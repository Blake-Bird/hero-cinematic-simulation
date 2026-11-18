"""Native GADGET configuration, IC handoffs, execution and frozen-run recovery.

This module translates explicit physical/numerical settings into compiled
features and runtime parameters, binds exact linear/IC inputs, records native
execution and verifies checkpoint continuation. It does not make scientific
measurements from snapshots; that belongs to analysis and domain readers.
Historical logical sections remain registered in configs/core_source_layout.json.
"""
from __future__ import annotations

# ========================================================================
# simulate: Initial conditions, GADGET-4 configuration, validation and execution.

# Workflow and units
# Creates model-matched ICs and run-specific GADGET executables, parameter files and provenance.
# GADGET code units are kpc/h, 10^10 Msun/h and km/s; cosmological snapshot velocities store peculiar velocity divided by sqrt(a).
# Matter particles evolve under Newtonian gravity/SPH. EDE supplies linear initial conditions and a smooth tabulated expansion/growth history; GADGET does not evolve nonlinear scalar-field modes.
# Bundled source patches: core/patches. Runtime choices are saved with each run.

import argparse
import copy
import hashlib
import json
import math
import os
import platform
import re
import subprocess
import time
import tempfile
from pathlib import Path
from core.native import native_engine_directory
from core.resources import runtime_paths

import h5py
import numpy as np
from core.data import verify_numeric_table
from core.data import hdf5_scalar_attr as _hdf5_scalar_attr, convert_pinned_native, exact_uint64_ids
import yaml

from core.cosmology import class_redshift_matches, class_background_table, choose_start_redshift, convert_pk_to_ngenic, load_config, match_wdm_cutoff_coverage, omega_radiation_from_config, pk_output_redshift, run_boltzmann, select_start_redshift, smooth_particle_growth_rate, validate_wdm_representation, validate_wdm_resolution_gate, wdm_fd_speed_moments, wdm_parameters, wdm_thermal_velocity_from_ids, write_monofonic_background_table, write_monofonic_ic_transfer_table, write_species_transfer_validation_table, write_wdm_velocity_transfer_table
from core.foundation import file_sha256, choose_run_directory
from core.native import external_tool_inventory
from core.resources import build_jobs, cpu_count, cpu_thread_budget, estimate_run_resources, memory_limit_gb, mpi_ranks, omp_threads, preflight_storage, initial_condition_census
from core.foundation import G_CGS, MPC_CM, KPC_CM, MSUN_G, RHO_CRIT_MSUN_H_MPC_H3
from zoom import softening_plan, validate_coarse_softening_masses
from core.foundation import atomic_copy, atomic_json, atomic_text
from core.data import recover_native_checkpoints, verify_native_checkpoints, stage_native_checkpoint_clock, verify_checkpoint_epochs
from core.native import native_run_owner, native_run_lock_fds, native_run_lock

ROOT = Path(__file__).resolve().parent


def require_particle_cosmology(cfg, require_ede_background_table=False):
    """Allow EDE-conditioned matter runs only with the matching smooth background table."""
    model = str(cfg.get("cosmology", {}).get("model", "")).lower()
    if model not in {"lcdm", "wdm", "ede"}:
        raise ValueError("Particle runs support LCDM, WDM, or EDE-conditioned matter inputs")
    enabled = bool(cfg.get("ede", {}).get("enabled", False))
    if enabled != (model == "ede"):
        raise ValueError("cosmology.model='ede' and ede.enabled must be selected together")
    if model == "ede" and require_ede_background_table:
        table = cfg.get("cosmology", {}).get("gadget_background_table")
        if not table or not Path(table).expanduser().is_file():
            raise ValueError("EDE particle evolution requires the run-matched AxiCLASS Gadget background table")


def validate_config_keys(cfg):
    """Reject misspelled/ignored settings while allowing documented runtime fields."""
    from core.sidm_physics import SIDM_CONFIGURATION_KEYS, SIDM_OPTIONAL_KEYS
    allowed = {
        'sidm':SIDM_CONFIGURATION_KEYS | SIDM_OPTIONAL_KEYS,
        'combination_specification': {'receipt', 'sha256'},
        "project": {"name", "runs_dir", "python", "software_dir"},
        "hardware": {"cpu_threads", "mpi_ranks", "omp_threads", "build_jobs", "memory_limit_gb"},
        "validation": {
            "background", "ic_power_spectrum", "ic_phase", "ic_velocity_divergence",
        },
        "spectrum": {"source", "file", "format", "k_unit", "p_unit", "redshift"},
        "cosmology": {
            "model", "engine", "H0", "omega_b", "omega_cdm", "A_s", "n_s", "tau_reio",
            "N_ur", "T_cmb", "N_ncdm", "m_ncdm_eV", "T_ncdm", "extra",
            "omega_radiation", "gadget_background_table",
        },
        "ede": {
            "enabled", "parameterization", "f_ede", "z_c", "theta_i", "n_axion", "extra_axiclass",
        },
        "wdm": {
            "enabled", "mode", "model", "thermal_relic_mass_keV", "max_thermal_speed_fraction_c",
            "include_thermal_velocities", "allow_underresolved_benchmark", "minimum_particles_per_half_mode",
            "lcdm_reference_pk", "lcdm_reference_redshift", "extra_class",
        },
        "simulation": {
            "box_mpc_h", "particles_per_dimension", "start_redshift", "start_redshift_guess",
            "auto_start_target_rms_displacement_fraction", "auto_start_min_fraction",
            "auto_start_max_fraction", "auto_start_min_redshift", "auto_start_max_redshift",
            "random_seed", "lpt_order_dm", "lpt_order_hydro", "ic_engine_dm", "ic_engine_hydro",
            "pm_grid", "softening_kpc_h", "softening_max_physical_kpc_h", "zoom_softening_mass_ratios", "accuracy",
            "asmth", "multipole_order",
            "max_size_timestep_ln_a", "min_size_timestep_ln_a", "err_tol_force_acc", "err_tol_int_accuracy", "err_tol_theta",
            "physics", "snapshot_z_max", "snapshot_z_min", "snapshot_dz", "snapshot_redshift_list", "highlight_redshifts", "end_redshift",
            "synchronized_outputs", "output_format", "restart_every_seconds", "num_files_per_snapshot",
            "time_limit_cpu_seconds", "output_acceleration_diagnostics", "restart_source_capsule", "restart_exercise_pause_step", "force_equal_timesteps",
            "monofonic_transfer_validation_source", "monofonic_background_validation_source",
            "native_generation_report", "native_flow_window_comparison",
        },
        "hydro": {
            "cooling", "star_formation", "uv_background", "courant_factor", "treecool_file",
            "allow_missing_treecool", "require_species_transfer_functions",
            "allow_gadget_particle_split_approximation",
            "max_sfr_timescale", "temp_clouds", "temp_supernova", "factor_evp",
            "factor_sn", "crit_phys_density", "crit_overdensity", "stellar_random_seed",
            "desired_neighbors", "neighbor_deviation", "artificial_viscosity", "initial_gas_temperature", "initial_gas_temperature_reference_scale_factor",
        },
        "black_holes": {"enabled", "radiative_efficiency", "feedback_coupling", "bondi_boost",
                        "eddington_factor", "kernel_radius_proper_kpc", "seed_mass_msun",
                        "source_fraction", "maximum_count", "seeding_mode",
                        "seed_density_hydrogen_cm3", "seed_exclusion_proper_kpc", "dynamics", "angular_momentum", "feedback"},
        "stellar_evolution": {"enabled","model_manifest","model_sha256","kernel_radius_proper_kpc",
                              "source_fraction","maximum_count","maximum_subcycles"},
        "chemistry": {"enabled","primordial_network","metal_cooling","uv_background","data_file","data_sha256",
                      "library_file","library_sha256","upstream_revision","deuterium_hydrogen_number_ratio","initial_species_number_fractions"},
        # Seed placement is a declared model choice, independent of accretion.
        "blender": {"particle_mode", "max_memory_fraction"},
        "halos": {
            "fof", "subfind", "rockstar", "consistent_trees", "native_mergertree", "fof_linking_length",
            "min_particles", "rockstar_strict_so_masses", "rockstar_overlap_length_mpc_h",
            "rockstar_exact_membership", "rockstar_wall_seconds",
        },
        "analysis": {
            "all", "products", "requested_outputs", "deferred_outputs", "hmf_mass_definition", "hmf_min_particles", "hmf_mass_range_msun_h",
            "max_accretion_histories",
            "hmf_resolution_relative_tolerance", "hmf_resolution_confidence",
            "hmf_resolution_min_halos_per_bin", "hmf_resolution_required_consecutive_bins",
            "finder_match_radius_particle_spacings",
            "cross_run_match_min_shared_particles", "cross_run_match_min_overlap_fraction",
            "cross_run_match_max_catalogue_ids", "mass_bins", "comparison_hmf_mass_range_msun_h",
            "power_grid", "halo_bias_min_halos", "halo_bias_min_modes_per_bin",
            "halo_bias_nyquist_fraction", "halo_profile_max_halos", "halo_profile_max_particles",
            "halo_profile_radial_bins", "halo_shape_max_halos", "halo_shape_max_particles",
            "halo_shape_min_particles", "correlation_bins", "correlation_min_halos",
            "correlation_min_separation_mpc_h", "density_grid",
            "rare_halo_thresholds_msun_h", "halo_shape_max_halos", "halo_shape_max_particles",
            "halo_shape_min_particles",
            "stellar_age_bins", "wdm_protohalo_shape_control_quantile",
            "wdm_protohalo_shape_min_control_halos", "wdm_spurious_min_particles",
        },
        "galaxies": {"enabled", "min_star_particles", "min_dm_particles"},
        "visualization": {
            "structure_maps", "comparison_maps", "movie", "interactive_3d", "blender_export",
            "map_fields", "map_projections",
            "max_matched_halo_overlays", "max_points_3d", "max_catalogue_points_3d",
            "blender_galaxy_overlays", "blender_max_galaxies", "blender_subhalo_limit",
            "hydro_map_grid", "r200_shell_max_count", "velocity_grid_size",
            "max_velocity_seeds_3d", "max_interpolation_points",
            "density_volume_grid_size", "cosmic_web_grid_size", "cosmic_web_smoothing_cells", "timeline_loop",
            "structure_catalogue_overlays", "filament_overlay", "blender_render",
            "blender_timeline", "blender_interpolate_motion", "blender_timeline_camera_orbit",
            "disperse_persistence_cut", "disperse_smoothing_cells", "disperse_timeout_s",
            "disperse_density_smoothing_cells",
            "disperse_executable", "skelconv_executable", "filament_method","disperse_output_coordinate_representation",
            "blender_executable", "filament_grid_size", "movie_fps", "slice_thickness_mpc_h", "slice_center_mpc_h",
            "slice_mode",
            "filament_density_percentile", "filament_curvature_threshold_factor", "filament_color",
            "filament_smoothing_mpc_h",
            "publication_style", "publication_font", "figure_layout", "figure_formats", "figure_dpi",
            "structure_story_figure", "structure_story_projections",
            "blender_sequence_max_points", "blender_frames_per_snapshot",
            "blender_render_quality",
            "density_scale_mode", "density_display_percentiles", "density_cmap",
            "temperature_cmap", "difference_cmap", "structure_map_overlays", "map_window_mpc_h",
        },
    }
    if not isinstance(cfg, dict):
        raise ValueError("configuration root must be a mapping")
    unknown_sections = sorted(set(cfg) - set(allowed))
    if unknown_sections:
        raise ValueError("Unsupported configuration section(s): " + ", ".join(unknown_sections))
    for section, options in cfg.items():
        if not isinstance(options, dict):
            raise ValueError("configuration.%s must be a mapping" % section)
        unknown_options = sorted(set(options) - allowed[section])
        if unknown_options:
            raise ValueError(
                "Unsupported configuration key(s) in %s: %s" % (section, ", ".join(unknown_options))
            )
    highlights = cfg.get('simulation', {}).get('highlight_redshifts', [])
    schedule = cfg.get('simulation', {}).get('snapshot_redshift_list')
    if (not isinstance(highlights, list) or any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in highlights)
            or len(set(highlights)) != len(highlights)
            or (highlights and (not isinstance(schedule, list) or not set(highlights) <= set(schedule)))):
        raise ValueError('simulation.highlight_redshifts must be distinct finite members of snapshot_redshift_list')
    deferred = cfg.get('analysis', {}).get('deferred_outputs', [])
    if not isinstance(deferred, list) or any(p not in {'stellar_components', 'stellar_histories', 'zoom_contamination', 'inference_diagnostics', 'radial_shapes'} for p in deferred):
        raise ValueError('analysis.deferred_outputs must list supported specialized measurements')
    if type(cfg.get('simulation',{}).get('output_acceleration_diagnostics',False))is not bool:
        raise ValueError('simulation.output_acceleration_diagnostics requires an explicit boolean')
    if type(cfg.get('simulation',{}).get('restart_source_capsule',False))is not bool:
        raise ValueError('simulation.restart_source_capsule requires an explicit boolean')
    if 'restart_exercise_pause_step' in cfg.get('simulation',{}):
        step=cfg['simulation']['restart_exercise_pause_step']
        if type(step) is not int or not 0<=step<=1000000 or cfg['simulation'].get('restart_source_capsule') is not True:
            raise ValueError('Restart exercise needs a bounded step and full native state diagnostics')
    blender = cfg.get("blender", {})
    particle_mode = str(blender.get("particle_mode", "sampled")).strip().lower()
    if particle_mode not in {"full", "sampled", "adaptive"}:
        raise ValueError("blender.particle_mode must be 'full', 'sampled', or 'adaptive'")
    try:
        memory_fraction = float(blender.get("max_memory_fraction", 0.70))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("blender.max_memory_fraction must be finite and in (0, 1]") from exc
    if not math.isfinite(memory_fraction) or not 0.0 < memory_fraction <= 1.0:
        raise ValueError("blender.max_memory_fraction must be finite and in (0, 1]")
    validation = cfg.get("validation", {})
    nested_allowed = {
        "background": {
            "max_relative_hubble_error", "max_relative_growth_rate_error", "max_relative_growth_error",
            "max_relative_drift_kick_error", "max_monofonic_interpolation_error",
        },
        "ic_power_spectrum": {"min_modes_per_bin", "min_bins", "false_rejection_probability", "null_monte_carlo_trials"},
        "ic_phase": {"max_mesh_size", "min_modes", "min_coherence", "max_weighted_rms_phase_rad"},
        "ic_velocity_divergence": {
            "max_mesh_size", "min_modes", "kmax_fraction_nyquist", "min_coherence",
            "max_response_error", "max_rms_fractional_residual",
        },
    }
    unknown_validation_sections = sorted(set(validation) - set(nested_allowed))
    if unknown_validation_sections:
        raise ValueError("Unsupported validation section(s): " + ", ".join(unknown_validation_sections))
    for name, values in validation.items():
        if not isinstance(values, dict):
            raise ValueError("validation.%s must be a mapping" % name)
        unknown = sorted(set(values) - nested_allowed[name])
        if unknown:
            raise ValueError("Unsupported configuration key(s) in validation.%s: %s" % (name, ", ".join(unknown)))
    return True


def validate_supported_options(cfg):
    """Fail before a run when a public setting is unsupported or ignored."""
    require_particle_cosmology(cfg)
    validate_config_keys(cfg)
    from core.research.capabilities import require_supported_combination
    interaction_kind = 'elastic_sidm' if cfg.get('sidm', {}).get('enabled') is True else 'collisionless'
    linear_kind = 'wdm' if cfg['cosmology']['model'] == 'wdm' or cfg.get('wdm', {}).get('enabled') is True else 'cdm'
    combined_model = {'cosmology': cfg['cosmology'], 'ede': cfg.get('ede', {}),
        'wdm': cfg.get('wdm', {}), 'sidm': cfg.get('sidm', {'enabled': False})}
    from core.foundation import canonical_digest
    background_kind = 'lcdm' if cfg['cosmology']['model'] == 'wdm' else cfg['cosmology']['model']
    require_supported_combination({'cosmology_model': background_kind,
        'dark_matter': {'linear_model': linear_kind, 'interactions': {'kind': interaction_kind}},
        'combination_specification': cfg.get('combination_specification'),
        'combination_model_sha256': canonical_digest(combined_model)})
    from core.sidm_physics import sidm_native_parameters
    sidm_native_parameters(cfg.get('sidm',{'enabled':False}),physics=cfg['simulation'].get('physics','dm'),
        mpi_ranks=mpi_ranks(cfg),max_mem_rank_mib=int(.85*memory_limit_gb(cfg)*1024/mpi_ranks(cfg)))
    from black_holes import native_parameters
    bh_parameters = native_parameters(cfg.get("black_holes", {"enabled": False}))
    from stars import stellar_evolution_native_parameters
    stellar_settings=cfg.get('stellar_evolution',{'enabled':False})
    stellar_parameters=stellar_evolution_native_parameters(stellar_settings,
        factor_sn=hydro_star_formation_parameters(cfg.get('hydro',{}))['FactorSN'] if stellar_settings.get('enabled',False) else .1)
    if stellar_parameters:
        if cfg['simulation']['physics']!='hydro' or not cfg.get('hydro',{}).get('cooling',True) or not cfg.get('hydro',{}).get('star_formation',True):
            raise ValueError('Delayed stellar returns require native hydro, cooling and SH03 birth state')
        start_a=1/(1+float(cfg['simulation']['start_redshift']));h=float(cfg['cosmology']['H0'])/100.
        if stellar_settings['kernel_radius_proper_kpc']>=500*float(cfg['simulation']['box_mpc_h'])*start_a/h:
            raise ValueError('Stellar source support exceeds the unique proper periodic region')
    from stars import grackle_native_parameters
    chemistry_parameters=grackle_native_parameters(cfg.get('chemistry',{'enabled':False}))
    if chemistry_parameters and not stellar_parameters:
        raise ValueError('Native chemistry requires the declared eleven-element stellar source model')
    if bh_parameters and cfg["simulation"]["physics"] != "hydro":
        raise ValueError("BH accretion/feedback requires hydro")
    if bh_parameters:
        h=float(cfg["cosmology"]["H0"])/100.
        start_a=1/(1+float(cfg["simulation"]["start_redshift"]))
        half_proper_box=500*float(cfg["simulation"]["box_mpc_h"])*start_a/h
        for name in ("kernel_radius_proper_kpc", "seed_exclusion_proper_kpc"):
            if cfg["black_holes"][name]>=half_proper_box:
                raise ValueError("BH "+name+" exceeds half the proper box at the starting epoch")
        if cfg["black_holes"].get("dynamics",{}).get("enabled",False):
            if cfg["black_holes"]["dynamics"]["background_radius_proper_kpc"]>=half_proper_box:
                raise ValueError("BH dynamics background support exceeds the unique proper periodic region")
            if cfg.get("sidm",{}).get("enabled",False):
                raise ValueError("Native BH collisionless friction cannot be applied to an uncalibrated SIDM background")
        if cfg["black_holes"]["seeding_mode"]=="density_exclusion":
            c=cfg["cosmology"]
            gas_mass=particle_mass_msun_h(cfg,dm_only=False)*float(c["omega_b"])/float(c["omega_cdm"])/h
            if cfg["black_holes"]["seed_mass_msun"]>gas_mass:
                raise ValueError("BH seed mass exceeds one gas particle; reduce seed mass or supply a separately validated IC model")
    visualization = cfg.get("visualization", {})
    filament_method = visualization.get("filament_method", "hessian_proxy")
    map_fields = visualization.get("map_fields", ["dm"])
    map_projections = visualization.get("map_projections", ["xy"])
    allowed_map_fields = {"dm", "gas", "gas_temperature", "stars", "total"}
    allowed_map_projections = {"xy", "xz", "yz"}
    if (not isinstance(map_fields, list) or not map_fields
            or any(not isinstance(value, str) or value not in allowed_map_fields for value in map_fields)
            or len(set(map_fields)) != len(map_fields)):
        raise ValueError(
            "visualization.map_fields must be a non-empty unique list from: "
            + ", ".join(sorted(allowed_map_fields))
        )
    if (not isinstance(map_projections, list) or not map_projections
            or any(not isinstance(value, str) or value not in allowed_map_projections for value in map_projections)
            or len(set(map_projections)) != len(map_projections)):
        raise ValueError(
            "visualization.map_projections must be a non-empty unique list from: "
            + ", ".join(sorted(allowed_map_projections))
        )
    story_projections = visualization.get("structure_story_projections")
    if visualization.get("structure_story_figure", True) and (
            not isinstance(story_projections, list) or not story_projections
            or any(not isinstance(value, str) or value not in allowed_map_projections for value in story_projections)
            or len(set(story_projections)) != len(story_projections)):
        raise ValueError("visualization.structure_story_projections must be a non-empty unique list from: xy, xz, yz")
    filament_color = visualization.get("filament_color", "#7af7ed")
    if not isinstance(filament_color, str) or not filament_color.strip():
        raise ValueError("visualization.filament_color must be a non-empty color string")
    smoothing = visualization.get("filament_smoothing_mpc_h")
    if smoothing is not None:
        if isinstance(smoothing, bool):
            raise ValueError("visualization.filament_smoothing_mpc_h must be null or finite and non-negative")
        try:
            smoothing = float(smoothing)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("visualization.filament_smoothing_mpc_h must be null or finite and non-negative") from exc
        if not math.isfinite(smoothing) or smoothing < 0:
            raise ValueError("visualization.filament_smoothing_mpc_h must be null or finite and non-negative")
    if (str(visualization.get("filament_method", "")).strip().lower() == "hessian_proxy"
            and bool(visualization.get("filament_overlay", False))
            and (smoothing is None or smoothing <= 0)):
        raise ValueError("visualization.filament_smoothing_mpc_h must be positive for Hessian filament overlays")
    simulation = cfg.get("simulation", {})
    box_size = float(simulation.get("box_mpc_h", np.nan))
    if not math.isfinite(box_size) or box_size <= 0:
        raise ValueError("simulation.box_mpc_h must be finite and positive")
    slice_mode = visualization.get("slice_mode")
    slice_thickness = visualization.get("slice_thickness_mpc_h")
    slice_center = visualization.get("slice_center_mpc_h")
    if slice_mode == "full_box":
        if slice_thickness is not None or slice_center is not None:
            raise ValueError("full_box slice mode cannot also declare slab dimensions")
    elif slice_mode == "slab":
        if isinstance(slice_thickness, bool) or not isinstance(slice_thickness, (int, float)):
            raise ValueError("slab slice mode requires finite thickness and center in Mpc/h")
        if not math.isfinite(float(slice_thickness)) or not 0 < float(slice_thickness) <= box_size:
            raise ValueError("visualization.slice_thickness_mpc_h must be finite and in (0, box size]")
        if isinstance(slice_center, bool) or not isinstance(slice_center, (int, float)) or not math.isfinite(float(slice_center)):
            raise ValueError("visualization.slice_center_mpc_h must be finite in slab mode")
    else:
        raise ValueError("visualization.slice_mode must explicitly be full_box or slab")
    particles_per_dimension = simulation.get("particles_per_dimension")
    if isinstance(particles_per_dimension, bool) or not isinstance(particles_per_dimension, int) or particles_per_dimension < 2:
        raise ValueError("simulation.particles_per_dimension must be an integer of at least 2")
    random_seed = simulation.get("random_seed")
    if isinstance(random_seed, bool) or not isinstance(random_seed, int) or random_seed < 0:
        raise ValueError("simulation.random_seed must be explicitly set to a non-negative integer")
    softening = simulation.get("softening_kpc_h")
    if isinstance(softening, bool):
        raise ValueError("simulation.softening_kpc_h must be explicitly set to a finite positive number")
    try:
        softening = float(softening)
    except (TypeError, ValueError) as exc:
        raise ValueError("simulation.softening_kpc_h must be explicitly set to a finite positive number") from exc
    if not math.isfinite(softening) or softening <= 0:
        raise ValueError("simulation.softening_kpc_h must be explicitly set to a finite positive number")
    max_physical_softening = simulation.get("softening_max_physical_kpc_h")
    if isinstance(max_physical_softening, bool):
        raise ValueError("simulation.softening_max_physical_kpc_h must be explicitly set and positive")
    try:
        max_physical_softening = float(max_physical_softening)
    except (TypeError, ValueError) as exc:
        raise ValueError("simulation.softening_max_physical_kpc_h must be explicitly set and positive") from exc
    if not math.isfinite(max_physical_softening) or max_physical_softening <= 0:
        raise ValueError("simulation.softening_max_physical_kpc_h must be explicitly set and positive")
    if max_physical_softening > softening:
        raise ValueError("maximum physical softening must not exceed the comoving softening")
    softening_plan(simulation)
    start_redshift = simulation.get("start_redshift")
    if start_redshift is None or start_redshift == "auto":
        raise ValueError("simulation.start_redshift must be explicitly set by the selected cosmology")
    try:
        start_redshift = float(start_redshift)
    except (TypeError, ValueError) as exc:
        raise ValueError("simulation.start_redshift must be finite and non-negative") from exc
    if not math.isfinite(start_redshift) or start_redshift < 0:
        raise ValueError("simulation.start_redshift must be finite and non-negative")
    for name in ["end_redshift", "snapshot_z_min", "snapshot_z_max", "snapshot_dz"]:
        try:
            value = float(simulation[name])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("simulation.%s must be a finite number" % name) from exc
        if not math.isfinite(value):
            raise ValueError("simulation.%s must be a finite number" % name)
    try:
        cpu_time_limit = float(simulation.get("time_limit_cpu_seconds"))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("simulation.time_limit_cpu_seconds must be finite and positive") from exc
    if not math.isfinite(cpu_time_limit) or cpu_time_limit <= 0:
        raise ValueError("simulation.time_limit_cpu_seconds must be finite and positive")
    restart_interval = simulation.get("restart_every_seconds")
    if isinstance(restart_interval, bool) or not isinstance(restart_interval, int) or restart_interval <= 0:
        raise ValueError("simulation.restart_every_seconds must be explicitly set to a positive integer")
    if float(simulation["end_redshift"]) < 0 or float(simulation["snapshot_z_min"]) < 0:
        raise ValueError("simulation.end_redshift and snapshot_z_min must be non-negative")
    if float(simulation["snapshot_dz"]) <= 0:
        raise ValueError("simulation.snapshot_dz must be positive")
    for name in ("monofonic_transfer_validation_source", "monofonic_background_validation_source", "native_generation_report"):
        value = simulation.get(name)
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ValueError("simulation.%s must be null or a non-empty path" % name)
    if simulation.get('native_flow_window_comparison') is not None:
        from core.music.flow import validate_original_flow_launch_request
        validate_original_flow_launch_request(simulation['native_flow_window_comparison'])
    if simulation.get("snapshot_redshift_list") is None and start_redshift != "auto" and float(simulation["snapshot_z_max"]) > start_redshift:
        raise ValueError("simulation.snapshot_z_max cannot exceed the initial-condition redshift")
    snapshot_redshifts(cfg)
    if simulation.get("physics") not in {"dm", "hydro"}:
        raise ValueError("simulation.physics must be 'dm' or 'hydro'")
    if type(cfg.get("wdm", {}).get("include_thermal_velocities", True)) is not bool:
        raise ValueError("wdm.include_thermal_velocities must be a YAML boolean")
    if simulation.get("ic_engine_dm") != "ngenic":
        raise ValueError("Only simulation.ic_engine_dm='ngenic' is implemented for DM-only initial conditions")
    if simulation.get("ic_engine_hydro") not in {'monofonic','native_music'}:
        raise ValueError("Mixed-species ICs require monofonic generation or an audited native_music handoff")
    if simulation.get('ic_engine_hydro')=='native_music' and (
            not simulation.get('native_generation_report') or type(simulation.get('lpt_order_hydro')) is not int
            or simulation.get('lpt_order_hydro') not in (1,2)):
        raise ValueError('native_music requires a generation report and declared supported LPT order')
    order = simulation.get("lpt_order_dm")
    if isinstance(order, bool) or order != 2:
        raise ValueError("N-GenIC is compiled with NGENIC_2LPT; simulation.lpt_order_dm must be 2")
    if simulation.get("synchronized_outputs") is not True:
        raise ValueError("Only synchronized GADGET outputs from simulation.output_times.txt are implemented")
    if simulation.get("output_format") != "hdf5":
        raise ValueError("Only simulation.output_format='hdf5' is implemented by the IC, halo, and analysis workflows")
    asmth = simulation.get("asmth")
    if isinstance(asmth, bool):
        raise ValueError("simulation.asmth must be explicitly set to a finite positive value")
    try:
        asmth = float(asmth)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("simulation.asmth must be explicitly set to a finite positive value") from exc
    if not math.isfinite(asmth) or asmth <= 0:
        raise ValueError("simulation.asmth must be explicitly set to a finite positive value")
    multipole_order = simulation.get("multipole_order")
    if isinstance(multipole_order, bool) or not isinstance(multipole_order, int) or multipole_order not in {1, 2, 3, 4, 5}:
        raise ValueError("simulation.multipole_order must be an integer from 1 to 5")
    gadget_numerics = {
        "max_size_timestep_ln_a": "simulation.max_size_timestep_ln_a",
        "err_tol_force_acc": "simulation.err_tol_force_acc",
        "err_tol_int_accuracy": "simulation.err_tol_int_accuracy",
        "err_tol_theta": "simulation.err_tol_theta",
    }
    for key, label in gadget_numerics.items():
        value = simulation.get(key)
        if isinstance(value, bool):
            raise ValueError(label + " must be explicitly set to a finite positive number")
        try:
            value = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(label + " must be explicitly set to a finite positive number") from exc
        if not math.isfinite(value) or value <= 0:
            raise ValueError(label + " must be explicitly set to a finite positive number")
    minimum_step = simulation.get("min_size_timestep_ln_a", 0.0)
    if isinstance(minimum_step, bool):
        raise ValueError("simulation.min_size_timestep_ln_a must be finite and non-negative")
    try:
        minimum_step = float(minimum_step)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("simulation.min_size_timestep_ln_a must be finite and non-negative") from exc
    if not math.isfinite(minimum_step) or not 0 <= minimum_step <= float(simulation["max_size_timestep_ln_a"]):
        raise ValueError("simulation.min_size_timestep_ln_a must be between zero and max_size_timestep_ln_a")
    schedule = snapshot_redshifts(cfg)
    if len(schedule) > 1:
        minimum_spacing = min(math.log((1.0 + left) / (1.0 + right))
                              for left, right in zip(schedule, schedule[1:]))
        if float(simulation["max_size_timestep_ln_a"]) > minimum_spacing:
            raise ValueError(
                "simulation.max_size_timestep_ln_a exceeds the smallest snapshot spacing "
                f"in ln(a) ({minimum_spacing:.8g}); reduce it before launching GADGET"
            )
    grid = simulation.get("pm_grid", "auto")
    if isinstance(grid, bool) or not isinstance(grid, int) or grid < 8:
        raise ValueError("simulation.pm_grid must be explicitly set to an integer of at least 8")
    snapshot_files = simulation.get("num_files_per_snapshot")
    if isinstance(snapshot_files, bool) or not isinstance(snapshot_files, int) or snapshot_files < 1:
        raise ValueError("simulation.num_files_per_snapshot must be a positive integer")
    if snapshot_files > mpi_ranks(cfg):
        raise ValueError("simulation.num_files_per_snapshot cannot exceed the configured MPI rank count")

    courant_factor = cfg.get("hydro", {}).get("courant_factor")
    if courant_factor is not None:
        if str(simulation.get("physics", "dm")).lower() != "hydro":
            raise ValueError("hydro.courant_factor is only meaningful when simulation.physics='hydro'")
        if isinstance(courant_factor, bool):
            raise ValueError("hydro.courant_factor must be null or a finite positive number")
        try:
            courant_factor = float(courant_factor)
        except (TypeError, ValueError) as exc:
            raise ValueError("hydro.courant_factor must be null or a finite positive number") from exc
        if not math.isfinite(courant_factor) or courant_factor <= 0:
            raise ValueError("hydro.courant_factor must be null or a finite positive number")

    analysis = cfg.get("analysis", {})
    run_all_analysis = analysis.get("all", True)
    if type(run_all_analysis) is not bool:
        raise ValueError("analysis.all must be a YAML boolean")
    selected_products = analysis.get("products")
    if run_all_analysis:
        if selected_products not in (None, [], ""):
            raise ValueError("analysis.products is used only when analysis.all is false")
    else:
        if selected_products is None:
            raise ValueError("analysis.all=false requires a non-empty analysis.products list")
        # Import lazily so simulation setup does not load the plotting stack
        # unless product-specific analysis validation is actually requested.
        from analysis import normalize_analysis_products
        try:
            normalize_analysis_products(selected_products)
        except ValueError as exc:
            raise ValueError("Invalid analysis.products: " + str(exc)) from exc
    for name, default, minimum in [
        ("mass_bins", 35, 1), ("hmf_min_particles", cfg.get("halos", {}).get("min_particles", 100), 0),
        ("power_grid", 256, 16), ("density_grid", 1024, 8),
    ]:
        value = analysis.get(name, default)
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError("analysis.%s must be an integer of at least %d" % (name, minimum))
    if int(analysis.get("power_grid", 256)) > 256:
        raise ValueError("analysis.power_grid must be at most 256 for the full-resolution halo-bias mesh")
    for name, default, minimum in [
        ("halo_bias_min_halos", 20, 2), ("halo_bias_min_modes_per_bin", 20, 1),
        ("correlation_bins", 35, 3), ("correlation_min_halos", 20, 2),
        ("halo_profile_max_halos", 8, 1), ("halo_profile_max_particles", 4_000_000, 1),
        ("halo_profile_radial_bins", 24, 3), ("halo_shape_max_halos", 40, 1),
        ("halo_shape_max_particles", 4_000_000, 1), ("halo_shape_min_particles", 100, 3),
    ]:
        value = analysis.get(name, default)
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError("analysis.%s must be an integer of at least %d" % (name, minimum))
    correlation_minimum = analysis.get("correlation_min_separation_mpc_h")
    if correlation_minimum is not None:
        if isinstance(correlation_minimum, bool):
            raise ValueError("analysis.correlation_min_separation_mpc_h must be finite and in (0, box/2)")
        try:
            correlation_minimum = float(correlation_minimum)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("analysis.correlation_min_separation_mpc_h must be finite and in (0, box/2)") from exc
        if not math.isfinite(correlation_minimum) or not 0 < correlation_minimum < box_size / 2.0:
            raise ValueError("analysis.correlation_min_separation_mpc_h must be finite and in (0, box/2)")
    bias_nyquist_fraction = float(analysis.get("halo_bias_nyquist_fraction", 0.25))
    if not math.isfinite(bias_nyquist_fraction) or not 0 < bias_nyquist_fraction <= 0.5:
        raise ValueError("analysis.halo_bias_nyquist_fraction must be in (0, 0.5]")
    finder_match_spacing = analysis.get("finder_match_radius_particle_spacings", 0.5)
    if isinstance(finder_match_spacing, bool):
        raise ValueError("analysis.finder_match_radius_particle_spacings must be finite and positive")
    try:
        finder_match_spacing = float(finder_match_spacing)
    except (TypeError, ValueError) as exc:
        raise ValueError("analysis.finder_match_radius_particle_spacings must be finite and positive") from exc
    if not math.isfinite(finder_match_spacing) or finder_match_spacing <= 0:
        raise ValueError("analysis.finder_match_radius_particle_spacings must be finite and positive")
    hmf_mass_definition = str(analysis.get("hmf_mass_definition", "fof")).lower()
    if hmf_mass_definition not in {"fof", "m200c", "m200b", "mvir"}:
        raise ValueError("analysis.hmf_mass_definition must be one of: fof, m200c, m200b, mvir")
    if hmf_mass_definition != "fof" and not bool(cfg.get("halos", {}).get("subfind", True)):
        raise ValueError("analysis.hmf_mass_definition=%s requires halos.subfind=true" % hmf_mass_definition)
    comparison_mass_range = analysis.get("comparison_hmf_mass_range_msun_h")
    if comparison_mass_range is not None:
        if not isinstance(comparison_mass_range, (list, tuple)) or len(comparison_mass_range) != 2:
            raise ValueError("analysis.comparison_hmf_mass_range_msun_h must be null or [minimum, maximum]")
        lower, upper = map(float, comparison_mass_range)
        if not (math.isfinite(lower) and math.isfinite(upper) and 0 < lower < upper):
            raise ValueError("analysis.comparison_hmf_mass_range_msun_h must be finite, positive, and increasing")
    hmf_mass_range = analysis.get("hmf_mass_range_msun_h")
    if hmf_mass_range is not None:
        if not isinstance(hmf_mass_range, (list, tuple)) or len(hmf_mass_range) != 2:
            raise ValueError("analysis.hmf_mass_range_msun_h must be null or [minimum, maximum]")
        lower, upper = map(float, hmf_mass_range)
        if not (math.isfinite(lower) and math.isfinite(upper) and 0 < lower < upper):
            raise ValueError("analysis.hmf_mass_range_msun_h must be finite, positive, and increasing")
    try:
        resolution_tolerance = float(analysis.get("hmf_resolution_relative_tolerance", 0.05))
    except (TypeError, ValueError) as exc:
        raise ValueError("analysis.hmf_resolution_relative_tolerance must be finite and in (0, 1)") from exc
    if not math.isfinite(resolution_tolerance) or not 0 < resolution_tolerance < 1:
        raise ValueError("analysis.hmf_resolution_relative_tolerance must be finite and in (0, 1)")
    try:
        resolution_confidence = float(analysis.get("hmf_resolution_confidence", 0.95))
    except (TypeError, ValueError) as exc:
        raise ValueError("analysis.hmf_resolution_confidence must be finite and in (0, 1)") from exc
    if not math.isfinite(resolution_confidence) or not 0 < resolution_confidence < 1:
        raise ValueError("analysis.hmf_resolution_confidence must be finite and in (0, 1)")
    for name, default, minimum in [
        ("hmf_resolution_min_halos_per_bin", 20, 1),
        ("hmf_resolution_required_consecutive_bins", 3, 2),
    ]:
        value = analysis.get(name, default)
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError("analysis.%s must be an integer of at least %d" % (name, minimum))
    shape_control_quantile = analysis.get("wdm_protohalo_shape_control_quantile", 0.10)
    if (isinstance(shape_control_quantile, bool) or not isinstance(shape_control_quantile, (int, float))
            or not math.isfinite(shape_control_quantile) or not 0 < shape_control_quantile <= 0.5):
        raise ValueError("analysis.wdm_protohalo_shape_control_quantile must be finite and in (0, 0.5]")
    for name, default, minimum in (
        ("wdm_protohalo_shape_min_control_halos", 20, 1),
        ("wdm_spurious_min_particles", 300, 1),
    ):
        value = analysis.get(name, default)
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError("analysis.%s must be an integer of at least %d" % (name, minimum))

    project = cfg.get("project", {})
    if project.get("python") not in (None, ""):
        raise ValueError("project.python is not used to launch the app; remove this setting and run the desired Python interpreter directly")
    if "software_dir" in project:
        raise ValueError("project.software_dir is not configurable in this build; remove the setting")

    halos = cfg.get("halos", {})
    for name in ["fof", "subfind", "rockstar", "consistent_trees", "native_mergertree", "rockstar_strict_so_masses"]:
        if name in halos and type(halos[name]) is not bool:
            raise ValueError("halos.%s must be a YAML boolean" % name)
    if halos.get("subfind", True) and not halos.get("fof", True):
        raise ValueError("halos.subfind requires halos.fof=true in this GADGET-4 build")
    if halos.get('native_mergertree',False) and not (halos.get('subfind',True) and halos.get('fof',True)):
        raise ValueError('Native merger trees require FOF and SUBFIND')
    if halos.get("consistent_trees", True) and not halos.get("rockstar", True):
        raise ValueError("halos.consistent_trees requires halos.rockstar=true")
    min_halo_particles = halos.get("min_particles", 100)
    if isinstance(min_halo_particles, bool) or not isinstance(min_halo_particles, int) or min_halo_particles < 1:
        raise ValueError("halos.min_particles must be a positive integer")

    galaxies = cfg.get("galaxies", {})
    if "enabled" in galaxies and type(galaxies["enabled"]) is not bool:
        raise ValueError('halos.enabled must be a YAML boolean')
    for name, default in (("min_star_particles", 100), ("min_dm_particles", 300)):
        value = galaxies.get(name, default)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError('halos.%s must be a positive integer' % name)

    visualization = cfg.get("visualization", {})
    for name in [
        "structure_maps", "comparison_maps", "movie", "interactive_3d", "blender_export",
        "structure_catalogue_overlays", "filament_overlay", "blender_render",
        "timeline_loop", "blender_timeline", "blender_interpolate_motion",
        "blender_timeline_camera_orbit", "blender_galaxy_overlays",
    ]:
        if type(visualization.get(name, True)) is not bool:
            raise ValueError("visualization.%s must be a YAML boolean" % name)
    for name, default, minimum in [
        ("max_points_3d", 3_000_000, 1),
        ("max_catalogue_points_3d", 50_000, 1),
        ("hydro_map_grid", 256, 32),
        ("r200_shell_max_count", 200, 1),
        ("velocity_grid_size", 32, 16),
        ("max_velocity_seeds_3d", 800, 1),
        ("max_interpolation_points", 250_000, 1),
        ("density_volume_grid_size", 64, 16),
        ("cosmic_web_grid_size", 128, 24),
        ("filament_grid_size", 64, 24),
        ("blender_sequence_max_points", 800_000, 1),
        ("blender_frames_per_snapshot", 12, 1),
        ("blender_max_galaxies", 200, 1),
        ("disperse_smoothing_cells", 2, 0),
        ("disperse_timeout_s", 3600, 1),
    ]:
        value = visualization.get(name, default)
        if name == "disperse_smoothing_cells" and value is None and filament_method == "hessian_proxy":
            continue
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError("visualization.%s must be an integer of at least %d" % (name, minimum))
    if visualization.get("hydro_map_grid", 256) > 2048:
        raise ValueError("visualization.hydro_map_grid must be at most 2048")
    if visualization.get("velocity_grid_size", 32) > 128:
        raise ValueError("visualization.velocity_grid_size must be at most 128")
    if visualization.get("density_volume_grid_size", 64) > 128:
        raise ValueError("visualization.density_volume_grid_size must be at most 128")
    if visualization.get("cosmic_web_grid_size", 128) > 128:
        raise ValueError("visualization.cosmic_web_grid_size must be at most 128")
    if visualization.get("filament_grid_size", 64) > 160:
        raise ValueError("visualization.filament_grid_size must be at most 160")
    if visualization.get("blender_sequence_max_points", 800_000) > 50_000_000:
        raise ValueError("visualization.blender_sequence_max_points must be at most 50000000")
    if visualization.get("blender_frames_per_snapshot", 12) > 240:
        raise ValueError("visualization.blender_frames_per_snapshot must be at most 240")
    if (visualization.get("blender_interpolate_motion", True)
            and visualization.get("blender_frames_per_snapshot", 12) < 2):
        raise ValueError("visualization.blender_frames_per_snapshot must be at least 2 when Blender motion interpolation is enabled")
    if (visualization.get("disperse_smoothing_cells", 2) is not None
            and visualization.get("disperse_smoothing_cells", 2) > 64):
        raise ValueError("visualization.disperse_smoothing_cells must be at most 64")
    if visualization.get("disperse_timeout_s", 3600) > 86_400:
        raise ValueError("visualization.disperse_timeout_s must be at most 86400 seconds")
    if not isinstance(filament_method, str) or filament_method not in {"hessian_proxy", "disperse_grid"}:
        raise ValueError("visualization.filament_method must be hessian_proxy or disperse_grid")
    representation=visualization.get('disperse_output_coordinate_representation')
    if (filament_method=='disperse_grid' or representation is not None) and representation not in ('comoving_mpc_h','grid_vertex_coordinates'):
        raise ValueError('visualization.disperse_output_coordinate_representation must explicitly declare comoving_mpc_h or grid_vertex_coordinates')
    persistence_cut = visualization.get("disperse_persistence_cut", 0.35)
    if (filament_method == "disperse_grid" or persistence_cut is not None) and (
            isinstance(persistence_cut, bool) or not isinstance(persistence_cut, (int, float))
            or not np.isfinite(persistence_cut) or persistence_cut <= 0 or persistence_cut > 10):
        raise ValueError("visualization.disperse_persistence_cut must be finite and in (0, 10]")
    density_smoothing = visualization.get("disperse_density_smoothing_cells", 1.0)
    if (isinstance(density_smoothing, bool) or not isinstance(density_smoothing, (int, float))
            or not np.isfinite(density_smoothing) or density_smoothing < 0 or density_smoothing > 8):
        raise ValueError("visualization.disperse_density_smoothing_cells must be finite and in [0, 8]")
    for name in ("disperse_executable", "skelconv_executable"):
        value = visualization.get(name)
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ValueError("visualization.%s must be null or a non-empty executable path" % name)
    render_quality = visualization.get("blender_render_quality", "preview")
    if not isinstance(render_quality, str) or render_quality not in {"preview", "high"}:
        raise ValueError("visualization.blender_render_quality must be preview or high")
    density_percentile = visualization.get("filament_density_percentile", 85)
    if (filament_method == "hessian_proxy" or density_percentile is not None) and (
            isinstance(density_percentile, bool) or not isinstance(density_percentile, int)
            or not 50 <= density_percentile <= 99):
        raise ValueError("visualization.filament_density_percentile must be an integer from 50 to 99")
    curvature_factor = visualization.get("filament_curvature_threshold_factor", 1.0)
    if (filament_method == "hessian_proxy" or curvature_factor is not None) and (
            isinstance(curvature_factor, bool) or not isinstance(curvature_factor, (int, float))
            or not np.isfinite(curvature_factor) or curvature_factor <= 0 or curvature_factor > 10):
        raise ValueError("visualization.filament_curvature_threshold_factor must be finite and in (0, 10]")
    cosmic_web_smoothing = visualization.get("cosmic_web_smoothing_cells", 1.25)
    if (isinstance(cosmic_web_smoothing, bool)
            or not isinstance(cosmic_web_smoothing, (int, float))
            or not np.isfinite(cosmic_web_smoothing)
            or not 0.5 <= cosmic_web_smoothing <= 4.0):
        raise ValueError("visualization.cosmic_web_smoothing_cells must be finite and in [0.5, 4]")
    blender_executable = visualization.get("blender_executable")
    if blender_executable is not None and (not isinstance(blender_executable, str) or not blender_executable.strip()):
        raise ValueError("visualization.blender_executable must be null or a non-empty path")
    publication_style = visualization.get("publication_style", "journal")
    if publication_style not in {"journal", "presentation"}:
        raise ValueError("visualization.publication_style must be journal or presentation")
    publication_font = visualization.get("publication_font", "STIX Two Text")
    if not isinstance(publication_font, str) or not publication_font.strip():
        raise ValueError("visualization.publication_font must be a non-empty font family name")
    if visualization.get("figure_layout", "double_column") not in {"single_column", "double_column", "presentation"}:
        raise ValueError("visualization.figure_layout must be single_column, double_column, or presentation")
    formats = visualization.get("figure_formats", ["png", "pdf", "svg"])
    if (not isinstance(formats, list) or not formats
            or any(not isinstance(fmt, str) or fmt.lower().lstrip(".") not in {"png", "pdf", "svg"} for fmt in formats)):
        raise ValueError("visualization.figure_formats must be a non-empty list chosen from png, pdf, svg")
    if isinstance(visualization.get("figure_dpi", 450), bool) or not isinstance(visualization.get("figure_dpi", 450), int) or not 100 <= visualization.get("figure_dpi", 450) <= 1200:
        raise ValueError("visualization.figure_dpi must be an integer from 100 to 1200")
    if visualization.get("density_scale_mode", "global") not in {"global", "per_snapshot"}:
        raise ValueError("visualization.density_scale_mode must be global or per_snapshot")
    percentiles = visualization.get("density_display_percentiles", [0.5, 99.95])
    if (not isinstance(percentiles, (list, tuple)) or len(percentiles) != 2
            or any(isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) for value in percentiles)
            or not 0.0 <= float(percentiles[0]) < float(percentiles[1]) <= 100.0):
        raise ValueError("visualization.density_display_percentiles must be [low, high] with 0 <= low < high <= 100")
    for name, default in [("density_cmap", "magma"), ("temperature_cmap", "inferno"), ("difference_cmap", "coolwarm")]:
        value = visualization.get(name, default)
        if not isinstance(value, str) or value not in {
            "magma", "inferno", "cividis", "viridis", "plasma", "coolwarm", "RdBu_r", "cosmic_web",
        }:
            raise ValueError("visualization.%s must name a supported perceptual Matplotlib colormap" % name)
    overlay_layers = visualization.get(
        "structure_map_overlays", ["fof", "subhalos", "rockstar", "rare_hosts", "r200c", "filaments"],
    )
    allowed_overlays = {"fof", "subhalos", "rockstar", "rare_hosts", "r200c", "filaments"}
    if (not isinstance(overlay_layers, list)
            or any(not isinstance(layer, str) or layer not in allowed_overlays for layer in overlay_layers)):
        raise ValueError("visualization.structure_map_overlays must be a list of fof, subhalos, rockstar, rare_hosts, r200c, filaments")
    map_window = visualization.get("map_window_mpc_h")
    if map_window is not None:
        if (not isinstance(map_window, list) or len(map_window) != 4
                or any(isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) for value in map_window)):
            raise ValueError("visualization.map_window_mpc_h must be null or [xmin, xmax, ymin, ymax]")
        box = float(cfg.get("simulation", {}).get("box_mpc_h", 0.0))
        xmin, xmax, ymin, ymax = (float(value) for value in map_window)
        if not (0.0 <= xmin < xmax <= box and 0.0 <= ymin < ymax <= box):
            raise ValueError("visualization.map_window_mpc_h must be ordered within the configured box")
    age_bins = cfg.get("analysis", {}).get("stellar_age_bins", 32)
    if isinstance(age_bins, bool) or not isinstance(age_bins, int) or not 4 <= age_bins <= 256:
        raise ValueError("analysis.stellar_age_bins must be an integer from 4 to 256")
    return True


def snapshot_redshifts(cfg):
    """Merge requested event windows into the explicit cosmological backbone."""
    backbone = _backbone_snapshot_redshifts(cfg)
    cadence = cfg['simulation'].get('event_cadence')
    if cadence is None:
        return backbone
    from evolution import merge_and_quantize_native_output_times
    if not isinstance(cadence, dict) or set(cadence) != {'windows', 'maximum_outputs'}:
        raise ValueError('event_cadence requires windows and maximum_outputs')
    start = cfg['simulation'].get('start_redshift')
    if start is None or start == 'auto':
        raise ValueError('Event cadence requires a resolved user-selected start epoch')
    clock = {'time_begin': 1/(1+float(start)), 'time_max': 1/(1+float(cfg['simulation']['end_redshift'])),
             'maximum_timestep': float(cfg['simulation']['max_size_timestep_ln_a']), 'timebins': 29, 'comoving': True,
             'output_mode': 'exact_synchronized'}
    plan = merge_and_quantize_native_output_times([1/(1+z) for z in backbone], cadence['windows'], clock,
        maximum_outputs=cadence['maximum_outputs'])
    return [1/a-1 for a in plan['requested_times']]


def _backbone_snapshot_redshifts(cfg):
    """Construct the explicit saved redshift schedule; a=1/(1+z) controls GADGET output times."""
    s = cfg["simulation"]
    explicit = s.get("snapshot_redshift_list")
    if explicit is not None:
        if not isinstance(explicit, list) or not explicit or len(explicit) > 1100:
            raise ValueError("snapshot_redshift_list must contain 1–1100 redshifts")
        try:
            redshifts = [float(value) for value in explicit]
        except (TypeError, ValueError) as exc:
            raise ValueError("snapshot_redshift_list must contain numeric redshifts") from exc
        end_z = float(s.get("end_redshift", 0.0))
        start = s.get("start_redshift", "auto")
        if (any(not math.isfinite(z) or z < end_z for z in redshifts)
                or any(a <= b for a, b in zip(redshifts, redshifts[1:]))
                or (start != "auto" and redshifts[0] > float(start))):
            raise ValueError("snapshot_redshift_list must decrease uniquely from z_start to z_end")
        if not math.isclose(redshifts[-1], end_z, rel_tol=0, abs_tol=1e-8):
            raise ValueError("snapshot_redshift_list must include the simulation end redshift")
        return redshifts
    zmax = float(s["snapshot_z_max"])
    zmin = float(s["snapshot_z_min"])
    dz = float(s["snapshot_dz"])
    if dz <= 0 or zmax < zmin:
        raise ValueError("Snapshot redshifts require z_max >= z_min and dz > 0")
    end_z = float(s.get("end_redshift", 0.0))
    if end_z < 0 or zmin < end_z:
        raise ValueError("The last requested snapshot must be at or above the simulation end redshift")
    start = s.get("start_redshift", "auto")
    if start != "auto" and zmax > float(start):
        raise ValueError("The first requested snapshot cannot be earlier than the initial-condition redshift")
    steps = (zmax - zmin) / dz
    n = int(round(steps))
    if not math.isclose(steps, n, rel_tol=0.0, abs_tol=1.0e-9):
        raise ValueError("Snapshot redshift range must be an exact multiple of dz")
    if n + 1 > 1100:
        raise ValueError("GADGET-4 supports at most 1100 requested output times in this build")
    return [round(zmax - i * dz, 10) for i in range(n + 1)]


def resolve_gadget_ic_files(prefix, *, maximum_shards):
    """Resolve the exact native HDF5 launch set using its declared shard count.

    Never infer completeness from a wildcard match or a lone first shard.
    Existing shared count/publication validators retain their semantics.
    """
    import h5py
    from core.data import validate_native_shard_counts,verify_native_hdf5_set
    if type(maximum_shards)is not int or maximum_shards<1:
        raise ValueError('Exact positive native IC shard bound required')
    prefix=Path(prefix).expanduser().resolve()
    single=Path(str(prefix)+'.hdf5');first=Path(str(prefix)+'.0.hdf5')
    if single.exists() and first.exists():raise ValueError('Native IC prefix has ambiguous single-file and sharded layouts')
    if single.is_file():paths=[single]
    elif first.is_file():
        with h5py.File(first,'r') as native:
            raw=np.asarray(native['Header'].attrs.get('NumFilesPerSnapshot'))
            if raw.ndim!=0 or raw.dtype.kind not in 'ui' or not 1<=int(raw)<=maximum_shards:
                raise ValueError('Native IC shard count is invalid or exceeds its allocation')
            count=int(raw)
        paths=[Path(str(prefix)+'.%d.hdf5'%index) for index in range(count)]
        if any(not path.is_file() for path in paths):raise ValueError('Native IC launch set is missing a declared shard')
        if Path(str(prefix)+'.%d.hdf5'%count).exists():raise ValueError('Native IC launch set has an extra contiguous shard')
    else:raise FileNotFoundError('Native IC launch prefix has no HDF5 source: '+str(prefix))
    if len(paths)>maximum_shards:raise MemoryError('Native IC shard inventory exceeds its allocation')
    validate_native_shard_counts(paths,'NumPart_Total')
    verify_native_hdf5_set(paths)
    return paths


def gadget_ic_prefix(path):
    """Return GADGET's HDF5 input prefix, which must omit `.hdf5` suffixes."""
    source = Path(path).expanduser()
    if source.is_dir():
        files = sorted(source.glob("*.hdf5"))
        if not files:
            raise FileNotFoundError("No HDF5 IC files found in " + str(source))
        source = files[0]
    name = source.name
    if name.endswith(".0.hdf5"):
        name = name[:-len(".0.hdf5")]
    elif name.endswith(".hdf5"):
        name = name[:-len(".hdf5")]
    return str(source.with_name(name))


def write_output_list(path, redshifts):
    """Validate and durably publish an increasing GADGET scale-factor schedule."""
    epochs = []
    for z in redshifts:
        if (isinstance(z, (bool, np.bool_)) or
                not isinstance(z, (int, float, np.integer, np.floating)) or
                not math.isfinite(float(z)) or float(z) <= -1.0):
            raise ValueError('Output redshifts must be finite numeric values greater than -1')
        epochs.append(1.0 / (1.0 + float(z)))
    if not epochs or any(right <= left for left, right in zip(epochs, epochs[1:])):
        raise ValueError('Output scale factors must be nonempty and strictly increasing')
    text = ''.join('%.12e\n' % a for a in epochs)
    saved = [float(line) for line in text.splitlines()]
    if any(right <= left for left, right in zip(saved, saved[1:])):
        raise ValueError('Output scale factors collide at native schedule precision')
    atomic_text(path, text)


def particle_mass_msun_h(cfg, dm_only=True):
    """Compute the particle mass from mean cosmological density and comoving box volume, in Msun/h."""
    c = cfg["cosmology"]
    h = float(c["H0"]) / 100.0
    omega_m = (float(c["omega_b"]) + float(c["omega_cdm"])) / h**2
    if dm_only:
        omega = omega_m
    else:
        omega = float(c["omega_cdm"]) / h**2
    rho_crit = RHO_CRIT_MSUN_H_MPC_H3
    L = float(cfg["simulation"]["box_mpc_h"])
    N = int(cfg["simulation"]["particles_per_dimension"])**3
    return omega * rho_crit * L**3 / N


def auto_softening_kpc_h(cfg):
    """Choose the configured spacing-based gravitational softening, in comoving kpc/h."""
    mean_spacing = mean_particle_spacing_kpc_h(cfg)
    # 1/40 mean inter-particle spacing is a conservative baseline, not a law.
    return mean_spacing / 40.0


def mean_particle_spacing_kpc_h(cfg):
    """Return the unperturbed cubic-load spacing in comoving kpc/h."""
    box_size = float(cfg["simulation"]["box_mpc_h"])
    particles_per_dimension = int(cfg["simulation"]["particles_per_dimension"])
    return box_size * 1000.0 / particles_per_dimension


def softening_scales(cfg):
    """Resolve both GADGET softenings and report their mean-spacing ratios."""
    simulation = cfg["simulation"]
    comoving = float(simulation["softening_kpc_h"])
    maximum_physical = float(simulation["softening_max_physical_kpc_h"])
    mean_spacing = mean_particle_spacing_kpc_h(cfg)
    return {
        "comoving_kpc_h": comoving,
        "maximum_physical_kpc_h": maximum_physical,
        "mean_particle_spacing_kpc_h": mean_spacing,
        "comoving_fraction_of_mean_spacing": comoving / mean_spacing,
        "maximum_physical_fraction_of_mean_spacing": maximum_physical / mean_spacing,
        "particle_type_and_mass_classes": softening_plan(simulation),
        "selection_note": "explicit comoving and maximum physical softenings; convergence remains to be demonstrated",
    }


HYDRO_STAR_FORMATION_PARAMETERS = {
    "max_sfr_timescale": ("MaxSfrTimescale", 1.5),
    "temp_clouds": ("TempClouds", 1000.0),
    "temp_supernova": ("TempSupernova", 1.0e8),
    "factor_evp": ("FactorEVP", 1000.0),
    "factor_sn": ("FactorSN", 0.1),
    "crit_phys_density": ("CritPhysDensity", 0.0),
    "crit_overdensity": ("CritOverDensity", 57.7),
}

HYDRO_SPH_PARAMETERS = {
    "desired_neighbors": ("DesNumNgb", 64.0),
    "neighbor_deviation": ("MaxNumNgbDeviation", 2.0),
    "artificial_viscosity": ("ArtBulkViscConst", 1.0),
    "initial_gas_temperature": ("InitGasTemp", 1.0e4),
}


def hydro_sph_parameters(hydro, *, start_scale_factor=None):
    """Resolve SPH controls, optionally scaling a reference temperature as a^-2.

    For a monatomic ideal gas this keeps the homogeneous adiabatic entropy
    constant across starts. It does not remove density perturbations or replace
    a thermal-history calculation. Without a reference, temperature is at start.
    """
    values = {}
    for key, (native, default) in HYDRO_SPH_PARAMETERS.items():
        raw = hydro.get(key, default)
        try:
            value = float(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError("hydro.%s must be a finite number" % key) from exc
        if isinstance(raw, bool) or not math.isfinite(value) or value < 0:
            raise ValueError("hydro.%s must be finite and non-negative" % key)
        values[native] = value
    if values["DesNumNgb"] <= 0 or values["MaxNumNgbDeviation"] >= values["DesNumNgb"]:
        raise ValueError("SPH requires positive desired_neighbors and 0 <= neighbor_deviation < desired_neighbors")
    reference = hydro.get("initial_gas_temperature_reference_scale_factor")
    if reference is not None:
        if isinstance(reference, bool) or not isinstance(reference, (int, float)) or not math.isfinite(reference) or not 0 < reference <= 1:
            raise ValueError("Gas temperature reference scale factor must be finite in (0, 1]")
        if start_scale_factor is not None:
            if isinstance(start_scale_factor, bool) or not math.isfinite(start_scale_factor) or not 0 < start_scale_factor <= 1:
                raise ValueError("Gas temperature start scale factor must be finite in (0, 1]")
            values["InitGasTemp"] *= (reference / start_scale_factor) ** 2
            if not math.isfinite(values["InitGasTemp"]):
                raise ValueError("Scaled initial gas temperature must be finite")
    return values


def hydro_star_formation_parameters(hydro):
    """Resolve and check the pinned Springel-Hernquist model parameters."""
    values = {}
    for key, (native, default) in HYDRO_STAR_FORMATION_PARAMETERS.items():
        raw = hydro.get(key, default)
        try:
            value = float(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError("hydro.%s must be a finite number" % key) from exc
        if isinstance(raw, bool) or not math.isfinite(value) or value < 0:
            raise ValueError("hydro.%s must be finite and non-negative" % key)
        if key != "crit_phys_density" and value == 0:
            raise ValueError("hydro.%s must be positive" % key)
        if key == "factor_sn" and value >= 1:
            raise ValueError("hydro.factor_sn must be strictly between zero and one")
        values[native] = value
    # Same neutral/ionized molecular weights as set_units_sfr(), X_H=0.76.
    cold_energy = values["TempClouds"] * (1 + 3 * 0.76) / 4
    sn_energy = values["TempSupernova"] * (8 - 5 * (1 - 0.76)) / 4
    beta = values["FactorSN"]
    if beta * sn_energy <= (1 - beta) * cold_energy:
        raise ValueError("Star-formation parameters give a non-positive net supernova heating energy")
    seed = hydro.get('stellar_random_seed', 0)
    if type(seed) is not int or not 0 <= seed < 2**64:
        raise ValueError('hydro.stellar_random_seed must be an exact uint64 integer')
    values['StellarRandomSeed'] = str(seed)
    return values


def validate_hydro_configuration(cfg, require_runtime_assets=True):
    """Reject hydro choices that this generator/evolution path cannot honor."""
    s = cfg["simulation"]
    hydro = cfg.get("hydro", {})
    sph = hydro_sph_parameters(hydro)
    if sph["DesNumNgb"] >= int(s["particles_per_dimension"])**3:
        raise ValueError("hydro.desired_neighbors must be smaller than the initial gas-particle count")
    if s.get("ic_engine_hydro") not in {'monofonic','native_music'}:
        raise ValueError("Hydro ICs require monofonic or an audited native_music handoff")

    expected_switches = {
        "cooling": True,
        "star_formation": True,
        "uv_background": True,
        "require_species_transfer_functions": True,
        "allow_gadget_particle_split_approximation": False,
    }
    switches = {}
    for name, default in expected_switches.items():
        value = hydro.get(name, default)
        if type(value) is not bool:
            raise ValueError("hydro.%s must be a YAML boolean" % name)
        switches[name] = value
    if not switches["require_species_transfer_functions"]:
        raise ValueError(
            "The supported mixed-species IC path requires CLASS baryon+CDM transfer functions; "
            "the no-species-transfer approximation is not implemented"
        )
    if switches["allow_gadget_particle_split_approximation"]:
        raise ValueError(
            "GADGET particle splitting is not a substitute for mixed-species transfer functions and is not implemented here"
        )
    if switches["uv_background"] != switches["cooling"]:
        raise ValueError(
            "This GADGET build reads the UV ionization/heating history from TREECOOL whenever cooling is enabled; "
            "hydro.uv_background must match hydro.cooling because the two cannot be switched independently"
        )
    if switches["star_formation"] and not switches["cooling"]:
        raise ValueError("The configured Springel-Hernquist star-formation model requires hydro.cooling: true")
    if switches["star_formation"]:
        hydro_star_formation_parameters(hydro)

    lpt_order = s.get("lpt_order_hydro", 3)
    if isinstance(lpt_order, bool) or not isinstance(lpt_order, int) or lpt_order not in (1, 2, 3):
        raise ValueError("simulation.lpt_order_hydro must be the integer 1, 2, or 3")
    if switches["cooling"] and require_runtime_assets:
        treecool = _treecool_file(cfg)
        if treecool is None:
            raise FileNotFoundError("Hydro cooling/UV background requires a GADGET TREECOOL table")
    elif switches["cooling"]:
        treecool = _treecool_file(cfg)
    else:
        treecool = None
    if treecool is not None:
        validate_treecool_table(treecool)
    return switches


def ensure_run_layout(run_dir):
    """Create the per-run input/data/results directories without fabricating scientific outputs."""
    run_dir = Path(run_dir)
    for name in ["input", "data", "results"]:
        (run_dir / name).mkdir(parents=True, exist_ok=True)
    return run_dir




def existing_ic_source_run(path, model):
    """Find the run directory that owns an IC and its Boltzmann provenance."""
    source = Path(path).expanduser().resolve()
    candidates = [source.parent if source.is_file() else source]
    candidates.extend(candidates[0].parents)
    metadata_name = str(model).lower() + "_cosmology.json"
    for candidate in candidates:
        if (candidate / "input" / metadata_name).is_file():
            return candidate
    raise FileNotFoundError(
        "--evolve-only needs an IC inside its source run, with input/%s and the matching solver artifacts"
        % metadata_name
    )


def load_existing_ic_cosmology(path, model):
    """Load and verify the original linear-theory inputs for evolve-only runs."""
    source_run = existing_ic_source_run(path, model)
    metadata_path = source_run / "input" / (str(model).lower() + "_cosmology.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict) or str(metadata.get("model", "")).lower() != str(model).lower():
        raise ValueError("The IC source run has mismatched or malformed cosmology metadata")
    manifest_path = source_run / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("The IC source run has no manifest for artifact verification: " + str(source_run))
    source_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(source_manifest, dict) or str(source_manifest.get("model", "")).lower() != str(model).lower():
        raise ValueError("The IC source manifest does not identify the requested cosmology model")
    input_artifacts = source_manifest.get("input_artifacts")
    if not isinstance(input_artifacts, dict):
        raise ValueError("The IC source manifest has no checksummed input-artifact records")

    required = ["pk_file", "background_file"]
    if model in {'wdm'}:
        required.append("transfer_file")
    missing = [name for name in required if not metadata.get(name)]
    if missing:
        raise ValueError("The IC source cosmology metadata is missing required artifacts: " + ", ".join(missing))
    manifest_names = {
        "pk_file": "linear_power_spectrum",
        "background_file": "boltzmann_background",
        "transfer_file": "transfer_functions",
        "gadget_background_file": "gadget_background_table",
    }
    resolved = {}
    for name in required:
        artifact = Path(str(metadata[name])).expanduser()
        if not artifact.is_absolute():
            artifact = source_run / artifact
        artifact = artifact.resolve()
        if not artifact.is_file():
            raise FileNotFoundError("The IC source %s artifact is missing: %s" % (name, artifact))
        record = input_artifacts.get(manifest_names[name])
        if not isinstance(record, dict) or not record.get("sha256"):
            raise ValueError("The IC source manifest lacks a checksum for %s" % name)
        manifest_artifact = Path(str(record.get("path", ""))).expanduser()
        if not manifest_artifact.is_absolute():
            manifest_artifact = source_run / manifest_artifact
        manifest_artifact = manifest_artifact.resolve()
        if manifest_artifact != artifact or file_sha256(artifact) != record["sha256"]:
            raise ValueError("The IC source %s artifact does not match its manifest checksum" % name)
        resolved[name] = str(artifact)
    if model == "wdm":
        reference = input_artifacts.get("wdm_lcdm_reference_spectrum")
        gate = source_manifest.get("wdm_resolution_gate", {})
        if (not isinstance(reference, dict) or not reference.get("path") or not reference.get("sha256")
                or not isinstance(gate, dict)
                or gate.get("status") not in {"PASS", "EXPLICITLY_ALLOWED_BENCHMARK"}):
            raise ValueError("The WDM IC source lacks a valid cutoff-resolution result and reference spectrum")
        reference_path = Path(str(reference["path"])).expanduser()
        if not reference_path.is_absolute():
            reference_path = source_run / reference_path
        reference_path = reference_path.resolve()
        if not reference_path.is_file() or file_sha256(reference_path) != reference["sha256"]:
            raise ValueError("The IC source WDM reference spectrum is missing or changed")
        metadata["wdm_lcdm_reference_spectrum"] = {
            **reference,
            "path": str(reference_path),
        }
    metadata.update(resolved)
    if metadata.get("pk_sha256") and file_sha256(metadata["pk_file"]) != metadata["pk_sha256"]:
        raise ValueError("The IC source linear power spectrum no longer matches its recorded SHA-256")
    metadata["source"] = "existing_ic; CLASS was not rerun"
    metadata["source_run"] = str(source_run)
    metadata["source_manifest_sha256"] = file_sha256(manifest_path)
    return source_run, metadata


def external_lcdm_ic_metadata(path, cfg):
    """Use an external collisionless IC without inventing linear-theory provenance.

    This path uses the declared GADGET units (kpc/h, 10^10 Msun/h, stored
    velocities v_pec/sqrt(a)). Header cosmology must match the configuration.
    Unknown phases, LPT order, and input P(k) remain unknown. Structural and
    mass validation still run, and GADGET measures the actual IC spectrum.
    WDM require model-specific background/transfer/thermal provenance and
    retain their existing strict source-run route.
    """
    if cfg["cosmology"]["model"] != "lcdm" or cfg["simulation"]["physics"] != "dm":
        raise ValueError("External ICs without solver provenance currently support LCDM DM only; WDM/hydro require their validated source artifacts")
    files = initial_condition_files(path)
    if not files:
        raise FileNotFoundError("No external GADGET HDF5 IC files: " + str(path))
    h = float(cfg["cosmology"]["H0"]) / 100.0
    omega_m = (float(cfg["cosmology"]["omega_b"]) + float(cfg["cosmology"]["omega_cdm"])) / h**2
    radiation = omega_radiation_from_config(cfg)
    for file in files:
        with h5py.File(file, "r") as handle:
            attrs = handle["Header"].attrs
            parameters = handle["Parameters"].attrs if "Parameters" in handle else {}
            for name, expected, tolerance in (("UnitLength_in_cm", KPC_CM, 1.e-6),
                                               ("UnitMass_in_g", 1.e10*MSUN_G, 1.e-3),
                                               ("UnitVelocity_in_cm_per_s", 1.e5, 1.e-6)):
                if name in parameters and not math.isclose(float(parameters[name]), expected, rel_tol=tolerance):
                    raise ValueError("External IC uses incompatible " + name + "; convert the IC into the configured GADGET units")
            if "ComovingIntegrationOn" in parameters and int(parameters["ComovingIntegrationOn"]) != 1:
                raise ValueError("External IC must use comoving cosmological coordinates")
            for name, expected in (("HubbleParam", h), ("Omega0", omega_m)):
                if name not in attrs:
                    raise ValueError("External IC lacks cosmology header field " + name)
                actual = _hdf5_scalar_attr(attrs[name], name)
                if not math.isclose(actual, expected, rel_tol=1.e-5, abs_tol=1.e-8):
                    raise ValueError("External IC %s=%g differs from configuration %g" % (name, actual, expected))
            if "OmegaLambda" in attrs:
                actual = _hdf5_scalar_attr(attrs["OmegaLambda"], "OmegaLambda")
                # Older GADGET ICs often close Omega_m+Omega_Lambda=1,
                # omitting today's small radiation density in this header.
                if not math.isfinite(actual) or min(abs(actual-(1-omega_m)), abs(actual-(1-omega_m-radiation))) > 1.e-5:
                    raise ValueError("External IC OmegaLambda differs from the configured flat LCDM cosmology")
    return {
        "model": "lcdm", "source": "existing_ic; external GADGET HDF5; CLASS was not run",
        "start_redshift": ic_header_redshift(path), "omega_radiation0": radiation,
        "linear_theory_validation": {"status": "UNAVAILABLE", "reason": "External IC has no original spectrum or transfer provenance"},
        "external_ic_files": [{"path": str(file.resolve()), "sha256": file_sha256(file)} for file in files],
    }


def validate_evolve_only_source_configuration(source_run, cfg, model):
    """Require reused ICs to evolve under their original cosmology and IC setup."""
    source_run = Path(source_run).expanduser().resolve()
    source_config_path = source_run / "config.yaml"
    source_manifest_path = source_run / "manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if not source_config_path.is_file():
        raise FileNotFoundError("The IC source run has no saved config.yaml")
    source_config_sha = source_manifest.get("config_sha256")
    if not source_config_sha or file_sha256(source_config_path) != source_config_sha:
        raise ValueError("The IC source configuration does not match its manifest checksum")
    source_cfg = yaml.safe_load(source_config_path.read_text(encoding="utf-8"))
    if not isinstance(source_cfg, dict):
        raise ValueError("The IC source configuration is not a mapping")
    if (str(source_cfg.get("cosmology", {}).get("model", "")).lower() != str(model).lower()
            or str(cfg.get("cosmology", {}).get("model", "")).lower() != str(model).lower()):
        raise ValueError("The IC source and evolve-only configurations identify different cosmology models")

    def normalized(source):
        result = {}
        result["cosmology"] = copy.deepcopy(source.get("cosmology", {}))
        for key in ("omega_radiation", "gadget_background_table"):
            result["cosmology"].pop(key, None)
        result["ede"] = copy.deepcopy(source.get("ede", {}))
        result["wdm"] = copy.deepcopy(source.get("wdm", {}))
        for key in ("lcdm_reference_pk", "lcdm_reference_redshift"):
            result["wdm"].pop(key, None)
        simulation_keys = [
            "box_mpc_h", "particles_per_dimension", "random_seed", "pm_grid", "physics",
        ]
        if str(source.get("simulation", {}).get("physics", "dm")).lower() == "hydro":
            simulation_keys.extend(("lpt_order_hydro", "ic_engine_hydro"))
        else:
            simulation_keys.extend(("lpt_order_dm", "ic_engine_dm"))
        result["simulation"] = {
            key: source.get("simulation", {}).get(key) for key in simulation_keys
        }
        if str(source.get("simulation", {}).get("physics", "dm")).lower() == "hydro":
            result["hydro"] = copy.deepcopy(source.get("hydro", {}))
        return result

    if normalized(source_cfg) != normalized(cfg):
        raise ValueError(
            "--evolve-only must preserve the IC source cosmology, model parameters, particle load, and IC generator settings"
        )






def validate_species_transfer_provenance(table_path, cosmo, model, start_redshift):
    """Ensure a reused species table came from this model's exact CLASS run."""
    table_path = Path(table_path).expanduser().resolve()
    sidecar_path = Path(str(table_path) + ".json")
    if not table_path.is_file() or not sidecar_path.is_file():
        raise FileNotFoundError("The source species-transfer table and provenance sidecar are both required")
    metadata = json.loads(sidecar_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict) or metadata.get("status") != "READY" or metadata.get("model") != model:
        raise ValueError("The source species-transfer provenance does not match model %s" % model)
    if (Path(str(metadata.get("validation_table", ""))).expanduser().resolve() != table_path
            or metadata.get("validation_table_sha256") != file_sha256(table_path)):
        raise ValueError("The source species-transfer table checksum does not match its sidecar")
    transfer_path = Path(str(metadata.get("source_transfer_file", ""))).expanduser().resolve()
    expected_transfer = Path(str(cosmo.get("transfer_file", ""))).expanduser().resolve()
    if (transfer_path != expected_transfer or not transfer_path.is_file()
            or metadata.get("source_transfer_sha256") != file_sha256(transfer_path)):
        raise ValueError("The species-transfer table was not derived from this IC's recorded CLASS transfer file")
    if (not class_redshift_matches(metadata.get("source_redshift"), start_redshift)
            or not math.isclose(float(metadata.get("expected_redshift", float("nan"))), float(start_redshift),
                                rel_tol=0.0, abs_tol=1.0e-8 * (1.0 + float(start_redshift)))):
        raise ValueError("The source species-transfer table redshift does not match the IC epoch")
    return metadata


def _project_source_hashes():
    """Record first-party sources and the setup inputs used by this checkout.

    Keys remain relative to core/ for existing manifest readers. Project-level
    setup files therefore use ../ paths; upstream engines have their own pins.
    Nested Python packages must enter provenance just like top-level modules.
    """
    paths = []
    for directory, folders, files in os.walk(ROOT):
        folders[:] = [name for name in folders if name not in
                      {'software', 'patches', '__pycache__', '.git'}]
        paths.extend(Path(directory)/name for name in files if name.endswith(('.py', '.cc')))
    paths.extend([ROOT/'config.yaml', ROOT/'software-revisions',
                  ROOT.parent/'bootstrap.py', ROOT.parent/'commands.py',
                  ROOT.parent/'scripts/setup.sh', ROOT.parent/'configs/core_source_layout.json'])
    paths.extend(sorted((ROOT.parent/'configs/runtime').glob('*.json')))
    paths.extend(sorted((ROOT/'patches').rglob('*.patch')))
    return {
        os.path.relpath(path, ROOT): file_sha256(path)
        for path in sorted(set(paths))
        if path.is_file()
    }


def _pinned_software_revisions():
    """Read declared vendor revisions even when the packaged trees lack .git."""
    path = ROOT / "software-revisions"
    revisions = {}
    if not path.is_file():
        return revisions
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        fields = line.split()
        if len(fields) >= 3 and not fields[0].startswith("#"):
            revisions[fields[0]] = {"repository": fields[1], "declared_commit": fields[2]}
    return revisions


def collect_boltzmann_solver_runs(run_dir):
    """Load the immutable provenance sidecars for every CLASS call."""
    directory = Path(run_dir) / "input" / "boltzmann_runs"
    records = []
    if not directory.is_dir():
        return records
    for path in sorted(directory.glob("*_provenance.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            records.append({
                "status": "UNREADABLE_PROVENANCE_RECORD",
                "record_path": str(path.resolve()),
                "reason": str(exc),
            })
            continue
        records.append({
            "status": value.get("status", "UNKNOWN"),
            "engine": value.get("engine"),
            "requested_redshift": value.get("requested_redshift"),
            "record_path": str(path.resolve()),
            "record_sha256": file_sha256(path),
            "executable_sha256": (value.get("executable") or {}).get("sha256"),
            "input_ini_sha256": (value.get("input_ini") or {}).get("sha256"),
            "source_tree_sha256": (value.get("source_tree") or {}).get("sha256"),
            "pinned_revision": value.get("pinned_revision"),
        })
    return records


def _write_json_atomic(path, value):
    """Publish finite JSON through the shared file/directory fsync boundary."""
    atomic_json(path, value)


def _record_gadget_build(run_dir, stage, signature):
    """Keep the exact run-specific GADGET build recipe in the run manifest."""
    path = Path(run_dir) / "manifest.json"
    if not path.is_file():
        return
    with open(path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    manifest.setdefault("gadget_builds", {})[stage] = signature
    _write_json_atomic(path, manifest)


def record_manifest(run_dir, key, value):
    """Update one top-level manifest field using an atomic replace."""
    record_manifest_fields(run_dir, {key: value})


def record_manifest_fields(run_dir, fields):
    """Commit related stage fields together; callers serialize run ownership."""
    path = Path(run_dir) / "manifest.json"
    with open(path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    manifest.update(fields)
    _write_json_atomic(path, manifest)


def _gadget_logged_internal_memory_mb(log_path):
    """Return the largest GADGET allocator value printed in a run log.

    This is GADGET's internal allocation accounting, not process RSS and not
    total MPI/job memory. Keep that distinction explicit in saved provenance.
    """
    maximum = None
    try:
        with Path(log_path).open("r", encoding="utf-8", errors="replace") as stream:
            for line in stream:
                match = re.search(r"presently allocated=([0-9.eE+-]+)\s*MB", line)
                if not match:
                    continue
                value = float(match.group(1))
                if math.isfinite(value) and value >= 0:
                    maximum = value if maximum is None else max(maximum, value)
    except OSError:
        return None
    return maximum


def _gadget_execution_measurement_record(log_path, attempt_kind, wall_seconds):
    """Build one measured GADGET process-time and internal-memory record."""
    memory_mb = _gadget_logged_internal_memory_mb(log_path)
    return {
        "attempt_kind": str(attempt_kind),
        "wall_seconds": float(wall_seconds),
        "log": str(Path(log_path).resolve()),
        "log_sha256": file_sha256(log_path) if Path(log_path).is_file() else None,
        "maximum_logged_internal_allocation_mb": memory_mb,
        "memory_measurement": (
            "maximum GADGET Mem.getAllocatedBytesInMB() value printed in the log; excludes process RSS and MPI runtime memory"
        ),
    }


def _append_gadget_execution_measurement(run_dir, log_path, attempt_kind, wall_seconds):
    """Append measured GADGET execution metrics to the run manifest."""
    path = Path(run_dir) / "manifest.json"
    with path.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    measurements = manifest.get("gadget_execution_measurements", [])
    if not isinstance(measurements, list):
        measurements = []
    measurements.append(_gadget_execution_measurement_record(log_path, attempt_kind, wall_seconds))
    manifest["gadget_execution_measurements"] = measurements
    _write_json_atomic(path, manifest)


def validate_gadget_used_values(parameter_path, used_values_path):
    """Verify every submitted GADGET parameter appears unchanged after parsing."""
    def parse(path):
        values = {}
        for line_number, line in enumerate(Path(path).read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            fields = stripped.split(None, 1)
            if len(fields) != 2 or fields[0] in values:
                raise ValueError("malformed or duplicate GADGET value on line %d in %s" % (line_number, path))
            values[fields[0]] = fields[1].strip()
        return values

    parameter_path = Path(parameter_path)
    used_values_path = Path(used_values_path)
    if not parameter_path.is_file() or not used_values_path.is_file():
        raise FileNotFoundError("GADGET parameter input or parsed used-values record is missing")
    supplied = parse(parameter_path)
    consumed = parse(used_values_path)
    missing = sorted(set(supplied) - set(consumed))
    if missing:
        raise ValueError("GADGET did not record submitted parameters as parsed inputs: " + ", ".join(missing))
    for key, expected_text in supplied.items():
        actual_text = consumed[key]
        try:
            expected_number = float(expected_text)
            actual_number = float(actual_text)
        except ValueError:
            if expected_text != actual_text:
                raise ValueError("GADGET parsed parameter %s=%r, expected %r" % (key, actual_text, expected_text))
        else:
            if not (math.isfinite(expected_number) and math.isfinite(actual_number)):
                raise ValueError("GADGET parsed parameter %s has a non-finite value" % key)
            if expected_number == 0.0:
                printed_precision_bound = 0.0
            else:
                # The pinned GADGET writer serializes PARAM_DOUBLE with %g,
                # which prints six significant digits. Compare at that exact
                # documented serialization resolution, not a loose relative
                # tolerance that would conceal changed inputs.
                printed_precision_bound = 0.500001 * 10.0 ** (math.floor(math.log10(abs(expected_number))) - 5)
            if abs(expected_number - actual_number) > printed_precision_bound:
                raise ValueError("GADGET parsed parameter %s=%r, expected %r" % (key, actual_text, expected_text))
    return {
        "status": "PASS",
        "parameter_file": str(parameter_path.resolve()),
        "used_values_file": str(used_values_path.resolve()),
        "used_values_sha256": file_sha256(used_values_path),
        "submitted_parameter_count": len(supplied),
        "comparison_precision": "GADGET's pinned %g output format (six significant digits)",
    }


def cosmology_input_artifacts(cosmology_meta):
    """Build checksummed artifact records from the cosmology metadata used by a run."""
    artifacts = {}
    for name, metadata_key in (
        ("linear_power_spectrum", "pk_file"),
        ("transfer_functions", "transfer_file"),
        ("boltzmann_background", "background_file"),
        ("cosmology_products", "cosmology_products_file"),
        ("gadget_background_table", "gadget_background_file"),
        ("imported_spectrum_source", "input_file"),
        ("boltzmann_solver_input", "solver_ini_file"),
        ("boltzmann_solver_log", "solver_log_file"),
    ):
        value = cosmology_meta.get(metadata_key)
        if value and Path(value).is_file():
            if name=='cosmology_products':
                from core.data import verify_hdf5_product
                verify_hdf5_product(value)
            elif name in ('linear_power_spectrum','gadget_background_table'):verify_numeric_table(value)
            artifacts[name] = {
                "path": str(Path(value).resolve()),
                "sha256": cosmology_meta.get("pk_sha256") if metadata_key == "pk_file" else file_sha256(value),
            }
    if cosmology_meta.get("pk_sha256") and "linear_power_spectrum" in artifacts:
        artifacts["linear_power_spectrum"]["sha256"] = cosmology_meta["pk_sha256"]
    if cosmology_meta.get("output_redshift") is not None:
        artifacts.setdefault("linear_power_spectrum", {})["redshift"] = cosmology_meta["output_redshift"]
    if cosmology_meta.get("power_spectrum_rescaled") is not None:
        artifacts.setdefault("linear_power_spectrum", {})["power_spectrum_rescaled"] = cosmology_meta["power_spectrum_rescaled"]
    wdm_reference = cosmology_meta.get("wdm_lcdm_reference_spectrum")
    if isinstance(wdm_reference, dict) and wdm_reference.get("path") and wdm_reference.get("sha256"):
        artifacts["wdm_lcdm_reference_spectrum"] = {
            "path": str(Path(wdm_reference["path"]).expanduser().resolve()),
            "sha256": wdm_reference["sha256"],
            "redshift": wdm_reference.get("redshift"),
        }
    return artifacts


def write_manifest(run_dir, cfg, model, start_z, cosmology_meta=None):
    """Publish resolved run parameters, sources, unit conventions and current run state."""
    import datetime
    from core.sidm_physics import sidm_manifest_record

    cosmology_meta = cosmology_meta or {}
    config_path = Path(run_dir) / "config.yaml"
    artifacts = cosmology_input_artifacts(cosmology_meta)

    h = float(cfg["cosmology"]["H0"]) / 100.0
    omega_b = float(cfg["cosmology"]["omega_b"]) / h**2
    omega_cdm = float(cfg["cosmology"]["omega_cdm"]) / h**2
    hydro = cfg["simulation"]["physics"] == "hydro"
    mass_dm = particle_mass_msun_h(cfg, dm_only=False) if hydro else particle_mass_msun_h(cfg, True)
    mass_gas = mass_dm * omega_b / omega_cdm if hydro and omega_cdm > 0 else None
    solver_runs = collect_boltzmann_solver_runs(run_dir)
    solver_provenance = cosmology_meta.get("solver_provenance")
    if not solver_runs:
        solver_status = (
            "NOT_RUN_EXISTING_IC" if str(cosmology_meta.get("source", "")).startswith("existing_ic")
            else "NOT_RUN"
        )
    elif solver_provenance and solver_provenance.get("status") == "RECORDED" and all(
        item.get("status") == "RECORDED" for item in solver_runs
    ):
        solver_status = "RECORDED"
    else:
        solver_status = "PARTIAL"

    manifest = {
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "project_name": cfg.get("project", {}).get("name", "DarkMatterSim"),
        "status": "PREPARING",
        "run_name": Path(run_dir).name,
        "model": model,
        "start_redshift": start_z,
        "cosmology_parameters": cfg.get("cosmology", {}),
        "model_parameters": {
            "ede": None,
            "wdm": cfg.get("wdm", {}) if model == "wdm" else None,
        },
        "simulation_parameters": cfg.get("simulation", {}),
        "hardware_parameters": cfg.get("hardware", {}),
        "hydro_parameters": cfg.get("hydro", {}) if cfg["simulation"]["physics"] == "hydro" else None,
        "stellar_evolution_parameters":cfg.get('stellar_evolution',{'enabled':False}),
        "chemistry_parameters":cfg.get('chemistry',{'enabled':False}),
        "sidm_parameters":cfg.get('sidm',{'enabled':False}),
        "sidm_model_record":sidm_manifest_record(cfg.get('sidm',{'enabled':False})),
        "halo_parameters": cfg.get("halos", {}),
        "analysis_parameters": cfg.get("analysis", {}),
        "visualization_parameters": cfg.get("visualization", {}),
        "box_mpc_h": cfg["simulation"]["box_mpc_h"],
        "particles_per_dimension": cfg["simulation"]["particles_per_dimension"],
        "random_seed": cfg["simulation"]["random_seed"],
        "physics": cfg["simulation"]["physics"],
        "requested_mpi_ranks": mpi_ranks(cfg),
        "requested_openmp_threads": omp_threads(cfg),
        "requested_cpu_threads_total": mpi_ranks(cfg) * omp_threads(cfg),
        "cpu_thread_budget": cpu_thread_budget(cfg),
        "requested_build_jobs": build_jobs(cfg),
        "available_cpu_slots": cpu_count(),
        "memory_limit_gb": memory_limit_gb(cfg),
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "config_file": str(config_path.resolve()),
        "config_sha256": file_sha256(config_path) if config_path.is_file() else None,
        "project_source_inventory_version": 2,
        "project_source_sha256": _project_source_hashes(),
        "mass_gravity_convention": {
            "G_cgs": G_CGS, "adopted_solar_mass_g": MSUN_G, "Mpc_cm": MPC_CM,
            "rho_critical0_msun_h_per_mpc_h_cubed": RHO_CRIT_MSUN_H_MPC_H3,
        },
        "input_artifacts": artifacts,
        "software_revisions_file_sha256": file_sha256(ROOT / "software-revisions")
        if (ROOT / "software-revisions").is_file() else None,
        "software_revisions": _pinned_software_revisions(),
        "external_tool_inventory": external_tool_inventory(),
        "boltzmann_solver_provenance": solver_provenance,
        "boltzmann_solver_runs": solver_runs,
        "boltzmann_solver_status": solver_status,
        "cosmology_input_metadata": cosmology_meta,
        "particle_mass_msun_h_dm_only": particle_mass_msun_h(cfg, True),
        "particle_mass_msun_h_per_species": {
            "dark_matter": mass_dm,
            "gas": mass_gas,
        },
        "softening_kpc_h": softening_scales(cfg)["comoving_kpc_h"],
        "softening_scales": softening_scales(cfg),
        "snapshot_redshifts_requested": snapshot_redshifts(cfg),
        "warnings": [],
        "validation": {},
        "software_commits": {},
    }
    if model == "wdm":
        spectrum_source = str(cfg.get("spectrum", {}).get("source", "boltzmann")).strip().lower()
        wdm_model = cfg["wdm"].get("model")
        thermal = wdm_model == "wdm_thermal_relic"
        manifest["dark_matter_model"] = {
            "classification": wdm_model,
            "linear_spectrum_origin": (
                "CLASS thermal-relic ncdm" if spectrum_source == "boltzmann" and cfg["wdm"].get("mode") == "class_ncdm"
                else "externally supplied linear spectrum"
            ),
            "particle_thermal_velocities": thermal,
            "thermal_velocity_distribution": "Fermi-Dirac relic phase space" if thermal else None,
            "full_thermal_relic_phase_space": thermal,
            "phase_space_scope": (
                "CLASS thermal-relic linear transfers plus sampled primordial Fermi-Dirac particle momenta; Newtonian N-body evolution"
                if thermal else "linear transfer suppression evolved with cold particles"
            ),
            "second_order_lpt_velocity": (
                "standard pressureless N-GenIC approximation; scale-dependent CLASS response is applied only to the first-order velocity"
                if cfg["simulation"].get("physics") == "dm" else
                "monofonIC integrates scale-independent smooth-background growth; thermal random velocities are added separately"
            ),
        }
    if cfg["simulation"].get("physics") == "hydro":
        hydro = cfg.get("hydro", {})
        cooling = bool(hydro.get("cooling", True))
        star_formation = bool(hydro.get("star_formation", True))
        hydro_record = {
            "initial_condition_generator": "monofonIC",
            "transfer_functions": "separate CLASS baryon and CDM transfer components with linear relative-velocity correction",
            "relative_baryon_cdm_velocity_ic": True,
            "gas_dynamics": "GADGET-4 density-entropy SPH with synchronized source timesteps" if cfg.get("black_holes", {}).get("enabled", False) or cfg.get("stellar_evolution",{}).get("enabled",False) else "GADGET-4 pressure-entropy SPH",
            "radiative_cooling": cooling,
            "treecool_uv_background": cooling and bool(hydro.get("uv_background", True)),
            "star_formation": (
                "GADGET-4 Springel-Hernquist effective multiphase subgrid model"
                if star_formation else "disabled"
            ),
            "interpretation_limit": "subgrid star formation and feedback; individual stars and resolved stellar feedback are not modeled",
            "sph_parameters": hydro_sph_parameters(hydro, start_scale_factor=1.0 / (1.0 + choose_start_redshift(cfg))),
            "star_formation_parameters": hydro_star_formation_parameters(hydro) if star_formation else None,
        }
        treecool = _treecool_file(cfg) if cooling else None
        if treecool is not None:
            hydro_record["treecool_file"] = str(treecool.resolve())
            hydro_record["treecool_sha256"] = hashlib.sha256(treecool.read_bytes()).hexdigest()
            hydro_record["treecool_table"] = validate_treecool_table(treecool)
        manifest["hydro_model"] = hydro_record
        if cfg.get("black_holes", {}).get("enabled", False):
            manifest["black_hole_model"] = {
                "model": "project thermal Bondi/Eddington model", "parameters": cfg["black_holes"],
                "kernel": "fixed proper-radius normalized cubic spline", "merging": "quasi-circular bound hard binary with proper-time PN/Peters evolution and a near-merger fit gate" if cfg["black_holes"].get("dynamics",{}).get("enabled",False) else "disabled",
                "seeding": "densest eligible gas particle with BH-centered exclusion; full gas gravity mass retained as unresolved reservoir",
                "evolution": "live gas consumption, radiation mass loss, momentum transfer, thermal energy deposition; globally synchronized hydro/gravity",
                "validation_scope": "Current source is unbuilt and untested; numerical controls, resolution and subgrid sensitivity are pending",
                "reference": "https://arxiv.org/abs/astro-ph/0411108",
            }
    for name in ["gadget4", "class", "axiclass", "monofonic", "rockstar", "consistent-trees"]:
        p = ROOT / "software" / name
        if not p.exists():
            continue
        try:
            commit = subprocess.check_output(
                ["git", "-C", str(p), "rev-parse", "HEAD"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
            manifest["software_commits"][name] = commit
        except Exception:
            pass
    _write_json_atomic(Path(run_dir) / "manifest.json", manifest)
    return manifest


def update_manifest(run_dir, key, value):
    """Update a named run stage in the provenance manifest without altering simulation data."""
    p = Path(run_dir) / "manifest.json"
    with open(p, "r", encoding="utf-8") as f:
        m = json.load(f)
    m.setdefault("validation", {})[key] = value
    _write_json_atomic(p, m)


def gadget_config_text(cfg, ngenic_creation=False):
    """Compile-time GADGET-4 physics options for this run.

    Keep FOF/SUBFIND parameters that GADGET documents as compile-time choices
    here, not in the runtime parameter file. N-GenIC is kept in a separate
    binary: a normal start with CREATE_GRID enabled can replace a supplied IC
    with a newly generated particle load.
    """
    require_particle_cosmology(cfg)
    s = cfg["simulation"]
    physics = s["physics"]
    n = int(s["particles_per_dimension"])
    pm = int(s["pm_grid"])
    halos = cfg.get("halos", {})
    options = [
        "PERIODIC",
        "SELFGRAVITY",
        "PMGRID=" + str(pm),
        "TREEPM_NOTIMESPLIT",
        "HIERARCHICAL_GRAVITY",
        "MULTIPOLE_ORDER=" + str(int(s["multipole_order"])),
        "POSITIONS_IN_64BIT",
        "DOUBLEPRECISION=1",
        "DOUBLEPRECISION_FFTW",
        "OUTPUT_IN_DOUBLEPRECISION",
        "IDS_64BIT",
        "POWERSPEC_ON_OUTPUT",
        "PRESERVE_SHMEM_BINARY_INVARIANCE",
        "OUTPUT_EXACT_SYNCHRONIZED",
    ]
    options.append("ASMTH=" + format(float(s["asmth"]), ".12g"))
    plan = softening_plan(s)
    if plan["coarse_mass_ratios"]:
        if ngenic_creation:
            raise ValueError("Mass-scaled zoom softenings require delivered zoom ICs")
        options += ["NSOFTCLASSES=" + str(len(plan["comoving"])), "INDIVIDUAL_GRAVITY_SOFTENING=4"]
    if halos.get("fof", True):
        options += [
            "FOF",
            "FOF_PRIMARY_LINK_TYPES=2",
            "FOF_SECONDARY_LINK_TYPES=1+16+32",
            "FOF_LINKLENGTH=" + str(cfg["halos"]["fof_linking_length"]),
            "FOF_GROUP_MIN_LEN=" + str(cfg["halos"]["min_particles"]),
        ]
        if halos.get("subfind", True):
            options.append("SUBFIND")
    if type(halos.get('native_mergertree',False)) is not bool:
        raise ValueError('Native merger-tree output control must be boolean')
    if halos.get('native_mergertree',False) and not ngenic_creation:
        if 'SUBFIND' not in options:raise ValueError('Native merger trees require SUBFIND')
        options.append('MERGERTREE')
    if physics == "dm" and ngenic_creation:
        options += ["NGENIC=" + str(n), "NGENIC_2LPT", "CREATE_GRID"]
    if physics == "hydro":
        if cfg.get("black_holes", {}).get("enabled", False) or cfg.get('stellar_evolution',{}).get('enabled',False):
            options.remove("HIERARCHICAL_GRAVITY")
            options += ["FORCE_EQUAL_TIMESTEPS"]
            if cfg.get('black_holes',{}).get('enabled',False):
                options.append('BLACK_HOLES')
                diagnostic=cfg['black_holes'].get('wake_diagnostics')
                if diagnostic is not None:
                    from black_holes import wake_diagnostic_settings
                    wake_diagnostic_settings(diagnostic)
                    options.extend(['RESEARCH_BH_WAKE_DIAGNOSTIC',
                        'RESEARCH_BH_WAKE_DIAGNOSTIC_SHA256='+diagnostic['schedule']['sha256'],
                        'RESEARCH_BH_WAKE_DIAGNOSTIC_MAX_BYTES='+str(diagnostic['maximum_bytes']),
                        'RESEARCH_BH_WAKE_DIAGNOSTIC_MAX_ROWS='+str(diagnostic['maximum_rows'])])
                    if 'OUTPUT_EXACT_SYNCHRONIZED' not in options:options.append('OUTPUT_EXACT_SYNCHRONIZED')
            if cfg.get('stellar_evolution',{}).get('enabled',False):options.append('STELLAR_EVOLUTION')
            if cfg.get('chemistry',{}).get('enabled',False):options.append('RESEARCH_GRACKLE')
        else:
            options.append("PRESSURE_ENTROPY_SPH")
        options += ["OUTPUT_DIVVEL", "OUTPUT_CURLVEL"]
        hydro = cfg.get("hydro", {})
        if bool(hydro.get("cooling", True)):
            options += ["OUTPUT_COOLHEAT", "COOLING"]
        if bool(hydro.get("star_formation", True)):
            options.append("STARFORMATION")
    equal_steps=s.get('force_equal_timesteps',False)
    if type(equal_steps) is not bool:raise ValueError('Force-equal timestep control must be an explicit boolean')
    if equal_steps and not ngenic_creation:
        if 'HIERARCHICAL_GRAVITY' in options:options.remove('HIERARCHICAL_GRAVITY')
        if 'FORCE_EQUAL_TIMESTEPS' not in options:options.append('FORCE_EQUAL_TIMESTEPS')
    if cfg.get('sidm',{}).get('enabled',False) and not ngenic_creation:
        from core.sidm_physics import sidm_native_parameters
        sidm_native_parameters(cfg['sidm'],physics=physics,mpi_ranks=mpi_ranks(cfg))
        if 'HIERARCHICAL_GRAVITY' in options:options.remove('HIERARCHICAL_GRAVITY')
        if 'FORCE_EQUAL_TIMESTEPS' not in options:options.append('FORCE_EQUAL_TIMESTEPS')
        options += ['RESEARCH_SIDM_REFERENCE','EVALPOTENTIAL','OUTPUT_POTENTIAL']
    capsule=cfg['simulation'].get('restart_source_capsule',False)
    if type(capsule) is not bool:raise ValueError('Restart source capsule output setting must be an explicit boolean')
    if capsule and not ngenic_creation:options.append('RESEARCH_RESTART_STATE_CAPSULE')
    if 'restart_exercise_pause_step' in s:
        step=s['restart_exercise_pause_step']
        if type(step) is not int or not 0<=step<=1000000 or not capsule:
            raise ValueError('Bounded restart exercise step requires complete source-state diagnostics')
        options.append('RESEARCH_CHECKPOINT_EXERCISE')
    diagnostics=s.get('output_acceleration_diagnostics',False)
    if type(diagnostics)is not bool:raise ValueError('Native acceleration diagnostic output requires an explicit boolean')
    if diagnostics and not ngenic_creation:options.append('OUTPUT_ACCELERATION')
    return "\n".join(dict.fromkeys(options)) + "\n"


def _treecool_file(cfg=None):
    """Resolve an explicitly selected TREECOOL input or the pinned project default."""
    configured = (cfg or {}).get("hydro", {}).get("treecool_file")
    if configured is not None:
        if not isinstance(configured, str) or not configured.strip():
            raise ValueError("hydro.treecool_file must be null or a non-empty path")
        path = Path(configured).expanduser()
        if not path.is_absolute():
            path = ROOT / path
        if not path.is_file():
            raise FileNotFoundError("hydro.treecool_file does not exist: " + str(path))
        return path.resolve()

    source = ROOT / "software/gadget4"
    if not source.exists():
        return None
    canonical = source / "data/TREECOOL"
    if canonical.is_file():
        return canonical
    candidates = []
    for pattern in ["**/TREECOOL", "**/TREECOOL*", "**/treecool*"]:
        candidates.extend(source.glob(pattern))
    files = sorted({p.resolve() for p in candidates if p.is_file()}, key=lambda p: str(p).lower())
    if not files:
        return None
    hashes = {hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
    if len(hashes) > 1:
        raise ValueError(
            "Multiple different GADGET TREECOOL tables were found; set hydro.treecool_file explicitly"
        )
    return files[0]


def validate_treecool_table(path):
    """Validate the numeric TREECOOL table exactly as consumed by GADGET-4."""
    table_path = Path(path).expanduser().resolve()
    if not table_path.is_file():
        raise FileNotFoundError("TREECOOL table does not exist: " + str(table_path))
    rows = []
    for line_number, raw_line in enumerate(table_path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue
        fields = line.split()
        if len(fields) != 7:
            raise ValueError(
                "TREECOOL line %d must contain exactly 7 numeric columns; GADGET's reader does not accept headers/comments"
                % line_number
            )
        try:
            row = [float(value) for value in fields]
        except ValueError as exc:
            raise ValueError("TREECOOL line %d contains a non-numeric value" % line_number) from exc
        if not all(math.isfinite(value) for value in row):
            raise ValueError("TREECOOL line %d contains a non-finite value" % line_number)
        rows.append(row)
    if not rows:
        raise ValueError("TREECOOL table contains no numeric rows")
    values = np.asarray(rows, dtype=float)
    axis = values[:, 0]
    if np.any(np.diff(axis) <= 0):
        raise ValueError("TREECOOL log10(1+z) column must be strictly increasing")
    if not math.isclose(float(axis[0]), 0.0, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError("TREECOOL must include its z=0 row so GADGET does not extrapolate below the table")
    if np.any(values[:, 1:] < 0):
        raise ValueError("TREECOOL ionization and heating rates must be non-negative")

    zero_rows = np.flatnonzero(values[:, 1] == 0.0)
    active_count = int(zero_rows[0]) if zero_rows.size else len(values)
    if active_count < 1:
        raise ValueError("TREECOOL has no active hydrogen-ionization rows")
    active = values[:active_count, 1:]
    # GADGET interpolates all six rates in log space and terminates its active
    # table at the first zero HI rate. A non-zero tail would be silently ignored.
    if np.any(active <= 0):
        raise ValueError("TREECOOL active ionization/heating rates must all be positive for GADGET log interpolation")
    if zero_rows.size and np.any(values[active_count:, 1:] != 0.0):
        raise ValueError("TREECOOL rows after the first zero HI rate must also contain six zeros")

    active_axis = axis[:active_count]
    redshifts = np.power(10.0, active_axis) - 1.0
    if not np.all(np.isfinite(redshifts)):
        raise ValueError("TREECOOL redshift axis is outside the representable range")
    return {
        "path": str(table_path),
        "sha256": hashlib.sha256(table_path.read_bytes()).hexdigest(),
        "row_count": int(len(values)),
        "active_row_count": active_count,
        "coordinate": "log10(1+z)",
        "rate_columns": ["Gamma_HI_s^-1", "Gamma_HeI_s^-1", "Gamma_HeII_s^-1",
                         "Heating_HI_erg_s^-1", "Heating_HeI_erg_s^-1", "Heating_HeII_erg_s^-1"],
        "active_redshift_min": float(redshifts[0]),
        "active_redshift_max": float(redshifts[-1]),
        "outside_active_table_behavior": "GADGET-4 sets UV ionization and heating rates to zero",
    }


def validate_native_timestep_span(values):
    """Reject steps outside GADGET's finite integer timeline before launch.

    This is a representability check, not an integration-accuracy certificate.
    Configuration is never silently shortened to make a control execute.
    """
    begin, end, maximum = (float(values[key]) for key in ('TimeBegin','TimeMax','MaxSizeTimestep'))
    if not all(math.isfinite(v) for v in (begin,end,maximum)) or end<=begin or maximum<=0:
        raise ValueError('Finite increasing native times and a positive maximum timestep required')
    if values['ComovingIntegrationOn']:
        if begin<=0:
            raise ValueError('Positive cosmological starting scale factor required')
        span = math.log(end/begin)
    else:
        span = end-begin
    if maximum >= span:
        raise ValueError(f'MaxSizeTimestep ({maximum:g}) must be smaller than the complete native timeline span ({span:g}); change the configuration explicitly')
    return span


def gadget_param_text(
    cfg, run_dir, ic_path, ngenic_pk=None, dry_run=False, ngenic_creation=False,
    snapshot_base=None, output_dir=None, num_files_per_snapshot=None,
    ngenic_velocity_transfer=None,
):
    """Translate supported physics/units into GADGET parameters."""
    require_particle_cosmology(cfg, require_ede_background_table=True)
    c = cfg["cosmology"]
    s = cfg["simulation"]
    if not ngenic_creation:
        validate_native_timestep_span({
            'TimeBegin':1.0/(1.0+choose_start_redshift(cfg)),
            'TimeMax':1.0/(1.0+float(s.get('end_redshift',0.0))),
            'MaxSizeTimestep':s['max_size_timestep_ln_a'],'ComovingIntegrationOn':1})
    physics = str(s.get("physics", "dm")).lower()
    h = float(c["H0"]) / 100.0
    omega_b = float(c["omega_b"]) / h**2
    omega_dm = float(c["omega_cdm"]) / h**2
    omega_m = omega_b + omega_dm
    omega_radiation = c.get("omega_radiation")
    omega_radiation = float(omega_radiation_from_config(cfg) if omega_radiation is None else omega_radiation)
    omega_lambda = 1.0 - omega_m - omega_radiation
    out = Path(output_dir) if output_dir is not None else Path(run_dir) / "data"
    output_list = Path(run_dir) / "input/output_times.txt"
    ranks = mpi_ranks(cfg)
    total_mem_mb = int(memory_limit_gb(cfg) * 1024)
    # GADGET's MaxMemSize is per MPI rank. Reserve margin for OS and MPI.
    max_mem_rank_mb = int(0.85 * total_mem_mb / ranks)
    if max_mem_rank_mb < 512:
        raise ValueError("Memory budget provides less than 512 MiB per MPI process after reserve; reduce MPI ranks or increase hardware.memory_limit_gb")

    values = {
        # GADGET's HDF5 reader appends `.hdf5` or `.0.hdf5` to this prefix.
        "InitCondFile": gadget_ic_prefix(ic_path),
        "OutputDir": str(out) + "/",
        # Restart flag 6 appends `_ics` to this base and writes snapshot 000.
        # Give it a dedicated deterministic base for generated initial data.
        "SnapshotFileBase": "ngenic" if ngenic_creation else (snapshot_base or "snapshot"),
        "OutputListFilename": output_list,
        "OutputListOn": 1,
        "ICFormat": 3,
        "SnapFormat": 3,
        "NumFilesPerSnapshot": int(
            cfg["simulation"]["num_files_per_snapshot"]
            if num_files_per_snapshot is None else num_files_per_snapshot
        ),
        "MaxFilesWithConcurrentIO": min(8, ranks),
        # GADGET treats zero as a request to exit after the first force pass.
        # A large positive wall limit disables that special exit path while
        # CpuTimeBetRestartFile continues to control restart checkpoints.
        "TimeLimitCPU": float(s["time_limit_cpu_seconds"]),
        "CpuTimeBetRestartFile": int(s["restart_every_seconds"]),
        "MaxMemSize": max_mem_rank_mb,
        "TimeBegin": 1.0 / (1.0 + choose_start_redshift(cfg)),
        "TimeMax": 1.0 / (1.0 + float(s.get("end_redshift", 0.0))),
        "TimeOfFirstSnapshot": 1.0 / (1.0 + float(snapshot_redshifts(cfg)[0])),
        "TimeBetSnapshot": 1.01,
        "TimeBetStatistics": 0.01,
        "ComovingIntegrationOn": 1,
        "BoxSize": float(s["box_mpc_h"]) * 1000.0,
        "Omega0": omega_m,
        "OmegaLambda": omega_lambda,
        "OmegaRadiation": omega_radiation,
        "CosmologyTableFile": "none",
        "CosmologyTableValidationTolerance": cfg.get("validation", {}).get("background", {}).get(
            "max_relative_drift_kick_error", 1.0e-4
        ),
        "OmegaBaryon": omega_b,
        "HubbleParam": h,
        "Hubble": 0.1,
        "UnitLength_in_cm": KPC_CM,
        "UnitMass_in_g": 1.0e10 * MSUN_G,
        "UnitVelocity_in_cm_per_s": 1.0e5,
        "GravityConstantInternal": 0,
        "TypeOfOpeningCriterion": 1,
        "ErrTolTheta": float(s["err_tol_theta"]),
        "ErrTolThetaMax": 1.0,
        "ErrTolForceAcc": float(s["err_tol_force_acc"]),
        "ActivePartFracForPMinsteadOfEwald": 0.1,
        "MaxSizeTimestep": float(s["max_size_timestep_ln_a"]),
        "MinSizeTimestep": float(s.get("min_size_timestep_ln_a", 0.0)),
        "ErrTolIntAccuracy": float(s["err_tol_int_accuracy"]),
        "CourantFac": float(
            cfg.get("hydro", {}).get("courant_factor")
            if cfg.get("hydro", {}).get("courant_factor") is not None else 0.1
        ),
        # GADGET registers this parameter for every build, including DM-only.
        "MinEgySpec": 0.0,
        "ActivePartFracForNewDomainDecomp": 0.01,
        "TopNodeFactor": 2.5,
    }
    values.update(hydro_sph_parameters(cfg.get("hydro", {}), start_scale_factor=values["TimeBegin"]))
    from black_holes import native_parameters
    values.update(native_parameters(cfg.get("black_holes", {"enabled": False})))
    from core.sidm_physics import sidm_native_parameters
    values.update(sidm_native_parameters(cfg.get('sidm',{'enabled':False}),physics=physics,
        mpi_ranks=ranks,max_mem_rank_mib=max_mem_rank_mb,ngenic_creation=ngenic_creation))
    # GADGET registers DesLinkNgb only in SUBFIND builds. An N-GenIC or
    # evolution binary without SUBFIND rejects the otherwise valid input.
    if cfg.get("halos", {}).get("subfind", False):
        values["DesLinkNgb"] = 20
    cosmology_table = c.get("gadget_background_table")
    if cosmology_table:
        values["CosmologyTableFile"] = str(Path(cosmology_table).expanduser().resolve())

    # N-GenIC parameters are harmless during the evolution run but required for
    # restart flag 6.  The spectrum is already evaluated at z_start, so do not
    # ask GADGET to normalize it backwards from z=0.
    if ngenic_pk is not None:
        values.update({
            "NSample": int(s["particles_per_dimension"]),
            "GridSize": int(s["particles_per_dimension"]),
            "PowerSpectrumType": 2,
            "PowerSpectrumFile": ngenic_pk,
            "InputSpectrum_UnitLength_in_cm": MPC_CM,
            "ShapeGamma": 0.21,
            "PrimordialIndex": 1.0,
            "Sigma8": 0.8,
            "ReNormalizeInputSpectrum": 0,
            "SphereMode": 0,
            "Seed": int(s["random_seed"]),
            "NgenicVelocityTransferFile": (
                str(Path(ngenic_velocity_transfer).expanduser().resolve())
                if ngenic_velocity_transfer is not None else "none"
            ),
        })

    hydro = cfg.get("hydro", {})
    if physics == "hydro" and bool(hydro.get("cooling", True)):
        treecool = _treecool_file(cfg)
        if treecool is None and not dry_run and not bool(hydro.get("allow_missing_treecool", False)):
            raise RuntimeError("Hydro+cooling requested but no TREECOOL table was found in the GADGET source/examples.")
        values["TreecoolFile"] = treecool if treecool else "TREECOOL"
    if physics == "hydro" and bool(hydro.get("star_formation", True)):
        values.update(hydro_star_formation_parameters(hydro))
    from stars import stellar_evolution_native_parameters
    values.update(stellar_evolution_native_parameters(cfg.get('stellar_evolution',{'enabled':False}),
        factor_sn=hydro_star_formation_parameters(hydro)['FactorSN']))
    from stars import grackle_native_parameters
    values.update(grackle_native_parameters(cfg.get('chemistry',{'enabled':False})))

    if 'restart_exercise_pause_step' in s and not ngenic_creation:
        step=s['restart_exercise_pause_step']
        if type(step) is not int or not 0<=step<=1000000 or not s.get('restart_source_capsule',False):
            raise ValueError('Restart exercise pause requires complete native state diagnostics and a bounded step')
        values['NativeControlPauseStep']=step
    lines = [str(k) + " " + str(v) for k, v in values.items()]
    plan = softening_plan(s)
    for t in range(6):
        lines.append("SofteningClassOfPartType%d %d" % (t, plan["type_classes"][t]))
    for k, value in enumerate(plan["comoving"]):
        lines.append("SofteningComovingClass%d %.17g" % (k, value))
        lines.append("SofteningMaxPhysClass%d %.17g" % (k, plan["maximum_physical"][k]))
    return "\n".join(lines) + "\n"







def validate_monofonic_growth_background_table(table_path, source_background_path, expected_model, cfg):
    """Verify the exact E(a), D(a), and f(a) table supplied to monofonIC."""
    table_path = Path(table_path).expanduser().resolve()
    metadata_path = Path(str(table_path) + ".json")
    if not table_path.is_file() or not metadata_path.is_file():
        raise RuntimeError("The model-matched monofonIC growth-background table or its provenance is missing")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if (metadata.get("status") != "PASS" or metadata.get("model") != expected_model
            or metadata.get("schema") != "darkmattersim-monofonic-background-v1"
            or Path(metadata.get("table", "")).resolve() != table_path
            or metadata.get("table_sha256") != file_sha256(table_path)):
        raise RuntimeError("monofonIC background-table provenance does not match the file/model used")
    source_path = Path(metadata.get("source_background", "")).expanduser().resolve()
    if (not source_path.is_file()
            or metadata.get("source_background_sha256") != file_sha256(source_path)):
        raise RuntimeError("monofonIC background table no longer matches its recorded CLASS source")

    expected_source_path = Path(source_background_path).expanduser().resolve()
    if not expected_source_path.is_file():
        raise FileNotFoundError("The current run's CLASS background is missing")
    columns, data = class_background_table(source_path)
    current_columns, current_data = class_background_table(expected_source_path)
    for name in ("z", "H", "growth_D", "growth_f"):
        if name not in columns or name not in current_columns:
            raise ValueError("CLASS background is missing the %s growth-validation column" % name)

    verify_numeric_table(table_path)
    table = np.loadtxt(table_path, comments="#", ndmin=2)
    if (table.shape[1] != 4 or len(table) < 6 or np.any(~np.isfinite(table))
            or np.any(table[:, 0] <= 0) or np.any(np.diff(table[:, 0]) <= 0)
            or np.any(table[:, 1:] <= 0) or not math.isclose(float(table[-1, 0]), 1.0, rel_tol=0.0, abs_tol=1e-14)):
        raise ValueError("The monofonIC growth-background table has invalid columns or scale-factor coverage")

    def ordered_background(source, source_columns):
        source_z = np.asarray(source[:, source_columns["z"]], dtype=np.float64)
        source_a = 1.0 / (1.0 + source_z)
        order = np.argsort(source_a)
        source_a = source_a[order]
        h = np.asarray(source[order, source_columns["H"]], dtype=np.float64)
        d = np.asarray(source[order, source_columns["growth_D"]], dtype=np.float64)
        f = np.asarray(source[order, source_columns["growth_f"]], dtype=np.float64)
        if (len(source_a) < 2 or np.any(~np.isfinite(source_a)) or np.any(np.diff(source_a) <= 0)
                or np.any(~np.isfinite(h)) or np.any(h <= 0)
                or np.any(~np.isfinite(d)) or np.any(d <= 0)
                or np.any(~np.isfinite(f)) or np.any(f <= 0)):
            raise ValueError("AxiCLASS background has invalid E(a), D(a), or f(a) data")
        today = int(np.argmin(np.abs(source_a - 1.0)))
        if abs(float(source_a[today]) - 1.0) > 1.0e-8:
            raise ValueError("AxiCLASS background does not include the present epoch")
        return source_a, h / h[today], d / d[today], f

    source_a, source_e, source_d, source_f = ordered_background(data, columns)
    current_a, current_e, current_d, current_f = ordered_background(current_data, current_columns)
    start_a = 1.0 / (1.0 + float(cfg["simulation"]["start_redshift"]))
    common_min = max(float(table[0, 0]), float(source_a[0]), float(current_a[0]))
    common_max = min(1.0, float(source_a[-1]), float(current_a[-1]))
    sample_a = source_a[(source_a >= common_min) & (source_a <= common_max)]
    if len(sample_a) < 2 or start_a < table[0, 0] or start_a > table[-1, 0]:
        raise ValueError("The monofonIC background table does not cover the validated run interval")
    log_table_a = np.log(table[:, 0])
    log_sample_a = np.log(sample_a)
    table_e = np.exp(np.interp(log_sample_a, log_table_a, np.log(table[:, 1])))
    table_d = np.exp(np.interp(log_sample_a, log_table_a, np.log(table[:, 2])))
    table_f = np.interp(log_sample_a, log_table_a, table[:, 3])
    reference_e = np.exp(np.interp(log_sample_a, np.log(source_a), np.log(source_e)))
    reference_d = np.exp(np.interp(log_sample_a, np.log(source_a), np.log(source_d)))
    reference_f = np.interp(log_sample_a, np.log(source_a), source_f)
    current_e = np.exp(np.interp(log_sample_a, np.log(current_a), np.log(current_e)))
    current_d = np.exp(np.interp(log_sample_a, np.log(current_a), np.log(current_d)))
    current_f = np.interp(log_sample_a, np.log(current_a), current_f)
    errors = {
        "max_relative_E_error_vs_source": float(np.max(np.abs(table_e / reference_e - 1.0))),
        "max_relative_D_error_vs_source": float(np.max(np.abs(table_d / reference_d - 1.0))),
        "max_absolute_f_error_vs_source": float(np.max(np.abs(table_f - reference_f))),
        "max_relative_E_error_vs_current_run": float(np.max(np.abs(reference_e / current_e - 1.0))),
        "max_relative_D_error_vs_current_run": float(np.max(np.abs(reference_d / current_d - 1.0))),
        "max_absolute_f_error_vs_current_run": float(np.max(np.abs(reference_f - current_f))),
    }
    tolerance = float(cfg.get("validation", {}).get("background", {}).get("max_monofonic_interpolation_error", 1.0e-4))
    max_error = max(errors.values())
    if not math.isfinite(tolerance) or tolerance <= 0 or max_error > tolerance:
        raise RuntimeError(
            "monofonIC's E(a), D(a), or f(a) differs from the matched CLASS history: "
            + json.dumps(errors, sort_keys=True)
        )
    return {
        "status": "PASS",
        "generator": "monofonIC",
        "table_file": str(table_path),
        "table_sha256": file_sha256(table_path),
        "metadata_file": str(metadata_path),
        "source_background": str(source_path),
        "source_background_sha256": file_sha256(source_path),
        "current_run_background": str(expected_source_path),
        "scale_factor_range": [float(table[0, 0]), float(table[-1, 0])],
        "start_scale_factor": start_a,
        "source_sample_count": int(len(sample_a)),
        "max_relative_E_error": errors["max_relative_E_error_vs_source"],
        "max_relative_D_error": errors["max_relative_D_error_vs_source"],
        "max_absolute_f_error": errors["max_absolute_f_error_vs_source"],
        "max_error_against_current_run": max(
            errors["max_relative_E_error_vs_current_run"],
            errors["max_relative_D_error_vs_current_run"],
            errors["max_absolute_f_error_vs_current_run"],
        ),
        "tolerance": tolerance,
        "interpolation_error_estimate": metadata.get("interpolation_error_estimate"),
        "limitation": metadata.get("limitations"),
    }





def gadget_binary(run_dir, ngenic_creation=False):
    """Locate the run-specific IC-generation or particle-evolution executable."""
    if ngenic_creation:
        return Path(run_dir) / "input/icsbuild/Gadget4"
    return Path(run_dir) / "input/Gadget4"


def verified_build_cache(binary, signature_path, receipt_path, signature):
    """A signature without a completed binary checksum receipt is not a cache hit."""
    from core.foundation import canonical_digest
    binary,signature_path,receipt_path=map(Path,(binary,signature_path,receipt_path))
    if not all(p.is_file() for p in (binary,signature_path,receipt_path)):
        return False
    try:
        receipt=json.loads(receipt_path.read_text())
        return (json.loads(signature_path.read_text())==signature
                and receipt.get('schema')=='native-build-cache-v1'
                and receipt.get('status')=='COMPLETE'
                and receipt.get('signature_sha256')==canonical_digest(signature)
                and receipt.get('binary_sha256')==file_sha256(binary))
    except (OSError,ValueError):
        return False


def native_physics_asset_bindings(cfg):
    """Record every file used by a compiled stellar/chemistry source operator."""
    from stars import read_stellar_return_model,grackle_native_parameters
    assets=[];stellar=cfg.get('stellar_evolution',{'enabled':False});chemistry=cfg.get('chemistry',{'enabled':False})
    diagnostic=cfg.get('black_holes',{}).get('wake_diagnostics')
    if diagnostic is not None:
        from black_holes import wake_diagnostic_settings
        path=wake_diagnostic_settings(diagnostic)
        assets.append({'role':'bh_wake_diagnostic_schedule','path':str(path),'sha256':file_sha256(path),'size_bytes':path.stat().st_size})
    combination = cfg.get('combination_specification')
    if combination is not None:
        receipt = Path(combination['receipt']).expanduser().resolve()
        if file_sha256(receipt) != combination['sha256']:
            raise ValueError('Combined cosmology/interaction campaign receipt changed before native launch')
        assets.append({'role': 'combined_physics_campaign', 'path': str(receipt),
            'sha256': combination['sha256'], 'size_bytes': receipt.stat().st_size})
    if cfg.get('sidm',{}).get('enabled',False):
        from core.sidm_physics import sidm_manifest_record
        assets.extend(sidm_manifest_record(cfg['sidm'])['assets'])
    if stellar.get('enabled',False):
        manifest=Path(stellar['model_manifest']).expanduser().resolve()
        if file_sha256(manifest)!=stellar['model_sha256']:raise ValueError('Stellar model changed before native launch')
        model=read_stellar_return_model(manifest)[0]
        for role,path in [('stellar_model',manifest),('stellar_table',Path(model['table']['path'])),('stellar_model_seal',manifest.with_name(manifest.name+'.sha256'))]:
            assets.append({'role':role,'path':str(path),'sha256':file_sha256(path),'size_bytes':path.stat().st_size})
        for role,asset in model['assets'].items():
            path=Path(asset['path'])
            if file_sha256(path)!=asset['sha256']:raise ValueError('Pinned stellar yield source changed before launch')
            assets.append({'role':role,**asset,'size_bytes':path.stat().st_size})
    if chemistry.get('enabled',False):
        grackle_native_parameters(chemistry)
        for role in ('data','library'):
            path=Path(chemistry[role+'_file']).expanduser().resolve()
            assets.append({'role':'grackle_'+role,'path':str(path),'sha256':chemistry[role+'_sha256'],'size_bytes':path.stat().st_size})
    return assets


def native_physics_environment(cfg,environment=None):
    """Pin chemistry loader search paths for every native launch and restart."""
    env=dict(os.environ if environment is None else environment)
    native_physics_asset_bindings(cfg)
    diagnostic=cfg.get('black_holes',{}).get('wake_diagnostics')
    if diagnostic is not None:
        from black_holes import wake_diagnostic_settings
        env['RESEARCH_BH_WAKE_DIAGNOSTIC_SCHEDULE']=str(wake_diagnostic_settings(diagnostic))
    else:env.pop('RESEARCH_BH_WAKE_DIAGNOSTIC_SCHEDULE',None)
    chemistry=cfg.get('chemistry',{'enabled':False})
    if chemistry.get('enabled',False):
        directory=str(Path(chemistry['library_file']).expanduser().resolve().parent)
        for key in ('LD_LIBRARY_PATH','DYLD_LIBRARY_PATH'):
            prior=env.get(key,'');env[key]=directory+(os.pathsep+prior if prior else '')
        # A preload can replace a pinned dependency even when rpath is fixed.
        if env.get('LD_PRELOAD') or env.get('DYLD_INSERT_LIBRARIES'):raise ValueError('Pinned native chemistry rejects injected loader libraries')
    return env


def build_gadget(cfg, run_dir, dry_run=False, ngenic_creation=False, *, control_profile=None):
    """Build/cache an executable keyed by actual source, toolchain and physics flags; record its run provenance."""
    source = ROOT / "software/gadget4"
    installation=source/"research-source-overlay.installation.json"
    if installation.exists() and json.loads(installation.read_text()).get("status")!="COMPLETE":
        raise ValueError("Native build refuses an interrupted source overlay installation")
    run_dir = Path(run_dir).expanduser().resolve()
    build_dir = run_dir / "input/icsbuild" if ngenic_creation else run_dir / "input"
    build_dir.mkdir(parents=True, exist_ok=True)
    conf = build_dir / "Config.sh"
    if control_profile is None:
        config_text = gadget_config_text(cfg, ngenic_creation=ngenic_creation)
    else:
        if ngenic_creation:
            raise ValueError("Numerical controls do not replace IC generators")
        from core.research.controls import control_config_text
        config_text = control_config_text(control_profile)
    if not conf.exists() or conf.read_text(encoding="utf-8") != config_text:
        atomic_text(conf, config_text)
    if dry_run:
        return gadget_binary(run_dir, ngenic_creation=ngenic_creation)
    if not source.exists():
        raise RuntimeError("GADGET-4 source not found. Restore the bundled solver sources.")
    env = native_physics_environment(cfg)
    env["SYSTYPE"] = "Generic-gcc"
    try:
        hdf5_cflags = subprocess.check_output(["pkg-config", "--cflags", "hdf5"], text=True).strip()
        hdf5_libs = subprocess.check_output(["pkg-config", "--libs", "hdf5"], text=True).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        try:
            hdf5_cflags = subprocess.check_output(["pkg-config", "--cflags", "hdf5-openmpi"], text=True).strip()
            hdf5_libs = subprocess.check_output(["pkg-config", "--libs", "hdf5-openmpi"], text=True).strip()
        except (FileNotFoundError, subprocess.CalledProcessError) as exc:
            raise RuntimeError("GADGET needs HDF5 development files and pkg-config metadata (hdf5 or hdf5-openmpi).") from exc
    # The upstream Makefile defaults to libraries below software/gadget4/lib.
    # The Windows/WSL installer instead installs system development packages;
    # pass their actual prefixes so a clean machine does not need vendored builds.
    library_flags = {}
    for package, include_key, library_key in [
        ("gsl", "GSL_INCL", "GSL_LIBS"),
        ("fftw3", "FFTW_INCL", "FFTW_LIBS"),
    ]:
        try:
            library_flags[include_key] = subprocess.check_output(
                ["pkg-config", "--cflags-only-I", package], text=True
            ).strip()
            # These are command-line Make variables. GNU make does not append
            # the upstream `+= -lgsl ...` defaults to a command-line override,
            # so include both search paths and library names explicitly.
            library_flags[library_key] = subprocess.check_output(
                ["pkg-config", "--libs", package], text=True
            ).strip()
        except (FileNotFoundError, subprocess.CalledProcessError) as exc:
            raise RuntimeError(
                "GADGET needs %s development files and pkg-config metadata. "
                "Install the %s development package before building." % (package, package)
            ) from exc
    binary = gadget_binary(run_dir, ngenic_creation=ngenic_creation)
    source_digest = hashlib.sha256()
    from core.native import native_build_input_paths
    source_files = native_build_input_paths(source)
    for path in sorted(source_files):
        source_digest.update(str(path.relative_to(source)).encode("utf-8"))
        source_digest.update(path.read_bytes())
    try:
        compiler_version = subprocess.check_output(["mpicxx", "--version"], text=True, stderr=subprocess.STDOUT).strip()
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("GADGET build requires a working MPI C++ compiler named mpicxx") from exc
    try:
        mpi_wrapper = subprocess.check_output(["mpicxx", "-show"], text=True, stderr=subprocess.STDOUT).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        mpi_wrapper = "unavailable"
    signature = {
        "config": config_text,
        "gadget_revision": subprocess.run(
            ["git", "-C", str(source), "rev-parse", "HEAD"],
            text=True, capture_output=True, check=False,
        ).stdout.strip() or "unknown",
        "darkmattersim_physics_patch": "lcdm-wdm-class-background-units-v7",
        "gadget_source_sha256": source_digest.hexdigest(),
        "macro_allowlist_sha256": file_sha256(source / 'defines_extra'),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "compiler_version": compiler_version,
        "mpi_wrapper_command": mpi_wrapper,
        "hdf5_cflags": hdf5_cflags,
        "hdf5_libs": hdf5_libs,
        "libraries": library_flags,
        "physical_source_assets":native_physics_asset_bindings(cfg),
        "stellar_model":cfg.get('stellar_evolution',{'enabled':False}),
        "native_chemistry":cfg.get('chemistry',{'enabled':False}),
    }
    if cfg.get('chemistry',{}).get('enabled',False):
        settings=cfg['chemistry']
        from stars import grackle_native_parameters
        grackle_native_parameters(settings)
        try:
            grackle_version=subprocess.check_output(['pkg-config','--modversion','grackle'],text=True).strip()
            grackle_include=subprocess.check_output(['pkg-config','--cflags','grackle'],text=True).strip()
            grackle_library=subprocess.check_output(['pkg-config','--variable=libdir','grackle'],text=True).strip()
        except (FileNotFoundError,subprocess.CalledProcessError) as exc:
            raise RuntimeError('Install the pinned Grackle source archive with GRACKLE_USE_DOUBLE=ON before native chemistry builds') from exc
        if grackle_version!='3.4.1' or Path(settings['library_file']).resolve().parent!=Path(grackle_library).resolve():
            raise ValueError('Grackle pkg-config installation differs from the configured pinned library')
        library_flags['GRACKLE_INCL']=grackle_include
        library_flags['GRACKLE_LIBS']=str(Path(settings['library_file']).resolve())+' -Wl,-rpath,'+str(Path(settings['library_file']).resolve().parent)
        signature['grackle_version']=grackle_version
        signature['grackle_data_sha256']=file_sha256(settings['data_file'])
        signature['grackle_library_sha256']=file_sha256(settings['library_file'])
    signature_path = build_dir / "build_signature.json"
    local_receipt = build_dir/'build_receipt.json'
    if verified_build_cache(binary,signature_path,local_receipt,signature):
        _record_gadget_build(run_dir, "initial_conditions" if ngenic_creation else "evolution", signature)
        return binary

    cache_key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode("utf-8")).hexdigest()
    cache_dir = runtime_paths().build_root / "gadget4" / cache_key
    cache_binary = cache_dir / "Gadget4"
    cache_signature = cache_dir / "build_signature.json"
    with native_run_lock(cache_dir, operation='shared GADGET build cache'):
        cache_receipt=cache_dir/'build_receipt.json'
        cached=verified_build_cache(cache_binary,cache_signature,cache_receipt,signature)

        if not cached:
            cache_dir.mkdir(parents=True, exist_ok=True)
            if cache_binary.exists():
                # Keep unsealed/corrupt prior build bytes for inspection, then
                # force a new link instead of trusting Make's old target time.
                preserved=cache_dir/('unverified-Gadget4-'+file_sha256(cache_binary))
                if not preserved.exists():atomic_copy(cache_binary,preserved)
                cache_binary.unlink()
            cache_conf = cache_dir / "Config.sh"
            if not cache_conf.exists() or cache_conf.read_text(encoding="utf-8") != config_text:
                atomic_text(cache_conf, config_text)
            make_args = ["make", "-j", str(build_jobs(cfg)), "DIR=" + str(cache_dir)]
            make_args.extend(["HDF5_INCL=" + hdf5_cflags, "HDF5_LIBS=" + hdf5_libs + " -lz"])
            make_args.extend(key + "=" + value for key, value in library_flags.items())
            subprocess.run(make_args, cwd=source, env=env, check=True, pass_fds=native_run_lock_fds(run_dir)+native_run_lock_fds(cache_dir))
        if not cache_binary.is_file():
            raise RuntimeError("GADGET build did not create " + str(binary))
        if not cached:
            after=hashlib.sha256()
            current_files = native_build_input_paths(source)
            for path in sorted(current_files):
                after.update(str(path.relative_to(source)).encode('utf-8'));after.update(path.read_bytes())
            if after.hexdigest()!=signature['gadget_source_sha256'] or file_sha256(source/'defines_extra')!=signature['macro_allowlist_sha256']:
                raise ValueError('GADGET source changed during build; refusing cache publication')
            _write_json_atomic(cache_signature, signature)
            from core.foundation import canonical_digest
            atomic_json(cache_receipt,{'schema':'native-build-cache-v1','status':'COMPLETE',
                        'signature_sha256':canonical_digest(signature),'binary_sha256':file_sha256(cache_binary)})
        binary.parent.mkdir(parents=True, exist_ok=True)
        atomic_copy(cache_binary, binary)
        _write_json_atomic(signature_path, signature)
        atomic_copy(cache_receipt,local_receipt)
    _record_gadget_build(run_dir, "initial_conditions" if ngenic_creation else "evolution", signature)
    return binary


def generate_dm_ic_with_gadget(cfg, run_dir, model, cosmo_meta, dry_run=False, velocity_transfer_path=None):
    """Generate lattice ICs with the run-matched P(k) and velocity transfer, preserving the declared h-scaled units."""
    run_dir = Path(run_dir)
    binary = build_gadget(cfg, run_dir, dry_run, ngenic_creation=True)
    ngenic_pk = run_dir / "input" / (model + "_ngenic_pk.txt")
    if not dry_run:
        class_pk = Path(cosmo_meta["pk_file"])
        convert_pk_to_ngenic(class_pk, ngenic_pk, float(cfg["cosmology"]["H0"]) / 100.0)

    # GADGET restart flag 6 creates cosmological ICs. The exact N-GenIC parameters
    # are read from the same parameter file; if upstream changes parameter names,
    # the bootstrap smoke test catches this before a production run.
    param = run_dir / "input/ngenic_param.txt"
    atomic_text(param,
        gadget_param_text(
            cfg, run_dir, run_dir / "data/ics", ngenic_pk=ngenic_pk, dry_run=dry_run,
            ngenic_creation=True, ngenic_velocity_transfer=velocity_transfer_path,
        ),
    )
    if dry_run:
        return run_dir / "data/ics"
    ranks = mpi_ranks(cfg)
    log = run_dir / "input/ngenic.log"
    with open(log, "w", encoding="utf-8") as out:
        subprocess.run(["mpirun", "-np", str(ranks), str(binary), str(param), "6"], cwd=run_dir / "input", stdout=out, stderr=subprocess.STDOUT, check=True, pass_fds=native_run_lock_fds(run_dir))
    # Multi-file outputs live below snapdir_NNN, just like evolution
    # snapshots. Select shard zero and let initial_condition_files validate
    # the complete set before the handoff.
    candidates = sorted((run_dir / "data").rglob("ngenic_ics_*.hdf5"))
    if not candidates:
        raise RuntimeError("GADGET N-GenIC did not produce ICs. See " + str(log))
    return candidates[0]


def add_wdm_thermal_velocities(paths, cfg):
    from core.data import transactional_ic_transform
    return transactional_ic_transform(paths, lambda staged: _add_wdm_thermal_velocities(staged, cfg),
            operation='wdm-thermal-velocity-v1', configuration=cfg)


def _add_wdm_thermal_velocities(paths, cfg):
    """Add a CLASS-normalized Fermi-Dirac relic velocity field to PartType1.

    The counter-based random stream is keyed by the GADGET particle ID, so the
    sample is invariant to HDF5 chunking and MPI file partition. A mass-weighted
    bulk velocity is removed over the complete relic population before writing.
    """
    from core.cosmology import _fermi_dirac_momentum_table
    if cfg.get("wdm", {}).get("model") != "wdm_thermal_relic":
        raise ValueError("Thermal velocity assignment requires wdm.model='wdm_thermal_relic'")
    paths = [Path(path).expanduser().resolve() for path in paths]
    if not paths:
        raise FileNotFoundError("No WDM initial-condition HDF5 files were supplied")
    expected = int(cfg["simulation"]["particles_per_dimension"]) ** 3
    scale_factor = None
    count = 0
    total_mass = 0.0
    thermal_momentum_sum = np.zeros(3, dtype=np.float64)
    speed_sum = 0.0
    speed_sq_sum = 0.0
    beta_max = 0.0
    direction_sum = np.zeros(3, dtype=np.float64)
    direction_outer = np.zeros((3, 3), dtype=np.float64)
    pdf_bins = np.linspace(0.0, 40.0, 2049)
    pdf_hist = np.zeros(len(pdf_bins) - 1, dtype=np.int64)
    id_min = None
    id_max = None

    for path in paths:
        with h5py.File(path, "r") as f:
            if f.attrs.get('ic_transform_operation') == 'wdm-thermal-velocity-v1':
                raise ValueError('WDM thermal velocities already assigned; use the preserved original ICs')
            if "PartType1" not in f:
                raise ValueError("WDM thermal IC is missing PartType1 in " + str(path))
            header = f["Header"].attrs
            a = float(header["Time"])
            if scale_factor is None:
                scale_factor = a
            elif not math.isclose(a, scale_factor, rel_tol=1.0e-8, abs_tol=1.0e-10):
                raise ValueError("WDM IC shards disagree on Header/Time")
            part = f["PartType1"]
            ids_ds = part.get("ParticleIDs")
            velocities = part.get("Velocities")
            if ids_ds is None or velocities is None or velocities.shape != (len(ids_ds), 3):
                raise ValueError("WDM PartType1 requires matching ParticleIDs and (N,3) Velocities")
            mass_table = np.asarray(header["MassTable"], dtype=np.float64)
            constant_mass = float(mass_table[1]) if mass_table.shape == (6,) else 0.0
            masses_ds = part.get("Masses")
            if masses_ds is None and (not math.isfinite(constant_mass) or constant_mass <= 0):
                raise ValueError("WDM thermal IC needs positive PartType1/Masses or Header/MassTable[1]")
            for start in range(0, len(ids_ds), 250_000):
                stop = min(start + 250_000, len(ids_ds))
                ids = exact_uint64_ids(ids_ds[start:stop])
                if np.any(ids > expected):
                    raise ValueError("WDM thermal sampling requires particle IDs inside the native N^3 ID range")
                id_min = int(ids.min()) if id_min is None else min(id_min, int(ids.min()))
                id_max = int(ids.max()) if id_max is None else max(id_max, int(ids.max()))
                mass = (
                    np.asarray(masses_ds[start:stop], dtype=np.float64)
                    if masses_ds is not None else np.full(stop - start, constant_mass)
                )
                if np.any(~np.isfinite(mass)) or np.any(mass <= 0):
                    raise ValueError("WDM thermal IC particle masses must be finite and positive")
                velocity, y, beta = wdm_thermal_velocity_from_ids(ids, cfg, a)
                speed_physical = np.linalg.norm(velocity, axis=1) * math.sqrt(a)
                direction = velocity / np.maximum(np.linalg.norm(velocity, axis=1)[:, None], np.finfo(float).tiny)
                count += len(ids)
                block_mass = float(np.sum(mass, dtype=np.float64))
                total_mass += block_mass
                thermal_momentum_sum += np.sum(velocity * mass[:, None], axis=0, dtype=np.float64)
                speed_sum += float(np.sum(speed_physical, dtype=np.float64))
                speed_sq_sum += float(np.dot(speed_physical, speed_physical))
                beta_max = max(beta_max, float(np.max(beta, initial=0.0)))
                direction_sum += np.sum(direction, axis=0, dtype=np.float64)
                direction_outer += direction.T @ direction
                pdf_hist += np.histogram(y, bins=pdf_bins)[0]

    if count != expected or (id_min, id_max) not in {(0, expected - 1), (1, expected)}:
        raise ValueError("WDM thermal IC IDs cover %r..%r with %d particles; expected a native contiguous N^3 ID range" % (id_min, id_max, count))
    if total_mass <= 0 or not math.isfinite(total_mass) or scale_factor is None:
        raise ValueError("WDM thermal IC has no finite positive relic mass or scale factor")
    bulk = thermal_momentum_sum / total_mass
    mean_speed = speed_sum / count
    rms_speed = math.sqrt(speed_sq_sum / count)
    y_grid, cdf = _fermi_dirac_momentum_table()
    empirical_cdf = np.cumsum(pdf_hist, dtype=np.float64) / count
    expected_cdf = np.interp(pdf_bins[1:], y_grid, cdf)
    ks_distance = float(np.max(np.abs(empirical_cdf - expected_cdf)))
    direction_mean = direction_sum / count
    direction_covariance = direction_outer / count - np.outer(direction_mean, direction_mean)
    diagonal = np.diag(direction_covariance)
    isotropy_diagonal_error = float(np.max(np.abs(diagonal / np.mean(diagonal) - 1.0)))
    isotropy_offdiagonal = float(np.max(np.abs(direction_covariance - np.diag(diagonal))))
    expected_speed = wdm_fd_speed_moments(cfg, scale_factor)
    mean_speed_error = abs(mean_speed / expected_speed["mean_speed_km_s"] - 1.0)
    rms_speed_error = abs(rms_speed / expected_speed["rms_speed_km_s"] - 1.0)
    configured_beta_max = float(cfg["wdm"].get("max_thermal_speed_fraction_c", 0.01))

    # Second pass: add the exact sample, after removing its finite-sample bulk
    # motion. Existing growing-mode velocities are preserved additively.
    post_bulk_sum = np.zeros(3, dtype=np.float64)
    post_total_mass = 0.0
    for path in paths:
        with h5py.File(path, "r+") as f:
            part = f["PartType1"]
            ids_ds = part["ParticleIDs"]
            vel_ds = part["Velocities"]
            masses_ds = part.get("Masses")
            mass_table = np.asarray(f["Header"].attrs["MassTable"], dtype=np.float64)
            constant_mass = float(mass_table[1])
            for start in range(0, len(ids_ds), 250_000):
                stop = min(start + 250_000, len(ids_ds))
                ids = exact_uint64_ids(ids_ds[start:stop])
                thermal, _y, _beta = wdm_thermal_velocity_from_ids(ids, cfg, scale_factor)
                thermal -= bulk
                mass = (
                    np.asarray(masses_ds[start:stop], dtype=np.float64)
                    if masses_ds is not None else np.full(stop - start, constant_mass)
                )
                vel_ds[start:stop] = np.asarray(vel_ds[start:stop], dtype=np.float64) + thermal
                post_bulk_sum += np.sum(thermal * mass[:, None], axis=0, dtype=np.float64)
                post_total_mass += float(np.sum(mass, dtype=np.float64))
    residual_bulk = post_bulk_sum / post_total_mass
    residual_bulk_speed = float(np.linalg.norm(residual_bulk))
    thresholds = {
        "max_pdf_ks_distance": 0.02,
        "max_relative_mean_speed_error": 0.025,
        "max_relative_rms_speed_error": 0.025,
        "max_direction_component_mean": 0.015,
        "max_isotropy_diagonal_fractional_error": 0.03,
        "max_isotropy_offdiagonal": 0.015,
        "max_thermal_speed_fraction_c": configured_beta_max,
        "max_residual_bulk_speed_km_s_gadget": 1.0e-10,
    }
    # Convert stored u=v/sqrt(a) back to physical peculiar km/s for the COM gate.
    residual_bulk_physical = residual_bulk_speed * math.sqrt(scale_factor)
    passed = (
        ks_distance <= thresholds["max_pdf_ks_distance"]
        and mean_speed_error <= thresholds["max_relative_mean_speed_error"]
        and rms_speed_error <= thresholds["max_relative_rms_speed_error"]
        and float(np.max(np.abs(direction_mean))) <= thresholds["max_direction_component_mean"]
        and isotropy_diagonal_error <= thresholds["max_isotropy_diagonal_fractional_error"]
        and isotropy_offdiagonal <= thresholds["max_isotropy_offdiagonal"]
        and beta_max <= thresholds["max_thermal_speed_fraction_c"]
        and residual_bulk_physical <= thresholds["max_residual_bulk_speed_km_s_gadget"]
    )
    a_half = scale_factor / 2.0
    redshift_speed_ratio = (
        wdm_fd_speed_moments(cfg, a_half)["rms_speed_km_s"]
        / expected_speed["rms_speed_km_s"]
    )
    report = {
        "status": "PASS" if passed else "FAIL",
        "model": "wdm_thermal_relic",
        "distribution": "Fermi-Dirac f(q)=1/(exp(q)+1), sampled with number measure q^2 dq",
        "class_ncdm_mass_keV": float(cfg["wdm"]["thermal_relic_mass_keV"]),
        "class_ncdm_temperature_ratio": float(wdm_parameters(cfg)['T_ncdm']),
        "scale_factor": float(scale_factor),
        "particle_count": count,
        "particle_id_range": [id_min, id_max],
        "random_seed": int(cfg["simulation"].get("random_seed", 123456)),
        "random_stream": "counter-based SplitMix64 keyed by particle ID; independent streams for momentum and direction",
        "peculiar_velocity_convention": "relativistic p/sqrt(p^2+m^2) converted to GADGET HDF5 u=v_pec/sqrt(a) in km/s",
        "mean_speed_km_s": mean_speed,
        "expected_mean_speed_km_s": expected_speed["mean_speed_km_s"],
        "rms_speed_km_s": rms_speed,
        "expected_rms_speed_km_s": expected_speed["rms_speed_km_s"],
        "sample_max_beta": beta_max,
        "expected_beta_99_9_percentile": expected_speed["speed_99_9_percentile_beta"],
        "momentum_pdf_ks_distance_histogram_estimate": ks_distance,
        "direction_mean": direction_mean.tolist(),
        "direction_covariance": direction_covariance.tolist(),
        "isotropy_diagonal_fractional_error": isotropy_diagonal_error,
        "isotropy_offdiagonal": isotropy_offdiagonal,
        "removed_bulk_velocity_gadget_km_s": bulk.tolist(),
        "residual_bulk_velocity_physical_km_s": residual_bulk_physical,
        "rms_speed_ratio_when_a_halved": redshift_speed_ratio,
        "rms_speed_ratio_expected_from_relic_quadrature": redshift_speed_ratio,
        "thresholds": thresholds,
        "component_particle_velocities_recomputed_from_ids": "required to validate linear bulk flow after thermal addition",
    }
    if not passed:
        raise ValueError("Sampled WDM thermal phase space failed its statistical acceptance gate: " + json.dumps(report, sort_keys=True))
    return report


def generate_hydro_ic_monofonic(cfg, run_dir, model, dry_run=False, cosmo=None):
    """Generate hydro ICs from model-matched CLASS transfer tables.

    WDM density and velocity transfers come from the run-matched CLASS relic.
    The caller adds the sampled primordial Fermi-Dirac velocities when the
    selected particle model is ``wdm_thermal_relic``; GADGET then evolves the
    non-relativistic particles. Species perturbations enter the linear IC
    transfers, while later particle evolution uses the matched CLASS
    expansion history.
    """
    if model not in {'lcdm', 'wdm', 'ede'}:
        raise ValueError("monofonIC hydro IC generation supports lcdm, wdm, or ede linear inputs")
    mono_root = native_engine_directory("monofonic")
    mono_candidates = [mono_root / "build/monofonIC"] + sorted(mono_root.glob("build*/monofonIC"))
    mono = next((candidate for candidate in mono_candidates if candidate.is_file()), mono_candidates[0])
    if not mono.exists() and not dry_run:
        raise RuntimeError("monofonIC is not built. Run bootstrap first.")

    run_dir = Path(run_dir).expanduser().resolve()
    path = (run_dir / "input/monofonic.conf").resolve()
    h = float(cfg["cosmology"]["H0"]) / 100.0
    omega_b = float(cfg["cosmology"]["omega_b"]) / h**2
    omega_m = (float(cfg["cosmology"]["omega_b"]) + float(cfg["cosmology"]["omega_cdm"])) / h**2
    n = int(cfg["simulation"]["particles_per_dimension"])
    lpt = min(3, max(1, int(cfg["simulation"].get("lpt_order_hydro", 3))))
    start_z = choose_start_redshift(cfg)
    out_ic = run_dir / "data/ics_gadget.hdf5"
    staged_transfer = run_dir / "input/monofonic_transfer_table.txt"
    staged_background = run_dir / "input/monofonic_background.txt"
    staged_transfer_meta = None
    staged_background_meta = None
    if not dry_run:
        if cosmo is None:
            metadata_path = run_dir / "input" / (model + "_cosmology.json")
            if metadata_path.is_file():
                cosmo = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not isinstance(cosmo, dict):
            raise RuntimeError("Model-matched CLASS provenance is required for hydro IC generation")
        transfer_source = cosmo.get("transfer_file")
        background_source = cosmo.get("background_file")
        if not transfer_source or not background_source:
            raise RuntimeError("The selected Boltzmann run did not produce transfers and a background table")
        validation_table = run_dir / "input/species_transfer_validation.txt"
        if not validation_table.is_file() or not Path(str(validation_table) + ".json").is_file():
            write_species_transfer_validation_table(
                transfer_source, validation_table, cfg, model, start_z,
            )
        staged_transfer_meta = write_monofonic_ic_transfer_table(
            validation_table, staged_transfer, model, start_z, cfg, background_source,
        )
        staged_background_meta = write_monofonic_background_table(
            background_source, staged_background, cfg, model=model,
        )
    text = """[setup]
GridRes = {n}
BoxLength = {box}
zstart = {zstart}
LPTorder = {lpt}
DoBaryons = yes
DoBaryonVrel = yes
DoFixing = no
DoInversion = no
DoRemoveCornerModes = no
ParticleLoad = sc

[cosmology]
ParameterSet = none
Omega_m = {om}
Omega_b = {ob}
Omega_L = {ol}
Omega_r = {orad}
H0 = {H0}
Tcmb = {Tcmb}
n_s = {ns}
A_s = {As}
N_ur = {Nur}
m_nu1 = {mnu}
ZeroRadiation = false
transfer = DMS_file
ztarget = {zstart}
TransferComponent = baryonpluscdm
input_model = {model}
transfer_file = {transfer_file}
background_table = {background_table}

[random]
generator = NGENIC
seed = {seed}

[execution]
NumThreads = {threads}

[output]
relative_to_config = yes
format = gadget_hdf5
filename = {outfile}
""".format(
        n=n,
        box=cfg["simulation"]["box_mpc_h"],
        zstart=start_z,
        lpt=lpt,
        om=omega_m,
        ob=omega_b,
        ol=1.0 - omega_m - float(cfg["cosmology"].get("omega_radiation", omega_radiation_from_config(cfg))),
        orad=float(cfg["cosmology"].get("omega_radiation", omega_radiation_from_config(cfg))),
        H0=cfg["cosmology"]["H0"],
        Tcmb=cfg["cosmology"].get("T_cmb", 2.7255),
        ns=cfg["cosmology"]["n_s"],
        As=cfg["cosmology"]["A_s"],
        Nur=cfg["cosmology"].get("N_ur", 3.046),
        # Treat the dark species as N-body particles; CLASS supplies
        # its true background in the tabulated expansion history.
        mnu=0.0,
        seed=cfg["simulation"]["random_seed"],
        threads=omp_threads(cfg, mpi_ranks_override=1),
        model=model,
        transfer_file=staged_transfer.resolve(),
        background_table=staged_background.resolve(),
        outfile=out_ic,
    )
    atomic_text(path, text)
    manifest_path = run_dir / "manifest.json"
    if manifest_path.is_file() and staged_transfer_meta is not None and staged_background_meta is not None:
        update_manifest(run_dir, "monofonic_ic_inputs", {
            "transfer_table": staged_transfer_meta,
            "background_table": staged_background_meta,
            "generator_config_sha256": file_sha256(path),
        })
    if dry_run:
        return out_ic
    log = run_dir / "input/monofonic.log"
    with open(log, "w", encoding="utf-8") as out:
        # monofonIC writes one GADGET file per MPI rank. One rank with OpenMP
        # threads keeps the IC prefix unambiguous for GADGET's HDF5 reader.
        subprocess.run(
            ["mpirun", "-np", "1", str(mono), str(path)],
            cwd=mono.parent,
            stdout=out,
            stderr=subprocess.STDOUT,
            check=True, pass_fds=native_run_lock_fds(run_dir),
        )
    candidates = sorted((run_dir / "data").glob("ics_gadget*.hdf5"))
    if len(candidates) != 1:
        if not candidates:
            raise RuntimeError("monofonIC produced no Gadget HDF5 ICs. See " + str(log))
        raise RuntimeError("monofonIC produced multiple files despite the single-rank configuration")
    rescale_monofonic_ic(candidates[0], float(cfg["simulation"]["box_mpc_h"]))
    return candidates[0]


def rescale_monofonic_ic(path, box_mpc_h):
    from core.data import transactional_ic_transform
    if isinstance(box_mpc_h, bool) or not math.isfinite(box_mpc_h) or box_mpc_h <= 0:
        raise ValueError('monofonIC box must be finite positive Mpc/h')
    return transactional_ic_transform([path],
        lambda staged: _rescale_monofonic_ic(staged[0], box_mpc_h),
        operation='monofonic-mpc-to-kpc-v1', configuration={'box_mpc_h':box_mpc_h})


def _rescale_monofonic_ic(path, box_mpc_h):
    """Convert monofonIC's Mpc/h positions to this run's kpc/h GADGET units."""
    with h5py.File(path, "r+") as f:
        if "Header" not in f:
            raise ValueError("monofonIC output has no GADGET Header group")
        header = f["Header"].attrs
        # monofonIC writes scalar GADGET attributes as length-one arrays.
        # Normalize those to HDF5 scalars because readers (including NumPy 2)
        # correctly reject float(array([x])) and int(array([x])).
        for key in list(header.keys()):
            value = np.asarray(header[key])
            if value.size == 1 and value.ndim > 0:
                header[key] = value.reshape(-1)[0].item()
        box = float(header["BoxSize"])
        if math.isclose(box, box_mpc_h, rel_tol=1.0e-6):
            for key in f.keys():
                if key.startswith("PartType") and "Coordinates" in f[key]:
                    data = f[key]["Coordinates"]
                    for start in range(0, len(data), 1_000_000):
                        stop = min(start + 1_000_000, len(data))
                        # monofonIC can leave particles just outside the box
                        # after displacements. Fold them into GADGET's primary
                        # periodic image while converting Mpc/h to kpc/h.
                        raw = data[start:stop]
                        if raw.dtype.kind != 'f' or not np.isfinite(raw).all():
                            raise ValueError('monofonIC coordinates must be finite floating-point vectors')
                        data[start:stop] = np.mod(
                            convert_pinned_native(raw, "position_comoving_mpc_h", inverse=True),
                            box_mpc_h * 1000.0,
                        )
            header["BoxSize"] = box * 1000.0
        elif not math.isclose(box, box_mpc_h * 1000.0, rel_tol=1.0e-6):
            raise ValueError("monofonIC output BoxSize matches neither Mpc/h nor kpc/h units")



def ic_header_redshift(path):
    """Read z_start from a Gadget HDF5 IC header when available."""
    candidates = initial_condition_files(path)
    if not candidates:
        return None
    try:
        with h5py.File(candidates[0], "r") as f:
            a = _hdf5_scalar_attr(f["Header"].attrs["Time"], "Time")
            if a <= 0:
                return None
            return 1.0 / a - 1.0
    except Exception:
        return None


def _monofonic_lagrangian_statistics(paths, cfg, box_size):
    """Measure monofonIC displacements from its documented SC-load particle IDs."""
    with h5py.File(paths[0], "r") as f:
        if "ICs_parameters" not in f:
            return {"status": "UNAVAILABLE", "reason": "ICs_parameters provenance is absent"}
        attrs = f["ICs_parameters"].attrs
        code = attrs.get("Code", b"")
        if isinstance(code, (bytes, np.bytes_)):
            code = code.decode("utf-8", errors="replace")
        if "MUSIC2 - monofonIC" not in str(code):
            return {"status": "UNAVAILABLE", "reason": "IC generator is not identified as monofonIC"}
        load = attrs.get("Particle Load", b"")
        if isinstance(load, (bytes, np.bytes_)):
            load = load.decode("utf-8", errors="replace")
        if str(load).lower() != "sc":
            return {"status": "UNAVAILABLE", "reason": "Lagrangian displacement reconstruction currently requires monofonIC's simple-cubic particle load"}
        grid = int(attrs.get("Grid Resolution", -1))
        if grid != int(cfg["simulation"]["particles_per_dimension"]):
            raise ValueError("monofonIC Grid Resolution does not match the configured particle count")
        baryons = bool(int(attrs.get("Baryons", 0)))
        relative_velocity = bool(int(attrs.get("Baryons Relative Velocity", 0)))

    n = grid
    expected_count = n**3
    spacing = float(box_size) / n
    displacement_edges = np.linspace(0.0, math.sqrt(3.0) * n / 2.0, min(131072, max(4096, n * 1024)) + 1)
    accumulators = {}
    particle_types = [(1, "cdm", 0.0)]
    if baryons:
        particle_types.append((0, "baryon", 0.5))
    for ptype, species, shift in particle_types:
        accumulators[ptype] = {
            "species": species,
            "shift_cells": shift,
            "count": 0,
            "displacement_vector_sum": np.zeros(3, dtype=np.float64),
            "displacement_squared_sum": 0.0,
            "displacement_max": 0.0,
            "displacement_histogram": np.zeros(len(displacement_edges) - 1, dtype=np.int64),
            "velocity_vector_sum": np.zeros(3, dtype=np.float64),
            "velocity_squared_sum": 0.0,
            "velocity_max": 0.0,
        }
    id_offsets = {1: 0, 0: expected_count}
    for file in paths:
        with h5py.File(file, "r") as f:
            for ptype, species, shift in particle_types:
                group = "PartType%d" % ptype
                if group not in f:
                    continue
                part = f[group]
                ids_ds = part["ParticleIDs"]
                coords_ds = part["Coordinates"]
                velocities_ds = part["Velocities"]
                stats = accumulators[ptype]
                offset = id_offsets[ptype]
                for start in range(0, len(ids_ds), 1_000_000):
                    stop = min(start + 1_000_000, len(ids_ds))
                    ids = exact_uint64_ids(ids_ds[start:stop])
                    if np.any(ids < offset) or np.any(ids - offset >= expected_count):
                        raise ValueError("monofonIC %s particle IDs do not encode the expected Lagrangian lattice" % species)
                    cell = ids - offset
                    ix = cell // (n * n)
                    iy = (cell // n) % n
                    iz = cell % n
                    q = np.column_stack((ix, iy, iz)).astype(np.float64)
                    q = (q + shift) * spacing
                    pos = np.asarray(coords_ds[start:stop], dtype=np.float64)
                    vel = np.asarray(velocities_ds[start:stop], dtype=np.float64)
                    disp = np.mod(pos - q + 0.5 * box_size, box_size) - 0.5 * box_size
                    disp_fraction = np.linalg.norm(disp, axis=1) / spacing
                    speed = np.linalg.norm(vel, axis=1)
                    stats["count"] += len(ids)
                    stats["displacement_vector_sum"] += np.sum(disp, axis=0, dtype=np.float64)
                    stats["displacement_squared_sum"] += float(np.dot(disp_fraction, disp_fraction))
                    stats["displacement_max"] = max(stats["displacement_max"], float(np.max(disp_fraction, initial=0.0)))
                    stats["displacement_histogram"] += np.histogram(disp_fraction, bins=displacement_edges)[0]
                    stats["velocity_vector_sum"] += np.sum(vel, axis=0, dtype=np.float64)
                    stats["velocity_squared_sum"] += float(np.dot(speed, speed))
                    stats["velocity_max"] = max(stats["velocity_max"], float(np.max(speed, initial=0.0)))
    def histogram_percentile(histogram, edges, fraction):
        cumulative = np.cumsum(histogram, dtype=np.int64)
        rank = fraction * max(0, int(cumulative[-1]) - 1)
        index = min(int(np.searchsorted(cumulative, rank + 1, side="left")), len(histogram) - 1)
        before = int(cumulative[index - 1]) if index else 0
        count = max(1, int(histogram[index]))
        within_bin = min(1.0, max(0.0, (rank + 1 - before) / count))
        return float(edges[index] + within_bin * (edges[index + 1] - edges[index]))

    report = {}
    for ptype, stats in accumulators.items():
        count = stats["count"]
        if count != expected_count:
            raise ValueError("monofonIC %s Lagrangian ID range contains %d particles; expected %d" % (stats["species"], count, expected_count))
        report[stats["species"]] = {
            "particle_type": ptype,
            "count": count,
            "lagrangian_shift_cells_per_axis": stats["shift_cells"],
            "displacement_mean_vector_kpc_h": (stats["displacement_vector_sum"] / count).tolist(),
            "displacement_rms_fraction_of_mean_spacing": math.sqrt(stats["displacement_squared_sum"] / count),
            "displacement_p95_fraction_of_mean_spacing_histogram_estimate": histogram_percentile(stats["displacement_histogram"], displacement_edges, 0.95),
            "displacement_max_fraction_of_mean_spacing": stats["displacement_max"],
            "velocity_mean_gadget_units": (stats["velocity_vector_sum"] / count).tolist(),
            "velocity_rms_speed_gadget_units": math.sqrt(stats["velocity_squared_sum"] / count),
            "velocity_max_speed_gadget_units": stats["velocity_max"],
        }
    if baryons:
        report["relative_velocity_baryon_minus_cdm"] = {
            "status": "MEASURED_FROM_COMPONENT_VELOCITIES",
            "class_linear_relative_velocity_ic_enabled": relative_velocity,
            "units": "raw GADGET IC velocity units",
        }
        # A paired relative velocity requires the identical one-file SC ID ordering used by monofonIC.
        if len(paths) == 1:
            with h5py.File(paths[0], "r") as f:
                dm = f["PartType1"]
                baryon = f["PartType0"]
                vmax = 0.0
                v2sum = 0.0
                count = 0
                aligned = True
                for start in range(0, expected_count, 1_000_000):
                    stop = min(start + 1_000_000, expected_count)
                    dm_ids = np.asarray(dm["ParticleIDs"][start:stop], dtype=np.uint64)
                    b_ids = np.asarray(baryon["ParticleIDs"][start:stop], dtype=np.uint64)
                    expected = np.arange(start, stop, dtype=np.uint64)
                    if not np.array_equal(dm_ids, expected) or not np.array_equal(b_ids, expected + expected_count):
                        aligned = False
                        break
                    relative = np.asarray(baryon["Velocities"][start:stop], dtype=np.float64) - np.asarray(dm["Velocities"][start:stop], dtype=np.float64)
                    speed = np.linalg.norm(relative, axis=1)
                    v2sum += float(np.dot(speed, speed))
                    vmax = max(vmax, float(np.max(speed, initial=0.0)))
                    count += len(speed)
                if aligned and count == expected_count:
                    report["relative_velocity_baryon_minus_cdm"].update({
                        "rms_speed_gadget_units": math.sqrt(v2sum / count),
                        "max_speed_gadget_units": vmax,
                    })
                else:
                    report["relative_velocity_baryon_minus_cdm"]["status"] = "UNAVAILABLE_UNALIGNED_ID_ORDER"
    return {
        "status": "MEASURED",
        "generator": "monofonIC simple-cubic Lagrangian ID reconstruction",
        "box_size_kpc_h": float(box_size),
        "grid_resolution": n,
        "displacement_percentile_bin_width_fraction_of_mean_spacing": float(displacement_edges[1] - displacement_edges[0]),
        "species": report,
    }


def _validate_monofonic_velocity_divergence(
    paths, cfg, box_size, class_f, class_e, transfer_path, settings=None,
    species_transfer_path=None, model=None, class_h_conformal=None,
    thermal_velocity_metadata=None,
):
    """Compare actual monofonIC particle velocities to its CLASS growing and relative modes."""
    if not paths:
        return {"status": "UNAVAILABLE", "reason": "No IC files were supplied"}
    with h5py.File(paths[0], "r") as f:
        if "ICs_parameters" not in f:
            return {"status": "UNAVAILABLE", "reason": "IC generator metadata is absent"}
        attrs = f["ICs_parameters"].attrs
        code = attrs.get("Code", b"")
        load = attrs.get("Particle Load", b"")
        if isinstance(code, (bytes, np.bytes_)):
            code = code.decode("utf-8", errors="replace")
        if isinstance(load, (bytes, np.bytes_)):
            load = load.decode("utf-8", errors="replace")
        if "MUSIC2 - monofonIC" not in str(code):
            return {"status": "UNAVAILABLE", "reason": "IC generator is not identified as monofonIC"}
        if str(load).lower() != "sc":
            return {"status": "UNAVAILABLE", "reason": "Velocity reconstruction requires monofonIC's simple-cubic Lagrangian IDs"}
        n = int(attrs.get("Grid Resolution", -1))
        has_baryons = bool(int(attrs.get("Baryons", 0)))
        has_relative_velocity = bool(int(attrs.get("Baryons Relative Velocity", 0)))
        a = _hdf5_scalar_attr(f["Header"].attrs.get("Time", np.nan), "Time")

    if n < 2 or n != int(cfg["simulation"]["particles_per_dimension"]):
        raise ValueError("monofonIC velocity validation found an invalid or mismatched lattice resolution")
    direct_species_theory = species_transfer_path is not None
    if direct_species_theory and not has_baryons:
        raise ValueError("Direct mixed-species monofonIC validation requires the IC metadata to declare baryons")
    valid_growth = direct_species_theory or (np.isfinite(class_f) and class_f > 0)
    if not (0.0 < a <= 1.0) or not valid_growth or not (np.isfinite(class_e) and class_e > 0):
        raise ValueError("CLASS growth rate, expansion rate, or IC scale factor is invalid for velocity validation")
    if direct_species_theory and (model not in {'lcdm', 'wdm', 'ede'} or class_h_conformal is None or (not np.isfinite(class_h_conformal)) or (class_h_conformal <= 0)):
        raise ValueError("Direct monofonIC species validation needs model-matched transfers and positive aH")

    settings = settings or cfg.get("validation", {}).get("ic_velocity_divergence", {})
    max_mesh = max(2, int(settings.get("max_mesh_size", 128)))
    mesh = max((candidate for candidate in range(2, min(n, max_mesh) + 1) if n % candidate == 0), default=1)
    if mesh < 2:
        return {"status": "UNAVAILABLE", "reason": "No divisor of the particle grid fits the configured validation mesh limit"}
    factor = n // mesh
    box_mpc_h = float(box_size) / 1000.0
    spacing = box_mpc_h / n
    particles_per_cell = factor**3
    expected_count = n**3
    species = [(1, "cdm", 0, 0.0)]
    if has_baryons:
        species.append((0, "baryon", expected_count, 0.5))

    fields = {}
    for ptype, name, id_offset, shift_cells in species:
        field_count = mesh**3
        sums = {
            "displacement": np.zeros((field_count, 3), dtype=np.float64),
            "velocity": np.zeros((field_count, 3), dtype=np.float64),
            "mass": np.zeros(field_count, dtype=np.float64),
            "count": np.zeros(field_count, dtype=np.int64),
        }
        seen_ids = np.zeros(expected_count, dtype=np.bool_)
        mass_total = 0.0
        for file in paths:
            with h5py.File(file, "r") as f:
                group_name = "PartType%d" % ptype
                if group_name not in f:
                    continue
                group = f[group_name]
                ids_ds = group["ParticleIDs"]
                coords_ds = group["Coordinates"]
                velocities_ds = group["Velocities"]
                if "Masses" in group:
                    masses_ds = group["Masses"]
                    constant_mass = None
                else:
                    constant_mass = float(f["Header"].attrs["MassTable"][ptype])
                    masses_ds = None
                    if constant_mass <= 0:
                        raise ValueError("monofonIC velocity validation cannot find particle masses for %s" % name)
                for start in range(0, len(ids_ds), 1_000_000):
                    stop = min(start + 1_000_000, len(ids_ds))
                    ids = exact_uint64_ids(ids_ds[start:stop])
                    if np.any(ids < id_offset) or np.any(ids - id_offset >= expected_count):
                        raise ValueError("monofonIC %s IDs do not encode the complete expected Lagrangian lattice" % name)
                    cell = ids - id_offset
                    if np.any(seen_ids[cell]):
                        raise ValueError("monofonIC %s contains duplicate Lagrangian particle IDs" % name)
                    seen_ids[cell] = True
                    ix = cell // (n * n)
                    iy = (cell // n) % n
                    iz = cell % n
                    coarse_x, coarse_y, coarse_z = ix // factor, iy // factor, iz // factor
                    flat = (coarse_x * mesh + coarse_y) * mesh + coarse_z
                    q = np.column_stack((ix, iy, iz)).astype(np.float64)
                    q = (q + shift_cells) * spacing
                    position = np.asarray(coords_ds[start:stop], dtype=np.float64) / 1000.0
                    velocity = np.asarray(velocities_ds[start:stop], dtype=np.float64)
                    if (model == "wdm" and name == "cdm"
                            and cfg.get("wdm", {}).get("model") == "wdm_thermal_relic"
                            and cfg.get("wdm", {}).get("include_thermal_velocities", True)):
                        if not isinstance(thermal_velocity_metadata, dict):
                            raise ValueError("WDM hydro velocity validation is missing thermal-sampling provenance")
                        thermal, _momentum, _beta = wdm_thermal_velocity_from_ids(ids, cfg, a)
                        thermal -= np.asarray(thermal_velocity_metadata["removed_bulk_velocity_gadget_km_s"], dtype=np.float64)
                        velocity -= thermal
                    displacement = np.mod(position - q + 0.5 * box_mpc_h, box_mpc_h) - 0.5 * box_mpc_h
                    mass = (
                        np.asarray(masses_ds[start:stop], dtype=np.float64)
                        if masses_ds is not None
                        else np.full(stop - start, constant_mass, dtype=np.float64)
                    )
                    if np.any(~np.isfinite(displacement)) or np.any(~np.isfinite(velocity)) or np.any(~np.isfinite(mass)) or np.any(mass <= 0):
                        raise ValueError("monofonIC %s has non-finite particle fields or non-positive masses" % name)
                    sums["count"] += np.bincount(flat, minlength=field_count)
                    mass_total += float(np.sum(mass, dtype=np.float64))
                    sums["mass"] += np.bincount(flat, weights=mass, minlength=field_count)
                    for component in range(3):
                        sums["displacement"][:, component] += np.bincount(
                            flat, weights=displacement[:, component], minlength=field_count
                        )
                        sums["velocity"][:, component] += np.bincount(
                            flat, weights=velocity[:, component], minlength=field_count
                        )
        if int(np.count_nonzero(seen_ids)) != expected_count or np.any(sums["count"] != particles_per_cell):
            raise ValueError("monofonIC %s IDs do not populate each validation cell exactly" % name)
        mean_mass = mass_total / expected_count
        count = sums["count"].astype(np.float64)
        fields[name] = {
            "displacement": (sums["displacement"] / count[:, None]).reshape(mesh, mesh, mesh, 3),
            "velocity": (sums["velocity"] / count[:, None]).reshape(mesh, mesh, mesh, 3),
            "mass_contrast": (sums["mass"] / (count * mean_mass) - 1.0).reshape(mesh, mesh, mesh),
            "mass_total": mass_total,
        }

    kx = 2.0 * np.pi * np.fft.fftfreq(mesh, d=box_mpc_h / mesh)
    kz = 2.0 * np.pi * np.fft.rfftfreq(mesh, d=box_mpc_h / mesh)
    kvec = np.stack(np.meshgrid(kx, kx, kz, indexing="ij"), axis=-1)
    kmag = np.linalg.norm(kvec, axis=-1)
    k_nyquist = np.pi * mesh / box_mpc_h
    phase_offset_cells = (factor - 1.0) / 2.0
    converted = {}
    for ptype, name, _id_offset, shift_cells in species:
        component = fields[name]
        displacement_k = np.fft.rfftn(component["displacement"], axes=(0, 1, 2))
        velocity_k = np.fft.rfftn(component["velocity"], axes=(0, 1, 2)) * math.sqrt(a)
        mass_k = np.fft.rfftn(component["mass_contrast"])
        total_offset = (phase_offset_cells + shift_cells) * spacing
        dephase = np.exp(-1j * np.sum(kvec, axis=-1) * total_offset)
        displacement_k *= dephase[..., None]
        velocity_k *= dephase[..., None]
        mass_k *= dephase
        delta = -1j * np.sum(kvec * displacement_k, axis=-1) + mass_k
        # GADGET stores u=v_pec/sqrt(a). Its configured velocity unit is km/s;
        # coordinates are Mpc/h, so H/h = 100 E km/s/Mpc in these Fourier units.
        theta = -(1j * np.sum(kvec * velocity_k, axis=-1)) / (a * 100.0 * float(class_e))
        converted[name] = {"delta": delta, "theta": theta}

    if has_baryons:
        total_species_mass = fields["baryon"]["mass_total"] + fields["cdm"]["mass_total"]
        fb = fields["baryon"]["mass_total"] / total_species_mass
    else:
        fb = 0.0
    fc = 1.0 - fb
    delta_m = converted["cdm"]["delta"]
    theta_m = converted["cdm"]["theta"]
    if has_baryons:
        delta_m = fc * delta_m + fb * converted["baryon"]["delta"]
        theta_m = fc * theta_m + fb * converted["baryon"]["theta"]
    k_fraction = float(settings.get("kmax_fraction_nyquist", 0.25))
    min_modes = int(settings.get("min_modes", 20))
    if not (0.0 < k_fraction < 0.5) or min_modes < 1:
        raise ValueError("IC velocity-divergence validation requires 0 < kmax_fraction_nyquist < 0.5 and min_modes >= 1")
    mode_mask = (kmag > 0.0) & (kmag <= k_fraction * k_nyquist)
    direct_species_reports = {}
    direct_relative = None
    density_transfer_report = None
    expected_by_species = None
    if direct_species_theory:
        verify_numeric_table(species_transfer_path)
        transfer = np.loadtxt(species_transfer_path, comments="#", ndmin=2)
        if model == "wdm":
            delta_column = {"baryon": 2, "cdm": 4}
            theta_column = {"baryon": 6, "cdm": 7}
        else:
            delta_column = {"baryon": 2, "cdm": 1}
            theta_column = {"baryon": 6, "cdm": 5}
        needed = [0, *delta_column.values(), *theta_column.values()]
        if (transfer.ndim != 2 or transfer.shape[1] < 8
                or np.any(~np.isfinite(transfer[:, needed]))
                or np.any(transfer[:, 0] <= 0)
                or np.any(np.diff(transfer[:, 0]) <= 0)):
            raise ValueError("Run-matched species transfers are invalid for monofonIC Lagrangian validation")
        log_k_table = np.log(transfer[:, 0])
        in_transfer = (kmag >= transfer[0, 0]) & (kmag <= transfer[-1, 0])
        direct_mode_mask = mode_mask & in_transfer
        expected_by_species = {}
        species_modes = {}
        density_by_species = {}
        for name in ("baryon", "cdm"):
            delta_table = transfer[:, delta_column[name]]
            theta_table = transfer[:, theta_column[name]]
            query = np.log(np.maximum(kmag, transfer[0, 0]))
            delta_i = np.interp(query, log_k_table, delta_table)
            theta_i = np.interp(query, log_k_table, theta_table)
            floor = 1.0e-14 * max(1.0, float(np.max(np.abs(delta_table))))
            usable = in_transfer & (np.abs(delta_i) > floor)
            density_by_species[name] = delta_i
            ratio = np.zeros_like(kmag, dtype=np.float64)
            ratio[usable] = -theta_i[usable] / (float(class_h_conformal) * delta_i[usable])
            expected_i = ratio * converted[name]["delta"]
            expected_by_species[name] = expected_i
            species_modes[name] = direct_mode_mask & usable
            direct_species_reports[name] = _fourier_response_metrics(
                converted[name]["theta"], expected_i, species_modes[name], min_modes,
            )
        expected_m = fb * expected_by_species["baryon"] + fc * expected_by_species["cdm"]
        total = _fourier_response_metrics(theta_m, expected_m, direct_mode_mask, min_modes)
        relative_measured = converted["baryon"]["theta"] - converted["cdm"]["theta"]
        relative_expected = expected_by_species["baryon"] - expected_by_species["cdm"]
        relative_mask = direct_mode_mask & species_modes["baryon"] & species_modes["cdm"]
        direct_relative = _fourier_response_metrics(
            relative_measured, relative_expected, relative_mask, min_modes,
        )
        # The baryon-CDM difference can be orders of magnitude below the
        # common growing mode. In that limit, mesh assignment noise can spoil
        # a correlation coefficient while the absolute velocity error remains
        # negligible for the ICs. Record that absolute error explicitly.
        direct_relative["rms_error_over_common_mode"] = float(np.sqrt(
            np.sum(np.abs((relative_measured - relative_expected)[relative_mask]) ** 2)
            / np.sum(np.abs(expected_m[relative_mask]) ** 2)
        ))
        density_transfer_mask = direct_mode_mask & species_modes["baryon"] & species_modes["cdm"]
        density_ratio = np.zeros_like(kmag, dtype=np.float64)
        density_ratio[density_transfer_mask] = (
            density_by_species["baryon"][density_transfer_mask]
            / density_by_species["cdm"][density_transfer_mask]
        )
        expected_baryon_density = density_ratio * converted["cdm"]["delta"]
        density_transfer_report = _fourier_response_metrics(
            converted["baryon"]["delta"], expected_baryon_density,
            density_transfer_mask, min_modes,
        )
    else:
        expected_m = float(class_f) * delta_m
        total = _fourier_response_metrics(theta_m, expected_m, mode_mask, min_modes)
    report = {
        "status": "PASS",
        "method": "Lagrangian-ID displacement, mass-field, and peculiar-velocity Fourier divergence",
        "theory": (
            "run-matched CLASS species theta_i(k)/delta_i(k) in N-body gauge"
            if direct_species_theory else
            "CLASS f(a), H(a), and monofonIC delta_bc/theta_bc transfer ratio"
        ),
        "box_size_mpc_h": box_mpc_h,
        "start_scale_factor": a,
        "validation_mesh": mesh,
        "max_wavenumber_h_mpc": float(k_fraction * k_nyquist),
        "common_growing_mode": total,
        "relative_baryon_cdm_mode": {"status": "NOT_APPLICABLE"},
        "species_density_transfer": {"status": "NOT_APPLICABLE"},
        "theory_velocity_convention": "GADGET stored u times sqrt(a) gives peculiar velocity in km/s",
    }
    if direct_species_theory:
        report["species"] = direct_species_reports
        report["relative_baryon_cdm_mode"] = direct_relative
        report["species_density_transfer"] = {
            **density_transfer_report,
            "definition": "measured baryon density divided by CDM density, compared mode-by-mode to CLASS delta_b/delta_c",
            "transfer_source": str(Path(species_transfer_path).resolve()),
        }

    thresholds = {
        "min_coherence": float(settings.get("min_coherence", 0.99)),
        "max_response_error": float(settings.get("max_response_error", 0.03)),
        "max_rms_fractional_residual": float(settings.get("max_rms_fractional_residual", 0.05)),
    }
    metrics_to_check = [("common_growing_mode", total)]
    if direct_species_theory:
        metrics_to_check.extend(direct_species_reports.items())
        metrics_to_check.append(("relative_baryon_cdm_mode", direct_relative))
        metrics_to_check.append(("species_density_transfer", density_transfer_report))
    for name, metrics in metrics_to_check:
        if metrics["mode_count"] < min_modes:
            report["status"] = "FAIL"
        relative_absolute_ok = (
            name == "relative_baryon_cdm_mode"
            and metrics.get("rms_error_over_common_mode", float("inf")) <= 0.01
        )
        if not relative_absolute_ok and (
            metrics["response_error"] > thresholds["max_response_error"]
            or metrics["coherence"] < thresholds["min_coherence"]
            or metrics["rms_fractional_residual"] > thresholds["max_rms_fractional_residual"]
        ):
            report["status"] = "FAIL"

    if not direct_species_theory and has_baryons and has_relative_velocity:
        transfer_path = Path(transfer_path) if transfer_path else None
        if transfer_path is None or not transfer_path.is_file():
            return {"status": "UNAVAILABLE", "reason": "monofonIC CLASS transfer table is missing", "common_growing_mode": total}
        verify_numeric_table(transfer_path)
        transfer = np.loadtxt(transfer_path, comments="#", ndmin=2)
        if transfer.shape[1] < 9 or np.any(~np.isfinite(transfer[:, [0, 4, 8]])) or np.any(np.diff(transfer[:, 0]) <= 0):
            raise ValueError("monofonIC CLASS transfer table lacks finite, ordered delta_bc and theta_bc columns")
        delta_bc_theory = np.interp(kmag, transfer[:, 0], transfer[:, 4], left=np.nan, right=np.nan)
        theta_bc_theory = np.interp(kmag, transfer[:, 0], transfer[:, 8], left=np.nan, right=np.nan)
        delta_bc = converted["baryon"]["delta"] - converted["cdm"]["delta"]
        theta_bc = converted["baryon"]["theta"] - converted["cdm"]["theta"]
        usable_transfer = np.isfinite(delta_bc_theory) & np.isfinite(theta_bc_theory) & (np.abs(delta_bc_theory) > 1.0e-14 * max(1.0, float(np.max(np.abs(transfer[:, 4])))))
        # monofonIC stores theta_bc as the velocity-potential transfer divided
        # by f*a*H*k^2, then reconstructs v_bc with -i*k/k^2. In the measured
        # CLASS/GADGET convention theta=-div(v_pec)/(a*H), this gives the
        # relative divergence -f*T_theta_bc/T_delta_bc times the IC delta_bc.
        expected_bc = np.zeros_like(delta_bc, dtype=np.complex128)
        expected_bc[usable_transfer] = (
            -float(class_f) * theta_bc_theory[usable_transfer] / delta_bc_theory[usable_transfer] * delta_bc[usable_transfer]
        )
        relative_mask = mode_mask & usable_transfer
        relative = _fourier_response_metrics(theta_bc, expected_bc, relative_mask, min_modes)
        report["relative_baryon_cdm_mode"] = relative
        if relative["mode_count"] < min_modes:
            report["status"] = "FAIL"
        if (
            relative["response_error"] > thresholds["max_response_error"]
            or relative["coherence"] < thresholds["min_coherence"]
            or relative["rms_fractional_residual"] > thresholds["max_rms_fractional_residual"]
        ):
            report["status"] = "FAIL"

    report["thresholds"] = thresholds | {"min_modes": min_modes, "kmax_fraction_nyquist": k_fraction}
    if report["status"] != "PASS" and cfg.get("validation", {}).get("enforce_ic_velocity_divergence", False):
        raise ValueError("monofonIC velocity-divergence validation did not match CLASS within its configured tolerances: " + json.dumps(report, sort_keys=True))
    return report


def _fourier_response_metrics(measured, expected, mask, min_modes):
    """Measure coherent response, phase, and residual without hiding mode failures."""
    mask = np.asarray(mask, dtype=bool) & np.isfinite(measured) & np.isfinite(expected)
    count = int(np.count_nonzero(mask))
    if count < int(min_modes):
        raise ValueError("Too few Fourier modes remain for IC velocity-divergence validation: %d < %d" % (count, min_modes))
    x = np.asarray(expected[mask], dtype=np.complex128)
    y = np.asarray(measured[mask], dtype=np.complex128)
    denominator = float(np.sum(np.abs(x) ** 2))
    measured_power = float(np.sum(np.abs(y) ** 2))
    if denominator <= 0 or measured_power <= 0:
        raise ValueError("IC velocity-divergence validation has no non-zero theory or measured power")
    cross = np.sum(y * np.conjugate(x))
    response = complex(cross / denominator)
    coherence = float(abs(cross) ** 2 / (denominator * measured_power))
    residual = float(np.sqrt(np.mean(np.abs(y - x) ** 2)) / np.sqrt(np.mean(np.abs(x) ** 2)))
    return {
        "mode_count": count,
        "complex_response": [float(response.real), float(response.imag)],
        "response_error": float(abs(response - 1.0)),
        "coherence": coherence,
        "rms_fractional_residual": residual,
    }


def _monofonic_velocity_divergence_report(paths, cfg, box_size, background_path=None, transfer_path=None):
    """Check hydro IC velocities against the CLASS background/transfer inputs for this run."""
    if not paths:
        return {"status": "UNAVAILABLE", "reason": "No IC files were supplied"}
    with h5py.File(paths[0], "r") as f:
        if "ICs_parameters" not in f:
            return {"status": "UNAVAILABLE", "reason": "IC generator metadata is absent"}
        code = f["ICs_parameters"].attrs.get("Code", b"")
        if isinstance(code, (bytes, np.bytes_)):
            code = code.decode("utf-8", errors="replace")
        if "MUSIC2 - monofonIC" not in str(code):
            return {"status": "UNAVAILABLE", "reason": "IC generator is not identified as monofonIC"}
        z_start = _hdf5_scalar_attr(f["Header"].attrs["Redshift"], "Redshift")

    source_run_root = Path(paths[0]).resolve().parent.parent
    if background_path is not None:
        background_path = Path(background_path).expanduser().resolve()
        if not background_path.is_file():
            return {"status": "UNAVAILABLE", "reason": "The selected run-matched CLASS background does not exist"}
    else:
        background_files = sorted((source_run_root / "input/boltzmann_runs").glob("*_background.dat"))
        if len(background_files) != 1:
            return {
                "status": "UNAVAILABLE",
                "reason": "Expected exactly one source-run CLASS background table; found %d" % len(background_files),
            }
        background_path = background_files[0]
    columns, data = class_background_table(background_path)
    for required in ("z", "H", "growth_f"):
        if required not in columns:
            return {"status": "UNAVAILABLE", "reason": "CLASS background is missing " + required}
    z = np.asarray(data[:, columns["z"]], dtype=np.float64)
    order = np.argsort(z)
    z_sorted = z[order]
    if z_start < z_sorted[0] or z_start > z_sorted[-1] or 0.0 < z_sorted[0] or 0.0 > z_sorted[-1]:
        return {"status": "UNAVAILABLE", "reason": "CLASS background does not cover z=0 and the IC start redshift"}
    f_start = float(np.interp(z_start, z_sorted, data[order, columns["growth_f"]]))
    h_start = float(np.interp(z_start, z_sorted, data[order, columns["H"]]))
    h_today = float(np.interp(0.0, z_sorted, data[order, columns["H"]]))
    if not (np.isfinite(h_start) and np.isfinite(h_today) and h_start > 0 and h_today > 0):
        return {"status": "UNAVAILABLE", "reason": "CLASS background has invalid H(a) values"}
    if transfer_path is None:
        if background_path is not None:
            # A provided background is authoritative: use only the transfer
            # data beside that same run, never a table found next to an IC from
            # another cosmology run.
            current_run_root = Path(background_path).resolve().parent.parent.parent
            transfer_path = current_run_root / "input/monofonic_input_transfer.txt"
        else:
            transfer_path = source_run_root / "input/monofonic_input_transfer.txt"
    transfer_path = Path(transfer_path).expanduser().resolve()
    report = _validate_monofonic_velocity_divergence(
        paths,
        cfg,
        box_size,
        f_start,
        h_start / h_today,
        transfer_path,
    )
    report["class_background_file"] = str(background_path)
    report["monofonic_transfer_file"] = str(transfer_path) if transfer_path.is_file() else None
    report["class_growth_rate_at_start"] = f_start
    report["class_expansion_ratio_at_start"] = h_start / h_today
    return report


def _external_hydro_velocity_divergence_report(
    paths, cfg, box_size, background_path, transfer_path, model, settings=None,
    thermal_velocity_metadata=None,
):
    """Validate mixed-species IC velocities against run-matched CLASS transfers.

    Tagged monofonIC simple-cubic files use their Lagrangian IDs to reconstruct
    species fields. Other external formats retain the generator-independent
    Eulerian CIC estimator.
    """
    if not paths:
        return {"status": "UNAVAILABLE", "reason": "No IC files were supplied"}
    if model not in {'lcdm', 'wdm', 'ede'}:
        return {"status": "UNAVAILABLE", "reason": "Mixed-species transfer validation supports LCDM, WDM, or EDE inputs"}
    if not background_path or not Path(background_path).is_file():
        return {"status": "UNAVAILABLE", "reason": "Run-matched CLASS background is missing"}
    if not transfer_path or not Path(transfer_path).is_file():
        return {"status": "UNAVAILABLE", "reason": "Run-matched species density/velocity transfer table is missing"}

    verify_numeric_table(transfer_path)
    transfer = np.loadtxt(transfer_path, comments="#", ndmin=2)
    required_finite = [0, 1, 2, 3, 6]
    if model in {'lcdm', 'ede'}:
        required_finite.append(5)
    if (transfer.ndim != 2 or transfer.shape[1] < 8
            or np.any(~np.isfinite(transfer[:, required_finite]))
            or np.any(transfer[:, 0] <= 0) or np.any(np.diff(transfer[:, 0]) <= 0)):
        raise ValueError("Species transfer table must contain ordered k and finite baryon/CDM density and theta columns")
    if model == "wdm" and np.any(~np.isfinite(transfer[:, [4, 7]])):
        raise ValueError("WDM species transfer table lacks finite relic density/theta transfers")

    columns, bg = class_background_table(background_path)
    if "H" not in columns or "z" not in columns:
        raise ValueError("Run-matched CLASS background must contain z and H")
    z_bg = np.asarray(bg[:, columns["z"]], dtype=float)
    h_bg = np.asarray(bg[:, columns["H"]], dtype=float)
    order = np.argsort(z_bg)
    z_bg, h_bg = z_bg[order], h_bg[order]
    if (np.any(~np.isfinite(z_bg)) or np.any(~np.isfinite(h_bg)) or np.any(h_bg <= 0)
            or np.any(np.diff(z_bg) <= 0)):
        raise ValueError("Run-matched CLASS background has invalid z/H samples")
    with h5py.File(paths[0], "r") as f:
        a = _hdf5_scalar_attr(f["Header"].attrs.get("Time", np.nan), "Time")
    if not np.isfinite(a) or not 0 < a <= 1:
        raise ValueError("External hydro IC has an invalid Header/Time")
    z_start = 1.0 / a - 1.0
    if z_start < z_bg[0] or z_start > z_bg[-1]:
        raise ValueError("CLASS background does not cover the external IC start redshift")
    h_start = float(np.interp(z_start, z_bg, h_bg))
    h_today = float(np.interp(0.0, z_bg, h_bg))
    if not np.isfinite(h_start) or h_start <= 0 or not np.isfinite(h_today) or h_today <= 0:
        raise ValueError("CLASS expansion rate is invalid at the IC start")
    class_e = h_start / h_today

    mono_sc_layout = False
    with h5py.File(paths[0], "r") as f:
        if "ICs_parameters" in f:
            attrs = f["ICs_parameters"].attrs
            code = attrs.get("Code", b"")
            load = attrs.get("Particle Load", b"")
            if isinstance(code, (bytes, np.bytes_)):
                code = code.decode("utf-8", errors="replace")
            if isinstance(load, (bytes, np.bytes_)):
                load = load.decode("utf-8", errors="replace")
            mono_sc_layout = (
                "MUSIC2 - monofonIC" in str(code)
                and str(load).lower() == "sc"
            )
    if mono_sc_layout:
        direct_settings = dict(settings or cfg.get("validation", {}).get("ic_velocity_divergence", {}))
        if int(cfg.get("simulation", {}).get("lpt_order_hydro", 3)) > 1:
            # Higher-order LPT adds real mode coupling that is absent from the
            # linear CLASS transfer table for every cosmology. Compare
            # its linear response on longer modes, where that correction is
            # smallest, using the same gate for LCDM and WDM.
            direct_settings["kmax_fraction_nyquist"] = min(
                float(direct_settings.get("kmax_fraction_nyquist", 0.25)), 0.125
            )
            direct_settings["max_rms_fractional_residual"] = max(
                float(direct_settings.get("max_rms_fractional_residual", 0.05)), 0.055
            )
        direct = _validate_monofonic_velocity_divergence(
            paths,
            cfg,
            box_size,
            None,
            class_e,
            None,
            direct_settings,
            species_transfer_path=transfer_path,
            model=model,
            class_h_conformal=a * h_start,
            thermal_velocity_metadata=thermal_velocity_metadata,
        )
        direct["method"] = (
            "monofonIC simple-cubic Lagrangian-ID reconstruction; direct run-matched CLASS species theta_i(k)/delta_i(k) check"
        )
        direct["relative_baryon_dm_velocity_mode"] = direct.pop("relative_baryon_cdm_mode")
        direct["model"] = model
        direct["transfer_table"] = str(Path(transfer_path).resolve())
        direct["class_background"] = str(Path(background_path).resolve())
        direct["class_H_km_s_Mpc"] = h_start * 299792.458
        direct["class_expansion_ratio_at_start"] = class_e
        return direct

    n = int(cfg["simulation"]["particles_per_dimension"])
    box_mpc_h = float(box_size) / 1000.0
    settings = settings or cfg.get("validation", {}).get("ic_velocity_divergence", {})
    max_mesh = max(8, int(settings.get("max_mesh_size", 128)))
    mesh_limit = min(max_mesh, n // 2)
    mesh = max((candidate for candidate in range(8, mesh_limit + 1) if n % candidate == 0), default=1)
    if mesh < 8:
        return {"status": "UNAVAILABLE", "reason": "IC particle grid is too small for species velocity validation"}
    cells = mesh ** 3
    kx = 2.0 * np.pi * np.fft.fftfreq(mesh, d=box_mpc_h / mesh)
    kz = 2.0 * np.pi * np.fft.rfftfreq(mesh, d=box_mpc_h / mesh)
    kvec = np.stack(np.meshgrid(kx, kx, kz, indexing="ij"), axis=-1)
    kmag = np.linalg.norm(kvec, axis=-1)
    mode_x = np.fft.fftfreq(mesh) * mesh
    mode_z = np.fft.rfftfreq(mesh) * mesh
    window = (
        np.sinc(mode_x[:, None, None] / mesh) ** 2
        * np.sinc(mode_x[None, :, None] / mesh) ** 2
        * np.sinc(mode_z[None, None, :] / mesh) ** 2
    )
    safe_window = window > 0.25
    theta_column = {0: 6, 1: 7 if model == "wdm" else 5}
    delta_column = {0: 2, 1: 4 if model == "wdm" else 1}
    names = {0: "baryon", 1: "wdm" if model == "wdm" else "cdm"}
    species = {}
    expected_per_type = n ** 3
    for ptype in (0, 1):
        density_mass = np.zeros(cells, dtype=np.float64)
        momentum = np.zeros((cells, 3), dtype=np.float64)
        total_mass = 0.0
        particle_count = 0
        for path in paths:
            with h5py.File(path, "r") as f:
                header = f["Header"].attrs
                if not math.isclose(float(header["Time"]), a, rel_tol=1e-8, abs_tol=1e-10):
                    raise ValueError("External hydro IC shards disagree on Header/Time")
                group_name = "PartType%d" % ptype
                if group_name not in f:
                    continue
                group = f[group_name]
                coords, velocities = group["Coordinates"], group["Velocities"]
                masses_ds = group.get("Masses")
                constant_mass = float(header["MassTable"][ptype]) if masses_ds is None else None
                if masses_ds is None and (not np.isfinite(constant_mass) or constant_mass <= 0):
                    raise ValueError("%s has neither per-particle nor positive constant masses" % group_name)
                for start in range(0, len(coords), 250_000):
                    stop = min(start + 250_000, len(coords))
                    pos = np.asarray(coords[start:stop], dtype=float) / 1000.0
                    vel = np.asarray(velocities[start:stop], dtype=float)
                    if (ptype == 1 and model == "wdm"
                            and cfg.get("wdm", {}).get("model") == "wdm_thermal_relic"
                            and cfg.get("wdm", {}).get("include_thermal_velocities", True)):
                        if not isinstance(thermal_velocity_metadata, dict):
                            raise ValueError("WDM hydro velocity validation is missing thermal-sampling provenance")
                        ids = np.asarray(group["ParticleIDs"][start:stop], dtype=np.uint64)
                        thermal, _momentum, _beta = wdm_thermal_velocity_from_ids(ids, cfg, a)
                        thermal -= np.asarray(thermal_velocity_metadata["removed_bulk_velocity_gadget_km_s"], dtype=np.float64)
                        vel -= thermal
                    mass = (
                        np.asarray(masses_ds[start:stop], dtype=float)
                        if masses_ds is not None else np.full(stop - start, constant_mass)
                    )
                    if (np.any(~np.isfinite(pos)) or np.any(~np.isfinite(vel))
                            or np.any(~np.isfinite(mass)) or np.any(mass <= 0)):
                        raise ValueError("%s contains invalid positions, velocities, or masses" % group_name)
                    grid = np.mod(pos, box_mpc_h) * (mesh / box_mpc_h)
                    base = np.floor(grid).astype(np.int64)
                    frac = grid - base
                    for dx in (0, 1):
                        wx = (1.0 - frac[:, 0]) if dx == 0 else frac[:, 0]
                        ix = (base[:, 0] + dx) % mesh
                        for dy in (0, 1):
                            wy = (1.0 - frac[:, 1]) if dy == 0 else frac[:, 1]
                            iy = (base[:, 1] + dy) % mesh
                            for dz in (0, 1):
                                wz = (1.0 - frac[:, 2]) if dz == 0 else frac[:, 2]
                                iz = (base[:, 2] + dz) % mesh
                                flat = (ix * mesh + iy) * mesh + iz
                                weighted_mass = mass * wx * wy * wz
                                density_mass += np.bincount(flat, weights=weighted_mass, minlength=cells)
                                for axis in range(3):
                                    momentum[:, axis] += np.bincount(
                                        flat, weights=weighted_mass * vel[:, axis], minlength=cells
                                    )
                    particle_count += len(pos)
                    total_mass += float(np.sum(mass, dtype=np.float64))
        if particle_count != expected_per_type or total_mass <= 0:
            raise ValueError(
                "%s has %d particles; expected %d for the configured N^3 species grid"
                % (names[ptype], particle_count, expected_per_type)
            )
        mean_cell_mass = total_mass / cells
        delta_k = np.fft.rfftn((density_mass / mean_cell_mass - 1.0).reshape(mesh, mesh, mesh))
        momentum_k = np.fft.rfftn(momentum.reshape(mesh, mesh, mesh, 3), axes=(0, 1, 2)) / mean_cell_mass
        delta_k = np.divide(delta_k, window, out=np.zeros_like(delta_k), where=safe_window)
        momentum_k = np.divide(momentum_k, window[..., None], out=np.zeros_like(momentum_k), where=safe_window[..., None])
        velocity_k = momentum_k * math.sqrt(a)
        theta_k = -(1j * np.sum(kvec * velocity_k, axis=-1)) / (a * 100.0 * class_e)
        # The denominator above is a*H/h = a*100*E in km/s per (Mpc/h).
        # CLASS theta uses 1/Mpc, so its dimensionless continuity response is
        # converted separately below using H/c in physical inverse Mpc.
        species[ptype] = {
            "delta": delta_k, "theta": theta_k, "mass": total_mass,
            "particle_count": particle_count,
        }

    k_table = transfer[:, 0]
    k_nyquist = np.pi * mesh / box_mpc_h
    k_fraction = float(settings.get("kmax_fraction_nyquist", 0.25))
    min_modes = int(settings.get("min_modes", 20))
    if not 0 < k_fraction < 0.5 or min_modes < 1:
        raise ValueError("IC velocity-divergence validation requires kmax_fraction_nyquist in (0, 0.5) and min_modes >= 1")
    mode_mask = (kmag > 0) & (kmag <= k_fraction * k_nyquist) & safe_window
    in_transfer = (kmag >= k_table[0]) & (kmag <= k_table[-1])
    mode_mask &= in_transfer
    # CLASS theta_i is the velocity divergence (c=1), so continuity gives
    # theta_code=-theta_CLASS/(a H/c)/delta_CLASS * delta_particle.
    # CLASS backgrounds use c=1 and report H in Mpc^-1.
    class_h_conformal = a * h_start
    expected = {}
    reports = {}
    for ptype in (0, 1):
        delta_i = delta_column[ptype]
        theta_i = theta_column[ptype]
        density_transfer = np.interp(np.log(np.maximum(kmag, k_table[0])), np.log(k_table), transfer[:, delta_i])
        velocity_transfer = np.interp(np.log(np.maximum(kmag, k_table[0])), np.log(k_table), transfer[:, theta_i])
        safe_density = np.abs(density_transfer) > 1e-14 * max(1.0, float(np.max(np.abs(transfer[:, delta_i]))))
        mode_i = mode_mask & safe_density
        ratio = np.zeros_like(kmag, dtype=float)
        ratio[mode_i] = -velocity_transfer[mode_i] / (class_h_conformal * density_transfer[mode_i])
        expected_i = ratio * species[ptype]["delta"]
        expected[ptype] = expected_i
        reports[names[ptype]] = _fourier_response_metrics(
            species[ptype]["theta"], expected_i, mode_i, min_modes,
        )
        reports[names[ptype]]["particle_count"] = species[ptype]["particle_count"]

    relative_measured = species[0]["theta"] - species[1]["theta"]
    relative_expected = expected[0] - expected[1]
    relative = _fourier_response_metrics(relative_measured, relative_expected, mode_mask, min_modes)
    fb = species[0]["mass"] / (species[0]["mass"] + species[1]["mass"])
    total_measured = fb * species[0]["theta"] + (1.0 - fb) * species[1]["theta"]
    total_expected = fb * expected[0] + (1.0 - fb) * expected[1]
    common = _fourier_response_metrics(total_measured, total_expected, mode_mask, min_modes)
    thresholds = {
        "min_coherence": float(settings.get("min_coherence", 0.99)),
        "max_response_error": float(settings.get("max_response_error", 0.03)),
        "max_rms_fractional_residual": float(settings.get("max_rms_fractional_residual", 0.05)),
        "min_modes": min_modes,
        "kmax_fraction_nyquist": k_fraction,
    }
    checks = list(reports.values()) + [relative, common]
    passed = all(
        row["mode_count"] >= min_modes
        and row["response_error"] <= thresholds["max_response_error"]
        and row["coherence"] >= thresholds["min_coherence"]
        and row["rms_fractional_residual"] <= thresholds["max_rms_fractional_residual"]
        for row in checks
    )
    report = {
        "status": "PASS" if passed else "FAIL",
        "method": "per-species CIC density/momentum; CIC-window deconvolution; CLASS theta_i(k) response",
        "model": model,
        "start_scale_factor": a,
        "start_redshift": z_start,
        "class_H_km_s_Mpc": h_start * 299792.458,
        "validation_mesh": mesh,
        "max_wavenumber_h_mpc": float(k_fraction * k_nyquist),
        "species": reports,
        "relative_baryon_dm_velocity_mode": relative,
        "common_matter_mode": common,
        "particle_mass_fraction_baryon": fb,
        "transfer_table": str(Path(transfer_path).resolve()),
        "class_background": str(Path(background_path).resolve()),
        "velocity_convention": "CLASS theta_i transfer mapped through theta=-div(v_pec)/(aH); GADGET stored u is converted with sqrt(a)",
        "thresholds": thresholds,
    }
    if not passed:
        raise ValueError("External mixed-species IC velocities do not match run-matched CLASS transfers: " + json.dumps(report, sort_keys=True))
    return report


def _particle_velocity_divergence_report(
    paths, cfg, box_size, background_path, species_transfer_path=None,
    thermal_velocity_metadata=None,
):
    """Compare mass-weighted particle velocity divergence with linear CLASS growth.

    This check is independent of particle-ID ordering. Small grids use exact
    particle Fourier sums to avoid CIC aliasing at very early epochs. Larger
    grids use density/momentum CIC with Fourier-window deconvolution.
    """
    if not background_path or not Path(background_path).is_file():
        return {"status": "UNAVAILABLE", "reason": "The run-matched CLASS background table is absent"}
    columns, data = class_background_table(background_path)
    if "H" not in columns or "growth_f" not in columns:
        return {"status": "UNAVAILABLE", "reason": "The CLASS background lacks H or growth_f"}
    z_table = np.asarray(data[:, 0], dtype=np.float64)
    order = np.argsort(z_table)
    z_table = z_table[order]
    h_table = np.asarray(data[order, columns["H"]], dtype=np.float64)
    f_table = np.asarray(data[order, columns["growth_f"]], dtype=np.float64)
    if z_table[0] > 1.0e-8 or not np.all(np.isfinite(h_table)) or np.any(h_table <= 0):
        return {"status": "UNAVAILABLE", "reason": "The CLASS background lacks a finite z=0 expansion rate"}

    with h5py.File(paths[0], "r") as f:
        a = _hdf5_scalar_attr(f["Header"].attrs.get("Time", np.nan), "Time")
    if not np.isfinite(a) or not (0.0 < a <= 1.0):
        raise ValueError("IC velocity-divergence validation found an invalid scale factor")
    z_start = 1.0 / a - 1.0
    if z_start < z_table[0] or z_start > z_table[-1]:
        return {"status": "UNAVAILABLE", "reason": "The CLASS background does not cover the IC start redshift"}
    class_f = float(np.interp(z_start, z_table, f_table))
    h0 = float(np.interp(0.0, z_table, h_table))
    h_start = float(np.interp(z_start, z_table, h_table))
    class_e = h_start / h0
    if str(cfg.get("cosmology", {}).get("model", "")).lower() == "wdm":
        # CLASS computes its background growth helper from baryon+CDM density;
        # in this benchmark the relic is ncdm in CLASS but cold matter in
        # GADGET, so use GADGET's radiation-aware smooth-particle growth rate.
        class_f = smooth_particle_growth_rate(cfg, z_start, background_path)
    if not (np.isfinite(class_f) and class_f > 0 and np.isfinite(class_e) and class_e > 0):
        raise ValueError("CLASS growth rate or expansion rate is invalid for IC velocity validation")
    report = _validate_particle_velocity_divergence(
        paths, cfg, box_size, class_f, class_e,
        species_transfer_path=species_transfer_path,
        thermal_velocity_metadata=thermal_velocity_metadata,
        hubble_rate_at_start=h_start,
    )
    if str(cfg.get("cosmology", {}).get("model", "")).lower() == "wdm":
        if cfg.get("wdm", {}).get("model") == "wdm_thermal_relic":
            report["theory"] = (
                "run-matched CLASS t_ncdm(k)/d_m(k), after removing the sampled thermal component from the measured flow"
                if cfg.get("wdm", {}).get("include_thermal_velocities", True)
                else "run-matched CLASS t_ncdm(k)/d_m(k), with Monte Carlo thermal kicks intentionally omitted"
            )
            report["growth_rate_source"] = "scale-dependent CLASS N-body-gauge relic velocity transfer"
        else:
            report["theory"] = "GADGET smooth matter+radiation particle-growth ODE; transfer-only WDM particles are cold"
            report["growth_rate_source"] = "smooth_particle_growth_rate"
    return report


def _direct_dm_fourier_modes(paths, cfg, wavevectors, box_mpc_h, scale_factor,
                             thermal_velocity_metadata=None):
    """Sum a bounded set of low-k density/momentum modes without grid assignment.

    Used only for grids up to 64^3 particles; chunks bound complex phase-array
    memory. The result has mean-mass normalization and excludes the DC mode.
    Density and momentum are measured independently of lattice IDs.
    """
    density = np.zeros(len(wavevectors), dtype=np.complex128)
    momentum = np.zeros((len(wavevectors), 3), dtype=np.complex128)
    total_mass = 0.0
    thermal = (cfg.get("cosmology", {}).get("model") == "wdm"
               and cfg.get("wdm", {}).get("model") == "wdm_thermal_relic"
               and cfg.get("wdm", {}).get("include_thermal_velocities", True))
    chunk = max(1, min(2048, 1_000_000 // max(1, len(wavevectors))))
    for path in paths:
        with h5py.File(path, "r") as f:
            if "PartType1" not in f:
                continue
            part = f["PartType1"]
            for start in range(0, len(part["Coordinates"]), chunk):
                stop = start + chunk
                pos = np.mod(np.asarray(part["Coordinates"][start:stop], dtype=np.float64) / 1000.0, box_mpc_h)
                vel = np.asarray(part["Velocities"][start:stop], dtype=np.float64)
                mass = (np.asarray(part["Masses"][start:stop], dtype=np.float64)
                        if "Masses" in part else np.full(len(pos), float(f["Header"].attrs["MassTable"][1])))
                if thermal:
                    kicks, _momentum, _beta = wdm_thermal_velocity_from_ids(
                        np.asarray(part["ParticleIDs"][start:stop], dtype=np.uint64), cfg, scale_factor)
                    kicks -= np.asarray(thermal_velocity_metadata["removed_bulk_velocity_gadget_km_s"], dtype=np.float64)
                    vel -= kicks
                phase = np.exp(-1j * (pos @ wavevectors.T))
                density += mass @ phase
                momentum += phase.T @ (mass[:, None] * vel)
                total_mass += float(mass.sum())
    if not np.isfinite(total_mass) or total_mass <= 0:
        raise ValueError("Direct Fourier validation requires positive total DM mass")
    return density / total_mass, momentum / total_mass


def _validate_particle_velocity_divergence(
    paths, cfg, box_size, class_f, class_e, settings=None,
    species_transfer_path=None, thermal_velocity_metadata=None,
    hubble_rate_at_start=None,
):
    """Measure theta=f*delta from the actual DM IC particle positions/velocities."""
    if not paths:
        return {"status": "UNAVAILABLE", "reason": "No IC files were supplied"}
    n = int(cfg["simulation"]["particles_per_dimension"])
    settings = settings or cfg.get("validation", {}).get("ic_velocity_divergence", {})
    max_mesh = max(2, int(settings.get("max_mesh_size", 128)))
    # A mesh at half the particle-grid resolution suppresses lattice and
    # particle-assignment noise while retaining enough low-k modes for a test.
    mesh_limit = min(max_mesh, n // 2)
    mesh = max((candidate for candidate in range(2, mesh_limit + 1) if n % candidate == 0), default=1)
    if mesh < 8:
        return {"status": "UNAVAILABLE", "reason": "The IC particle grid is too small for a resolved velocity-divergence test"}
    if not (np.isfinite(class_f) and class_f > 0 and np.isfinite(class_e) and class_e > 0):
        raise ValueError("CLASS growth rate and expansion rate must be finite and positive")

    box_mpc_h = float(box_size) / 1000.0
    cells = mesh ** 3
    density_mass = np.zeros(cells, dtype=np.float64)
    momentum = np.zeros((cells, 3), dtype=np.float64)
    particle_count = 0
    total_mass = 0.0
    scale_factor = None
    for file in paths:
        with h5py.File(file, "r") as f:
            header = f["Header"].attrs
            this_a = float(header["Time"])
            if scale_factor is None:
                scale_factor = this_a
            elif not math.isclose(this_a, scale_factor, rel_tol=1.0e-8, abs_tol=1.0e-10):
                raise ValueError("IC shards disagree on scale factor during velocity-divergence validation")
            if "PartType1" not in f:
                continue
            group = f["PartType1"]
            coords_ds = group["Coordinates"]
            velocities_ds = group["Velocities"]
            ids_ds = group["ParticleIDs"]
            if "Masses" in group:
                masses_ds = group["Masses"]
                constant_mass = None
            else:
                constant_mass = float(header["MassTable"][1])
                masses_ds = None
                if constant_mass <= 0:
                    raise ValueError("DM IC velocity validation cannot find positive particle masses")
            for start in range(0, len(coords_ds), 250_000):
                stop = min(start + 250_000, len(coords_ds))
                pos = np.asarray(coords_ds[start:stop], dtype=np.float64) / 1000.0
                vel = np.asarray(velocities_ds[start:stop], dtype=np.float64)
                thermal_model = (
                    cfg.get("cosmology", {}).get("model") == "wdm"
                    and cfg.get("wdm", {}).get("model") == "wdm_thermal_relic"
                )
                if thermal_model and cfg.get("wdm", {}).get("include_thermal_velocities", True):
                    if not isinstance(thermal_velocity_metadata, dict):
                        raise ValueError("WDM thermal velocity-divergence validation is missing sampling provenance")
                    ids = exact_uint64_ids(ids_ds[start:stop])
                    thermal, _momentum, _beta = wdm_thermal_velocity_from_ids(ids, cfg, this_a)
                    thermal -= np.asarray(thermal_velocity_metadata["removed_bulk_velocity_gadget_km_s"], dtype=np.float64)
                    vel -= thermal
                mass = (
                    np.asarray(masses_ds[start:stop], dtype=np.float64)
                    if masses_ds is not None
                    else np.full(stop - start, constant_mass, dtype=np.float64)
                )
                if (np.any(~np.isfinite(pos)) or np.any(~np.isfinite(vel))
                        or np.any(~np.isfinite(mass)) or np.any(mass <= 0)):
                    raise ValueError("DM IC positions, velocities, and masses must be finite and masses positive")
                grid = np.mod(pos, box_mpc_h) * (mesh / box_mpc_h)
                base = np.floor(grid).astype(np.int64)
                frac = grid - base
                for dx in (0, 1):
                    wx = (1.0 - frac[:, 0]) if dx == 0 else frac[:, 0]
                    ix = (base[:, 0] + dx) % mesh
                    for dy in (0, 1):
                        wy = (1.0 - frac[:, 1]) if dy == 0 else frac[:, 1]
                        iy = (base[:, 1] + dy) % mesh
                        for dz in (0, 1):
                            wz = (1.0 - frac[:, 2]) if dz == 0 else frac[:, 2]
                            iz = (base[:, 2] + dz) % mesh
                            flat = (ix * mesh + iy) * mesh + iz
                            weighted_mass = mass * wx * wy * wz
                            density_mass += np.bincount(flat, weights=weighted_mass, minlength=cells)
                            for axis in range(3):
                                momentum[:, axis] += np.bincount(
                                    flat, weights=weighted_mass * vel[:, axis], minlength=cells
                                )
                particle_count += len(pos)
                total_mass += float(np.sum(mass, dtype=np.float64))

    if particle_count != n ** 3 or not np.isfinite(total_mass) or total_mass <= 0:
        raise ValueError(
            "DM IC velocity validation found %d type-1 particles and %.12g total mass; expected %d particles"
            % (particle_count, total_mass, n ** 3)
        )
    if scale_factor is None:
        raise ValueError("DM IC velocity validation could not read a snapshot scale factor")

    mean_cell_mass = total_mass / cells
    density_k = np.fft.rfftn((density_mass / mean_cell_mass - 1.0).reshape(mesh, mesh, mesh))
    momentum_k = np.fft.rfftn(momentum.reshape(mesh, mesh, mesh, 3), axes=(0, 1, 2)) / mean_cell_mass
    kx = 2.0 * np.pi * np.fft.fftfreq(mesh, d=box_mpc_h / mesh)
    kz = 2.0 * np.pi * np.fft.rfftfreq(mesh, d=box_mpc_h / mesh)
    kvec = np.stack(np.meshgrid(kx, kx, kz, indexing="ij"), axis=-1)
    kmag = np.linalg.norm(kvec, axis=-1)
    mode_x = np.fft.fftfreq(mesh) * mesh
    mode_z = np.fft.rfftfreq(mesh) * mesh
    window = (
        np.sinc(mode_x[:, None, None] / mesh) ** 2
        * np.sinc(mode_x[None, :, None] / mesh) ** 2
        * np.sinc(mode_z[None, None, :] / mesh) ** 2
    )
    safe_window = window > 0.25
    density_k = np.divide(density_k, window, out=np.zeros_like(density_k), where=safe_window)
    momentum_k = np.divide(momentum_k, window[..., None], out=np.zeros_like(momentum_k), where=safe_window[..., None])
    peculiar_velocity_k = momentum_k * math.sqrt(scale_factor)
    theta_k = -(1j * np.sum(kvec * peculiar_velocity_k, axis=-1)) / (scale_factor * 100.0 * class_e)
    k_nyquist = np.pi * mesh / box_mpc_h
    k_fraction = float(settings.get("kmax_fraction_nyquist", 0.25))
    min_modes = int(settings.get("min_modes", 20))
    if not (0.0 < k_fraction < 0.5) or min_modes < 1:
        raise ValueError("IC velocity-divergence validation requires 0 < kmax_fraction_nyquist < 0.5 and min_modes >= 1")
    low_k_mask = (kmag > 0.0) & (kmag <= k_fraction * k_nyquist) & safe_window
    exact_modes = n <= 64
    if exact_modes:
        # CIC knots coincide with the undeformed lattice at tiny displacement.
        # Direct sums resolve the low-k modes without relaxing any tolerance.
        direct_density, direct_momentum = _direct_dm_fourier_modes(
            paths, cfg, kvec[low_k_mask], box_mpc_h, scale_factor, thermal_velocity_metadata)
        density_k[low_k_mask] = direct_density * cells
        theta_k[low_k_mask] = -(1j * np.sum(kvec[low_k_mask] * direct_momentum, axis=-1)) * (
            cells * math.sqrt(scale_factor) / (scale_factor * 100.0 * class_e))
    thermal_model = (
        cfg.get("cosmology", {}).get("model") == "wdm"
        and cfg.get("wdm", {}).get("model") == "wdm_thermal_relic"
    )
    scale_dependent_model = thermal_model
    if scale_dependent_model:
        if not species_transfer_path or not Path(species_transfer_path).is_file():
            raise ValueError("Scale-dependent velocity validation requires the run-matched CLASS species-transfer table")
        if not (hubble_rate_at_start is not None and np.isfinite(hubble_rate_at_start)
                and hubble_rate_at_start > 0):
            raise ValueError("Scale-dependent velocity validation requires CLASS H(a) at the IC epoch")
        verify_numeric_table(species_transfer_path)
        transfer = np.loadtxt(species_transfer_path, comments="#", ndmin=2)
        theta_column = 7 if thermal_model else 5
        if transfer.shape[1] < 8 or np.any(~np.isfinite(transfer[:, [0, 3, theta_column]])):
            raise ValueError("CLASS species-transfer table lacks k, delta_m, or the selected particle velocity")
        in_transfer = (kmag >= transfer[0, 0]) & (kmag <= transfer[-1, 0])
        log_k = np.log(transfer[:, 0])
        query = np.log(np.maximum(kmag, transfer[0, 0]))
        delta_m_transfer = np.interp(query, log_k, transfer[:, 3])
        if thermal_model:
            theta_particle_transfer = np.interp(query, log_k, transfer[:, 7])
        else:
            omega_b = float(cfg["cosmology"]["omega_b"])
            omega_cdm = float(cfg["cosmology"]["omega_cdm"])
            theta_b_transfer = np.interp(query, log_k, transfer[:, 6])
            theta_cdm_transfer = np.interp(query, log_k, transfer[:, 5])
            theta_particle_transfer = (
                omega_b * theta_b_transfer + omega_cdm * theta_cdm_transfer
            ) / (omega_b + omega_cdm)
        floor = 1.0e-14 * max(1.0, float(np.max(np.abs(transfer[:, 3]))))
        valid_transfer = in_transfer & (np.abs(delta_m_transfer) > floor)
        response = np.zeros_like(kmag, dtype=np.float64)
        response[valid_transfer] = -theta_particle_transfer[valid_transfer] / (
            scale_factor * float(hubble_rate_at_start) * delta_m_transfer[valid_transfer]
        )
        expected_theta_k = response * density_k
    else:
        valid_transfer = np.ones_like(kmag, dtype=bool)
        expected_theta_k = float(class_f) * density_k

    mask = low_k_mask & valid_transfer
    metrics = _fourier_response_metrics(theta_k, expected_theta_k, mask, min_modes)
    thresholds = {
        "min_coherence": float(settings.get("min_coherence", 0.99)),
        "max_response_error": float(settings.get("max_response_error", 0.03)),
        "max_rms_fractional_residual": float(settings.get("max_rms_fractional_residual", 0.05)),
    }
    passed = (
        metrics["response_error"] <= thresholds["max_response_error"]
        and metrics["coherence"] >= thresholds["min_coherence"]
        and metrics["rms_fractional_residual"] <= thresholds["max_rms_fractional_residual"]
    )
    report = {
        "status": "PASS" if passed else "FAIL",
        "method": ("exact mass-weighted particle Fourier sums for low-k density/momentum and divergence"
                   if exact_modes else "mass-weighted particle CIC density/momentum, CIC-window deconvolution, and low-k Fourier divergence"),
        "fourier_backend": "direct_particle_sum" if exact_modes else "cic_deconvolved",
        "theory": (
            (
                "run-matched CLASS t_ncdm(k)/d_m(k), with the ID-seeded thermal component removed before the coherent-flow check"
                if cfg.get("wdm", {}).get("include_thermal_velocities", True)
                else "run-matched CLASS t_ncdm(k)/d_m(k), with Monte Carlo thermal kicks intentionally omitted"
            ) if thermal_model else
            "run-matched CLASS growth_f and H(a); theta=-div(v_pec)/(a H) = f delta"
        ),
        "particle_type": 1,
        "particle_count": particle_count,
        "box_size_mpc_h": box_mpc_h,
        "start_scale_factor": float(scale_factor),
        "class_growth_rate": float(class_f),
        "class_expansion_ratio": float(class_e),
        "validation_mesh": mesh,
        "max_wavenumber_h_mpc": float(k_fraction * k_nyquist),
        "common_growing_mode": metrics,
        "thresholds": thresholds | {"min_modes": min_modes, "kmax_fraction_nyquist": k_fraction},
        "caveat": "A low-k linear continuity check; it validates the IC velocity response, not nonlinear accuracy or species-relative transfer modes.",
    }
    if not passed and cfg.get("validation", {}).get("enforce_ic_velocity_divergence", False):
        raise ValueError("DM IC velocity divergence does not match the CLASS growing mode: " + json.dumps(report, sort_keys=True))
    if thermal_model and cfg.get("wdm", {}).get("include_thermal_velocities", True):
        report["thermal_particle_component"] = {
            "status": thermal_velocity_metadata["status"],
            "thermal_component_removed_before_coherent_flow_check": True,
        }
    return report


def validate_ic_hdf5(
    path, cfg, background_path=None, require_velocity_validation=False,
    species_transfer_path=None, model=None, thermal_velocity_metadata=None,
    native_validation_dir=None,
):
    """Check IC species counts, native units, mass closure, positions and model-matched velocity/transfer metadata."""
    paths = initial_condition_files(path)
    if not paths:
        raise FileNotFoundError("No readable HDF5 initial-condition file found for " + str(path))
    if cfg['simulation'].get('ic_engine_hydro')=='native_music':
        if native_validation_dir is None:
            raise ValueError('Native IC validation requires a retained audit destination')
        from zoom import verify_native_handoff
        report=verify_native_handoff(path,cfg['simulation']['native_generation_report'],cfg,
                                     native_validation_dir,memory_gib=memory_limit_gb(cfg))
        counts=report['resource_census']['particles_by_type']
        return {'status':'PASS','physics_validation_scope':report.get('technical_scope','Native mixed-fluid technical handoff; science NOT_ACCEPTED'),
                'files':1,'particles_by_type_all':dict(enumerate(counts)),
                'particles_by_type':{t:counts[t] for t in range(6) if counts[t]},
                'native_handoff_report':str((Path(native_validation_dir)/'handoff.json').resolve()),
                'native_handoff_report_sha256':file_sha256(Path(native_validation_dir)/'handoff.json'),
                'velocity_divergence_theory_validation':report['velocity_audit'],
                'native_structural_audit':report['structural_audit'],'science_acceptance':'NOT_ACCEPTED'}

    n = int(cfg["simulation"]["particles_per_dimension"])
    expected_per_type = n**3
    expected_types = [0, 1] if cfg["simulation"]["physics"] == "hydro" else [1]
    counts = {ptype: 0 for ptype in range(6)}
    mass_totals_internal = {ptype: 0.0 for ptype in expected_types}
    finite_positions = True
    finite_velocities = True
    coordinates_in_box = True
    positive_masses = True
    fields_valid = True
    header_time = None
    header_box = None
    header_totals = None
    header_mass_table = None
    h = float(cfg["cosmology"]["H0"]) / 100.0
    expected_omega_b = float(cfg["cosmology"]["omega_b"]) / h**2
    expected_omega_m = (float(cfg["cosmology"]["omega_b"]) + float(cfg["cosmology"]["omega_cdm"])) / h**2
    expected_box = float(cfg["simulation"]["box_mpc_h"]) * 1000.0
    required_header = ("NumPart_ThisFile", "NumPart_Total", "MassTable", "Time", "Redshift", "BoxSize", "NumFilesPerSnapshot")
    for file in paths:
        with h5py.File(file, "r") as f:
            if "Header" not in f:
                raise ValueError("IC is missing the GADGET Header group: " + str(file))
            attrs = f["Header"].attrs
            missing_header = [key for key in required_header if key not in attrs]
            if missing_header:
                raise ValueError("IC is missing required GADGET header fields %s: %s" % (missing_header, file))
            num_this = np.asarray(attrs["NumPart_ThisFile"])
            from core.data import snapshot_total_counts
            num_total = snapshot_total_counts(attrs)
            mass_table = np.asarray(attrs["MassTable"], dtype=float)
            if (num_this.shape != (6,) or num_total.shape != (6,)
                    or not np.issubdtype(num_this.dtype, np.integer)
                    or not np.issubdtype(num_total.dtype, np.integer)
                    or np.any(num_this < 0) or np.any(num_total < 0)):
                raise ValueError("IC GADGET particle-count header fields must be six non-negative integers")
            if mass_table.shape != (6,) or not np.all(np.isfinite(mass_table)) or np.any(mass_table < 0):
                raise ValueError("IC GADGET MassTable must contain six finite non-negative values")
            num_files_value = np.asarray(attrs["NumFilesPerSnapshot"])
            if num_files_value.shape != () or not np.issubdtype(num_files_value.dtype, np.integer):
                raise ValueError("IC NumFilesPerSnapshot header value must be an integer scalar")
            num_files = int(num_files_value)
            if num_files != len(paths):
                raise ValueError("IC NumFilesPerSnapshot does not match the discovered file count")
            if header_totals is None:
                header_totals = num_total.astype(np.uint64)
                header_mass_table = mass_table
            elif not np.array_equal(num_total, header_totals) or not np.array_equal(mass_table, header_mass_table):
                raise ValueError("IC HDF5 files disagree on total particle counts or MassTable")
            if header_time is None:
                header_time = _hdf5_scalar_attr(attrs.get("Time", np.nan), "Time")
                header_box = _hdf5_scalar_attr(attrs.get("BoxSize", np.nan), "BoxSize")
            this_time = _hdf5_scalar_attr(attrs.get("Time", np.nan), "Time")
            this_box = _hdf5_scalar_attr(attrs.get("BoxSize", np.nan), "BoxSize")
            this_redshift = _hdf5_scalar_attr(attrs.get("Redshift", np.nan), "Redshift")
            if (not np.isfinite(this_time) or this_time <= 0.0 or not math.isclose(this_time, header_time, rel_tol=1.0e-8, abs_tol=1.0e-10)
                    or not np.isfinite(this_box) or not math.isclose(this_box, header_box, rel_tol=1.0e-8)
                    or not np.isfinite(this_redshift) or not math.isclose(this_redshift, 1.0 / this_time - 1.0, rel_tol=1.0e-6, abs_tol=1.0e-6)):
                raise ValueError("IC HDF5 files disagree on scale factor, redshift, or box size")
            if "HubbleParam" in attrs and not math.isclose(_hdf5_scalar_attr(attrs["HubbleParam"], "HubbleParam"), h, rel_tol=1.0e-5):
                raise ValueError("IC HubbleParam does not match the configured cosmology")
            if "Omega0" in attrs and not math.isclose(_hdf5_scalar_attr(attrs["Omega0"], "Omega0"), expected_omega_m, rel_tol=1.0e-4):
                raise ValueError("IC Omega0 does not match the configured baryon+CDM density")
            if cfg["simulation"]["physics"] == "hydro" and "OmegaBaryon" in attrs:
                if not math.isclose(_hdf5_scalar_attr(attrs["OmegaBaryon"], "OmegaBaryon"), expected_omega_b, rel_tol=1.0e-4):
                    raise ValueError("IC OmegaBaryon does not match the configured cosmology")
            actual_file_counts = np.zeros(6, dtype=np.uint64)
            for ptype in range(6):
                group = "PartType%d" % ptype
                if group not in f:
                    continue
                part = f[group]
                if "Coordinates" not in part:
                    if int(num_this[ptype]) != 0:
                        raise ValueError("IC header counts particles in %s but its Coordinates dataset is missing" % group)
                    continue
                coordinates = part["Coordinates"]
                if coordinates.ndim != 2 or coordinates.shape[1] != 3:
                    raise ValueError("IC %s/Coordinates must have shape (N, 3)" % group)
                actual_file_counts[ptype] = len(coordinates)
            if not np.array_equal(actual_file_counts, num_this.astype(np.uint64)):
                raise ValueError("IC NumPart_ThisFile does not match the particle datasets in " + str(file))
            for ptype in expected_types:
                group = "PartType%d" % ptype
                if group not in f:
                    continue
                part = f[group]
                if "Coordinates" not in part or "Velocities" not in part or "ParticleIDs" not in part:
                    fields_valid = False
                    continue
                coords = part["Coordinates"]
                velocities = part["Velocities"]
                ids = part["ParticleIDs"]
                if (coords.ndim != 2 or coords.shape[1] != 3 or velocities.shape != coords.shape
                        or ids.ndim != 1 or len(ids) != len(coords)
                        or not np.issubdtype(ids.dtype, np.integer)):
                    fields_valid = False
                    continue
                counts[ptype] += len(coords)
                if ptype >= len(mass_table) or mass_table[ptype] == 0:
                    if "Masses" not in part or len(part["Masses"]) != len(coords):
                        fields_valid = False
                    else:
                        masses = part["Masses"]
                        for start in range(0, len(masses), 1_000_000):
                            block = np.asarray(masses[start:min(start + 1_000_000, len(masses))])
                            positive_masses = positive_masses and bool(np.all(np.isfinite(block)) and np.all(block > 0))
                            mass_totals_internal[ptype] += float(np.sum(block, dtype=np.float64))
                elif mass_table[ptype] < 0:
                    positive_masses = False
                else:
                    mass_totals_internal[ptype] += float(mass_table[ptype]) * len(coords)
                for start in range(0, len(coords), 1_000_000):
                    stop = min(start + 1_000_000, len(coords))
                    pos = np.asarray(coords[start:stop])
                    vel = np.asarray(velocities[start:stop])
                    finite_positions = finite_positions and bool(np.all(np.isfinite(pos)))
                    finite_velocities = finite_velocities and bool(np.all(np.isfinite(vel)))
                    coordinates_in_box = coordinates_in_box and bool(np.all(pos >= -1.0e-6) and np.all(pos <= expected_box + 1.0e-6))
            for ptype in range(6):
                counts[ptype] += int(actual_file_counts[ptype]) if ptype not in expected_types else 0
    unexpected_types = {ptype: count for ptype, count in counts.items() if count and ptype not in expected_types}
    if unexpected_types:
        raise ValueError("IC contains populated particle types not enabled by this run: " + str(unexpected_types))
    if header_totals is None or not np.array_equal(np.asarray([counts[i] for i in range(6)], dtype=np.uint64), header_totals):
        raise ValueError("IC NumPart_Total does not match the particle datasets across all files")
    expected_a = 1.0 / (1.0 + float(cfg["simulation"]["start_redshift"]))
    if not np.isfinite(header_time) or abs(header_time - expected_a) > max(1.0e-8, expected_a * 1.0e-6):
        raise ValueError("IC scale factor does not match the requested start redshift")
    if not np.isfinite(header_box) or not math.isclose(header_box, expected_box, rel_tol=1.0e-6):
        raise ValueError("IC BoxSize does not match the configured box in GADGET's kpc/h units")
    bad_counts = {ptype: counts[ptype] for ptype in expected_types if counts[ptype] != expected_per_type}
    if bad_counts:
        raise ValueError("IC particle counts do not match N^3 for required types: " + str(bad_counts))
    rho_crit = RHO_CRIT_MSUN_H_MPC_H3
    volume = float(cfg["simulation"]["box_mpc_h"])**3
    if cfg["simulation"]["physics"] == "hydro":
        expected_mass_totals = {
            0: float(cfg["cosmology"]["omega_b"]) / h**2 * rho_crit * volume / 1.0e10,
            1: float(cfg["cosmology"]["omega_cdm"]) / h**2 * rho_crit * volume / 1.0e10,
        }
    else:
        expected_mass_totals = {
            1: (float(cfg["cosmology"]["omega_b"]) + float(cfg["cosmology"]["omega_cdm"])) / h**2 * rho_crit * volume / 1.0e10,
        }
    bad_masses = {
        ptype: (mass_totals_internal[ptype], expected)
        for ptype, expected in expected_mass_totals.items()
        if not math.isclose(mass_totals_internal[ptype], expected, rel_tol=1.0e-4, abs_tol=1.0e-8)
    }
    if bad_masses:
        raise ValueError("IC total particle masses do not match the configured cosmology: " + str(bad_masses))
    if not finite_positions:
        raise ValueError("IC positions contain a non-finite value")
    if not finite_velocities:
        raise ValueError("IC velocities are missing or contain a non-finite value")
    if not coordinates_in_box:
        raise ValueError("IC coordinates fall outside the configured periodic box")
    if not positive_masses or not fields_valid:
        raise ValueError("IC particle fields, IDs, or masses are missing or invalid")
    monofonic_stats = _monofonic_lagrangian_statistics(paths, cfg, expected_box)
    model = model or cfg.get("cosmology", {}).get("model", "lcdm")
    if cfg["simulation"]["physics"] == "hydro" and species_transfer_path is not None:
        velocity_divergence = _external_hydro_velocity_divergence_report(
            paths, cfg, expected_box, background_path, species_transfer_path, model,
            thermal_velocity_metadata=thermal_velocity_metadata,
        )
    elif cfg['simulation']['physics'] == 'hydro' and model in {'wdm'}:
        velocity_divergence = _external_hydro_velocity_divergence_report(
            paths, cfg, expected_box, background_path, species_transfer_path, model,
            thermal_velocity_metadata=thermal_velocity_metadata,
        )
    elif cfg["simulation"]["physics"] == "hydro":
        velocity_divergence = _monofonic_velocity_divergence_report(
            paths, cfg, expected_box, background_path=background_path
        )
    else:
        velocity_divergence = _particle_velocity_divergence_report(
            paths, cfg, expected_box, background_path,
            species_transfer_path=species_transfer_path,
            thermal_velocity_metadata=thermal_velocity_metadata,
        )
    if require_velocity_validation and velocity_divergence.get("status") != "PASS":
        raise ValueError("This run requires a passing CLASS initial-velocity check: " + str(velocity_divergence))
    return {
        "status": "PASS",
        "files": len(paths),
        "particles_by_type": {ptype: counts[ptype] for ptype in expected_types},
        "particles_by_type_all": counts,
        "expected_per_type": expected_per_type,
        "finite_positions": True,
        "finite_velocities": True,
        "coordinates_within_box": True,
        "particle_fields_and_masses_valid": True,
        "total_mass_by_type_internal_1e10_msun_h": mass_totals_internal,
        "scale_factor": header_time,
        "box_size_kpc_h": header_box,
        "lagrangian_displacement_and_velocity_statistics": monofonic_stats,
        "velocity_divergence_theory_validation": velocity_divergence,
    }




def initial_condition_files(path):
    """Discover the complete numerically ordered HDF5 initial-condition shard set."""
    source = Path(path).expanduser().resolve()
    from core.data import assert_ic_publication_complete
    assert_ic_publication_complete([source/'placeholder' if source.is_dir() else source])
    if source.is_file():
        with h5py.File(source, "r") as f:
            if "Header" not in f:
                nfiles_value = np.asarray(1, dtype=np.int64)
            else:
                nfiles_value = np.asarray(f["Header"].attrs.get("NumFilesPerSnapshot", 1))
            if nfiles_value.size != 1:
                raise ValueError("IC NumFilesPerSnapshot header must contain exactly one value")
            nfiles_scalar = nfiles_value.reshape(-1)[0]
            if not np.issubdtype(np.asarray(nfiles_scalar).dtype, np.integer):
                raise ValueError("IC NumFilesPerSnapshot header must be an integer")
            nfiles = int(nfiles_scalar)
            if nfiles < 1:
                raise ValueError("IC NumFilesPerSnapshot header must be positive")
        if nfiles <= 1:
            return [source]
        suffix = ".hdf5"
        stem = source.name[:-len(suffix)] if source.name.endswith(suffix) else source.name
        if stem.endswith(".0"):
            stem = stem[:-2]
        prefix = source.with_name(stem)
        paths = [Path(str(prefix) + ".%d.hdf5" % index) for index in range(nfiles)]
        missing = [str(item) for item in paths if not item.is_file()]
        if missing:
            raise FileNotFoundError("Multipart IC snapshot is incomplete; missing: " + ", ".join(missing))
        return paths
    if source.is_dir():
        return sorted(source.glob("*.hdf5"))
    single = Path(str(source) + ".hdf5")
    if single.is_file():
        return initial_condition_files(single)
    first_piece = Path(str(source) + ".0.hdf5")
    if first_piece.is_file():
        return initial_condition_files(first_piece)
    return sorted(source.parent.glob(source.name + "*.hdf5"))


def load_wdm_thermal_ic_sidecar(path, cfg):
    """Load campaign-generated WDM thermal sampling provenance for a supplied IC."""
    source = Path(path).expanduser().resolve()
    sidecar = Path(str(source) + ".wdm_thermal_velocity.json")
    if not sidecar.is_file():
        raise ValueError(
            "wdm_thermal_relic external ICs require the matching .wdm_thermal_velocity.json sampling sidecar"
        )
    record = json.loads(sidecar.read_text(encoding="utf-8"))
    if record.get("schema") != "darkmattersim-wdm-thermal-ic-v1" or record.get("status") != "PASS":
        raise ValueError("WDM thermal IC sidecar has an unsupported schema or failed sampling status")
    files = initial_condition_files(source)
    if not files:
        raise FileNotFoundError("WDM thermal IC sidecar was provided without readable HDF5 IC files")
    expected_files = [
        {"path": str(item.resolve()), "sha256": file_sha256(item)}
        for item in files
    ]
    if record.get("initial_condition_files") != expected_files:
        raise ValueError("WDM thermal IC contents do not match the hashes recorded in their sampling sidecar")
    sampling = record.get("sampling")
    if not isinstance(sampling, dict) or sampling.get("status") != "PASS" or sampling.get("model") != "wdm_thermal_relic":
        raise ValueError("WDM thermal IC sidecar is missing a passing phase-space sampling report")
    if int(sampling.get("random_seed", -1)) != int(cfg["simulation"].get("random_seed", 123456)):
        raise ValueError("WDM thermal IC sidecar uses a different configured random seed")
    if not math.isclose(
        float(sampling.get("class_ncdm_mass_keV", float("nan"))),
        float(cfg["wdm"]["thermal_relic_mass_keV"]), rel_tol=0.0, abs_tol=1.0e-12,
    ):
        raise ValueError("WDM thermal IC sidecar uses a different relic mass")
    expected_temperature_ratio = float(wdm_parameters(cfg)["T_ncdm"])
    if not math.isclose(
        float(sampling.get("class_ncdm_temperature_ratio", float("nan"))),
        expected_temperature_ratio, rel_tol=1.0e-10, abs_tol=1.0e-14,
    ):
        raise ValueError("WDM thermal IC sidecar uses a different CLASS relic temperature")
    with h5py.File(files[0], "r") as stream:
        scale_factor = float(stream["Header"].attrs["Time"])
    if not math.isclose(
        float(sampling.get("scale_factor", float("nan"))),
        scale_factor, rel_tol=1.0e-8, abs_tol=1.0e-10,
    ):
        raise ValueError("WDM thermal IC sidecar scale factor does not match the supplied IC")
    return sampling


def validate_gadget_ic_id_uniqueness_log(log_path):
    """Require GADGET's global particle-ID uniqueness check to complete."""
    path = Path(log_path)
    if not path.is_file():
        raise FileNotFoundError("GADGET IC validation log is missing: " + str(path))
    log = path.read_text(encoding="utf-8", errors="replace")
    if "non-unique ID=" in log:
        raise ValueError("GADGET found duplicate particle IDs in the initial conditions")
    marker = "INIT: Testing ID uniqueness..."
    success = "INIT: success."
    marker_at = log.find(marker)
    success_at = log.find(success, marker_at + len(marker)) if marker_at >= 0 else -1
    if marker_at < 0 or success_at < 0:
        raise RuntimeError("GADGET IC validation did not record a completed global particle-ID uniqueness check")
    return {
        "status": "PASS",
        "method": "GADGET-4 global parallel particle-ID sort and duplicate check",
        "log": str(path.resolve()),
    }


def measure_and_validate_ic_power_spectrum(cfg, run_dir, ic_path, target_pk_path=None):
    """Measure IC P(k) with GADGET itself before allowing time evolution."""
    from analysis import read_gadget_power_spectrum_segments, validate_ic_power_spectrum, validate_ic_species_power_spectra

    run_dir = Path(run_dir).resolve()
    sources = initial_condition_files(ic_path)
    if not sources:
        raise FileNotFoundError("No HDF5 files were found for the IC power-spectrum check")
    for source in sources:
        with h5py.File(source,'r') as initial:
            header=initial['Header'].attrs
            if header.get('Research_twofluid_flow_schema')==1 and header.get('Research_twofluid_projection_mode')=='finite_volume_material_cells':
                if cfg['simulation'].get('ic_engine_hydro')!='native_music' or len(sources)!=1:raise ValueError('Material-window launch gate requires the exact single-file native hydro handoff')
                from initial_conditions import native_twofluid_window_gate
                return native_twofluid_window_gate(cfg,run_dir,ic_path)
            if header.get('Music_levelmax',0)>header.get('Music_levelmin',0):
                if cfg.get('simulation',{}).get('ic_engine_hydro')!='native_music' or len(sources)!=1:
                    raise RuntimeError('Nested MUSIC requires a window-aware species power gate on the exact single-file native hydro handoff')
                from core.music.flow import native_original_flow_window_gate
                return native_original_flow_window_gate(cfg,run_dir,ic_path)
    binary = build_gadget(cfg, run_dir)
    measurement_dir = run_dir / "data/ic_validation"
    measurement_dir.mkdir(parents=True, exist_ok=True)
    links = []
    if len(sources) == 1:
        destinations = [measurement_dir / "validation_ic_000.hdf5"]
    else:
        snapshot_dir = measurement_dir / "snapdir_000"
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        destinations = [snapshot_dir / ("validation_ic_000.%d.hdf5" % i) for i in range(len(sources))]
    try:
        for source, destination in zip(sources, destinations):
            if destination.exists() or destination.is_symlink():
                raise FileExistsError("IC validation snapshot path already exists: " + str(destination))
            destination.symlink_to(source)
            links.append(destination)
        param = run_dir / "input/ic_power_spectrum.param"
        atomic_text(param,
            gadget_param_text(
                cfg,
                run_dir,
                str(ic_path),
                snapshot_base="validation_ic",
                output_dir=measurement_dir,
                num_files_per_snapshot=len(sources),
            ),
        )
        env = native_physics_environment(cfg)
        env["OMP_NUM_THREADS"] = str(omp_threads(cfg))
        log = run_dir / "input/ic_power_spectrum.log"
        with open(log, "w", encoding="utf-8") as out:
            subprocess.run(
                ["mpirun", "-np", str(mpi_ranks(cfg)), str(binary), str(param), "4", "0"],
                cwd=run_dir / "input",
                env=env,
                stdout=out,
                stderr=subprocess.STDOUT,
                check=True, pass_fds=native_run_lock_fds(run_dir),
            )
    finally:
        for link in links:
            if link.is_symlink():
                link.unlink()
    id_validation = validate_gadget_ic_id_uniqueness_log(log)
    spectrum_path = measurement_dir / "powerspecs/powerspec_000.txt"
    if not spectrum_path.is_file():
        raise RuntimeError("GADGET-4 did not write the IC power spectrum; see " + str(run_dir / "input/ic_power_spectrum.log"))
    segments = read_gadget_power_spectrum_segments(spectrum_path)
    if not target_pk_path:
        first = segments[0]
        return {
            "status": "MEASURED_NO_TARGET",
            "estimator": "GADGET-4 restartflag 4; full mass-weighted particles; unfolded modes",
            "measured_spectrum": str(spectrum_path),
            "scale_factor": first["time"],
            "box_size_kpc_h": first["box_size_internal"],
            "pm_grid": first["mesh"],
            "particle_ids_unique": True,
            "particle_id_validation": id_validation,
        }
    result = validate_ic_power_spectrum(spectrum_path, target_pk_path, cfg)
    result["scale_factor"] = segments[0]["time"]
    result["box_size_kpc_h"] = segments[0]["box_size_internal"]
    result["pm_grid"] = segments[0]["mesh"]
    result["particle_ids_unique"] = True
    result["particle_id_validation"] = id_validation
    if cfg["simulation"]["physics"] == "hydro":
        model = cfg["cosmology"].get("model", "lcdm")
        species_transfer_file = run_dir / "input/species_transfer_validation.txt"
        generator_transfer_file = run_dir / "input/monofonic_transfer_table.txt"
        transfer_file = (
            species_transfer_file if species_transfer_file.is_file()
            else generator_transfer_file if generator_transfer_file.is_file()
            else run_dir / "input/monofonic_input_transfer.txt"
        )
        if not transfer_file.is_file():
            raise FileNotFoundError(
                "Hydro IC validation needs the run-matched species transfer table; supply matching transfer data before evolving external ICs"
            )
        species_result = validate_ic_species_power_spectra(
            measurement_dir / "powerspecs",
            target_pk_path,
            transfer_file,
            cfg,
            dm_species="wdm" if model == "wdm" else "cdm",
        )
        result["species_transfer_spectra"] = species_result
        if species_result["status"] != "PASS":
            result["status"] = "FAIL"
    return result


def collect_gadget_snapshot_schedule(run_dir, cfg):
    """Validate output indices/shards and record target versus synchronized headers."""
    run_dir = Path(run_dir).expanduser().resolve()
    data_dir = run_dir / "data"
    pattern = re.compile(r"(?:snapshot|snap)[_-]?(\d+)(?:\.(\d+))?\.hdf5$", re.IGNORECASE)
    by_snapshot = {}
    if data_dir.is_dir():
        for path in data_dir.rglob("*.hdf5"):
            match = pattern.fullmatch(path.name)
            if match:
                by_snapshot.setdefault(int(match.group(1)), []).append((
                    int(match.group(2)) if match.group(2) is not None else 0, path,
                ))
    requested = snapshot_redshifts(cfg)
    expected_ids = list(range(len(requested)))
    if sorted(by_snapshot) != expected_ids:
        raise RuntimeError(
            "GADGET snapshot IDs %s do not match the requested output indices %s"
            % (sorted(by_snapshot), expected_ids)
        )
    expected_shards = int(cfg["simulation"].get("num_files_per_snapshot", 1))
    from core.cosmology import run_proper_time_clock, interpolate_proper_time
    from core.data import epoch_contract
    time_scale, time_values, clock_source = run_proper_time_clock(run_dir)

    def time_at(scale_factor):
        if time_scale is None:
            return None
        return interpolate_proper_time(time_scale,time_values,scale_factor)

    rows = []
    for snapshot, requested_z in enumerate(requested):
        shards = sorted(by_snapshot[snapshot], key=lambda item: item[0])
        if len(shards) != expected_shards:
            raise RuntimeError(
                "GADGET snapshot %d has %d HDF5 shards; NumFilesPerSnapshot requests %d"
                % (snapshot, len(shards), expected_shards)
            )
        if [index for index, _ in shards] != list(range(expected_shards)):
            raise RuntimeError("GADGET snapshot %d has missing or duplicate HDF5 shard indices" % snapshot)
        actual_redshifts, actual_scales = [], []
        for _, path in shards:
            with h5py.File(path, "r") as snapshot_file:
                if "Header" not in snapshot_file:
                    raise ValueError("GADGET snapshot shard has no Header group: " + str(path))
                header = snapshot_file["Header"].attrs
                if "Redshift" not in header or "Time" not in header:
                    raise ValueError("GADGET snapshot header lacks Redshift or Time: " + str(path))
                actual_redshifts.append(float(np.asarray(header["Redshift"]).reshape(-1)[0]))
                actual_scales.append(float(np.asarray(header["Time"]).reshape(-1)[0]))
                if "NumFilesPerSnapshot" in header:
                    file_count = int(np.asarray(header["NumFilesPerSnapshot"]).reshape(-1)[0])
                    if file_count != expected_shards:
                        raise ValueError("Snapshot header shard count disagrees with configured NumFilesPerSnapshot")
        if not np.isfinite(actual_redshifts).all() or not np.isfinite(actual_scales).all():
            raise ValueError('Snapshot header has invalid cosmological epoch')
        if not np.allclose(actual_redshifts, actual_redshifts[0], rtol=0.0, atol=1.0e-10):
            raise ValueError("Snapshot %d shards disagree on their synchronized redshift" % snapshot)
        if not np.allclose(actual_scales, actual_scales[0], rtol=0.0, atol=1.0e-12):
            raise ValueError("Snapshot %d shards disagree on their synchronized scale factor" % snapshot)
        actual_z = float(actual_redshifts[0])
        actual_a = float(actual_scales[0])
        if (not np.isfinite(actual_z) or not np.isfinite(actual_a) or actual_z < 0 or actual_a <= 0
                or not np.isclose(actual_a, 1.0 / (1.0 + actual_z), rtol=1.e-10, atol=1.e-12)):
            raise ValueError('Snapshot header has invalid or inconsistent cosmological epoch')
        if rows and actual_a <= rows[-1]['actual_scale_factor']:
            raise ValueError('Saved snapshot epochs must increase strictly')
        requested_a = 1.0 / (1.0 + float(requested_z))
        offset_a = actual_a - requested_a
        target_status = "MATCHED" if abs(offset_a) <= max(1.0e-10, 1.0e-7 * requested_a) else "DEVIATED"
        rows.append({
            "snapshot": snapshot,
            "requested_redshift": float(requested_z),
            "actual_redshift": actual_z,
            "redshift_offset_actual_minus_requested": actual_z - float(requested_z),
            "requested_scale_factor": requested_a,
            "actual_scale_factor": actual_a,
            "scale_factor_offset_actual_minus_requested": offset_a,
            "cosmic_time_requested_gyr": time_at(requested_a),
            "cosmic_time_actual_gyr": time_at(actual_a),
            'epoch_contract': epoch_contract(actual_a,requested_scale_factor=requested_a,clock=(time_scale,time_values,clock_source)),
            "synchronization_status": target_status,
            "files": [str(path.resolve()) for _, path in shards],
        })
    return {
        "status": "MATCHED" if all(row["synchronization_status"] == "MATCHED" for row in rows) else "DEVIATED",
        'state_synchronization_policy':'native top-level block-step synchronization',
        'epoch_selection_policy':'first top-level synchronized step at or after the requested time',
        'epoch_precision_scope':'MATCHED/DEVIATED compares saved and requested epochs; it is not a claim of unsynchronized particle state',
        "requested_snapshot_count": len(requested),
        "actual_snapshot_count": len(rows),
        "files_per_snapshot": expected_shards,
        "cosmic_time_source": clock_source,
        "rows": rows,
    }


@native_run_owner
def run_gadget(cfg, run_dir, ic_path, dry_run=False, validated_binary=None):
    """Execute the configured MPI/OpenMP particle integrator and preserve its logs/restart state."""
    run_dir = Path(run_dir)
    if not dry_run:
        from core.native import preflight_native_ic_capacity
        capacity=preflight_native_ic_capacity(initial_condition_files(ic_path),mpi_ranks(cfg))
        update_manifest(run_dir,'native_particle_capacity',capacity)
        from core.data import validate_gadget_native_particle_ids
        validate_gadget_native_particle_ids(initial_condition_files(ic_path))
        coarse_softening = validate_coarse_softening_masses(initial_condition_files(ic_path), cfg["simulation"])
        if coarse_softening is not None:
            update_manifest(run_dir, "zoom_softening", coarse_softening)
    if validated_binary is not None:
        source_binary = Path(validated_binary)
        local_binary = gadget_binary(run_dir)
        atomic_text(run_dir / "input/Config.sh", gadget_config_text(cfg))
        if not dry_run:
            if source_binary.resolve()!=local_binary.resolve():
                atomic_copy(source_binary, local_binary)
            local_binary.chmod(local_binary.stat().st_mode | 0o111)
        binary = local_binary
    else:
        binary = build_gadget(cfg, run_dir, dry_run)
    write_output_list(run_dir / "input/output_times.txt", snapshot_redshifts(cfg))
    param = run_dir / "input/param.txt"
    snapshot_file_count = int(cfg["simulation"].get("num_files_per_snapshot", 1))
    atomic_text(param,
        gadget_param_text(
            cfg, run_dir, ic_path, dry_run=dry_run,
            num_files_per_snapshot=snapshot_file_count,
        ),
    )
    if dry_run:
        return
    from core.foundation import run_namespace_binding
    run_namespace_binding(run_dir, initialize=True)
    checkpoint_clock = stage_native_checkpoint_clock(run_dir)
    config_file = run_dir / "input/Config.sh"
    output_times = run_dir / "input/output_times.txt"
    ic_files = initial_condition_files(ic_path)
    record_manifest(run_dir, "gadget_handoff", {
        "parameter_file": {"path": str(param.resolve()), "sha256": file_sha256(param)},
        "compiled_options_file": {
            "path": str(config_file.resolve()),
            "sha256": file_sha256(config_file) if config_file.is_file() else None,
        },
        "executable": {
            "path": str(Path(binary).resolve()),
            "sha256": file_sha256(binary),
            "size_bytes": Path(binary).stat().st_size,
        },
        "initial_conditions": [
            {"path": str(path.resolve()), "sha256": file_sha256(path), "size_bytes": path.stat().st_size}
            for path in ic_files
        ],
        "initial_condition_prefix_passed_to_gadget": gadget_ic_prefix(ic_path),
        "snapshot_schedule_scale_factors": [
            float(line) for line in output_times.read_text(encoding="utf-8").splitlines() if line.strip()
        ],
        "snapshot_schedule_file": {
            "path": str(output_times.resolve()), "sha256": file_sha256(output_times),
        },
        "snapshot_file_count": int(cfg["simulation"].get("num_files_per_snapshot", 1)),
        "units": {
            "length_cm_per_h": KPC_CM,
            "mass_g_per_h": 1.0e10 * MSUN_G,
            "velocity_cm_per_s": 1.0e5,
            "internal_hubble": 0.1,
        },
        "physical_source_assets":native_physics_asset_bindings(cfg),
        "checkpoint_contract": "gadget-rank-checkpoint-v1",
        "checkpoint_clock": checkpoint_clock,
        "mpi_ranks": mpi_ranks(cfg),
        "openmp_threads_per_rank": omp_threads(cfg),
        "working_directory": str((run_dir / "input").resolve()),
    })
    ranks = mpi_ranks(cfg)
    env = native_physics_environment(cfg)
    env["OMP_NUM_THREADS"] = str(omp_threads(cfg))
    log = run_dir / "input/gadget.log"
    gadget_execution_started = time.monotonic()
    record_manifest(run_dir, "status", "SIMULATION_RUNNING")
    try:
        with open(log, "w", encoding="utf-8") as out:
            subprocess.run(["mpirun", "-np", str(ranks), str(binary), str(param)], cwd=run_dir / "input", env=env, stdout=out, stderr=subprocess.STDOUT, check=True, pass_fds=native_run_lock_fds(run_dir))
    except BaseException as exc:
        record_manifest_fields(run_dir, {
            "status": "SIMULATION_INTERRUPTED" if isinstance(exc, KeyboardInterrupt) else "SIMULATION_FAILED",
            "gadget_run": {"status": "EXECUTION_FAILED", "error": str(exc), "log": str(log.resolve())},
        })
        raise
    finally:
        _append_gadget_execution_measurement(
            run_dir, log, "initial", time.monotonic() - gadget_execution_started,
        )
    used_values = validate_gadget_used_values(param, Path(str(param) + "-usedvalues"))
    record_manifest(run_dir, "gadget_used_values", used_values)
    try:
        snapshot_schedule = collect_gadget_snapshot_schedule(run_dir, cfg)
    except Exception as exc:
        record_manifest_fields(run_dir, {
            "status": "SIMULATION_INCOMPLETE",
            "gadget_run": {"status": "INCOMPLETE_OUTPUTS", "error": str(exc), "log": str(log.resolve())},
        })
        raise
    record_manifest_fields(run_dir, {"snapshot_schedule": snapshot_schedule, "gadget_run": {
        "status": "COMPLETE",
        "log": str(log.resolve()),
        "log_sha256": file_sha256(log),
        "snapshot_schedule_status": snapshot_schedule["status"],
        "snapshot_schedule": snapshot_schedule["rows"],
        "snapshot_files": [
            {"path": str(path.resolve()), "size_bytes": path.stat().st_size, "sha256": file_sha256(path)}
            for path in sorted((run_dir / "data").rglob("snapshot_*.hdf5"))
        ],
    }})


def completed_gadget_run_record(previous, snapshot_schedule, log):
    """Keep failed-attempt history without presenting its error as current state."""
    record = dict(previous)
    if 'error' in record:
        record['previous_failed_attempt'] = {
            'status': record.get('status'), 'error': record.pop('error'),
            'log': record.get('log'),
        }
    record.update(status='COMPLETE', log=str(Path(log).resolve()), log_sha256=file_sha256(log),
                  snapshot_schedule_status=snapshot_schedule['status'],
                  snapshot_schedule=snapshot_schedule['rows'],
                  snapshot_files=[{'path':str(path),'size_bytes':Path(path).stat().st_size,
                                   'sha256':file_sha256(path)}
                                  for row in snapshot_schedule['rows'] for path in row['files']])
    return record


@native_run_owner
def resume_gadget_run(run_dir, *, recover_checkpoint=False):
    """Resume an interrupted GADGET run from its complete rank-local checkpoint set."""
    run_dir = Path(run_dir).expanduser().resolve()
    manifest_path = run_dir / "manifest.json"
    parameter_path = run_dir / "input" / "param.txt"
    if not manifest_path.is_file() or not parameter_path.is_file():
        raise FileNotFoundError("Resume requires an existing run manifest and input/param.txt")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    from core.foundation import run_namespace_binding
    run_namespace_binding(run_dir)  # Validate declared ownership without inventing legacy ancestry.
    if manifest.get("status") == "SIMULATION_COMPLETE" or manifest.get("gadget_run", {}).get("status") == "COMPLETE":
        raise RuntimeError("This run is already complete; refusing to resume or overwrite its outputs")
    cfg_path = run_dir / "config.yaml"
    if not cfg_path.is_file():
        raise FileNotFoundError("The original run config.yaml is required to resume safely")
    recorded_config_sha = manifest.get("config_sha256")
    if recorded_config_sha and file_sha256(cfg_path) != recorded_config_sha:
        raise RuntimeError("Saved run config.yaml changed since the original run; refusing an unsafe resume")
    for entry in [*manifest.get('input_artifacts',{}).values(),
                  *manifest.get('gadget_handoff',{}).get('initial_conditions',[])]:
        if entry.get('path') and entry.get('sha256') and file_sha256(entry['path'])!=entry['sha256']:
            raise RuntimeError('Saved resume input changed: '+entry['path'])
    cfg = load_config(cfg_path)
    require_particle_cosmology(cfg)
    validate_supported_options(cfg)
    handoff = manifest.get("gadget_handoff", {})
    recorded_parameter = handoff.get("parameter_file", {}).get("sha256")
    if recorded_parameter and file_sha256(parameter_path) != recorded_parameter:
        raise RuntimeError("GADGET parameter file changed since the original run; refusing an unsafe resume")
    output_times = run_dir / "input" / "output_times.txt"
    if not output_times.is_file():
        raise FileNotFoundError("The original synchronized-output schedule is missing")
    schedule_record = handoff.get('snapshot_schedule_file')
    if schedule_record and file_sha256(output_times) != schedule_record.get('sha256'):
        raise RuntimeError('Synchronized-output schedule checksum changed; refusing an unsafe resume')
    try:
        current_schedule = [float(line) for line in output_times.read_text(encoding="utf-8").splitlines() if line.strip()]
    except ValueError as exc:
        raise RuntimeError("The saved synchronized-output schedule is malformed") from exc
    recorded_schedule = handoff.get("snapshot_schedule_scale_factors")
    if not isinstance(recorded_schedule, list) or len(recorded_schedule) != len(current_schedule) or not np.allclose(
        np.asarray(recorded_schedule, dtype=float), np.asarray(current_schedule, dtype=float), rtol=0.0, atol=1.0e-12,
    ):
        raise RuntimeError("Synchronized-output schedule changed since the original run; refusing an unsafe resume")
    executable_path = handoff.get("executable", {}).get("path")
    binary = Path(executable_path).expanduser().resolve() if executable_path else gadget_binary(run_dir)
    if not binary.is_file():
        raise FileNotFoundError("The original GADGET executable is missing: " + str(binary))
    recorded_binary_sha = handoff.get("executable", {}).get("sha256")
    if recorded_binary_sha and file_sha256(binary) != recorded_binary_sha:
        raise RuntimeError("GADGET executable changed since the original run; refusing an unsafe resume")

    ranks = int(manifest.get("requested_mpi_ranks", mpi_ranks(cfg)))
    restart_dir = run_dir / "data" / "restartfiles"
    recovery_transaction = restart_dir / 'checkpoint_recovery.json'
    if recovery_transaction.is_file() and json.loads(recovery_transaction.read_text()).get('status')=='PENDING' and not recover_checkpoint:
        raise RuntimeError('Checkpoint recovery is pending; resume with --recover-checkpoint to finish its verified transaction')
    stage_native_checkpoint_clock(run_dir)
    checkpoint_recovery = recover_native_checkpoints(restart_dir, ranks) if recover_checkpoint else None
    restart_files = {int(match.group(1)): path for path in restart_dir.glob("restart.*")
                     if (match := re.fullmatch(r"restart\.(\d+)", path.name))}
    expected_ranks = set(range(ranks))
    if set(restart_files) != expected_ranks or any(not path.is_file() or path.stat().st_size == 0 for path in restart_files.values()):
        raise RuntimeError(
            "Resume needs one non-empty restartfiles/restart.<rank> checkpoint for every original MPI rank; "
            "found ranks %s, expected %s" % (sorted(restart_files), sorted(expected_ranks))
        )

    checkpoint_integrity = verify_native_checkpoints(restart_dir, ranks, require_receipts=handoff.get('checkpoint_contract')=='gadget-rank-checkpoint-v1')
    if checkpoint_integrity.get('a') is not None:
        from core.cosmology import run_proper_time_clock
        from core.data import epoch_contract
        checkpoint_integrity['epoch'] = epoch_contract(checkpoint_integrity['a'], clock=run_proper_time_clock(run_dir))
        checkpoint_integrity['periodic_epochs'] = verify_checkpoint_epochs(restart_dir,checkpoint_integrity,run_dir)
    import datetime
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    attempts = manifest.get("resume_attempts", [])
    if not isinstance(attempts, list):
        attempts = []
    attempts.append({"started_utc": started, "restart_ranks": sorted(restart_files), "checkpoint_integrity": checkpoint_integrity, "checkpoint_recovery": checkpoint_recovery, "status": "RUNNING"})
    manifest["resume_attempts"] = attempts
    manifest["status"] = "SIMULATION_RESUMING"
    _write_json_atomic(manifest_path, manifest)

    env = native_physics_environment(cfg)
    original_threads = int(manifest.get("requested_openmp_threads", omp_threads(cfg)))
    env["OMP_NUM_THREADS"] = str(original_threads)
    log = run_dir / "input" / ("gadget_resume_%03d.log" % len(attempts))
    gadget_execution_started = time.monotonic()
    execution_recorded = False
    try:
        with log.open("w", encoding="utf-8") as output:
            subprocess.run(
                ["mpirun", "-np", str(ranks), str(binary), str(parameter_path), "1"],
                cwd=run_dir / "input", env=env, stdout=output, stderr=subprocess.STDOUT, check=True, pass_fds=native_run_lock_fds(run_dir),
            )
        execution_wall_seconds = time.monotonic() - gadget_execution_started
        execution_record = _gadget_execution_measurement_record(log, "resume", execution_wall_seconds)
        manifest.setdefault("gadget_execution_measurements", []).append(execution_record)
        attempts[-1]["execution_wall_seconds"] = execution_wall_seconds
        attempts[-1]["maximum_logged_internal_allocation_mb"] = execution_record[
            "maximum_logged_internal_allocation_mb"
        ]
        execution_recorded = True
        used_values = validate_gadget_used_values(parameter_path, Path(str(parameter_path) + "-usedvalues"))
        snapshot_schedule = collect_gadget_snapshot_schedule(run_dir, cfg)
        finished = datetime.datetime.now(datetime.timezone.utc).isoformat()
        attempts[-1].update({
            "finished_utc": finished,
            "status": "COMPLETE",
            "log": str(log.resolve()),
            "log_sha256": file_sha256(log),
            "snapshot_schedule_status": snapshot_schedule["status"],
        })
        manifest["resume_attempts"] = attempts
        manifest["gadget_used_values"] = used_values
        manifest["snapshot_schedule"] = snapshot_schedule
        manifest['gadget_run'] = completed_gadget_run_record(
            manifest.get('gadget_run',{}), snapshot_schedule, log)
        manifest["status"] = "SIMULATION_COMPLETE"
        manifest["simulation_finished_utc"] = finished
        _write_json_atomic(manifest_path, manifest)
    except BaseException as exc:
        if not execution_recorded:
            execution_wall_seconds = time.monotonic() - gadget_execution_started
            execution_record = _gadget_execution_measurement_record(log, "resume", execution_wall_seconds)
            manifest.setdefault("gadget_execution_measurements", []).append(execution_record)
            attempts[-1]["execution_wall_seconds"] = execution_wall_seconds
            attempts[-1]["maximum_logged_internal_allocation_mb"] = execution_record[
                "maximum_logged_internal_allocation_mb"
            ]
        attempts[-1].update({"status": "INTERRUPTED", "error": str(exc)})
        manifest["resume_attempts"] = attempts
        manifest["status"] = "SIMULATION_INTERRUPTED"
        _write_json_atomic(manifest_path, manifest)
        raise
    return {"status": "SIMULATION_COMPLETE", "run_dir": str(run_dir), "log": str(log)}


def print_run_summary(cfg, run_dir, model, existing_ic=None, ic_only=False, prepare_only=False, *, ic_census=None):
    """Expose the selected science, output scope and cost before solver work."""
    simulation = cfg["simulation"]
    schedule = snapshot_redshifts(cfg)
    sizing = estimate_run_resources(cfg, len(schedule),
                                    particle_counts=None if ic_census is None else ic_census['particles_by_type'])
    n = int(simulation["particles_per_dimension"])
    physics = simulation["physics"]
    stage = "prepare GADGET" if prepare_only else "IC generation" if ic_only else "GADGET evolution" if existing_ic else "simulation"
    print("\nDarkMatterSim launch summary")
    print("  Model: %s; physics: %s; stage: %s" % (model.upper(), physics, stage))
    if ic_census is None:
        print("  Box: %g Mpc/h; particles: %d^3%s; seed: %s" %
              (simulation["box_mpc_h"], n, " DM + gas" if physics == "hydro" else "", simulation["random_seed"]))
    else:
        print('  Box: %g Mpc/h; delivered species counts: %s' %
              (simulation['box_mpc_h'],ic_census['particles_by_type']))
    print("  Start z: %s; end z: %s; requested snapshots: %d" %
          (simulation["start_redshift"], simulation.get("end_redshift", 0), len(schedule)))
    print("  Rough RAM estimate: %.2f GiB; RAM budget: %.2f GiB; disk estimate: %.2f GiB" %
          (sizing["memory_gib"], memory_limit_gb(cfg), sizing["disk_gib"]))
    print("  Memory use depends on tree/SPH/MPI allocation; estimates are planning aids.")
    if physics == "hydro":
        hydro = cfg.get("hydro", {})
        print("  Gas physics: cooling/UV %s; Springel-Hernquist star formation %s" % (
            "enabled" if hydro.get("cooling", True) else "disabled",
            "enabled" if hydro.get("star_formation", True) else "disabled"))
        if cfg.get("black_holes", {}).get("enabled", False):
            print("  Black-hole physics: native thermal Bondi/Eddington model; seeding=" + cfg["black_holes"]["seeding_mode"] + "; dynamics=" + str(cfg["black_holes"].get("dynamics",{}).get("enabled",False)))
            print("  BH integration: synchronized timesteps; density-entropy SPH; fixed proper-radius gas kernel")
        else:
            print("  Black-hole seeding/accretion/feedback: disabled")
    print("  Halo finders:", ", ".join(name for name in ("fof", "subfind", "rockstar", "consistent_trees") if cfg.get("halos", {}).get(name)) or "none")
    analysis = cfg.get("analysis", {})
    print("  Configured analysis:", "all" if analysis.get("all", False) else analysis.get("products", "selected"))
    visual = cfg.get("visualization", {})
    print("  Configured visualization:", ", ".join(name for name in ("structure_maps", "movie", "blender_export", "interactive_3d") if visual.get(name)) or "none")
    print("  Output folder:", Path(run_dir).expanduser().resolve())
    if existing_ic:
        print("  Existing IC:", Path(existing_ic).expanduser().resolve())
    print()


@native_run_owner
def prepare_and_run(
    cfg, run_dir, model=None, dry_run=False, existing_ic=None,
    evolve_only=False, ic_only=False, prepare_only=False,
):
    """Prepare exact linear/IC inputs and evolve the selected LCDM, WDM or EDE run."""
    require_particle_cosmology(cfg)
    simulation_started = time.monotonic()
    model = model or cfg["cosmology"].get("model", "lcdm")
    run_dir = Path(run_dir).expanduser().resolve()
    if existing_ic:
        existing_ic = str(Path(existing_ic).expanduser().resolve())
    if evolve_only and not existing_ic and not dry_run:
        raise ValueError("--evolve-only requires --existing-ic")
    if ic_only and (existing_ic or evolve_only):
        raise ValueError("--ic-only generates new initial conditions; do not combine it with --existing-ic or --evolve-only")
    if prepare_only and (not existing_ic or dry_run or (
            not evolve_only and cfg['simulation'].get('ic_engine_hydro')!='native_music')):
        raise ValueError('--prepare-only requires an existing IC with evolve-only provenance or an audited native_music handoff')
    cfg = json.loads(json.dumps(cfg))
    cfg["cosmology"]["model"] = model
    cfg["ede"]["enabled"] = model == "ede"
    cfg["wdm"]["enabled"] = model == "wdm"
    validate_supported_options(cfg)
    if cfg['simulation'].get('ic_engine_hydro')=='native_music' and (not existing_ic or evolve_only):
        raise ValueError('native_music requires --existing-ic and the exact modular spectrum source; omit --evolve-only')
    thermal_velocity_metadata = None
    if cfg["simulation"].get("physics") == "hydro":
        validate_hydro_configuration(cfg, require_runtime_assets=not dry_run)
    if model == "wdm":
        validate_wdm_representation(cfg)
        if (cfg["wdm"].get("model") == "wdm_thermal_relic"
                and cfg["wdm"].get("include_thermal_velocities", True) and existing_ic):
            if not dry_run:
                if cfg['simulation'].get('ic_engine_hydro') == 'native_music':
                    from initial_conditions import verify_native_wdm_thermal
                    thermal_velocity_metadata = verify_native_wdm_thermal(existing_ic,cfg)
                else:
                    thermal_velocity_metadata = load_wdm_thermal_ic_sidecar(existing_ic, cfg)
    if model != "wdm" and (int(cfg["cosmology"].get("N_ncdm", 0)) > 0 or float(cfg["cosmology"].get("m_ncdm_eV", 0.0)) > 0):
        raise RuntimeError(
            "This GADGET run evolves baryons and CDM particles only. Massive neutrinos are present in the CLASS parameters "
            "but have no separate particle or validated linear-response evolution here. Set N_ncdm=0 and m_ncdm_eV=0, "
            "or use a solver workflow that explicitly evolves the neutrino component."
        )
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(
            "Run folder already contains files; choose a new run name or move the old results first: " + str(run_dir)
        )
    ic_census = initial_condition_census(initial_condition_files(existing_ic)) if existing_ic and not dry_run else None
    if ic_census is not None:
        from core.native import validate_native_particle_capacity
        validate_native_particle_capacity(ic_census,mpi_ranks(cfg))
    elif not existing_ic:
        from core.native import validate_native_particle_capacity
        n=cfg['simulation']['particles_per_dimension']
        if type(n) is not int or n<1:raise ValueError('Exact positive uniform IC lattice side required')
        uniform={'status':'PLANNED_UNIFORM_IC','particles_by_type':[n**3 if cfg['simulation']['physics']=='hydro' else 0,n**3,0,0,0,0],'files':[]}
        validate_native_particle_capacity(uniform,mpi_ranks(cfg))
    print_run_summary(cfg, run_dir, model, existing_ic, ic_only, prepare_only, ic_census=ic_census)
    storage_preflight = preflight_storage(
        cfg,
        run_dir,
        snapshot_count=len(snapshot_redshifts(cfg)),
        require_confirmation=not dry_run,
        particle_counts=None if ic_census is None else ic_census['particles_by_type'],
    )
    run_dir = ensure_run_layout(run_dir)
    # If a validated external HDF5 IC is supplied, its header is authoritative
    # for the simulation start time. Never evolve it from a mismatched TimeBegin.
    if existing_ic and not dry_run:
        z_ic = ic_header_redshift(existing_ic)
        if z_ic is None:
            raise ValueError("Could not read a valid Header/Time from the supplied Gadget HDF5 IC")
        if z_ic is not None:
            requested = cfg["simulation"].get("start_redshift", "auto")
            if requested != "auto" and abs(float(requested) - z_ic) > 1.0e-6:
                raise RuntimeError("Configured start redshift does not match existing IC header: %.12g vs %.12g" % (float(requested), z_ic))
            cfg["simulation"]["start_redshift"] = float(z_ic)

    source_run = None
    if evolve_only and not dry_run:
        start_z = ic_header_redshift(existing_ic)
        if start_z is None:
            raise ValueError("--evolve-only needs a Gadget HDF5 IC with a valid Header/Time")
        try:
            source_run = existing_ic_source_run(existing_ic, model)
        except FileNotFoundError:
            cosmo = external_lcdm_ic_metadata(existing_ic, cfg)
        else:
            source_run, cosmo = load_existing_ic_cosmology(existing_ic, model)
            validate_evolve_only_source_configuration(source_run, cfg, model)
        source_redshift = cosmo.get("start_redshift", cosmo.get("output_redshift"))
        if source_redshift is not None and not math.isclose(
            float(source_redshift), float(start_z), rel_tol=0.0, abs_tol=1.0e-6 * (1.0 + start_z),
        ):
            raise ValueError("The IC scale factor does not match the source run's saved linear-theory redshift")
    else:
        start_z, cosmo = select_start_redshift(cfg, run_dir, model, dry_run)
    cfg["simulation"]["start_redshift"] = float(start_z)
    cfg["cosmology"]["omega_radiation"] = float(cosmo.get("omega_radiation0", omega_radiation_from_config(cfg)))
    if model == "ede":
        background_table = cosmo.get("gadget_background_file")
        if not background_table or not Path(background_table).is_file():
            raise RuntimeError("EDE particle runs require the run-matched AxiCLASS Gadget background table")
        cfg["cosmology"]["gadget_background_table"] = str(Path(background_table).resolve())
    atomic_text(run_dir / "config.yaml", yaml.safe_dump(cfg, sort_keys=False))
    write_manifest(run_dir, cfg, model, start_z, cosmo)
    record_manifest(run_dir, "storage_preflight", storage_preflight)
    if ic_census is not None:
        record_manifest(run_dir, 'initial_condition_resource_census', ic_census)
    write_output_list(run_dir / "input/output_times.txt", snapshot_redshifts(cfg))


    if model == "wdm" and not dry_run and not evolve_only:
        wdm_pk = cosmo.get("pk_file")
        if not wdm_pk:
            raise RuntimeError("WDM resolution validation needs the actual WDM P(k) used for the initial conditions")
        reference_pk = cfg.get("wdm", {}).get("lcdm_reference_pk")
        reference_z = cfg.get("wdm", {}).get("lcdm_reference_redshift")
        observed_reference_z = pk_output_redshift(reference_pk) if reference_pk and Path(reference_pk).is_file() else None
        reference_matches = (
            reference_z is not None
            and observed_reference_z is not None
            and math.isclose(float(reference_z), float(start_z), rel_tol=0.0, abs_tol=1.0e-8)
            and class_redshift_matches(observed_reference_z, start_z)
        )
        if reference_pk and Path(reference_pk).is_file() and reference_matches:
            lcdm_meta = {"pk_file": str(Path(reference_pk).expanduser().resolve())}
        else:
            lcdm_meta = run_boltzmann(cfg, run_dir, "lcdm", start_z, False)
        try:
            cosmo, lcdm_meta, report = match_wdm_cutoff_coverage(
                cfg, run_dir, start_z, cosmo, lcdm_meta,
            )
        except (OSError, ValueError, RuntimeError) as exc:
            raise RuntimeError("Could not measure the WDM half-mode cutoff from paired CLASS spectra: " + str(exc)) from exc
        wdm_pk = cosmo["pk_file"]
        lcdm_pk = Path(lcdm_meta["pk_file"]).expanduser().resolve()
        cosmo["wdm_lcdm_reference_spectrum"] = {
            "path": str(lcdm_pk),
            "sha256": file_sha256(lcdm_pk),
            "redshift": pk_output_redshift(lcdm_pk),
        }
        artifacts = cosmology_input_artifacts(cosmo)
        record_manifest(run_dir, "input_artifacts", artifacts)
        solver_runs = collect_boltzmann_solver_runs(run_dir)
        record_manifest(run_dir, "boltzmann_solver_runs", solver_runs)
        record_manifest(
            run_dir,
            "boltzmann_solver_status",
            "RECORDED" if solver_runs and all(item.get("status") == "RECORDED" for item in solver_runs) else "PARTIAL",
        )
        cfg.setdefault("wdm", {})["lcdm_reference_pk"] = str(lcdm_pk)
        cfg["wdm"]["lcdm_reference_redshift"] = float(start_z)
        try:
            resolution_gate = validate_wdm_resolution_gate(report, cfg)
        except ValueError as exc:
            resolution_gate = {"status": "BLOCKED", "reason": str(exc)}
        report["evolution_gate"] = resolution_gate
        report["start_redshift"] = float(start_z)
        report["lcdm_control_pk"] = str(lcdm_pk)
        report["wdm_pk"] = str(wdm_pk)
        report_path = run_dir / "input/wdm_resolution_validation.json"
        _write_json_atomic(report_path, report)
        atomic_text(run_dir / "config.yaml", yaml.safe_dump(cfg, sort_keys=False))
        record_manifest(run_dir, "config_sha256", file_sha256(run_dir / "config.yaml"))
        update_manifest(run_dir, "wdm_resolution", report)
        update_manifest(run_dir, "wdm_resolution_gate", resolution_gate)
        if report["resolution_warning"]:
            print("WDM RESOLUTION WARNING:", report["resolution_warning"])
            print("  Half-mode mass [Msun/h]: %.6g; empirical spurious mass [Msun/h]: %.6g" % (
                report["half_mode_mass_msun_h"], report["wang_white_spurious_mass_msun_h"]
            ))
            print("  Particles per half-mode mass: %.3g; suggested N per dimension for the configured reference: %d" % (
                report["particles_per_half_mode_mass"], report["suggested_particles_per_dimension_for_reference"]
            ))
        if resolution_gate["status"] == "BLOCKED":
            record_manifest(run_dir, "status", "STOPPED_WDM_UNDERRESOLVED")
            raise RuntimeError(resolution_gate["reason"])

    # Species-specific CLASS transfers define the baryon/CDM relative flow.
    species_transfer_path = None
    ngenic_velocity_transfer_path = None
    need_species_transfers = (
        cfg["simulation"]["physics"] == "hydro"
        or (model == "wdm" and cfg["wdm"].get("model") == "wdm_thermal_relic")
    )
    if not evolve_only and not dry_run and need_species_transfers:
        transfer_source = cosmo.get("transfer_file")
        if not transfer_source:
            raise RuntimeError(
                "%s IC velocity validation needs this run's CLASS transfer functions; "
                "no transfer table was produced at the IC start redshift." % model.upper()
            )
        species_transfer_path = run_dir / "input/species_transfer_validation.txt"
        species_transfer = write_species_transfer_validation_table(
            transfer_source,
            species_transfer_path,
            cfg,
            model,
            start_z,
        )
        update_manifest(run_dir, "species_transfer_validation", species_transfer)
        if (model == "wdm" and cfg["wdm"].get("model") == "wdm_thermal_relic"
                and cfg["simulation"]["physics"] == "dm"):
            baseline_f = smooth_particle_growth_rate(cfg, start_z, cosmo.get("background_file"))
            ngenic_velocity_transfer_path = run_dir / "input/wdm_velocity_transfer.txt"
            velocity_transfer = write_wdm_velocity_transfer_table(
                transfer_source,
                cosmo.get("background_file"),
                ngenic_velocity_transfer_path,
                start_z,
                baseline_f,
            )
            update_manifest(run_dir, "wdm_linear_velocity_transfer", velocity_transfer)
    elif evolve_only and model in {'wdm'}:
        source_transfer = source_run / "input" / "species_transfer_validation.txt"
        if not source_transfer.is_file():
            raise FileNotFoundError(
                "--evolve-only requires the source run's validated species transfer table: " + str(source_transfer)
            )
        transfer_provenance = validate_species_transfer_provenance(
            source_transfer, cosmo, model, start_z,
        )
        species_transfer_path = run_dir / "input" / "species_transfer_validation.txt"
        atomic_copy(source_transfer, species_transfer_path)
        source_sidecar = Path(str(source_transfer) + ".json")
        if source_sidecar.is_file():
            atomic_copy(source_sidecar, Path(str(species_transfer_path) + ".json"))
        update_manifest(run_dir, "species_transfer_validation", {
            "status": "REUSED_SOURCE_VALIDATION_TABLE",
            "source": str(source_transfer.resolve()),
            "staged_copy": str(species_transfer_path.resolve()),
            "sha256": file_sha256(species_transfer_path),
            "source_transfer_sha256": transfer_provenance["source_transfer_sha256"],
            "source_redshift": transfer_provenance["source_redshift"],
        })
    validated_binary = None

    if existing_ic:
        ic = Path(existing_ic)
    elif cfg["simulation"]["physics"] == "hydro":
        if dry_run and model in ['wdm']:
            ic = run_dir / "data/validated_external_ics.hdf5"
        else:
            ic = generate_hydro_ic_monofonic(cfg, run_dir, model, dry_run, cosmo=cosmo)
    else:
        ic = generate_dm_ic_with_gadget(
            cfg, run_dir, model, cosmo, dry_run,
            velocity_transfer_path=ngenic_velocity_transfer_path,
        )

    if (not dry_run and existing_ic is None and model == "wdm"
            and cfg["wdm"].get("model") == "wdm_thermal_relic"
            and cfg["wdm"].get("include_thermal_velocities", True)):
        thermal_velocity_metadata = add_wdm_thermal_velocities(initial_condition_files(ic), cfg)
        update_manifest(run_dir, "wdm_thermal_velocities", thermal_velocity_metadata)

    if not dry_run:
        # A campaign can generate one validated LCDM hydro IC during its
        # all-branches preflight, then reuse it here.  monofonIC's transfer
        # table is needed to validate that IC's relative baryon-CDM velocity
        # mode, so stage the exact source-run file beside this run's CLASS
        # background before the IC checks begin.
        if ic_census is not None and any(file_sha256(row['path']) != row['sha256'] for row in ic_census['files']):
            raise ValueError('Delivered IC changed after its resource census')
        source_transfer = cfg.get("simulation", {}).get("monofonic_transfer_validation_source")
        if (source_transfer and existing_ic and cfg["simulation"]["physics"] == "hydro"
                and model == "lcdm"):
            source_transfer = Path(source_transfer).expanduser().resolve()
            if not source_transfer.is_file():
                raise FileNotFoundError("Configured monofonIC validation transfer table is missing: " + str(source_transfer))
            staged_transfer = run_dir / "input/monofonic_input_transfer.txt"
            atomic_copy(source_transfer, staged_transfer)
            update_manifest(run_dir, "monofonic_transfer_validation_source", {
                "source": str(source_transfer),
                "staged_copy": str(staged_transfer.resolve()),
                "sha256": file_sha256(staged_transfer),
            })
        update_manifest(run_dir, "initial_condition_source", {
            "path": str(Path(ic).expanduser().resolve()),
            "generated_by_run": existing_ic is None,
        })
        if evolve_only:
            update_manifest(run_dir, "evolve_only", {
                "status": "PASS",
                "source_ic": str(Path(ic).expanduser().resolve()),
                "source_run": str(source_run.resolve()) if source_run else None,
                "source_cosmology_metadata": str((source_run / "input" / (model + "_cosmology.json")).resolve()) if source_run else None,
                "source_power_spectrum": cosmo.get("pk_file"),
                "source_background": cosmo.get("background_file"),
                "CLASS_run": False,
            })
        ic_validation = validate_ic_hdf5(
            ic,
            cfg,
            background_path=cosmo.get("background_file"),
            require_velocity_validation=(
                False or cfg["simulation"]["physics"] == "hydro"
                or (model == "wdm" and cfg["wdm"].get("model") == "wdm_thermal_relic")
            ),
            species_transfer_path=species_transfer_path,
            model=model,
            thermal_velocity_metadata=thermal_velocity_metadata,
            native_validation_dir=run_dir/'input/native-ic-handoff',
        )
        if cfg['simulation'].get('ic_engine_hydro')=='native_music':
            native_transfer=run_dir/'input/native-ic-handoff/reference/transfers.txt'
            staged_transfer=run_dir/'input/monofonic_input_transfer.txt'
            atomic_copy(native_transfer,staged_transfer)
            update_manifest(run_dir,'native_ic_handoff',{
                'report':ic_validation['native_handoff_report'],
                'report_sha256':ic_validation['native_handoff_report_sha256'],
                'species_transfer':str(staged_transfer),'species_transfer_sha256':file_sha256(staged_transfer),
                'science_acceptance':'NOT_ACCEPTED'})
        update_manifest(run_dir, "initial_conditions", ic_validation)
        if prepare_only:
            checkpoint_clock = stage_native_checkpoint_clock(run_dir)
            # Write configuration and compile only. IC Fourier measurements and
            # particle evolution are deferred until --run-prepared is requested.
            binary = build_gadget(cfg, run_dir)
            parameter = run_dir / "input/gadget.param"
            atomic_text(parameter, gadget_param_text(
                cfg, run_dir, ic,
            ))
            record_manifest_fields(run_dir, {"prepared_gadget": {
                    "checkpoint_clock": checkpoint_clock,
                "ic_path": str(Path(ic).resolve()),
                "ic_files": [{"path": str(file.resolve()), "sha256": file_sha256(file)} for file in initial_condition_files(ic)],
                "parameter_path": str(parameter), "parameter_sha256": file_sha256(parameter),
                "binary_path": str(binary), "binary_sha256": file_sha256(binary),
                "output_schedule_path": str(run_dir / "input/output_times.txt"),
                "output_schedule_sha256": file_sha256(run_dir / "input/output_times.txt"),
            }, "status": "GADGET_PREPARED"})
            return str(run_dir)
        target_pk = None if evolve_only else cosmo.get("pk_file")
        if not evolve_only and not target_pk:
            raise RuntimeError("No target linear spectrum is available to validate the generated initial conditions")
        try:
            spectrum_validation = measure_and_validate_ic_power_spectrum(cfg, run_dir, ic, target_pk)
        except Exception as exc:
            update_manifest(run_dir, "initial_ic_power_spectrum", {"status": "FAIL", "error": str(exc)})
            raise RuntimeError("Initial-condition power-spectrum validation failed before GADGET evolution: " + str(exc)) from exc
        update_manifest(run_dir, "initial_ic_power_spectrum", spectrum_validation)
        ic_validation["particle_ids_unique"] = spectrum_validation["particle_ids_unique"]
        ic_validation["particle_id_validation"] = spectrum_validation["particle_id_validation"]
        update_manifest(run_dir, "initial_conditions", ic_validation)
        if spectrum_validation["status"] == "FAIL" and cfg.get("validation", {}).get("enforce_ic_power_spectrum", False):
            raise RuntimeError(
                "Initial-condition P(k) differs from the requested linear spectrum; GADGET evolution was stopped. "
                + json.dumps(spectrum_validation, sort_keys=True)
            )
    if ic_only:
        record_manifest(run_dir, "initial_condition_stage", {
            "status": "DRY_RUN" if dry_run else "COMPLETE",
            "path": str(Path(ic).expanduser().resolve()),
            "generated_by_run": True,
            "validated": not dry_run,
            "linear_spectrum": cosmo.get("pk_file"),
        })
        record_manifest(run_dir, "status", "IC_STAGE_COMPLETE" if not dry_run else "IC_STAGE_DRY_RUN")
        return str(run_dir)
    run_gadget(cfg, run_dir, ic, dry_run, validated_binary)
    if dry_run:
        record_manifest(run_dir, "status", "DRY_RUN_PLAN")
    else:
        import datetime
        record_manifest_fields(run_dir, {
            "status": "SIMULATION_COMPLETE",
            "simulation_finished_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "simulation_wall_seconds": time.monotonic() - simulation_started,
        })
    return str(run_dir)


@native_run_owner
def run_saved_finder_campaign(run_dir, request, *, dry_run=False):
    """Run independent finder/tree stages against saved evolution products."""
    from halos import run_requested_native_finders
    from core.native import native_run_lock
    with native_run_lock(run_dir,operation='saved finder campaign'):
        return run_requested_native_finders(run_dir,load_config(Path(run_dir)/'config.yaml'),dry_run=dry_run,**request)


def run_prepared_gadget(run_dir):
    """Execute a reviewed prepared configuration after checking immutable inputs.

    No Boltzmann solver or IC generator is called. As with evolve-only runs,
    the actual IC power is measured and ParticleID uniqueness is checked before
    time integration. No agreement with an unavailable target P(k) is claimed.
    """
    run_dir = Path(run_dir).expanduser().resolve()
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "GADGET_PREPARED":
        raise ValueError("--run-prepared requires a GADGET_PREPARED run; use --resume for an interrupted evolution")
    prepared = manifest["prepared_gadget"]
    if file_sha256(run_dir / "config.yaml") != manifest["config_sha256"]:
        raise ValueError("Prepared config.yaml changed; prepare a new run to review the new settings")
    for path_key, hash_key in (("parameter_path", "parameter_sha256"), ("binary_path", "binary_sha256")):
        if file_sha256(prepared[path_key]) != prepared[hash_key]:
            raise ValueError("Prepared input changed: " + prepared[path_key])
    if prepared.get('output_schedule_path'):
        if file_sha256(prepared['output_schedule_path']) != prepared.get('output_schedule_sha256'):
            raise ValueError('Prepared output schedule changed; prepare a new run')
    for record in prepared["ic_files"]:
        if file_sha256(record["path"]) != record["sha256"]:
            raise ValueError("Prepared IC changed: " + record["path"])
    cfg = load_config(run_dir / "config.yaml")
    require_particle_cosmology(cfg)
    validate_supported_options(cfg)
    from core.native import preflight_native_ic_capacity
    capacity=preflight_native_ic_capacity(initial_condition_files(prepared['ic_path']),mpi_ranks(cfg))
    update_manifest(run_dir,'native_particle_capacity',capacity)
    binary = build_gadget(cfg, run_dir)
    if file_sha256(binary) != prepared["binary_sha256"]:
        raise ValueError("GADGET source/toolchain changed since preparation; prepare a new run")
    expected_parameter = gadget_param_text(cfg, run_dir, prepared["ic_path"],
                                           )
    if Path(prepared["parameter_path"]).read_text(encoding="utf-8") != expected_parameter:
        raise ValueError("Prepared parameter recipe no longer matches the configuration")
    # These tables define the CLASS initial spectrum and expansion history. Validate their original
    # checksums even when the IC and executable themselves have not changed.
    for record in manifest.get("input_artifacts", {}).values():
        if record.get("path") and record.get("sha256") and file_sha256(record["path"]) != record["sha256"]:
            raise ValueError("Prepared cosmology artifact changed: " + record["path"])
    if cfg['simulation'].get('ic_engine_hydro')=='native_music':
        import datetime
        audit_dir=run_dir/'input'/('native-prepared-recheck-'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
        native_validation=validate_ic_hdf5(prepared['ic_path'],cfg,native_validation_dir=audit_dir)
        update_manifest(run_dir,'native_prepared_recheck',native_validation)
        staged_transfer=run_dir/'input/monofonic_input_transfer.txt'
        original=manifest.get('validation',{}).get('native_ic_handoff',{})
        if not original or file_sha256(staged_transfer)!=original.get('species_transfer_sha256'):
            raise ValueError('Prepared native species-transfer reference changed')
    target_pk=None
    if cfg['simulation'].get('ic_engine_hydro')=='native_music':
        source=json.loads(Path(cfg['spectrum']['file']).read_text())
        relative=source['source_files']['start']['pk']
        target_pk=(Path(cfg['spectrum']['file']).parent/relative).resolve()
        if not target_pk.is_file() or file_sha256(target_pk)!=source['checksums_sha256'].get(relative):
            raise ValueError('Prepared native target linear power source changed')
    spectrum = measure_and_validate_ic_power_spectrum(cfg, run_dir, prepared["ic_path"],target_pk)
    if spectrum['status']=='FAIL':
        update_manifest(run_dir,'initial_ic_power_spectrum',spectrum)
        raise ValueError('Prepared native IC power spectrum failed before evolution')
    update_manifest(run_dir, "initial_ic_power_spectrum", spectrum)
    ic_validation = manifest.get("validation", {}).get("initial_conditions", {})
    ic_validation.update(particle_ids_unique=spectrum["particle_ids_unique"],
                         particle_id_validation=spectrum["particle_id_validation"])
    update_manifest(run_dir, "initial_conditions", ic_validation)
    started = time.monotonic()
    try:
        run_gadget(cfg, run_dir, prepared["ic_path"], validated_binary=binary)
    except BaseException as error:
        record_manifest(run_dir,'prepared_launch_failure',{
            'type':type(error).__name__,'message':str(error),
            'wall_seconds':time.monotonic()-started})
        raise
    record_manifest_fields(run_dir, {
        "status": "SIMULATION_COMPLETE",
        "simulation_wall_seconds": time.monotonic() - started,
    })
    return str(run_dir)


def main():
    """Parse command arguments and dispatch this module's documented workflow."""
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(ROOT / "config.yaml"))
    p.add_argument("--run-dir", default=None, help="output folder (prompted with a unique default when omitted)")
    p.add_argument("--model", choices=["lcdm", "wdm", "ede"])
    p.add_argument("--existing-ic")
    p.add_argument("--evolve-only", action="store_true", help="Evolve an existing Gadget HDF5 IC without running CLASS or generating new ICs")
    p.add_argument("--ic-only", action="store_true", help="Run CLASS/import P(k), generate and validate initial conditions, then stop before GADGET evolution")
    p.add_argument("--prepare-only", action="store_true", help="validate/build without executing; requires existing IC plus evolve-only provenance or native_music configuration")
    p.add_argument("--run-prepared", action="store_true", help="execute a previously prepared GADGET configuration; requires --run-dir")
    p.add_argument("--resume", action="store_true", help="Resume an interrupted run from its complete GADGET restart checkpoint set")
    p.add_argument("--recover-checkpoint", action="store_true", help="With --resume, recover the newest coherent checksummed current/backup rank set; preserve failed originals")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    if args.recover_checkpoint and not args.resume:
        p.error("--recover-checkpoint requires --resume")
    if args.prepare_only and (not args.existing_ic or args.dry_run or args.ic_only or args.resume or args.run_prepared):
        p.error('--prepare-only requires --existing-ic and cannot be combined with dry-run/IC-only/resume/run-prepared')
    if args.run_prepared:
        if not args.run_dir or args.existing_ic or args.ic_only or args.evolve_only or args.resume or args.dry_run:
            p.error("--run-prepared requires --run-dir and cannot be combined with IC/resume/dry-run options")
        run_prepared_gadget(args.run_dir)
        return
    if args.ic_only and (args.existing_ic or args.evolve_only or args.resume):
        p.error("--ic-only cannot be combined with --existing-ic, --evolve-only, or --resume")
    if args.resume:
        if not args.run_dir:
            p.error("--resume requires --run-dir pointing to an existing interrupted run")
        print(json.dumps(resume_gadget_run(args.run_dir, recover_checkpoint=args.recover_checkpoint), indent=2))
        return
    if args.ic_only and (args.existing_ic or args.evolve_only):
        p.error("--ic-only cannot be combined with --existing-ic or --evolve-only")
    cfg = load_config(args.config)
    model = args.model or cfg["cosmology"].get("model", "lcdm")
    physics = str(cfg.get("simulation", {}).get("physics", "dm")).lower()
    run_dir = choose_run_directory(args.run_dir)
    prepare_and_run(
        cfg, run_dir, model, args.dry_run, args.existing_ic,
        args.evolve_only, args.ic_only,
        args.prepare_only,
    )



# ========================================================================
# event_cadence: Event windows, native clock quantization and immutable restart branches.
import shutil
from core.foundation import canonical_digest
from core.native import directory_owner

EVENT_KINDS = {'pericenter', 'stripping', 'encounter', 'major_merger', 'bh_pair', 'bh_merger', 'bh_recoil', 'starburst'}


def number(value, label, *, positive=True):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)) or not math.isfinite(float(value)) or (positive and value <= 0):
        raise ValueError(label+' requires a finite '+('positive ' if positive else '')+'number')
    return float(value)


def validate_clock(clock):
    required = {'time_begin', 'time_max', 'maximum_timestep', 'timebins', 'comoving'}
    if not isinstance(clock, dict) or not required <= set(clock) or set(clock) - required - {'output_mode'}:
        raise ValueError('Native clock requires begin/end, maximum timestep, timebins and comoving')
    mode = clock.get('output_mode', 'timestep_grid')
    if mode not in ('timestep_grid', 'exact_synchronized'):
        raise ValueError('Unsupported native output mode')
    begin = number(clock['time_begin'], 'time_begin', positive=bool(clock['comoving']))
    end = number(clock['time_max'], 'time_max')
    if end <= begin or type(clock['comoving']) is not bool or type(clock['timebins']) is not int or clock['timebins'] not in (29, 60):
        raise ValueError('Unsupported native integer clock')
    maximum = number(clock['maximum_timestep'], 'maximum_timestep')
    span = math.log(end/begin) if clock['comoving'] else end-begin
    interval = span/(1 << clock['timebins'])
    ticks = int(maximum/interval)
    if ticks < 1:
        raise ValueError('Maximum timestep is smaller than a native clock tick')
    subdivision = 1 << clock['timebins']
    while subdivision > ticks:
        subdivision >>= 1
    if mode == 'exact_synchronized':
        if ticks < 2:
            raise ValueError('Maximum timestep is smaller than the minimum legal synchronized step')
        subdivision = 2
    return interval, subdivision


def quantize_native_output_time(requested, clock):
    interval, subdivision = validate_clock(clock)
    value = number(requested, 'requested_output', positive=clock['comoving'])
    if not clock['time_begin'] <= value <= clock['time_max']:
        raise ValueError('Requested output lies outside native clock')
    delta = math.log(value/clock['time_begin']) if clock['comoving'] else value-clock['time_begin']
    raw = int(delta/interval)
    if clock.get('output_mode', 'timestep_grid') == 'exact_synchronized':
        tick = (1 << clock['timebins']) if value == clock['time_max'] else (raw // 2) * 2
    else:
        tick = int(raw/float(subdivision)+.5)*subdivision
    planned = clock['time_begin']*math.exp(tick*interval) if clock['comoving'] else clock['time_begin']+tick*interval
    return {'requested_time': value, 'native_integer_tick': tick, 'planned_quantized_time': planned,
            'quantization_error': planned-value, 'actual_output_time': None, 'status': 'PLANNED'}


def validate_user_schedule(epochs):
    if not isinstance(epochs, dict) or set(epochs) != {'z_start', 'z_end', 'backbone_redshifts'}:
        raise ValueError('Choose start, end and backbone redshifts explicitly')
    start, end = (number(epochs[k], k, positive=False) for k in ('z_start', 'z_end'))
    redshifts = epochs['backbone_redshifts']
    if start <= end or end < 0 or not isinstance(redshifts, list) or not 1 <= len(redshifts) <= 1100:
        raise ValueError('Invalid explicit epoch schedule')
    values = [number(x, 'backbone_redshift', positive=False) for x in redshifts]
    if values[0] > start or values[-1] != end or any(z < end for z in values) or any(b >= a for a,b in zip(values, values[1:])):
        raise ValueError('Backbone must decrease uniquely to the chosen end redshift')
    return [1/(1+z) for z in values]


def merge_and_quantize_native_output_times(backbone, windows, clock, *, maximum_outputs=1100):
    validate_clock(clock)
    if type(maximum_outputs) is not int or not 1 <= maximum_outputs <= 1100:
        raise ValueError('maximum_outputs must be 1 to 1100')
    if not isinstance(backbone, (list, tuple)) or not backbone or any(b <= a for a,b in zip(backbone, backbone[1:])):
        raise ValueError('Backbone native times must increase uniquely')
    times = list(backbone)
    seen = set()
    for window in windows:
        if not isinstance(window, dict) or set(window) != {'event_id', 'start', 'end', 'maximum_interval'}:
            raise ValueError('Event window needs event_id, start, end and maximum_interval')
        lo, hi, spacing = (number(window[k], k, positive=k!='start' or clock['comoving']) for k in ('start','end','maximum_interval'))
        if hi <= lo or lo < clock['time_begin'] or hi > clock['time_max']:
            raise ValueError('Event window must lie within native clock')
        key = canonical_digest(window)
        if key in seen:
            continue
        seen.add(key)
        # Uniform spacing in ln(a) for comoving clocks, native time otherwise.
        left, right = (math.log(lo), math.log(hi)) if clock['comoving'] else (lo, hi)
        n = max(1, math.ceil((right-left)/spacing))
        if n+1 > maximum_outputs:
            raise ValueError('Event refinement exceeds output budget')
        additions = np.linspace(left, right, n+1)
        times.extend(np.exp(additions).tolist() if clock['comoving'] else additions.tolist())
        if len(set(times)) > maximum_outputs:
            raise ValueError('Merged event schedule exceeds output budget')
    # Match serialization before quantizing; decimal rounding can cross a tick.
    times = sorted(set(float('%.12e' % t) for t in times))
    if len(times)>maximum_outputs:
        raise ValueError('Merged event schedule exceeds output budget')
    rows = [quantize_native_output_time(t, clock) for t in times]
    groups = {}
    for row in rows:
        groups.setdefault(row['native_integer_tick'], []).append(row['requested_time'])
    return {'schema':'event-output-schedule-v1','clock':clock,'requested_times':times,'rows':rows,
            'unique_planned_output_times':[clock['time_begin']*math.exp(k*validate_clock(clock)[0]) if clock['comoving'] else clock['time_begin']+k*validate_clock(clock)[0] for k in sorted(groups)],
            'collisions':[{'native_integer_tick':k,'requested_times':v} for k,v in groups.items() if len(v)>1],
            'science_acceptance':'NOT_ACCEPTED'}


def predict_events(histories):
    """Derive candidates from quality-masked proper trajectories and transitions.

    Histories carry stable string object IDs, explicit cosmic time and proper
    positions relative to the selected host/encounter partner. Input periodic
    wrapping/host-frame conversion belongs to the existing history producer.
    No epoch-local catalogue row is promoted to a persistent object identity.
    """
    events = []
    for history in histories:
        required = {'object_id', 'times_myr', 'relative_positions_proper_kpc', 'valid', 'encounter_radius_proper_kpc', 'transitions'}
        if set(history) != required or not isinstance(history['object_id'], str) or not history['object_id']:
            raise ValueError('History needs stable identity, explicit times, frame, quality and transitions')
        times = np.asarray(history['times_myr'], dtype=float)
        xyz = np.asarray(history['relative_positions_proper_kpc'], dtype=float)
        valid = np.asarray(history['valid'])
        radius = number(history['encounter_radius_proper_kpc'], 'encounter_radius_proper_kpc')
        if times.ndim != 1 or len(times)<2 or not np.isfinite(times).all() or np.any(np.diff(times)<=0) or xyz.shape!=(len(times),3) or valid.shape!=times.shape or valid.dtype.kind!='b' or not np.isfinite(xyz[valid]).all():
            raise ValueError('Invalid history times, positions or quality masks')
        distance=np.linalg.norm(xyz,axis=1)
        saved_minima=set()
        for i in range(1,len(times)-1):
            if valid[i-1:i+2].all() and distance[i]<distance[i-1] and distance[i]<distance[i+1]:
                saved_minima.add(i)
                kind='encounter' if distance[i]<=radius else 'pericenter'
                events.append(_candidate(history['object_id'],kind,float(times[i]),
                    [float(times[i-1]),float(times[i+1])],
                    {'method':'saved-state local minimum','distance_proper_kpc_estimate':float(distance[i])}))
        for i in range(len(times)-1):
            if not valid[i:i+2].all() or i in saved_minima or i+1 in saved_minima:
                continue
            delta = xyz[i+1]-xyz[i]
            norm = float(delta@delta)
            if norm == 0:
                continue
            fraction = float(-xyz[i]@delta/norm)
            if not 0 < fraction < 1:
                continue
            closest = float(np.linalg.norm(xyz[i]+fraction*delta))
            kind = 'encounter' if closest <= radius else 'pericenter'
            events.append(_candidate(history['object_id'],kind,float(times[i]+fraction*(times[i+1]-times[i])),[float(times[i]),float(times[i+1])],
                {'method':'linear segment closest approach','distance_proper_kpc_estimate':closest}))
        for transition in history['transitions']:
            if set(transition) != {'kind','left_index','right_index'} or transition['kind'] not in EVENT_KINDS:
                raise ValueError('Unsupported declared history transition')
            left,right=transition['left_index'],transition['right_index']
            if type(left) is not int or type(right) is not int or not 0<=left<right<len(times) or not valid[left:right+1].all():
                raise ValueError('Event transition spans missing or invalid samples')
            events.append(_candidate(history['object_id'],transition['kind'],float((times[left]+times[right])/2),[float(times[left]),float(times[right])],{'method':'declared saved-state transition'}))
    return sorted({e['event_id']:e for e in events}.values(),key=lambda x:(x['predicted_time_myr'],x['event_id']))


def _candidate(identity,kind,time,bracket,method):
    record={'object_id':identity,'kind':kind,'predicted_time_myr':time,'input_bracket_myr':bracket,
            'interpolation_uncertainty_myr':max(time-bracket[0],bracket[1]-time),'status':'CANDIDATE','prediction':method}
    return {'event_id':canonical_digest(record),**record}


def event_window(event, required_time_error_myr):
    error=number(required_time_error_myr,'required_time_error_myr')
    if event['kind'] not in EVENT_KINDS or event['status']!='CANDIDATE':
        raise ValueError('Only declared candidate events can be refined')
    lo,hi=event['input_bracket_myr'];time=event['predicted_time_myr']
    if not all(math.isfinite(x) for x in (lo,hi,time)) or not lo<=time<=hi or lo==hi:
        raise ValueError('Invalid event bracket')
    return {'event_id':event['event_id'],'start_myr':lo,'end_myr':hi,'required_time_error_myr':error,
            'maximum_interval_myr':2*error,'predicted_time_myr':time}


def saved_outputs_resolve(window, actual_times_myr):
    times=np.asarray(actual_times_myr,dtype=float)
    if times.ndim!=1 or not np.isfinite(times).all() or np.any(np.diff(times)<=0):
        raise ValueError('Actual saved times must increase uniquely')
    lo,hi=window['start_myr'],window['end_myr']
    # Resolve the entire uncertain candidate window, including both edges.
    left=int(np.searchsorted(times,lo,side='right'))-1
    right=int(np.searchsorted(times,hi,side='left'))
    covered=left>=0 and right<len(times)
    gap=float(np.diff(times[left:right+1]).max()) if covered and right>left else None
    passed=covered and gap is not None and gap<=window['maximum_interval_myr']
    return {'status':'RESOLVED_CADENCE' if passed else 'UNRESOLVED_CADENCE',
            'actual_bracket_myr':[float(times[left]),float(times[right])] if covered else None,
            'maximum_saved_interval_myr':gap,'time_error_bound_myr':None if gap is None else gap/2,
            'interpretation':'Sampling support only; event must be remeasured from finer history before physical acceptance'}


def analyze_event(event, histories, required_time_error_myr, *, clock_error_myr=0.):
    """Measure a bracket from saved states; interpolation alone never resolves it.

    A distance event requires a saved interior local minimum. Encounters also
    require a saved position inside the declared encounter aperture. Declared
    state transitions retain their measured endpoint bracket and remain
    candidates until the history producer supplies intrinsic native event data.
    """
    error=number(required_time_error_myr,'required_time_error_myr')
    clock_error=number(clock_error_myr,'clock_error_myr',positive=False)
    if clock_error<0:raise ValueError('Clock error must be nonnegative')
    selected=[h for h in histories if h['object_id']==event['object_id']]
    if len(selected)!=1:raise ValueError('Event requires one exact identity history')
    history=selected[0]
    # Shared prediction validation catches nonmonotonic clocks/missing frames.
    predict_events([history])
    times=np.asarray(history['times_myr'],dtype=float)
    xyz=np.asarray(history['relative_positions_proper_kpc'],dtype=float)
    valid=np.asarray(history['valid'],dtype=bool)
    lo,hi=event['input_bracket_myr']
    indices=np.flatnonzero(valid & (times>=lo) & (times<=hi))
    result={'event_id':event['event_id'],'kind':event['kind'],'object_id':event['object_id'],
            'predicted_time_myr':event['predicted_time_myr'],'status':'UNRESOLVED_EVENT',
            'measured_bracket_myr':None,'measured_time_myr':None,'time_error_bound_myr':None,
            'science_acceptance':'NOT_ACCEPTED'}
    if event['kind'] not in {'encounter','pericenter','bh_pair'}:
        result['reason']='Saved-state transition remains a candidate without intrinsic native event evidence'
        return result
    candidates=[]
    distances=np.linalg.norm(xyz,axis=1)
    for i in indices:
        if i==0 or i==len(times)-1 or not valid[i-1:i+2].all():continue
        if distances[i]<distances[i-1] and distances[i]<distances[i+1]:
            if event['kind'] in {'encounter','bh_pair'} and distances[i]>history['encounter_radius_proper_kpc']:continue
            candidates.append(int(i))
    if len(candidates)!=1:
        result['reason']='No unique saved-state local minimum in the candidate window'
        return result
    i=candidates[0];bound=max(times[i]-times[i-1],times[i+1]-times[i])+clock_error
    result.update(measured_bracket_myr=[float(times[i-1]),float(times[i+1])],measured_time_myr=float(times[i]),
                  time_error_bound_myr=float(bound),measured_distance_proper_kpc=float(distances[i]),
                  status='MEASURED_EVENT' if bound<=error else 'UNRESOLVED_EVENT',
                  reason='Temporal bracket from saved local-minimum samples; dynamics accuracy requires separate convergence')
    return result


def choose_restart(checkpoint_directories, ranks, start_native_time, *, clock_representation=None):
    from core.data import verify_native_checkpoints
    valid, failures=[],[]
    for path in checkpoint_directories:
        try:
            report=verify_native_checkpoints(path,ranks,require_receipts=True)
            if clock_representation is not None and report['clock_representation']!=clock_representation:
                raise ValueError('Checkpoint has a different physical clock representation')
            if report['clock_value']<=start_native_time:
                valid.append((report['clock_value'],str(Path(path).resolve()),report))
            else:
                failures.append({'path':str(path),'reason':'Checkpoint is later than the event window'})
        except (ValueError,OSError,KeyError) as exc:
            failures.append({'path':str(path),'reason':str(exc)})
    if not valid:
        return {'status':'NO_AVAILABLE_RESTART','failures':failures}
    _,path,report=max(valid,key=lambda x:(x[0],x[1]))
    return {'status':'AVAILABLE','directory':path,'checkpoint':report,'failures':failures}


def history_cadence_support(epochs):
    """History consumers retain actual physical output times and clock error."""
    values=[]
    for epoch in epochs:
        contract=epoch.get('epoch_contract',{})
        age=contract.get('cosmic_time_gyr')
        if age is None:
            return {'status':'UNRESOLVED_CLOCK','reason':'Saved epochs lack verified background-derived cosmic time',
                    'event_resolution':'NOT_MEASURED'}
        error=contract.get('cosmic_time_numerical_error_gyr')
        bound=error.get('absolute_bound_gyr') if isinstance(error,dict) else error
        if bound is not None and number(bound,'clock_error',positive=False)<0:
            raise ValueError('Clock error must be nonnegative')
        values.append({'snapshot':epoch['snapshot'],'actual_scale_factor':epoch['scale_factor'],
                       'actual_time_myr':number(age,'cosmic_time_gyr')*1000,
                       'clock_error_myr':None if bound is None else float(bound)*1000,
                       'requested_scale_factor':contract.get('requested_scale_factor')})
    times=[x['actual_time_myr'] for x in values]
    if any(b<=a for a,b in zip(times,times[1:])):
        raise ValueError('History actual cosmic times do not increase')
    return {'status':'MEASURED_OUTPUT_TIMES','outputs':values,'intervals_myr':np.diff(times).tolist(),
            'event_resolution':'NOT_MEASURED','scope':'Actual saved cadence does not establish event occurrence or a precise event time'}


@native_run_owner
@directory_owner('destination')
def publish_history_candidates(run_dir, destination, *, history_path, population,
                               selected_object_ids, host_centers_comoving_kpc,
                               periodic_box_comoving_kpc, encounter_radius_proper_kpc,
                               transitions_by_object, memory_gib):
    """Adapt existing stellar/BH HDF5 histories using exact IDs and saved units.

    Stellar candidate centers use mass-weighted minimum-image offsets about
    each explicit host center. Absent populations retain invalid rows. BH IDs
    are exact; an absent sink is never interpolated into a merger/recoil event.
    """
    import h5py
    from core.data import exact_uint64_ids,run_epoch_contract,verify_hdf5_dataset_payloads
    from core.foundation import verify_product_provenance
    parent=Path(run_dir).resolve();path=Path(history_path).resolve();destination=Path(destination).resolve()
    if destination.exists():raise FileExistsError(destination)
    if population not in {'stellar_candidate','black_hole'}:raise ValueError('Choose stellar_candidate or black_hole history')
    selected=exact_uint64_ids(selected_object_ids,allow_empty=False)
    if len(selected)!=len(np.unique(selected)):raise ValueError('Selected event identities must be unique')
    box=number(periodic_box_comoving_kpc,'periodic_box_comoving_kpc')
    radius=number(encounter_radius_proper_kpc,'encounter_radius_proper_kpc')
    memory=number(memory_gib,'memory_gib')*1024**3
    sources={str(path):file_sha256(path),str(Path(__file__).resolve()):file_sha256(__file__)}
    verify_hdf5_dataset_payloads(path)
    histories={int(i):{'object_id':population+':'+str(int(i)), 'times_myr':[], 'relative_positions_proper_kpc':[],
                      'valid':[], 'encounter_radius_proper_kpc':radius,
                      'transitions':transitions_by_object.get(str(int(i)),[])} for i in selected}
    with h5py.File(path) as f:
        provenance=json.loads(f.attrs['data_contract_json'])
        verify_product_provenance(parent,provenance)
        for source in provenance['input_hashes']:
            sources[source['path']]=source['sha256']
        label_field='CandidateID' if population=='stellar_candidate' else 'ParticleIDs'
        if population=='stellar_candidate' and f['CandidateID'].size*80>memory:
            raise MemoryError('Event candidate identity selection exceeds workspace')
        stellar_labels=exact_uint64_ids(f['CandidateID'][:],unique=False) if population=='stellar_candidate' else None
        epochs=[]
        for _,epoch in sorted(f['Epochs'].items()):
            meta=json.loads(epoch.attrs['metadata_json'])
            contract=run_epoch_contract(parent,int(meta['snapshot']),float(meta['scale_factor']))
            age=contract['cosmic_time_gyr']
            if age is None:raise ValueError('Event histories require verified actual cosmic times')
            if epoch['Coordinates'].size*32>memory:raise MemoryError('Event epoch exceeds declared workspace')
            center=np.asarray(host_centers_comoving_kpc[str(meta['snapshot'])],dtype=float)
            if center.shape!=(3,) or not np.isfinite(center).all():raise ValueError('Declare three finite host coordinates at every saved epoch')
            xyz=np.asarray(epoch['Coordinates'][:],dtype=float)
            a=float(meta['scale_factor'])
            expected_units='comoving kpc' if population=='stellar_candidate' else 'proper kpc'
            units=epoch['Coordinates'].attrs.get('units')
            if units!=expected_units:raise ValueError('History position units differ from population adapter')
            comoving=xyz if population=='stellar_candidate' else xyz/a
            offsets=((comoving-center+box/2)%box-box/2)*a
            labels=stellar_labels if stellar_labels is not None else exact_uint64_ids(epoch[label_field][:])
            present=np.asarray(epoch['Present'][:],dtype=bool) if population=='stellar_candidate' else np.ones(len(labels),dtype=bool)
            masses=np.asarray(epoch['Masses'][:],dtype=float)
            for identity,history in histories.items():
                mask=(labels==np.uint64(identity)) & present
                valid=bool(mask.any())
                if valid and (not np.isfinite(masses[mask]).all() or np.any(masses[mask]<=0) or not np.isfinite(offsets[mask]).all()):
                    raise ValueError('Present event population has invalid positions/masses')
                relative=np.average(offsets[mask],axis=0,weights=masses[mask]).tolist() if valid else [None,None,None]
                history['times_myr'].append(age*1000);history['relative_positions_proper_kpc'].append(relative);history['valid'].append(valid)
            epochs.append({**meta,'epoch_contract':contract})
        for meta in epochs:
            original=meta.get('source_sha256',{})
            if isinstance(original,list):original={s['path']:s['sha256'] for s in original}
            sources.update(original)
    data=list(histories.values());events=predict_events(data)
    support=history_cadence_support(epochs)
    result={'schema':'event-history-candidates-v1','run_dir':str(parent),'population':population,
            'histories':data,'events':events,'actual_output_times_myr':[r['actual_time_myr'] for r in support['outputs']],
            'source_hashes':sources,'host_centers_comoving_kpc':host_centers_comoving_kpc,
            'periodic_box_comoving_kpc':box,'science_acceptance':'NOT_ACCEPTED'}
    if any(file_sha256(p)!=h for p,h in sources.items()):raise ValueError('History source changed before event publication')
    atomic_json(destination,result);atomic_text(destination.with_name(destination.name+'.sha256'),file_sha256(destination)+'\n')
    return result


@native_run_owner
@directory_owner('destination')
def plan_event_followups(run_dir, destination, *, history_product, required_time_error_myr,
                         checkpoint_directories, ranks, maximum_branches, maximum_outputs,
                         maximum_timestep, wall_hours, memory_gib, disk_gib, omp_threads):
    """Verify saved candidate histories, then emit executable bounded branches.

    The product is generated by the history adapter below. Its event brackets
    and source hashes are rechecked; the source run's verified background clock
    converts physical event windows to native a. No fixed H0 approximation.
    """
    from core.cosmology import run_proper_time_clock,interpolate_proper_time
    from scipy.optimize import brentq
    parent=Path(run_dir).resolve();destination=Path(destination).resolve()
    product=Path(history_product).resolve()
    if destination.exists():raise FileExistsError(destination)
    if type(ranks) is not int or ranks<1 or type(omp_threads) is not int or omp_threads<1:
        raise ValueError('Event followups require explicit ranks and threads')
    for key,value in [('wall_hours',wall_hours),('memory_gib',memory_gib),('disk_gib',disk_gib)]:number(value,key)
    if type(maximum_branches) is not int or maximum_branches<0:raise ValueError('Choose a nonnegative branch limit')
    if product.with_name(product.name+'.sha256').read_text().strip()!=file_sha256(product):
        raise ValueError('Event history product seal changed')
    data=json.loads(product.read_text())
    if data.get('schema')!='event-history-candidates-v1' or data.get('run_dir')!=str(parent):
        raise ValueError('Event history product belongs to a different run')
    for path,checksum in data['source_hashes'].items():
        if file_sha256(path)!=checksum:raise ValueError('Event history input changed')
    events=reconstruct_published_source_events(data) if data.get('event_origin')=='committed_native_operators' else predict_events(data['histories'])
    if events!=data['events']:raise ValueError('Stored candidates differ from measured histories')
    scale,ages,source=run_proper_time_clock(parent)
    if scale is None:raise ValueError('Event refinement requires a verified physical background clock')
    params=parameter_values(parent/'input/param.txt')
    begin,end=float(params['TimeBegin']),float(params['TimeMax'])
    original_step=float(params['MaxSizeTimestep'])
    number(maximum_timestep,'maximum_timestep')
    if maximum_timestep>original_step:raise ValueError('Event branch cannot increase the native timestep')
    native_config=(parent/'input/Config.sh').read_text().splitlines()
    clock={'time_begin':begin,'time_max':end,'maximum_timestep':maximum_timestep,
           'timebins':60 if 'ENLARGE_DYNAMIC_RANGE_IN_TIME' in native_config else 29,'comoving':True,
           'output_mode':'exact_synchronized' if 'OUTPUT_EXACT_SYNCHRONIZED' in native_config else 'timestep_grid'}
    merge_and_quantize_native_output_times([begin,end],[],clock,maximum_outputs=maximum_outputs)
    if int(params['ComovingIntegrationOn'])!=1:raise ValueError('Physical event followups require cosmological clocks')
    def native_time(t):
        number(t,'event cosmic time',positive=False)
        lower=interpolate_proper_time(scale,ages,begin)*1000
        upper=interpolate_proper_time(scale,ages,end)*1000
        if not lower<=t<=upper:raise ValueError('Event lies outside the evolved interval')
        return float(brentq(lambda a:interpolate_proper_time(scale,ages,a)*1000-t,begin,end,xtol=1e-14))
    actual_times=data['actual_output_times_myr']
    if not data['histories'] or any(h['times_myr']!=actual_times for h in data['histories']):
        raise ValueError('Actual output times must match every saved identity history')
    queued=[];decisions=[];seen={}
    for event in events:
        window=event_window(event,required_time_error_myr)
        from core.data import proper_time_error_at_epoch
        native_window=[native_time(window['start_myr']),native_time(window['end_myr'])]
        errors=[proper_time_error_at_epoch(source,a).get('absolute_bound_gyr') for a in native_window]
        if any(e is None for e in errors):
            decisions.append({'event':event,'window':window,'status':'UNBOUNDED_BACKGROUND_CLOCK'});continue
        bound=max(errors)*1000
        if bound>=window['required_time_error_myr']:
            decisions.append({'event':event,'window':window,'status':'REQUIRES_REFINED_BACKGROUND_CLOCK','clock_error_bound_myr':bound});continue
        window['clock_error_bound_myr']=bound
        window['maximum_interval_myr']=2*(window['required_time_error_myr']-bound)
        measured=saved_outputs_resolve(window,actual_times)
        decision={'event':event,'window':window,'saved_cadence':measured,'status':measured['status']}
        if measured['status']=='RESOLVED_CADENCE':
            if data.get('event_origin')=='committed_native_operators':
                uncertainty=event['interpolation_uncertainty_myr']
                resolved=event['prediction']['clock_error_status']!='UNBOUNDED' and uncertainty<=required_time_error_myr
                decision['event_measurement']={'status':'RESOLVED_OPERATOR_TIME' if resolved else 'REQUIRES_FINER_SOURCE_STEP',
                    'maximum_time_error_myr':uncertainty,'source':'committed native source operation',
                    'scope':'Native operator timing; continuous astrophysical event timing and physical timestep convergence remain conditional'}
                decision['status']=decision['event_measurement']['status']
                if not resolved:measured['status']='UNRESOLVED_CADENCE'
            else:
                decision['event_measurement']=analyze_event(event,data['histories'],required_time_error_myr,clock_error_myr=bound)
                decision['status']=decision['event_measurement']['status']
        if measured['status']=='UNRESOLVED_CADENCE':
            left,right=native_time(window['start_myr']),native_time(window['end_myr'])
            available=choose_restart(checkpoint_directories,ranks,left,clock_representation='scale_factor')
            decision['restart_availability']=available
            key=(left,right)
            if available['status']!='AVAILABLE':decision['status']='REQUIRES_EARLIER_CHECKPOINT_OR_IC_RERUN'
            elif key in seen:
                queued[seen[key]]['event_ids'].append(event['event_id'])
                decision.update(status='SHARED_QUEUED_BRANCH',branch_index=seen[key])
            elif len(queued)>=maximum_branches:decision['status']='FOLLOWUP_LIMIT_REACHED'
            else:
                # Sample uniformly in physical time, then quantize the actual
                # inverse background clock. Avoid uniform-a assumptions.
                n=max(1,math.ceil((window['end_myr']-window['start_myr'])/window['required_time_error_myr']))
                if n+1>maximum_outputs:
                    decision['status']='OUTPUT_LIMIT_REACHED'
                else:
                    times=[native_time(t) for t in np.linspace(window['start_myr'],window['end_myr'],n+1)]
                    padded_left=max(begin,left*math.exp(-maximum_timestep))
                    padded_right=min(end,right*math.exp(maximum_timestep))
                    times=sorted(set([padded_left,*times,padded_right,end]))
                    if len(times)>maximum_outputs:
                        decision['status']='OUTPUT_LIMIT_REACHED'
                        decisions.append(decision)
                        continue
                    schedule=merge_and_quantize_native_output_times(times,[],clock,maximum_outputs=maximum_outputs)
                    quantized_physical=[interpolate_proper_time(scale,ages,a)*1000 for a in schedule['unique_planned_output_times']]
                    planned=saved_outputs_resolve(window,quantized_physical)
                    if planned['status']!='RESOLVED_CADENCE':
                        decision['status']='REQUIRES_SMALLER_NATIVE_TIMESTEP'
                    else:
                        branch={'run_dir':str(parent),'checkpoint_directory':available['directory'],'schedule':schedule,
                            'event_ids':[event['event_id']],'ranks':ranks,'omp_threads':omp_threads,'wall_hours':wall_hours,
                            'memory_gib':memory_gib,'disk_gib':disk_gib}
                        queued.append(branch);seen[key]=len(queued)-1;decision.update(status='QUEUED_CHECKPOINT_BRANCH',branch_index=len(queued)-1)
        decisions.append(decision)
    destination.mkdir(parents=True)
    for index,branch in enumerate(queued):atomic_json(destination/f'branch-{index:03d}.json',branch)
    sources={str(product):file_sha256(product),str(Path(__file__).resolve()):file_sha256(__file__),
             str(parent/'input/param.txt'):file_sha256(parent/'input/param.txt'),
             str(parent/'input/Config.sh'):file_sha256(parent/'input/Config.sh'),**data['source_hashes']}
    result={'schema':'event-followup-plan-v1','status':'PLANNED','events':decisions,'branch_requests':len(queued),
            'source_hashes':sources,'background_clock':source,'science_acceptance':'NOT_ACCEPTED',
            'execution_command':'commands.py branch-NNN.json FRESH_BRANCH --execute-branch'}
    if any(file_sha256(p)!=h for p,h in sources.items()):raise ValueError('Event followup sources changed before publication')
    atomic_json(destination/'plan.json',result)
    atomic_text(destination/'plan.json.sha256',file_sha256(destination/'plan.json')+'\n')
    return result


def parameter_values(path):
    result={}
    for line in Path(path).read_text().splitlines():
        fields=line.split()
        if not fields or fields[0].startswith(('%','#')):
            continue
        if len(fields)!=2 or fields[0] in result:
            raise ValueError('Ambiguous native parameter file')
        result[fields[0]]=fields[1]
    return result


def verify_event_branch(path):
    """Recheck a completed branch and its original parent/native payloads."""
    path=Path(path).resolve()
    if path.with_name(path.name+'.sha256').read_text().strip()!=file_sha256(path):
        raise ValueError('Event branch lineage seal mismatch')
    report=json.loads(path.read_text())
    if report.get('schema')!='event-checkpoint-branch-v1' or report.get('status')!='COMPLETE':
        raise ValueError('Event branch is incomplete')
    if report['algorithm_sha256']!=file_sha256(__file__):
        raise ValueError('Event branch algorithm changed')
    for p,h in report['source_hashes'].items():
        if file_sha256(p)!=h:raise ValueError('Event branch parent source changed')
    for row in report['measured_outputs']:
        for p,h in row['files'].items():
            if file_sha256(p)!=h:raise ValueError('Event branch saved output changed')
    from core.native import verify_invocation_inputs
    invocation=json.loads((path.parent/'invocation.json').read_text())
    if invocation.get('status')!='RETURNED' or invocation.get('returncode')!=0:
        raise ValueError('Event branch native invocation did not complete')
    verify_invocation_inputs(invocation)
    return report


@native_run_owner
@directory_owner('destination')
def execute_checkpoint_branch(run_dir, destination, *, checkpoint_directory,
                              schedule, event_ids, ranks, omp_threads,
                              wall_hours, memory_gib, disk_gib, reuse_completed=False):
    """Run the original binary/ranks from a verified copied interior restart.

    The parent's binary, numerical physics and native clock range are preserved.
    Only output directory/list, a reduced maximum step and wall-time limit may
    change. Parent files are checksum-checked before and after execution.
    """
    from core.native import run_native_command
    from core.data import verify_native_checkpoints
    from analysis import validate_gadget_snapshot_set
    import h5py
    parent=Path(run_dir).resolve(); destination=Path(destination).resolve()
    settings={'ranks':ranks,'omp_threads':omp_threads,'wall_hours':wall_hours,'memory_gib':memory_gib,'disk_gib':disk_gib}
    if destination.exists() and reuse_completed:
        report=verify_event_branch(destination/'lineage.json')
        if report['parent_run']!=str(parent) or report['schedule']!=schedule or report['event_ids']!=sorted(set(event_ids)) or report['execution_settings']!=settings or str(Path(checkpoint_directory).resolve())!=report['checkpoint_directory']:
            raise ValueError('Completed event branch does not match this request')
        return report
    if destination.exists() or destination==parent or destination.is_relative_to(parent/'data'):
        raise FileExistsError('Event branch requires a fresh separate destination')
    if type(ranks) is not int or ranks<1 or type(omp_threads) is not int or omp_threads<1:
        raise ValueError('Branch ranks and threads must be explicit positive integers')
    for name,value in [('wall_hours',wall_hours),('memory_gib',memory_gib),('disk_gib',disk_gib)]:number(value,name)
    if not isinstance(event_ids,list) or not event_ids or any(not isinstance(x,str) or not x for x in event_ids):
        raise ValueError('Branch must retain event identities')
    source_param=parent/'input/param.txt';binary=parent/'input/Gadget4'
    if not binary.is_file():
        raise FileNotFoundError('Original native binary is required')
    params=parameter_values(source_param)
    clock=schedule['clock'];validate_clock(clock)
    if (clock['time_begin']!=float(params['TimeBegin']) or clock['time_max']!=float(params['TimeMax']) or clock['comoving']!=(int(params['ComovingIntegrationOn'])==1) or clock['maximum_timestep']>float(params['MaxSizeTimestep'])):
        raise ValueError('Branch cannot change original native clock or increase the timestep')
    native_config=(parent/'input/Config.sh').read_text()
    timebins=60 if 'ENLARGE_DYNAMIC_RANGE_IN_TIME' in native_config.splitlines() else 29
    output_mode='exact_synchronized' if 'OUTPUT_EXACT_SYNCHRONIZED' in native_config.splitlines() else 'timestep_grid'
    if (clock['timebins']!=timebins or clock.get('output_mode','timestep_grid')!=output_mode
            or 'OUTPUT_NON_SYNCHRONIZED_ALLOWED' in native_config.splitlines()):
        raise ValueError('Branch schedule does not match native compiled clock')
    verified=verify_native_checkpoints(checkpoint_directory,ranks,require_receipts=True)
    if (verified['clock_representation']=='scale_factor')!=clock['comoving']:
        raise ValueError('Restart and schedule clock representations differ')
    actual_schedule=merge_and_quantize_native_output_times(schedule['requested_times'],[],clock)
    if actual_schedule['rows']!=schedule['rows']:
        raise ValueError('Event schedule quantization changed')
    requested=[r['requested_time'] for r in schedule['rows'] if r['planned_quantized_time']>verified['clock_value']]
    if not requested:
        raise ValueError('No refined outputs after the available restart')
    retained=list((parent/'input').glob('*'))
    sources={str(p.resolve()):file_sha256(p) for p in retained if p.is_file()}
    for helper in ('evolution.py','data.py','foundation.py','data.py','data.py'):
        p=Path(__file__).with_name(helper);sources[str(p.resolve())]=file_sha256(p)
    checkpoint_files=[]
    for row in verified['files']:
        p=Path(row['path'])
        checkpoint_files += [p,Path(str(p)+'.checksum.json')]
        if row['epoch_metadata']:checkpoint_files.append(Path(row['epoch_metadata']['path']))
    sources.update({str(p.resolve()):file_sha256(p) for p in checkpoint_files})
    # Copy footprint bounds disk use; native allocations are capped per rank.
    footprint=sum(p.stat().st_size for p in retained+checkpoint_files if p.is_file())
    if footprint>disk_gib*1024**3 or footprint>shutil.disk_usage(destination.parent).free:
        raise MemoryError('Event branch retained inputs exceed disk budget')
    destination.mkdir();(destination/'input').mkdir();(destination/'data/restartfiles').mkdir(parents=True)
    lineage={'schema':'event-checkpoint-branch-v1','parent_run':str(parent),'event_ids':sorted(set(event_ids)),
             'checkpoint_directory':str(Path(checkpoint_directory).resolve()),'execution_settings':settings,
             'checkpoint':verified,'schedule':schedule,'source_hashes':sources,'status':'PREPARED',
             'algorithm_sha256':file_sha256(__file__),'science_acceptance':'NOT_ACCEPTED'}
    atomic_json(destination/'lineage.json',lineage)
    try:
        for p in retained:
            if p.is_file():atomic_copy(p,destination/'input'/p.name)
        for p in checkpoint_files:atomic_copy(p,destination/'data/restartfiles'/p.name)
        params.update(OutputDir=str(destination/'data')+'/',OutputListFilename=str(destination/'input/output_times.txt'),
                      MaxSizeTimestep=format(clock['maximum_timestep'],'.17g'),TimeLimitCPU=str(wall_hours*3600),
                      MaxMemSize=str(max(1,int(memory_gib*1024/ranks))))
        for key in ('InitCondFile','TreecoolFile','CosmologyTableFile'):
            value=params.get(key)
            if value and value!='none' and not Path(value).is_absolute():
                params[key]=str((parent/'input'/value).resolve())
        atomic_text(destination/'input/param.txt',''.join(k+' '+v+'\n' for k,v in params.items()))
        atomic_text(destination/'input/output_times.txt',''.join('%.12e\n'%x for x in requested))
        env=os.environ.copy();env['OMP_NUM_THREADS']=str(omp_threads)
        with (destination/'native.log').open('w') as log:
            run_native_command(['mpirun','-np',str(ranks),str(destination/'input/Gadget4'),str(destination/'input/param.txt'),'1'],
                cwd=destination/'input',receipt_path=destination/'invocation.json',inputs=[*sources,destination/'input/output_times.txt'],
                env=env,stdout=log,stderr=__import__('subprocess').STDOUT,timeout=wall_hours*3600)
        groups={}
        for p in (destination/'data').glob('snapshot_*.hdf5'):
            index=int(p.name.split('_')[1].split('.')[0]);groups.setdefault(index,[]).append(p)
        measured=[]
        for index,paths in sorted(groups.items()):
            validate_gadget_snapshot_set(paths,index)
            with h5py.File(paths[0]) as f:actual=float(f['Header'].attrs['Time'])
            measured.append({'snapshot':index,'actual_output_time':actual,'files':{str(p.resolve()):file_sha256(p) for p in paths}})
        expected=sorted(set(r['planned_quantized_time'] for r in schedule['rows'] if r['planned_quantized_time']>verified['clock_value']))
        if len(measured)!=len(expected) or not np.allclose([r['actual_output_time'] for r in measured],expected,rtol=1e-12,atol=1e-14):
            raise ValueError('Native branch did not save every refined quantized output')
        if any(file_sha256(p)!=h for p,h in sources.items()):
            raise ValueError('Parent inputs changed during branch execution')
        lineage.update(status='COMPLETE',measured_outputs=measured)
        atomic_json(destination/'lineage.json',lineage)
        atomic_text(destination/'lineage.json.sha256',file_sha256(destination/'lineage.json')+'\n')
        return lineage
    except BaseException as exc:
        atomic_json(destination/'failure.json',{'status':'FAIL','reason':str(exc),'science_acceptance':'NOT_ACCEPTED'})
        raise

# ========================================================================
# start_epoch_campaign: Matched start-epoch campaigns, actual measurements and comparisons.
from core.foundation import file_sha256 as digest, publish_directory
from initial_conditions import validate_linear_bundle, resolve_start_epoch
from core.data import export_numeric_audit, verify_numeric_audit
from core.data import epoch_contract, CANONICAL_SCHEMA
from core.data import verify_native_hdf5_set
import datetime
import struct


# Start epoch campaign
#
# Matched, user-selected start epochs with frozen noise and paired measurements.
#
# No transfer is rescaled between epochs. Covariance is estimated across matched
# independent seeds, never across bins or snapshots of a single realization.

def _number(value, name, minimum=0., strict=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(name+' must be finite and numeric')
    if value < minimum or (strict and value == minimum):
        raise ValueError(name+' is outside its domain')
    return float(value)


def require_distinct_valid_epochs(redshifts):
    if not isinstance(redshifts, list) or not 2 <= len(redshifts) <= 64:
        raise ValueError('Declare 2 to 64 distinct start redshifts')
    values=[_number(z,'start redshift',strict=True) for z in redshifts]
    if any(abs(a-b)<=1e-8 for i,a in enumerate(values) for b in values[i+1:]):
        raise ValueError('Start epochs must be distinct at solver epoch precision')
    return values


def _checked(record):
    if not isinstance(record,dict) or set(record) != {'path','sha256'}:
        raise ValueError('Artifact requires exactly path and sha256')
    if not isinstance(record['path'],str) or not isinstance(record['sha256'],str):
        raise ValueError('Artifact path/hash must be strings')
    path=Path(record['path']).expanduser().resolve()
    if not path.is_file() or digest(path)!=record['sha256']:
        raise ValueError('Campaign source changed or is missing: '+str(path))
    return path


def _record(path):
    path=Path(path).resolve()
    return {'path':str(path),'sha256':digest(path)}


def validate_start_epoch_request(request, observables=None):
    required={'schema_version','configuration','redshifts','reference_redshift','seeds',
              'common_scale_factors','epoch_tolerance','confidence_sigma','observables','linear_bundles'}
    if not isinstance(request,dict) or set(request)-required-{'noise_by_seed'} or not required<=set(request):
        raise ValueError('Malformed start_epoch_campaign request')
    if type(request['schema_version']) is not int or request['schema_version']!=1:
        raise ValueError('Start-epoch request requires schema_version=1')
    zs=require_distinct_valid_epochs(request['redshifts'])
    reference=_number(request['reference_redshift'],'reference redshift',strict=True)
    if reference not in zs:raise ValueError('Reference start redshift must be explicitly selected')
    seeds=request['seeds']
    if (not isinstance(seeds,list) or not 1<=len(seeds)<=64 or len(zs)*len(seeds)>256
            or any(type(s) is not int or not 0<=s<=2**64-1 for s in seeds) or len(set(seeds))!=len(seeds)):
        raise ValueError('Declare distinct exact uint64 seeds; at most 256 branches')
    epochs=request['common_scale_factors']
    if not isinstance(epochs,list) or not epochs:
        raise ValueError('Declare common actual output scale factors')
    epochs=[_number(a,'common scale factor',strict=True) for a in epochs]
    if any(a>1 for a in epochs) or any(b<=a for a,b in zip(epochs,epochs[1:])):
        raise ValueError('Common scale factors must increase within (0,1]')
    tolerance=_number(request['epoch_tolerance'],'epoch tolerance',strict=True)
    if tolerance>1e-6 or any(b-a<=2*tolerance for a,b in zip(epochs,epochs[1:])):
        raise ValueError('Actual epoch matching tolerance is too broad or ambiguous')
    if epochs[0]<=max(1/(1+z) for z in zs)+tolerance:
        raise ValueError('Common outputs must follow every selected start epoch')
    _number(request['confidence_sigma'],'confidence sigma')
    rules=request['observables']
    if not isinstance(rules,dict) or not rules or (observables is not None and set(rules)-set(observables)):
        raise ValueError('Campaign observables must be declared research observables')
    for name,rule in rules.items():
        if not isinstance(rule,dict) or set(rule)!={'units','bin_labels','absolute_tolerance','relative_tolerance','absolute_floor'}:
            raise ValueError('Declare units, aligned bins and numerical tolerances: '+name)
        bins=rule['bin_labels']
        if (not isinstance(rule['units'],str) or not rule['units'] or not isinstance(bins,list) or not bins
                or any(not isinstance(b,str) or not b for b in bins) or len(set(bins))!=len(bins)):
            raise ValueError('Observable units and unique bin labels are required')
        absolute=_number(rule['absolute_tolerance'],'absolute tolerance')
        relative=_number(rule['relative_tolerance'],'relative tolerance')
        if not absolute+relative>0:raise ValueError('At least one convergence tolerance must be positive')
        _number(rule['absolute_floor'],'absolute residual floor',strict=True)
        if observables is not None and rule['units']!=observables[name]['units']:
            raise ValueError('Campaign observable units disagree with the research definition')
    _checked(request['configuration'])
    bundles=request['linear_bundles']
    if not isinstance(bundles,list) or len(bundles)!=len(zs):
        raise ValueError('One exact-epoch linear bundle is required per selected start')
    bundle_z=[]
    for entry in bundles:
        if not isinstance(entry,dict) or set(entry)!={'redshift','manifest'}:
            raise ValueError('Linear bundle entry requires redshift and manifest')
        bundle_z.append(_number(entry['redshift'],'bundle redshift',strict=True));_checked(entry['manifest'])
    if sorted(bundle_z)!=sorted(zs):raise ValueError('Bundle epochs do not match the selected starts')
    if 'noise_by_seed' in request:
        noise=request['noise_by_seed']
        if not isinstance(noise,dict) or set(noise)!=set(map(str,seeds)):
            raise ValueError('External noise must cover every declared seed exactly')
        for entry in noise.values():_checked(entry)
    return request


def _physics_payload(configuration):
    result=copy.deepcopy(configuration)
    # These are the only varied fields. Everything else remains byte-semantically matched.
    result['simulation'].pop('start_redshift',None)
    result['simulation'].pop('random_seed',None)
    result['spectrum'].pop('file',None)
    result['spectrum'].pop('redshift',None)
    return result


def _bundle_physics(info):
    parameters=copy.deepcopy(info['parameters']);parameters.pop('z_start',None)
    # run_class names each published solver run independently. The label is
    # output bookkeeping, while every cosmological/numerical input stays bound.
    parameters.pop('name',None)
    solver=copy.deepcopy(info['class_parameters']['start']);solver.pop('z_pk',None)
    # Solver output directories carry no cosmological parameter.
    solver.pop('root',None)
    return {'model':info['model'],'solver':info['solver'],'parameters':parameters,'solver_input':solver}


def _match_configuration(info,cfg):
    c=cfg['cosmology'];h=float(c['H0'])/100
    expected={'H0':c['H0'],'Omega_b':c['omega_b']/h**2,'Omega_cdm':c['omega_cdm']/h**2,
              'A_s':c['A_s'],'n_s':c['n_s']}
    if info['model']!=c['model']:raise ValueError('Campaign configuration and bundle models differ')
    if c['model']=='ede':
        e=cfg['ede'];expected.update(f_ede=e['f_ede'],log10_zc=math.log10(e['z_c']),theta_i=e['theta_i'],n_axion=e['n_axion'])
    for key,value in expected.items():
        if not math.isclose(float(value),float(info['parameters'][key]),rel_tol=1e-12,abs_tol=0):
            raise ValueError('Campaign bundle changes configured physics: '+key)
    for key in ('T_cmb','N_ur'):
        if not math.isclose(float(c[key]),float(info['class_parameters']['start'][key]),rel_tol=1e-12):
            raise ValueError('Campaign bundle changes radiation: '+key)


def _noise_shape(path, expected_size=None):
    with Path(path).open('rb') as stream:
        header=stream.read(24)
    if len(header)!=24:raise ValueError('Truncated GRAFIC noise header')
    marker,nx,ny,nz,_,end=struct.unpack('<6i',header)
    if marker!=16 or end!=16 or nx!=ny or nx!=nz or nx<2 or nx&(nx-1):
        raise ValueError('Shared noise requires a cubic power-of-two float64 GRAFIC field')
    if expected_size is not None and nx!=expected_size:raise ValueError('External noise grid differs from the campaign')
    if Path(path).stat().st_size!=24+nx*(8+nx*nx*8):
        raise ValueError('Shared noise record sizes or float64 encoding are invalid')
    with Path(path).open('rb') as stream:
        stream.seek(24)
        for _ in range(nx):
            count=struct.unpack('<i',stream.read(4))[0]
            values=np.frombuffer(stream.read(nx*nx*8),dtype='<f8')
            end=struct.unpack('<i',stream.read(4))[0]
            if count!=nx*nx*8 or end!=count or not np.isfinite(values).all():
                raise ValueError('Shared noise contains corrupt records or nonfinite values')
    return nx


@directory_owner('destination')
def freeze_phase_realization(seed, destination, *, grid_size, memory_gib, external=None):
    """Persist shared Gaussian noise once, retaining the full uint64 seed.

    The GRAFIC header uses a sentinel seed of zero; the immutable metadata keeps
    the real uint64 seed. No narrowing of a user seed changes the random draw.
    """
    if type(seed) is not int or not 0<=seed<=2**64-1:raise ValueError('Noise seed must be exact uint64')
    if type(grid_size) is not int or grid_size<2 or grid_size&(grid_size-1):
        raise ValueError('Noise grid must be a power of two')
    budget=_number(memory_gib,'noise memory budget',strict=True)*1024**3
    if 16*grid_size**3+64*1024**2>budget:raise MemoryError('Frozen noise exceeds declared memory budget')
    destination=Path(destination).resolve()
    if destination.exists():raise FileExistsError('Keep previous frozen phase realization')
    destination.parent.mkdir(parents=True,exist_ok=True)
    stage=Path(tempfile.mkdtemp(prefix='.phase-',dir=destination.parent))
    try:
        path=stage/'noise.grafic'
        if external is not None:
            source=_checked(external)
            if _noise_shape(source,grid_size)!=grid_size:raise ValueError('External noise grid differs from the campaign')
            shutil.copyfile(source,path);_checked(external)
            generator='externally supplied immutable realization; seed label retained, RNG origin not inferred'
        else:
            values=np.random.Generator(np.random.Philox(seed)).standard_normal((grid_size,)*3)
            generator='NumPy Philox standard normal float64; homogeneous noise retained, native density DC removed'
            with path.open('wb') as stream:
                stream.write(struct.pack('<6i',16,grid_size,grid_size,grid_size,0,16))
                size=grid_size*grid_size*8
                for k in range(grid_size):
                    stream.write(struct.pack('<i',size));stream.write(values[:,:,k].T.astype('<f8').tobytes())
                    stream.write(struct.pack('<i',size))
        _noise_shape(path)
        report={'schema':'start-epoch-frozen-noise-v1','seed':seed,'grid_size':grid_size,'generator':generator,
                'noise_sha256':digest(path),'algorithm_sha256':digest(__file__),'science_acceptance':'NOT_ACCEPTED'}
        atomic_json(stage/'phase.json',report);publish_directory(stage,destination)
        return report
    finally:
        if stage.exists():shutil.rmtree(stage)


@directory_owner('destination')
def build_start_epoch_campaign(base_spec, redshifts, destination, *, regenerate=None, allow_regeneration=False, resume=False):
    """Publish matched branch configurations/specifications and one noise per seed."""
    from core.foundation import validate_spec
    validate_spec(base_spec)
    request=copy.deepcopy(base_spec.get('start_epoch_campaign'))
    validate_start_epoch_request(request,base_spec['observables'])
    zs=require_distinct_valid_epochs(redshifts)
    if zs!=request['redshifts']:raise ValueError('Build epochs must equal the user-selected request in order')
    destination=Path(destination).resolve()
    if destination.exists():
        if resume is not True:raise FileExistsError('Keep the previous campaign or explicitly resume it')
        existing=verify_start_epoch_campaign(destination/'campaign.json')
        if existing['base_spec_sha256']!=canonical_digest(base_spec) or existing['request']!=request:
            raise ValueError('Resume requires the identical specification and selected epochs')
        return existing
    cfg_path=_checked(request['configuration']);cfg=yaml.safe_load(cfg_path.read_text())
    if not isinstance(cfg,dict) or cfg['spectrum']['source']!='modular':
        raise ValueError('Start campaigns require an explicit modular solver-bundle configuration')
    n=cfg['simulation']['particles_per_dimension']
    if type(n) is not int or n<2 or n&(n-1):raise ValueError('Campaign grid size must be a power of two')
    base_seed=cfg['simulation']['random_seed']
    if type(base_seed) is not int or not 0<=base_seed<=2**64-1:raise ValueError('Base configuration needs an exact uint64 seed')
    level=n.bit_length()-1
    if not 1<=level<=20:raise ValueError('Campaign refinement level is outside the supported domain')
    budget=base_spec['resource_budget'];noise_bytes=len(request['seeds'])*(24+n*(8+8*n*n))
    if noise_bytes+64*1024**2>budget['disk_gib']*1024**3:raise OSError('Frozen phases exceed declared disk budget')
    schedule=cfg['simulation'].get('snapshot_redshift_list')
    if not isinstance(schedule,list) or not schedule:raise ValueError('Retain an explicit common output schedule')
    output_a=[1/(1+_number(z,'output redshift')) for z in schedule]
    if min(output_a)<=max(1/(1+z) for z in zs):raise ValueError('Base output schedule precedes a selected start')
    for a in request['common_scale_factors']:
        if sum(abs(a-b)<=request['epoch_tolerance'] for b in output_a)!=1:
            raise ValueError('Common epochs must be present exactly once in the unchanged output schedule')
    if max(output_a)>1/(1+_number(cfg['simulation']['end_redshift'],'end redshift'))+request['epoch_tolerance']:
        raise ValueError('Output schedule extends beyond the configured integration interval')
    inputs={str(cfg_path):digest(cfg_path),str(Path(__file__).resolve()):digest(__file__)}
    base_z=_number(cfg['simulation']['start_redshift'],'base start redshift',strict=True)
    base=validate_linear_bundle(cfg['spectrum']['file'],redshift=base_z,
        box_mpc_h=cfg['simulation']['box_mpc_h'],refinement_level=level,memory_gib=budget['memory_gib'])
    base_info=json.loads(Path(base['manifest']['path']).read_text());_match_configuration(base_info,cfg)
    physics=_bundle_physics(base_info)
    for item in [base['manifest'],*base['sources'].values()]:inputs[item['path']]=item['sha256']
    bundles={};synthetic='SYNTHETIC' in str(base_info.get('fixture','')).upper()
    for entry in request['linear_bundles']:
        source=_checked(entry['manifest']);z=float(entry['redshift'])
        validated=resolve_start_epoch(source,z,regenerate=regenerate,allow_regeneration=allow_regeneration,
            box_mpc_h=cfg['simulation']['box_mpc_h'],refinement_level=level,memory_gib=budget['memory_gib'])
        if max(output_a)>validated['background_a_range'][1]:
            raise ValueError('Solver background does not cover the common integration/output clock range')
        info=json.loads(Path(validated['manifest']['path']).read_text());_match_configuration(info,cfg)
        signature=_bundle_physics(info)
        if physics!=signature:raise ValueError('Start-epoch bundles change unrelated solver physics')
        synthetic|='SYNTHETIC' in str(info.get('fixture','')).upper()
        for item in [validated['manifest'],*validated['sources'].values()]:inputs[item['path']]=item['sha256']
        bundles[z]=validated
    destination=Path(destination).resolve()
    if destination.exists():raise FileExistsError('Keep the previous campaign')
    destination.parent.mkdir(parents=True,exist_ok=True)
    stage=Path(tempfile.mkdtemp(prefix='.start-epoch-',dir=destination.parent))
    try:
        phases={};branches=[];invariant=canonical_digest(_physics_payload(cfg))
        for seed in request['seeds']:
            relative=Path('phases')/str(seed)
            external=request.get('noise_by_seed',{}).get(str(seed))
            phase=freeze_phase_realization(seed,stage/relative,grid_size=n,memory_gib=budget['memory_gib'],external=external)
            if external:inputs[str(_checked(external))]=external['sha256']
            phases[str(seed)]={**phase,'noise_path':str(destination/relative/'noise.grafic'),
                              'phase_id':canonical_digest(phase)}
            for index,z in enumerate(zs):
                identity=f'seed-{seed}-start-{index:02d}'
                relative=Path('branches')/identity;folder=stage/relative;folder.mkdir(parents=True)
                branch=copy.deepcopy(cfg)
                branch['simulation'].update(start_redshift=z,random_seed=seed)
                branch['spectrum'].update(file=bundles[z]['manifest']['path'],redshift=z)
                if canonical_digest(_physics_payload(branch))!=invariant:raise AssertionError('Branch changed unrelated physics')
                (folder/'config.yaml').write_text(yaml.safe_dump(branch,sort_keys=False))
                atomic_json(folder/'research_spec.json',base_spec)
                branches.append({'id':identity,'seed':seed,'start_redshift':z,'start_scale_factor':1/(1+z),
                    'phase_id':phases[str(seed)]['phase_id'],'noise':{'path':phases[str(seed)]['noise_path'],'sha256':phase['noise_sha256']},
                    'configuration':{'path':str(destination/relative/'config.yaml'),'sha256':digest(folder/'config.yaml')},
                    'research_spec':{'path':str(destination/relative/'research_spec.json'),'sha256':digest(folder/'research_spec.json')},
                    'bundle':bundles[z],'physics_fingerprint':invariant})
        if len({phase['noise_sha256'] for phase in phases.values()})!=len(phases):
            raise ValueError('Independent seeds cannot reuse the same frozen noise')
        # Locks coordinate preparation, but are not immutable campaign products.
        shutil.rmtree(stage/'.native-run-locks',ignore_errors=True)
        for path in stage.rglob('.native-run-locks'):shutil.rmtree(path)
        for path,checksum in inputs.items():
            if digest(path)!=checksum:raise ValueError('Campaign input changed while preparing: '+path)
        report={'schema':'start-epoch-campaign-v1','status':'PREPARED','request':request,
            'request_sha256':canonical_digest(request),'base_spec_sha256':canonical_digest(base_spec),
            'physics_fingerprint':invariant,'solver_physics_fingerprint':canonical_digest(physics),
            'grid_level':level,'phases':phases,'branches':branches,'source_hashes':inputs,
            'synthetic_inputs':synthetic,'science_acceptance':'NOT_ACCEPTED',
            'artifacts':{p.relative_to(stage).as_posix():digest(p) for p in stage.rglob('*') if p.is_file()}}
        atomic_json(stage/'campaign.json',report);(stage/'campaign.json.sha256').write_text(digest(stage/'campaign.json')+'\n')
        publish_directory(stage,destination)
        return verify_start_epoch_campaign(destination/'campaign.json')
    finally:
        if stage.exists():shutil.rmtree(stage)


def verify_start_epoch_campaign(path):
    path=Path(path).resolve();before=digest(path)
    if before!=path.with_name(path.name+'.sha256').read_text().strip():raise ValueError('Campaign seal changed')
    report=json.loads(path.read_text())
    if report.get('schema')!='start-epoch-campaign-v1' or report.get('status')!='PREPARED':
        raise ValueError('Campaign is not prepared')
    validate_start_epoch_request(report['request'])
    if report['request_sha256']!=canonical_digest(report['request']):raise ValueError('Campaign request changed')
    for relative,checksum in report['artifacts'].items():
        source=(path.parent/relative).resolve()
        if not source.is_relative_to(path.parent) or not source.is_file() or digest(source)!=checksum:
            raise ValueError('Campaign product changed: '+relative)
    for source,checksum in report['source_hashes'].items():
        if digest(source)!=checksum:raise ValueError('Campaign source changed: '+source)
    for branch in report['branches']:
        cfg=yaml.safe_load(_checked(branch['configuration']).read_text())
        if (cfg['simulation']['start_redshift']!=branch['start_redshift'] or cfg['simulation']['random_seed']!=branch['seed']
                or canonical_digest(_physics_payload(cfg))!=report['physics_fingerprint']):
            raise ValueError('Branch changed start, seed or unrelated physics')
        phase=report['phases'][str(branch['seed'])]
        if branch['phase_id']!=phase['phase_id'] or _checked(branch['noise'])!=Path(phase['noise_path']):
            raise ValueError('Branch phase realization changed')
        bundle=validate_linear_bundle(branch['bundle']['manifest']['path'],redshift=branch['start_redshift'],
            box_mpc_h=cfg['simulation']['box_mpc_h'],refinement_level=report['grid_level'])
        if bundle!=branch['bundle']:raise ValueError('Exact-epoch bundle changed')
    if digest(path)!=before:raise RuntimeError('Campaign changed while reading')
    return report


def _branch(plan,identity):
    matches=[branch for branch in plan['branches'] if branch['id']==identity]
    if len(matches)!=1:raise ValueError('Unknown or ambiguous campaign branch')
    return matches[0]


def stage_start_epoch_branch(campaign, branch_id, destination, *, wdm_request=None):
    """Connect the frozen phase asset to the existing native MUSIC staging path."""
    from zoom import stage_zoom_inputs
    import configparser
    plan=verify_start_epoch_campaign(campaign);branch=_branch(plan,branch_id)
    cfg=yaml.safe_load(_checked(branch['configuration']).read_text());level=plan['grid_level']
    if wdm_request is not None and wdm_request.get('thermal_seed')!=branch['seed']:
        raise ValueError('WDM thermal seed must match the branch seed')
    if cfg['simulation']['physics']!='hydro' or cfg['simulation'].get('ic_engine_hydro','native_music_external')!='native_music_external':
        raise ValueError('Campaign native staging supports the explicitly selected two-fluid MUSIC route')
    if level<5:raise ValueError('Native MUSIC campaign staging requires a grid of at least 32 per dimension')
    destination=Path(destination).resolve()
    if destination.exists():raise FileExistsError('Keep previous branch staging')
    destination.mkdir(parents=True)
    staged=stage_zoom_inputs(branch['bundle']['manifest']['path'],destination/'native-inputs',
        box_mpc_h=cfg['simulation']['box_mpc_h'],parent_level=level,refinement_level=level,
        seeds_by_level={str(level):branch['seed']%(2**31)},padding_cells=2,poisson_accuracy=1e-14,
        lpt_order=cfg['simulation'].get('lpt_order_hydro',1),wdm_request=wdm_request,
        phase_noise_by_level={str(level):branch['noise']})
    # Preserve the sealed staging recipe. The separate run recipe changes only RNG input.
    recipe=configparser.ConfigParser(interpolation=None);recipe.optionxform=str
    recipe.read(destination/'native-inputs/parent.conf')
    recipe['random']['seed['+str(level)+']']=str(_checked(branch['noise']))
    recipe['random']['grafic_sign']='yes'
    from core.foundation import atomic_text
    import io
    stream=io.StringIO();recipe.write(stream)
    atomic_text(Path(destination)/'campaign-parent.conf',stream.getvalue())
    atomic_json(Path(destination)/'campaign-branch.json',{'campaign':_record(campaign),'branch_id':branch_id,
        'configuration':branch['configuration'],'phase_id':branch['phase_id'],'noise':branch['noise'],
        'native_configuration':_record(Path(destination)/'campaign-parent.conf'),'science_acceptance':'NOT_ACCEPTED'})
    return staged


def _snapshot_epoch(path,branch,cfg):
    with h5py.File(path) as source:
        a=np.asarray(source['Header'].attrs['Time'])
        if a.size!=1 or a.dtype.kind not in 'fiu':raise ValueError('Snapshot Time must be a numeric scalar')
        a=_number(float(a.reshape(-1)[0]),'actual snapshot scale factor',strict=True)
        params=source['Parameters'].attrs
        if not math.isclose(float(params['TimeBegin']),branch['start_scale_factor'],rel_tol=0,abs_tol=1e-12):
            raise ValueError('Snapshot integration start differs from branch')
        for name,value in (('HubbleParam',cfg['cosmology']['H0']/100),
                           ('OmegaBaryon',cfg['cosmology']['omega_b']/(cfg['cosmology']['H0']/100)**2)):
            if not math.isclose(float(params[name]),float(value),rel_tol=1e-12):
                raise ValueError('Snapshot changes branch cosmology: '+name)
        if 'Redshift' in source['Header'].attrs and not math.isclose(float(source['Header'].attrs['Redshift']),1/a-1,rel_tol=0,abs_tol=1e-8):
            raise ValueError('Snapshot redshift and actual Time disagree')
    return a


@directory_owner('destination')
def write_start_epoch_measurements(campaign, branch_id, snapshots, values, destination, *, ic_phase_audit, execution_receipt=None, execution_kind=None, measurement_source_hashes=None):
    """Archive actual-epoch observable vectors with native input and cost bindings.

    Each values[name] is (saved epochs, declared bins) in declared units. Values
    remain producer measurements; this function does not derive them from a
    snapshot or substitute requested epochs for actual Header/Time.
    """
    plan=verify_start_epoch_campaign(campaign);branch=_branch(plan,branch_id)
    cfg=yaml.safe_load(_checked(branch['configuration']).read_text())
    phase_report=verify_start_epoch_phase_audit(ic_phase_audit,campaign,branch_id)
    paths=[Path(p).resolve() for p in snapshots]
    if not paths or len(set(paths))!=len(paths):raise ValueError('Measurements require distinct saved snapshots')
    sources={str(Path(campaign).resolve()):digest(campaign),branch['configuration']['path']:branch['configuration']['sha256']}
    sources.update(phase_report['sources'])
    sources[str(Path(ic_phase_audit).resolve())]=digest(ic_phase_audit)
    if measurement_source_hashes is not None:
        if not isinstance(measurement_source_hashes,dict):raise ValueError('Measurement sources require an explicit path/hash mapping')
        for path,checksum in measurement_source_hashes.items():
            checked=_checked({'path':path,'sha256':checksum})
            sources[str(checked)]=checksum
    epochs=[]
    for path in paths:
        sources[str(path)]=digest(path);epochs.append(_snapshot_epoch(path,branch,cfg))
    if any(b<=a for a,b in zip(epochs,epochs[1:])):raise ValueError('Actual saved epochs must strictly increase')
    if min(epochs)<=branch['start_scale_factor'] or max(epochs)>1/(1+cfg['simulation']['end_redshift'])+plan['request']['epoch_tolerance']:
        raise ValueError('Measured epochs lie outside the branch integration interval')
    rules=plan['request']['observables']
    if not isinstance(values,dict) or set(values)!=set(rules):raise ValueError('Measurements must cover exactly the declared observables')
    arrays={'actual_scale_factors':np.asarray(epochs)}
    specs={'actual_scale_factors':{'units':'dimensionless','quantity':'actual Header/Time, never storage index'}}
    for name,rule in rules.items():
        array=np.asarray(values[name])
        if array.shape!=(len(paths),len(rule['bin_labels'])) or array.dtype.kind not in 'fiu' or not np.isfinite(array).all():
            raise ValueError('Observable requires finite aligned epoch/bin vectors: '+name)
        arrays['observable_'+name]=array.astype(float)
        specs['observable_'+name]={'units':rule['units'],'quantity':'producer measurement: '+name,'annotations':{'bin_labels':rule['bin_labels'],'axis_labels':['actual_epoch','bin']}}
    cost={'status':'UNAVAILABLE','reason':'No measured native execution receipt supplied'}
    if execution_receipt is not None:
        if execution_kind not in {'evolution','ic_generation','numerical_control'}:
            raise ValueError('Explicitly classify the measured invocation as evolution, ic_generation or numerical_control')
        from core.native import verify_invocation_inputs
        execution=Path(execution_receipt).resolve();sources[str(execution)]=digest(execution)
        record=json.loads(execution.read_text())
        if record.get('schema')!='native-command-invocation-v1' or record.get('status')!='RETURNED' or record.get('returncode')!=0:
            raise ValueError('Costs require a completed successful native invocation')
        if record['input_hashes'].get(branch['configuration']['path'])!=branch['configuration']['sha256']:
            raise ValueError('Native invocation must bind this exact branch configuration')
        binding=record.get('campaign_branch')
        if binding is not None and binding!={'campaign':_record(campaign),'branch_id':branch_id,
                'phase_id':branch['phase_id'],'execution_kind':execution_kind}:
            raise ValueError('Native cost receipt changes branch, phase or execution kind')
        verify_invocation_inputs(record)
        begin=datetime.datetime.fromisoformat(record['recorded_utc']);end=datetime.datetime.fromisoformat(record['finished_utc'])
        if begin.tzinfo is None or end.tzinfo is None or end<begin:raise ValueError('Native cost timestamps are invalid')
        sources.update(record['input_hashes'])
        cost={'status':'MEASURED','execution_kind':execution_kind,'native_wall_seconds':(end-begin).total_seconds(),
              'definition':'whole retained invocation duration; not an interval-matched transient metric',
              'receipt':_record(execution)}
    report={'schema':'start-epoch-measurements-v1','status':'RECORDED','campaign':_record(campaign),
        'branch_id':branch_id,'configuration':branch['configuration'],'phase_id':branch['phase_id'],
        'ic_phase_audit':_record(ic_phase_audit),'seed':branch['seed'],'start_redshift':branch['start_redshift'],'actual_scale_factors':epochs,
        'snapshots':[_record(p) for p in paths],'observable_rules':rules,'cost':cost,
        'sources':sources,'synthetic_inputs':plan['synthetic_inputs'],'science_acceptance':'NOT_ACCEPTED'}
    return export_numeric_audit(destination,report,arrays,specs,algorithm=__file__,source_hashes=sources,
        context={'definition':'Matched start-epoch producer measurements with actual saved epochs',
            'epoch':{'schema':CANONICAL_SCHEMA,'representation':'actual per-row scale-factor epochs',
                     'axis':0,'rows':[epoch_contract(a) for a in epochs]}})


def _read_measurement(path,plan,campaign):
    report=verify_numeric_audit(path)
    if report.get('schema')!='start-epoch-measurements-v1' or report['campaign']!=_record(campaign):
        raise ValueError('Measurements belong to another campaign')
    branch=_branch(plan,report['branch_id']);cfg=yaml.safe_load(_checked(branch['configuration']).read_text())
    if report['configuration']!=branch['configuration'] or report['phase_id']!=branch['phase_id'] or report['observable_rules']!=plan['request']['observables']:
        raise ValueError('Measurement branch/phase/observable contract changed')
    for source,checksum in report['sources'].items():
        if digest(source)!=checksum:raise ValueError('Measurement source changed: '+source)
    verify_start_epoch_phase_audit(_checked(report['ic_phase_audit']),campaign,branch['id'])
    epochs=[_snapshot_epoch(_checked(p),branch,cfg) for p in report['snapshots']]
    if epochs!=report['actual_scale_factors']:raise ValueError('Measurement actual epochs changed')
    with np.load(str(Path(path).resolve())+'.data/arrays.npz',allow_pickle=False) as archive:
        arrays={key:archive[key] for key in archive.files}
    if set(arrays)!={'actual_scale_factors',*('observable_'+name for name in plan['request']['observables'])}:
        raise ValueError('Measurement archive changes observable inventory')
    for name,rule in plan['request']['observables'].items():
        values=arrays['observable_'+name]
        if values.shape!=(len(epochs),len(rule['bin_labels'])) or values.dtype.kind not in 'fiu' or not np.isfinite(values).all():
            raise ValueError('Measurement archive changes observable shape or values')
    if not np.array_equal(arrays['actual_scale_factors'],epochs):raise ValueError('Archived epochs differ from native snapshots')
    return report,arrays


@directory_owner('destination')
def compare_start_epoch_campaign(campaign, measurements, destination, *, memory_gib=1.):
    """Pair seeds at common actual epochs; retain residuals, covariance and decisions."""
    plan=verify_start_epoch_campaign(campaign);request=plan['request']
    if not isinstance(measurements,list) or len(measurements)!=len(plan['branches']):
        raise ValueError('One measurement archive is required for every campaign branch')
    # Dense covariance workspace is budgeted before loading any observable arrays.
    dimension=sum(len(rule['bin_labels']) for rule in request['observables'].values())*len(request['common_scale_factors'])*(len(request['redshifts'])-1)
    if 32*dimension*dimension+64*1024**2>_number(memory_gib,'comparison memory',strict=True)*1024**3:
        raise MemoryError('Paired covariance exceeds declared comparison memory budget')
    loaded={};sources={str(Path(campaign).resolve()):digest(campaign)}
    for path in measurements:
        path=Path(path).resolve();sources[str(path)]=digest(path)
        report,arrays=_read_measurement(path,plan,campaign)
        if report['branch_id'] in loaded:raise ValueError('Duplicate branch measurements')
        loaded[report['branch_id']]=(report,arrays)
        sources.update(report['sources'])
        for asset in Path(str(path)+'.data').rglob('*'):
            if asset.is_file():sources[str(asset)]=digest(asset)
    if set(loaded)!=set(b['id'] for b in plan['branches']):raise ValueError('Missing campaign branch measurements')
    expected=request['common_scale_factors'];tolerance=request['epoch_tolerance']
    aligned={};costs=[]
    for branch in plan['branches']:
        report,arrays=loaded[branch['id']];epochs=np.asarray(report['actual_scale_factors'])
        indices=[]
        for a in expected:
            match=np.flatnonzero(np.abs(epochs-a)<=tolerance)
            if len(match)!=1:raise ValueError('Common actual epoch missing or ambiguous; interpolation is not authorized')
            indices.append(int(match[0]))
        aligned[(branch['start_redshift'],branch['seed'])]={name:arrays['observable_'+name][indices] for name in request['observables']}
        costs.append({'branch_id':branch['id'],'start_scale_factor':branch['start_scale_factor'],
            'common_actual_scale_factors':epochs[indices].tolist(),
            'integration_intervals_ln_a':np.log(epochs[indices]/branch['start_scale_factor']).tolist(),
            'cost':report['cost'],'scope':'unequal integration intervals/costs retained separately from observable residuals'})
    zs=[float(z) for z in request['redshifts'] if z!=request['reference_redshift']]
    seeds=request['seeds'];arrays={};specs={};decisions={};recommendations=[]
    arrays['common_scale_factors']=np.asarray(expected);arrays['comparison_redshifts']=np.asarray(zs)
    arrays['independent_seeds']=np.asarray(seeds,dtype=np.uint64)
    for key in arrays:specs[key]={'units':'dimensionless','quantity':'declared matched campaign coordinate: '+key}
    for name,rule in request['observables'].items():
        reference=np.stack([aligned[(request['reference_redshift'],s)][name] for s in seeds],axis=1)
        candidate=np.stack([np.stack([aligned[(z,s)][name] for s in seeds],axis=1) for z in zs])
        residual=candidate-reference[None,...]
        # Ordering: start, actual epoch, independent realization, bin.
        mean=residual.mean(axis=2)
        paired=residual.transpose(2,0,1,3).reshape(len(seeds),-1)
        covariance=None;sem=np.zeros_like(mean)
        if len(seeds)>1:
            covariance=np.atleast_2d(np.cov(paired,rowvar=False,ddof=1))
            sem=np.sqrt(np.maximum(np.diag(covariance),0)/len(seeds)).reshape(mean.shape)
        bound=rule['absolute_tolerance']+rule['relative_tolerance']*np.abs(reference.mean(axis=1))[None,...]
        upper=np.abs(mean)+request['confidence_sigma']*sem
        passed=upper<=bound
        status=('PASS' if passed.all() else 'FAIL') if covariance is not None else 'INCOMPLETE'
        decisions[name]={'status':status,'passed_bins':int(passed.sum()),'tested_bins':int(passed.size),
            'covariance_status':'ESTIMATED_FROM_PAIRED_INDEPENDENT_SEEDS' if covariance is not None else 'UNAVAILABLE_SINGLE_REALIZATION',
            'confidence_sigma':request['confidence_sigma'],'bin_labels':rule['bin_labels'],
            'definition':'absolute mean paired residual plus declared sigma times SEM versus absolute+relative tolerance',
            'reference_role':'explicit numerical comparator; no claim that an earlier start is intrinsically more accurate'}
        for suffix,value in (('reference',reference),('candidate',candidate),('paired_residual',residual),
                             ('mean_residual',mean),('standard_error',sem),('tolerance_bound',bound),('confidence_upper',upper)):
            key=name+'_'+suffix;arrays[key]=value
            specs[key]={'units':rule['units'],'quantity':suffix+' at common actual epochs and matched seeds'}
        key=name+'_relative_residual';arrays[key]=residual/np.maximum(np.abs(reference)[None,...],rule['absolute_floor'])
        specs[key]={'units':'dimensionless','quantity':'paired residual with explicitly declared denominator floor'}
        if covariance is not None:
            for suffix,value in (('paired_covariance',covariance),('mean_covariance',covariance/len(seeds))):
                key=name+'_'+suffix;arrays[key]=value
                specs[key]={'units':'('+rule['units']+')^2','quantity':'covariance over independent paired seeds; axes flattened start, epoch, bin'}
        if status!='PASS':recommendations.append({'observable':name,'action':
            'Add independent matched seeds before a covariance or convergence claim' if covariance is None else
            'Add user-selected starts and separately vary LPT order, integration accuracy and resolution to identify the transient; retain shared noise',
            'automatic_physics_changes':False})
    if any(c['cost']['status']!='MEASURED' for c in costs):
        recommendations.append({'action':'Retain completed native invocation receipts to measure integration costs','automatic_physics_changes':False})
    status='FAIL' if any(v['status']=='FAIL' for v in decisions.values()) else ('INCOMPLETE' if any(v['status']=='INCOMPLETE' for v in decisions.values()) else 'PASS')
    report={'schema':'start-epoch-comparison-v1','status':status,'campaign':_record(campaign),
        'request_sha256':plan['request_sha256'],'physics_fingerprint':plan['physics_fingerprint'],
        'decisions':decisions,'costs':costs,'recommended_refinements':recommendations,
        'common_scale_factors':expected,'reference_redshift':request['reference_redshift'],
        'seed_count':len(seeds),'synthetic_inputs':plan['synthetic_inputs'],'sources':sources,
        'scope':'matched observable differences; intervals and runtime reported separately; no causal attribution from cost',
        'science_acceptance':'NOT_ACCEPTED'}
    return export_numeric_audit(destination,report,arrays,specs,algorithm=__file__,source_hashes=sources,
        context={'definition':'Matched start-epoch residuals and paired covariance; start/epoch/bin axis order',
            'epoch':{'schema':CANONICAL_SCHEMA,'representation':'matched actual epochs by branch; requested values are alignment coordinates',
                'alignment_tolerance':tolerance,'rows_by_branch':{item['branch_id']:
                    [epoch_contract(actual,requested_scale_factor=target) for actual,target in
                     zip(item['common_actual_scale_factors'],expected)] for item in costs}}})


def verify_start_epoch_comparison(record, request):
    """Acceptance intake: check sealed measurements, source freshness and request."""
    path=_checked(record);report=verify_numeric_audit(path)
    if report.get('schema')!='start-epoch-comparison-v1' or report.get('request_sha256')!=canonical_digest(request):
        raise ValueError('Start-epoch comparison is for another specification')
    plan=verify_start_epoch_campaign(_checked(report['campaign']))
    if plan['request']!=request or report['physics_fingerprint']!=plan['physics_fingerprint']:
        raise ValueError('Start-epoch comparison changed matched physics')
    for source,checksum in report['sources'].items():
        if digest(source)!=checksum:raise ValueError('Start-epoch comparison source changed: '+source)
    if report['status']!='PASS' or report['seed_count']<2:
        raise ValueError('Start-epoch convergence has not passed with paired covariance')
    if any(item['cost']['status']!='MEASURED' or item['cost'].get('execution_kind')!='evolution' for item in report['costs']):
        raise ValueError('Start-epoch scientific intake requires measured evolution costs')
    if report['synthetic_inputs']:raise ValueError('Synthetic numerical controls cannot grant scientific convergence acceptance')
    return {'status':'PASS','report':record,'scope':report['scope']}


@directory_owner('destination')
def audit_start_epoch_phases(campaign, realizations, destination, *, modes, maximum_phase_error=.05, memory_gib=1.):
    """Measure start-epoch phases and native q/IDs, allowing physical amplitudes to vary.

    Each record names branch_id, ic and native_noise. The native_noise is the
    actual MUSIC wnoise cache; it must be bitwise equal across starts of a seed.
    The source GRAFIC asset is checked separately by the prepared campaign.
    """
    from zoom import linear_density_modes
    from core.data import hdf5_numeric_payload_sha256
    plan=verify_start_epoch_campaign(campaign)
    limit=_number(maximum_phase_error,'maximum phase error',strict=True)
    if limit>np.pi:raise ValueError('Phase tolerance exceeds pi')
    if not isinstance(realizations,list) or len(realizations)!=len(plan['branches']):
        raise ValueError('Phase audit requires every actual native IC realization')
    epochs_by_branch={}
    fields={};identities={};noises={};sources={str(Path(campaign).resolve()):digest(campaign)}
    for item in realizations:
        if not isinstance(item,dict) or set(item)!={'branch_id','ic','native_noise'}:
            raise ValueError('Native phase realization requires branch_id, ic and native_noise')
        branch=_branch(plan,item['branch_id'])
        if branch['id'] in fields:raise ValueError('Duplicate native phase branch')
        ic=Path(item['ic']).resolve();noise=Path(item['native_noise']).resolve()
        sources[str(ic)]=digest(ic);sources[str(noise)]=digest(noise)
        with noise.open('rb') as stream:
            header=stream.read(12)
            if len(header)!=12:raise ValueError('Truncated native noise cache')
            shape=struct.unpack('<3I',header)
            n=2**plan['grid_level']
            if shape!=(n,n,n) or noise.stat().st_size!=12+8*n**3:
                raise ValueError('Native noise cache differs from the frozen grid')
            while block:=stream.read(8*65536):
                if not np.isfinite(np.frombuffer(block,dtype='<f8')).all():
                    raise ValueError('Native noise cache contains nonfinite values')
        cfg=yaml.safe_load(_checked(branch['configuration']).read_text())
        with h5py.File(ic) as source:
            header=source['Header'].attrs
            if not math.isclose(float(header['Time']),branch['start_scale_factor'],rel_tol=0,abs_tol=1e-12):
                raise ValueError('Native IC epoch differs from the selected branch')
            if not math.isclose(float(header['BoxSize']),cfg['simulation']['box_mpc_h']*1000,rel_tol=1e-12):
                raise ValueError('Native IC changes the campaign box')
            if int(header.get('Music_lpt_order',0))!=cfg['simulation'].get('lpt_order_hydro',1):
                raise ValueError('Native IC changes the declared LPT order')
            epochs_by_branch[branch['id']]=epoch_contract(float(header['Time']),requested_scale_factor=branch['start_scale_factor'])
            h=cfg['cosmology']['H0']/100
            for key,value in (('HubbleParam',h),('Omega0',(cfg['cosmology']['omega_b']+cfg['cosmology']['omega_cdm'])/h**2),
                              ('OmegaBaryon',cfg['cosmology']['omega_b']/h**2)):
                if not math.isclose(float(header[key]),value,rel_tol=1e-12):
                    raise ValueError('Native phase IC changes cosmological frame: '+key)
            identity={}
            for t in (0,1,2):
                group=source.get('PartType'+str(t))
                if group is None:continue
                if group['ParticleIDs'].dtype!=np.uint64 or group['LagrangianCoordinates'].dtype!=np.float64:
                    raise ValueError('Phase proof requires native uint64 IDs and float64 q')
                for field in ('ParticleIDs','LagrangianCoordinates'):
                    identity[str(t)+'/'+field]=hdf5_numeric_payload_sha256(group[field])
        fields[branch['id']]=linear_density_modes(ic,modes,memory_gib=memory_gib)
        identities[branch['id']]=identity;noises[branch['id']]=sources[str(noise)]
    cache_ids={noises[next(b['id'] for b in plan['branches'] if b['seed']==seed)] for seed in plan['request']['seeds']}
    if len(cache_ids)!=len(plan['request']['seeds']):
        raise ValueError('Independent seeds reuse the same native noise cache')
    arrays={'mode_vectors':np.asarray(modes,dtype=np.int64)}
    specs={'mode_vectors':{'units':'dimensionless','quantity':'integer Fourier wavevectors'}}
    checks={}
    for seed in plan['request']['seeds']:
        branches=[b for b in plan['branches'] if b['seed']==seed]
        reference=next(b for b in branches if b['start_redshift']==plan['request']['reference_redshift'])
        ref=fields[reference['id']]
        if np.any(np.abs(ref)<=1e-30):raise ValueError('Phase reference has zero measured modes')
        for branch in branches:
            value=fields[branch['id']]
            if np.any(np.abs(value)<=1e-30):raise ValueError('Native IC has zero measured phase modes')
            phase=np.angle(value*np.conj(ref));error=float(np.max(np.abs(phase)))
            equal=identities[branch['id']]==identities[reference['id']] and noises[branch['id']]==noises[reference['id']]
            checks[branch['id']]={'status':'PASS' if equal and error<=limit else 'FAIL',
                'native_noise_and_q_ids_equal':equal,'maximum_phase_error_rad':error}
            arrays[branch['id']+'_density']=value
            specs[branch['id']+'_density']={'units':'dimensionless','quantity':'native linear total-matter complex density'}
    report={'schema':'start-epoch-phases-v1','status':'PASS' if all(c['status']=='PASS' for c in checks.values()) else 'FAIL',
        'campaign':_record(campaign),'checks':checks,'sources':sources,'maximum_phase_error_rad':limit,
        'scope':'native shared-noise equality and original q/ID equality; selected large-scale density phases, amplitudes may evolve',
        'science_acceptance':'NOT_ACCEPTED'}
    return export_numeric_audit(destination,report,arrays,specs,algorithm=__file__,source_hashes=sources,
        context={'definition':'Actual native IC start-epoch phase invariance',
            'epoch':{'schema':CANONICAL_SCHEMA,'representation':'native initial-condition epochs by branch',
                     'epochs_by_branch':epochs_by_branch}})


def verify_start_epoch_phase_audit(path,campaign,branch_id):
    report=verify_numeric_audit(path)
    if (report.get('schema')!='start-epoch-phases-v1' or report['campaign']!=_record(campaign)
            or report['status']!='PASS' or report.get('checks',{}).get(branch_id,{}).get('status')!='PASS'):
        raise ValueError('Native start-epoch phase invariance has not passed')
    for source,checksum in report['sources'].items():
        if digest(source)!=checksum:raise ValueError('Native phase proof source changed: '+source)
    return report


@directory_owner('run_directory')
def execute_start_epoch_branch(campaign, branch_id, command, run_directory, *, execution_kind, receipt_name='execution.json'):
    """Track a prepared native command with the exact branch config and noise.

    The caller prepares the native evolution inputs through the existing run
    pipeline. This wrapper executes an argv vector, not a shell script, and
    labels IC generation/control time separately from evolution time.
    """
    from core.native import run_native_command
    plan=verify_start_epoch_campaign(campaign);branch=_branch(plan,branch_id)
    if execution_kind not in {'evolution','ic_generation','numerical_control'}:
        raise ValueError('Declare the actual execution kind')
    if not isinstance(command,list) or not command or any(not isinstance(part,str) or not part for part in command):
        raise ValueError('Native branch command requires a nonempty argv vector')
    directory=Path(run_directory).resolve()
    if not directory.is_dir():raise ValueError('Prepare the native run directory before execution')
    path=(directory/receipt_name).resolve()
    if not path.is_relative_to(directory) or path.exists():raise ValueError('Use a fresh receipt within the owned run directory')
    spec=json.loads(_checked(branch['research_spec']).read_text())
    run_native_command(command,cwd=directory,receipt_path=path,
        inputs=[_checked(branch['configuration']),_checked(branch['noise']),Path(campaign).resolve()],
        timeout=spec['resource_budget']['wall_hours']*3600)
    record=json.loads(path.read_text());record['campaign_branch']={'campaign':_record(campaign),
        'branch_id':branch_id,'phase_id':branch['phase_id'],'execution_kind':execution_kind}
    atomic_json(path,record);verify_start_epoch_campaign(campaign)
    return record


# Snapshot density modes
#
# Bounded, mass-weighted periodic Fourier measurements from saved snapshots.

def measure_periodic_density_modes(files, modes, *, species=(0, 1), memory_gib=1.):
    """Return V|delta_k|^2 at exact nonzero reciprocal-box wavevectors.

    This is the discrete point-particle estimator: no mesh window, shot-noise
    subtraction, interpolation or inference of simulation convergence is used.
    Coordinates and BoxSize share native units; the declared native length unit
    supplies the comoving Mpc/h volume. All selected species contribute by mass.
    """
    wave=np.asarray(modes)
    if (wave.ndim!=2 or wave.shape[1]!=3 or not 1<=len(wave)<=64
            or wave.dtype.kind not in 'iu' or np.any(np.abs(wave.astype(float))>2**20)
            or np.any(np.all(wave==0,axis=1)) or len(np.unique(wave,axis=0))!=len(wave)):
        raise ValueError('Declare 1–64 distinct nonzero integer reciprocal-box modes')
    if (not species or len(set(species))!=len(species)
            or any(type(p) is not int or not 0<=p<6 for p in species)):
        raise ValueError('Select distinct exact native particle types')
    if isinstance(memory_gib,bool) or not math.isfinite(memory_gib) or memory_gib<=.05:
        raise MemoryError('Density-mode workspace requires more than 0.05 GiB')
    paths=[Path(p).resolve() for p in files]
    publication=verify_native_hdf5_set(paths)
    before={str(p):file_sha256(p) for p in paths}
    # Phase matrix, complex exponential, weighted product and particle buffers.
    chunk=max(1,min(65536,int((memory_gib*1024**3-32*1024**2)/(64*len(wave)+128))))
    amplitudes=np.zeros(len(wave),dtype=complex)
    total_mass=0.;squared_mass=0.;particle_count=0;baseline=None
    count_sum=np.zeros(6,dtype=object)
    for path in paths:
        with h5py.File(path,'r') as source:
            header=source['Header'].attrs;params=source['Parameters'].attrs
            if params['ComovingIntegrationOn']!=1:
                raise ValueError('Periodic density modes require cosmological coordinates')
            a=float(header['Time']);box=float(header['BoxSize'])
            length=float(params['UnitLength_in_cm']);h=float(params['HubbleParam'])
            mass_unit=float(params['UnitMass_in_g'])
            if not all(math.isfinite(v) and v>0 for v in (a,box,length,h,mass_unit)) or a>1:
                raise ValueError('Invalid native epoch, box or length/Hubble unit metadata')
            counts=np.asarray(header['NumPart_ThisFile'])
            totals=np.asarray(header['NumPart_Total'])
            if counts.shape!=(6,) or totals.shape!=(6,) or counts.dtype.kind not in 'iu' or totals.dtype.kind not in 'iu' or np.any(counts<0) or np.any(totals<0):
                raise ValueError('Native particle counts require six exact nonnegative integers')
            if 'NumPart_Total_HighWord' in header:
                high=np.asarray(header['NumPart_Total_HighWord'])
                if high.shape!=(6,) or high.dtype.kind not in 'iu' or np.any(high<0) or np.any(high>2**32-1):
                    raise ValueError('Invalid native particle count high words')
                if np.any(high>0) and np.any(totals>2**32-1):
                    raise ValueError('Native count high words require uint32 low words')
                totals=np.asarray([int(lo)+(int(hi)<<32) for lo,hi in zip(totals,high)],dtype=object)
            mass_table=np.asarray(header['MassTable'],dtype=float)
            if mass_table.shape!=(6,) or not np.isfinite(mass_table).all() or np.any(mass_table<0):
                raise ValueError('Invalid native constant-mass table')
            signature=(a,box,length,h,tuple(map(int,totals)),mass_unit,tuple(mass_table))
            if baseline is None:baseline=signature
            elif signature!=baseline:raise ValueError('Native density-mode shards disagree on epoch, box, units or totals')
            if int(header['NumFilesPerSnapshot'])!=len(paths):
                raise ValueError('Density-mode snapshot is missing native shards')
            count_sum+=counts.astype(object)
            for ptype in species:
                count=int(counts[ptype]);name='PartType'+str(ptype)
                if not count:continue
                group=source[name];positions=group['Coordinates']
                if positions.shape!=(count,3) or positions.dtype.kind not in 'fiu':
                    raise ValueError('Native coordinates do not match particle counts')
                for key,expected in (('to_cgs',length),('a_scaling',1.),('h_scaling',-1.)):
                    if key not in positions.attrs or not math.isclose(float(positions.attrs[key]),expected,rel_tol=1e-12,abs_tol=0):
                        raise ValueError('Density modes require declared native comoving coordinate units')
                variable=mass_table[ptype]==0
                if variable and (group['Masses'].shape!=(count,) or group['Masses'].dtype.kind not in 'fiu'):
                    raise ValueError('Native variable masses do not match particle counts')
                if variable:
                    for key,expected in (('to_cgs',mass_unit),('a_scaling',0.),('h_scaling',-1.)):
                        if key not in group['Masses'].attrs or not math.isclose(float(group['Masses'].attrs[key]),expected,rel_tol=1e-12,abs_tol=0):
                            raise ValueError('Density modes require declared native mass units')
                for start in range(0,count,chunk):
                    xyz=np.asarray(positions[start:start+chunk],dtype=float)
                    mass=(np.asarray(group['Masses'][start:start+chunk],dtype=float)
                          if variable else np.full(len(xyz),mass_table[ptype]))
                    if not np.isfinite(xyz).all() or not np.isfinite(mass).all() or np.any(mass<=0):
                        raise ValueError('Density modes require finite coordinates and positive mass')
                    phase=2*np.pi*((xyz%box)/box)@wave.astype(float).T
                    amplitudes+=np.sum(mass[:,None]*np.exp(-1j*phase),axis=0)
                    total_mass+=float(mass.sum());squared_mass+=float(mass@mass);particle_count+=len(mass)
    if tuple(map(int,count_sum))!=baseline[4]:
        raise ValueError('Native shard particle census does not match global totals')
    if not math.isfinite(total_mass) or total_mass<=0 or not math.isfinite(squared_mass):
        raise ValueError('Density-mode selection has no finite positive mass')
    if any(file_sha256(path)!=checksum for path,checksum in before.items()):
        raise ValueError('Saved snapshot changed during density-mode measurement')
    delta=amplitudes/total_mass;box_mpc_h=baseline[1]*baseline[2]/MPC_CM
    power=box_mpc_h**3*np.abs(delta)**2
    if not np.isfinite(power).all():raise ValueError('Nonfinite density-mode power')
    return {'schema':'periodic-density-modes-v1','status':'MEASURED',
        'scale_factor':baseline[0],'box_mpc_h':box_mpc_h,'species':list(species),
        'modes':wave.tolist(),'delta_real':delta.real.tolist(),'delta_imaginary':delta.imag.tolist(),
        'power_mpc_h3':power.tolist(),'particle_count':particle_count,
        'total_mass_native':total_mass,'poisson_reference_mpc_h3':box_mpc_h**3*squared_mass/total_mass**2,
        'window':'direct point-particle Fourier sum; window=1',
        'shot_noise_subtracted':False,'poisson_reference_scope':'reference for independent Poisson points; not an inferred correction for lattice ICs',
        'source_hashes':before,'native_publication':publication,'algorithm_sha256':file_sha256(__file__),
        'science_acceptance':'NOT_ACCEPTED'}


@directory_owner('destination')
def write_start_epoch_analysis_measurements(campaign, branch_id, products, destination, *,
                                          ic_phase_audit, execution_receipt=None, execution_kind=None,
                                          memory_gib=1.):
    """Connect saved analysis producers to matched start-epoch comparisons.

    Each observable selects a measured vector and its numeric bin-coordinate
    vector from a sealed analysis bundle at each complete native epoch. Labels
    are canonical .17g coordinates; no user-supplied values or interpolation are
    used. P(k), HMF, concentration and later measured products share this intake.
    Missing source physics or failed/empty bins remain explicit errors.
    """
    from analysis import verify_analysis_bundle
    from core.data import verify_dataset_semantics
    plan = verify_start_epoch_campaign(campaign); branch = _branch(plan, branch_id)
    cfg = yaml.safe_load(_checked(branch['configuration']).read_text())
    rules = plan['request']['observables']
    if not isinstance(products, dict) or set(products) != set(rules):
        raise ValueError('Producer products must cover exactly the requested campaign observables')
    budget = _number(memory_gib, 'producer measurement memory', strict=True)*1024**3
    sources = {str(Path(__file__).resolve()): digest(__file__)}; values = {}; reference_sets = None
    total_bytes = 0; verified = {}
    for name, rows in products.items():
        if not isinstance(rows, list) or not rows: raise ValueError('Every campaign observable needs ordered producer epochs')
        measured = []; selected_sets = []
        for row in rows:
            if not isinstance(row, dict) or set(row) != {'directory', 'file', 'dataset', 'coordinate_dataset', 'snapshot_set'}:
                raise ValueError('Producer selection requires directory, file, measured dataset, coordinates and complete native snapshot set')
            directory = Path(row['directory']).resolve()
            if directory not in verified: verified[directory] = verify_analysis_bundle(directory, verify_sources=True)
            receipt = verified[directory]; run = Path(receipt['source_run']).resolve()
            actual_cfg = yaml.safe_load((run/'config.yaml').read_text())
            if (_physics_payload(actual_cfg) != _physics_payload(cfg)
                    or actual_cfg['simulation']['random_seed'] != branch['seed']
                    or actual_cfg['simulation']['start_redshift'] != branch['start_redshift']):
                raise ValueError('Analysis producer changes matched branch physics, start epoch or seed')
            if not plan['synthetic_inputs'] and not receipt['data_contract'].get('initial_conditions'):
                raise ValueError('Native campaign producer lacks verified IC lineage')
            snapshot_set = [Path(p).resolve() for p in row['snapshot_set']]
            verify_native_hdf5_set(snapshot_set)
            if not snapshot_set: raise ValueError('Producer requires a complete nonempty native epoch set')
            source_inventory = {entry['path']: entry['sha256'] for entry in receipt['data_contract']['input_hashes']}
            for snapshot in snapshot_set:
                if source_inventory.get(str(snapshot)) != digest(snapshot):
                    raise ValueError('Selected native epoch is not an input of this measured producer')
                sources[str(snapshot)] = digest(snapshot)
            actual_epochs = [_snapshot_epoch(snapshot, branch, cfg) for snapshot in snapshot_set]
            if len(set(actual_epochs)) != 1: raise ValueError('Producer native shards differ in actual epochs')
            actual_epoch = actual_epochs[0]
            selected_sets.append([str(p) for p in snapshot_set])
            file = (directory/row['file']).resolve()
            artifacts = {(directory/entry['relative_path']).resolve(): entry['sha256'] for entry in receipt['output_artifacts']}
            if not file.is_relative_to(directory) or file not in artifacts or file.suffix not in {'.h5', '.hdf5'}:
                raise ValueError('Campaign measurement must select a sealed HDF5 analysis artifact')
            with h5py.File(file, 'r') as handle:
                data = handle[row['dataset']]; coordinate = handle[row['coordinate_dataset']]
                for dataset in (data, coordinate):
                    verify_dataset_semantics(dataset)
                    epoch = json.loads(dataset.attrs['epoch_json'])
                    if epoch.get('actual_scale_factor') != actual_epoch:
                        raise ValueError('Producer measurement and coordinates differ from the actual native epoch')
                    parent = dataset.parent
                    while parent.name != '/':
                        if 'status' in parent.attrs and parent.attrs['status'] not in {'COMPLETE', 'MEASURED', 'PASS'}:
                            raise ValueError('Failed or unresolved producer bins cannot enter a start-epoch comparison')
                        parent = parent.parent
                if data.attrs['units'] != rules[name]['units'] or data.dtype.kind not in 'fiu' or coordinate.dtype.kind not in 'fiu':
                    raise ValueError('Campaign producer units or numeric dtype mismatch')
                if data.shape != (len(rules[name]['bin_labels']),) or coordinate.shape != data.shape:
                    raise ValueError('Campaign producer requires aligned nonempty observable/bin vectors')
                total_bytes += data.size*8+coordinate.size*8
                if total_bytes+32*1024**2 > budget: raise MemoryError('Campaign producer vectors exceed the declared memory budget')
                vector = np.asarray(data[:], dtype=float); bins = np.asarray(coordinate[:], dtype=float)
                if not np.isfinite(vector).all() or not np.isfinite(bins).all():
                    raise ValueError('Empty or unresolved producer measurements cannot grant convergence')
                if [format(float(value), '.17g') for value in bins] != rules[name]['bin_labels']:
                    raise ValueError('Measured producer bin coordinates differ from the frozen campaign labels')
                measured.append(vector)
            for artifact, checksum in artifacts.items(): sources[str(artifact)] = checksum
            for entry in receipt['data_contract']['algorithm_sources']: sources[entry['path']] = entry['sha256']
            for file_name in ('analysis_provenance.json', 'analysis_provenance.json.sha256'):
                sources[str(directory/file_name)] = digest(directory/file_name)
        if reference_sets is not None and selected_sets != reference_sets:
            raise ValueError('Campaign observables must use the same complete actual snapshot sets')
        reference_sets = selected_sets; values[name] = np.asarray(measured)
    for directory in verified: verify_analysis_bundle(directory, verify_sources=True)
    return write_start_epoch_measurements(campaign, branch_id, [files[0] for files in reference_sets], values,
        destination, ic_phase_audit=ic_phase_audit, execution_receipt=execution_receipt,
        execution_kind=execution_kind, measurement_source_hashes=sources)


def write_start_epoch_density_modes(campaign, branch_id, snapshot_sets, modes, destination, *,
                                   ic_phase_audit, execution_receipt=None, execution_kind=None,
                                   species=(0,1), memory_gib=1.):
    """Measure complete native epoch sets and bind every shard to comparison.

    Kept measurement reports include the estimator, window and uncorrected
    Poisson reference. The common campaign writer supplies epochs, cost and
    covariance contracts; secondary shards cannot escape its source binding.
    """
    from core.foundation import atomic_json
    plan=verify_start_epoch_campaign(campaign)
    if branch_id not in {branch['id'] for branch in plan['branches']}:
        raise ValueError('Unknown start-epoch branch')
    import yaml
    branch=next(b for b in plan['branches'] if b['id']==branch_id)
    cfg=yaml.safe_load(Path(branch['configuration']['path']).read_text())
    rules=plan['request']['observables']
    labels=[','.join(map(str,m)) for m in modes]
    if set(rules)!={'density_modes'} or rules['density_modes']['units']!='(Mpc/h)^3' or rules['density_modes']['bin_labels']!=labels:
        raise ValueError('Measured modes must match the declared density_modes units and bins')
    if not isinstance(snapshot_sets,list) or not snapshot_sets or any(not isinstance(s,list) or not s for s in snapshot_sets):
        raise ValueError('Provide nonempty complete shard sets in actual epoch order')
    destination=Path(destination).resolve()
    if destination.exists():raise FileExistsError('Keep the previous campaign measurements')
    reports_directory=Path(str(destination)+'.density-modes');reports_directory.mkdir(parents=True,exist_ok=False)
    reports=[];sources={str(Path(__file__).resolve()):file_sha256(__file__)}
    for index,files in enumerate(snapshot_sets):
        report=measure_periodic_density_modes(files,modes,species=species,memory_gib=memory_gib)
        if not math.isclose(report['box_mpc_h'],float(cfg['simulation']['box_mpc_h']),rel_tol=1e-6):
            raise ValueError('Measured periodic box differs from the matched branch configuration')
        path=reports_directory/f'epoch-{index:04d}.json';atomic_json(path,report)
        sources.update(report['source_hashes']);sources[str(path)]=file_sha256(path);reports.append(report)
    return write_start_epoch_measurements(campaign,branch_id,[files[0] for files in snapshot_sets],
        {'density_modes':np.asarray([r['power_mpc_h3'] for r in reports])},destination,
        ic_phase_audit=ic_phase_audit,execution_receipt=execution_receipt,execution_kind=execution_kind,
        measurement_source_hashes=sources)


def native_source_event_candidates(report, arrays, *, starburst_minimum_locked_sfr_msun_per_year):
    """Connect committed source operators to physical event windows.

    Operator application times are measured. Continuous formation/merger times
    within an integration step remain bracketed; a journal is not substep clock
    evidence. Starbursts use an explicitly frozen interval-averaged net threshold.
    """
    from core.cosmology import run_proper_time_clock, interpolate_proper_time
    from core.data import proper_time_error_at_epoch
    minimum=number(starburst_minimum_locked_sfr_msun_per_year,'starburst threshold')
    root=Path(report['request']['run_directory']).resolve()
    scale,ages,clock=run_proper_time_clock(root)
    if scale is None:raise ValueError('Source cadence requires a pinned physical background clock')
    if not report['epochs'] or any(not e['parameters']['comoving'] for e in report['epochs']):raise ValueError('Current checkpoint event planner requires comoving epochs')
    def time(native):return interpolate_proper_time(scale,ages,native)*1000
    def bound(native):
        value=proper_time_error_at_epoch(clock,native).get('absolute_bound_gyr')
        return None if value is None else value*1000
    events=[]
    def emit(identity,kind,applied,left,right,details):
        # Never extrapolate or call an absent continuous event time exact.
        if not left<right or not left<=applied<=right:raise ValueError('Invalid native source operator clock bracket')
        error_left,error_right=bound(left),bound(right)
        bracket=[time(left)-(error_left or 0),time(right)+(error_right or 0)]
        details={**details,'native_application_time':applied,'native_integration_bracket':[left,right],
            'applied_time_convention':'operator state at synchronized native step end',
            'clock_error_myr':[error_left,error_right],
            'clock_error_status':'UNBOUNDED' if error_left is None or error_right is None else 'MODEL_CONDITIONAL_BOUND',
            'continuous_event_time':'UNRESOLVED_WITHIN_OPERATOR_STEP'}
        events.append(_candidate(identity,kind,time(applied),bracket,details))
    for ordinal,interval in enumerate(report['intervals']):
        before,after=report['epochs'][ordinal:ordinal+2]
        lo,hi=before['native_time'],after['native_time']
        for row in interval.get('black_hole_rows',[]):
            kind=row['kind'];identity='native-black-hole:'+str(row['particle_id'])
            left=max(lo,row.get('native_step_begin',lo));applied=row['native_time']
            if kind=='binary_capture':emit(identity,'bh_pair',applied,left,applied,{'method':'committed bound/hard native pair capture','counter':row['counter'],'companion_id':row['companion_id']})
            elif kind=='merger':
                details={'method':'committed private inspiral physical gate','counter':row['counter'],'parent_ids':row['parent_ids'],
                    'radiated_mass_msun':row['radiated_mass_msun'],'remnant_spin':row['spin']}
                emit(identity,'bh_merger',applied,left,applied,details)
                if np.linalg.norm(row['physical_recoil_cm_s'])>0:
                    emit(identity,'bh_recoil',applied,left,applied,{**details,'physical_recoil_cm_s':row['physical_recoil_cm_s']})
        key=f'interval_{ordinal:03d}_event_locked_mass'
        if key not in arrays:raise ValueError('Native source history lacks complete birth event masses')
        duration=(time(hi)-time(lo))*1.e6
        rate=float(np.sum(arrays[key]))/duration
        if rate>=minimum:
            emit('native-run:stellar-population','starburst',hi,lo,hi,{'method':'measured interval net SSP birth rate above frozen threshold',
                'locked_sfr_msun_per_year':rate,'minimum_locked_sfr_msun_per_year':minimum,
                'population_scope':'entire audited native run, not an inferred host galaxy'})
    return sorted({e['event_id']:e for e in events}.values(),key=lambda e:(e['predicted_time_myr'],e['event_id']))


def publish_native_source_events(request,destination):
    """Publish sealed source-event windows usable by the checkpoint planner."""
    from stars import verify_native_hydro_history
    from core.data import numeric_audit_arrays_path
    from core.foundation import atomic_json, atomic_text
    if not isinstance(request,dict) or set(request)!={'hydro_history','starburst_minimum_locked_sfr_msun_per_year','memory_gib'}:raise ValueError('Declare sealed native history, threshold and memory allowance')
    budget=number(request['memory_gib'],'memory_gib')*1024**3
    source=Path(request['hydro_history']).resolve();destination=Path(destination).resolve()
    if not destination.parent.is_dir() or destination.exists():raise FileExistsError('Use a new file in an existing directory for source events')
    report=verify_native_hydro_history(source);archive=numeric_audit_arrays_path(source,report)
    import zipfile
    with zipfile.ZipFile(archive) as z:
        if sum(i.file_size for i in z.infolist())*2+source.stat().st_size*6+32*1024**2>budget:raise MemoryError('Source event reconstruction exceeds declared memory')
    with np.load(archive,allow_pickle=False) as arrays:
        events=native_source_event_candidates(report,arrays,starburst_minimum_locked_sfr_msun_per_year=request['starburst_minimum_locked_sfr_msun_per_year'])
    sources={**report['source_hashes'],str(source):file_sha256(source),str(archive):file_sha256(archive),str(Path(__file__).resolve()):file_sha256(__file__)}
    root=Path(report['request']['run_directory']).resolve()
    output={'schema':'event-history-candidates-v1','run_dir':str(root),'population':'native_physical_sources',
        'histories':[],'events':events,'actual_output_times_myr':[e['epoch']['cosmic_time_gyr']*1000 for e in report['epochs']],
        'source_hashes':sources,'source_authority':request,'event_origin':'committed_native_operators',
        'science_acceptance':'NOT_ACCEPTED','cadence_acceptance':'UNMEASURED; candidate brackets retain operator timing and clock uncertainty'}
    if any(file_sha256(path)!=checksum for path,checksum in sources.items()):raise ValueError('Native event input changed during publication')
    atomic_json(destination,output);atomic_text(destination.with_name(destination.name+'.sha256'),file_sha256(destination)+'\n')
    return output


def reconstruct_published_source_events(product):
    """Replay source windows against the sealed physical source report."""
    from stars import verify_native_hydro_history
    from core.data import numeric_audit_arrays_path
    authority=product['source_authority'];report=verify_native_hydro_history(authority['hydro_history'])
    if str(Path(report['request']['run_directory']).resolve())!=product['run_dir']:raise ValueError('Native source event run identity changed')
    import zipfile
    archive=numeric_audit_arrays_path(authority['hydro_history'],report)
    with zipfile.ZipFile(archive) as z:
        if sum(i.file_size for i in z.infolist())*2+32*1024**2>number(authority['memory_gib'],'memory_gib')*1024**3:raise MemoryError('Source event replay exceeds workspace')
    with np.load(archive,allow_pickle=False) as arrays:
        return native_source_event_candidates(report,arrays,starburst_minimum_locked_sfr_msun_per_year=authority['starburst_minimum_locked_sfr_msun_per_year'])


def native_source_overlay_manifest(manifest_path,*,maximum_bytes=64*1024**2):
    """Read the pinned full-file overlay without applying patches or building."""
    import zipfile
    from pathlib import PurePosixPath
    path=Path(manifest_path).expanduser().resolve()
    if type(maximum_bytes) is not int or maximum_bytes<1 or path.stat().st_size>1024**2:raise ValueError('Bounded native source manifest required')
    if file_sha256(path)!=path.with_name(path.name+'.sha256').read_text().strip():raise ValueError('Native source overlay manifest seal changed')
    record=json.loads(path.read_text())
    if record.get('schema')!='research2-native-source-overlay-v1' or not isinstance(record.get('files'),list) or not 1<=len(record['files'])<=256:raise ValueError('Unknown or unbounded native source overlay')
    archive=path.parent/record['archive']['name']
    if archive.parent!=path.parent or archive.is_symlink() or file_sha256(archive)!=record['archive']['sha256']:raise ValueError('Native source overlay archive changed or escaped its package')
    entries={};total=0
    for entry in record['files']:
        if set(entry)!={'path','sha256','bytes'}:raise ValueError('Incomplete source overlay entry')
        name=entry['path'];safe=PurePosixPath(name)
        if not isinstance(name,str) or str(safe)!=name or safe.is_absolute() or '..' in safe.parts or not (name.startswith('src/') or name.startswith('buildsystem/') or name in {'Makefile','Template-Config.sh','LICENSE','defines_extra'}) or name in entries:raise ValueError('Unsafe/duplicate source overlay path')
        if type(entry['bytes']) is not int or entry['bytes']<1 or not re.fullmatch('[0-9a-f]{64}',entry['sha256']):raise ValueError('Invalid native source inventory')
        total+=entry['bytes'];entries[name]=entry
    if total>maximum_bytes:raise MemoryError('Native source overlay exceeds declared memory')
    with zipfile.ZipFile(archive) as z:
        infos=z.infolist()
        if len(infos)!=len(entries) or {i.filename for i in infos}!=set(entries):raise ValueError('Native source archive differs from manifest inventory')
        for info in infos:
            if info.file_size!=entries[info.filename]['bytes'] or info.is_dir() or (info.external_attr>>16)&0o170000==0o120000:raise ValueError('Native overlay archive has unexpected sizes or links')
            content=z.read(info)
            if hashlib.sha256(content).hexdigest()!=entries[info.filename]['sha256']:raise ValueError('Native source file hash differs from overlay inventory')
    return record,archive


def install_native_source_overlay(manifest_path,source_root,*,expected_existing_hashes,maximum_bytes=64*1024**2):
    """Install exact native source into existing directories with explicit prior hashes.

    This is a full-file source overlay after the retained ordered upstream
    patches, not a claim that patch/build/restart validation has run. Conflicting
    local edits require their exact expected hash, never an implicit overwrite.
    Staging uses adjacent files, no temporary directory or external extraction.
    """
    import zipfile,uuid
    from core.foundation import atomic_json, atomic_text
    record,archive=native_source_overlay_manifest(manifest_path,maximum_bytes=maximum_bytes)
    root=Path(source_root).expanduser().resolve()
    if not root.is_dir() or not (root/'src').is_dir() or not isinstance(expected_existing_hashes,dict):raise ValueError('Declare an existing native checkout and prior-file hashes')
    names={e['path'] for e in record['files']}
    if set(expected_existing_hashes)!=names:raise ValueError('Bind every target to its existing SHA256 or null for a new source file')
    staged=[];prior={};generation=uuid.uuid4().hex
    # Validate the complete input and target census before publishing any source.
    for entry in record['files']:
        path=root/entry['path']
        if not path.parent.is_dir() or path.is_symlink() or path.parent.resolve()!=path.parent:raise ValueError('Native overlay requires existing direct target directories')
        existing=file_sha256(path) if path.is_file() else None
        expected=expected_existing_hashes[entry['path']]
        if expected is not None and (not isinstance(expected,str) or not re.fullmatch('[0-9a-f]{64}',expected)):raise ValueError('Invalid prior native source SHA256')
        if existing!=expected and existing!=entry['sha256']:raise ValueError('Native local edits conflict with the source overlay: '+entry['path'])
        prior[entry['path']]=existing
    journal=root/'research-source-overlay.installation.json'
    if journal.exists():
        previous=json.loads(journal.read_text())
        if previous.get('status')=='PENDING':raise ValueError('Retain/recover the previous interrupted source installation before a new overlay')
    atomic_json(journal,{'schema':'native-source-installation-v1','status':'PENDING','generation':generation,'prior_hashes':prior,
        'archive_sha256':record['archive']['sha256'],'validation_status':'NOT_RUN'})
    try:
        with zipfile.ZipFile(archive) as z:
            for entry in record['files']:
                path=root/entry['path'];stage=path.with_name('.'+path.name+'.research-stage-'+generation)
                if path.is_file() and file_sha256(path)==entry['sha256']:continue
                content=z.read(entry['path'])
                if hashlib.sha256(content).hexdigest()!=entry['sha256']:raise ValueError('Overlay changed after manifest intake')
                with stage.open('xb') as f:f.write(content);f.flush();os.fsync(f.fileno())
                if path.exists():os.chmod(stage,path.stat().st_mode&0o777)
                staged.append((stage,path,entry))
        for stage,path,entry in staged:
            current=file_sha256(path) if path.is_file() else None
            if current!=prior[entry['path']]:raise ValueError('Native target changed during source staging')
            os.replace(stage,path)
            fd=os.open(path.parent,os.O_RDONLY)
            try:os.fsync(fd)
            finally:os.close(fd)
        if any(file_sha256(root/e['path'])!=e['sha256'] for e in record['files']):raise ValueError('Incomplete native source installation')
        output={'schema':'native-source-installation-v1','status':'COMPLETE','generation':generation,'archive_sha256':record['archive']['sha256'],
            'files':record['files'],'prior_hashes':prior,'build_execution':'NOT_RUN','test_execution':'NOT_RUN','physics_acceptance':'NOT_ACCEPTED'}
        atomic_json(journal,output);atomic_text(journal.with_name(journal.name+'.sha256'),file_sha256(journal)+'\n')
        return output
    except BaseException as error:
        atomic_json(journal,{'schema':'native-source-installation-v1','status':'PENDING','generation':generation,'prior_hashes':prior,
            'archive_sha256':record['archive']['sha256'],'failure':{'type':type(error).__name__,'message':str(error)},
            'recovery':'Inspect retained adjacent stage files and exact target hashes before explicit recovery; no build may use a PENDING installation'})
        raise


if __name__ == "__main__":
    main()
