"""Cosmology: consolidated scientific implementation."""
from __future__ import annotations

# ========================================================================
# cosmology: Generate and validate linear cosmology inputs with CLASS or AxiCLASS.

# Workflow and units
# Translates config.yaml to CLASS/AxiCLASS inputs and exports linear spectra/backgrounds for run manifests.
# k is h/Mpc and P is (Mpc/h)^3; omega_i=Omega_i*h^2. a=1/(1+z), and all growth rates are dimensionless.
# Axion EDE follows PoulinV/AxiCLASS. Particle runs use EDE-conditioned linear inputs and its tabulated smooth background/growth; they do not evolve nonlinear scalar-field modes. Thermal relic distributions follow the declared CLASS ncdm model.
# Bundled source patches: core/patches. Runtime choices are saved with each run.

import argparse
import hashlib
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import uuid
from pathlib import Path
from core.native import native_engine_directory
from core.resources import runtime_paths

import numpy as np
import h5py
import yaml
from scipy.integrate import solve_ivp
from core.foundation import RHO_CRIT_MSUN_H_MPC_H3
from core.foundation import file_sha256 as sha256
from core.native import directory_owner, native_run_owner, native_run_lock
from core.foundation import atomic_json, atomic_copy, atomic_text
from core.data import publish_numeric_table, verify_numeric_table

ROOT = Path(__file__).resolve().parent


def load_config(path="config.yaml"):
    """Read project YAML inputs without changing their physical parameters."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_config(cfg, path):
    """Write the resolved configuration used by subsequent solver/run stages."""
    atomic_text(path, yaml.safe_dump(cfg, sort_keys=False))




def executable(name):
    """Resolve only the selected solver; return None when it has no executable."""
    if name not in {'class', 'axiclass'}:
        raise ValueError('Supported Boltzmann solver required')
    legacy = 'class_public' if name == 'class' else 'AxiCLASS'
    candidates = [native_engine_directory(name) / 'class', ROOT/'software'/legacy/'class']
    for p in candidates:
        if p.is_file() and os.access(p, os.X_OK):
            return p
    return None


def _pinned_software_revision(name):
    """Read the packaged upstream revision without requiring a Git checkout."""
    path = ROOT / "software-revisions"
    if not path.is_file():
        return {"status": "UNAVAILABLE", "reason": "software-revisions is missing"}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        fields = line.split()
        if len(fields) >= 3 and fields[0] == name:
            return {
                "status": "DECLARED",
                "repository": fields[1],
                "commit": fields[2],
                "revision_file": str(path.resolve()),
                "revision_file_sha256": sha256(path),
            }
    return {"status": "UNAVAILABLE", "reason": "no pinned revision for " + name}


def _solver_source_inventory(source_root):
    """Hash the vendored solver sources and build files, excluding generated output."""
    source_root = Path(source_root).resolve()
    ignored_directories = {
        ".git", "build", "output", "__pycache__", "notebooks", "doc", "docs",
        "test", "tests", "python",
    }
    source_suffixes = {
        ".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx",
        ".inc", ".f", ".for", ".f90", ".f95", ".f03", ".s", ".S", ".pl", ".pm", ".py", ".sh",
    }
    files = []
    for path in sorted(source_root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(source_root)
        if any(part in ignored_directories for part in relative.parts[:-1]):
            continue
        name = path.name.lower()
        if path.suffix in source_suffixes or name == "makefile" or name.startswith("makefile.") or path.suffix in {".mk", ".cmake"}:
            files.append({
                "path": relative.as_posix(),
                "sha256": sha256(path),
                "size_bytes": path.stat().st_size,
            })
    digest = hashlib.sha256()
    for item in files:
        digest.update(item["path"].encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(item["sha256"]))
    return {
        "path": str(source_root),
        "sha256": digest.hexdigest(),
        "file_count": len(files),
        "files": files,
    }



def native_source_manifest(engine,source_root):
    """Measure shipped source/build-recipe bytes without inventing compiled equivalence."""
    root=Path(source_root).resolve()
    if not root.is_dir():raise FileNotFoundError('Native source root is absent: '+str(root))
    return {'schema':'native-source-manifest-v1','engine':engine,'source_tree':_solver_source_inventory(root),
        'pinned_revision':_pinned_software_revision(engine),
        'build_equivalence_status':'UNDECLARED; source bytes alone do not prove how an existing executable was compiled',
        'science_acceptance':'NOT_ACCEPTED'}


def verify_native_source_manifest(record):
    """Reject changed source or recipe inventories during a native operation."""
    current=_solver_source_inventory(record['source_tree']['path'])
    if current!=record['source_tree']:raise ValueError('Native source/build recipe changed during operation')
    pin=_pinned_software_revision(record['engine'])
    if pin!=record['pinned_revision']:raise ValueError('Native pinned-revision declaration changed during operation')
    return record


def _solver_build_recipe(source_root):
    """Capture the effective CLASS make recipe and the toolchain versions."""
    source_root = Path(source_root).resolve()
    try:
        database = subprocess.run(
            ["make", "-C", str(source_root), "-pn", "class"],
            text=True, capture_output=True, check=False, timeout=30,
        )
        recipe = subprocess.run(
            ["make", "-C", str(source_root), "-B", "-n", "class"],
            text=True, capture_output=True, check=False, timeout=30,
        )
        target_check = subprocess.run(
            ["make", "-C", str(source_root), "-q", "class"],
            text=True, capture_output=True, check=False, timeout=30,
        )
        make_version = subprocess.run(
            ["make", "--version"], text=True, capture_output=True, check=False, timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"status": "UNAVAILABLE", "reason": str(exc)}

    variable_names = ("CC", "CPP", "OPTFLAG", "OMPFLAG", "CCFLAG", "LDFLAG", "INCLUDES")
    variables = {}
    for name in variable_names:
        match = re.search(
            r"^" + re.escape(name) + r"\s*[:+?]?=\s*(.*?)\s*$",
            database.stdout, re.MULTILINE,
        )
        if match:
            variables[name] = match.group(1)
    recipe_lines = [line.rstrip() for line in recipe.stdout.splitlines() if line.strip()]
    compiler_versions = {}
    for name in ("CC", "CPP"):
        tokens = shlex.split(variables.get(name, ""))
        if not tokens:
            continue
        try:
            version = subprocess.run(
                [tokens[0], "--version"], text=True, capture_output=True, check=False, timeout=5,
            )
            compiler_versions[name] = {
                "command": tokens[0],
                "status": "RECORDED" if version.returncode == 0 else "UNAVAILABLE",
                "version": "\n".join((version.stdout or version.stderr).splitlines()[:8]),
            }
        except (OSError, subprocess.TimeoutExpired) as exc:
            compiler_versions[name] = {"command": tokens[0], "status": "UNAVAILABLE", "reason": str(exc)}
    try:
        make_version_text = (make_version.stdout or make_version.stderr).splitlines()[0]
    except IndexError:
        make_version_text = "unavailable"
    target_status = (
        "UP_TO_DATE" if target_check.returncode == 0 else
        "STALE" if target_check.returncode == 1 else
        "CHECK_FAILED"
    )
    environment = {
        key: os.environ[key]
        for key in ("CC", "CPP", "CFLAGS", "CXXFLAGS", "FFLAGS", "LDFLAGS", "OPTFLAG", "OMPFLAG")
        if key in os.environ
    }
    status = (
        "RECORDED"
        if database.returncode == 0 and recipe.returncode == 0 and recipe_lines
        and target_status == "UP_TO_DATE"
        else "PARTIAL"
    )
    result = {
        "status": status,
        "target_dependency_status": target_status,
        "make_version": make_version_text,
        "effective_make_variables": variables,
        "build_environment_overrides": environment,
        "compiler_versions": compiler_versions,
        "build_recipe_command": "make -C %s -B -n class" % source_root,
        "build_recipe_lines": recipe_lines,
    }
    if status != "RECORDED":
        result["reason"] = "Could not recover the complete effective make recipe"
        result["make_database_returncode"] = database.returncode
        result["make_recipe_returncode"] = recipe.returncode
        result["make_target_check_returncode"] = target_check.returncode
        result["stderr"] = (database.stderr + recipe.stderr)[-4000:]
    return result


def solver_execution_provenance(engine, binary, ini, log, outputs, requested_redshift):
    """Return reproducible source, toolchain, executable, input, and output identity."""
    binary = Path(binary).expanduser().resolve()
    ini = Path(ini).expanduser().resolve()
    log = Path(log).expanduser().resolve()
    source_root = binary.parent
    source = _solver_source_inventory(source_root)
    git_revision = git_status = None
    git_error = None
    try:
        git_revision = subprocess.run(
            ["git", "-C", str(source_root), "rev-parse", "HEAD"],
            text=True, capture_output=True, check=False, timeout=5,
        )
        git_status = subprocess.run(
            ["git", "-C", str(source_root), "status", "--porcelain", "--untracked-files=no"],
            text=True, capture_output=True, check=False, timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        # Portable source releases need no Git installation or .git checkout.
        # Source bytes and the shipped revision declaration remain authoritative.
        git_error = str(exc)

    def artifact(path):
        path = Path(path).expanduser().resolve()
        if not path.is_file():
            return None
        return {"path": str(path), "sha256": sha256(path), "size_bytes": path.stat().st_size}

    recorded_outputs = {}
    for name, path in outputs.items():
        if path is not None:
            value = artifact(path)
            if value is not None:
                recorded_outputs[name] = value
    build = _solver_build_recipe(source_root)
    pinned_revision = _pinned_software_revision(engine)
    expected_outputs = {name for name, path in outputs.items() if path is not None}
    complete = (
        source["file_count"] > 0 and binary.is_file() and ini.is_file() and log.is_file()
        and build.get("status") == "RECORDED" and pinned_revision.get("status") == "DECLARED"
        and bool(expected_outputs) and set(recorded_outputs) == expected_outputs
    )
    return {
        "status": "RECORDED" if complete else "PARTIAL",
        "engine": engine,
        "requested_redshift": float(requested_redshift),
        "pinned_revision": pinned_revision,
        "git_revision": git_revision.stdout.strip() if git_revision is not None and git_revision.returncode == 0 else None,
        "git_worktree_clean": (not git_status.stdout.strip()) if git_status is not None and git_status.returncode == 0 else None,
        "git_inspection_error": git_error,
        "missing_outputs": sorted(expected_outputs - set(recorded_outputs)),
        "source_tree": source,
        "executable": artifact(binary),
        "input_ini": artifact(ini),
        "solver_log": artifact(log),
        "outputs": recorded_outputs,
        "build": build,
    }


def k_limits(box_mpc_h, n):
    """Choose the required linear-spectrum coverage in h/Mpc from the declared resolution and configuration."""
    k_fund = 2.0 * math.pi / box_mpc_h
    k_nyq = math.pi * n / box_mpc_h
    # Margin ensures interpolation and second-to-last CLASS sample cover 3D corner modes (sqrt(3)*k_nyq).
    return k_fund / 8.0, max(k_nyq * 2.5, math.sqrt(3.0) * k_nyq * 1.35)


def class_common(cfg, z_pk, root, kmax_hmpc):
    """Build common linear solver inputs using physical densities omega_i=Omega_i*h^2."""
    c = cfg["cosmology"]
    params = {
        "H0": c["H0"],
        "omega_b": c["omega_b"],
        "omega_cdm": c["omega_cdm"],
        "A_s": c["A_s"],
        "n_s": c["n_s"],
        "tau_reio": c["tau_reio"],
        "T_cmb": c.get("T_cmb", 2.7255),
        "N_ur": c.get("N_ur", 3.046),
        "N_ncdm": c.get("N_ncdm", 0),
        "m_ncdm": c.get("m_ncdm_eV", 0.0),
        "T_ncdm": c.get("T_ncdm", 0.71611),
        # Velocity transfers are required to validate species-specific mixed
        # hydro ICs. Use CLASS's synchronous solver gauge and transform the
        # transfer functions into N-body gauge. The bundled CLASS/AxiCLASS
        # sources include a small output fix so this also writes t_cdm when
        # the synchronous-gauge CDM velocity is identically zero.
        "output": "mPk,mTk,vTk",
        "gauge": "synchronous",
        "nbody_gauge_transfer_functions": "yes",
        "P_k_max_h/Mpc": max(10.0, kmax_hmpc),
        "z_pk": z_pk,
        "root": str(root),
        "write background": "yes",
        "write parameters": "yes",
        "headers": "yes",
        "format": "class",
    }
    extra = c.get("extra", {}) or {}
    if not isinstance(extra, dict):
        raise ValueError("cosmology.extra must be a mapping of CLASS parameters")
    managed = {'root', 'z_pk', 'headers', 'format', 'write background', 'write parameters'}
    if managed.intersection(extra):
        raise ValueError('cosmology.extra cannot override workflow-owned solver outputs: '+', '.join(sorted(managed.intersection(extra))))
    if "gauge" in extra and str(extra["gauge"]).strip().lower() not in {
        "sync", "synchronous"
    }:
        raise ValueError(
            "cosmology.extra.gauge must be synchronous for the validated N-body-gauge "
            "species transfer output; AxiCLASS scalar-field perturbations do not support Newtonian gauge"
        )
    if "nbody_gauge_transfer_functions" in extra and str(
        extra["nbody_gauge_transfer_functions"]
    ).strip().lower() not in {"yes", "y", "true", "1"}:
        raise ValueError(
            "N-body-gauge transfer functions are required for mixed-species IC validation"
        )
    params.update(extra)
    # Retain any caller-requested outputs while guaranteeing the fields used
    # by the IC validator are present.
    output_value = extra.get("output", "mPk,mTk,vTk")
    output_tokens = [
        token for token in re.split(r"[,\s]+", str(output_value).strip()) if token
    ]
    for required_output in ("mPk", "mTk", "vTk"):
        if required_output not in output_tokens:
            output_tokens.append(required_output)
    params["output"] = ",".join(output_tokens)
    # Keep these validated transfer conventions authoritative if a generic
    # extra-parameter mapping attempts to override them.
    params["gauge"] = "synchronous"
    params["nbody_gauge_transfer_functions"] = "yes"
    return params


def omega_radiation_from_config(cfg):
    """Return CLASS's present radiation density for the supported LCDM budget.

    The fallback is used when an imported P(k) table has no CLASS background
    file.  Its constants intentionally match the pinned CLASS source; solver
    runs use ``Omega_r(z=0)`` read from CLASS itself instead.
    """
    c = cfg["cosmology"]
    if "omega_radiation" in c:
        value = float(c["omega_radiation"])
        if not math.isfinite(value) or value <= 0 or value >= 0.01:
            raise ValueError("cosmology.omega_radiation must be finite and in (0, 0.01)")
        return value

    extra = c.get("extra", {}) or {}
    h = float(c["H0"]) / 100.0
    temperature = float(extra.get("T_cmb", c.get("T_cmb", 2.7255)))
    if not math.isfinite(h) or h <= 0 or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("H0 and T_cmb must be finite and positive to derive Omega_r")

    # CLASS's pinned constants, used in its photon blackbody density.
    grav_const = 6.67428e-11
    boltzmann = 1.3806504e-23
    planck = 6.62606896e-34
    speed_of_light = 2.99792458e8
    mpc_m = 3.085677581282e22
    h100_si = 1.0e5 / mpc_m
    rho_gamma = 8.0 * math.pi**5 * boltzmann**4 * temperature**4 / (15.0 * planck**3 * speed_of_light**5)
    omega_gamma_h2 = (8.0 * math.pi * grav_const * rho_gamma / (3.0 * h100_si**2))

    omega_g = extra.get("Omega_g")
    omega_g_h2 = extra.get("omega_g")
    if omega_g is not None and omega_g_h2 is not None:
        raise ValueError("Specify at most one of cosmology.extra.Omega_g and omega_g")
    if omega_g is not None:
        omega_gamma = float(omega_g)
    elif omega_g_h2 is not None:
        omega_gamma = float(omega_g_h2) / h**2
    else:
        omega_gamma = omega_gamma_h2 / h**2

    omega_ur = extra.get("Omega_ur")
    omega_ur_h2 = extra.get("omega_ur")
    n_ur = extra.get("N_ur", c.get("N_ur", 3.046))
    if sum(value is not None for value in (omega_ur, omega_ur_h2)) > 1:
        raise ValueError("Specify at most one of cosmology.extra.Omega_ur and omega_ur")
    if omega_ur is not None:
        omega_neutrinos = float(omega_ur)
    elif omega_ur_h2 is not None:
        omega_neutrinos = float(omega_ur_h2) / h**2
    else:
        # CLASS's default massless-neutrino temperature ratio and density.
        omega_neutrinos = omega_gamma * (7.0 / 8.0) * (4.0 / 11.0) ** (4.0 / 3.0) * float(n_ur)

    unsupported = {"Neff", "N_eff", "N_idr", "N_dg", "xi_idr", "xi_ur", "Omega_idr", "omega_idr"}
    if unsupported.intersection(extra):
        raise ValueError(
            "The analytic radiation fallback cannot derive CLASS's full radiation budget from "
            "these extra parameters: " + ", ".join(sorted(unsupported.intersection(extra))
        ) + ". Run the Boltzmann solver to obtain its background, or provide a validated omega_radiation."
        )

    value = omega_gamma + omega_neutrinos
    if not math.isfinite(value) or value <= 0 or value >= 0.01:
        raise ValueError("Implausible radiation density derived from cosmology inputs: " + str(value))
    return value


def class_background_table(path):
    """Load a CLASS background table and map its self-described columns."""
    labels, data = _class_named_table(path)
    return _background_columns(labels), data


def _background_columns(labels):
    """Keep exact CLASS column labels and add the aliases used by consumers."""
    header = ' '.join(labels)
    if not labels or labels[0] != 'z':
        raise ValueError('CLASS background requires redshift z as its first column')
    if not ('Omega_r(z)' in header or ('rho_g' in header and 'rho_crit' in header)):
        raise ValueError('CLASS background output has no recognizable density columns')
    columns = {}
    for index, label in enumerate(labels):
        columns[label] = index
        if label.startswith("H "):
            columns.setdefault("H", index)
        if label.startswith("Omega_r("):
            columns.setdefault("Omega_r(z)", index)
        if label.startswith("Omega_m("):
            columns.setdefault("Omega_m(z)", index)
        if "gr.fac. D" in label:
            columns.setdefault("growth_D", index)
        if "gr.fac. f" in label:
            columns.setdefault("growth_f", index)
        density = re.search(r"rho_[A-Za-z0-9_]+", label)
        if density:
            columns.setdefault(density.group(0), index)
    return columns


def validated_proper_time_samples(scale_factors, times_gyr):
    """Validate the full source clock; never discard malformed rows or duplicates."""
    scale, times = np.asarray(scale_factors, dtype=float), np.asarray(times_gyr, dtype=float)
    if (scale.ndim != 1 or len(scale) < 2 or times.shape != scale.shape
            or not np.isfinite(scale).all() or not np.isfinite(times).all()
            or np.any(scale <= 0) or np.any(times < 0)):
        raise ValueError('Cosmic clock requires finite positive scale factors and nonnegative proper times')
    order = np.argsort(scale, kind='stable')
    scale, times = scale[order], times[order]
    if np.any(np.diff(scale) <= 0) or np.any(np.diff(times) <= 0):
        raise ValueError('Cosmic clock scale factors and proper times must increase strictly')
    return scale, times


def interpolate_proper_time(scale_factors, times_gyr, scale_factor):
    """Piecewise-linear CLASS proper time in canonical a; no extrapolation."""
    scale, times = validated_proper_time_samples(scale_factors, times_gyr)
    if (isinstance(scale_factor, (bool, np.bool_)) or not isinstance(scale_factor, (int, float, np.integer, np.floating))
            or not np.isfinite(scale_factor) or not scale[0] <= scale_factor <= scale[-1]):
        raise ValueError('Epoch is outside the validated CLASS proper-time domain')
    return float(np.interp(scale_factor, scale, times))


def proper_time_numerical_envelope(scale_factors, times_gyr, hubble_inverse_mpc):
    """Bound linear-t interpolation relative to a declared piecewise-linear H(a).

    CLASS H is in 1/Mpc with c=1; physical H [s^-1] = c[cm/s] H/Mpc[cm].
    Anchor t at the first supplied proper-time row. On an interval H=H0+s(a-a0),
    f=dt/da=1/(a H), t''=-1/(a^2 H)-s/(a H^2).
    Positive endpoint H and a supply conservative absolute derivative bounds.
    Linear interpolation error <= max|t''| delta_a^2/8. Source-node residuals
    against integrated H and accumulated quadrature errors are also retained.
    This bounds the declared tabulated model, not the unknown continuum H or
    solver/systematic accuracy. QUADPACK errors are numerical estimates.
    """
    from scipy.integrate import quad
    from core.foundation import C_SI, MPC_CM, SECONDS_PER_YEAR
    raw_a = np.asarray(scale_factors,dtype=float)
    order = np.argsort(raw_a,kind='stable')
    a,t = validated_proper_time_samples(raw_a,times_gyr)
    H = np.asarray(hubble_inverse_mpc,dtype=float)
    if H.shape != raw_a.shape or not np.isfinite(H).all() or np.any(H<=0):
        raise ValueError('CLASS clock expansion samples require finite positive H in 1/Mpc')
    H = H[order]
    factor = MPC_CM/(C_SI*100*SECONDS_PER_YEAR*1.e9)
    widths = np.diff(a)
    slopes = np.diff(H)/widths
    low_H = np.minimum(H[:-1],H[1:])
    derivative_bound = factor*(1/(a[:-1]**2*low_H) + np.abs(slopes)/(a[:-1]*low_H**2))
    integrated = [float(t[0])]
    errors = [0.]
    for i,width in enumerate(widths):
        value,error = quad(lambda x:factor/(x*(H[i]+slopes[i]*(x-a[i]))),
                           a[i],a[i+1],epsabs=1.e-12,epsrel=1.e-11)
        integrated.append(integrated[-1]+value)
        errors.append(errors[-1]+error)
    residual = np.abs(t-np.asarray(integrated))
    bounds = np.maximum(residual[:-1],residual[1:])+derivative_bound*widths**2/8+np.asarray(errors[1:])
    if not np.isfinite(bounds).all():
        raise ValueError('Clock error envelope overflowed')
    return {'schema':'proper-time-numerical-envelope-v1','status':'MODEL_CONDITIONAL_BOUND',
        'reference_model':'piecewise-linear positive CLASS H(a), anchored at first source t',
        'scale_factors':a.tolist(),'interval_absolute_bound_gyr':bounds.tolist(),
        'source_node_residual_gyr':residual.tolist(),'quadrature_error_estimate_gyr':errors,
        'maximum_absolute_bound_gyr':float(np.max(bounds)),
        'extrapolation':'FORBIDDEN','solver_and_background_sampling_error':'NOT_BOUNDED',
        'quadrature':'scipy QUADPACK epsabs=1e-12 Gyr epsrel=1e-11; error estimate, not rigorous proof'}


_CLOCK_NUMERICAL_ENVELOPES = {}

def background_proper_time_clock(background_path, *, expected_sha256=None):
    """Read the shared CLASS clock directly, with no filtering or extrapolation.

    A supplied checksum binds it to the caller's manifest. Without one, the
    returned checksum identifies the measured source, whose bytes must stay
    unchanged throughout loading and numerical-envelope construction.
    """
    path = Path(background_path).expanduser().resolve()
    checksum = sha256(path)
    if expected_sha256 is not None and checksum != expected_sha256:
        raise ValueError('CLASS proper-time source is missing or changed')
    columns, table = class_background_table(path)
    if 'z' not in columns or 'proper time [Gyr]' not in columns:
        raise ValueError('CLASS background lacks redshift/proper-time columns')
    z = table[:, columns['z']]
    if not np.isfinite(z).all() or np.any(z < 0):
        raise ValueError('CLASS clock redshift samples must be finite and nonnegative')
    scale, times = validated_proper_time_samples(
        1 / (1 + z), table[:, columns['proper time [Gyr]']],
    )
    envelope = None
    if 'H' in columns:
        key = (str(path), checksum)
        envelope = _CLOCK_NUMERICAL_ENVELOPES.get(key)
        if envelope is None:
            envelope = proper_time_numerical_envelope(
                1 / (1 + z), table[:, columns['proper time [Gyr]']], table[:, columns['H']],
            )
            if len(_CLOCK_NUMERICAL_ENVELOPES) >= 2:
                _CLOCK_NUMERICAL_ENVELOPES.pop(next(iter(_CLOCK_NUMERICAL_ENVELOPES)))
            _CLOCK_NUMERICAL_ENVELOPES[key] = envelope
    if sha256(path) != checksum:
        raise RuntimeError('CLASS proper-time source changed during reading')
    return scale, times, {
        'status': 'VERIFIED' if expected_sha256 is not None else 'MEASURED',
        'path': str(path), 'sha256': checksum, 'units': 'Gyr',
        'interpolation': 'piecewise linear in scale factor; no extrapolation',
        'interpolation_error': (
            'Conditional numerical envelope; continuum background accuracy still requires convergence'
            if envelope else 'H absent; numerical envelope unavailable'
        ),
        'numerical_error_envelope': envelope,
        'scale_factor_domain': [float(scale[0]), float(scale[-1])],
    }


def run_proper_time_clock(run_dir):
    """Read a manifest-bound CLASS/AxiCLASS clock for every run product."""
    root = Path(run_dir).expanduser().resolve()
    manifest_path = root / 'manifest.json'
    if not manifest_path.is_file():
        return None, None, {'status': 'UNAVAILABLE', 'reason': 'No native run manifest with a hashed CLASS background'}
    manifest_hash = sha256(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    entry = manifest.get('input_artifacts', {}).get('boltzmann_background')
    if entry is None:
        return None, None, {'status': 'UNAVAILABLE', 'reason': 'Run manifest has no hashed CLASS background'}
    checksum = entry.get('sha256')
    if not isinstance(checksum, str) or not re.fullmatch('[0-9a-f]{64}', checksum):
        raise ValueError('Run CLASS proper-time source requires its recorded SHA256')
    path = Path(entry['path']).expanduser()
    if not path.is_absolute():
        path = root / path
    clock = background_proper_time_clock(path, expected_sha256=checksum)
    if sha256(manifest_path) != manifest_hash:
        raise RuntimeError('Run proper-time manifest changed during reading')
    return clock


def proper_time_intervals(clock, scale_factors):
    """Evaluate ordered epochs and propagate endpoint clock errors into gaps.

    Endpoint errors add for an absolute interval bound. Missing H samples leave
    that bound unresolved; nominal interpolation never supplies a zero error.
    Bounds describe the declared tabulated background, not cosmological error.
    """
    from .data import proper_time_error_at_epoch
    scale, times, source = clock
    scale, times = validated_proper_time_samples(scale, times)
    requested = np.asarray(scale_factors)
    if (requested.ndim != 1 or len(requested) < 2 or requested.dtype.kind not in 'fiu'
            or not np.isfinite(requested).all() or np.any(requested <= 0)
            or np.any(np.diff(requested.astype(float)) <= 0)
            or requested.min() < scale[0] or requested.max() > scale[-1]):
        raise ValueError('At least two increasing epochs inside the CLASS clock domain required')
    ages = np.interp(requested, scale, times)
    intervals = np.diff(ages)
    if not np.isfinite(intervals).all() or np.any(intervals <= 0):
        raise ValueError('Proper-time gaps are not numerically resolved by the CLASS table')
    errors = [proper_time_error_at_epoch(source, float(a)) for a in requested]
    bounded = all(error['status'] == 'MODEL_CONDITIONAL_BOUND' for error in errors)
    epoch_bounds = interval_bounds = upper = None
    if bounded:
        epoch_bounds = np.array([error['absolute_bound_gyr'] for error in errors])
        interval_bounds = epoch_bounds[:-1] + epoch_bounds[1:]
        upper = intervals + interval_bounds
        if not np.isfinite(upper).all():
            raise ValueError('Proper-time interval error bound overflows')
    return {
        'scale_factors': requested.astype(float).tolist(), 'proper_times_gyr': ages.tolist(),
        'intervals_gyr': intervals.tolist(),
        'bound_status': 'MODEL_CONDITIONAL_BOUND' if bounded else 'UNBOUNDED',
        'epoch_absolute_bounds_gyr': epoch_bounds.tolist() if bounded else None,
        'interval_absolute_bounds_gyr': interval_bounds.tolist() if bounded else None,
        'interval_upper_bounds_gyr': upper.tolist() if bounded else None,
        'clock_source': {key: value for key, value in source.items() if key != 'numerical_error_envelope'},
        'bound_scope': 'Tabulated-model numerical time error only; solver and continuum-background errors excluded',
    }


def _class_named_table(path, required_header_text=None):
    """Read a CLASS numeric table and its numbered, self-described columns."""
    header = None
    matching_description = required_header_text is None
    with open(path, "r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            if not line.startswith("#"):
                continue
            if required_header_text and required_header_text.casefold() in line.casefold():
                matching_description = True
            if re.search(r"#\s*1:\s*", line):
                header = line
    if header is None or not matching_description:
        raise ValueError("CLASS table has no numbered column header: " + str(path))
    markers = list(re.finditer(r"(\d+):", header))
    if [int(marker.group(1)) for marker in markers] != list(range(1, len(markers) + 1)):
        raise ValueError('CLASS table column numbers must run consecutively from one: '+str(path))
    labels = []
    for index, marker in enumerate(markers):
        stop = markers[index + 1].start() if index + 1 < len(markers) else len(header)
        labels.append(re.sub(r"\s+", " ", header[marker.end():stop]).strip())
    if any(not label for label in labels) or len(set(labels)) != len(labels):
        raise ValueError('CLASS table column labels must be nonempty and distinct: '+str(path))
    data = np.loadtxt(path, comments="#", ndmin=2)
    if data.size == 0 or data.shape[1] != len(labels):
        raise ValueError("CLASS table data do not match its numbered header: " + str(path))
    return labels, data


def _hdf5_name(label, index):
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(label)).strip("_")
    return (cleaned or "column") + "_%02d" % int(index)


def _class_table_redshift(path):
    """Return a CLASS table's printed redshift without inferring from its filename."""
    if not path or not Path(path).is_file():
        return None
    with open(path, "r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            if not line.startswith("#"):
                break
            match = re.search(r"\bredshift\s+z\s*=\s*([-+0-9.eE]+)", line, re.IGNORECASE)
            if match:
                value = float(match.group(1))
                return value if math.isfinite(value) else None
    return None


def class_redshift_matches(observed, requested):
    """Match an exact epoch or the pinned CLASS/AxiCLASS `%g` header value.

    Both output.c implementations print six significant digits. A fixed
    decimal tolerance either rejects high-z rounding or merges distinct low-z
    epochs. The unrounded solver input remains the authoritative epoch.
    """
    if observed is None or isinstance(observed, (bool, np.bool_)) or isinstance(requested, (bool, np.bool_)):
        return False
    actual, expected = float(observed), float(requested)
    if not math.isfinite(actual) or not math.isfinite(expected) or actual < 0 or expected < 0:
        return False
    return actual == expected or actual == float(format(expected, '.6g'))


def class_output_files(root, requested_redshift):
    """Select one epoch-matched linear spectrum and its sibling transfers."""
    root = Path(root)
    matches = [path for path in sorted(root.parent.glob(root.name+'*pk.dat'))
               if class_redshift_matches(pk_output_redshift(path), requested_redshift)]
    if len(matches) != 1:
        raise ValueError('Expected one CLASS linear spectrum at the requested epoch under '+str(root))
    power = matches[0]
    # Native output.c uses the same run index/redshift prefix for pk and tk.
    # Selecting an unrelated first transfer file can attach the wrong epoch.
    transfer = power.with_name(power.name[:-len('pk.dat')]+'tk.dat')
    if not transfer.is_file() or not class_redshift_matches(_class_table_redshift(transfer), requested_redshift):
        raise ValueError('Missing or epoch-mismatched CLASS transfer sibling: '+str(transfer))
    backgrounds = sorted(root.parent.glob(root.name+'*background.dat'))
    if len(backgrounds) != 1:
        raise ValueError('Expected one CLASS background under '+str(root))
    return {'pk': power, 'transfer': transfer, 'background': backgrounds[0]}


def _class_transfer_table(path, expected_redshift):
    """Read named species transfers with the shared header and epoch rules."""
    labels, table = _class_named_table(path, required_header_text='Transfer functions')
    columns = {label.split()[0].lower(): index for index, label in enumerate(labels)}
    if len(columns) != len(labels):
        raise ValueError('Ambiguous CLASS transfer column names: '+str(path))
    observed = _class_table_redshift(path)
    if not class_redshift_matches(observed, expected_redshift):
        raise ValueError('CLASS transfer epoch does not match the requested initial conditions: '+str(path))
    return columns, table, observed


def second_order_growth_diagnostic(columns, data):
    """Return smooth-background D2 and f2, explicitly diagnostic rather than IC input.

    This integrates the standard pressureless second-order growth equation using
    CLASS D1, f1, H, and Omega_m. The early boundary uses the EdS asymptote at
    the first CLASS sample with Omega_m >= 0.8 (or the earliest sample if no
    such point exists). It is not a scale-dependent EDE/WDM perturbation solver.
    """
    required = ("H", "growth_D", "growth_f", "Omega_m(z)")
    missing = [key for key in required if key not in columns]
    if missing:
        return {"status": "UNAVAILABLE", "reason": "missing CLASS columns: " + ", ".join(missing)}

    z = np.asarray(data[:, 0], dtype=np.float64)
    a = 1.0 / (1.0 + z)
    order = np.argsort(a)
    a = a[order]
    h = np.asarray(data[order, columns["H"]], dtype=np.float64)
    d1_raw = np.asarray(data[order, columns["growth_D"]], dtype=np.float64)
    f1 = np.asarray(data[order, columns["growth_f"]], dtype=np.float64)
    omega_m = np.asarray(data[order, columns["Omega_m(z)"]], dtype=np.float64)
    if (len(a) < 4 or np.any(~np.isfinite(a)) or np.any(a <= 0)
            or np.any(np.diff(a) <= 0) or np.any(~np.isfinite(h))
            or np.any(h <= 0) or np.any(~np.isfinite(d1_raw))
            or np.any(d1_raw <= 0) or np.any(~np.isfinite(f1))
            or np.any(~np.isfinite(omega_m))):
        return {"status": "UNAVAILABLE", "reason": "invalid CLASS background samples"}
    today = int(np.argmin(np.abs(a - 1.0)))
    if abs(float(a[today]) - 1.0) > 1.0e-8:
        return {"status": "UNAVAILABLE", "reason": "CLASS background does not include a=1"}

    x = np.log(a)
    d1 = d1_raw / d1_raw[today]
    start_candidates = np.flatnonzero(omega_m >= 0.8)
    start = int(start_candidates[0]) if len(start_candidates) else 0
    boundary_method = (
        "first CLASS sample with Omega_m>=0.8" if len(start_candidates)
        else "earliest CLASS sample because Omega_m never reaches 0.8"
    )
    if start >= today - 1:
        return {"status": "UNAVAILABLE", "reason": "insufficient samples after matter-dominated boundary"}
    dlnh_dx = np.gradient(np.log(h), x, edge_order=2)
    x_eval = x[start:today + 1]
    d1_interp = lambda value: np.interp(value, x, d1)
    om_interp = lambda value: np.interp(value, x, omega_m)
    friction_interp = lambda value: np.interp(value, x, dlnh_dx)
    y0 = -3.0 / 7.0 * d1[start] ** 2
    dy0 = 2.0 * f1[start] * y0

    def rhs(value, state):
        second = -1.5 * om_interp(value) * d1_interp(value) ** 2
        acceleration = second - (2.0 + friction_interp(value)) * state[1] + 1.5 * om_interp(value) * state[0]
        return state[1], acceleration

    solution = solve_ivp(
        rhs, (float(x_eval[0]), float(x_eval[-1])), (float(y0), float(dy0)),
        t_eval=x_eval, method="DOP853", rtol=2.0e-9, atol=1.0e-11,
    )
    if not solution.success or solution.y.shape[1] != len(x_eval):
        return {"status": "UNAVAILABLE", "reason": "D2 integration failed: " + solution.message}
    d2 = np.full_like(a, np.nan)
    f2 = np.full_like(a, np.nan)
    d2[start:today + 1] = solution.y[0]
    d2_prime = solution.y[1]
    nonzero = np.abs(solution.y[0]) > np.finfo(np.float64).tiny
    f2_values = np.full_like(solution.y[0], np.nan)
    f2_values[nonzero] = d2_prime[nonzero] / solution.y[0][nonzero]
    f2[start:today + 1] = f2_values
    return {
        "status": "DIAGNOSTIC_ONLY",
        "a": a,
        "D1_normalized": d1,
        "D2": d2,
        "f2": f2,
        "integration_start_a": float(a[start]),
        "integration_start_redshift": float(1.0 / a[start] - 1.0),
        "boundary_condition": "D2=-3/7 D1^2; dD2/dln(a)=2 f1 D2 at the selected early boundary",
        "boundary_method": boundary_method,
        "equation": "D2''+(2+dlnH/dlna)D2'-(3/2)Omega_m D2=-(3/2)Omega_m D1^2",
        "limitations": "smooth pressureless-background approximation; not used for IC velocities or displacements; not a scale-dependent EDE/WDM perturbation calculation",
    }


@directory_owner('destination')
def write_class_cosmology_products(model, requested_redshift, background_path, transfer_path, pk_path, destination):
    """Persist run-matched CLASS/AxiCLASS outputs in a self-describing HDF5 product."""
    destination = Path(destination).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    import tempfile
    final=destination
    stage=Path(tempfile.mkdtemp(prefix='.cosmology-product-',dir=final.parent))
    destination=stage/final.name
    source_hashes={str(Path(path).resolve()):sha256(path) for path in (background_path,transfer_path,pk_path) if path}
    provenance={'algorithm_sha256':sha256(__file__),'source_hashes':source_hashes,'model':str(model),
        'engine':'axiclass' if model=='ede' else 'class','scope':'Saved linear solver tables; no nonlinear acceptance'}
    try:
        with h5py.File(destination, "w") as product:
            product.attrs["schema"] = "darkmattersim.cosmology-products.v1"
            product.attrs["model"] = str(model)
            product.attrs["requested_redshift"] = float(requested_redshift)
            product.attrs["engine"] = "axiclass" if model == "ede" else "class"
    
            if background_path and Path(background_path).is_file():
                raw_labels, data = _class_named_table(background_path)
                columns = _background_columns(raw_labels)
                bg = product.create_group("background")
                bg.attrs["source_file"] = str(Path(background_path).resolve())
                bg.attrs["source_sha256"] = sha256(background_path)
                # Keep the exact source table in column order, with the labels from
                # the numbered header available as metadata for downstream readers.
                bg.create_dataset("raw_table", data=data, compression="gzip", shuffle=True)
                bg.create_dataset("column_labels", data=np.asarray(raw_labels, dtype=h5py.string_dtype("utf-8")))
                raw_columns = bg.create_group("columns")
                for column_index, label in enumerate(raw_labels):
                    column = raw_columns.create_dataset(
                        _hdf5_name(label, column_index + 1), data=data[:, column_index],
                        compression="gzip", shuffle=True,
                    )
                    column.attrs["class_column_label"] = label
                    column.attrs["units"] = "CLASS native units; see column label and source table"
                z = np.asarray(data[:, 0], dtype=np.float64)
                a = 1.0 / (1.0 + z)
                bg.create_dataset("redshift", data=z).attrs["units"] = "dimensionless"
                bg.create_dataset("scale_factor", data=a).attrs["units"] = "dimensionless"
                if "proper time [Gyr]" in columns:
                    bg.create_dataset("cosmic_time", data=data[:, columns["proper time [Gyr]"]]).attrs["units"] = "Gyr"
                if "H" in columns:
                    h = np.asarray(data[:, columns["H"]], dtype=np.float64)
                    bg.create_dataset("H_1_per_Mpc", data=h).attrs["units"] = "Mpc^-1 (CLASS c=1 convention)"
                    bg.create_dataset("H_km_s_Mpc", data=h * 299792.458).attrs["units"] = "km s^-1 Mpc^-1"
                for alias, output_name, units in (
                    ("Omega_m(z)", "Omega_m", "dimensionless"),
                    ("Omega_r(z)", "Omega_r", "dimensionless"),
                    ("growth_D", "D1_CLASS", "CLASS growth normalization"),
                    ("growth_f", "f1_CLASS", "dimensionless"),
                ):
                    if alias in columns:
                        bg.create_dataset(output_name, data=data[:, columns[alias]]).attrs["units"] = units
                if "rho_crit" in columns:
                    rho_crit = np.asarray(data[:, columns["rho_crit"]], dtype=np.float64)
                    components = bg.create_group("density_components")
                    components.attrs["source_units"] = "CLASS background density units (Mpc^-2)"
                    for density_name, index in columns.items():
                        if not density_name.startswith("rho_") or density_name == "rho_crit":
                            continue
                        values = np.asarray(data[:, index], dtype=np.float64)
                        raw = components.create_dataset(density_name, data=values)
                        raw.attrs["units"] = "CLASS background density units (Mpc^-2)"
                        with np.errstate(divide="ignore", invalid="ignore"):
                            fraction = values / rho_crit
                        frac = components.create_dataset("Omega_" + density_name[4:], data=fraction)
                        frac.attrs["units"] = "dimensionless; rho_component/rho_crit"
                growth = second_order_growth_diagnostic(columns, data)
                d2_group = bg.create_group("second_order_growth")
                d2_group.attrs["status"] = growth["status"]
                for key, value in growth.items():
                    if isinstance(value, np.ndarray):
                        dataset = d2_group.create_dataset(key, data=value, compression="gzip")
                        dataset.attrs["units"] = "dimensionless"
                    elif key != "status":
                        d2_group.attrs[key] = value
                product.attrs["background_source_sha256"] = sha256(background_path)
                if "redshift" in bg:
                    product.attrs["background_contains_today"] = bool(np.any(np.isclose(a, 1.0, rtol=0.0, atol=1.0e-8)))
    
            if transfer_path and Path(transfer_path).is_file():
                labels, values = _class_named_table(transfer_path, required_header_text="Transfer functions")
                transfer = product.create_group("transfer_functions")
                transfer.attrs["source_file"] = str(Path(transfer_path).resolve())
                transfer.attrs["source_sha256"] = sha256(transfer_path)
                transfer.create_dataset("raw_table", data=values, compression="gzip", shuffle=True)
                transfer.create_dataset("column_labels", data=np.asarray(labels, dtype=h5py.string_dtype("utf-8")))
                transfer_columns = transfer.create_group("columns")
                for column_index, label in enumerate(labels):
                    column = transfer_columns.create_dataset(
                        _hdf5_name(label, column_index + 1), data=values[:, column_index],
                        compression="gzip", shuffle=True,
                    )
                    column.attrs["class_column_label"] = label
                    if column_index == 0 and re.search(r"\(h/Mpc\)", label):
                        column.attrs["units"] = "h/Mpc"
                    else:
                        column.attrs["units"] = "CLASS native units; see column label and source table"
                transfer.attrs["requested_redshift"] = float(requested_redshift)
                transfer_redshift = _class_table_redshift(transfer_path)
                if transfer_redshift is not None:
                    transfer.attrs["actual_redshift"] = transfer_redshift
                transfer.attrs["k_column_units"] = "h/Mpc when present in CLASS header"
    
            if pk_path and Path(pk_path).is_file():
                pk_data = np.loadtxt(pk_path, comments="#", ndmin=2)
                if pk_data.ndim != 2 or pk_data.shape[1] < 2 or not np.all(np.isfinite(pk_data)):
                    raise ValueError("CLASS P(k) output is malformed: " + str(pk_path))
                pk = product.create_group("linear_power_spectrum")
                pk.attrs["source_file"] = str(Path(pk_path).resolve())
                pk.attrs["source_sha256"] = sha256(pk_path)
                pk.attrs["requested_redshift"] = float(requested_redshift)
                pk_redshift = pk_output_redshift(pk_path)
                if pk_redshift is not None:
                    pk.attrs["actual_redshift"] = float(pk_redshift)
                    product.attrs["linear_power_spectrum_actual_redshift"] = float(pk_redshift)
                pk.attrs["column_order"] = "k, P(k) as emitted by CLASS/AxiCLASS"
                pk.create_dataset("k_h_Mpc", data=pk_data[:, 0]).attrs["units"] = "h/Mpc"
                pk.create_dataset("P_mpc_h3", data=pk_data[:, 1]).attrs["units"] = "(Mpc/h)^3"
                pk.create_dataset("raw_table", data=pk_data, compression="gzip", shuffle=True)
    
        from core.data import seal_hdf5_numeric_datasets,write_dataset_contract,epoch_contract
        with h5py.File(destination,'r+') as product:
            product.attrs['data_contract_json']=json.dumps(provenance,sort_keys=True)
            product.attrs['science_acceptance']='NOT_ACCEPTED'
            seal_hdf5_numeric_datasets(product)
            def annotate(name,dataset):
                if not isinstance(dataset,h5py.Dataset):return
                family=name.split('/')[0]
                if family=='background':
                    epoch={'canonical_time':'scale_factor','scale_factor_dataset':'/background/scale_factor',
                        'cosmic_time_dataset':'/background/cosmic_time' if 'background/cosmic_time' in product else None}
                    if name.startswith('background/second_order_growth/') and 'background/second_order_growth/a' in product:
                        epoch['scale_factor_dataset']='/background/second_order_growth/a'
                else:
                    actual=product[family].attrs.get('actual_redshift')
                    epoch=(epoch_contract(1/(1+float(actual)),requested_scale_factor=1/(1+float(requested_redshift)))
                        if actual is not None else {'canonical_time':'scale_factor','actual_epoch_status':'UNDECLARED',
                            'requested_scale_factor':1/(1+float(requested_redshift))})
                unit=str(dataset.attrs.get('units','heterogeneous columns' if name.endswith('raw_table') else
                    'UTF-8 labels' if name.endswith('column_labels') else 'UNDECLARED'))
                write_dataset_contract(dataset,units=unit,quantity='Saved CLASS/AxiCLASS '+name,
                    epoch=epoch,provenance=provenance,missing_policy='Native solver values and diagnostic unavailable/nonfinite values retained explicitly')
            product.visititems(annotate)
        for path,checksum in source_hashes.items():
            if sha256(path)!=checksum:raise ValueError('CLASS product source changed during export')
        from core.data import publish_hdf5_product
        publish_hdf5_product(destination,final)
        destination.unlink();stage.rmdir()
        return str(final)
    except BaseException as exc:
        atomic_json(stage/'failure.json',{'status':'FAILED','type':type(exc).__name__,'message':str(exc)})
        raise



@directory_owner('output_path')
def write_gadget_background_table(background_path, output_path, minimum_scale_factor, *, expected_h0_km_s_mpc=None):
    """Convert an AxiCLASS background into the compact table used by GADGET.

    Columns are ``a, H(a)/H0, D(a)/D(1), f(a)``.  The table keeps every CLASS
    sample between the requested early-time bound and today, with exact rows
    inserted at both endpoints.  GADGET uses log-linear interpolation in a.
    """
    source_before=sha256(background_path)
    columns, data = class_background_table(background_path)
    required = ("H", "growth_D", "growth_f")
    missing = [name for name in required if name not in columns]
    if missing:
        raise ValueError("CLASS background is missing columns needed for GADGET: " + ", ".join(missing))

    z = data[:, 0]
    a = 1.0 / (1.0 + z)
    order = np.argsort(a)
    a = a[order]
    h = data[order, columns["H"]]
    growth = data[order, columns["growth_D"]]
    rate = data[order, columns["growth_f"]]
    if np.any(~np.isfinite(a)) or np.any(a <= 0) or np.any(np.diff(a) <= 0):
        raise ValueError("CLASS background scale factors must be finite, positive, and unique")
    if np.any(~np.isfinite(h)) or np.any(h <= 0):
        raise ValueError("CLASS background contains invalid H(a) values")
    if np.any(~np.isfinite(growth)) or np.any(growth <= 0):
        raise ValueError("CLASS background contains invalid linear growth factors")
    if np.any(~np.isfinite(rate)) or np.any(rate <= 0):
        raise ValueError("CLASS background contains invalid linear growth rates")

    a_min = float(minimum_scale_factor)
    if not math.isfinite(a_min) or a_min <= 0 or a_min >= 1:
        raise ValueError("The requested GADGET background-table scale-factor range is invalid")
    z0 = int(np.argmin(np.abs(z)))
    if abs(float(z[z0])) > 1.0e-8:
        raise ValueError("CLASS background must contain the present epoch a=1")
    h0 = float(data[z0, columns["H"]])
    d0 = float(data[z0, columns["growth_D"]])
    if expected_h0_km_s_mpc is not None:
        expected_h0 = float(expected_h0_km_s_mpc)
        if (isinstance(expected_h0_km_s_mpc, (bool, np.bool_)) or not math.isfinite(expected_h0)
                or expected_h0 <= 0 or not math.isclose(h0 * 299792.458, expected_h0, rel_tol=1.0e-6, abs_tol=1.0e-8)):
            raise ValueError('CLASS/AxiCLASS H0 differs from the configured tabulated-background evolution')
    if a[0] > a_min or a[-1] < 1.0:
        raise ValueError("CLASS background does not cover the GADGET integration and growth-normalization range")

    log_a = np.log(a)
    interior = (a > a_min) & (a < 1.0)
    table_a = np.unique(np.concatenate(([a_min], a[interior], [1.0])))
    table_e = np.exp(np.interp(np.log(table_a), log_a, np.log(h / h0)))
    table_d = np.exp(np.interp(np.log(table_a), log_a, np.log(growth / d0)))
    table_f = np.interp(np.log(table_a), log_a, rate)
    table = np.column_stack((table_a, table_e, table_d, table_f))
    if np.any(~np.isfinite(table)) or np.any(table[:, 1:] <= 0):
        raise ValueError("The converted CLASS background table contains invalid values")

    sample = (a >= a_min) & (a <= 1.0)
    log_table_a = np.log(table_a)
    e_reconstructed = np.exp(np.interp(np.log(a[sample]), log_table_a, np.log(table_e)))
    d_reconstructed = np.exp(np.interp(np.log(a[sample]), log_table_a, np.log(table_d)))
    f_reconstructed = np.interp(np.log(a[sample]), log_table_a, table_f)
    e_reference = h[sample] / h0
    d_reference = growth[sample] / d0
    e_error = float(np.max(np.abs(e_reconstructed / e_reference - 1.0)))
    d_error = float(np.max(np.abs(d_reconstructed / d_reference - 1.0)))
    f_error = float(np.max(np.abs(f_reconstructed - rate[sample])))
    if max(e_error, d_error, f_error) > 1.0e-10:
        raise ValueError("Converted GADGET background table does not reproduce the CLASS source columns")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table_contract=publish_numeric_table(output_path,table,
        {'a':{'units':'dimensionless','quantity':'Canonical scale factor'},
         'E':{'units':'dimensionless','quantity':'H(a)/H0'},
         'D':{'units':'dimensionless','quantity':'D(a)/D(1)'},
         'f':{'units':'dimensionless','quantity':'dlnD/dln(a)'}},
        algorithm=__file__,source_hashes={str(background_path):source_before},
        context={'canonical_time':'scale_factor','epoch_column':'a',
            'interpolation':'linear in ln(a) for ln(E), ln(D); linear in ln(a) for f'},
        header='a E(a)=H(a)/H0 D(a)/D(1) f(a)=dlnD/dlna; source CLASS/AxiCLASS')

    return {
        "table_contract":table_contract,
        "status": "PASS",
        "source_background": str(Path(background_path).resolve()),
        "table_file": str(output_path.resolve()),
        "table_sha256": sha256(output_path),
        "rows": int(len(table)),
        "scale_factor_range": [float(table_a[0]), float(table_a[-1])],
        "H0_1_per_Mpc": h0,
        "expected_native_H0_km_s_Mpc": expected_h0_km_s_mpc,
        "growth_normalization_D_at_a1": d0,
        "interpolation": "linear in ln(a) for ln(E) and ln(D); linear in ln(a) for f",
        "max_relative_H_difference_at_source_samples": e_error,
        "max_relative_D_difference_at_source_samples": d_error,
        "max_absolute_f_difference_at_source_samples": f_error,
        "validation_tolerance": 1.0e-10,
    }


def omega_radiation_from_background(path):
    """Read CLASS's total present radiation density from its background table."""
    columns, data = class_background_table(path)
    radiation_column = columns.get("Omega_r(z)")
    if radiation_column is None:
        if "rho_g" not in columns or "rho_crit" not in columns:
            raise ValueError("Could not identify CLASS Omega_r or density columns in " + str(path))
        radiation_columns = [columns[name] for name in ("rho_g", "rho_ur", "rho_idr") if name in columns]
        rho_crit_column = columns["rho_crit"]
        used_columns = radiation_columns + [rho_crit_column]
    else:
        used_columns = [radiation_column]
    if data.shape[1] <= max(used_columns):
        raise ValueError("CLASS background output is empty or malformed: " + str(path))
    row = data[int(np.argmin(np.abs(data[:, 0])))]
    if abs(row[0]) > 1.0e-8:
        raise ValueError("CLASS background output does not include z=0")
    if radiation_column is not None:
        value = row[radiation_column]
    else:
        if row[rho_crit_column] <= 0:
            raise ValueError("CLASS background output has non-positive critical density at z=0")
        value = sum(row[idx] for idx in radiation_columns) / row[rho_crit_column]
    if not math.isfinite(value) or value <= 0 or value >= 0.01:
        raise ValueError("Implausible present radiation density in CLASS output: " + str(value))
    return float(value)


def validate_gadget_background(background_path, cfg, start_redshift, end_redshift=0.0, tolerance=1.0e-4):
    """Compare GADGET's matter+radiation+Lambda H(a) to CLASS over the run."""
    if isinstance(tolerance, (bool, np.bool_)) or not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError('Background tolerance must be finite and positive')
    columns, data = class_background_table(background_path)
    for name in ("H", "Omega_m(z)", "Omega_r(z)"):
        if name not in columns:
            raise ValueError("CLASS background is missing required column " + name)

    redshift = data[:, columns['z']]
    if not np.isfinite(redshift).all() or np.any(redshift < 0):
        raise ValueError('CLASS background redshifts must be finite and nonnegative')
    order = np.argsort(redshift)
    if np.any(np.diff(redshift[order]) <= 0):
        raise ValueError('CLASS background redshifts must be distinct')
    z0_index = int(np.argmin(np.abs(redshift)))
    if abs(redshift[z0_index]) > 1.0e-8:
        raise ValueError("CLASS background output does not include z=0")
    h = float(cfg["cosmology"]["H0"]) / 100.0
    if not math.isfinite(h) or h <= 0:
        raise ValueError('Native Hubble parameter must be finite and positive')
    omega_m_input = (float(cfg["cosmology"]["omega_b"]) + float(cfg["cosmology"]["omega_cdm"])) / h**2
    omega_m_class = float(data[z0_index, columns["Omega_m(z)"]])
    omega_r_class = float(data[z0_index, columns["Omega_r(z)"]])
    omega_lambda = 1.0 - omega_m_input - omega_r_class
    if (not all(math.isfinite(value) for value in (omega_m_input, omega_m_class, omega_r_class, omega_lambda))
            or omega_m_input <= 0 or omega_m_class <= 0 or omega_r_class < 0):
        raise ValueError('Native background requires finite positive matter and nonnegative radiation densities')
    if not math.isclose(omega_m_class, omega_m_input, rel_tol=tolerance, abs_tol=1.0e-10):
        raise ValueError(
            "CLASS matter density at z=0 (%.12g) does not match the GADGET matter input (%.12g)"
            % (omega_m_class, omega_m_input)
        )
    if omega_lambda <= 0:
        raise ValueError("The flat GADGET matter+radiation+Lambda closure gives non-positive OmegaLambda")

    h0_class = float(data[z0_index, columns["H"]])
    if not math.isfinite(h0_class) or h0_class <= 0:
        raise ValueError("CLASS background has a non-positive Hubble rate at z=0")
    h0_native = h * 100.0 / 299792.458  # CLASS H is in 1/Mpc with c=1.
    if not math.isclose(h0_class, h0_native, rel_tol=tolerance, abs_tol=0.0):
        raise ValueError('CLASS H0 disagrees with the configured native Hubble rate')
    low_z = float(end_redshift)
    high_z = float(start_redshift)
    if not (0.0 <= low_z <= high_z):
        raise ValueError("The GADGET background validation redshift interval is invalid")
    use = (redshift >= low_z) & (redshift <= high_z)
    if np.min(redshift) > low_z or np.max(redshift) < high_z:
        raise ValueError("CLASS background table does not cover the complete GADGET redshift interval")

    # Include exact requested endpoints; an interior-only comparison can miss
    # the very epochs at which native evolution starts or ends.
    samples = np.unique(np.concatenate(([low_z], redshift[use], [high_z])))
    one_plus_z = 1.0 + samples
    h_gadget = h0_native * np.sqrt(
        omega_m_input * one_plus_z**3 + omega_r_class * one_plus_z**4 + omega_lambda
    )
    source_h = data[:, columns['H']]
    if np.any(~np.isfinite(source_h)) or np.any(source_h <= 0):
        raise ValueError("CLASS background contains non-finite or non-positive Hubble values")
    h_class = np.interp(samples, redshift[order], source_h[order])
    residual = h_gadget / h_class - 1.0
    if not np.isfinite(residual).all():
        raise ValueError('Native/CLASS Hubble comparison is not numerically representable')
    max_index = int(np.argmax(np.abs(residual)))
    max_residual = float(abs(residual[max_index]))
    if max_residual > float(tolerance):
        raise ValueError(
            "CLASS and GADGET H(a) differ by %.6g at z=%.8g, exceeding tolerance %.6g; "
            "the configured cosmology includes background physics this GADGET model does not evolve"
            % (max_residual, samples[max_index], tolerance)
        )
    return {
        "status": "PASS",
        "model": "matter+radiation+cosmological constant",
        "omega_m0": omega_m_input,
        "omega_r0": omega_r_class,
        "omega_lambda0": omega_lambda,
        "native_H0_km_s_Mpc": h * 100.0,
        "class_H0_km_s_Mpc": h0_class * 299792.458,
        "redshift_interval": [low_z, high_z],
        "max_fractional_hubble_residual": max_residual,
        "max_residual_redshift": float(samples[max_index]),
        "endpoint_interpolation": "linear in CLASS redshift; source sampling error not bounded",
        "tolerance": float(tolerance),
    }


def validate_lcdm_growth_rate(background_path, cfg, start_redshift, end_redshift=0.0, tolerance=1.0e-3):
    """Compare GADGET's radiation-aware smooth-growth ODE with CLASS over the run."""
    from scipy.integrate import solve_ivp
    if isinstance(tolerance, (bool, np.bool_)) or not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError('Growth tolerance must be finite and positive')

    columns, data = class_background_table(background_path)
    for name in ("growth_D", "growth_f"):
        if name not in columns:
            raise ValueError("CLASS background is missing its linear " + name + " column")

    high_z = float(start_redshift)
    low_z = float(end_redshift)
    if not (0.0 <= low_z <= high_z):
        raise ValueError("The growth validation redshift interval is invalid")
    z_table = data[:, 0]
    if not np.isfinite(z_table).all() or np.any(z_table < 0) or len(np.unique(z_table)) != len(z_table):
        raise ValueError('CLASS growth redshifts must be finite, nonnegative and distinct')
    if np.min(z_table) > low_z or np.max(z_table) < high_z:
        raise ValueError("CLASS background table does not cover the complete growth-validation interval")

    # Include the requested endpoints even when CLASS's logarithmic output grid
    # does not land on them exactly.  The CLASS reference is linearly
    # interpolated in redshift, matching the interpolation used for its other
    # tabulated background quantities.
    z_samples = np.concatenate(([low_z], z_table[(z_table > low_z) & (z_table < high_z)], [high_z]))
    z_samples = np.unique(z_samples)
    z_order = np.argsort(z_table)
    d_class = np.interp(z_samples, z_table[z_order], data[z_order, columns["growth_D"]])
    f_class = np.interp(z_samples, z_table[z_order], data[z_order, columns["growth_f"]])
    if np.any(~np.isfinite(d_class)) or np.any(d_class <= 0):
        raise ValueError("CLASS background has invalid linear growth factors in the requested interval")
    if np.any(~np.isfinite(f_class)) or np.any(f_class <= 0):
        raise ValueError("CLASS background has invalid linear growth rates in the requested interval")

    c = cfg["cosmology"]
    h = float(c["H0"]) / 100.0
    omega_m = (float(c["omega_b"]) + float(c["omega_cdm"])) / h**2
    omega_r = omega_radiation_from_background(background_path)
    omega_lambda = 1.0 - omega_m - omega_r
    omega_k = 1.0 - omega_m - omega_r - omega_lambda
    amin = 1.0e-5
    if 1.0 / (1.0 + high_z) < amin:
        raise ValueError('Native smooth-growth validation cannot extrapolate before its a=1e-5 seed; use a supported tabulated-background route')
    aeq = omega_r / omega_m
    y = amin / aeq
    initial = (1.0 + 1.5 * y, 1.5 / aeq)
    a_end = 1.0 / (1.0 + low_z)

    def derivatives(a, state):
        e2 = omega_m / a**3 + omega_r / a**4 + omega_k / a**2 + omega_lambda
        dln_e_da = (-3.0 * omega_m / a**4 - 4.0 * omega_r / a**5 - 2.0 * omega_k / a**3) / (2.0 * e2)
        omega_m_a = (omega_m / a**3) / e2
        return [state[1], -(3.0 / a + dln_e_da) * state[1] + 1.5 * omega_m_a * state[0] / a**2]

    solution = solve_ivp(
        derivatives,
        (amin, a_end),
        initial,
        rtol=1.0e-10,
        atol=1.0e-12,
        dense_output=True,
    )
    if not solution.success:
        raise ValueError("Radiation-aware linear-growth ODE failed: " + solution.message)

    a_samples = 1.0 / (1.0 + z_samples)
    growth_solution = solution.sol(a_samples)
    d_gadget = growth_solution[0]
    f_gadget = a_samples * growth_solution[1] / d_gadget
    if (not np.isfinite(d_gadget).all() or np.any(d_gadget <= 0)
            or not np.isfinite(f_gadget).all() or np.any(f_gadget <= 0)):
        raise ValueError('Native growth comparison returned nonfinite or nonpositive values')
    # CLASS and GADGET may choose different arbitrary normalizations for the
    # growing mode. Compare its shape after normalizing both at the final
    # simulation redshift; compare f=dln(D)/dln(a) without normalization.
    d_gadget /= float(solution.sol(a_end)[0])
    d_class /= float(np.interp(low_z, z_table[z_order], data[z_order, columns["growth_D"]]))
    d_relative = np.abs(d_gadget / d_class - 1.0)
    f_relative = np.abs(f_gadget / f_class - 1.0)
    d_worst = int(np.argmax(d_relative))
    f_worst = int(np.argmax(f_relative))
    max_relative = float(max(d_relative[d_worst], f_relative[f_worst]))
    result = {
        "status": "PASS" if max_relative <= tolerance else "FAIL",
        "method": "regular Meszaros seed plus smooth matter-radiation growth ODE",
        "redshift_interval": [low_z, high_z],
        "max_relative_growth_factor_difference": float(d_relative[d_worst]),
        "growth_factor_worst_redshift": float(z_samples[d_worst]),
        "max_relative_growth_rate_difference": float(f_relative[f_worst]),
        "growth_rate_worst_redshift": float(z_samples[f_worst]),
        "growth_rate_gadget_ode_at_start": float(f_gadget[-1]),
        "growth_rate_class_at_start": float(f_class[-1]),
        "max_relative_difference": max_relative,
        "tolerance": float(tolerance),
    }
    if result["status"] != "PASS":
        raise ValueError(
            "Radiation-aware GADGET growth differs from CLASS by %.6g over z=[%.8g, %.8g]"
            % (max_relative, low_z, high_z)
        )
    return result


def smooth_particle_growth_history(cfg, scale_factors, background_path=None):
    """Return normalized D(a) and f(a) for the smooth cold-particle model.

    CLASS's default background growth helper omits non-cold relics from its
    pressureless source. That is appropriate for its free-streaming transfer
    calculation, but not for this project's explicitly cold-particle WDM
    evolution after the transfer cutoff has been applied. The particle solver
    evolves all matter as cold after the IC epoch, so its growth table uses the
    radiation-aware smooth matter ODE with the WDM density included in Omega_m.
    """
    from scipy.integrate import solve_ivp

    query_a = np.asarray(scale_factors, dtype=np.float64)
    if (query_a.ndim != 1 or len(query_a) == 0 or np.any(~np.isfinite(query_a))
            or np.any(query_a <= 0) or np.any(query_a > 1.0)):
        raise ValueError("Growth scale factors must be a nonempty finite vector in (0, 1]")
    c = cfg["cosmology"]
    h = float(c["H0"]) / 100.0
    omega_m = (float(c["omega_b"]) + float(c["omega_cdm"])) / h**2
    omega_r = omega_radiation_from_background(background_path) if background_path else omega_radiation_from_config(cfg)
    omega_lambda = 1.0 - omega_m - omega_r
    omega_k = 1.0 - omega_m - omega_r - omega_lambda
    if not all(math.isfinite(value) for value in (omega_m, omega_r, omega_lambda)) or omega_m <= 0 or omega_r <= 0:
        raise ValueError("Smooth particle growth needs positive finite matter and radiation densities")

    a_seed = min(1.0e-6, 1.0e-5 * omega_r / omega_m)
    # CLASS can start its dense background grid slightly before the exact
    # monofonIC seed. Starting the regular Meszaros solution at that earlier
    # sample lets the interpolation audit include every source row.
    a_seed = min(a_seed, float(np.min(query_a)))
    a_eq = omega_r / omega_m
    y = a_seed / a_eq
    initial = (1.0 + 1.5 * y, 1.5 / a_eq)

    def derivatives(a, state):
        e2 = omega_m / a**3 + omega_r / a**4 + omega_k / a**2 + omega_lambda
        dln_e_da = (-3.0 * omega_m / a**4 - 4.0 * omega_r / a**5 - 2.0 * omega_k / a**3) / (2.0 * e2)
        omega_m_a = (omega_m / a**3) / e2
        return [state[1], -(3.0 / a + dln_e_da) * state[1] + 1.5 * omega_m_a * state[0] / a**2]

    solution = solve_ivp(
        derivatives, (a_seed, 1.0), initial, rtol=1.0e-10, atol=1.0e-12,
        dense_output=True,
    )
    if not solution.success:
        raise ValueError("Smooth particle-growth ODE failed: " + solution.message)
    state = solution.sol(query_a)
    today = float(solution.sol(1.0)[0])
    growth = state[0] / today
    rate = query_a * state[1] / state[0]
    if np.any(~np.isfinite(growth)) or np.any(growth <= 0) or np.any(~np.isfinite(rate)) or np.any(rate <= 0):
        raise ValueError("Smooth particle-growth ODE returned invalid growth values")
    return growth.astype(np.float64), rate.astype(np.float64)


def smooth_particle_growth_rate(cfg, redshift, background_path=None):
    """Return f=d ln D/d ln a for the smooth cold-particle growth model."""
    z = float(redshift)
    if not math.isfinite(z) or z < 0:
        raise ValueError("Growth redshift must be finite and non-negative")
    _growth, rate = smooth_particle_growth_history(
        cfg, [1.0 / (1.0 + z)], background_path,
    )
    return float(rate[0])


def ede_parameters(cfg):
    """Map axion f_EDE, z_c, theta_i [rad] and potential exponent to protected AxiCLASS input keys."""
    e = cfg["ede"]
    parameterization = str(e.get("parameterization", "axion")).strip().lower()
    if parameterization != "axion":
        raise ValueError(
            "Unsupported ede.parameterization %r; this code implements only 'axion'"
            % e.get("parameterization")
        )

    try:
        f_ede = float(e["f_ede"])
        z_c = float(e["z_c"])
        theta_i = float(e["theta_i"])
        n_axion = float(e.get("n_axion", 3))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("EDE f_ede, z_c, theta_i, and n_axion must be numeric") from exc
    if not math.isfinite(f_ede) or not 0.0 < f_ede < 1.0:
        raise ValueError("ede.f_ede must be finite and strictly between 0 and 1")
    if not math.isfinite(z_c) or z_c <= 0.0:
        raise ValueError("ede.z_c must be finite and positive")
    if not math.isfinite(theta_i) or not 0.0 < theta_i < math.pi:
        raise ValueError("ede.theta_i must be finite and strictly between 0 and pi radians")
    if not math.isfinite(n_axion) or n_axion < 2.0:
        raise ValueError("ede.n_axion must be finite and >= 2 for the configured full-scalar AxiCLASS path")
    a_c = 1.0 / (1.0 + z_c)
    # AxiCLASS supports the axion EDE parameterization using fraction_axion_ac
    # and log10_axion_ac. theta_i is the first scalar-field initial condition.
    params = {
        "scf_potential": "axion",
        # AxiCLASS defines n as a real exponent; e.g. its official example uses
        # n_axion=2.6. Keeping this as a float avoids silently changing models.
        "n_axion": n_axion,
        "fraction_axion_ac": f_ede,
        "log10_axion_ac": math.log10(a_c),
        "scf_parameters": f"{theta_i},0.0",
        # Upstream's n>=2 example uses the full scalar-field evolution rather
        # than switching between scalar and fluid approximations.
        "scf_evolve_as_fluid": "no",
        "scf_evolve_like_axionCAMB": "no",
        "do_shooting": "yes",
        "do_shooting_scf": "yes",
        "scf_has_perturbations": "yes",
        "attractor_ic_scf": "no",
        "loop_over_background_for_closure_relation": "yes",
        "include_scf_in_delta_m": "no",
        "nbody_gauge_transfer_functions": "yes",
        "include_scf_in_delta_cb": "no",
        "background_Nloga": 100000,
    }
    extra = e.get("extra_axiclass", {}) or {}
    if not isinstance(extra, dict):
        raise ValueError("ede.extra_axiclass must be a mapping of AxiCLASS input names to values")
    protected = set(params) | {
        "output", "gauge", "nbody_gauge_transfer_functions",
    }
    overridden = sorted(protected.intersection(extra))
    if overridden:
        raise ValueError(
            "ede.extra_axiclass cannot override parameters that define the axion model or validated transfer outputs: %s"
            % ", ".join(overridden)
        )
    params.update(extra)
    return params


def validate_wdm_mode(cfg):
    """Reject WDM mode strings without an implemented CLASS mapping."""
    mode = cfg.get("wdm", {}).get("mode")
    if mode != "class_ncdm":
        raise ValueError(
            "Unsupported wdm.mode %r; the only implemented transfer-function mode is 'class_ncdm'"
            % mode
        )


def validate_wdm_representation(cfg):
    """Validate the selected WDM particle model and its linear spectrum source."""
    validate_wdm_mode(cfg)
    w = cfg.get("wdm", {})
    representation = w.get("model")
    supported = {"wdm_transfer_only_benchmark", "wdm_thermal_relic"}
    if representation not in supported:
        raise ValueError(
            "wdm.model must explicitly select 'wdm_transfer_only_benchmark' or 'wdm_thermal_relic'"
        )
    spectrum_source = str(cfg.get("spectrum", {}).get("source", "boltzmann")).strip().lower()
    if representation == "wdm_thermal_relic" and spectrum_source not in {"boltzmann", "modular"}:
        raise ValueError(
            "wdm_thermal_relic requires the run-matched CLASS ncdm spectrum; an imported spectrum "
            "does not establish consistency with the configured relic abundance and phase-space distribution"
        )
    max_beta = float(w.get("max_thermal_speed_fraction_c", 0.01))
    if not math.isfinite(max_beta) or not 0.0 < max_beta < 1.0:
        raise ValueError("wdm.max_thermal_speed_fraction_c must be finite and in (0,1)")


def _wdm_cmb_temperature(cfg):
    """Use the same photon temperature that class_common submits to CLASS."""
    cosmology = cfg['cosmology']
    extra = cosmology.get('extra', {}) or {}
    if not isinstance(extra, dict):
        raise ValueError('cosmology.extra must be a mapping of CLASS parameters')
    value = extra.get('T_cmb', cosmology.get('T_cmb', 2.7255))
    if isinstance(value, (bool, np.bool_)):
        raise ValueError('WDM photon temperature must be finite and positive')
    temperature = float(value)
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('WDM photon temperature must be finite and positive')
    return temperature


def wdm_parameters(cfg):
    """Translate the declared relic abundance/mass into CLASS ncdm parameters without a fitted substitute spectrum."""
    validate_wdm_representation(cfg)
    w = cfg["wdm"]
    mass_keV = float(w["thermal_relic_mass_keV"])
    if not math.isfinite(mass_keV) or mass_keV <= 0:
        raise ValueError("wdm.thermal_relic_mass_keV must be finite and positive")
    mass_eV = mass_keV * 1000.0
    if not math.isfinite(mass_eV):
        raise ValueError('WDM relic mass conversion to eV overflows')
    # For one Fermi-Dirac thermal relic, rho_x is proportional to m_x T_x^3.
    # Choose T_x so CLASS derives the requested physical density from the relic
    # mass and distribution, rather than forcing omega_ncdm and renormalizing
    # the phase-space distribution to a non-thermal abundance.
    target_omega_dm = float(cfg["cosmology"]["omega_cdm"])
    standard_t = float(cfg["cosmology"].get("T_ncdm", 0.71611))
    if not math.isfinite(target_omega_dm) or target_omega_dm <= 0:
        raise ValueError("cosmology.omega_cdm must be finite and positive for a thermal-relic WDM mapping")
    if not math.isfinite(standard_t) or standard_t <= 0:
        raise ValueError("cosmology.T_ncdm must be finite and positive for a thermal-relic WDM mapping")
    if standard_t != 0.71611:
        raise ValueError('Thermal-relic temperature is derived from mass and abundance; remove custom cosmology.T_ncdm')
    # Pinned CLASS input.c declares m/omega≈93.14 eV at T_ncdm=0.71611,
    # T_cmb=2.7255 K and deg_ncdm=1. Number density scales as T_x^3.
    # T_ncdm is a ratio to the actual CMB temperature, not a temperature in K.
    temperature = standard_t * (target_omega_dm * 93.14 / mass_eV) ** (1.0 / 3.0) * (2.7255 / _wdm_cmb_temperature(cfg))
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('Derived WDM temperature is not representable')
    params = {
        "omega_cdm": 1.0e-15,
        "N_ncdm": 1,
        "m_ncdm": mass_eV,
        "T_ncdm": temperature,
        "deg_ncdm": 1,
        "ksi_ncdm": 0,
    }
    extra = w.get("extra_class", {}) or {}
    if not isinstance(extra, dict):
        raise ValueError("wdm.extra_class must be a mapping of CLASS input names to values")
    protected = set(params) | {
        "output", "gauge", "nbody_gauge_transfer_functions",
        "omega_ncdm", "Omega_ncdm", "use_ncdm_psd_files",
        "ncdm_psd_filenames", "ncdm_psd_parameters", "T_cmb", "Omega_g", "omega_g",
    }
    cosmology_extra = cfg['cosmology'].get('extra', {}) or {}
    incompatible = (protected - {'output', 'gauge', 'nbody_gauge_transfer_functions', 'T_cmb'}).intersection(cosmology_extra)
    if incompatible:
        raise ValueError('cosmology.extra cannot redefine the normalized thermal relic: '+', '.join(sorted(incompatible)))
    overridden = sorted(protected.intersection(extra))
    if overridden:
        raise ValueError(
            "wdm.extra_class cannot override parameters that define the normalized thermal relic or validated transfer outputs: %s"
            % ", ".join(overridden)
        )
    params.update(extra)
    return params


_FD_MOMENTUM_GRID = None


def _fermi_dirac_momentum_table():
    """Fixed-grid CDF for p/(k_B T) under f(p) d^3p ∝ p²/(e^p+1).

    The grid ends at q=40; its exponentially small tail is truncated. A fixed
    prescription keeps chunking and repeated calls independent of call order.
    """
    global _FD_MOMENTUM_GRID
    if _FD_MOMENTUM_GRID is None:
        y = np.linspace(0.0, 40.0, 200001, dtype=np.float64)
        pdf = y * y / (np.exp(y) + 1.0)
        increments = 0.5 * (pdf[1:] + pdf[:-1]) * np.diff(y)
        cdf = np.concatenate(([0.0], np.cumsum(increments, dtype=np.float64)))
        cdf /= cdf[-1]
        _FD_MOMENTUM_GRID = (y, cdf)
    return _FD_MOMENTUM_GRID


def _counter_uniform(ids, seed, lane):
    """A deterministic counter-based SplitMix64 stream indexed by particle ID."""
    mask = np.uint64(0xFFFFFFFFFFFFFFFF)
    lane_offset = np.uint64((int(lane) * 0xD2B74407B1CE6E93) & 0xFFFFFFFFFFFFFFFF)
    with np.errstate(over="ignore"):
        x = np.asarray(ids, dtype=np.uint64) + np.uint64(seed) + lane_offset
        x = (x + np.uint64(0x9E3779B97F4A7C15)) & mask
        x = ((x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)) & mask
        x = ((x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)) & mask
        x ^= x >> np.uint64(31)
    return (x >> np.uint64(11)).astype(np.float64) * (1.0 / 9007199254740992.0)


def wdm_thermal_velocity_from_ids(particle_ids, cfg, scale_factor):
    """Sample Fermi-Dirac relic velocities in GADGET IC units (km/s / sqrt(a)).

    The relic temperature follows the same CLASS ncdm normalization as
    ``wdm_parameters``. CLASS defines the distribution in comoving momentum
    q=p/(k_B T_ncdm T_gamma,0); physical momentum is q k_B T_0/a. The exact
    special-relativistic momentum-to-speed relation is used. GADGET-4's HDF5
    IC convention stores u=v_pec/sqrt(a), as documented in its snapshot manual.
    See CLASS explanatory.ini (massive relic input) and GADGET-4 snapshot format.
    """
    ids = np.asarray(particle_ids)
    if ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer):
        raise ValueError("WDM thermal velocity sampling requires a one-dimensional integer ParticleIDs array")
    if np.issubdtype(ids.dtype, np.signedinteger) and np.any(ids < 0):
        raise ValueError("WDM thermal velocity sampling requires non-negative particle IDs")
    a = float(scale_factor)
    if isinstance(scale_factor, (bool, np.bool_)) or not math.isfinite(a) or a <= 0:
        raise ValueError("WDM thermal velocity scale factor must be finite and positive")
    w = cfg.get("wdm", {})
    mass_ev = float(w.get("thermal_relic_mass_keV", np.nan)) * 1000.0
    if not math.isfinite(mass_ev) or mass_ev <= 0:
        raise ValueError("wdm.thermal_relic_mass_keV must be finite and positive")
    relic = wdm_parameters(cfg)
    temperature_ratio = float(relic["T_ncdm"])
    cmb_kelvin = _wdm_cmb_temperature(cfg)
    k_b_ev_k = 8.617333262145e-5
    momentum_temperature_ev = k_b_ev_k * cmb_kelvin * temperature_ratio / a
    if not math.isfinite(momentum_temperature_ev) or momentum_temperature_ev <= 0:
        raise ValueError('WDM momentum temperature is not representable')
    y_grid, cdf_grid = _fermi_dirac_momentum_table()
    seed = cfg['simulation'].get('random_seed', 123456)
    if (isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer))
            or not 0 <= int(seed) <= 0xFFFFFFFFFFFFFFFF):
        raise ValueError('WDM random seed must be an exact uint64 integer')
    seed = int(seed)
    quantile = _counter_uniform(ids, seed, 0)
    y = np.interp(quantile, cdf_grid, y_grid)
    mu = 2.0 * _counter_uniform(ids, seed, 1) - 1.0
    phi = 2.0 * math.pi * _counter_uniform(ids, seed, 2)
    sin_theta = np.sqrt(np.maximum(0.0, 1.0 - mu * mu))
    p_ev = y * momentum_temperature_ev
    if not np.isfinite(p_ev).all():
        raise ValueError('WDM momentum is not representable')
    beta = p_ev / np.hypot(p_ev, mass_ev)
    speed_kms = beta * 299792.458
    velocity = np.column_stack((
        speed_kms * sin_theta * np.cos(phi),
        speed_kms * sin_theta * np.sin(phi),
        speed_kms * mu,
    )) / math.sqrt(a)
    if not np.isfinite(velocity).all():
        raise ValueError('WDM native velocity is not representable')
    return velocity, y, beta


def wdm_fd_speed_moments(cfg, scale_factor):
    """Numerically integrate mean, RMS, and 99.9-percentile relic speed."""
    relic = wdm_parameters(cfg)
    y, cdf = _fermi_dirac_momentum_table()
    mass_ev = float(cfg["wdm"]["thermal_relic_mass_keV"]) * 1000.0
    a = float(scale_factor)
    if isinstance(scale_factor, (bool, np.bool_)) or not math.isfinite(a) or a <= 0:
        raise ValueError('WDM speed moments require a finite positive scale factor')
    t_ev = 8.617333262145e-5 * _wdm_cmb_temperature(cfg) * float(relic['T_ncdm']) / a
    momentum = y * t_ev
    if not math.isfinite(t_ev) or t_ev <= 0 or not np.isfinite(momentum).all():
        raise ValueError('WDM momentum is not representable')
    beta = momentum / np.hypot(momentum, mass_ev)
    pdf_cdf = cdf
    density = y * y / (np.exp(y) + 1.0)
    norm = float(np.trapezoid(density, y))
    mean_beta = float(np.trapezoid(beta * density, y) / norm)
    mean_beta2 = float(np.trapezoid(beta * beta * density, y) / norm)
    q999 = float(np.interp(0.999, pdf_cdf, y))
    beta999 = q999 * t_ev / math.hypot(q999 * t_ev, mass_ev)
    c_kms = 299792.458
    return {
        "mean_speed_km_s": mean_beta * c_kms,
        "rms_speed_km_s": math.sqrt(mean_beta2) * c_kms,
        "speed_99_9_percentile_km_s": beta999 * c_kms,
        "mean_beta": mean_beta,
        "rms_beta": math.sqrt(mean_beta2),
        "speed_99_9_percentile_beta": beta999,
    }


@directory_owner('destination')
def write_wdm_velocity_transfer_table(transfer_path, background_path, destination, expected_redshift, baseline_growth_rate):
    """Write CLASS's scale-dependent WDM linear velocity response for N-GenIC.

    The table stores f_WDM(k)/f_GADGET(a), so the patched N-GenIC multiplies
    its scale-independent growing-mode velocity by the exact run-matched
    CLASS ncdm response while leaving density displacements and phases intact.
    """
    transfer_path = Path(transfer_path)
    source_hashes={str(transfer_path):sha256(transfer_path),str(background_path):sha256(background_path)}
    if not transfer_path.is_file():
        raise FileNotFoundError("CLASS WDM transfer file is missing: " + str(transfer_path))
    columns, transfer, observed_redshift = _class_transfer_table(transfer_path, expected_redshift)
    required = ("k", "d_m", "t_ncdm[0]")
    missing = [name for name in required if name not in columns]
    if missing:
        raise ValueError("CLASS WDM transfer lacks the required scale-dependent velocity columns: " + ", ".join(missing))
    k = transfer[:, columns["k"]]
    delta_m = transfer[:, columns["d_m"]]
    theta_wdm = transfer[:, columns["t_ncdm[0]"]]
    bg_columns, background = class_background_table(background_path)
    if "z" not in bg_columns or "H" not in bg_columns:
        raise ValueError("CLASS background must contain z and H for WDM velocity conversion")
    bg_z = background[:, bg_columns["z"]]
    order = np.argsort(bg_z)
    bg_z = bg_z[order]
    bg_h = background[order, bg_columns["H"]]
    h_start = float(np.interp(float(expected_redshift), bg_z, bg_h))
    a = 1.0 / (1.0 + float(expected_redshift))
    if not math.isfinite(h_start) or h_start <= 0 or not math.isfinite(baseline_growth_rate) or baseline_growth_rate <= 0:
        raise ValueError("CLASS background or GADGET WDM growth rate is invalid at the IC epoch")
    floor = 1.0e-14 * max(1.0, float(np.max(np.abs(delta_m))))
    f_wdm = -theta_wdm / (a * h_start * delta_m)
    correction = f_wdm / float(baseline_growth_rate)
    if (len(k) < 3 or np.any(~np.isfinite(k)) or np.any(k <= 0) or np.any(np.diff(k) <= 0)
            or np.any(~np.isfinite(correction)) or np.any(np.abs(delta_m) <= floor)
            or np.any(correction <= 0)):
        raise ValueError("CLASS WDM scale-dependent velocity response is invalid or changes sign")
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    from core.data import epoch_contract
    table_contract=publish_numeric_table(destination,np.column_stack((k,correction)),
        {'k':{'units':'h/Mpc','quantity':'Comoving Fourier wavenumber'},
         'velocity_response':{'units':'dimensionless','quantity':'CLASS ncdm growth response / GADGET baseline growth rate'}},
        algorithm=__file__,source_hashes=source_hashes,context={'epoch':epoch_contract(a),'gauge':'N-body',
            'baseline_growth_rate':float(baseline_growth_rate)},
        header='k[h/Mpc] f_wdm(k)/f_gadget(a); CLASS N-body-gauge ncdm theta divided by total-matter density')
    metadata = {
        "status": "READY",
        "model": "wdm_thermal_relic",
        "source_transfer_file": str(transfer_path.resolve()),
        "source_transfer_sha256": sha256(transfer_path),
        "source_background_file": str(Path(background_path).resolve()),
        "source_redshift": float(expected_redshift),
        "scale_factor": a,
        "baseline_gadget_growth_rate": float(baseline_growth_rate),
        "k_range_h_mpc": [float(k[0]), float(k[-1])],
        "response_range": [float(np.min(correction)), float(np.max(correction))],
        "formula": "f_wdm(k)=-theta_ncdm(k)/(a H delta_m(k)); N-GenIC multiplies its GADGET f(a) by f_wdm(k)/f_GADGET(a)",
        "application_scope": "Scale-dependent CLASS response is applied to N-GenIC first-order velocities; the upstream second-order LPT velocity remains the standard pressureless approximation",
        "velocity_transfer_table": str(destination.resolve()),
        "velocity_transfer_table_sha256": sha256(destination),
    }
    metadata['table_contract']=table_contract
    atomic_json(destination.with_suffix(destination.suffix+'.json'),metadata)
    return metadata


def write_ini(params, path):
    """Serialize explicit solver key/value inputs; no observational data or default cosmology is invented."""
    rows = []
    for key, value in params.items():
        if isinstance(value, bool): value = 'yes' if value else 'no'
        rows.append(str(key) + ' = ' + str(value) + '\n')
    atomic_text(path, ''.join(rows))


@native_run_owner
def run_boltzmann(cfg, run_dir, model, start_z, dry_run=False, kmax_hmpc=None):
    """Run the selected pinned solver, exporting linear spectra and background tables with epoch provenance."""
    run_dir = Path(run_dir).expanduser().resolve()
    input_dir = run_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    box = float(cfg["simulation"]["box_mpc_h"])
    n = int(cfg["simulation"]["particles_per_dimension"])
    kmin, mesh_kmax = k_limits(box, n)
    kmax = max(mesh_kmax, float(kmax_hmpc or 0.0))

    engine = "axiclass" if model == "ede" else "class"
    binary = executable(engine)
    # CLASS can leave several indexed outputs behind when AUTO evaluates more
    # than one trial redshift in the same run directory. Isolate every solver
    # call and select only files produced for this exact requested redshift.
    output_dir = input_dir / "boltzmann_runs"
    output_dir.mkdir(parents=True, exist_ok=True)
    z_token = ("%.10g" % float(start_z)).replace("-", "m").replace(".", "p")
    root = output_dir / (model + "_z" + z_token + "_" + uuid.uuid4().hex[:10] + "_")
    params = class_common(cfg, start_z, root, kmax)

    if model == "ede":
        params.update(ede_parameters(cfg))
    elif model == "wdm":
        params.update(wdm_parameters(cfg))
    # A user extra may request more k coverage, but it must never truncate the
    # configured particle mesh or an adaptive WDM half-mode search.
    params["P_k_max_h/Mpc"] = max(kmax, float(params.get("P_k_max_h/Mpc", 0.0)))

    ini = output_dir / (root.name + "solver.ini")
    write_ini(params, ini)

    if dry_run:
        return {"model": model, "ini": str(ini), "kmin": kmin, "kmax": kmax}

    if binary is None:
        raise RuntimeError(engine + " executable not found. Run bootstrap first.")

    with native_run_lock(binary.parent, operation='Boltzmann solver and provenance'):
        log = output_dir / (root.name + "boltzmann.log")
        with open(log, "w", encoding="utf-8") as out:
            from core.native import run_native_command
            source_manifest = native_source_manifest(engine, binary.parent)
            run_native_command([str(binary), str(ini)], cwd=binary.parent,
                receipt_path=output_dir / (root.name + 'invocation.json'),
                stdout=out, stderr=subprocess.STDOUT, check=True)
            verify_native_source_manifest(source_manifest)

        outputs = class_output_files(root, start_z)
        pk = outputs['pk']
        validate_pk(pk, kmin, kmax)

        meta = {
            "model": model,
            "engine": engine,
            "perturbation_solver_gauge": "synchronous",
            "transfer_function_gauge": "N-body",
            "nbody_gauge_transfer_functions": True,
            "start_redshift": float(start_z),
            "output_redshift": pk_output_redshift(pk),
            "k_required_min_h_Mpc": kmin,
            "k_required_max_h_Mpc": kmax,
            "particle_mesh_required_max_h_Mpc": mesh_kmax,
            "pk_file": str(pk),
            "pk_sha256": sha256(pk),
            "transfer_file": str(outputs["transfer"]),
            "background_file": str(outputs["background"]),
            "solver_ini_file": str(ini.resolve()),
            "solver_log_file": str(log.resolve()),
            "native_invocation": {"path":str(output_dir / (root.name + "invocation.json")),
                                  "sha256":sha256(output_dir / (root.name + "invocation.json"))},
            "native_source_manifest": source_manifest,
            "science_acceptance": "NOT_ACCEPTED",
        }
        if meta["background_file"]:
            meta["background_sha256"] = sha256(meta["background_file"])
            meta["omega_radiation0"] = omega_radiation_from_background(meta["background_file"])
            if model in {"lcdm", "wdm"}:
                background_tolerance = float(
                    cfg.get("validation", {}).get("background", {}).get("max_relative_hubble_error", 1.0e-4)
                )
                meta["background_validation"] = validate_gadget_background(
                    meta["background_file"],
                    cfg,
                    start_z,
                    float(cfg.get("simulation", {}).get("end_redshift", 0.0)),
                    background_tolerance,
                )
            if model == "lcdm":
                growth_tolerance = float(
                    cfg.get("validation", {}).get("background", {}).get("max_relative_growth_rate_error", 1.0e-3)
                )
                meta["growth_validation"] = validate_lcdm_growth_rate(
                    meta["background_file"],
                    cfg,
                    start_z,
                    float(cfg.get("simulation", {}).get("end_redshift", 0.0)),
                    growth_tolerance,
                )
            if model == "ede":
                table_path = input_dir / "ede_gadget_background.txt"
                meta["gadget_background_validation"] = write_gadget_background_table(
                    meta["background_file"], table_path, min(1.0 / (1.0 + float(start_z)), 1.0e-3),
                    expected_h0_km_s_mpc=cfg['cosmology']['H0'],
                )
                meta["gadget_background_file"] = str(table_path.resolve())
                meta["gadget_background_sha256"] = sha256(table_path)
                meta["particle_model_scope"] = (
                    "AxiCLASS linear transfer functions and smooth H(a), D(a), f(a) table; "
                    "GADGET evolves matter particles and does not evolve nonlinear EDE scalar-field modes"
                )
        products_path = input_dir / (model + "_cosmology_products.h5")
        meta["cosmology_products_file"] = write_class_cosmology_products(
            model,
            start_z,
            meta.get("background_file"),
            meta.get("transfer_file"),
            meta.get("pk_file"),
            products_path,
        )
        meta["solver_provenance"] = solver_execution_provenance(
            engine,
            binary,
            ini,
            log,
            {
                "linear_power_spectrum": pk,
                "transfer_functions": meta.get("transfer_file"),
                "boltzmann_background": meta.get("background_file"),
                "canonical_cosmology_products": meta.get("cosmology_products_file"),
                "gadget_background_table": meta.get("gadget_background_file"),
                "monofonic_background_table": meta.get("monofonic_background_file"),
            },
            start_z,
        )
        provenance_path = output_dir / (root.name + "provenance.json")
        atomic_json(provenance_path, meta['solver_provenance'])
        atomic_text(provenance_path.with_name(provenance_path.name + '.sha256'), sha256(provenance_path) + '\n')
        meta["solver_provenance_file"] = str(provenance_path.resolve())
        atomic_json(input_dir / (model + '_cosmology.json'), meta)
        return meta


@directory_owner('destination')
def write_species_transfer_validation_table(source_path, destination, cfg, model, expected_redshift):
    """Write run-matched matter and species transfers for IC generation/checks.

    The CLASS/AxiCLASS transfer output is parsed by its column names rather
    than assuming that every model writes the same species in the same order.
    This validation table is not consumed directly by an IC generator. Its raw
    CLASS/AxiCLASS amplitudes retain the configured primordial normalization;
    a separate staging function can convert it to the monofonIC plugin format.
    """
    source_path = Path(source_path)
    if not source_path.is_file():
        raise FileNotFoundError("Run-matched CLASS/AxiCLASS transfer output is missing: " + str(source_path))

    source_before=sha256(source_path)
    columns, table, observed_redshift = _class_transfer_table(source_path, expected_redshift)
    for required in ("k", "d_b", "d_cdm", "t_b",
                     "h_t_nb_prime", "k2gamma_nb"):
        if required not in columns:
            raise ValueError("CLASS/AxiCLASS transfer output is missing the required %s column" % required)
    if model in {"lcdm", "ede"} and "t_cdm" not in columns:
        raise ValueError("%s transfer output is missing the t_cdm velocity column" % model.upper())

    k = table[:, columns["k"]]
    delta_b = table[:, columns["d_b"]]
    delta_cdm = table[:, columns["d_cdm"]]
    theta_b = table[:, columns["t_b"]]
    # In the transfer-function-only WDM model, the relic is the particle dark
    # matter species and CLASS's cold-CDM density is intentionally negligible.
    # Its t_cdm transfer is consequently absent; retain the stable output
    # column as NaN and validate the actual relic with t_ncdm[0].
    theta_cdm = (
        table[:, columns["t_cdm"]]
        if "t_cdm" in columns else np.full_like(delta_cdm, np.nan)
    )
    if model == "wdm":
        if "d_ncdm[0]" not in columns or "t_ncdm[0]" not in columns:
            raise ValueError("WDM transfer output is missing the d_ncdm[0]/t_ncdm[0] relic-particle transfers")
        if "d_m" not in columns:
            raise ValueError("WDM transfer output is missing CLASS's total-matter d_m transfer")
        delta_wdm = table[:, columns["d_ncdm[0]"]]
        theta_wdm = table[:, columns["t_ncdm[0]"]]
        delta_m = table[:, columns["d_m"]]
        matter_method = "CLASS d_m, which includes the configured non-cold relic matter"
    elif model in {"lcdm", "ede"}:
        h = float(cfg["cosmology"]["H0"]) / 100.0
        omega_b = float(cfg["cosmology"]["omega_b"]) / h**2
        omega_cdm = float(cfg["cosmology"]["omega_cdm"]) / h**2
        omega_m = omega_b + omega_cdm
        if not math.isfinite(omega_m) or omega_m <= 0:
            raise ValueError("%s species validation needs positive baryon plus CDM density" % model.upper())
        delta_m = (omega_b * delta_b + omega_cdm * delta_cdm) / omega_m
        delta_wdm = np.full_like(delta_m, np.nan)
        theta_wdm = np.full_like(delta_m, np.nan)
        matter_method = "(Omega_b delta_b + Omega_cdm delta_cdm)/(Omega_b + Omega_cdm); excludes radiation perturbations"
    else:
        raise ValueError("Species transfer validation table supports LCDM, WDM, or EDE inputs")

    # Keep the first five columns stable because the IC power validator uses
    # them. CLASS theta has units of inverse Mpc and is explicitly transformed
    # into N-body gauge by the solver; the particle validator combines it with
    # the same run's H(a).
    result = np.column_stack([
        k, delta_cdm, delta_b, delta_m, delta_wdm,
        theta_cdm, theta_b, theta_wdm,
    ])
    if (not np.all(np.isfinite(result[:, :4])) or np.any(k <= 0)
            or np.any(np.diff(k) <= 0) or np.any(np.abs(delta_m) <= np.finfo(float).tiny)):
        raise ValueError("CLASS/AxiCLASS species transfers contain invalid k or density values")
    if model == "wdm" and not np.all(np.isfinite(result[:, [4, 7]])):
        raise ValueError("CLASS WDM transfer contains invalid d_ncdm[0]/t_ncdm[0] values")
    if not np.all(np.isfinite(result[:, [6]])):
        raise ValueError("CLASS/AxiCLASS baryon velocity transfer contains invalid values")
    if model in {"lcdm", "ede"} and not np.all(np.isfinite(result[:, [5]])):
        raise ValueError("CLASS/AxiCLASS %s CDM velocity transfer contains invalid values" % model.upper())
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    from core.data import epoch_contract
    names=['k_h_per_Mpc','delta_cdm','delta_b','delta_m','delta_wdm',
           'theta_cdm_Mpc^-1','theta_b_Mpc^-1','theta_wdm_Mpc^-1']
    specifications={name:{'units':'h/Mpc' if index==0 else 'dimensionless' if index<5 else 'Mpc^-1',
        'quantity':'N-body-gauge CLASS species reference '+name,
        'missing_policy':'NaN for model-unavailable WDM species' if index in (4,7) and model!='wdm' else 'No missing values'}
        for index,name in enumerate(names)}
    table_contract=publish_numeric_table(destination,result,specifications,algorithm=__file__,
        source_hashes={str(source_path):source_before},
        context={'epoch':epoch_contract(1/(1+float(expected_redshift))),'model':model,'gauge':'N-body',
            'matter_transfer_method':matter_method,'purpose':'Validation reference; not direct IC input'},
        header=' '.join(names)+' (validation only; not an IC input)')
    metadata = {
        "status": "READY",
        "model": model,
        "perturbation_solver_gauge": "synchronous",
        "transfer_function_gauge": "N-body",
        "theta_convention": "CLASS vTk theta_i transformed into N-body gauge; converted with -theta_i/(a H) for the GADGET peculiar-velocity divergence",
        "source_transfer_file": str(source_path.resolve()),
        "source_transfer_sha256": sha256(source_path),
        "source_redshift": observed_redshift,
        "expected_redshift": float(expected_redshift),
        "matter_transfer_method": matter_method,
        "columns": [
            "k_h_per_Mpc", "delta_cdm", "delta_b", "delta_m", "delta_wdm",
            "theta_cdm_Mpc^-1", "theta_b_Mpc^-1", "theta_wdm_Mpc^-1",
        ],
        "validation_table": str(destination.resolve()),
        "validation_table_sha256": sha256(destination),
        "purpose": "Validation-only species reference; not an IC generator input. Convert it explicitly before staging any generator inputs.",
    }
    metadata['table_contract']=table_contract
    atomic_json(str(destination)+'.json',metadata)
    return metadata


@directory_owner('destination')
def write_monofonic_ic_transfer_table(
    validation_table_path, destination, model, expected_redshift, cfg, background_path,
):
    """Stage N-body-gauge CLASS species transfers for the monofonIC file plugin.

    The tabulated fields are the run-matched linear density contrasts and
    velocity divergences at the IC epoch. They are deliberately kept in CLASS
    units here; the pinned monofonIC plugin applies its documented
    ``-T(k)/k^2`` convention and A_s normalization when evaluating them.
    """
    if model not in {"lcdm", "wdm", "ede"}:
        raise ValueError("monofonIC tabulated hydro transfers support LCDM, WDM, or EDE linear inputs")
    validation_table_path = Path(validation_table_path).expanduser().resolve()
    metadata_path = Path(str(validation_table_path) + ".json")
    if not validation_table_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(
            "The model-matched species validation table and provenance are required before staging monofonIC inputs"
        )
    source_hashes={str(path):sha256(path) for path in (validation_table_path,metadata_path,Path(background_path))}
    source_meta = json.loads(metadata_path.read_text(encoding="utf-8"))
    if source_meta.get("status") != "READY" or source_meta.get("model") != model:
        raise ValueError("Species validation table provenance does not match the requested monofonIC model")
    if (Path(source_meta.get("validation_table", "")).expanduser().resolve() != validation_table_path
            or source_meta.get("validation_table_sha256") != sha256(validation_table_path)):
        raise ValueError("Species validation table contents do not match their recorded provenance hash")
    if not class_redshift_matches(source_meta.get("source_redshift"), expected_redshift):
        raise ValueError("Species validation table redshift does not match the monofonIC IC epoch")
    background_path = Path(background_path).expanduser().resolve()
    background_columns, background_data = class_background_table(background_path)
    if not all(name in background_columns for name in ("z", "H", "growth_f")):
        raise ValueError("The monofonIC velocity normalization needs CLASS H(a) and f(a) background columns")
    background_z = np.asarray(background_data[:, background_columns["z"]], dtype=np.float64)
    background_a = 1.0 / (1.0 + background_z)
    background_order = np.argsort(background_a)
    background_a = background_a[background_order]
    background_H = np.asarray(background_data[background_order, background_columns["H"]], dtype=np.float64)
    background_f = np.asarray(background_data[background_order, background_columns["growth_f"]], dtype=np.float64)
    a_start = 1.0 / (1.0 + float(expected_redshift))
    if (len(background_a) < 2 or a_start < background_a[0] or a_start > background_a[-1]
            or np.any(~np.isfinite(background_H)) or np.any(background_H <= 0)
            or np.any(~np.isfinite(background_f)) or np.any(background_f <= 0)):
        raise ValueError("CLASS/AxiCLASS background does not provide valid H(a), f(a) at the IC epoch")
    log_a = np.log(background_a)
    aH_start = a_start * float(np.exp(np.interp(np.log(a_start), log_a, np.log(background_H))))
    f_start = float(np.interp(np.log(a_start), log_a, background_f))
    if model == "wdm":
        # CLASS's gr.fac. f follows its pressureless cb source, which excludes
        # this thermal relic in the transfer-only model. The particles below
        # are all treated as cold, so use the matching smooth total-matter rate.
        f_start = smooth_particle_growth_rate(cfg, expected_redshift, background_path)
    if not (math.isfinite(aH_start) and aH_start > 0 and math.isfinite(f_start) and f_start > 0):
        raise ValueError("CLASS/AxiCLASS returned invalid aH or f at the IC epoch")

    verify_numeric_table(validation_table_path)
    source = np.loadtxt(validation_table_path, comments="#", ndmin=2)
    if source.ndim != 2 or source.shape[1] < 8 or len(source) < 8:
        raise ValueError("Species validation table must contain at least eight rows and eight columns")
    dark_delta_column, dark_theta_column = (4, 7) if model == "wdm" else (1, 5)
    rows = source[:, [0, dark_delta_column, 2, dark_theta_column, 6]].astype(np.float64, copy=True)
    if (np.any(~np.isfinite(rows)) or np.any(rows[:, 0] <= 0)
            or np.any(np.diff(rows[:, 0]) <= 0)):
        raise ValueError("monofonIC transfer inputs must have finite values and strictly increasing positive k")

    h = float(cfg["cosmology"]["H0"]) / 100.0
    box = float(cfg["simulation"]["box_mpc_h"])
    n = int(cfg["simulation"]["particles_per_dimension"])
    k_required = math.sqrt(3.0) * math.pi * n / box
    # The monofonIC spline deliberately excludes both endpoint samples. Check
    # coverage after that exclusion, or edge modes would be silently zeroed.
    if rows[1, 0] > 2.0 * math.pi / box or rows[-2, 0] < k_required:
        raise ValueError(
            "CLASS species transfers cover k=[%.6g, %.6g] h/Mpc; monofonIC's particle mesh needs [<= %.6g, >= %.6g] h/Mpc"
            % (rows[1, 0], rows[-2, 0], 2.0 * math.pi / box, k_required)
        )
    if not math.isfinite(h) or h <= 0:
        raise ValueError("The monofonIC transfer staging needs a finite positive Hubble parameter")

    destination = Path(destination).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    from core.data import epoch_contract
    table_contract=publish_numeric_table(destination,rows,
        {'k':{'units':'h/Mpc','quantity':'Comoving Fourier wavenumber'},
         'delta_dark':{'units':'dimensionless','quantity':'N-body-gauge dark-species density transfer'},
         'delta_baryon':{'units':'dimensionless','quantity':'N-body-gauge baryon density transfer'},
         'theta_dark':{'units':'Mpc^-1','quantity':'N-body-gauge dark-species velocity-divergence transfer'},
         'theta_baryon':{'units':'Mpc^-1','quantity':'N-body-gauge baryon velocity-divergence transfer'}},
        algorithm=__file__,source_hashes=source_hashes,
        context={'epoch':epoch_contract(a_start),'model':model,'gauge':'N-body',
            'normalization':'CLASS per unit primordial curvature; monofonIC applies A_s and primordial tilt',
            'growth_rate_start':f_start,'aH_start_Mpc^-1':aH_start},
        header=('DarkMatterSim monofonIC transfer table v1\n'
            'Columns: k[h/Mpc] delta_dark delta_baryon theta_dark[1/Mpc] theta_baryon[1/Mpc]\n'
            'Model: %s; redshift: %.12g; gauge: N-body; source_table_sha256: %s\n'
            'growth_f_start: %.17e\n'
            'aH_start_Mpc^-1: %.17e' % (model,float(expected_redshift),source_hashes[str(validation_table_path)],f_start,aH_start)))
    metadata = {
        "status": "READY",
        "schema": "darkmattersim-monofonic-species-transfers-v1",
        "model": model,
        "redshift": float(expected_redshift),
        "transfer_function_gauge": "N-body",
        "columns": [
            "k_h_per_Mpc", "delta_dark_particle_species", "delta_baryon",
            "theta_dark_per_Mpc", "theta_baryon_per_Mpc",
        ],
        "dark_species_source": "CLASS d_ncdm[0], t_ncdm[0]" if model == "wdm" else "CLASS d_cdm, t_cdm",
        "source_validation_table": str(validation_table_path),
        "source_validation_table_sha256": sha256(validation_table_path),
        "source_validation_metadata_sha256": sha256(metadata_path),
        "source_background": str(background_path),
        "source_background_sha256": sha256(background_path),
        "growth_rate_start": f_start,
        "growth_rate_source": (
            "radiation-aware smooth total cold-particle growth ODE" if model == "wdm"
            else "CLASS/AxiCLASS background growth_f"
        ),
        "conformal_hubble_rate_start_per_Mpc": aH_start,
        "table": str(destination),
        "table_sha256": sha256(destination),
        "rows": int(len(rows)),
        "k_range_h_per_Mpc": [float(rows[1, 0]), float(rows[-2, 0])],
        "required_particle_mesh_kmax_h_per_Mpc": k_required,
        "normalization": "CLASS transfer functions per unit primordial curvature; monofonIC applies A_s and primordial tilt",
        "theta_convention": "CLASS N-body-gauge velocity-divergence transfer, in Mpc^-1",
        "particle_model": "WDM transfer-function-only cold particles" if model == "wdm" else model,
    }
    metadata['table_contract']=table_contract
    atomic_json(str(destination)+'.json',metadata)
    return metadata




@directory_owner('destination')
def write_monofonic_background_table(background_path, destination, cfg, model="lcdm"):
    """Write the full CLASS/AxiCLASS smooth background and linear growth for LPT.

    The table extends to the radiation-era regular seed epoch used by the
    monofonIC growth integrator, not only to the requested simulation start.
    This lets its higher-order growth ODE use the selected H(a). For WDM the
    transfer cutoff comes from CLASS, while particle growth after the IC epoch
    treats those particles as cold and uses the matching smooth-matter ODE.
    """
    if model not in {"lcdm", "wdm", "ede"}:
        raise ValueError("monofonIC particle backgrounds support LCDM, WDM, or EDE smooth backgrounds")
    source_before=sha256(background_path)
    columns, data = class_background_table(background_path)
    required = ("H", "growth_D", "growth_f")
    missing = [name for name in required if name not in columns]
    if missing:
        raise ValueError("CLASS/AxiCLASS background lacks monofonIC fields: " + ", ".join(missing))
    z = np.asarray(data[:, columns["z"]], dtype=np.float64)
    a = 1.0 / (1.0 + z)
    order = np.argsort(a)
    a = a[order]
    h = np.asarray(data[order, columns["H"]], dtype=np.float64)
    growth = np.asarray(data[order, columns["growth_D"]], dtype=np.float64)
    rate = np.asarray(data[order, columns["growth_f"]], dtype=np.float64)
    if (len(a) < 6 or np.any(~np.isfinite(a)) or np.any(a <= 0)
            or np.any(np.diff(a) <= 0) or np.any(~np.isfinite(h)) or np.any(h <= 0)
            or np.any(~np.isfinite(growth)) or np.any(growth <= 0)
            or np.any(~np.isfinite(rate)) or np.any(rate <= 0)):
        raise ValueError("CLASS/AxiCLASS monofonIC background samples must be finite and physically ordered")
    omega_r = omega_radiation_from_background(background_path)
    h_reduced = float(cfg["cosmology"]["H0"]) / 100.0
    omega_m = (
        float(cfg["cosmology"]["omega_b"]) + float(cfg["cosmology"]["omega_cdm"])
    ) / h_reduced**2
    if not (math.isfinite(omega_r) and omega_r > 0 and math.isfinite(omega_m) and omega_m > 0):
        raise ValueError("A positive radiation and matter density is required for the monofonIC early-time seed")
    a_seed = min(1.0e-6, 1.0e-5 * omega_r / omega_m)
    if a[0] > a_seed:
        raise ValueError(
            "CLASS/AxiCLASS background starts at a=%.6g, later than the monofonIC radiation-era seed a=%.6g"
            % (a[0], a_seed)
        )
    at_today = int(np.argmin(np.abs(a - 1.0)))
    if abs(float(a[at_today]) - 1.0) > 1.0e-8:
        raise ValueError("CLASS/AxiCLASS background must include the present epoch for monofonIC normalization")
    h0 = float(h[at_today])
    class_d0 = float(growth[at_today])
    h0_config = float(cfg["cosmology"]["H0"])
    if not math.isclose(299792.458 * h0, h0_config, rel_tol=1.0e-6, abs_tol=1.0e-8):
        raise ValueError("CLASS/AxiCLASS H0 does not match the selected monofonIC cosmology")
    if a[-1] < 1.0:
        raise ValueError("CLASS/AxiCLASS background does not reach a=1")
    growth_source = "CLASS/AxiCLASS background growth_D and growth_f"
    class_growth_rate_at_start = None
    if model == "wdm":
        class_growth_rate_at_start = float(np.interp(
            math.log(1.0 / (1.0 + choose_start_redshift(cfg))),
            np.log(a), rate,
        ))
        growth, rate = smooth_particle_growth_history(cfg, a, background_path)
        d0 = 1.0
        growth_source = "radiation-aware smooth total cold-particle growth ODE; CLASS supplies the WDM transfer cutoff"
    else:
        d0 = class_d0
    keep = (a > a_seed) & (a < 1.0)
    table = np.column_stack((a[keep], h[keep] / h0, growth[keep] / d0, rate[keep]))
    # CLASS's logarithmic output grid usually omits both requested endpoints.
    # Add exact seed and present rows by log-a interpolation; never extrapolate.
    loga = np.log(a)
    seed_row = np.asarray([
        a_seed,
        np.exp(np.interp(math.log(a_seed), loga, np.log(h / h0))),
        np.exp(np.interp(math.log(a_seed), loga, np.log(growth / d0))),
        np.interp(math.log(a_seed), loga, rate),
    ])
    today_row = np.asarray([
        1.0,
        1.0,
        1.0,
        float(np.interp(0.0, np.log(a), rate)),
    ])
    table = np.vstack((seed_row, table, today_row))
    if len(table) < 6 or np.any(np.diff(table[:, 0]) <= 0) or table[-1, 0] != 1.0:
        raise ValueError("The staged monofonIC background must be strictly ordered from the exact seed through a=1")

    # Estimate the linear-in-log(a) interpolation error at interval midpoints
    # against a monotone-cubic reconstruction of the dense CLASS/AxiCLASS
    # samples. Refuse a background grid that is too coarse across a rapid expansion-history feature.
    from scipy.interpolate import PchipInterpolator

    log_table_a = np.log(table[:, 0])
    midpoint_log_a = 0.5 * (log_table_a[:-1] + log_table_a[1:])
    log_source_a = np.log(a)
    source_e = h / h0
    source_d = growth / d0
    e_reference = np.exp(PchipInterpolator(log_source_a, np.log(source_e))(midpoint_log_a))
    d_reference = np.exp(PchipInterpolator(log_source_a, np.log(source_d))(midpoint_log_a))
    f_reference = PchipInterpolator(log_source_a, rate)(midpoint_log_a)
    e_mid = np.exp(0.5 * (np.log(table[:-1, 1]) + np.log(table[1:, 1])))
    d_mid = np.exp(0.5 * (np.log(table[:-1, 2]) + np.log(table[1:, 2])))
    f_mid = 0.5 * (table[:-1, 3] + table[1:, 3])
    interpolation_errors = {
        "max_relative_E_midpoint_error": float(np.max(np.abs(e_mid / e_reference - 1.0))),
        "max_relative_D_midpoint_error": float(np.max(np.abs(d_mid / d_reference - 1.0))),
        "max_absolute_f_midpoint_error": float(np.max(np.abs(f_mid - f_reference))),
    }
    interpolation_tolerance = float(
        cfg.get("validation", {}).get("background", {}).get("max_monofonic_interpolation_error", 1.0e-4)
    )
    if (not math.isfinite(interpolation_tolerance) or interpolation_tolerance <= 0
            or max(interpolation_errors.values()) > interpolation_tolerance):
        raise ValueError(
            "The CLASS/AxiCLASS background grid is too coarse for monofonIC interpolation: %s (tolerance %.6g)"
            % (json.dumps(interpolation_errors, sort_keys=True), interpolation_tolerance)
        )
    destination = Path(destination).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    table_contract=publish_numeric_table(destination,table,
        {'a':{'units':'dimensionless','quantity':'Canonical scale factor'},
         'E':{'units':'dimensionless','quantity':'H(a)/H0'},
         'D':{'units':'dimensionless','quantity':'D(a)/D(1)'},
         'f':{'units':'dimensionless','quantity':'dlnD/dln(a)'}},
        algorithm=__file__,source_hashes={str(background_path):source_before},
        context={'model':model,'canonical_time':'scale_factor','epoch_column':'a','growth_source':growth_source,
            'interpolation_error_estimate':interpolation_errors,'interpolation_error_tolerance':interpolation_tolerance},
        header=('DarkMatterSim monofonIC background table v1\n'
            'Model: %s; H0_km_s_Mpc: %.17g; Omega_m0: %.17g; Omega_r0: %.17g\n'
            'Columns: a E(a)=H(a)/H0 D(a)/D(1) f(a)=dlnD/dln(a); source CLASS/AxiCLASS'
            % (model,299792.458*h0,omega_m,omega_r)))
    metadata = {
        "status": "PASS",
        "schema": "darkmattersim-monofonic-background-v1",
        "model": model,
        "source_solver": "CLASS/AxiCLASS",
        "source_background": str(Path(background_path).resolve()),
        "source_background_sha256": sha256(background_path),
        "table": str(destination),
        "table_sha256": sha256(destination),
        "rows": int(len(table)),
        "scale_factor_range": [float(table[0, 0]), float(table[-1, 0])],
        "radiation_era_seed_scale_factor": a_seed,
        "omega_radiation0": omega_r,
        "omega_matter0": omega_m,
        "H0_1_per_Mpc": h0,
        "growth_normalization_D_at_a1": d0,
        "growth_source": growth_source,
        "class_reported_growth_f_at_start": class_growth_rate_at_start,
        "columns": ["a", "H_over_H0", "D_over_D_today", "dlnD_dln_a"],
        "interpolation": "monofonIC interpolates E and D log-linearly in log(a), and f linearly in log(a)",
        "interpolation_error_estimate": interpolation_errors,
        "interpolation_error_estimate_method": "log-a midpoint comparison against monotone-cubic interpolation of dense CLASS/AxiCLASS output",
        "interpolation_error_tolerance": interpolation_tolerance,
        "purpose": "Model-matched first-order growth and background for mixed-species LPT initial conditions",
        "limitations": "Growth sourced by smooth background expansion; scalar-field perturbations are not evolved by the Newtonian particle generator",
    }
    metadata['table_contract']=table_contract
    atomic_json(str(destination)+'.json',metadata)
    return metadata


def pk_output_redshift(path):
    """Read the redshift written into a CLASS/AxiCLASS P(k) header."""
    return _class_table_redshift(path)
























@native_run_owner
def run_external_pk(cfg, run_dir, model, start_z, dry_run=False):
    """Load an external linear spectrum without changing its redshift amplitude.

    Supported inputs are CLASS-style linear P(k) tables and two-column
    N-GenIC-style log10(k), log10(Delta^2) tables.  The redshift is mandatory;
    the requested IC start redshift must match it exactly so we never silently
    extrapolate a spectrum with an assumed growth law.
    """
    spectrum = cfg.get("spectrum", {})
    source = Path(str(spectrum.get("file", ""))).expanduser()
    if not str(spectrum.get("file", "")).strip():
        raise ValueError("External P(k) was selected but spectrum.file is empty")
    if not source.is_absolute():
        source = (Path.cwd() / source).resolve()
    if not source.is_file():
        raise FileNotFoundError("External P(k) file does not exist: " + str(source))

    reference_z = spectrum.get("redshift")
    if reference_z is None:
        raise ValueError("External P(k) needs spectrum.redshift; its amplitude will not be rescaled")
    reference_z = float(reference_z)
    if not math.isfinite(reference_z) or reference_z < 0:
        raise ValueError("External P(k) reference redshift must be finite and non-negative")
    if not math.isclose(float(start_z), reference_z, rel_tol=0.0, abs_tol=1.0e-8 * (1.0 + reference_z)):
        raise ValueError(
            "External P(k) is at z=%.12g but the requested IC start is z=%.12g. "
            "This workflow does not rescale or extrapolate imported spectra; "
            "provide P(k) at the desired start redshift." % (reference_z, float(start_z))
        )

    source_before=sha256(source)
    table = _load_numeric_table(source)
    if table.ndim != 2 or table.shape[1] != 2:
        raise ValueError("External P(k) must contain exactly two numeric columns; choose/prepare the intended linear column first")
    values = table
    input_format = str(spectrum.get("format", "class_pk")).strip().lower()
    if input_format == "class_pk":
        k = values[:, 0].copy()
        p = values[:, 1].copy()
        k_unit = str(spectrum.get("k_unit", "h/Mpc"))
        p_unit = str(spectrum.get("p_unit", "(Mpc/h)^3"))
        h = float(cfg["cosmology"]["H0"]) / 100.0
        if k_unit == "1/Mpc":
            k /= h
        elif k_unit != "h/Mpc":
            raise ValueError("spectrum.k_unit must be 'h/Mpc' or '1/Mpc'")
        if p_unit == "Mpc^3":
            p *= h**3
        elif p_unit != "(Mpc/h)^3":
            raise ValueError("spectrum.p_unit must be '(Mpc/h)^3' or 'Mpc^3'")
        conversion = "CLASS P(k) converted to k[h/Mpc], P[(Mpc/h)^3]"
    elif input_format == "ngenic_logdelta2":
        logk, log_delta2 = values[:, 0], values[:, 1]
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            k = np.power(10.0, logk)
            delta2 = np.power(10.0, log_delta2)
            p = delta2 * (2.0 * math.pi**2) / k**3
        k_unit = "h/Mpc"
        p_unit = "(Mpc/h)^3"
        conversion = "N-GenIC log10(k[h/Mpc]), log10(Delta^2) converted to CLASS P(k) convention"
    else:
        raise ValueError("spectrum.format must be 'class_pk' or 'ngenic_logdelta2'")

    if not np.all(np.isfinite(k)) or not np.all(np.isfinite(p)):
        raise ValueError("External P(k) contains NaN, infinity, or values outside the numeric range")
    if np.any(k <= 0) or np.any(p <= 0):
        raise ValueError("External P(k) needs positive k and P(k) values")
    if np.any(np.diff(k) <= 0):
        raise ValueError("External P(k) wavenumbers must be strictly increasing")

    run_dir = Path(run_dir).expanduser().resolve()
    input_dir = run_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    kmin, kmax = k_limits(float(cfg["simulation"]["box_mpc_h"]), int(cfg["simulation"]["particles_per_dimension"]))
    pk_path = input_dir / (model + "_external_pk.dat")
    if k[0]>kmin or k[-1]<kmax:raise ValueError('External P(k) does not cover required box/Nyquist domain')
    from core.data import epoch_contract
    table_contract=publish_numeric_table(pk_path,np.column_stack([k,p]),
        {'k':{'units':'h/Mpc','quantity':'Comoving Fourier wavenumber'},
         'P':{'units':'(Mpc/h)^3','quantity':'Linear matter power spectrum at declared reference redshift'}},
        algorithm=__file__,source_hashes={str(source):source_before},fmt='%.12e',
        context={'epoch':epoch_contract(1/(1+reference_z)),'conversion':conversion,'input_format':input_format,
            'input_k_unit':k_unit,'input_p_unit':p_unit,'amplitude_rescaled':False},
        header='k_h_per_Mpc P_Mpc_over_h_cubed')
    validate_pk(pk_path, kmin, kmax)

    source_copy = input_dir / ("external_pk_source" + (source.suffix or ".dat"))
    if source.resolve() != source_copy.resolve():
        atomic_copy(source, source_copy)
    meta = {
        "model": model,
        "engine": "external_pk",
        "start_redshift": reference_z,
        "k_required_min_h_Mpc": kmin,
        "k_required_max_h_Mpc": kmax,
        "pk_file": str(pk_path),
        "pk_sha256": sha256(pk_path),
        "input_file": str(source),
        "input_file_copy": str(source_copy),
        "input_file_sha256": sha256(source),
        "input_format": input_format,
        "input_k_unit": k_unit,
        "input_p_unit": p_unit,
        "conversion": conversion,
        "power_spectrum_rescaled": False,
        "transfer_file": None,
        "background_file": None,
        "omega_radiation0": omega_radiation_from_config(cfg),
    }
    if sha256(source)!=source_before:raise ValueError('External spectrum source changed during staging')
    meta['table_contract']=table_contract
    atomic_json(input_dir/(model+'_cosmology.json'),meta)
    return meta


@native_run_owner
def run_linear_spectrum(cfg, run_dir, model, start_z, dry_run=False):
    """Use CLASS/AxiCLASS by default, or a strictly validated user spectrum."""
    source = str(cfg.get("spectrum", {}).get("source", "boltzmann")).strip().lower()
    if source == "modular":
        return run_modular_spectrum(cfg, run_dir, model, start_z, dry_run)
    if source == "file":
        return run_external_pk(cfg, run_dir, model, start_z, dry_run)
    if source != "boltzmann":
        raise ValueError("spectrum.source must be 'boltzmann', 'modular', or 'file'")
    return run_boltzmann(cfg, run_dir, model, start_z, dry_run)


@native_run_owner
def run_modular_spectrum(cfg, run_dir, model, start_z, dry_run=False):
    """Reuse a checksummed Component 1 solver run with its native transfers.

    Copying all three native tables preserves the exact linear epoch and gives
    the supported particle validation paths the same files as a direct solver run.
    """
    import shutil

    if model not in {"lcdm", "wdm", "ede"}:
        raise ValueError("Modular particle inputs support LCDM, WDM, or EDE linear inputs")
    manifest = Path(str(cfg.get("spectrum", {}).get("file", ""))).expanduser().resolve()
    if not manifest.is_file():
        raise FileNotFoundError("Modular cosmology spectrum.json is missing: " + str(manifest))
    from core.linear import verify_spectrum_directory
    source_manifest_hash = sha256(manifest)
    record = verify_spectrum_directory(manifest.parent)
    params = record.get("parameters", {})
    if record.get("status") != "complete" or record.get("model") != model:
        raise ValueError("Modular cosmology is incomplete or has a different model")
    if not math.isclose(float(params["z_start"]), float(start_z), rel_tol=0, abs_tol=1e-8):
        raise ValueError("Modular cosmology redshift differs from the requested IC epoch")
    h = float(cfg["cosmology"]["H0"]) / 100.0
    matched = {
        "H0": float(cfg["cosmology"]["H0"]),
        "Omega_b": float(cfg["cosmology"]["omega_b"]) / h**2,
        "Omega_cdm": float(cfg["cosmology"]["omega_cdm"]) / h**2,
        "A_s": float(cfg["cosmology"]["A_s"]),
        "n_s": float(cfg["cosmology"]["n_s"]),
    }
    if model == "ede":
        ede_cfg = cfg.get("ede", {})
        matched.update({
            "f_ede": float(ede_cfg["f_ede"]),
            "log10_zc": math.log10(float(ede_cfg["z_c"])),
            "theta_i": float(ede_cfg["theta_i"]),
            "n_axion": float(ede_cfg["n_axion"]),
        })
    for key, expected in matched.items():
        if not math.isclose(float(params[key]), expected, rel_tol=1e-7, abs_tol=1e-12):
            raise ValueError("Modular cosmology parameter mismatch: " + key)
    source_files = record["source_files"]["start"]
    checked = {}
    for kind in ("pk", "transfer", "background", "ini", "log"):
        path = (manifest.parent / source_files[kind]).resolve()
        if not path.is_file() or sha256(path) != record["checksums_sha256"][source_files[kind]]:
            raise ValueError("Modular cosmology checksum failed: " + kind)
        checked[kind] = path
    kmin, kmax = k_limits(float(cfg["simulation"]["box_mpc_h"]), int(cfg["simulation"]["particles_per_dimension"]))
    validate_pk(checked["pk"], kmin, kmax)
    if dry_run:
        return {"model": model, "engine": "modular", "start_redshift": float(start_z),
                "pk_file": str(checked["pk"]), "kmin": kmin, "kmax": kmax}
    input_dir = Path(run_dir).expanduser().resolve() / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    generation_dir = input_dir / 'modular_runs' / (model + '_' + uuid.uuid4().hex)
    generation_dir.mkdir(parents=True)
    staged = {}
    for kind, path in checked.items():
        dest = generation_dir / ("modular_" + kind + path.suffix)
        atomic_copy(path, dest)
        staged[kind] = dest
    meta = {
        "model": model, "engine": "axiclass" if model == "ede" else "class",
        "start_redshift": float(start_z), "output_redshift": pk_output_redshift(staged["pk"]),
        "k_required_min_h_Mpc": kmin, "k_required_max_h_Mpc": kmax,
        "pk_file": str(staged["pk"]), "pk_sha256": sha256(staged["pk"]),
        "transfer_file": str(staged["transfer"]), "background_file": str(staged["background"]),
        "background_sha256": sha256(staged["background"]),
        "solver_ini_file": str(staged["ini"]), "solver_log_file": str(staged["log"]),
        "omega_radiation0": omega_radiation_from_background(staged["background"]),
        "modular_source_manifest": str(manifest), "modular_source_manifest_sha256": sha256(manifest),
    }
    if model == "ede":
        table_path = input_dir / "ede_gadget_background.txt"
        meta["gadget_background_validation"] = write_gadget_background_table(
            staged["background"], table_path,
            min(1.0 / (1.0 + float(start_z)), 1.0e-3),
            expected_h0_km_s_mpc=cfg['cosmology']['H0'],
        )
        meta["gadget_background_file"] = str(table_path.resolve())
        meta["gadget_background_sha256"] = sha256(table_path)
        meta["particle_model_scope"] = (
            "AxiCLASS linear transfer functions and smooth H(a), D(a), f(a) table; "
            "GADGET evolves matter particles and does not evolve nonlinear EDE scalar-field modes"
        )
    meta["cosmology_products_file"] = write_class_cosmology_products(
        model, start_z, staged["background"], staged["transfer"], staged["pk"],
        input_dir / (model + "_cosmology_products.h5"))
    if sha256(manifest) != source_manifest_hash or verify_spectrum_directory(manifest.parent) != record:
        raise ValueError('Modular cosmology source changed during copying')
    meta['science_acceptance'] = 'NOT_ACCEPTED'
    atomic_json(input_dir / (model + '_cosmology.json'), meta)
    return meta


def wdm_resolution_report(wdm_pk_path, lcdm_pk_path, cfg):
    """Measure WDM half-mode and the empirical Wang-White particle limit.

    The report applies to both supported WDM models; particle velocities and
    their exact representation are recorded separately.
    """
    validate_wdm_representation(cfg)
    wdm = np.loadtxt(wdm_pk_path, comments="#", ndmin=2)
    lcdm = np.loadtxt(lcdm_pk_path, comments="#", ndmin=2)
    for name, table in (("WDM", wdm), ("LCDM control", lcdm)):
        if table.ndim != 2 or table.shape[1] != 2 or len(table) < 3:
            raise ValueError(name + " P(k) needs at least three k,P rows")
        if np.any(~np.isfinite(table)) or np.any(table[:, 0] <= 0) or np.any(table[:, 1] <= 0) or np.any(np.diff(table[:, 0]) <= 0):
            raise ValueError(name + " P(k) must be finite, positive, and strictly ordered")
    kmin = max(float(wdm[0, 0]), float(lcdm[0, 0]))
    kmax = min(float(wdm[-1, 0]), float(lcdm[-1, 0]))
    if not 0 < kmin < kmax:
        raise ValueError("WDM and LCDM P(k) tables have no common positive k range")
    k = np.geomspace(kmin, kmax, max(2048, min(20000, 4 * len(wdm))))
    p_wdm = np.exp(np.interp(np.log(k), np.log(wdm[:, 0]), np.log(wdm[:, 1])))
    p_lcdm = np.exp(np.interp(np.log(k), np.log(lcdm[:, 0]), np.log(lcdm[:, 1])))
    transfer = np.sqrt(p_wdm / p_lcdm)
    below = np.flatnonzero(transfer <= 0.5)
    if not len(below) or below[0] == 0:
        raise ValueError("The shared P(k) coverage does not bracket the WDM half-mode transfer amplitude T=0.5")
    i = int(below[0])
    log_t0, log_t1 = np.log(transfer[i - 1:i + 1])
    log_k0, log_k1 = np.log(k[i - 1:i + 1])
    fraction = (math.log(0.5) - float(log_t0)) / float(log_t1 - log_t0)
    log_khm = float(log_k0 + fraction * (log_k1 - log_k0))
    k_hm = math.exp(log_khm)

    c = cfg["cosmology"]
    h = float(c["H0"]) / 100.0
    omega_wdm = float(c["omega_cdm"]) / h**2
    omega_particles = omega_wdm
    if cfg["simulation"].get("physics", "dm") != "hydro":
        # In DM-only runs each particle carries the baryons plus CDM mass.
        omega_particles += float(c["omega_b"]) / h**2
    rho_crit = RHO_CRIT_MSUN_H_MPC_H3
    rho_wdm = omega_wdm * rho_crit
    rho_particles = omega_particles * rho_crit
    box = float(cfg["simulation"]["box_mpc_h"])
    n = int(cfg["simulation"]["particles_per_dimension"])
    spacing = box / n
    particle_mass = rho_particles * spacing**3
    half_mode_mass = (4.0 * math.pi / 3.0) * rho_wdm * (math.pi / k_hm)**3

    # Wang & White (2007), equation 7: empirical estimate of the spurious
    # filament-fragment mass. It is a resolution diagnostic, not a universal
    # halo-mass cut for every transfer-function shape.
    delta2_wdm = wdm[:, 0]**3 * wdm[:, 1] / (2.0 * math.pi**2)
    peak_idx = int(np.argmax(delta2_wdm))
    k_peak = float(wdm[peak_idx, 0])
    spurious_mass = 10.1 * rho_particles * spacing / k_peak**2
    particles_per_half_mode = half_mode_mass / particle_mass
    target_particles = int(cfg.get("wdm", {}).get("minimum_particles_per_half_mode", 100))
    n_required = int(math.ceil(n * (target_particles / particles_per_half_mode)**(1.0 / 3.0)))
    underresolved = particles_per_half_mode < target_particles or spurious_mass >= half_mode_mass
    thermal_model = cfg.get("wdm", {}).get("model") == "wdm_thermal_relic"
    return {
        "status": "UNDERRESOLVED" if underresolved else "PASS",
        "simulation_approximation": (
            "CLASS thermal-relic transfer with sampled Fermi-Dirac particle velocities"
            if thermal_model else "transfer-function-only benchmark with cold particles"
        ),
        "interpretation_limit": (
            "linear CLASS ncdm transfer and primordial thermal velocities are represented; Newtonian nonlinear evolution and the standard pressureless second-order LPT term still require observable-level convergence"
            if thermal_model else "cold-particle benchmark; primordial relic thermal velocities are omitted"
        ),
        "second_order_lpt_velocity": "standard pressureless N-GenIC approximation; scale-dependent CLASS correction is applied only to the first-order velocity",
        "particle_thermal_velocities": thermal_model,
        "full_thermal_relic_phase_space": thermal_model,
        "transfer_definition": "T(k)=sqrt(P_WDM/P_LCDM); half-mode is T=0.5 (power ratio 0.25)",
        "start_redshift": pk_output_redshift(wdm_pk_path),
        "k_half_mode_h_Mpc": k_hm,
        "half_mode_mass_msun_h": half_mode_mass,
        "k_peak_dimensionless_power_h_Mpc": k_peak,
        "wang_white_spurious_mass_msun_h": spurious_mass,
        "wang_white_reference": "Wang & White 2007, https://arxiv.org/abs/astro-ph/0702575",
        "particle_spacing_mpc_h": spacing,
        "particle_mass_msun_h": particle_mass,
        "particles_per_half_mode_mass": particles_per_half_mode,
        "reference_minimum_particles_per_half_mode": target_particles,
        "suggested_particles_per_dimension_for_reference": n_required,
        "resolution_warning": (
            "WDM cutoff halos are below the requested particle-count reference or at/below the empirical spurious-fragment mass. "
            "Do not interpret the low-mass halo mass function as converged."
            if underresolved else None
        ),
    }


def match_wdm_cutoff_coverage(cfg, run_dir, start_redshift, wdm_meta, lcdm_meta, max_expansions=8,
                              model_run_dirs=None):
    """Extend paired CLASS output until it brackets the WDM half-mode cutoff.

    The required particle mesh may have a Nyquist scale below the physical WDM
    half-mode k. Reporting the cutoff requires extending both linear spectra;
    extrapolating beyond their shared tabulated range would hide an unknown.
    """
    for expansion in range(max_expansions + 1):
        try:
            report = wdm_resolution_report(wdm_meta["pk_file"], lcdm_meta["pk_file"], cfg)
            return wdm_meta, lcdm_meta, report
        except ValueError as exc:
            if "does not bracket the WDM half-mode" not in str(exc):
                raise
            if str(cfg.get("spectrum", {}).get("source", "boltzmann")).strip().lower() != "boltzmann":
                raise ValueError(
                    "The imported WDM/control spectra do not cover the half-mode cutoff; supply both spectra "
                    "with wider k coverage. The code will not extrapolate their ratio."
                ) from exc
            if expansion == max_expansions:
                raise ValueError(
                    "CLASS output still does not bracket the WDM half-mode after %d range expansions; "
                    "check the relic parameters and requested k coverage." % max_expansions
                ) from exc
            wdm_table = np.loadtxt(wdm_meta["pk_file"], comments="#", ndmin=2)
            lcdm_table = np.loadtxt(lcdm_meta["pk_file"], comments="#", ndmin=2)
            current_kmax = min(float(wdm_table[-1, 0]), float(lcdm_table[-1, 0]))
            requested_kmax = max(current_kmax * 2.0, float(wdm_meta.get("k_required_max_h_Mpc", current_kmax)) * 2.0)
            print(
                "WDM half-mode lies beyond shared CLASS coverage k=%.6g h/Mpc; "
                "recomputing both spectra through k=%.6g h/Mpc..." % (current_kmax, requested_kmax)
            )
            model_run_dirs = model_run_dirs or {}
            wdm_meta = run_boltzmann(
                cfg, model_run_dirs.get("wdm", run_dir), "wdm", start_redshift,
                kmax_hmpc=requested_kmax,
            )
            lcdm_meta = run_boltzmann(
                cfg, model_run_dirs.get("lcdm", run_dir), "lcdm", start_redshift,
                kmax_hmpc=requested_kmax,
            )
    raise RuntimeError("Unreachable WDM spectrum coverage state")


def validate_wdm_resolution_gate(report, cfg):
    """Stop an underresolved WDM evolution unless the config explicitly labels it a benchmark."""
    if not isinstance(report, dict) or report.get("status") not in {"PASS", "UNDERRESOLVED"}:
        raise ValueError("WDM resolution gate needs a valid resolution report")
    if report["status"] == "PASS":
        return {"status": "PASS", "interpretation": "meets the configured half-mode particle-count and spurious-mass screening criteria"}
    allowed = cfg.get("wdm", {}).get("allow_underresolved_benchmark", False)
    if allowed is True:
        return {
            "status": "EXPLICITLY_ALLOWED_BENCHMARK",
            "interpretation": "underresolved cutoff-scale results are not converged science measurements",
        }
    raise ValueError(
        "WDM particle evolution is blocked: the configured run is UNDERRESOLVED "
        "(%s particles per half-mode mass; configured reference %s; estimated "
        "N=%s per dimension for that reference). Change the box size, particle "
        "count, or relic mass. To run only as an explicitly labeled underresolved "
        "benchmark, set wdm.allow_underresolved_benchmark: true." % (
            report.get("particles_per_half_mode_mass", "unknown"),
            report.get("reference_minimum_particles_per_half_mode", "unknown"),
            report.get("suggested_particles_per_dimension_for_reference", "unknown"),
        )
    )


def _load_numeric_table(path):
    verify_numeric_table(path)
    before=sha256(path)
    rows = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            try:
                rows.append([float(x) for x in parts])
            except ValueError:
                continue
    if not rows:
        raise ValueError("No numeric rows found in " + str(path))
    if sha256(path)!=before:raise ValueError('Numeric table changed during read')
    return np.asarray(rows)


def validate_pk(path, kmin, kmax):
    """Check the positive, ordered k/P grid and requested physical coverage before downstream use."""
    table = _load_numeric_table(path)
    k = table[:, 0]
    p = table[:, 1]
    if not np.all(np.isfinite(k)) or not np.all(np.isfinite(p)):
        raise ValueError("Power spectrum contains NaN/Inf")
    if np.any(k <= 0) or np.any(p <= 0):
        raise ValueError("Power spectrum must have positive k and P(k)")
    if np.any(np.diff(k) <= 0):
        raise ValueError("Power-spectrum k values are not strictly increasing")
    if k[0] > kmin:
        raise ValueError("P(k) does not extend far enough below box fundamental mode")
    if k[-1] < kmax:
        raise ValueError("P(k) does not extend beyond requested Nyquist margin")
    return True


@directory_owner('out_path')
def convert_pk_to_ngenic(class_pk, out_path, h):
    """Convert conventional CLASS P(k) into GADGET-4's N-GenIC table.

    CLASS uses the Fourier convention with Delta^2 = k^3 P(k)/(2*pi^2).
    GADGET-4's N-GenIC table is documented as 4*pi*k^3*P_gadget, where its
    internal P_gadget is smaller than the conventional P(k) by (2*pi)^3.
    Therefore the value written to N-GenIC is the conventional dimensionless
    spectrum, k^3 P_CLASS/(2*pi^2), not 4*pi*k^3 P_CLASS.

    The CLASS ``format = class`` output uses k[h/Mpc] and P[(Mpc/h)^3].
    """
    before=sha256(class_pk)
    table = _load_numeric_table(class_pk)
    k = table[:, 0]
    p = table[:, 1]
    if table.shape[1]!=2 or len(table)<2 or not np.isfinite(table).all() or np.any(table<=0) or np.any(np.diff(k)<=0):
        raise ValueError('N-GenIC conversion requires positive finite ordered two-column k/P')
    if not math.isfinite(h) or h<=0:raise ValueError('N-GenIC conversion requires positive finite h')
    delta2 = k**3 * p / (2.0 * math.pi**2)
    if np.any(~np.isfinite(delta2)) or np.any(delta2<=0):raise ValueError('Dimensionless power exceeds numerical range')
    data = np.column_stack([np.log10(k), np.log10(delta2)])
    # GADGET's N-GenIC reader uses fscanf("%lg %lg") and stops on any header
    # text, including a comment marker. Keep this two-column file numeric only;
    # the JSON sidecar below carries units and convention metadata.
    table_contract=publish_numeric_table(out_path,data,
        {'log10_k':{'units':'dimensionless','quantity':'log10(k / (h/Mpc))'},
         'log10_delta2':{'units':'dimensionless','quantity':'log10(k^3 P_CLASS / (2 pi^2))'}},
        algorithm=__file__,source_hashes={str(class_pk):before},fmt='%.12e',
        context={'h':float(h),'Fourier_convention':'Delta^2 = k^3 P_CLASS / (2 pi^2)'})
    sidecar = {
        "source": str(class_pk),
        "k_unit": "h/Mpc",
        "P_unit": "(Mpc/h)^3",
        "h": float(h),
        "class_fourier_convention": "Delta^2 = k^3 P(k) / (2 pi^2)",
        "ngenic_table_convention": "Delta^2 = 4 pi k^3 P_gadget; P_gadget = P_CLASS / (2 pi)^3",
        "conversion_factor_from_class_pk": "k^3 / (2 pi^2)",
        "ngenic_columns": ["log10(k)", "log10(Delta^2)"],
    }
    sidecar['table_contract']=table_contract
    atomic_json(str(out_path)+'.json',sidecar)
    return out_path



def lpt_rms_displacement_fraction(pk_path, cfg):
    """Return exact discrete-box 1-D rms linear displacement / particle spacing.

    This sums the CLASS spectrum over the Fourier modes represented by the
    configured periodic particle mesh. It excludes super-box and unresolved
    modes instead of integrating an infinite-volume spectrum outside the run's
    actual mode set.
    """
    table = _load_numeric_table(pk_path)
    k_table = table[:, 0]
    p_table = table[:, 1]
    L = float(cfg["simulation"]["box_mpc_h"])
    n = int(cfg["simulation"]["particles_per_dimension"])
    k_modes = 2.0 * math.pi * np.fft.fftfreq(n, d=L / n)
    k_squared_yz = k_modes[:, None]**2 + k_modes[None, :]**2
    variance = 0.0
    for kx in k_modes:
        k_squared = k_squared_yz + kx**2
        nonzero = k_squared > 0
        k_mag = np.sqrt(k_squared[nonzero])
        p_mode = np.interp(k_mag, k_table, p_table, left=0.0, right=0.0)
        variance += float(np.sum(p_mode * (kx**2) / (k_squared[nonzero]**2)))
    variance /= L**3
    sigma = math.sqrt(max(variance, 0.0))
    spacing = L / n
    return sigma / spacing


def _publish_start_epoch_selection(run_dir, report, trials):
    """Save measured displacement trials through the shared numeric audit."""
    from core.data import export_numeric_audit
    from core.data import epoch_contract
    from core.foundation import physical_constants_contract
    root = Path(run_dir).resolve()
    history = root / 'input/start_epoch_diagnostics' / uuid.uuid4().hex
    history.mkdir(parents=True)
    audit = history / 'trials.json'
    source_hashes = {row['source_pk']: row['source_pk_sha256'] for row in trials}
    arrays = {'redshift':np.asarray([row['z'] for row in trials]),
              'scale_factor':np.asarray([1. / (1. + row['z']) for row in trials]),
              'rms_displacement_fraction':np.asarray([row['rms_displacement_fraction'] for row in trials])}
    specifications = {name:{'units':'dimensionless','quantity':{
        'redshift':'Trial linear power spectrum redshift',
        'scale_factor':'Trial requested canonical scale factor',
        'rms_displacement_fraction':'One-dimensional rms LPT displacement divided by mean particle spacing'}[name]}
        for name in arrays}
    measured = export_numeric_audit(audit, report, arrays, specifications, algorithm=__file__,
        context={'epoch':epoch_contract(float(arrays['scale_factor'][-1])),
                 'adopted_constants':physical_constants_contract(),
                 'validity':'Linear displacement diagnostic only; no nonlinear start-epoch convergence claim'},
        source_hashes=source_hashes)
    pointer = root / 'input/start_redshift_validation.json'
    record = {**report,'science_acceptance':'NOT_ACCEPTED','input_pk_hashes':source_hashes,
        'numeric_audit':{'path':audit.relative_to(root).as_posix(),'sha256':sha256(audit)}}
    atomic_json(pointer, record)
    atomic_text(pointer.with_name(pointer.name + '.sha256'), sha256(pointer) + '\n')
    verify_start_epoch_selection(pointer)
    return record


def verify_start_epoch_selection(filename):
    """Check durable scalar/numeric start-epoch diagnostic before reuse."""
    from core.data import verify_numeric_audit
    filename = Path(filename).resolve()
    before = sha256(filename)
    record = json.loads(filename.read_text())
    if 'numeric_audit' not in record: return {**record,'integrity_status':'LEGACY_UNSEALED'}
    seal = filename.with_name(filename.name + '.sha256')
    if not seal.is_file() or seal.read_text().strip() != before:
        raise ValueError('Start-epoch diagnostic completion seal missing or changed')
    root = filename.parent.parent
    relative = Path(record['numeric_audit']['path'])
    audit = root / relative
    if relative.is_absolute() or '..' in relative.parts or not audit.resolve().is_relative_to(root):
        raise ValueError('Start-epoch diagnostic numeric path escapes its run')
    if sha256(audit) != record['numeric_audit']['sha256']:
        raise ValueError('Start-epoch diagnostic numeric report changed')
    measured = verify_numeric_audit(audit)
    for name, value in measured.items():
        if name not in ('numeric_data','science_acceptance') and record.get(name) != value:
            raise ValueError('Start-epoch scalar/numeric report mismatch')
    for source, expected in record['input_pk_hashes'].items():
        if sha256(source) != expected: raise ValueError('Start-epoch diagnostic input spectrum changed')
    if sha256(filename) != before: raise ValueError('Start-epoch diagnostic changed while reading')
    return {**record,'integrity_status':'INTEGRITY_VERIFIED'}


@native_run_owner
def select_start_redshift(cfg, run_dir, model, dry_run=False):
    """Choose z_start from an explicit value or a measured LPT displacement gate.

    AUTO evaluates the actual CLASS/AxiCLASS P(k,z) and iterates until the 1-D
    rms displacement is in the configured fraction of one particle spacing.
    This avoids blindly hard-coding z=49/99 while keeping the criterion visible.
    """
    value = cfg["simulation"].get("start_redshift", "auto")
    if value != "auto":
        z = float(value)
        return z, run_linear_spectrum(cfg, run_dir, model, z, dry_run)
    if str(cfg.get("spectrum", {}).get("source", "boltzmann")).strip().lower() == "file":
        z = float(cfg["spectrum"].get("redshift", -1))
        meta = run_linear_spectrum(cfg, run_dir, model, z, dry_run)
        ratio = lpt_rms_displacement_fraction(meta["pk_file"], cfg)
        lower = float(cfg["simulation"].get("auto_start_min_fraction", 0.05))
        upper = float(cfg["simulation"].get("auto_start_max_fraction", 0.15))
        if not lower <= ratio <= upper:
            raise RuntimeError(
                "Imported P(k) at its fixed redshift gives an AUTO initial-displacement fraction of %.5g "
                "(accepted %.5g–%.5g). The imported spectrum is not rescaled; provide P(k) at a suitable "
                "initial redshift or select a different source." % (ratio, lower, upper)
            )
        _publish_start_epoch_selection(run_dir, {
            'criterion':'1D rms LPT displacement / mean spacing','spectrum_redshift':z,
            'rms_displacement_fraction':ratio,'accepted_range':[lower,upper],
            'power_spectrum_rescaled':False},
            [{'z':z,'rms_displacement_fraction':ratio,'source_pk':str(meta['pk_file']),
              'source_pk_sha256':sha256(meta['pk_file'])}])
        return z, meta
    if dry_run:
        z = float(cfg["simulation"].get("start_redshift_guess", 99.0))
        return z, run_boltzmann(cfg, run_dir, model, z, True)

    s = cfg["simulation"]
    z = float(s.get("start_redshift_guess", 99.0))
    target = float(s.get("auto_start_target_rms_displacement_fraction", 0.10))
    lower = float(s.get("auto_start_min_fraction", 0.05))
    upper = float(s.get("auto_start_max_fraction", 0.15))
    zmin = float(s.get("auto_start_min_redshift", 24.0))
    zmax = float(s.get("auto_start_max_redshift", 199.0))
    history = []
    meta = None
    accepted = False
    for _ in range(5):
        z = min(zmax, max(zmin, z))
        meta = run_linear_spectrum(cfg, run_dir, model, z, False)
        ratio = lpt_rms_displacement_fraction(meta["pk_file"], cfg)
        history.append({'z':z,'rms_displacement_fraction':ratio,
                        'source_pk':str(meta['pk_file']),'source_pk_sha256':sha256(meta['pk_file'])})
        if lower <= ratio <= upper:
            accepted = True
            break
        if ratio <= 0:
            raise RuntimeError("AUTO start-redshift diagnostic returned a non-positive displacement amplitude")
        # Linear displacement amplitude is approximately proportional to growth.
        # This update is only a proposal; every proposed z is re-evaluated with
        # CLASS/AxiCLASS, so the final accepted criterion uses the actual model.
        proposal = (1.0 + z) * (ratio / target) - 1.0
        next_z = min(zmax, max(zmin, proposal))
        if math.isclose(next_z, z, rel_tol=0.0, abs_tol=1.0e-12):
            break
        z = next_z
    if not accepted:
        raise RuntimeError("AUTO start-redshift selector did not converge inside configured bounds: " + str(history))

    _publish_start_epoch_selection(run_dir,
        {'criterion':'1D rms LPT displacement / mean spacing','target':target,
         'accepted_range':[lower,upper],'history':history}, history)
    return float(z), meta

def choose_start_redshift(cfg):
    """Select the model-matched IC epoch from the configured displacement criterion and allowed redshift range."""
    value = cfg["simulation"].get("start_redshift", "auto")
    if value != "auto":
        return float(value)
    # Conservative first guess. The IC validator in simulate.py checks actual
    # displacements and requires a rerun at higher z if this is too low.
    return float(cfg["simulation"].get("start_redshift_guess", 99.0))


def doctor():
    print("CLASS:   ", executable("class") or "MISSING")
    print("AxiCLASS:", executable("axiclass") or "MISSING")


def main():
    """Parse command arguments and dispatch this module's documented workflow."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "config.yaml"))
    parser.add_argument("--run-dir", default=None, help="output folder (required interactively when omitted)")
    parser.add_argument("--model", choices=["lcdm", "wdm", "ede"], default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--doctor", action="store_true")
    args = parser.parse_args()
    if args.doctor:
        doctor()
        return
    cfg = load_config(args.config)
    model = args.model or cfg["cosmology"].get("model")
    if model not in {"lcdm", "wdm", "ede"}:
        parser.error("enter --model lcdm, wdm, or ede, or set cosmology.model in the config")
    cfg["cosmology"]["model"] = model
    missing = ["cosmology.%s" % key for key in (
        "H0", "omega_b", "omega_cdm", "A_s", "n_s", "tau_reio", "N_ur", "T_cmb")
        if cfg["cosmology"].get(key) is None]
    if model == "ede":
        missing += ["ede.%s" % key for key in ("f_ede", "z_c", "theta_i", "n_axion")
                    if cfg.get("ede", {}).get(key) is None]
    if model == "wdm" and cfg.get("wdm", {}).get("thermal_relic_mass_keV") is None:
        missing.append("wdm.thermal_relic_mass_keV")
    missing += ["simulation.%s" % key for key in (
        "box_mpc_h", "particles_per_dimension", "random_seed", "start_redshift",
        "end_redshift", "snapshot_z_max", "snapshot_z_min", "snapshot_dz")
        if cfg.get("simulation", {}).get(key) is None]
    if cfg.get("simulation", {}).get("start_redshift") == "auto" and cfg.get("simulation", {}).get("start_redshift_guess") is None:
        missing.append("simulation.start_redshift_guess")
    if missing:
        parser.error("enter required cosmology values: " + ", ".join(missing))
    from core.foundation import choose_run_directory
    run_dir = choose_run_directory(args.run_dir)
    z, meta = select_start_redshift(cfg, run_dir, model, args.dry_run)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
from dataclasses import dataclass
from scipy.integrate import cumulative_trapezoid, simpson
from scipy.interpolate import PchipInterpolator

RHO_CRIT_0 = RHO_CRIT_MSUN_H_MPC_H3
DELTA_C = 1.686


@dataclass(frozen=True)
class HMFModelInfo:
    """Reference equation and its mass convention; calibration is not universal."""
    name: str
    mass_definition: str
    citation: str


MODELS = {
    1: HMFModelInfo("Sheth-Tormen reference", "ellipsoidal-collapse reference", "https://arxiv.org/abs/astro-ph/9901122"),
    2: HMFModelInfo("Tinker 2008 SO 200m", "M_200m", "https://arxiv.org/abs/0803.2706"),
    5: HMFModelInfo("Press-Schechter reference", "spherical-collapse reference", "https://doi.org/10.1086/152650"),
}


def top_hat_window_fourier(x):
    """W(kR)=3[sin(kR)-kR cos(kR)]/(kR)^3; W(0)=1 without cancellation."""
    x = np.asarray(x, dtype=float)
    small = np.abs(x) < 1.e-3
    safe = np.where(small, 1., x)
    exact = 3. * (np.sin(safe) - safe * np.cos(safe)) / safe**3
    series = 1. - x**2/10. + x**4/280. - x**6/15120.
    return np.where(small, series, exact)


def lagrangian_radius_from_mass(m_msun_h, omega_m):
    """R=[3M/(4 pi rho_m,0)]^(1/3) in Mpc/h, with positive inputs."""
    mass = np.asarray(m_msun_h, dtype=float)
    if not np.isfinite(omega_m) or omega_m <= 0 or np.any(~np.isfinite(mass)) or np.any(mass <= 0):
        raise ValueError("Mass [Msun/h] and Omega_m must be finite and positive")
    return np.cbrt(3.*mass/(4.*math.pi*omega_m*RHO_CRIT_0))


def compute_sigma_and_dlnsigma(k_h_mpc, p_k_linear, masses_msun_h, omega_m):
    """Integrate sigma and -dln(sigma)/dln(M) over the supplied linear P(k).

    sigma^2 = integral Delta^2(k) W(kR)^2 dln(k). The derivative is evaluated
    under the same integral, avoiding a derivative of a sparse mass spline.
    dW/dln(R)=3 sin(kR)/(kR)-3W and dln(R)/dln(M)=1/3.
    Endpoint coverage is required for every smoothing radius. A shape-preserving
    log interpolation samples only inside the supplied k interval.
    """
    k, power, mass = (np.asarray(v, dtype=float) for v in (k_h_mpc, p_k_linear, masses_msun_h))
    if k.ndim != 1 or power.shape != k.shape or len(k) < 8 or mass.ndim != 1 or not len(mass):
        raise ValueError("P(k) needs at least eight aligned samples and a nonempty mass vector")
    if (np.any(~np.isfinite(k)) or np.any(~np.isfinite(power)) or np.any(k <= 0)
            or np.any(power <= 0) or np.any(np.diff(k) <= 0)):
        raise ValueError("k and P must be finite, positive; k must strictly increase")
    radii = lagrangian_radius_from_mass(mass, omega_m)
    if np.any(k[0]*radii > .1) or np.any(k[-1]*radii < 10.):
        raise ValueError("Linear spectrum needs broader coverage: k_min R <= 0.1 and k_max R >= 10")
    log_k = np.linspace(np.log(k[0]), np.log(k[-1]), max(8192, len(k)))
    dense_k = np.exp(log_k)
    dense_power = np.exp(PchipInterpolator(np.log(k), np.log(power), extrapolate=False)(log_k))
    delta2 = dense_k**3 * dense_power / (2.*math.pi**2)
    sigma, slope = np.empty(len(mass)), np.empty(len(mass))
    for i, radius in enumerate(radii):
        x = dense_k*radius
        window = top_hat_window_fourier(x)
        small = np.abs(x) < 1.e-3
        derivative = np.where(small, -x**2/5. + x**4/70. - x**6/2520., 3.*np.sin(x)/x-3.*window)
        variance = simpson(delta2*window**2, x=log_k)
        if not np.isfinite(variance) or variance <= 0:
            raise ValueError("Top-hat variance must be finite and positive")
        sigma[i] = math.sqrt(variance)
        slope[i] = -simpson(delta2*window*derivative, x=log_k)/(3.*variance)
    if np.any(~np.isfinite(slope)) or np.any(slope < 0):
        raise ValueError("Variance derivative is unresolved or increases with mass; broaden/refine the spectrum")
    return sigma, slope


def f_sheth_tormen(sigma, z=0., *, delta_c=DELTA_C, A=.3222, a=.707, p=.3):
    """Dimensionless f(sigma); equation 10 of Sheth & Tormen (1999)."""
    if not all(math.isfinite(v) and v > 0 for v in (delta_c, A, a)) or not math.isfinite(p) or not 0 <= p < .5:
        raise ValueError("ST coefficients require positive delta_c/A/a and 0 <= p < 0.5")
    sigma = np.asarray(sigma, dtype=float)
    if np.any(~np.isfinite(sigma)) or np.any(sigma <= 0):
        raise ValueError("sigma must be finite and positive")
    nu = math.sqrt(a)*delta_c/sigma
    return A*math.sqrt(2./math.pi)*nu*(1.+nu**(-2.*p))*np.exp(-.5*nu**2)


def f_tinker_2008(sigma, z=0., delta_so=200.):
    """Tinker eqs. 3-8 and Table 2 for SO 200m, without mass-definition switching."""
    if delta_so != 200. or not np.isfinite(z) or not 0 <= z <= 2.5:
        raise ValueError("This calibration requires SO 200m and 0 <= z <= 2.5")
    sigma = np.asarray(sigma, dtype=float)
    if np.any(~np.isfinite(sigma)) or np.any(sigma <= 0):
        raise ValueError("sigma must be finite and positive")
    alpha = 10.**(-(0.75/math.log10(200./75.))**1.2)
    amplitude = .186*(1.+z)**(-.14)
    exponent = 1.47*(1.+z)**(-.06)
    scale = 2.57*(1.+z)**(-alpha)
    return amplitude*((sigma/scale)**(-exponent)+1.)*np.exp(-1.19/sigma**2)


def evaluate_theoretical_hmf(model_id, masses_msun_h, k_h_mpc, p_k_linear, omega_m, redshift):
    """Return differential abundance and a cumulative integral with a finite upper mass.

    dn/dln(M) = rho_m,0/M * f(sigma) * [-dln(sigma)/dln(M)]. Multiply by
    ln(10) for dn/dlog10(M). The cumulative is integral_M^Mmax, where Mmax
    is explicitly returned; it is never described as an integral to infinity.
    """
    if model_id not in MODELS or not np.isfinite(redshift) or redshift < 0:
        raise ValueError("Select reference 1, 2 or 5 and a finite nonnegative redshift")
    masses = np.asarray(masses_msun_h, dtype=float)
    if masses.ndim != 1 or len(masses) < 2 or np.any(np.diff(masses) <= 0):
        raise ValueError("Reference masses must be a strictly increasing vector")
    def multiplicity(sig):
        if model_id == 1:
            return f_sheth_tormen(sig)
        if model_id == 2:
            return f_tinker_2008(sig, redshift)
        nu = DELTA_C/sig
        return math.sqrt(2./math.pi)*nu*np.exp(-.5*nu**2)
    # Integrate on exactly the declared mass interval; consumers decide a
    # scientifically suitable upper bound rather than inventing an unseen tail.
    dense_mass = np.geomspace(masses[0], masses[-1], max(512, len(masses)))
    sigma, derivative = compute_sigma_and_dlnsigma(k_h_mpc, p_k_linear, dense_mass, omega_m)
    f_val = multiplicity(sigma)
    rate = omega_m*RHO_CRIT_0/dense_mass*f_val*derivative
    reverse_integral = -cumulative_trapezoid(rate[::-1], x=np.log(dense_mass[::-1]), initial=0.)
    log_mass = np.log(masses)
    diff = np.interp(log_mass, np.log(dense_mass), rate)
    return {
        "masses_msun_h": masses,
        "sigma": np.interp(log_mass, np.log(dense_mass), sigma),
        "dlnsigma_dlnm": np.interp(log_mass, np.log(dense_mass), derivative),
        "f_sigma": np.interp(log_mass, np.log(dense_mass), f_val),
        "dn_dlnM_mpc_h3": diff, "dn_dlog10M_mpc_h3": math.log(10.)*diff,
        "cumulative_n_mpc_h3": np.interp(log_mass, np.log(dense_mass), reverse_integral[::-1]),
        "cumulative_upper_mass_msun_h": float(dense_mass[-1]), "model_info": MODELS[model_id],
    }
