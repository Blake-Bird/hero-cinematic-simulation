"""CPU, memory, storage estimates and external runtime locations.

Resource estimates describe planning limits, not physical resolution or measured
peak usage. Scientific dependencies are imported only by the census/estimators
that need them; basic resource discovery also works in portable Blender Python.
"""
from __future__ import annotations
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
import math
import json
import os
import shutil
import sys
import threading
import time
from core.foundation import atomic_json, canonical_digest

ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class RuntimePaths:
    """External writable state; source assets remain independent of the host."""
    source_root: Path
    state_root: Path
    environment_root: Path
    build_root: Path
    cache_root: Path
    runs_root: Path
    evidence_root: Path


def runtime_paths(source_root=None, state_root=None):
    """Resolve locations without creating directories, including during preflight."""
    source = Path(source_root or ROOT.parent).expanduser().resolve()
    selected = state_root or os.environ.get('RESEARCH2_STATE_ROOT')
    state = Path(selected).expanduser().resolve() if selected else source.parent/'research2-runtime'
    if state == source or state.is_relative_to(source):
        raise ValueError('Runtime state must be outside the source tree')
    return RuntimePaths(source, state, state/'environments', state/'builds',
                        state/'cache', state/'runs', state/'evidence')


def resolve_runtime_asset(reference, paths=None):
    """Resolve an explicit source/state asset; reject unqualified relative paths."""
    paths = paths or runtime_paths()
    if isinstance(reference, (str, Path)):
        value = Path(reference).expanduser()
        if not value.is_absolute():
            raise ValueError('Relative assets must declare source_path or state_path')
        return value.resolve()
    if not isinstance(reference, dict) or len(reference) != 1:
        raise ValueError('Asset reference requires exactly one source_path or state_path')
    key, value = next(iter(reference.items()))
    if key not in {'source_path', 'state_path'} or not isinstance(value, str):
        raise ValueError('Unknown asset reference')
    root = paths.source_root if key == 'source_path' else paths.state_root
    relative = Path(value)
    target = (root/relative).resolve()
    if relative.is_absolute() or not target.is_relative_to(root):
        raise ValueError('Asset reference escapes its declared root')
    return target



def _cgroup_limit_bytes(paths):
    """Return the smallest finite cgroup memory limit, if one is exposed."""
    limits = []
    for path in paths:
        try:
            with open(path,"r",encoding="ascii") as stream:
                raw=stream.read().strip()
            if raw == "max":
                continue
            value = int(raw)
            # Some cgroup v1 systems use a huge sentinel for unlimited memory.
            if 0 < value < (1 << 60):
                limits.append(value)
        except (OSError, ValueError):
            continue
    return min(limits) if limits else None



def cgroup_available_memory(limit_path, usage_path, stat_path, *, inactive_key='inactive_file'):
    """Cgroup headroom including reclaimable inactive file cache, never anon RAM."""
    try:
        raw = Path(limit_path).read_text(encoding='ascii').strip()
        if raw == 'max':
            return None
        limit = int(raw)
        current = int(Path(usage_path).read_text(encoding='ascii').strip())
        if not 0 < limit < (1 << 60) or current < 0:
            return None
    except (OSError, ValueError):
        return None
    inactive = 0
    try:
        for line in Path(stat_path).read_text(encoding='ascii').splitlines():
            fields = line.split()
            if len(fields) == 2 and fields[0] == inactive_key:
                inactive = min(current, max(0, int(fields[1])))
                break
    except (OSError, ValueError):
        pass  # Missing cache metadata keeps the conservative charged-byte bound.
    return min(limit, max(0, limit - current + inactive))


def cpu_count():
    """Return usable CPU slots, respecting affinity and cgroup quotas."""
    try:
        count = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        count = os.cpu_count() or 1
    count = max(1, count)
    # Slurm may leave affinity unrestricted even for a small shared-node
    # allocation. Its current-node CPU declaration caps our local workers.
    allocated = os.environ.get('SLURM_CPUS_ON_NODE')
    if allocated is not None:
        if not allocated.isdigit() or int(allocated) < 1:
            raise ValueError('SLURM_CPUS_ON_NODE must declare positive allocated local CPUs')
        count = min(count, int(allocated))

    # nproc and sched_getaffinity respect CPU pinning, but can miss a container
    # CPU quota. A quota of 4 CPU-seconds per second permits four build jobs.
    try:
        with open("/sys/fs/cgroup/cpu.max", "r", encoding="ascii") as f:
            quota, period = f.read().split()[:2]
        if quota != "max":
            count = min(count, max(1, math.ceil(int(quota) / int(period))))
    except (OSError, ValueError, ZeroDivisionError):
        try:
            with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us", "r", encoding="ascii") as qf:
                quota = int(qf.read().strip())
            with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us", "r", encoding="ascii") as pf:
                period = int(pf.read().strip())
            if quota > 0 and period > 0:
                count = min(count, max(1, math.ceil(quota / period)))
        except (OSError, ValueError, ZeroDivisionError):
            pass
    return count


def _positive_integer(value, name):
    """Read counts exactly: never truncate fractions or promote invalid zeros."""
    if isinstance(value, Integral) and not isinstance(value, bool):
        result = int(value)
    elif isinstance(value, str):
        text = value.strip()
        if text.startswith('+'):
            text = text[1:]
        if not text or not text.isascii() or not text.isdecimal():
            raise ValueError(name + ' must be a positive integer')
        result = int(text)
    else:
        raise ValueError(name + ' must be a positive integer')
    if result < 1:
        raise ValueError(name + ' must be a positive integer')
    return result


def _configured_count(cfg, name, maximum):
    value = (cfg or {}).get('hardware', {}).get(name, 'auto')
    if isinstance(value, str) and value.strip().lower() == 'auto':
        return maximum
    return min(_positive_integer(value, 'hardware.' + name), maximum)


def cpu_thread_budget(cfg=None):
    """Return the configured CPU pool, capped to the allocated host slots."""
    return _configured_count(cfg, 'cpu_threads', cpu_count())


def mpi_ranks(cfg):
    """Resolve the number of MPI processes within the allocated CPU budget."""
    return _configured_count(cfg, 'mpi_ranks', cpu_thread_budget(cfg))


def build_jobs(cfg=None):
    """Resolve the native compilation parallelism within available CPU resources."""
    return _configured_count(cfg, 'build_jobs', cpu_thread_budget(cfg))


def omp_threads(cfg, mpi_ranks_override=None):
    """Choose threads per MPI rank without oversubscribing the CPU budget."""
    budget = cpu_thread_budget(cfg)
    ranks = (
        mpi_ranks(cfg) if mpi_ranks_override is None
        else _positive_integer(mpi_ranks_override, 'MPI rank override')
    )
    if ranks > budget:
        raise ValueError('MPI rank override exceeds the allocated CPU budget')
    return _configured_count(cfg, 'omp_threads', budget // ranks)


def memory_limit_gb(cfg):
    """Cap a configured budget at assigned host RAM; auto uses 80% of it."""
    value = cfg.get("hardware", {}).get("memory_limit_gb", "auto")
    requested = None
    if not (isinstance(value, str) and value.strip().lower() == 'auto'):
        try:
            requested = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("hardware.memory_limit_gb must be finite and positive, or auto") from exc
        if isinstance(value, bool) or not math.isfinite(requested) or requested <= 0:
            raise ValueError("hardware.memory_limit_gb must be finite and positive, or auto")
    visible_bytes = None
    try:
        with open("/proc/meminfo", "r", encoding="ascii") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    kib = float(line.split()[1])
                    visible_bytes = int(kib * 1024)
                    break
    except (OSError, ValueError, IndexError):
        pass
    if visible_bytes is None:
        # macOS has no /proc/meminfo. POSIX sysconf also provides a physical
        # cap when procfs is unavailable; scheduler/cgroup limits below still
        # take precedence over that host total.
        try:
            pages=os.sysconf('SC_PHYS_PAGES');page_size=os.sysconf('SC_PAGE_SIZE')
            if isinstance(pages,int) and isinstance(page_size,int) and pages>0 and page_size>0:
                visible_bytes=pages*page_size
        except (OSError,ValueError,AttributeError):pass
    # Slurm's memory limit is MiB; some sites enforce it through cgroups,
    # others expose only these allocation variables.
    scheduler_bytes = None
    for variable, multiplier in (('SLURM_MEM_PER_NODE', 1), ('SLURM_MEM_PER_CPU', cpu_count())):
        raw = os.environ.get(variable)
        if raw is None:
            continue
        if not raw.isdigit() or int(raw) < 0:
            raise ValueError(variable+' must declare allocated memory in MiB')
        if int(raw) > 0:  # Slurm zero means the node's full memory.
            candidate = int(raw) * multiplier * 1024**2
            scheduler_bytes = candidate if scheduler_bytes is None else min(scheduler_bytes,candidate)
    if scheduler_bytes is not None:
        visible_bytes = scheduler_bytes if visible_bytes is None else min(visible_bytes,scheduler_bytes)
    cgroup_bytes = _cgroup_limit_bytes((
        "/sys/fs/cgroup/memory.max",
        "/sys/fs/cgroup/memory/memory.limit_in_bytes",
    ))
    if visible_bytes is not None:
        if cgroup_bytes is not None:
            visible_bytes = min(visible_bytes, cgroup_bytes)
    elif cgroup_bytes is not None:
        visible_bytes = cgroup_bytes
    if visible_bytes is not None:
        assigned_gib = visible_bytes / (1024.0 ** 3)
        return min(requested, assigned_gib) if requested is not None else 0.80 * assigned_gib
    return requested if requested is not None else 4.0


def initial_condition_census(paths):
    """Read complete HDF5 IC species counts without allocating particle arrays.

    This is a resource census, not a mass/phase/physics validation. File hashes
    bind the estimate to its inputs; the scientific IC validator is separate.
    """
    from core.foundation import file_sha256
    import h5py
    import numpy as np
    from core.data import snapshot_total_counts
    paths = [Path(p).expanduser().resolve() for p in paths]
    if not paths or len(set(paths)) != len(paths):
        raise ValueError('IC resource census requires distinct complete shards')
    totals = None
    # Python integers cannot wrap if malformed shards overstate a uint64 total.
    observed = [0] * 6
    files=[]
    for path in paths:
        checksum=file_sha256(path)
        with h5py.File(path) as source:
            header=source['Header'].attrs
            declared=snapshot_total_counts(header)
            shards=np.asarray(header['NumFilesPerSnapshot'])
            if shards.shape != () or shards.dtype.kind not in 'iu' or int(shards) != len(paths):
                raise ValueError('IC resource census is missing declared shards')
            if totals is None:
                totals=declared
            elif not np.array_equal(totals,declared):
                raise ValueError('IC resource census shards disagree on total counts')
            this=np.asarray(header['NumPart_ThisFile'])
            if this.shape != (6,) or this.dtype.kind not in 'iu' or np.any(this<0):
                raise ValueError('IC resource census requires integer species counts')
            for t,n in enumerate(this):
                group=source.get(f'PartType{t}')
                actual=0 if group is None else len(group['Coordinates'])
                if actual != int(n):
                    raise ValueError('IC resource census count differs from datasets')
                if actual and (group['Coordinates'].shape != (actual,3)
                               or group['Velocities'].shape != (actual,3)
                               or group['ParticleIDs'].shape != (actual,)):
                    raise ValueError('IC resource census has incomplete particle fields')
            for species, count in enumerate(this):
                observed[species] += int(count)
        if file_sha256(path) != checksum:
            raise ValueError('IC changed during resource census')
        files.append({'path':str(path),'sha256':checksum})
    if observed != [int(count) for count in totals] or not sum(observed):
        raise ValueError('IC resource census does not close declared counts')
    return {'status':'CENSUS_COMPLETE','particles_by_type':[int(n) for n in observed],
            'files':files,'scope':'Dataset/header resource counts only; physics remains separately validated'}


PERIODIC_IMAGE_STATE_BYTES = 48  # two uint64, one double, three int64; native struct/snapshot layout


def periodic_image_resource_allowance(particle_count):
    """Explicit additive state budget; legacy snapshot calibration predates it.

    Public flagship builds enable PERIODIC. Retain this conservative allowance
    for externally planned profiles too; it never certifies allocator/RSS use.
    """
    if type(particle_count) is not int or particle_count<0:
        raise ValueError('Exact nonnegative native periodic-state particle count required')
    return particle_count*PERIODIC_IMAGE_STATE_BYTES


def estimate_run_resources(cfg, snapshot_count, *, particle_counts=None):
    """Planning estimates, not a memory allocation guarantee.

    Snapshot disk calibration comes from the pinned 32^3 integration checks.
    The rough RAM model budgets 256 bytes per gravity particle, another 512
    per gas particle, and 64 bytes per global PM cell for density/FFT work.
    Actual tree/domain/SPH allocation depends on MPI decomposition and load;
    GADGET's measured allocation log is authoritative after execution.
    """
    simulation = cfg["simulation"]
    particles_per_dimension = _positive_integer(
        simulation['particles_per_dimension'], 'simulation.particles_per_dimension',
    )
    snapshot_count = _positive_integer(snapshot_count, 'snapshot_count')

    physics = str(simulation.get("physics", "dm")).lower()
    if physics not in {"dm", "hydro"}:
        raise ValueError("simulation.physics must be dm or hydro for storage estimation")
    calibration_bytes = 6.3 * 1024**2 if physics == "hydro" else 1.8 * 1024**2
    counts = ([particles_per_dimension**3 if physics == 'hydro' else 0,
               particles_per_dimension**3,0,0,0,0] if particle_counts is None else particle_counts)
    if (not isinstance(counts,(list,tuple)) or len(counts)!=6
            or any(type(value) is not int or value<0 for value in counts)
            or not sum(counts) or (physics=='dm' and counts[0])):
        raise ValueError('Resource census must contain six nonnegative integer species counts compatible with physics')
    gas_count, gravity_count = counts[0], sum(counts)
    # Keep both original measured calibrations for uniform runs. External
    # multilevel ICs scale with actual gas/gravity counts, not the finest N^3.
    snapshot_bytes = int(math.ceil((1.8*1024**2*(gravity_count-gas_count)
                                  +4.5*1024**2*gas_count)/32**3))
    snapshot_bytes += 8*counts[2]  # variable coarse-mass field allowance
    periodic_image_bytes=periodic_image_resource_allowance(gravity_count)
    snapshot_bytes+=periodic_image_bytes
    diagnostics=simulation.get('output_acceleration_diagnostics',False)
    if type(diagnostics)is not bool:raise ValueError('Acceleration diagnostic storage selection requires an explicit boolean')
    # Normal output uses doubles: three gravity components on every carrier,
    # plus three stored hydro components on gas, before the existing reserve.
    acceleration_bytes=(24*(gravity_count+gas_count)+16*gas_count) if diagnostics else 0
    snapshot_bytes+=acceleration_bytes
    bh_enabled = cfg.get('black_holes', {}).get('enabled', False)
    if type(bh_enabled) is not bool:
        raise ValueError('black_holes.enabled must be an explicit boolean')
    bh_count = 0
    if bh_enabled:
        from core.black_holes import native_parameters
        native_parameters(cfg["black_holes"])
        bh_count = min(gas_count, cfg["black_holes"]["maximum_count"])
        # Four doubles saved per sink. A seed converts an existing gas particle.
        snapshot_bytes += 32 * bh_count
    requested_snapshot_bytes = snapshot_bytes * int(snapshot_count)
    initial_condition_allowance = snapshot_bytes
    before_reserve = requested_snapshot_bytes + initial_condition_allowance
    estimate = int(math.ceil(before_reserve * 1.5 + 256 * 1024**2))

    n = particles_per_dimension
    pm = simulation.get("pm_grid", "auto")
    pm = n if pm == 'auto' else _positive_integer(pm, 'simulation.pm_grid')
    # Hydro starts with N^3 gas plus N^3 DM particles: both need the
    # gravity allocation, and gas additionally needs the SPH allocation.
    particle_bytes = 256*gravity_count + 512*gas_count
    particle_bytes+=periodic_image_bytes
    acceleration_epoch_bytes=16*gas_count if diagnostics else 0
    particle_bytes+=acceleration_epoch_bytes
    # Four BH doubles on every gravity particle, one transient double per gas
    # particle; replicated MPI source records need separate heap headroom.
    bh_memory_bytes = (32*gravity_count + 8*gas_count + 512*bh_count) if bh_enabled else 0
    particle_bytes += bh_memory_bytes
    memory = particle_bytes + 64 * pm**3
    return {"physics": physics, "particles_per_dimension": n,
            'particles_by_type':list(counts),
            'particle_count_source':'uniform_configuration' if particle_counts is None else 'external_ic_census',
            "snapshot_calibration_bytes": int(calibration_bytes),
            "black_hole_memory_allowance_bytes": bh_memory_bytes,
            "periodic_image_memory_allowance_bytes":periodic_image_bytes,
            "periodic_image_bytes_per_snapshot":periodic_image_bytes,
            "acceleration_diagnostic_bytes_per_snapshot":acceleration_bytes,
            "hydro_force_epoch_memory_allowance_bytes":acceleration_epoch_bytes,
            "disk_bytes": estimate, "memory_bytes": memory,
            "disk_gib": estimate / 1024**3, "memory_gib": memory / 1024**3}


def resolution_followup_resources(cfg, particle_counts, snapshot_count, refinement):
    """Bound refinement costs by actual species counts, retaining coarse levels.

    Forecasts are preflight estimates; regenerated IC census/native MUSIC mesh
    planning are mandatory before launch. No uniform finest N^3 assumption.
    """
    if type(refinement['maximum_levels']) is not int or refinement['maximum_levels'] < 0:
        raise ValueError('Resolution follow-up levels must be a nonnegative integer')
    limits = {}
    for name in ('mass_reduction_factor', 'memory_gib', 'disk_gib'):
        value = refinement[name]
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value <= 0:
            raise ValueError('Resolution follow-up '+name+' must be finite and positive')
        limits[name] = float(value)
    factor = limits['mass_reduction_factor']
    if factor <= 1:
        raise ValueError('Resolution mass reduction must be greater than one')
    rows = []
    for level in range(refinement['maximum_levels']+1):
        counts = [int(math.ceil(n*factor**level)) if t in (0,1,4) else n for t,n in enumerate(particle_counts)]
        sizing = estimate_run_resources(cfg,snapshot_count,particle_counts=counts)
        feasible = sizing['memory_gib'] <= limits['memory_gib'] and sizing['disk_gib'] <= limits['disk_gib']
        rows.append({'level':level,'mass_reduction':factor**level,'estimated_resources':sizing,
                     'status':'WITHIN_ESTIMATED_BUDGET' if feasible else 'RESOURCE_LIMIT'})
    return {'schema':'resolution-followup-resource-v1','levels':rows,
            'scope':'Fixed aperture particle forecast; native MUSIC mesh planning and regenerated complete IC census required',
            'maximum_feasible_level':max((r['level'] for r in rows if r['status']=='WITHIN_ESTIMATED_BUDGET'),default=None)}


def preflight_storage(cfg, run_dir, snapshot_count, require_confirmation=True, *, particle_counts=None):
    """Estimate run storage from measured pinned-build snapshots before work starts.

    The calibration is the package's completed 32^3 GADGET runs (about 1.8 MB
    per DM-only snapshot and 6.3 MB per two-species hydro snapshot). Particle
    data scale with species counts (N^3 by default). The estimate adds one snapshot-sized IC allowance,
    50% for analysis/intermediate products and file-size variation, and a
    fixed 256 MiB reserve. It is a planning bound, not a promise about output
    size; callers can explicitly continue when available disk is below it.
    """
    sizing = estimate_run_resources(cfg, snapshot_count, particle_counts=particle_counts)
    if particle_counts is not None and sizing['memory_gib'] > memory_limit_gb(cfg):
        raise MemoryError('Delivered IC/PM resource estimate exceeds the configured total memory budget')
    physics = sizing["physics"]
    particles_per_dimension = sizing["particles_per_dimension"]
    calibration_bytes = sizing["snapshot_calibration_bytes"]
    estimate = sizing["disk_bytes"]

    target = Path(run_dir).expanduser().resolve()
    probe = target if target.exists() else target.parent
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    available = int(shutil.disk_usage(probe).free)
    result = {
        "status": "PASS" if available >= estimate else "INSUFFICIENT_SPACE_REQUIRES_CONFIRMATION",
        "estimate_bytes": estimate,
        "estimate_gib": estimate / 1024**3,
        "available_bytes": available,
        "available_gib": available / 1024**3,
        "filesystem_probe_path": str(probe),
        "physics": physics,
        "particles_per_dimension": particles_per_dimension,
        'particles_by_type':sizing['particles_by_type'],
        'particle_count_source':sizing['particle_count_source'],
        'estimated_memory_gib':sizing['memory_gib'],
        "requested_snapshot_count": int(snapshot_count),
        "measured_32_cubed_snapshot_calibration_bytes": int(calibration_bytes),
        "method": "measured 32^3 pinned-build snapshot size scaled by species counts; coarse mass allowance; one IC allowance; 50% downstream/intermediate reserve; additional 256 MiB",
    }
    print("Estimated disk space for this run: %.1f GiB; available: %.1f GiB (%s)." % (
        result["estimate_gib"], result["available_gib"], probe,
    ))
    if available < estimate and require_confirmation:
        print("Available space is below the estimate. The run may stop before completion if output fills the filesystem.")
        try:
            answer = input("Type CONTINUE to proceed anyway, or press Enter to stop: ").strip()
        except EOFError:
            answer = ""
        if answer != "CONTINUE":
            raise RuntimeError(
                "Run stopped by the storage preflight. Free disk space, reduce the resolution/output count, "
                "or choose another filesystem."
            )
        result["status"] = "USER_ACCEPTED_BELOW_ESTIMATE"
    return result




# Production allocation models and measured local process usage.
GIB = 1024**3


def _number(value, name, *, integer=False, zero=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 or not zero and value == 0:
        raise ValueError('Invalid resource input: ' + name)
    if integer and type(value) is not int:
        raise ValueError('Integer count required: ' + name)
    return value


def estimate_production_resources(spec):
    """Budget the declared particle census and mesh without altering the run.

    Memory is a peak for sequential follow-ups; disk includes every retained
    child. For an average of N_ngb neighbors, SIDM has N_dm*N_ngb/2 distinct
    pairs. Exchange bytes multiply that pair count by boundary fraction,
    endpoint payload, operator intervals and conflict rounds. Pair timings are
    caller-supplied calibrations, not timings inferred from source inspection.
    """
    _number(spec['rank_imbalance_factor'], 'rank_imbalance_factor')
    if spec['rank_imbalance_factor'] < 1:
        raise ValueError('Rank imbalance factor must be at least one')
    counts = spec['particle_counts']
    if not counts:
        raise ValueError('Actual/preflight particle census required')
    for kind, count in counts.items():
        _number(count, 'particle_counts.'+kind, integer=True, zero=True)
    n = sum(counts.values())
    if not n:
        raise ValueError('Empty production census')
    r = spec['resources']
    ranks = _number(r['mpi_ranks'], 'mpi_ranks', integer=True)
    threads = _number(r['omp_threads'], 'omp_threads', integer=True)
    for key in ('memory_gib', 'disk_gib'):
        _number(r[key], key)
    settings = spec['storage_model']
    for key in ('particle_bytes', 'hydro_extra_bytes', 'stellar_extra_bytes', 'bh_extra_bytes',
                'snapshot_particle_bytes', 'tree_bytes_per_particle', 'mesh_bytes_per_cell',
                'restart_particle_bytes', 'memory_safety_factor', 'disk_safety_factor'):
        _number(settings[key], key)
    if settings['memory_safety_factor'] < 1 or settings['disk_safety_factor'] < 1:
        raise ValueError('Resource safety factors must be at least one')
    snapshots = _number(spec['snapshot_count'], 'snapshot_count', integer=True)
    mesh_side = _number(spec['mesh_side'], 'mesh_side', integer=True)
    followups = spec['followup_limits']
    children = _number(followups['maximum_children'], 'maximum_children', integer=True, zero=True)
    worst_multiplier = _number(followups['maximum_particle_multiplier'], 'maximum_particle_multiplier')
    mesh_multiplier = _number(followups.get('maximum_mesh_cell_multiplier',worst_multiplier),'maximum_mesh_cell_multiplier')
    if mesh_multiplier < 1:
        raise ValueError('Maximum child mesh multiplier must be at least one')
    if worst_multiplier < 1:
        raise ValueError('Maximum child particle multiplier must be at least one')
    gas, stars, bh = (counts.get(q, 0) for q in ('gas', 'stars', 'black_holes'))
    particle_memory = n*(settings['particle_bytes']+settings['tree_bytes_per_particle'])
    particle_memory += gas*settings['hydro_extra_bytes']+stars*settings['stellar_extra_bytes']+bh*settings['bh_extra_bytes']
    mesh_distribution = settings.get('mesh_distribution', 'distributed')
    if mesh_distribution not in {'distributed', 'replicated'}:
        raise ValueError('Explicit distributed or replicated PM mesh allocation required')
    mesh_memory = mesh_side**3*settings['mesh_bytes_per_cell']*(ranks if mesh_distribution == 'replicated' else 1)
    sidm = spec.get('sidm')
    sidm_memory = sidm_storage = exchange_bytes = sidm_replicated = 0
    sidm_cost = None
    if sidm is not None:
        if sidm.get('synchronized_timesteps') is not True:
            raise ValueError('Production SIDM resource model requires synchronized operator scheduling')
        for key in ('neighbors_per_particle', 'neighbor_bytes', 'per_particle_state_bytes',
                    'kernel_table_bytes', 'angle_table_bytes', 'selected_event_count', 'event_bytes',
                    'operator_intervals', 'conflict_rounds_per_interval', 'boundary_pair_fraction',
                    'endpoint_exchange_bytes', 'frequent_pair_seconds', 'rare_pair_seconds'):
            _number(sidm[key], 'sidm.'+key, zero=key in {'selected_event_count', 'boundary_pair_fraction','kernel_table_bytes','angle_table_bytes'})
        if sidm['boundary_pair_fraction'] > 1:
            raise ValueError('Boundary pair fraction cannot exceed one')
        ndm = counts.get('dm', 0)
        if not ndm:
            raise ValueError('Enabled SIDM resource plan requires an actual DM census')
        for key in ('neighbors_per_particle', 'selected_event_count', 'operator_intervals', 'conflict_rounds_per_interval'):
            _number(sidm[key], 'sidm.'+key, integer=True, zero=key == 'selected_event_count')
        neighbors = sidm['neighbors_per_particle']
        pair_count = ndm*neighbors/2
        sidm_memory = ndm*(sidm['per_particle_state_bytes']+neighbors*sidm['neighbor_bytes'])
        sidm_replicated = ranks*(sidm['kernel_table_bytes']+sidm['angle_table_bytes'])
        sidm_memory += sidm_replicated
        sidm_storage = sidm['selected_event_count']*sidm['event_bytes']
        exchange_bytes = pair_count*sidm['boundary_pair_fraction']*sidm['endpoint_exchange_bytes']*sidm['operator_intervals']*sidm['conflict_rounds_per_interval']
        frequent_fraction = _number(sidm['frequent_pair_fraction'], 'frequent_pair_fraction', zero=True)
        if frequent_fraction > 1:
            raise ValueError('Invalid frequent pair fraction')
        source_scheme=sidm.get('source_scheme','rare_frequent_mixture')
        if source_scheme not in {'rare','frequent','hybrid','rare_frequent_mixture'}:
            raise ValueError('Unsupported declared SIDM resource source scheme')
        if source_scheme=='hybrid':
            frequent_calls_per_pair,rare_calls_per_pair=2.,1.
        elif source_scheme=='frequent':
            frequent_calls_per_pair,rare_calls_per_pair=1.,0.
        elif source_scheme=='rare':
            frequent_calls_per_pair,rare_calls_per_pair=0.,1.
        else:
            frequent_calls_per_pair,rare_calls_per_pair=frequent_fraction,1-frequent_fraction
        pair_seconds = frequent_calls_per_pair*sidm['frequent_pair_seconds']+rare_calls_per_pair*sidm['rare_pair_seconds']
        sidm_cost = {'pair_operations': pair_count*sidm['operator_intervals'],
            'source_scheme':source_scheme,
            'frequent_operator_calls':pair_count*sidm['operator_intervals']*frequent_calls_per_pair,
            'rare_operator_calls':pair_count*sidm['operator_intervals']*rare_calls_per_pair,
            'estimated_serial_pair_seconds': pair_count*sidm['operator_intervals']*pair_seconds,
            'exchange_bytes': exchange_bytes, 'synchronization_rounds': sidm['operator_intervals']*sidm['conflict_rounds_per_interval'],
            'estimated_serial_pair_seconds_per_rank':pair_count*sidm['operator_intervals']*pair_seconds/ranks*spec['rank_imbalance_factor'],
            'assumption': 'pair timings and boundary fraction are explicit calibration inputs; synchronization latency is separate'}
    snapshot_bytes = snapshots*n*settings['snapshot_particle_bytes']
    restart_generations = _number(spec['retained_restart_generations'], 'retained_restart_generations', integer=True)
    restart_bytes = restart_generations*n*settings['restart_particle_bytes']
    analysis_bytes = _number(spec['analysis_bytes'], 'analysis_bytes', zero=True)
    assets_bytes = _number(spec['asset_bytes'], 'asset_bytes', zero=True)
    memory = (particle_memory+mesh_memory+sidm_memory)*settings['memory_safety_factor']
    # Replicated tables contribute once per rank; imbalance scales distributed
    # particle/mesh state, not a second copy of each rank's lookup tables.
    replicated_mesh = mesh_memory if mesh_distribution == 'replicated' else 0
    replicated = sidm_replicated+replicated_mesh
    rank_memory = ((particle_memory+mesh_memory+sidm_memory-replicated)/ranks*spec['rank_imbalance_factor']+replicated/ranks)*settings['memory_safety_factor']
    child_replicated = sidm_replicated+replicated_mesh*mesh_multiplier
    child_distributed = (particle_memory+sidm_memory-sidm_replicated)*worst_multiplier+(mesh_memory-replicated_mesh)*mesh_multiplier
    child_memory = (child_distributed+child_replicated)*settings['memory_safety_factor']
    child_rank_memory = (child_distributed/ranks*spec['rank_imbalance_factor']+child_replicated/ranks)*settings['memory_safety_factor']
    worst_memory = max(memory,child_memory) if children else memory
    worst_rank_memory = max(rank_memory,child_rank_memory) if children else rank_memory
    base_storage = snapshot_bytes+restart_bytes+analysis_bytes+assets_bytes+sidm_storage
    worst_storage = base_storage*(1+children*worst_multiplier)*settings['disk_safety_factor']
    return {'schema': 'production-resource-plan-v1', 'specification_hash': canonical_digest(spec),
        'particle_counts': counts, 'mesh_cells': mesh_side**3, 'mpi_ranks': ranks, 'omp_threads': threads,
        'base_memory_bytes':math.ceil(memory),'worst_campaign_peak_memory_bytes':math.ceil(worst_memory),
        'worst_peak_per_rank_memory_bytes':math.ceil(worst_rank_memory), 'peak_per_rank_memory_bytes':math.ceil(rank_memory),
        'base_storage_bytes': math.ceil(base_storage), 'worst_campaign_storage_bytes': math.ceil(worst_storage),
        'sidm': sidm_cost, 'components': {'particles': particle_memory, 'mesh': mesh_memory,
            'sidm_memory':sidm_memory,'sidm_replicated_table_memory':sidm_replicated,'snapshots': snapshot_bytes, 'restarts': restart_bytes,
            'analysis': analysis_bytes, 'assets': assets_bytes, 'sidm_events': sidm_storage},
        'fits_declared_memory':worst_memory <= r['memory_gib']*GIB,
        'fits_declared_disk': worst_storage <= r['disk_gib']*GIB,
        'assumptions': settings, 'followup_limits': followups, 'calibration_status': 'ESTIMATE',
        'mesh_allocation':mesh_distribution,
        'followup_memory_assumption':{'scheduling':'sequential child runs; peak is maximum, not sum',
            'maximum_particle_multiplier':worst_multiplier,'maximum_mesh_cell_multiplier':mesh_multiplier,
            'mesh_multiplier_origin':'explicit' if 'maximum_mesh_cell_multiplier' in followups else 'conservative particle multiplier fallback'},
        'changes_to_requested_configuration':[]}


def require_resources(plan, resources, destination, *, available_memory_bytes=None, available_cpu_count=None):
    if plan['mpi_ranks'] != resources['mpi_ranks'] or plan['omp_threads'] != resources['omp_threads']:
        raise ValueError('Production estimate and execution MPI/thread allocations differ; configuration unchanged')
    topology = resources.get('execution_topology')
    if topology is not None:
        return require_resource_topology(plan, resources, destination, topology)
    available_cpu_count = cpu_count() if available_cpu_count is None else available_cpu_count
    cpu_needed = resources['mpi_ranks']*resources['omp_threads']
    destination = Path(destination).resolve()
    ancestor = destination
    while not ancestor.exists():
        ancestor = ancestor.parent
    free = shutil.disk_usage(ancestor).free
    required_memory = plan.get('worst_campaign_peak_memory_bytes',plan['base_memory_bytes'])
    problems = []
    if available_cpu_count is not None and cpu_needed > available_cpu_count:
        problems.append({'resource': 'CPU slots', 'required': cpu_needed, 'available': available_cpu_count})
    if not plan['fits_declared_memory'] or required_memory > resources['memory_gib']*GIB:
        problems.append({'resource': 'declared total memory', 'required':required_memory, 'available': resources['memory_gib']*GIB})
    if resources.get('per_rank_memory_gib') is not None:
        per_rank = _number(resources['per_rank_memory_gib'],'per_rank_memory_gib')*GIB
        needed = plan.get('worst_peak_per_rank_memory_bytes',plan.get('peak_per_rank_memory_bytes'))
        if needed is None or needed > per_rank:
            problems.append({'resource':'declared per-rank memory','required':needed,'available':per_rank})
    if available_memory_bytes is not None and required_memory > available_memory_bytes:
        problems.append({'resource': 'available memory', 'required':required_memory, 'available': available_memory_bytes})
    if not plan['fits_declared_disk'] or plan['worst_campaign_storage_bytes'] > min(free,resources['disk_gib']*GIB):
        problems.append({'resource': 'campaign disk', 'required': plan['worst_campaign_storage_bytes'], 'available': min(free, resources['disk_gib']*GIB)})
    if problems:
        raise RuntimeError('Production preflight failed; change configuration explicitly: ' + str(problems))
    return {'status': 'PASS', 'plan_hash': canonical_digest(plan), 'changes_to_configuration': []}


def require_resource_topology(plan, resources, destination, topology):
    """Check declared scheduler hosts without comparing remote ranks to local CPUs.

    Host capacities must be supplied by the selected allocation. This function
    neither discovers remote machines nor changes MPI placement to fit them.
    """
    if not isinstance(topology, dict) or topology.get('schema') != 'production-host-allocation-v1':
        raise ValueError('Explicit production-host-allocation-v1 required')
    hosts = topology.get('hosts', [])
    if not hosts or len({host['host_id'] for host in hosts}) != len(hosts):
        raise ValueError('Distinct actual scheduler host allocations required')
    ranks = 0
    problems = []
    host_rows = []
    per_rank = plan['worst_peak_per_rank_memory_bytes']
    for host in hosts:
        if not isinstance(host['host_id'], str) or not host['host_id']:
            raise ValueError('Scheduler host identity required')
        count = _number(host['mpi_ranks'], 'host mpi_ranks', integer=True)
        slots = _number(host['allocated_cpu_slots'], 'allocated_cpu_slots', integer=True)
        memory = _number(host['allocated_memory_gib'], 'allocated_memory_gib')*GIB
        rank_memory = _number(host['maximum_rank_memory_gib'], 'maximum_rank_memory_gib')*GIB
        needed_slots = count*resources['omp_threads']
        needed_memory = count*per_rank
        ranks += count
        if slots < needed_slots or memory < needed_memory or rank_memory < per_rank:
            problems.append({'host_id':host['host_id'], 'required_cpu_slots':needed_slots,
                'allocated_cpu_slots':slots, 'required_memory_bytes':needed_memory,
                'allocated_memory_bytes':memory, 'required_rank_memory_bytes':per_rank,
                'maximum_rank_memory_bytes':rank_memory})
        host_rows.append({'host_id':host['host_id'], 'mpi_ranks':count,
            'estimated_host_peak_bytes':needed_memory, 'estimated_cpu_slots':needed_slots})
    if ranks != resources['mpi_ranks']:
        raise ValueError('Scheduler host rank census differs from the explicit requested MPI count')
    filesystem = topology.get('output_filesystem', {})
    allocated = _number(filesystem['allocated_disk_gib'], 'allocated output filesystem disk')*GIB
    if filesystem.get('capacity_scope') not in {'local_destination', 'scheduler_declared_shared'}:
        raise ValueError('Output filesystem capacity scope must identify actual local or shared storage')
    free = allocated
    if filesystem['capacity_scope'] == 'local_destination':
        ancestor = Path(destination).resolve()
        while not ancestor.exists():
            ancestor = ancestor.parent
        free = min(free, shutil.disk_usage(ancestor).free)
    required = plan['worst_campaign_storage_bytes']
    if not plan['fits_declared_memory'] or not plan['fits_declared_disk'] or required > min(free, resources['disk_gib']*GIB):
        problems.append({'resource':'campaign allocation', 'required_storage_bytes':required,
            'available_storage_bytes':min(free, resources['disk_gib']*GIB),
            'required_memory_bytes':plan['worst_campaign_peak_memory_bytes']})
    if problems:
        raise RuntimeError('Production host preflight failed; change configuration explicitly: '+str(problems))
    return {'status':'PASS', 'plan_hash':canonical_digest(plan), 'topology_hash':canonical_digest(topology),
        'hosts':host_rows, 'capacity_scope':filesystem['capacity_scope'], 'changes_to_configuration':[]}


def update_from_actual_stage_receipts(plan, receipts):
    """Compare retained payload sizes and sealed, scoped runtime observations.

    Sampling is reported alongside exact native telemetry when provided. It
    never becomes an exact RSS peak or a substitute for missing MPI rank data.
    """
    import json
    from core.workflow import verify_completed_receipt
    from core.foundation import file_sha256
    measured = []
    seen = set()
    retained_payloads = {}
    for path in receipts:
        receipt_path = Path(path).resolve()
        if receipt_path in seen:
            raise ValueError('Repeated resource receipt would double-count retained storage')
        seen.add(receipt_path)
        record = verify_completed_receipt(receipt_path)
        for name, digest in record['files_sha256'].items():
            payload = str((Path(record['generation'])/name).resolve())
            if payload in retained_payloads:
                raise ValueError('Overlapping resource stage payloads would double-count storage')
            retained_payloads[payload] = digest
        metrics = record.get('measurements', {})
        sampled = None
        entry = metrics.get('resource_samples_path')
        if entry is not None:
            generation = Path(record['generation']).resolve()
            sample_path = (generation/entry).resolve()
            if not sample_path.is_relative_to(generation) or entry not in record['files_sha256']:
                raise ValueError('Native resource observations must be sealed in their stage generation')
            if file_sha256(sample_path) != record['files_sha256'][entry]:
                raise ValueError('Native resource observations changed')
            sampled = json.loads(sample_path.read_text())
            if sampled.get('schema') != 'native-local-resource-samples-v1' or sampled['sampled_local_peak_rss_bytes'] != metrics.get('sampled_local_peak_rss_bytes'):
                raise ValueError('Resource measurement fields differ from retained native observations')
        values = {}
        for key in ('peak_rss_bytes','sidm_pair_operations','sidm_exchange_bytes','sidm_synchronization_rounds',
                    'sidm_pair_cpu_seconds','sidm_rare_cpu_seconds','sidm_frequent_cpu_seconds',
                    'sidm_peak_rank_rss_bytes','sidm_sum_rank_rss_peaks_upper_bound_bytes',
                    'sidm_exchange_buffer_payload_peak_rank_bytes','sidm_kernel_lookup_table_memory_bytes',
                    'sidm_angular_lookup_table_memory_bytes','sidm_frequent_updates','sidm_event_log_bytes'):
            value = metrics.get(key)
            if value is not None:
                _number(value,key,zero=True)
            values[key] = value
        measured.append({'stage_id':record['stage_id'],'fingerprint':record['fingerprint'],
            'receipt_path':str(receipt_path),'elapsed_seconds':record['elapsed_seconds'],**values,
            'output_bytes':sum((Path(record['generation'])/q).stat().st_size for q in record['files_sha256']),
            'sampled_local_peak_rss_bytes':sampled['sampled_local_peak_rss_bytes'] if sampled else None,
            'local_sampling_coverage':sampled['coverage_status'] if sampled else 'UNAVAILABLE',
            'sampling_limits_reached':sampled['limits_reached'] if sampled else [],
            'observed_local_cpu_seconds':sampled['observed_process_cpu_seconds'] if sampled else None})
    actual_storage = sum(q['output_bytes'] for q in measured)
    sampled_rss = [q['sampled_local_peak_rss_bytes'] for q in measured if q['sampled_local_peak_rss_bytes'] is not None]
    exact_rss = [q['peak_rss_bytes'] for q in measured if q['peak_rss_bytes'] is not None]
    sidm = plan.get('sidm')
    comparison = {}
    if sidm:
        for metric,estimate in (('sidm_pair_operations','pair_operations'),('sidm_exchange_bytes','exchange_bytes'),
                                ('sidm_synchronization_rounds','synchronization_rounds'),
                                ('sidm_pair_cpu_seconds','estimated_serial_pair_seconds')):
            available = [q[metric] for q in measured if q[metric] is not None]
            comparison[metric] = {'estimated':sidm[estimate],'measured_supplied_stage_sum':sum(available) if available else None,
                'unmeasured_stage_ids':[q['stage_id'] for q in measured if q[metric] is None]}
    return {'schema':'resource-estimate-measurement-v1','plan_hash':canonical_digest(plan),
        'estimated_base_storage_bytes':plan['base_storage_bytes'],'measured_storage_bytes':actual_storage,
        'storage_residual_bytes':actual_storage-plan['base_storage_bytes'],'stages':measured,
        'estimated_total_memory_bytes':plan['base_memory_bytes'],
        'maximum_supplied_exact_rss_bytes':max(exact_rss) if exact_rss else None,
        'maximum_sampled_local_rss_bytes':max(sampled_rss) if sampled_rss else None,
        'sidm_comparison':comparison,
        'native_sidm_resources':{
            'measured_operator_cpu_seconds':sum(q['sidm_pair_cpu_seconds'] for q in measured if q['sidm_pair_cpu_seconds'] is not None),
            'operator_cpu_measurement_stage_ids':[q['stage_id'] for q in measured if q['sidm_pair_cpu_seconds'] is not None],
            'maximum_measured_rank_process_rss_bytes':max((q['sidm_peak_rank_rss_bytes'] for q in measured if q['sidm_peak_rank_rss_bytes'] is not None),default=None),
            'rss_scope':'rank process lifetime peaks; separate from simultaneous total MPI memory',
            'maximum_single_exchange_payload_rank_bytes':max((q['sidm_exchange_buffer_payload_peak_rank_bytes'] for q in measured if q['sidm_exchange_buffer_payload_peak_rank_bytes'] is not None),default=None)},
        'coverage':{'supplied_stage_count':len(measured), 'all_supplied_stages_have_exact_rss':bool(measured) and len(exact_rss)==len(measured),
            'all_supplied_stages_have_sidm_counters':bool(measured) and all(q['sidm_pair_operations'] is not None for q in measured) if sidm else None},
        'missing_rss_measurements':[q['stage_id'] for q in measured if q['peak_rss_bytes'] is None],
        'calibration_scope':'retained supplied stages only; local session sampling excludes remote MPI and is not exact physical or peak memory',
        'configuration_changes':[]}


def publish_resource_comparison(plan, receipts, destination):
    """Retain the exact estimate, measured receipt pins and calibrated pair costs."""
    from core.foundation import atomic_json, atomic_text, file_sha256
    receipts = tuple(receipts)
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError('Resource comparison publications cannot overwrite retained estimates')
    result = update_from_actual_stage_receipts(plan, receipts)
    pins = {str(Path(path).resolve()):file_sha256(path) for path in receipts}
    operation_rows = [row for row in result['stages'] if row['sidm_pair_operations'] is not None and row['sidm_pair_cpu_seconds'] is not None]
    operations = sum(row['sidm_pair_operations'] for row in operation_rows)
    seconds = sum(row['sidm_pair_cpu_seconds'] for row in operation_rows)
    result.update(plan=plan, input_hashes=pins, source_hashes={str(Path(__file__).resolve()):file_sha256(__file__)},
        execution_status='COMPLETE', calibration={'scope':'supplied actual operator measurements only',
            'measured_pair_operations':operations if operation_rows else None,
            'measured_pair_cpu_seconds':seconds if operation_rows else None,
            'measured_cpu_seconds_per_pair':seconds/operations if operations else None,
            'stage_ids':[row['stage_id'] for row in operation_rows],
            'applied_configuration_changes':[]})
    for path, digest in pins.items():
        if file_sha256(path) != digest:
            raise ValueError('Measured resource receipt changed during publication')
    destination.mkdir(parents=True)
    path = destination/'resource-comparison.json'
    atomic_json(path, result)
    atomic_text(path.with_suffix('.sha256'), file_sha256(path)+'\n')
    return result


def parse_proc_stat(text, *, page_bytes, clock_ticks):
    """Parse comm safely even when it includes spaces or closing parentheses."""
    first,last = text.find('('),text.rfind(')')
    if first < 0 or last <= first:
        raise ValueError('Invalid Linux process stat record')
    pid = int(text[:first].strip())
    fields = text[last+1:].split()
    if len(fields) < 22 or page_bytes <= 0 or clock_ticks <= 0:
        raise ValueError('Incomplete Linux process stat record')
    return {'pid':pid,'parent_pid':int(fields[1]),'session_id':int(fields[3]),
        'start_ticks':int(fields[19]),'rss_bytes':max(0,int(fields[21]))*page_bytes,
        'cpu_seconds':(int(fields[11])+int(fields[12]))/clock_ticks}


def validate_sampling_options(options):
    allowed = {'interval_seconds','maximum_samples','maximum_processes','maximum_proc_entries'}
    if not isinstance(options,dict) or not set(options) <= allowed:
        raise ValueError('Unknown native resource sampling options')
    interval_seconds = options.get('interval_seconds',.5)
    maximum_samples = options.get('maximum_samples',10000)
    maximum_processes = options.get('maximum_processes',10000)
    maximum_proc_entries = options.get('maximum_proc_entries',100000)
    if isinstance(interval_seconds,bool) or not isinstance(interval_seconds,(int,float)) or not math.isfinite(interval_seconds) or not .05 <= interval_seconds <= 5:
        raise ValueError('Resource sampling interval must be within [0.05,5] seconds')
    for value in (maximum_samples,maximum_processes,maximum_proc_entries):
        if type(value) is not int or value < 1:
            raise ValueError('Resource sampling bounds must be positive integers')


class LocalProcessResourceSampler:
    def __init__(self, output, *, interval_seconds=.5, maximum_samples=10000,
                 maximum_processes=10000, maximum_proc_entries=100000):
        validate_sampling_options({'interval_seconds':interval_seconds,'maximum_samples':maximum_samples,
            'maximum_processes':maximum_processes,'maximum_proc_entries':maximum_proc_entries})
        self.output = Path(output)
        self.interval = interval_seconds
        self.maximum_samples = maximum_samples
        self.maximum_processes = maximum_processes
        self.maximum_proc_entries = maximum_proc_entries
        self.stop_event = threading.Event()
        self.samples,self.processes = [],{}
        self.sample_count = 0
        self.peak = None
        self.limits_reached = set()
        self.errors = []
        self.thread = None
        self.root_pid = None
        self.root_start = None
        self.started = None
        self.finished = False
        self.available = sys.platform.startswith('linux') and Path('/proc/self/stat').is_file()
        if self.available:
            self.page_bytes = os.sysconf('SC_PAGE_SIZE')
            self.clock_ticks = os.sysconf('SC_CLK_TCK')

    def start(self,pid):
        if self.root_pid is not None:
            raise ValueError('Resource sampler cannot observe a second invocation')
        self.root_pid = pid
        self.started = time.monotonic()
        if not self.available:
            return
        try:
            root = parse_proc_stat((Path('/proc')/str(pid)/'stat').read_text(),
                page_bytes=self.page_bytes,clock_ticks=self.clock_ticks)
            self.root_start = root['start_ticks']
        except (OSError,ValueError):
            self.errors.append('root process exited or was unreadable before its first observation')
            return
        self._sample()
        self.thread = threading.Thread(target=self._loop,name='native-resource-observer',daemon=True)
        self.thread.start()

    def _loop(self):
        while not self.stop_event.wait(self.interval):
            try:
                self._sample()
            except (OSError,ValueError) as error:
                if len(self.errors) < 32:
                    self.errors.append(type(error).__name__+': '+str(error))

    def _sample(self):
        # A new-session invocation is the owner. Retaining start ticks prevents
        # treating a reused root PID/session as the original native process.
        root_path = Path('/proc')/str(self.root_pid)
        try:
            root = parse_proc_stat((root_path/'stat').read_text(),page_bytes=self.page_bytes,clock_ticks=self.clock_ticks)
        except (OSError,ValueError):
            return
        if root['start_ticks'] != self.root_start:
            self.limits_reached.add('root_pid_reused')
            return
        rows = []
        scanned = 0
        for path in Path('/proc').iterdir():
            if not path.name.isdecimal():
                continue
            scanned += 1
            if scanned > self.maximum_proc_entries:
                self.limits_reached.add('proc_entry_scan_bound')
                break
            try:
                row = parse_proc_stat((path/'stat').read_text(),page_bytes=self.page_bytes,clock_ticks=self.clock_ticks)
            except (OSError,ValueError):
                continue  # Process races and proc visibility cannot imply zero use.
            if row['session_id'] != self.root_pid:
                continue
            identity = (row['pid'],row['start_ticks'])
            if identity not in self.processes and len(self.processes) >= self.maximum_processes:
                self.limits_reached.add('retained_process_bound')
                continue
            old = self.processes.get(identity)
            self.processes[identity] = {**row,'maximum_observed_rss_bytes':max(row['rss_bytes'],old['maximum_observed_rss_bytes'] if old else 0)}
            rows.append(row)
        total = sum(row['rss_bytes'] for row in rows)
        if rows:
            self.peak = max(total,self.peak or 0)
        self.sample_count += 1
        if len(self.samples) < self.maximum_samples:
            self.samples.append({'elapsed_seconds':time.monotonic()-self.started,
                'observed_process_count':len(rows),'local_session_rss_bytes':total if rows else None})
        else:
            self.limits_reached.add('retained_sample_bound')

    def stop(self):
        if self.finished:
            return
        self.finished = True
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=7.)
            if self.thread.is_alive():
                raise RuntimeError('Resource observer did not stop within its bounded shutdown')
        report = {'schema':'native-local-resource-samples-v1','root_pid':self.root_pid,
            'root_start_ticks':self.root_start,'sampling_interval_seconds':self.interval,
            'sample_count':self.sample_count,'retained_samples':self.samples,
            'sampled_local_peak_rss_bytes':self.peak,'observed_processes':list(self.processes.values()),
            'observed_process_cpu_seconds':sum(q['cpu_seconds'] for q in self.processes.values()) if self.processes else None,
            'coverage_status':'SAMPLED_LOCAL_SESSION' if self.available and self.peak is not None else 'UNAVAILABLE',
            'limits_reached':sorted(self.limits_reached),'observation_errors':self.errors,
            'bounds':{'maximum_samples':self.maximum_samples,'maximum_processes':self.maximum_processes,
                      'maximum_proc_entries':self.maximum_proc_entries},
            'scope':'local owned session only; remote ranks and processes that escape the session are unmeasured',
            'rss_semantics':'approximate sampled sum; shared resident pages can be counted by more than one process; no exact peak claim'}
        atomic_json(self.output,report)


def resources_main():
    import argparse
    p = argparse.ArgumentParser(description='Estimate explicit production resources without changing resolution or ranks')
    p.add_argument('request', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--measurements', type=Path)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError('Resource reports cannot overwrite history')
    plan = estimate_production_resources(json.loads(a.request.read_text()))
    if a.measurements:
        plan = update_from_actual_stage_receipts(plan, json.loads(a.measurements.read_text()))
    atomic_json(a.output, plan)
    print(json.dumps(plan, indent=2))


def resource_comparison_main():
    """Publish measured/estimated resources without changing run controls."""
    import argparse
    parser = argparse.ArgumentParser(description='Compare a frozen resource plan with actual completed stage measurements')
    parser.add_argument('plan', type=Path)
    parser.add_argument('--receipts', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    publish_resource_comparison(json.loads(args.plan.read_text()), json.loads(args.receipts.read_text()), args.output)
