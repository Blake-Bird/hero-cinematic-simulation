"""Zoom: consolidated scientific implementation."""
from __future__ import annotations

# ========================================================================
# zoom_target: Zoom targeting, catalogue selection, host tracking and coverage.


from pathlib import Path
from core.data import hdf5_scalar_attr, snapshot_total_counts, write_dataset_contract, coordinate_contract, epoch_contract, verify_hdf5_dataset_payloads, snapshot_id_shards, read_selected_group_ids
from core.foundation import KPC_CM
from core.foundation import file_sha256
from core.native import directory_owner
from core.foundation import atomic_json, file_sha256 as digest, publish_directory, product_provenance, verify_product_provenance, canonical_digest, publish_file
from halos import _group_files, _snapshot_files, _read_catalogue_series, _unit_conversion
from scipy.spatial import cKDTree
import json
import math
import os
import shutil
import tempfile
import h5py
import numpy as np


# Zoom target
#
# Exact saved-aperture DM tracing to native pre-displacement coordinates.
#
# The result is a conservative refinement proposal. It does not establish that
# the aperture is a halo, that phases match a regenerated zoom, or that the final
# science region is free of coarse particles.

def periodic_region(points, box, guard):
    """Smallest periodic axis intervals around q, expanded by a declared guard."""
    points = np.asarray(points, dtype=float)
    if (
        points.ndim != 2
        or points.shape[1] != 3
        or not len(points)
        or np.any(~np.isfinite(points))
        or not math.isfinite(box)
        or box <= 0
        or not math.isfinite(guard)
        or guard <= 0
    ):
        raise ValueError(
            "A finite nonempty particle region and positive box/guard are required"
        )
    if np.any(points < 0) or np.any(points >= box):
        raise ValueError("Native Lagrangian coordinates must be wrapped into the box")
    center, extent = [], []
    for axis in range(3):
        values = np.sort(points[:, axis])
        gaps = np.diff(np.r_[values, values[0] + box])
        largest = int(np.argmax(gaps))
        span = box - float(gaps[largest])
        start = float(values[(largest + 1) % len(values)])
        center.append((start + span / 2) % box)
        extent.append(span + 2 * guard)
    if np.any(np.asarray(extent) >= box / 2):
        raise ValueError(
            "Traced region plus guard does not fit the supported nested half-box geometry"
        )
    return np.asarray(center), np.asarray(extent)


def _exact_ids(values):
    values = np.asarray(values)
    if values.ndim != 1 or values.dtype.kind not in "ui" or np.any(values < 0):
        raise ValueError("Tracing requires nonnegative exact integer ParticleIDs")
    return values.astype(np.uint64, copy=False)


def apply_traced_region(cfg, filename):
    filename = Path(filename).expanduser().resolve()
    region = verify_traced_region(filename)
    if (
        region.get("schema") != "darkmattersim-traced-zoom-region-v1"
        or region.get("status") != "TRACED_PROPOSAL"
    ):
        raise ValueError("Refinement requires a native traced-region proposal")
    for entry in [
        region["initial_ic"],
        region["source_manifest"],
        *region["target_snapshots"],
        *region.get('catalogue_binding', {}).get('source_catalogues', []),
    ]:
        if file_sha256(Path(entry["path"])) != entry["sha256"]:
            raise ValueError("Traced-region source changed since particle tracing")
    if (
        file_sha256(filename.parent / "selected_particles.hdf5")
        != region["selected_particles_sha256"]
    ):
        raise ValueError("Traced particle membership changed")
    expected = [
        ("setup", "boxlength", region["box_comoving_kpc_h"] / 1000),
        ("cosmology", "H0", region["h"] * 100),
        ("cosmology", "Omega_m", region["cosmology"]["Omega0"]),
        ("cosmology", "Omega_b", region["cosmology"]["OmegaBaryon"]),
        ("cosmology", "Omega_L", region["cosmology"]["OmegaLambda"]),
    ]
    for section, key, value in expected:
        if not math.isclose(
            cfg.getfloat(section, key), value, rel_tol=1e-12, abs_tol=1e-14
        ):
            raise ValueError("Generator cosmology/box differs from the traced parent")
    for key in ("ref_center", "ref_extent"):
        values = region["music"][key]
        if len(values) != 3 or any(not math.isfinite(x) for x in values):
            raise ValueError("Malformed traced refinement geometry")
        if key == "ref_extent" and any(not 0 < x < 0.5 for x in values):
            raise ValueError(
                "Traced refinement extent exceeds supported nested geometry"
            )
        if key == "ref_center" and any(not 0 <= x < 1 for x in values):
            raise ValueError("Traced refinement center must be periodic box fractions")
        cfg.set("setup", key, ",".join(format(x, ".17g") for x in values))
    cfg.set("setup", "region", "box")
    cfg.set("output", "shift_back", "yes")
    return {"path": str(filename), "sha256": file_sha256(filename)}


def _plot_region(traced, center, extent, box, report, destination):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    for ax, (i, j) in zip(axes.flat, ((0, 1), (0, 2), (1, 2))):
        for t, q in traced.items():
            local = (q - center + box / 2) % box - box / 2
            # Deterministic display thinning; all exact IDs/q remain in HDF5.
            local = local[:: max(1, math.ceil(len(local) / 50_000))]
            ax.scatter(
                local[:, i],
                local[:, j],
                s=5,
                alpha=0.7,
                label="Fine DM" if t == 1 else "Coarse combined matter",
            )
        ax.add_patch(
            Rectangle(
                (-extent[i] / 2, -extent[j] / 2),
                extent[i],
                extent[j],
                fill=False,
                linestyle="--",
                color="black",
                label="Proposed bounds with guard",
            )
        )
        ax.set(
            xlabel=f"q{'xyz'[i]} − center [comoving kpc/h]",
            ylabel=f"q{'xyz'[j]} − center [comoving kpc/h]",
            xlim=(-extent[i] * 0.56, extent[i] * 0.56),
            ylim=(-extent[j] * 0.56, extent[j] * 0.56),
        )
        ax.set_aspect("equal", adjustable="box")
        ax.legend(fontsize=7)
    axes[1, 1].axis("off")
    axes[1, 1].text(
        0.02,
        0.95,
        "Exact native particle trace\n\n"
        + f"Target scale factor: {report['target_scale_factor']:.6g}\n"
        + f"Fine DM selected: {report['selected_counts']['1']}\nCoarse selected: {report['selected_counts']['2']}\n"
        + f"Cell guard: {report['guard']['parent_cell_comoving_kpc_h']:.6g} kpc/h\n"
        + f"Extra padding: {report['guard']['additional_comoving_kpc_h']:.6g} kpc/h\n\n"
        + ("q is before IC displacements.\nExact saved FOF membership is retained.\nEvolving identity, phase agreement and\nnonlinear contamination remain unvalidated."
           if report.get('catalogue_binding') else
           "q is before IC displacements.\nThis aperture is an explicit saved-frame input.\nHalo identity, phase agreement and\nnonlinear contamination remain unvalidated."),
        va="top",
        fontsize=10,
    )
    title = "Traced initial particle region — refinement proposal"
    if report.get("synthetic_fixture"):
        title += "\nSYNTHETIC VALIDATION FIXTURE"
    fig.suptitle(title)
    from figures import export_figure
    export_figure(fig,destination/'traced_zoom_region',formats=['png','pdf'],dpi=160,
        metadata={'epoch':report['initial_epoch'],'selection_epoch':report['target_epoch'],
            'data_contract':report['data_contract'],'science_acceptance':'NOT_ACCEPTED',
            'coordinate_units':'comoving kpc/h; native pre-displacement q'})


@directory_owner('destination')
def trace_aperture(
    initial_ic,
    target_paths,
    source_manifest,
    destination,
    *,
    center_comoving_kpc_h,
    radius_proper_kpc,
    padding_comoving_kpc_h,
    memory_gib=1.0,
    _catalogue_binding=None,
    _member_ids=None,
    _component_members=None,
):
    from analysis import validate_gadget_snapshot_set
    from halos import _unit_conversion

    initial_ic, source_manifest, destination = map(
        lambda p: Path(p).expanduser().resolve(),
        (initial_ic, source_manifest, destination),
    )
    target_paths = [Path(p).expanduser().resolve() for p in target_paths]
    if destination.exists():
        raise FileExistsError("Refusing to overwrite target tracing artifacts")
    center = np.asarray(center_comoving_kpc_h, dtype=float)
    if center.shape != (3,) or np.any(~np.isfinite(center)):
        raise ValueError(
            "Target center must be three finite saved comoving kpc/h coordinates"
        )
    if (
        not math.isfinite(radius_proper_kpc)
        or radius_proper_kpc <= 0
        or not math.isfinite(padding_comoving_kpc_h)
        or padding_comoving_kpc_h < 0
        or not math.isfinite(memory_gib)
        or memory_gib <= 0
    ):
        raise ValueError(
            "Radius/memory must be positive and padding finite/nonnegative"
        )
    if not target_paths:
        raise ValueError("Target tracing requires a complete saved snapshot set")
    source_files=[initial_ic,source_manifest,*target_paths]
    if _catalogue_binding is not None:source_files.extend(Path(item['path']) for item in _catalogue_binding['source_catalogues'])
    provenance=product_provenance(source_manifest.parent,algorithm=__file__,source_files=source_files,
        selection='Exact native target aperture and declared saved membership traced to parent pre-displacement q',
        coordinates=coordinate_contract(position_units='comoving kpc/h',velocity_convention='not exported',
            origin='periodic parent simulation box origin',periodic_wrapping='[0, BoxSize)'))
    manifest = json.loads(source_manifest.read_text())
    handoff = manifest.get("gadget_handoff", {})
    lineage = handoff.get("initial_conditions", [])
    if len(lineage) != 1 or lineage[0].get("sha256") != file_sha256(initial_ic):
        raise ValueError(
            "Target source manifest must identify the unchanged single-file parent IC"
        )
    prefix = handoff.get("initial_condition_prefix_passed_to_gadget")
    if not isinstance(prefix, str) or not prefix:
        raise ValueError(
            "Source handoff must record its exact native initial-condition prefix"
        )
    with h5py.File(target_paths[0]) as source:
        header, params = source["Header"].attrs, source["Parameters"].attrs
        a = hdf5_scalar_attr(header["Time"], "Time")
        box = hdf5_scalar_attr(header["BoxSize"], "BoxSize")
        h = float(params["HubbleParam"])
        cosmology = {
            name: float(params[name])
            for name in ("Omega0", "OmegaLambda", "OmegaBaryon")
        }
        count = int(sum(int(n) for n in snapshot_total_counts(header)[1:3]))
    if (
        not all(math.isfinite(x) and x > 0 for x in (a, box, h))
        or radius_proper_kpc >= box * a / h / 2
    ):
        raise ValueError(
            "Target aperture must fit within half the saved proper periodic box"
        )
    # IDs, selection, q, sort/search workspace and chunk headroom. Reject
    # before reading particle payloads, even for an aperture that ends empty.
    component_count = sum(map(len, _component_members.values())) if _component_members is not None else 0
    if 128 * (count + component_count) + 64 * 1024**2 > memory_gib * 1024**3:
        raise MemoryError("Target tracing exceeds the declared selection memory budget")
    validate_gadget_snapshot_set(target_paths)
    _unit_conversion(target_paths)
    with h5py.File(initial_ic) as source:
        header = source["Header"].attrs
        if header.get("Music_lagrangian_schema") != 1 or not math.isclose(
            float(header.get("Music_lagrangian_unitlength_cm", math.nan)),
            KPC_CM,
            rel_tol=1e-12,
        ):
            raise ValueError(
                "Parent IC needs native pre-displacement Lagrangian schema 1 in comoving kpc/h"
            )
        if int(header["NumFilesPerSnapshot"]) != 1 or not np.array_equal(
            snapshot_total_counts(header), header["NumPart_ThisFile"]
        ):
            raise ValueError("Parent IC must be a complete single-file particle load")
        if not math.isclose(
            float(header["BoxSize"]), box, rel_tol=1e-12
        ) or not math.isclose(float(header["HubbleParam"]), h, rel_tol=1e-12):
            raise ValueError("Parent IC and target disagree about the periodic frame")
        for name, expected in cosmology.items():
            if not math.isclose(
                float(header[name]), expected, rel_tol=1e-12, abs_tol=1e-14
            ):
                raise ValueError("Parent IC and target cosmology differ: " + name)
        if a < float(header["Time"]):
            raise ValueError("Target epoch precedes its parent IC")
        from core.cosmology import run_proper_time_clock
        clock=run_proper_time_clock(source_manifest.parent)
        initial_epoch=epoch_contract(float(header['Time']),clock=clock)
        target_epoch=epoch_contract(a,clock=clock)
        level = int(header["Music_levelmin"])
        if not 1 <= level <= 20:
            raise ValueError("Parent refinement level is invalid")
        # One full coarsest parent cell is a conservative cell-footprint guard,
        # including the baryonic COM staggering. This is not a measured
        # displacement or a guarantee about subsequent nonlinear contamination.
        cell_guard = box / 2**level
    selected = {1: [], 2: []}
    if _member_ids is not None:
        if _catalogue_binding is None or set(_member_ids) != {1, 2}:
            raise ValueError('Catalogue membership requires an explicit source binding for types 1 and 2')
        _member_ids = {t: np.sort(_exact_ids(ids)) for t, ids in _member_ids.items()}
    for path in target_paths:
        with h5py.File(path) as source:
            recorded = source["Parameters"].attrs.get("InitCondFile")
            if isinstance(recorded, bytes):
                recorded = recorded.decode()
            if recorded != prefix:
                # A separate native FoF invocation records its input snapshot
                # as InitCondFile. Follow its sealed input/output mapping to
                # the original evolution snapshot before accepting lineage.
                report_path = source_manifest.parent/'input/halo_finding.json'
                if _catalogue_binding is None or not report_path.is_file():
                    raise ValueError('Target snapshot does not identify the recorded parent IC')
                source_files.extend(Path(item['path']) for item in
                                    _grouped_snapshot_lineage(path,source_manifest,prefix))
            for t in (1, 2):
                group = source.get(f"PartType{t}")
                if group is None:
                    continue
                for first in range(0, len(group["ParticleIDs"]), 100_000):
                    ids = _exact_ids(group["ParticleIDs"][first : first + 100_000])
                    pos = np.asarray(
                        group["Coordinates"][first : first + 100_000], dtype=float
                    )
                    if pos.shape != (len(ids), 3) or np.any(~np.isfinite(pos)):
                        raise ValueError(
                            "Target particle coordinates must be finite and aligned with IDs"
                        )
                    delta = (pos - center + box / 2) % box - box / 2
                    inside = np.linalg.norm(delta, axis=1) * a / h <= radius_proper_kpc
                    if _member_ids is not None and len(_member_ids[t]):
                        # FOF extent can exceed the spherical science aperture.
                        # Retain every original member as well as aperture DM.
                        inside |= np.isin(ids, _member_ids[t], assume_unique=False)
                    selected[t].append(ids[inside])
    selected = {
        t: np.sort(np.concatenate(parts)) if parts else np.empty(0, dtype=np.uint64)
        for t, parts in selected.items()
    }
    combined = np.concatenate(list(selected.values()))
    if not len(combined) or len(np.unique(combined)) != len(combined):
        raise ValueError("Target aperture is empty or contains duplicate selected IDs")
    if _member_ids is not None:
        for t, ids in _member_ids.items():
            if not np.isin(ids, selected[t]).all():
                raise ValueError('Selected catalogue membership is not present in the traced snapshot')
    traced = {}
    with h5py.File(initial_ic) as source:
        for t, wanted in selected.items():
            if not len(wanted):
                continue
            group = source[f"PartType{t}"]
            dataset = group["LagrangianCoordinates"]
            if dataset.dtype != np.float64 or dataset.shape != (
                len(group["ParticleIDs"]),
                3,
            ):
                raise ValueError(
                    "Parent q must be double precision and aligned with exact IDs"
                )
            q, matches = (
                np.full((len(wanted), 3), np.nan),
                np.zeros(len(wanted), dtype=np.int64),
            )
            for first in range(0, len(group["ParticleIDs"]), 100_000):
                ids = _exact_ids(group["ParticleIDs"][first : first + 100_000])
                rows = np.searchsorted(wanted, ids)
                valid = rows < len(wanted)
                valid[valid] &= wanted[rows[valid]] == ids[valid]
                if not np.any(valid):
                    continue
                values = np.asarray(dataset[first : first + len(ids)])[valid]
                q[rows[valid]] = values
                np.add.at(matches, rows[valid], 1)
            if np.any(matches != 1):
                raise ValueError(
                    "Selected target IDs are missing or duplicated in the parent IC"
                )
            traced[t] = q
    points = np.concatenate(list(traced.values()))
    region_center, extent = periodic_region(
        points, box, cell_guard + padding_comoving_kpc_h
    )
    provenance=product_provenance(source_manifest.parent,algorithm=__file__,source_files=source_files,
        selection='Exact native target aperture and declared saved membership traced to parent pre-displacement q',
        coordinates=coordinate_contract(position_units='comoving kpc/h',velocity_convention='not exported',
            origin='periodic parent simulation box origin',periodic_wrapping='[0, BoxSize)'))
    report = {
        "schema": "darkmattersim-traced-zoom-region-v1",
        "synthetic_fixture": "SYNTHETIC" in str(manifest.get("fixture", "")).upper(),
        "status": "TRACED_PROPOSAL",
        "initial_ic": {"path": str(initial_ic), "sha256": file_sha256(initial_ic)},
        "source_manifest": {
            "path": str(source_manifest),
            "sha256": file_sha256(source_manifest),
        },
        "target_snapshots": [
            {"path": str(p), "sha256": file_sha256(p)} for p in target_paths
        ],
        "target_scale_factor": a,
        "data_contract": provenance,
        "initial_epoch": initial_epoch,
        "target_epoch": target_epoch,
        "h": h,
        "box_comoving_kpc_h": box,
        "cosmology": cosmology,
        "selection": {
            "center_comoving_kpc_h": center.tolist(),
            "radius_proper_kpc": radius_proper_kpc,
            "species": [1, 2],
        },
        "selected_counts": {str(t): len(ids) for t, ids in selected.items()},
        "guard": {
            "parent_cell_comoving_kpc_h": cell_guard,
            "additional_comoving_kpc_h": padding_comoving_kpc_h,
        },
        "music": {
            "ref_center": (region_center / box).tolist(),
            "ref_extent": (extent / box).tolist(),
        },
        "science_acceptance": "NOT_ACCEPTED",
        "phase_matching": "UNVALIDATED",
        "nonlinear_contamination": "UNVALIDATED",
    }
    if _catalogue_binding is not None:
        report['catalogue_binding'] = _catalogue_binding
        report['selection']['definition'] = 'union of exact selected FOF collisionless membership and declared R200c aperture'
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".target-", dir=destination.parent))
    try:
        with h5py.File(temporary / "selected_particles.hdf5", "w") as output:
            output.attrs['data_contract_json']=json.dumps(provenance,sort_keys=True)
            output.attrs["coordinate_units"] = (
                "comoving kpc/h; native pre-displacement q"
            )
            for t, q in traced.items():
                group = output.create_group(f"PartType{t}")
                identity=group.create_dataset('ParticleIDs',data=selected[t],chunks=True,fletcher32=True)
                write_dataset_contract(identity,units='identity',quantity='Exact selected parent particle identity',
                    identity_namespace=provenance['identity_namespaces']['dark_matter' if t==1 else 'native_particle'],epoch=target_epoch,provenance=provenance,
                    ancestry={'initial_ic_sha256':report['initial_ic']['sha256'],'species':t,'relationship':'unchanged native particle ID'})
                positions=group.create_dataset('LagrangianCoordinates',data=q,chunks=True,fletcher32=True)
                write_dataset_contract(positions,units='comoving kpc/h',quantity='Native parent pre-displacement Lagrangian position',
                    epoch=initial_epoch,coordinates=provenance['coordinates'],provenance=provenance)
                if _member_ids is not None:
                    mask=group.create_dataset('SelectedHostMember',data=np.isin(selected[t],_member_ids[t]),chunks=True,fletcher32=True)
                    write_dataset_contract(mask,units='dimensionless',quantity='Membership in exact saved target host',
                        epoch=target_epoch,provenance=provenance)
            if _component_members is not None:
                component = output.create_group('SelectedGalaxyBoundMembers')
                for t, ids in _component_members.items():
                    if not len(ids):continue
                    identity=component.create_group(f'PartType{t}').create_dataset('ParticleIDs',data=_exact_ids(ids),chunks=True,fletcher32=True)
                    kind={0:'gas',1:'dark_matter',2:'dark_matter',4:'star_population',5:'black_hole'}.get(t)
                    if kind is None:raise ValueError('Unsupported bound-component species namespace')
                    write_dataset_contract(identity,units='identity',quantity='Exact selected galaxy bound member ID',
                        identity_namespace=provenance['identity_namespaces'][kind],epoch=target_epoch,provenance=provenance)
        report["selected_particles_sha256"] = file_sha256(
            temporary / "selected_particles.hdf5"
        )
        _plot_region(traced, region_center, extent, box, report, temporary)
        report["figures_sha256"] = {
            name: file_sha256(temporary / name)
            for name in ("traced_zoom_region.png", "traced_zoom_region.pdf")
        }
        atomic_json(temporary/'region.json',report)
        (temporary/'region.sha256').write_text(digest(temporary/'region.json')+'\n')
        verify_product_provenance(source_manifest.parent,provenance)
        if _catalogue_binding is not None:
            for entry in _catalogue_binding['source_catalogues']:
                if file_sha256(entry['path']) != entry['sha256']:
                    raise ValueError('Catalogue source changed while tracing the selected host')
        publish_directory(temporary,destination)
    except BaseException as exc:
        if temporary.exists():atomic_json(temporary/'failure.json',{'status':'FAILED','type':type(exc).__name__,'message':str(exc)})
        raise
    return report


def verify_traced_region(filename):
    """Verify sealed tracing products, numeric payloads and generated figures."""
    filename = Path(filename).resolve()
    before = digest(filename)
    record = json.loads(filename.read_text())
    seal = filename.parent / 'region.sha256'
    if 'data_contract' in record and not seal.is_file():
        raise ValueError('Traced-region completion seal is missing')
    if seal.exists() and seal.read_text().strip() != before:
        raise ValueError('Traced-region checksum mismatch')
    particles = filename.parent / 'selected_particles.hdf5'
    if digest(particles) != record['selected_particles_sha256']:
        raise ValueError('Traced particle membership changed')
    verify_hdf5_dataset_payloads(particles, require_contract='data_contract' in record)
    for name, expected in record.get('figures_sha256', {}).items():
        figure = filename.parent / name
        if Path(name).name != name or figure.is_symlink():
            raise ValueError('Traced-region figure path escapes its product')
        if digest(figure) != expected:
            raise ValueError('Traced-region figure checksum mismatch')
    figure_manifest = filename.parent / 'traced_zoom_region.figure.json'
    if figure_manifest.exists():
        from figures import verify_figure_export
        verify_figure_export(figure_manifest)
    if digest(filename) != before:
        raise ValueError('Traced-region report changed during verification')
    return record


# Zoom catalogue target
#
# Bind a zoom refinement proposal to exact native FOF host membership.
#
# Catalogue rows identify objects only at the saved epoch. Checksummed catalogue
# and snapshot sources plus exact member IDs define this selection; subsequent
# identity requires particle continuity or a validated tree, never row equality.

def read_host_catalogue(run_dir, snapshot, *, memory_gib=1., host_row=None):
    """Read a complete native FOF frame and its exact membership spans."""
    run_dir=Path(run_dir).expanduser().resolve()
    if (type(snapshot) is not int or snapshot<0 or isinstance(memory_gib,bool) or not math.isfinite(memory_gib) or memory_gib<=0
            or (host_row is not None and (type(host_row) is not int or host_row<0))):
        raise ValueError('Declare a saved snapshot, valid optional host row and positive memory')
    catalogues, snapshots = _group_files(run_dir, snapshot), _snapshot_files(run_dir, snapshot)
    if not catalogues or not snapshots:
        raise FileNotFoundError('Catalogue target needs complete FOF catalogues and group-ordered snapshots')
    sources = [{'path': str(p.resolve()), 'sha256': file_sha256(p)} for p in catalogues]
    snapshot_sources=[{'path':str(p.resolve()),'sha256':file_sha256(p)} for p in snapshots]
    totals, shards = [], 0
    for path in catalogues:
        with h5py.File(path) as source:
            header = source['Header'].attrs
            counts = [header.get(key) for key in ('NumFiles', 'Ngroups_Total', 'Ngroups_ThisFile')]
            if any(v is None or np.asarray(v).ndim != 0 or np.asarray(v).dtype.kind not in 'ui' or int(v) < 0 for v in counts):
                raise ValueError('FOF catalogue requires exact complete shard/group counts')
            if int(counts[0]) != len(catalogues):
                raise ValueError('FOF catalogue shard set is incomplete')
            totals.append(int(counts[1]))
            shards += int(counts[2])
    if len(set(totals)) != 1 or shards != totals[0] or (host_row is not None and host_row >= totals[0]):
        raise ValueError('FOF catalogue totals or selected row are invalid')
    # Check before allocating whole catalogue vectors or member IDs.
    if totals[0] * 512 + 64 * 1024**2 > memory_gib * 1024**3:
        raise MemoryError('Host catalogue selection exceeds declared memory')
    _unit_conversion(catalogues + snapshots)
    frame = None
    for path in catalogues + snapshots:
        with h5py.File(path) as source:
            header, params = source['Header'].attrs, source['Parameters'].attrs
            values = (hdf5_scalar_attr(header['Time'], 'Time'), hdf5_scalar_attr(header['BoxSize'], 'BoxSize'),
                      float(params['HubbleParam']))
            if not np.isfinite(values).all() or min(values) <= 0:
                raise ValueError('Invalid catalogue/snapshot physical frame')
            if frame is not None and not np.allclose(frame, values, rtol=1e-12, atol=0):
                raise ValueError('Host catalogue and snapshot epoch/box/h mismatch')
            frame = values
    a, box, h = frame
    offsets = _read_catalogue_series(catalogues, 'Group', 'GroupOffsetType')
    lengths = _read_catalogue_series(catalogues, 'Group', 'GroupLenType')
    positions = _read_catalogue_series(catalogues, 'Group', 'GroupPos')
    radii = _read_catalogue_series(catalogues, 'Group', 'Group_R_Crit200')
    if (offsets.shape != (totals[0], 6) or lengths.shape != offsets.shape
            or offsets.dtype.kind not in 'ui' or lengths.dtype.kind not in 'ui'
            or np.any(offsets < 0) or np.any(lengths < 0) or positions.shape != (totals[0], 3)
            or radii.shape != (totals[0],) or not np.isfinite(positions).all()
            or not np.isfinite(radii).all() or np.any(radii < 0) or (host_row is not None and radii[host_row] <= 0)):
        raise ValueError('Invalid FOF host spans, centres or measured R200c')
    for t in (1,2):
        _, total=snapshot_id_shards(snapshots,t)
        spans=sorted((int(o[t]),int(o[t])+int(n[t])) for o,n in zip(offsets,lengths) if n[t])
        if any(stop>total for _,stop in spans) or any(current[0]<previous[1] for previous,current in zip(spans,spans[1:])):
            raise ValueError('FOF collisionless membership is out of range or overlaps')
    for entry in sources+snapshot_sources:
        if file_sha256(entry['path'])!=entry['sha256']:
            raise ValueError('Host catalogue/snapshot changed while reading frame')
    return {'catalogues':catalogues,'snapshots':snapshots,'sources':sources,'snapshot_sources':snapshot_sources,'frame':frame,
            'offsets':offsets,'lengths':lengths,'positions':positions,'radii':radii,'group_count':totals[0]}


def trace_catalogue_host(run_dir, snapshot, host_row, initial_ic, source_manifest, destination, *,
                         aperture_r200c_factor, padding_comoving_kpc_h, memory_gib=1., galaxy_subhalo_row=None):
    """Trace all selected host DM/coarse IDs plus a catalogue-centred aperture.

    aperture_r200c_factor is explicit: this is a scientific region definition,
    not a magic guarantee against contamination. Optional galaxy selection
    must have native stellar members and a valid exact most-bound ID.
    """
    run_dir = Path(run_dir).expanduser().resolve()
    if (type(snapshot) is not int or snapshot < 0 or type(host_row) is not int or host_row < 0
            or isinstance(aperture_r200c_factor, bool) or not math.isfinite(aperture_r200c_factor)
            or aperture_r200c_factor <= 0 or isinstance(memory_gib, bool)
            or not math.isfinite(memory_gib) or memory_gib <= 0):
        raise ValueError('Declare integer snapshot/host and positive aperture factor/memory')
    catalogue=read_host_catalogue(run_dir,snapshot,memory_gib=memory_gib,host_row=host_row)
    catalogues,snapshots,sources=[catalogue[key] for key in ('catalogues','snapshots','sources')]
    a,box,h=catalogue['frame']
    offsets,lengths,positions,radii=[catalogue[key] for key in ('offsets','lengths','positions','radii')]
    totals=[catalogue['group_count']]
    member_ids = {}
    for t in (1, 2):
        records, _ = snapshot_id_shards(snapshots, t)
        count = int(lengths[host_row, t])
        if count * 128 + totals[0] * 512 + 64 * 1024**2 > memory_gib * 1024**3:
            raise MemoryError('Exact host membership exceeds declared selection memory')
        ids, _ = read_selected_group_ids(offsets, lengths, np.array([host_row], dtype=np.int64), records,
                                          particle_type=t, max_ids=count)
        member_ids[t] = ids
    if sum(map(len, member_ids.values())) == 0:
        raise ValueError('Selected host has no collisionless membership to trace')
    galaxy = None
    galaxy_members = None
    if galaxy_subhalo_row is not None:
        if type(galaxy_subhalo_row) is not int or galaxy_subhalo_row < 0:
            raise ValueError('Galaxy subhalo row must be a nonnegative integer')
        hosts = _read_catalogue_series(catalogues, 'Subhalo', 'SubhaloGroupNr')
        stars = _read_catalogue_series(catalogues, 'Subhalo', 'SubhaloLenType')
        cores = _read_catalogue_series(catalogues, 'Subhalo', 'SubhaloIDMostbound')
        bound_offsets = _read_catalogue_series(catalogues, 'Subhalo', 'SubhaloOffsetType')
        row = galaxy_subhalo_row
        if (hosts.ndim != 1 or stars.shape != (len(hosts), 6) or cores.shape != hosts.shape
                or any(v.dtype.kind not in 'ui' or np.any(v < 0) for v in (hosts, stars, cores))
                or row >= len(hosts) or int(hosts[row]) != host_row or int(stars[row, 4]) == 0):
            raise ValueError('Selected galaxy lacks stars or does not belong to the selected FOF host')
        galaxy = {'subhalo_row_at_selected_epoch': row, 'most_bound_particle_id': int(cores[row]),
                  'stellar_member_count': int(stars[row, 4])}
        galaxy_count = sum(map(int, stars[row]))
        host_total_count = sum(map(int, lengths[host_row]))
        if (galaxy_count + host_total_count + sum(map(len, member_ids.values()))) * 128 + totals[0]*512 + 64*1024**2 > memory_gib*1024**3:
            raise MemoryError('Selected galaxy/host exact memberships exceed declared memory')
        galaxy_members = {}
        for t in range(6):
            records, _ = snapshot_id_shards(snapshots, t)
            ids, _ = read_selected_group_ids(bound_offsets, stars, np.array([row], dtype=np.int64), records,
                                              particle_type=t, max_ids=int(stars[row,t]))
            host_ids, _ = read_selected_group_ids(offsets, lengths, np.array([host_row], dtype=np.int64), records,
                                                  particle_type=t, max_ids=int(lengths[host_row,t]))
            if not np.isin(ids, host_ids).all():
                raise ValueError('Galaxy bound members are not contained in their declared FOF host')
            galaxy_members[t] = ids
        all_bound = np.concatenate(list(galaxy_members.values()))
        if len(np.unique(all_bound)) != len(all_bound) or not np.any(all_bound == cores[row]):
            raise ValueError('Galaxy most-bound ID is missing from exact bound membership or IDs overlap')
    binding = {'schema': 'darkmattersim-catalogue-target-v1', 'snapshot': snapshot, 'host_fof_row': host_row,
               'identity_namespace': canonical_digest({'catalogue_sha256': [s['sha256'] for s in sources], 'snapshot': snapshot, 'host_row': host_row}),
               'source_catalogues': sources, 'group_ordered_snapshots': [str(p.resolve()) for p in snapshots],
               'galaxy': galaxy, 'host_member_counts_by_type': lengths[host_row].astype(object).tolist(),
               'host_r200c_comoving_kpc_h': float(radii[host_row]),
               'aperture_r200c_factor': float(aperture_r200c_factor),
               'identity_scope': 'epoch-local exact FOF selection; cross-epoch identity requires bound/core continuity'}
    return trace_aperture(initial_ic, snapshots, source_manifest, destination,
                           center_comoving_kpc_h=(positions[host_row] % box).tolist(),
                           radius_proper_kpc=float(radii[host_row])*a/h*aperture_r200c_factor,
                           padding_comoving_kpc_h=padding_comoving_kpc_h, memory_gib=memory_gib,
                           _catalogue_binding=binding, _member_ids=member_ids,
                           _component_members=galaxy_members)


# Zoom host tracking
#
# Recover saved-epoch FOF apertures through exact anchor-particle continuity.
#
# This tracks an anchor host's collisionless material in the same simulation.
# Rows are epoch-local. Splits, weak retention and competing matches fail rather
# than inventing an orbit or treating a row number as a persistent identity.

def _run_lineage(run_dir):
    path=Path(run_dir).expanduser().resolve()/'manifest.json'
    checksum=file_sha256(path)
    manifest=json.loads(path.read_text())
    handoff=manifest.get('gadget_handoff',{})
    prefix=handoff.get('initial_condition_prefix_passed_to_gadget')
    initial=handoff.get('initial_conditions')
    if not isinstance(prefix,str) or not prefix or not isinstance(initial,list) or not initial:
        raise ValueError('Host tracking requires a recorded native IC namespace in the run manifest')
    return {'path':str(path),'sha256':checksum,'prefix':prefix,**fine_dm_ic_inventory(initial)}


def fine_dm_ic_inventory(initial):
    """Stream actual fine-DM IC masses, retaining their exact source namespace.

    MUSIC's compensated mixed-fluid masses can vary within the fine level.
    The pinned GADGET/MUSIC mass unit is 1e10 Msun/h; this is an IC inventory,
    not a mass estimate inferred from a nominal resolution or halo selection.
    """
    if not isinstance(initial,list) or not initial:
        raise ValueError('Exact nonempty initial IC source bindings required')
    fine_mass=math.inf
    fine_mass_max=0.
    mixed_flags=set()
    hubble=None
    for entry in initial:
        if file_sha256(entry['path'])!=entry['sha256']:
            raise ValueError('Host-tracking initial IC changed after native handoff')
        with h5py.File(entry['path']) as source:
            mixed_flags.add(source['Header'].attrs.get('Music_mixed_fluid_schema')==1)
            h=float(source['Header'].attrs['HubbleParam'])
            if not math.isfinite(h) or h<=0 or (hubble is not None and not math.isclose(h,hubble,rel_tol=1e-12)):
                raise ValueError('Host-tracking IC shards disagree about valid Hubble parameter')
            hubble=h
            group=source.get('PartType1')
            if group is None:
                continue
            size=len(group['Coordinates'])
            dataset=group.get('Masses')
            if dataset is not None and dataset.shape!=(size,):
                raise ValueError('Host-tracking IC mass payload is incomplete')
            for first in range(0,size,100_000):
                masses=dataset[first:first+100_000] if dataset is not None else np.array([source['Header'].attrs['MassTable'][1]])
                if np.any(~np.isfinite(masses)) or np.any(masses<=0):
                    raise ValueError('Host tracking needs actual positive initial DM masses')
                fine_mass=min(fine_mass,float(np.min(masses))*1e10/h)
                fine_mass_max=max(fine_mass_max,float(np.max(masses))*1e10/h)
    if not math.isfinite(fine_mass):
        raise ValueError('Host tracking IC namespace contains no fine DM')
    for entry in initial:
        if file_sha256(entry['path'])!=entry['sha256']:
            raise ValueError('Initial IC changed during fine DM mass inventory')
    return {'initial_conditions':initial,
            'fine_dm_mass_msun':fine_mass,'fine_dm_mass_range_msun':[fine_mass,fine_mass_max],
            'mixed_fluid_fine_masses':mixed_flags=={True},'h':hubble}


def _grouped_snapshot_lineage(path, source_manifest, prefix):
    """Follow an exact completed FoF output mapping back to native ICs."""
    path=Path(path).resolve()
    report_path=Path(source_manifest).parent/'input/halo_finding.json'
    if not report_path.is_file():
        raise ValueError('Tracked snapshot does not belong to the recorded IC namespace')
    checksum=file_sha256(report_path)
    report=json.loads(report_path.read_text())
    matches=[row for row in report.get('snapshots',[])
             if any(Path(item['path']).resolve()==path and item['sha256']==file_sha256(path)
                    for item in row.get('outputs',[]))]
    if report.get('status')!='COMPLETE' or len(matches)!=1:
        raise ValueError('Grouped snapshot has no completed exact native input/output lineage')
    inputs=matches[0]['inputs_sha256']
    if not inputs or any(file_sha256(name)!=expected for name,expected in inputs.items()):
        raise ValueError('Grouped snapshot native source changed')
    for name in inputs:
        with h5py.File(name) as original:
            original_prefix=original['Parameters'].attrs.get('InitCondFile')
            if isinstance(original_prefix,bytes):original_prefix=original_prefix.decode()
            if original_prefix!=prefix:
                raise ValueError('Grouped snapshot source does not identify the parent IC')
    return [{'path':str(report_path.resolve()),'sha256':checksum},
            *[{'path':str(Path(name).resolve()),'sha256':expected} for name,expected in inputs.items()]]


def _check_lineage(catalogue,lineage):
    sources=[]
    for path in catalogue['snapshots']:
        with h5py.File(path) as source:
            prefix=source['Parameters'].attrs.get('InitCondFile')
            if isinstance(prefix,bytes):
                prefix=prefix.decode()
            if prefix!=lineage['prefix']:
                sources.extend(_grouped_snapshot_lineage(path,lineage['path'],lineage['prefix']))
            if not math.isclose(float(source['Parameters'].attrs['HubbleParam']),lineage['h'],rel_tol=1e-12):
                raise ValueError('Tracked snapshot Hubble parameter differs from its initial IC namespace')
    return sources


def recover_host_regions(run_dir, *, anchor_snapshot, anchor_host_row, snapshots,
                         aperture_r200c_factor, minimum_anchor_fraction,
                         minimum_shared_particles, maximum_runner_up_fraction,
                         memory_gib):
    """Return apertures and auditable ID-overlap evidence, never guessed rows.

    Tracers are fixed type-1 DM members of the actual anchor FOF host. The
    strongest candidate must retain the required fraction/count; runner-up
    fraction is defined relative to the same fixed anchor tracer count.
    This is material continuity, not persistent central-galaxy/core identity.
    """
    if (not isinstance(snapshots,list) or not snapshots
            or any(type(s) is not int or s<0 for s in snapshots) or len(set(snapshots))!=len(snapshots)
            or type(anchor_snapshot) is not int or anchor_snapshot<0
            or type(anchor_host_row) is not int or anchor_host_row<0
            or type(minimum_shared_particles) is not int or minimum_shared_particles<1
            or isinstance(aperture_r200c_factor,bool) or not math.isfinite(aperture_r200c_factor) or aperture_r200c_factor<=0
            or isinstance(memory_gib,bool) or not math.isfinite(memory_gib) or memory_gib<=0
            or isinstance(minimum_anchor_fraction,bool) or not math.isfinite(minimum_anchor_fraction) or not 0<minimum_anchor_fraction<=1
            or isinstance(maximum_runner_up_fraction,bool) or not math.isfinite(maximum_runner_up_fraction) or not 0<=maximum_runner_up_fraction<minimum_anchor_fraction):
        raise ValueError('Declare distinct saved epochs and explicit aperture/overlap/count/memory criteria')
    lineage=_run_lineage(run_dir)
    anchor=read_host_catalogue(run_dir,anchor_snapshot,memory_gib=memory_gib,host_row=anchor_host_row)
    anchor_lineage_sources=_check_lineage(anchor,lineage)
    count=int(anchor['lengths'][anchor_host_row,1])
    if count<minimum_shared_particles:
        raise ValueError('Anchor host has too few DM tracers for declared identity criteria')
    if count*128+anchor['group_count']*512+64*1024**2>memory_gib*1024**3:
        raise MemoryError('Anchor ID tracking exceeds declared workspace before reading IDs')
    shards,_=snapshot_id_shards(anchor['snapshots'],1)
    ids,_=read_selected_group_ids(anchor['offsets'],anchor['lengths'],np.array([anchor_host_row]),shards,
                                  particle_type=1,max_ids=count)
    ids=np.sort(ids)
    if len(np.unique(ids))!=count:
        raise ValueError('Anchor DM membership contains duplicate exact IDs')
    anchor_sources=anchor['sources']+anchor['snapshot_sources']+anchor_lineage_sources
    report={'schema':'darkmattersim-host-continuity-v1','status':'IN_PROGRESS',
            'anchor_snapshot':anchor_snapshot,'anchor_host_row':anchor_host_row,
            'anchor_sources':anchor_sources,'anchor_tracer_count':count,
            'run_lineage':lineage,
            'anchor_particle_ids_sha256':canonical_digest([int(value) for value in ids]),
            'criteria':{'minimum_anchor_fraction':minimum_anchor_fraction,
                        'minimum_shared_particles':minimum_shared_particles,
                        'maximum_runner_up_fraction':maximum_runner_up_fraction},
            'identity_scope':'same-run fixed anchor DM material continuity; no galaxy-core or cross-run identity claim',
            'records':[],'regions':[],'science_acceptance':'NOT_ACCEPTED'}
    last_a=None
    frame=anchor['frame']
    for snap in snapshots:
        cat=anchor if snap==anchor_snapshot else read_host_catalogue(run_dir,snap,memory_gib=memory_gib)
        lineage_sources=anchor_lineage_sources if snap==anchor_snapshot else _check_lineage(cat,lineage)
        a,box,h=cat['frame']
        if (last_a is not None and a<=last_a) or not np.allclose((box,h),frame[1:],rtol=1e-12,atol=0):
            raise ValueError('Tracked epochs must increase in saved time and share box/h')
        last_a=a
        if count*128+cat['group_count']*512+64*1024**2>memory_gib*1024**3:
            raise MemoryError('Host ID matching exceeds declared memory')
        source_records=cat['sources']+cat['snapshot_sources']+lineage_sources
        shards,total=snapshot_id_shards(cat['snapshots'],1)
        # Map only matched anchor tracers to exact catalogue spans. Reading all
        # snapshot IDs at once is unnecessary; offsets include ungrouped gaps.
        active=np.flatnonzero(cat['lengths'][:,1]>0)
        active=active[np.argsort(cat['offsets'][active,1])]
        starts=cat['offsets'][active,1]
        ends=starts+cat['lengths'][active,1]
        votes=np.zeros(cat['group_count'],dtype=np.int64)
        seen=np.zeros(count,dtype=bool)
        ungrouped=0
        for shard in shards:
            with h5py.File(shard['path']) as source:
                group=source.get('PartType1')
                if group is None:
                    continue
                dataset=group['ParticleIDs']
                for first in range(0,len(dataset),100_000):
                    values=np.asarray(dataset[first:first+100_000])
                    if values.dtype.kind not in 'ui' or np.any(values<0):
                        raise ValueError('Tracking requires exact nonnegative integer IDs')
                    values=values.astype(np.uint64,copy=False)
                    matches=np.searchsorted(ids,values)
                    valid=matches<count
                    valid[valid]=ids[matches[valid]]==values[valid]
                    positions=np.flatnonzero(valid)
                    indices=matches[valid]
                    if len(np.unique(indices))!=len(indices) or np.any(seen[indices]):
                        raise ValueError('Tracked anchor ID appears more than once in saved snapshot')
                    seen[indices]=True
                    locations=shard['start']+first+positions
                    candidates=np.searchsorted(starts,locations,side='right')-1
                    grouped=candidates>=0
                    grouped[grouped]=locations[grouped]<ends[candidates[grouped]]
                    np.add.at(votes,active[candidates[grouped]],1)
                    ungrouped+=int(np.count_nonzero(~grouped))
        ranked=np.argsort(-votes,kind='stable')
        winner=int(ranked[0]) if len(ranked) else None
        shared=int(votes[winner]) if winner is not None else 0
        runner=int(votes[ranked[1]]) if len(ranked)>1 else 0
        accepted=(shared>=minimum_shared_particles and shared/count>=minimum_anchor_fraction
                  and runner/count<=maximum_runner_up_fraction and shared>runner)
        row={'snapshot':snap,'scale_factor':a,'host_fof_row':winner,'shared_anchor_particles':shared,
             'anchor_fraction':shared/count,'runner_up_shared_particles':runner,'runner_up_anchor_fraction':runner/count,
             'anchor_particles_present':int(seen.sum()),'anchor_particles_ungrouped':ungrouped,
             'source_files':source_records,'status':'MATCHED' if accepted else 'UNRESOLVED_IDENTITY'}
        if accepted:
            radius=float(cat['radii'][winner])*a/h*aperture_r200c_factor
            if not math.isfinite(radius) or not 0<radius<box*a/h/2:
                raise ValueError('Tracked host needs a positive R200c aperture within half the proper box')
            region={'snapshot':snap,'center_comoving_kpc_h':(cat['positions'][winner]%box).tolist(),
                    'radius_proper_kpc':radius}
            row['region']=region
            report['regions'].append(region)
        report['records'].append(row)
        for entry in source_records:
            if file_sha256(entry['path'])!=entry['sha256']:
                raise ValueError('Host tracking source changed while reading membership')
    for entry in anchor_sources:
        if file_sha256(entry['path'])!=entry['sha256']:
            raise ValueError('Anchor host source changed during tracking')
    for entry in [lineage,*lineage['initial_conditions']]:
        if file_sha256(entry['path'])!=entry['sha256']:
            raise ValueError('Host-tracking run/IC namespace changed during matching')
    report['status']='PASS' if all(row['status']=='MATCHED' for row in report['records']) else 'FAIL'
    return report,ids


def parent_zoom_identity_query(graph_directory, *, namespace, object_id, snapshot):
    """Expose recorded Lagrangian links without converting match confidence to ancestry."""
    from halos import IdentityQueries
    return IdentityQueries(graph_directory).parent_zoom_match((namespace,object_id,snapshot))


# Zoom target coverage
#
# Static octree coverage of traced parent guards by native fine IC cells.
#
# Passing this check does not establish phase matching or later contamination.
# MUSIC coarse q is a baryon+CDM COM; undo its documented staggering before
# testing the underlying octree cell's footprint.

def _query_tree(tree, coarse_centers, cell_widths, box, guard):
    radii = guard + cell_widths / 2 - 1e-12 * box
    counts = tree.query_ball_point(coarse_centers, radii, p=np.inf, return_length=True)
    return np.asarray(counts) > 0


def coarse_intersections(points, coarse_centers, cell_widths, *, box, guard):
    points, coarse_centers, cell_widths = map(
        np.asarray, (points, coarse_centers, cell_widths)
    )
    if (
        points.ndim != 2
        or points.shape[1] != 3
        or not len(points)
        or coarse_centers.ndim != 2
        or coarse_centers.shape[1] != 3
        or cell_widths.shape != (len(coarse_centers),)
        or any(np.any(~np.isfinite(x)) for x in (points, coarse_centers, cell_widths))
        or not math.isfinite(box)
        or box <= 0
        or not math.isfinite(guard)
        or guard <= 0
        or np.any(cell_widths <= 0)
        or np.any(cell_widths >= box)
        or np.any(points < 0)
        or np.any(points >= box)
        or np.any(coarse_centers < 0)
        or np.any(coarse_centers >= box)
    ):
        raise ValueError(
            "Coverage needs finite periodic q and positive cell widths/guard"
        )
    tree = cKDTree(points, boxsize=box)
    # Axis-aligned cubes overlap if all three periodic coordinate distances
    # are smaller than the sum of their half-widths. p=infinity gives exactly
    # that test, without an enclosing-sphere approximation.
    return _query_tree(tree, coarse_centers, cell_widths, box, guard)


@directory_owner('destination')
def audit_target_coverage(
    generation_report, traced_region, destination, *, memory_gib=1.0
):
    generation_report, traced_region, destination = map(
        lambda p: Path(p).expanduser().resolve(),
        (generation_report, traced_region, destination),
    )
    if destination.exists():
        raise FileExistsError("Refusing to overwrite static target coverage")
    if not math.isfinite(memory_gib) or memory_gib <= 0:
        raise ValueError("Coverage memory budget must be finite and positive")
    source_hashes={path:digest(path) for path in (generation_report,traced_region)}
    from zoom import verify_generated_zoom
    material_generation=json.loads(generation_report.read_text()).get('schema')=='twofluid-native-ic-generation-v1'
    if material_generation:
        from initial_conditions import verify_twofluid_native_generation
        generation,projection,arrays=verify_twofluid_native_generation(generation_report,memory_gib=memory_gib)
        del arrays
        base_reference=generation['base_native_generation'];base_path=Path(base_reference['path']).resolve()
        if file_sha256(base_path)!=base_reference['sha256']:raise ValueError('Material target coverage original generation changed')
        basis=verify_generated_zoom(base_path);source_hashes[base_path]=base_reference['sha256']
        ic=Path(generation['native_ic']['path']).resolve();ic_sha=generation['native_ic']['sha256']
        completed=generation['execution_status']=='COMPLETE' and projection.get('projection_mode')=='finite_volume_material_cells'
    else:
        generation=verify_generated_zoom(generation_report);basis=generation
        ic=generation_report.parent/'ics_gadget.hdf5';ic_sha=generation['artifacts']['ics_gadget.hdf5'];completed=generation.get('status')=='GENERATED_STRUCTURAL_PASS'
    region = verify_traced_region(traced_region)
    if (
        not completed
        or basis.get("traced_region", {}).get("sha256")
        != file_sha256(traced_region)
        or region.get("status") != "TRACED_PROPOSAL"
    ):
        raise ValueError(
            "Coverage requires structurally audited generation of this exact traced region"
        )
    if ic_sha != file_sha256(ic):
        raise ValueError("Generated IC changed after its structural audit")
    particles = traced_region.parent / "selected_particles.hdf5"
    if file_sha256(particles) != region["selected_particles_sha256"]:
        raise ValueError("Traced parent membership changed")
    count = sum(int(n) for n in region["selected_counts"].values())
    if count * 256 + 64 * 1024**2 > memory_gib * 1024**3:
        raise MemoryError("Static coverage exceeds the declared q/tree memory budget")
    with h5py.File(particles) as source:
        points = np.concatenate(
            [source[f'PartType{ptype}']["LagrangianCoordinates"][:]
             for ptype, number in region['selected_counts'].items() if int(number) > 0]
        )
    if len(points) != count:
        raise ValueError("Traced q count differs from its recorded membership")
    guard = float(region["guard"]["parent_cell_comoving_kpc_h"])
    if not math.isfinite(guard) or guard <= 0:
        raise ValueError("Coverage requires a positive declared parent cell guard")
    touched_ids = []
    with h5py.File(ic) as source:
        header = source["Header"].attrs
        box = float(header["BoxSize"])
        if header.get("Music_lagrangian_schema") != 1 or not math.isclose(
            float(header.get("Music_lagrangian_unitlength_cm", math.nan)),
            KPC_CM,
            rel_tol=1e-12,
        ):
            raise ValueError("Static coverage needs native MUSIC Lagrangian schema 1")
        if not math.isclose(box, region["box_comoving_kpc_h"], rel_tol=1e-12):
            raise ValueError("Traced and generated periodic boxes differ")
        tree = cKDTree(points, boxsize=box)
        high = int(header["Music_levelmax"])
        low = int(header["Music_levelmin"])
        if header.get('Music_mixed_fluid_schema') == 1:
            means = np.asarray(header.get('Music_mean_species_masses_native',[]),dtype=float)
            if means.shape != (6,) or not np.isfinite(means).all() or np.any(means[:2] <= 0):
                raise ValueError('Mixed-fluid coverage requires finite positive mean species masses')
            fine_mass = float(means[0]+means[1])
        else:
            fine_mass = float(header["MassTable"][0] + header["MassTable"][1])
        material_cells=material_generation and header.get('Music_material_cell_schema')==1
        if (not material_cells and not fine_mass > 0) or not 1 <= low < high <= 20:
            raise ValueError(
                "Coverage requires finite octree mass scales and nested levels"
            )
        fb = float(header["OmegaBaryon"]) / float(header["Omega0"])
        if not 0 <= fb < 1:
            raise ValueError("Invalid baryon fraction in generated octree")
        shift = fb * box / 2 ** (high + 1)
        group = source["PartType2"]
        q = group["LagrangianCoordinates"]
        if count * 256 + len(q) * 128 + 64 * 1024**2 > memory_gib * 1024**3:
            raise MemoryError(
                "Static coverage exceeds the coarse-ID/report memory budget"
            )
        if q.dtype != np.float64 or q.shape != (len(group["ParticleIDs"]), 3):
            raise ValueError("Coarse q must be double and align with exact IDs")
        for first in range(0, len(q), 100_000):
            end = min(first + 100_000, len(q))
            mass = (
                np.asarray(group["Masses"][first:end])
                if "Masses" in group
                else np.full(end - first, header["MassTable"][2])
            )
            if np.any(~np.isfinite(mass)) or np.any(mass <= 0):
                raise ValueError("Coarse cell masses must be finite and positive")
            if material_cells:
                widths=np.asarray(group['LagrangianCellWidths'][first:end],float)
                if widths.shape!=(end-first,) or not np.isfinite(widths).all() or np.any(widths<=0):raise ValueError('Invalid recorded material coarse cell widths')
                levels=np.rint(np.log2(box/widths)).astype(int)
                if not np.allclose(widths,box/np.exp2(levels),rtol=1e-12,atol=0):raise ValueError('Recorded material coarse widths are not dyadic')
            else:
                differences = np.log(mass / fine_mass) / np.log(8)
                if np.any(np.abs(differences - np.rint(differences)) > 1e-10):
                    raise ValueError("Coarse cell mass is not an octree level")
                levels = high - np.rint(differences).astype(int)
            if np.any(levels < low) or np.any(levels >= high):
                raise ValueError(
                    "Coarse cell level lies outside the declared hierarchy"
                )
            native_q = np.asarray(q[first:end])
            if (
                np.any(~np.isfinite(native_q))
                or np.any(native_q < 0)
                or np.any(native_q >= box)
            ):
                raise ValueError(
                    "Coarse native q must be finite and periodically wrapped"
                )
            centers = np.asarray(group['LagrangianCellCenters'][first:end],float) if material_cells else (native_q-shift)%box
            if centers.shape!=(end-first,3) or not np.isfinite(centers).all() or np.any(centers<0) or np.any(centers>=box):raise ValueError('Invalid recorded common material cell centers')
            widths = box / 2.0**levels
            intersects = _query_tree(tree, centers, widths, box, guard)
            ids = np.asarray(group["ParticleIDs"][first:end])
            if ids.dtype.kind not in "ui" or np.any(ids < 0):
                raise ValueError("Coarse IDs must remain exact nonnegative integers")
            touched_ids.extend(int(n) for n in ids[intersects])
    report = {
        "schema": "darkmattersim-static-target-coverage-v1",
        "status": "PASS" if not touched_ids else "FAIL",
        "generation_report_sha256": file_sha256(generation_report),
        "traced_region_sha256": file_sha256(traced_region),
        "ic_sha256": ic_sha,
        "selected_parent_count": count,
        "conservative_parent_guard_comoving_kpc_h": guard,
        "coarse_cells_intersecting_parent_guards": len(touched_ids),
        "coarse_particle_ids": touched_ids,
        "coarse_COM_shift_removed_comoving_kpc_h": 0. if material_cells else shift,
        "cell_geometry_source": 'recorded original native material cell centers/widths' if material_cells else 'original coarse octree mass/q geometry',
        "scope": "Static original-grid cell coverage only; phase matching and nonlinear contamination remain unvalidated",
        "science_acceptance": "NOT_ACCEPTED",
    }
    for path,before in source_hashes.items():
        if digest(path)!=before:raise ValueError('Coverage report source changed during measurement')
    if digest(ic)!=report['ic_sha256'] or digest(particles)!=region['selected_particles_sha256']:
        raise ValueError('Coverage numeric source changed during measurement')
    from core.data import export_model_arrays
    report['numeric_ids']=export_model_arrays(destination,{'coarse_particle_ids':np.asarray(touched_ids,dtype=np.uint64)},
        {'coarse_particle_ids':{'units':'identity','quantity':'Generated coarse cells intersecting traced parent guards',
            'identity_namespace':'generated-ic:'+report['ic_sha256']+':native_particle_union',
            'ancestry':{'generated_ic_sha256':report['ic_sha256'],'parent_traced_region_sha256':report['traced_region_sha256'],
                'relationship':'Geometric guard overlap; no identity correspondence with parent particles implied'}}},
        algorithm=__file__,context={'generation_report_sha256':report['generation_report_sha256'],
            'traced_region_sha256':report['traced_region_sha256'],'ic_sha256':report['ic_sha256']})
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Keep failures as reviewable evidence instead of discarding their IDs.
    stage=Path(tempfile.mkdtemp(prefix='.coverage-',dir=destination.parent))
    temporary=stage/destination.name
    atomic_json(temporary,report)
    publish_file(temporary,destination)
    stage.rmdir()
    return report

# ========================================================================
# zoom_initial_conditions: Native IC structure, density/velocity modes and Lagrangian windows.
from core.foundation import G_MPC_KMS2_MSUN, MSUN_G
from core.foundation import atomic_text
from core.data import external_hdf5_contract
from core.data import export_numeric_audit, export_model_arrays
from scipy.interpolate import CubicSpline
from core.data import verify_numeric_table
from figures import export_figure
import configparser
import itertools
import re


# Zoom initial conditions
#
# Audit native MUSIC2 zoom ICs without inventing linear-transfer provenance.
#
# The exporter must reserve type 2 for coarse combined matter, leaving type 5 for
# BH sinks. Mass/structure checks do not validate phases, linear spectra or a
# chosen halo's Lagrangian coverage; those are separately required science gates.

def audit_music_ic(filename, configuration, *, memory_gib=1.0, input_kind='zoom'):
    filename, configuration = Path(filename), Path(configuration)
    before={str(p.resolve()):file_sha256(p) for p in (filename,configuration)}
    cfg = configparser.ConfigParser()
    cfg.read(configuration)
    if input_kind not in {'zoom','parent'}:
        raise ValueError('MUSIC structural audit requires explicit zoom or parent input kind')
    if not math.isfinite(memory_gib) or memory_gib <= 0:
        raise ValueError("Audit memory budget must be finite and positive")
    for key, expected in [
        ("arepo_unitlength", KPC_CM),
        ("arepo_unitmass", 1e10 * MSUN_G),
        ("arepo_unitvel", 1e5),
    ]:
        if not math.isclose(
            cfg.getfloat("output", key, fallback=expected), expected, rel_tol=1e-12
        ):
            raise ValueError("Zoom adapter requires pinned units: " + key)
    if (
        cfg.get("output", "format") != "arepo"
        or cfg.getint("output", "arepo_coarsetype") != 2
    ):
        raise ValueError(
            "Zoom adapter requires HDF5 arepo output and coarse type 2, reserving type 5 for BHs"
        )
    if not cfg.getboolean("output", "arepo_doubleprec") or not cfg.getboolean(
        "output", "arepo_longids"
    ):
        raise ValueError("Zoom ICs require double coordinates and exact 64-bit IDs")
    if cfg.getint("output", "arepo_num_files") != 1:
        raise ValueError("This audit currently requires a single complete IC file")
    if any(cfg.getfloat("cosmology", key) != 0 for key in ("m_nu1", "m_nu2", "m_nu3")):
        raise ValueError("This adapter does not yet evolve massive-neutrino particles")
    high, low = cfg.getint("setup", "levelmax"), cfg.getint("setup", "levelmin")
    if (not 1 <= low <= high <= 20 or (input_kind == 'zoom' and low == high)
            or (input_kind == 'parent' and low != high)):
        raise ValueError(
            "Zoom requires bounded, distinct coarse/fine refinement levels"
        )
    h = cfg.getfloat("cosmology", "H0") / 100
    omega = cfg.getfloat("cosmology", "Omega_m")
    baryons = cfg.getboolean("setup", "baryons")
    omega_b = cfg.getfloat("cosmology", "Omega_b") if baryons else 0
    box = cfg.getfloat("setup", "boxlength")
    if (
        not all(math.isfinite(x) and x > 0 for x in (h, omega, box))
        or not 0 <= omega_b < omega
    ):
        raise ValueError("Invalid zoom cosmology or box")
    total_mass = 0.0
    species = {}
    all_ids = []
    with h5py.File(filename) as source:
        header = source["Header"].attrs
        declared_order = 2 if cfg.getboolean('setup','use_2LPT',fallback=False) else 1
        if header.get('Music_lpt_order',1) != declared_order:
            raise ValueError('Delivered native perturbation order differs from its configuration')
        mixed = declared_order == 2
        if mixed:
            for option,attribute in (('lpt2_displacement','Music_2lpt_displacement_coefficient'),
                                      ('lpt2_velocity','Music_2lpt_velocity_coefficient')):
                expected=cfg.getfloat('cosmology',option)
                actual=float(header.get(attribute,math.nan))
                if not math.isfinite(expected) or expected <= 0 or not math.isclose(actual,expected,rel_tol=1e-12):
                    raise ValueError('Delivered higher-order kernel differs from background recipe')
        mean_mass = np.asarray(header.get('Music_mean_species_masses_native',np.zeros(6)),dtype=float)
        if mixed and (header.get('Music_mixed_fluid_schema') != 1 or mean_mass.shape != (6,)
                or not np.isfinite(mean_mass).all() or np.any(mean_mass[:2] <= 0)
                or not cfg.getboolean('cosmology','mixed_2lpt',fallback=False)):
            raise ValueError('Second-order native IC requires retained mixed-fluid masses and kernel metadata')
        counts = snapshot_total_counts(header)
        # Includes concatenated IDs, sort/unique workspace and one chunk of
        # phase-space data; reject before any particle allocation.
        if 48 * int(sum(map(int, counts))) + 64 * 1024**2 > memory_gib * 1024**3:
            raise MemoryError("Zoom audit exceeds declared ID/sort memory budget")
        if int(header["NumFilesPerSnapshot"]) != 1 or not np.array_equal(
            counts, header["NumPart_ThisFile"]
        ):
            raise ValueError("IC shards/counts incomplete")
        for name, value in [
            ("BoxSize", 1000 * box),
            ("Time", 1 / (1 + cfg.getfloat("setup", "zstart"))),
            ("HubbleParam", h),
            ("Omega0", omega),
        ]:
            if not math.isclose(float(header[name]), value, rel_tol=1e-10):
                raise ValueError("IC/config mismatch: " + name)
        if (
            any(counts[t] for t in (3, 4, 5))
            or not counts[1]
            or (not counts[2] if input_kind == 'zoom' else bool(counts[2]))
            or bool(counts[0]) != baryons
        ):
            raise ValueError(
                "Zoom requires fine DM, coarse type 2 and only actual requested gas"
            )
        for ptype in (0, 1, 2):
            count = int(counts[ptype])
            if count == 0:
                continue
            group = source[f"PartType{ptype}"]
            if input_kind == 'parent':
                if (counts[1] != 2**(3*low) or (baryons and counts[0] != counts[1])
                        or header.get('Music_lagrangian_schema') != 1
                        or not math.isclose(float(header.get('Music_lagrangian_unitlength_cm',math.nan)),KPC_CM,rel_tol=1e-12)):
                    raise ValueError('Uniform parent requires full species counts and native Lagrangian schema 1')
                q = group.get('LagrangianCoordinates')
                if q is None or q.shape != (count,3) or q.dtype != np.float64:
                    raise ValueError('Uniform parent q must be double and align with exact species IDs')
                # A finite array with the right length can still duplicate a
                # cell or invent non-lattice coordinates. Prove one native q
                # per periodic parent cell, allowing each species its offset.
                cells_per_axis = 2**low
                spacing = 1000*box/cells_per_axis
                origin = q[0]/spacing
                occupied = np.zeros(count, dtype=bool)
                for first in range(0,count,100_000):
                    values=q[first:first+100_000]
                    if not np.isfinite(values).all() or np.any(values<0) or np.any(values>=1000*box):
                        raise ValueError('Native parent q must be finite and wrapped into the box')
                    lattice = values/spacing-origin
                    rounded = np.rint(lattice)
                    if np.any(np.abs(lattice-rounded) > 1e-9):
                        raise ValueError('Native parent q does not form a uniform periodic lattice')
                    indices = rounded.astype(np.int64) % cells_per_axis
                    linear = (indices[:,0]*cells_per_axis+indices[:,1])*cells_per_axis+indices[:,2]
                    if len(np.unique(linear)) != len(linear) or np.any(occupied[linear]):
                        raise ValueError('Duplicate native parent Lagrangian lattice cell')
                    occupied[linear] = True
                del occupied
            ids = np.asarray(group["ParticleIDs"])
            if (
                ids.dtype.kind not in "iu"
                or ids.dtype.itemsize != 8
                or ids.shape != (count,)
                or np.any(ids < 0)
            ):
                raise ValueError(
                    "Zoom IDs must be exact nonnegative 64-bit integer vectors"
                )
            all_ids.append(ids.astype(np.uint64, copy=False))
            for key in (("Coordinates", "Velocities", "FirstOrderDisplacements", "FirstOrderVelocities")
                        if mixed else ("Coordinates", "Velocities")):
                data = group[key]
                if data.shape != (count, 3) or data.dtype.kind != 'f' or data.dtype.itemsize != 8:
                    raise ValueError("Invalid double phase-space field: " + key)
                for first in range(0, count, 1_000_000):
                    if not np.all(np.isfinite(data[first : first + 1_000_000])):
                        raise ValueError("Nonfinite zoom phase-space: " + key)
            mass = (
                np.asarray(group["Masses"])
                if "Masses" in group
                else np.full(count, header["MassTable"][ptype])
            )
            if (
                mass.shape != (count,)
                or not np.all(np.isfinite(mass))
                or np.any(mass <= 0)
            ):
                raise ValueError("Invalid zoom masses")
            total_mass += float(mass.sum())
            species[str(ptype)] = {
                "count": count,
                "mass_levels_native": ([float(mean_mass[ptype])] if mixed and ptype in (0,1) else np.unique(mass).tolist()),
                "actual_mass_range_native": [float(mass.min()),float(mass.max())],
                "actual_total_mass_native": float(mass.sum()),
            }
        ids = np.concatenate(all_ids)
        if len(np.unique(ids)) != len(ids):
            raise ValueError("Duplicate IDs across refinement/species")
        # Low-resolution particles carry combined baryon+DM mass. Fine gas is
        # emitted only within the zoom. This matters for global mass closure.
        fine_mass = float(mean_mass[0]+mean_mass[1]) if mixed else float(header["MassTable"][1]) + float(header["MassTable"][0])
        if input_kind == 'zoom':
            coarse_mass = np.asarray(species["2"]["mass_levels_native"])
            ratios = np.log(coarse_mass / fine_mass) / np.log(8)
            if np.any(np.abs(ratios - np.rint(ratios)) > 1e-10) or np.any(ratios < 1):
                raise ValueError("Coarse mass levels do not match nested octree refinement")
        else:
            density = 3*100**2/(8*np.pi*G_MPC_KMS2_MSUN)
            for t, fraction in ((0,omega_b),(1,omega-omega_b)):
                if t == 0 and not baryons:
                    continue
                levels=species[str(t)]['mass_levels_native']
                expected=fraction*density*box**3/(2**(3*low)*1e10)
                if mixed and not math.isclose(species[str(t)]['actual_total_mass_native'],expected*species[str(t)]['count'],rel_tol=1e-5):
                    raise ValueError('Compensated parent species mass fails cosmological closure')
                if len(levels)!=1 or not math.isclose(levels[0],expected,rel_tol=1e-5):
                    raise ValueError('Uniform parent species mass does not match declared cosmology')
    expected_mass = omega * 3 * 100**2 / (8 * np.pi * G_MPC_KMS2_MSUN) * box**3 / 1e10
    residual = abs(total_mass / expected_mass - 1)
    if residual > 1e-5:
        raise ValueError("Zoom global mass closure failed")
    specifications={}
    namespace='native_music_ic:'+before[str(filename.resolve())]
    units={'Coordinates':'comoving kpc/h','LagrangianCoordinates':'comoving kpc/h',
           'LagrangianCellCenters':'comoving kpc/h','LagrangianCellWidths':'comoving kpc/h',
           'Velocities':'peculiar km/s divided by sqrt(a)',
           'ThermalVelocities':'peculiar km/s divided by sqrt(a)',
           'FirstOrderVelocities':'peculiar km/s divided by sqrt(a)',
           'FirstOrderDisplacements':'comoving kpc/h','Masses':'1e10 Msun/h','ParticleIDs':'identity'}
    with h5py.File(filename) as source:
        def describe(name,item):
            if not isinstance(item,h5py.Dataset):return
            leaf=name.rsplit('/',1)[-1]
            specification={'units':units.get(leaf,'UNDECLARED native field'),
                'quantity':{'Coordinates':'displaced periodic particle position',
                    'LagrangianCoordinates':'native original q; coarse type2 is baryon+CDM centre of mass',
                    'LagrangianCellCenters':'original common dyadic material cell center before species staggering and displacement',
                    'LagrangianCellWidths':'original common dyadic material cell side length; not inferred from compensated mass',
                    'Velocities':'native IC peculiar velocity scaled by inverse sqrt(a)',
                    'ThermalVelocities':'ID-keyed physical thermal relic component scaled by inverse sqrt(a)',
                    'FirstOrderDisplacements':'common growing-mode first-order displacement at native q',
                    'FirstOrderVelocities':'exact first-order species bulk velocity before global momentum closure',
                    'Masses':'particle mass in pinned native units','ParticleIDs':'native generated particle identity'}.get(leaf)}
            if leaf=='ParticleIDs':
                specification.update(identity_namespace=namespace,
                    ancestry={'relation':'native generated IC identity; no parent ID equivalence',
                              'species':name.split('/')[0]})
            specifications[name]=specification
        source.visititems(describe)
        actual_a=float(source['Header'].attrs['Time'])
    contract=external_hdf5_contract(filename,specifications,epoch=epoch_contract(actual_a),
        coordinates=coordinate_contract(position_units='comoving kpc/h',origin='restored original periodic box',
            periodic_wrapping='native displaced and original q wrapped to box',velocity_convention='peculiar km/s divided by sqrt(a)'),
        provenance={'input_hashes':before,'algorithm_sha256':file_sha256(__file__),
                    'native_exporter_sha256':file_sha256(Path(__file__).parent/'software/music2/src/plugins/output_arepo.cc'),
                    'build_equivalence':'UNDECLARED','science_acceptance':'NOT_ACCEPTED'})
    if any(file_sha256(path)!=expected for path,expected in before.items()):
        raise RuntimeError('Native IC or configuration changed during structural audit')
    return {
        "native_dataset_contract":contract,
        "science_acceptance":"NOT_ACCEPTED",
        "input_artifacts":{
            "initial_ic":{"path":str(filename.resolve()),"sha256":before[str(filename.resolve())]},
            "configuration":{"path":str(configuration.resolve()),"sha256":before[str(configuration.resolve())]}},
        "schema_version": 1,
        "status": "STRUCTURAL_PASS",
        'input_kind':input_kind,
        "ic_sha256": file_sha256(filename),
        "config_sha256": file_sha256(configuration),
        "species": species,
        "mass_closure_relative_error": residual,
        "units": {
            "coordinates": "comoving kpc/h",
            "mass": "1e10 Msun/h",
            "velocities": "peculiar km/s divided by sqrt(a)",
        },
        "coarse_particle_type": 2,
        "black_hole_particle_type": 5,
        "linear_transfers": {
            "status": "UNVALIDATED",
            "plugin": cfg.get("cosmology", "transfer"),
        },
        "lagrangian_target_coverage": {"status": "UNVALIDATED"},
        "science_readiness": "Requires matched transfer/phase checks, target tracing, contamination and convergence",
    }


@directory_owner('destination')
def write_audit(filename, configuration, destination, *, memory_gib=1., input_kind='zoom'):
    """Publish immutable native structural evidence with a completion seal."""
    target=Path(destination).resolve()
    if target.exists():raise FileExistsError('Keep previous native IC audit evidence')
    target.parent.mkdir(parents=True,exist_ok=True)
    stage=Path(tempfile.mkdtemp(prefix='.'+target.name+'-stage-',dir=target.parent))
    try:
        report=audit_music_ic(filename,configuration,memory_gib=memory_gib,input_kind=input_kind)
        report['publication_contract']='native-ic-structural-audit-v1'
        atomic_json(stage/'audit.json',report)
        atomic_text(stage/'audit.sha256',digest(stage/'audit.json')+'\n')
        publish_file(stage/'audit.json',target)
        publish_file(stage/'audit.sha256',target.with_name(target.name+'.sha256'))
        return report
    except BaseException as exc:
        atomic_json(stage/'failure.json',{'status':'FAIL','science_acceptance':'NOT_ACCEPTED',
            'type':type(exc).__name__,'message':str(exc)})
        raise


@directory_owner('path')
def verify_music_ic_audit(path):
    """Recheck native structural evidence and original numeric payload contract."""
    from core.data import verify_external_hdf5_contract
    path=Path(path).resolve();before=digest(path)
    if path.with_name(path.name+'.sha256').read_text().strip()!=before:
        raise ValueError('Native IC audit seal mismatch')
    report=json.loads(path.read_text())
    if report.get('publication_contract')!='native-ic-structural-audit-v1' or report.get('status')!='STRUCTURAL_PASS':
        raise ValueError('Native IC structural audit is incomplete or unsupported')
    for name,entry in report['input_artifacts'].items():
        if digest(entry['path'])!=entry['sha256']:raise ValueError('Native IC audit input changed: '+name)
    verify_external_hdf5_contract(report['input_artifacts']['initial_ic']['path'],report['native_dataset_contract'])
    if digest(path)!=before:raise RuntimeError('Native IC audit report changed during reading')
    return report


# Zoom phases
#
# Measure parent/zoom large-scale linear density phases from delivered ICs.
#
# Uses delta=-div(Psi) at native original q, mass-weighting gas, DM and combined
# coarse cells. This tests the realization, not the physical transfer spectrum,
# nonlinear target recovery or the quadratic LPT component. Two-fluid 1LPT and
# retained linear components of mixed 2LPT are supported; the native displacement
# bound prevents winding ambiguity.

def linear_density_modes(filename, modes, *, memory_gib=1.0, particle_types=(0,1,2), refinement_levels=None):
    """Linear displacement modes; a species subset is a Lagrangian window."""
    return _matter_modes(filename, modes, memory_gib=memory_gib, velocity=False, particle_types=particle_types, refinement_levels=refinement_levels)


def velocity_divergence_modes(filename, modes, *, memory_gib=1.0, particle_types=(0,1,2), refinement_levels=None):
    """Return div(v_bulk), subtracting declared thermal vectors, with comoving k in (kpc/h)^-1 and v in km/s."""
    return _matter_modes(filename, modes, memory_gib=memory_gib, velocity=True, particle_types=particle_types, refinement_levels=refinement_levels)


def _native_cell_levels(widths, box, low, high):
    """Decode exact dyadic native cell widths without guessing from mass."""
    widths=np.asarray(widths)
    if (widths.ndim!=1 or widths.dtype.kind not in 'fiu' or not np.isfinite(widths).all()
            or np.any(widths<=0) or type(low) is not int or type(high) is not int
            or not 1<=low<=high<=30 or not math.isfinite(box) or box<=0):
        raise ValueError('Declared native dyadic hierarchy and positive cell widths required')
    levels=np.rint(np.log2(box)-np.log2(widths))
    if (np.any(levels<low) or np.any(levels>high)
            or not np.allclose(widths,np.ldexp(box,-levels.astype(int)),rtol=1e-12,atol=0)):
        raise ValueError('Native cell widths disagree with declared dyadic hierarchy')
    return levels.astype(np.int16)


def _native_hierarchy_levels(header):
    low,high=header.get('Music_levelmin'),header.get('Music_levelmax')
    if (any(isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,np.integer)) for value in (low,high))
            or not 1<=low<=high<=30):
        raise ValueError('Species-window measurement requires exact native hierarchy levels')
    return int(low),int(high)


def measure_native_species_window_modes(filename, modes, destination, *, memory_gib=1.0):
    """Publish delivered linear modes per actual species and its Lagrangian window.

    This is a measurement, not a periodic-power acceptance test. Mixed 2LPT
    retains the actual first-order components and compensated mass term through
    the existing estimator. Quadratic/handoff acceptance remains separate.
    """
    from core.data import export_numeric_audit,epoch_contract
    if isinstance(memory_gib,(bool,np.bool_)) or not isinstance(memory_gib,(int,float,np.integer,np.floating)) or not np.isfinite(memory_gib) or memory_gib<=0:
        raise ValueError('Positive species-window measurement memory required')
    waves=_modes(modes)
    if np.any(np.abs(waves)>2**53-1):
        raise ValueError('Species-window modes must be exactly representable integers')
    if 256*len(waves)>memory_gib*1024**3/4:
        raise MemoryError('Species-window mode outputs exceed declared memory')
    path=Path(filename).resolve();before=file_sha256(path)
    with h5py.File(path) as native:
        header=native['Header'].attrs;counts=snapshot_total_counts(header)
        species=[t for t in (0,1,2) if counts[t]>0]
        a=float(header['Time']);box=float(header['BoxSize'])
        order=int(header.get('Music_lpt_order',0))
        if not species:raise ValueError('Native window measurement requires actual matter populations')
        masses=[];windows={}
        low,high=_native_hierarchy_levels(header)
        for t in species:
            group=native['PartType'+str(t)];total=0.
            if 'LagrangianCellWidths' not in group or group['LagrangianCellWidths'].shape!=(int(counts[t]),):
                raise ValueError('Species-window measurement requires aligned native cell widths')
            mass_chunk=min(100000,max(1,int(memory_gib*1024**3/64)))
            for first in range(0,int(counts[t]),mass_chunk):
                stop=min(first+mass_chunk,int(counts[t]))
                values=np.asarray(group['Masses'][first:stop]) if 'Masses' in group else np.full(stop-first,float(header['MassTable'][t]))
                if not np.isfinite(values).all() or np.any(values<=0):raise ValueError('Positive actual species-window masses required')
                total+=float(values.sum())
                levels=_native_cell_levels(group['LagrangianCellWidths'][first:stop],box,low,high)
                for level in np.unique(levels):
                    key=(t,int(level));count,mass=windows.get(key,(0,0.));keep=levels==level
                    windows[key]=(count+int(keep.sum()),mass+float(values[keep].sum()))
            if not math.isfinite(total) or total<=0:raise ValueError('Species-window normalization overflowed')
            masses.append(total)
    density=np.asarray([linear_density_modes(path,waves,memory_gib=memory_gib/2,particle_types=(t,)) for t in species])
    velocity=np.asarray([velocity_divergence_modes(path,waves,memory_gib=memory_gib/2,particle_types=(t,)) for t in species])
    keys=sorted(windows)
    if 128*len(waves)*len(keys)>memory_gib*1024**3/4:
        raise MemoryError('Species/hierarchy window mode outputs exceed declared memory')
    window_density=np.asarray([linear_density_modes(path,waves,memory_gib=memory_gib/2,particle_types=(t,),refinement_levels=(level,)) for t,level in keys])
    window_velocity=np.asarray([velocity_divergence_modes(path,waves,memory_gib=memory_gib/2,particle_types=(t,),refinement_levels=(level,)) for t,level in keys])
    if file_sha256(path)!=before:raise ValueError('Native species-window IC changed during measurement')
    arrays={'particle_types':np.asarray(species,dtype=np.int16),'integer_wavevectors':waves.astype(np.int64),
        'actual_particle_counts':np.asarray([counts[t] for t in species],dtype=np.uint64),
        'species_total_mass_native':np.asarray(masses),
        'linear_density_mode_real':density.real,'linear_density_mode_imaginary':density.imag,
        'velocity_divergence_mode_real':velocity.real,'velocity_divergence_mode_imaginary':velocity.imag,
        'window_particle_types':np.asarray([key[0] for key in keys],dtype=np.int16),
        'window_refinement_levels':np.asarray([key[1] for key in keys],dtype=np.int16),
        'window_particle_counts':np.asarray([windows[key][0] for key in keys],dtype=np.uint64),
        'window_total_mass_native':np.asarray([windows[key][1] for key in keys]),
        'window_linear_density_mode_real':window_density.real,'window_linear_density_mode_imaginary':window_density.imag,
        'window_velocity_divergence_mode_real':window_velocity.real,'window_velocity_divergence_mode_imaginary':window_velocity.imag}
    units={'particle_types':'native species enum','integer_wavevectors':'dimensionless integer mode',
        'actual_particle_counts':'particle count','species_total_mass_native':'1e10 Msun/h',
        'linear_density_mode_real':'dimensionless','linear_density_mode_imaginary':'dimensionless',
        'velocity_divergence_mode_real':'km/s per comoving kpc/h','velocity_divergence_mode_imaginary':'km/s per comoving kpc/h'}
    units.update(window_particle_types='native species enum',window_refinement_levels='dyadic refinement level',
        window_particle_counts='particle count',window_total_mass_native='1e10 Msun/h',
        window_linear_density_mode_real='dimensionless',window_linear_density_mode_imaginary='dimensionless',
        window_velocity_divergence_mode_real='km/s per comoving kpc/h',window_velocity_divergence_mode_imaginary='km/s per comoving kpc/h')
    report={'schema':'native-music-species-window-modes-v2','execution_status':'COMPLETE',
        'actual_scale_factor':a,'box_comoving_kpc_h':box,'native_lpt_order':order,
        'native_levelmin':low,'native_levelmax':high,
        'window_definition':'Each actual native species and each separate species/refinement-level original-q population with actual mass normalization; levels derive from native cell widths, not masses. Mixed 2LPT uses retained first-order flow and compensated density.',
        'density_quantity':'-i k dot Psi plus mixed-fluid compensated mass mode',
        'velocity_quantity':'+i k dot physical peculiar first-order/bulk velocity',
        'linear_transfer_acceptance':'NOT_ESTABLISHED','quadratic_acceptance':'NOT_ESTABLISHED',
        'science_acceptance':'NOT_ACCEPTED','scope':'Delivered species/window measurement only; reference/operator/tail/error-budget comparison is mandatory before launch acceptance'}
    sources={str(path):before,str(Path(__file__).resolve()):file_sha256(__file__)}
    return export_numeric_audit(destination,report,arrays,{key:{'units':units[key],'quantity':key} for key in arrays},
        algorithm=__file__,source_hashes=sources,context={'epoch':epoch_contract(a)})


def _matter_modes(filename, modes, *, memory_gib, velocity, particle_types=(0,1,2), refinement_levels=None):
    if (not isinstance(particle_types,tuple) or not particle_types
            or any(type(t) is not int or t not in (0,1,2) for t in particle_types)
            or len(set(particle_types)) != len(particle_types)):
        raise ValueError('Declare distinct native matter particle types')
    if refinement_levels is not None and (not isinstance(refinement_levels,tuple) or not refinement_levels
            or any(type(level) is not int or not 1<=level<=30 for level in refinement_levels)
            or len(set(refinement_levels))!=len(refinement_levels)):
        raise ValueError('Declare distinct supported native refinement levels')
    modes = np.asarray(modes, dtype=float)
    if (
        modes.ndim != 2
        or modes.shape[1] != 3
        or not len(modes)
        or np.any(~np.isfinite(modes))
        or np.any(modes != np.rint(modes))
        or np.any(np.all(modes == 0, axis=1))
    ):
        raise ValueError("Density phases require nonzero integer wavevectors")
    budget = memory_gib * 1024**3 - 64 * 1024**2
    if not math.isfinite(budget) or budget < 80 * len(modes) + 200:
        raise MemoryError("Phase measurement exceeds its declared workspace budget")
    chunk = min(10_000, int(budget // (80 * len(modes) + 200)))
    result, total = np.zeros(len(modes), complex), 0.0
    with h5py.File(filename) as source:
        header = source["Header"].attrs
        hierarchy=_native_hierarchy_levels(header) if refinement_levels is not None else None
        box = float(header["BoxSize"])
        a = float(header["Time"])
        if not math.isfinite(a) or a <= 0:
            raise ValueError("Native mode measurement needs a positive scale factor")
        if not math.isfinite(box) or box <= 0 or header.get("Music_lpt_order") not in (1,2):
            raise ValueError(
                "Density phase measurement requires a positive box and retained native LPT components"
            )
        mixed = header.get('Music_lpt_order') == 2
        if mixed and header.get('Music_mixed_fluid_schema') != 1:
            raise ValueError('Higher-order modes require the mixed-fluid native component schema')
        mean_mass = np.asarray(header.get('Music_mean_species_masses_native',np.zeros(6)),dtype=float)
        if mixed and (mean_mass.shape != (6,) or not np.isfinite(mean_mass).all() or np.any(mean_mass[:2] <= 0)):
            raise ValueError('Missing mixed-fluid reference masses')
        totals = snapshot_total_counts(header)
        if header.get("NumFilesPerSnapshot") != 1 or np.any(totals[3:]):
            raise ValueError("Density phases require a complete single-file matter IC")
        if not np.array_equal(totals, header.get("NumPart_ThisFile")):
            raise ValueError("Incomplete particle counts in phase IC")
        for t in range(6):
            group = source.get(f"PartType{t}")
            actual = len(group["ParticleIDs"]) if group is not None else 0
            if actual != totals[t]:
                raise ValueError("Particle payload disagrees with phase IC counts")
        if header.get("Music_lagrangian_schema") != 1 or not math.isclose(
            float(header.get("Music_lagrangian_unitlength_cm", math.nan)),
            KPC_CM,
            rel_tol=1e-12,
        ):
            raise ValueError(
                "Phase measurement requires native original q in comoving kpc/h"
            )
        bound = np.asarray(header.get("Music_max_abs_displacement_native", []))
        if (
            bound.shape != (3,)
            or np.any(~np.isfinite(bound))
            or np.any(bound < 0)
            or np.any(bound >= box / 4)
        ):
            raise ValueError(
                "Native unwrapped displacement bounds are absent or too large"
            )
        k = modes * 2 * np.pi / box
        for t in particle_types:
            group = source.get(f"PartType{t}")
            if group is None:
                continue
            n = len(group["ParticleIDs"])
            for first in range(0, n, chunk):
                end = min(n, first + chunk)
                q = np.asarray(group["LagrangianCoordinates"][first:end])
                x = np.asarray(group["Coordinates"][first:end])
                mass = (
                    np.asarray(group["Masses"][first:end])
                    if "Masses" in group
                    else np.full(end - first, header["MassTable"][t])
                )
                if (
                    q.shape != (end - first, 3)
                    or x.shape != q.shape
                    or any(np.any(~np.isfinite(v)) for v in (q, x, mass))
                    or np.any(mass <= 0)
                    or np.any(q < 0)
                    or np.any(q >= box)
                    or np.any(x < 0)
                    or np.any(x >= box)
                    or mass.shape != (end - first,)
                ):
                    raise ValueError("Invalid original particle coordinates or masses")
                keep=slice(None)
                if refinement_levels is not None:
                    if 'LagrangianCellWidths' not in group or group['LagrangianCellWidths'].shape!=(n,):
                        raise ValueError('Refinement windows require aligned native LagrangianCellWidths')
                    widths=group['LagrangianCellWidths'][first:end]
                    levels=_native_cell_levels(widths,box,*hierarchy)
                    keep=np.isin(levels,refinement_levels)
                    if not np.any(keep):continue
                    q,x,mass=q[keep],x[keep],mass[keep]
                psi = (x - q + box / 2) % box - box / 2
                if np.any(np.max(np.abs(psi), axis=0) > bound + box * 1e-12):
                    raise ValueError("Delivered displacement exceeds its native bound")
                reference_mass = np.full(len(mass),mean_mass[t]) if mixed and t in (0,1) else mass
                vector = np.asarray(group['FirstOrderDisplacements'][first:end])[keep] if mixed else psi
                if vector.shape != q.shape or not np.isfinite(vector).all():
                    raise ValueError('Invalid retained first-order displacement')
                sign = -1j
                if velocity:
                    vector = np.asarray(group["FirstOrderVelocities" if mixed else "Velocities"][first:end])[keep]
                    if not mixed and 'ThermalVelocities' in group:
                        thermal = np.asarray(group['ThermalVelocities'][first:end])[keep]
                        if thermal.shape != vector.shape or not np.isfinite(thermal).all():
                            raise ValueError('Invalid native thermal velocity component')
                        vector = vector-thermal
                    if not mixed and header.get('Music_momentum_schema') == 1:
                        frame = np.asarray(header.get('Music_removed_bulk_velocity_gadget',[]),dtype=float)
                        if frame.shape != (3,) or not np.isfinite(frame).all():
                            raise ValueError('Invalid native momentum frame shift')
                        # Restore the solver transfer frame for the mode test;
                        # conservation separately measures delivered velocities.
                        vector = vector+frame
                    vector = vector * math.sqrt(a)
                    if vector.shape != q.shape or np.any(~np.isfinite(vector)):
                        raise ValueError("Invalid native velocity field")
                    sign = 1j
                result += np.sum(
                    sign * (vector @ k.T) * np.exp(-1j * (q @ k.T)) * reference_mass[:, None],
                    axis=0,
                )
                if mixed and not velocity:
                    result += np.sum((mass-reference_mass)[:,None]*np.exp(-1j*(q@k.T)),axis=0)
                total += float(mass.sum())
    if not math.isfinite(total) or not total > 0 or np.any(~np.isfinite(result)):
        raise ValueError("Phase measurement has no finite matter normalization")
    return result / total


def _noise(filename, n, shift, memory_gib):
    filename = Path(filename)
    with filename.open("rb") as stream:
        dimensions = np.fromfile(stream, dtype="=u4", count=3)
    if (
        not np.array_equal(dimensions, [n] * 3)
        or filename.stat().st_size != 12 + 8 * n**3
    ):
        raise ValueError(
            "Coarse cache must be the native-endian full double MUSIC cube"
        )
    if 32 * n**3 + 64 * 1024**2 > memory_gib * 1024**3:
        raise MemoryError("Coarse noise comparison exceeds its workspace budget")
    array = np.memmap(filename, mode="r", dtype="=f8", offset=12, shape=(n, n, n))
    if np.any(~np.isfinite(array)):
        raise ValueError("Coarse noise cache contains nonfinite values")
    return np.roll(array, tuple(shift), axis=(0, 1, 2))


@directory_owner('destination')
def compare_density_phases(
    parent_ic,
    zoom_ic,
    parent_noise,
    zoom_noise,
    destination,
    *,
    minimum_unique_modes=20,
    minimum_coherence=0.99,
    maximum_phase_rms_rad=0.05,
    memory_gib=1.0,
):
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError("Refusing to overwrite phase evidence")
    if (
        type(minimum_unique_modes) is not int
        or minimum_unique_modes < 1
        or not 0 < minimum_coherence <= 1
        or not 0 < maximum_phase_rms_rad <= np.pi
        or not math.isfinite(memory_gib)
        or memory_gib <= 0
    ):
        raise ValueError("Invalid declared phase criteria or workspace budget")
    source_hashes = {str(Path(p).resolve()):file_sha256(p) for p in (parent_ic, zoom_ic, parent_noise, zoom_noise)}
    frames = []
    for filename in (parent_ic, zoom_ic):
        with h5py.File(filename) as source:
            h = source["Header"].attrs
            frame = tuple(
                float(h[key])
                for key in (
                    "BoxSize",
                    "Time",
                    "HubbleParam",
                    "Omega0",
                    "OmegaBaryon",
                    "OmegaLambda",
                )
            )
            level = int(h["Music_levelmin"])
            if (
                not 1 <= level <= 30
                or not all(math.isfinite(v) for v in frame)
                or min(frame[:3]) <= 0
            ):
                raise ValueError("Invalid native phase frame or coarse level")
            shift = (
                np.asarray(h["Music_output_coordinate_shift_box_fraction"]) * 2**level
            )
            if (
                shift.shape != (3,)
                or np.any(~np.isfinite(shift))
                or np.any(np.abs(shift - np.rint(shift)) > 1e-10)
                or h["Music_shift_back"] != 1
            ):
                raise ValueError(
                    "Phase comparison requires restored integer-cell parent frames"
                )
            frames.append((frame, level, np.rint(shift).astype(int)))
    if (
        not np.allclose(frames[0][0], frames[1][0], rtol=1e-12, atol=0)
        or frames[0][1] != frames[1][1]
    ):
        raise ValueError("Parent/zoom cosmology, epoch or coarse grid differ")
    n = 2 ** frames[0][1]
    radius = min(4, n / 8)
    limit = math.floor(radius)
    modes = [
        v
        for v in itertools.product(range(-limit, limit + 1), repeat=3)
        if 0 < np.dot(v, v) <= radius**2 and next(a for a in v if a) > 0
    ]
    if len(modes) < minimum_unique_modes:
        raise ValueError(
            "Too few independent large-scale modes for declared phase criteria"
        )
    parent = linear_density_modes(parent_ic, modes, memory_gib=memory_gib)
    zoom = linear_density_modes(zoom_ic, modes, memory_gib=memory_gib)
    norm_parent, norm_zoom = (
        float(np.vdot(parent, parent).real),
        float(np.vdot(zoom, zoom).real),
    )
    if norm_parent <= np.finfo(float).tiny or norm_zoom <= np.finfo(float).tiny:
        raise ValueError("Zero perturbations cannot establish delivered density phases")
    coherence = float(abs(np.vdot(parent, zoom)) ** 2 / (norm_parent * norm_zoom))
    phases = np.angle(zoom * np.conj(parent))
    phase_rms = float(np.sqrt(np.sum(abs(parent) ** 2 * phases**2) / norm_parent))
    noise_a = _noise(parent_noise, n, frames[0][2], memory_gib)
    noise_b = _noise(zoom_noise, n, frames[1][2], memory_gib)
    noise_equal = np.array_equal(noise_a, noise_b)
    passed = (
        noise_equal
        and coherence >= minimum_coherence
        and phase_rms <= maximum_phase_rms_rad
    )
    report = {
        "status": "PASS" if passed else "FAIL",
        "scope": "Native first-order large-scale total-matter density phases; transfer correctness, velocity matching, nonlinear target recovery and convergence are separate gates",
        "unique_modes": len(modes),
        "coarse_noise_bitwise_equal": noise_equal,
        "density_coherence_squared": coherence,
        "weighted_phase_rms_rad": phase_rms,
        "density_amplitude_norm_ratio": math.sqrt(norm_zoom / norm_parent),
        "criteria": {
            "minimum_unique_modes": minimum_unique_modes,
            "minimum_coherence": minimum_coherence,
            "maximum_phase_rms_rad": maximum_phase_rms_rad,
        },
        "artifacts": {
            str(Path(p).resolve()): file_sha256(p)
            for p in (parent_ic, zoom_ic, parent_noise, zoom_noise)
        },
        "science_acceptance": "NOT_ACCEPTED",
    }
    report['artifacts'] = source_hashes
    return export_numeric_audit(destination, report,
        {'mode_vectors':np.asarray(modes,dtype=np.int64), 'parent_density':parent,
         'zoom_density':zoom, 'phase_difference_rad':phases},
        {'mode_vectors':{'units':'dimensionless','quantity':'integer Fourier wavevector in fundamental box modes'},
         'parent_density':{'units':'dimensionless','quantity':'mass-normalized complex linear density mode'},
         'zoom_density':{'units':'dimensionless','quantity':'mass-normalized complex linear density mode'},
         'phase_difference_rad':{'units':'rad','quantity':'argument of zoom density times conjugate parent density'}},
        algorithm=__file__, source_hashes=source_hashes,
        context={'epoch':epoch_contract(frames[0][0][1]),
                 'coordinates':coordinate_contract(position_units='comoving kpc/h',origin='native restored original Lagrangian box',periodic_wrapping='periodic Fourier box',velocity_convention='peculiar km/s'),
                 'definition':report['scope'], 'criteria':report['criteria']})


# Zoom velocity
#
# Measured large-scale linear velocity/density consistency in native 1LPT/2LPT.
#
# Tests delivered bulk particle velocities against signed CLASS N-body transfer
# ratios; separately stored thermal velocities are excluded from this linear test.
# Source provenance, small-scale/species transfer fidelity, phase and power
# normalization remain separate checks; this audit cannot grant science readiness.

@directory_owner('destination')
def audit_zoom_velocity(
    ic,
    transfer_table,
    destination,
    *,
    model,
    maximum_relative_l2_error=0.05,
    minimum_independent_modes=20,
    memory_gib=1.0,
):
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError("Refusing to overwrite zoom velocity evidence")
    if (
        model not in {"lcdm", "ede", "wdm"}
        or not math.isfinite(maximum_relative_l2_error)
        or not 0 < maximum_relative_l2_error < 1
        or type(minimum_independent_modes) is not int
        or minimum_independent_modes < 1
    ):
        raise ValueError("Invalid declared zoom velocity criteria")
    source_hashes = {str(Path(p).resolve()):file_sha256(p) for p in (ic, transfer_table)}
    verify_numeric_table(transfer_table)
    with h5py.File(ic) as source:
        h = source["Header"].attrs
        box, a, hubble = (float(h[k]) for k in ("BoxSize", "Time", "HubbleParam"))
        omega, baryon = float(h["Omega0"]), float(h["OmegaBaryon"])
        level = int(h["Music_levelmin"])
        if (
            not all(math.isfinite(v) and v > 0 for v in (box, a, hubble, omega))
            or not math.isfinite(baryon)
            or not 0 < baryon < omega
            or not 1 <= level <= 20
        ):
            raise ValueError("Invalid two-fluid native zoom velocity frame")
    table_path = Path(transfer_table)
    # A table-size budget is checked before loading its numeric payload.
    if (
        not math.isfinite(memory_gib)
        or memory_gib <= 0
        or table_path.stat().st_size * 8 + 64 * 1024**2 > memory_gib * 1024**3
    ):
        raise MemoryError("Transfer table exceeds declared velocity audit memory")
    comments = []
    with table_path.open() as stream:
        for line in stream:
            if line.startswith("#"):
                comments.append(line.rstrip())
    header = "\n".join(comments)
    redshift = re.search(r"redshift:\s*([0-9.eE+\-]+)", header)
    if (
        "DarkMatterSim monofonIC transfer table v1" not in header
        or "gauge: N-body" not in header
        or f"Model: {model};" not in header
        or redshift is None
        or not math.isclose(float(redshift[1]), 1 / a - 1, abs_tol=1e-8, rel_tol=0)
    ):
        raise ValueError(
            "Velocity audit requires matching signed N-body start-epoch transfers"
        )
    table = np.loadtxt(table_path, comments="#", ndmin=2)
    if (
        table.shape[1] != 5
        or len(table) < 6
        or np.any(~np.isfinite(table))
        or np.any(table[:, 0] <= 0)
        or np.any(np.diff(table[:, 0]) <= 0)
    ):
        raise ValueError("Malformed signed density/velocity transfer table")
    radius = min(4, 2**level / 8)
    limit = math.floor(radius)
    modes = np.array(
        [
            v
            for v in itertools.product(range(-limit, limit + 1), repeat=3)
            if 0 < np.dot(v, v) <= radius**2 and next(x for x in v if x) > 0
        ]
    )
    if len(modes) < minimum_independent_modes:
        raise ValueError(
            "Too few independent coarse modes for declared velocity criteria"
        )
    k = np.linalg.norm(modes, axis=1) * 2 * np.pi / (box / 1000)
    if k.min() < table[1, 0] or k.max() > table[-2, 0]:
        raise ValueError("Transfer spline does not cover the measured zoom modes")
    values = CubicSpline(np.log(table[:, 0]), table[:, 1:], bc_type="natural")(
        np.log(k)
    )
    fb = baryon / omega
    delta = (1 - fb) * values[:, 0] + fb * values[:, 1]
    theta = (1 - fb) * values[:, 2] + fb * values[:, 3]
    if np.any(delta == 0):
        raise ValueError(
            "Total-matter transfer ratio has a density zero at a measured mode"
        )
    density = linear_density_modes(ic, modes, memory_gib=memory_gib)
    measured = velocity_divergence_modes(ic, modes, memory_gib=memory_gib)
    # CLASS theta=div(v)/c in Mpc^-1; comoving IC k uses kpc/h.
    predicted = density * (theta / delta) * 299792.458 / (1000 * hubble)
    norm = float(np.linalg.norm(predicted))
    if not math.isfinite(norm) or norm <= np.finfo(float).tiny:
        raise ValueError(
            "Zero velocity perturbations cannot establish transfer consistency"
        )
    energetic = int(np.count_nonzero(abs(predicted) > norm * 1e-8))
    if energetic < minimum_independent_modes:
        raise ValueError("Too few measurable velocity modes for declared criteria")
    error = float(np.linalg.norm(measured - predicted) / norm)
    if not math.isfinite(error):
        raise ValueError("Nonfinite measured velocity consistency")
    report = {
        "status": "PASS" if error <= maximum_relative_l2_error else "FAIL",
        "scope": "Large-scale total-matter bulk velocity/density transfer ratio only",
        "thermal_velocity_handling": "subtract aligned ThermalVelocities when supplied; validate the relic distribution separately",
        "velocity_frame": "restore recorded native global momentum shift for transfer comparison; conservation uses delivered velocities",
        "relative_l2_error": error,
        "independent_modes": len(modes),
        "measurable_modes": energetic,
        "maximum_coarse_nyquist_fraction": radius / (2**level / 2),
        "criteria": {
            "maximum_relative_l2_error": maximum_relative_l2_error,
            "minimum_independent_modes": minimum_independent_modes,
        },
        "source_provenance": "NOT_VALIDATED_BY_THIS_AUDIT",
        "science_acceptance": "NOT_ACCEPTED",
        "artifacts": {
            str(Path(p).resolve()): file_sha256(p) for p in (ic, transfer_table)
        },
    }
    report['artifacts'] = source_hashes
    return export_numeric_audit(destination, report,
        {'mode_vectors':modes.astype(np.int64), 'k_h_Mpc':k, 'density':density,
         'measured_divergence':measured, 'predicted_divergence':predicted,
         'density_transfer':delta, 'velocity_divergence_transfer':theta},
        {'mode_vectors':{'units':'dimensionless','quantity':'integer Fourier wavevector in fundamental box modes'},
         'k_h_Mpc':{'units':'h/Mpc','quantity':'comoving wave number magnitude'},
         'density':{'units':'dimensionless','quantity':'mass-normalized complex linear density mode'},
         'measured_divergence':{'units':'km s^-1 (kpc/h)^-1','quantity':'complex divergence of peculiar velocity with comoving wave number'},
         'predicted_divergence':{'units':'km s^-1 (kpc/h)^-1','quantity':'CLASS signed total-matter transfer prediction'},
         'density_transfer':{'units':'dimensionless','quantity':'signed total-matter N-body density transfer'},
         'velocity_divergence_transfer':{'units':'Mpc^-1','quantity':'signed CLASS total-matter theta transfer in c=1 convention'}},
        algorithm=__file__, source_hashes=source_hashes,
        context={'epoch':epoch_contract(a),
                 'coordinates':coordinate_contract(position_units='comoving kpc/h',origin='native restored original Lagrangian box',periodic_wrapping='periodic Fourier box',velocity_convention='peculiar km/s'),
                 'model':model,'definition':report['scope'],'criteria':report['criteria']})


# Independent finite-volume hierarchy reference. This diagnoses the response to
# measured source cells; it does not certify their noise/transfer statistics.
def conservative_cell_prolongation(values, factor, *, degree=3, memory_gib=1.0):
    """Periodically prolong 3D cell averages while preserving parent masses.

    degree=0 is piecewise constant. Odd degree=3/5 integrates symmetric averages
    of left/right polynomial fits to neighboring cell averages over each child
    cell; it is not point interpolation.
    """
    if type(factor) is not int or not 1 <= factor <= 16 or type(degree) is not int or degree not in (0, 3, 5):
        raise ValueError('Declare factor 1..16 and reconstruction degree 0, 3 or 5')
    source = np.asarray(values)
    if source.ndim != 3 or source.dtype.kind not in 'fiu' or min(source.shape) < (degree+1 if degree else 1):
        raise ValueError('Finite three-dimensional cell-average grid required')
    if (not isinstance(memory_gib, (int,float,np.integer,np.floating)) or isinstance(memory_gib, bool) or not np.isfinite(memory_gib) or memory_gib <= 0
            or 8 * (3 * source.size * factor**3 + source.size) > memory_gib * 1024**3):
        raise ValueError('Conservative reconstruction exceeds declared memory budget')
    if not np.isfinite(source).all():
        raise ValueError('Nonfinite cell-average source')
    result = source.astype(np.float64, copy=True)
    if factor == 1:
        return result
    if degree == 0:
        for axis in range(3):
            result = np.repeat(result, factor, axis=axis)
        return result
    # Average left/right polynomial reconstructions to preserve reflection.
    # Cubic uses four cells per fit; quintic uses six, with a shared center.
    radius = (degree+1)//2
    offsets = np.arange(-float(radius), float(radius)+1)
    powers = np.arange(1, degree+2)
    inverses = []
    for stencil in (offsets[:-1], offsets[1:]):
        moments = ((stencil[:, None] + .5)**powers - (stencil[:, None] - .5)**powers) / powers
        inverses.append(np.linalg.inv(moments))
    for axis in range(3):
        shape = list(result.shape); shape[axis] *= factor
        refined = np.empty(shape, dtype=np.float64)
        for child in range(factor):
            left = child / factor - .5; right = (child + 1) / factor - .5
            child_moments = (right**powers - left**powers) / powers * factor
            weights = np.zeros(degree+2)
            weights[:-1] += .5 * (child_moments @ inverses[0])
            weights[1:] += .5 * (child_moments @ inverses[1])
            selection = [slice(None)] * 3; selection[axis] = slice(child, None, factor)
            refined[tuple(selection)] = 0.
            for offset, weight in zip(offsets.astype(int), weights):
                refined[tuple(selection)] += weight * np.roll(result, -offset, axis=axis)
        result = refined
    return result


def _prepare_cell_average_hierarchy(hierarchy, degree, memory_gib, workspace_cells=None, *, sampling='cell_average'):
    if sampling not in ('cell_average','cell_centered_point'):
        raise ValueError('Explicit supported hierarchy source sampling convention required')
    if not isinstance(hierarchy, (list, tuple)) or not hierarchy or type(degree) is not int or degree not in (0, 3, 5):
        raise ValueError('Declare a nonempty hierarchy and reconstruction degree 0, 3 or 5')
    records = []
    for record in hierarchy:
        if not isinstance(record, dict) or set(record) != {'level', 'origin', 'values'}:
            raise ValueError('Hierarchy records require level, origin and values')
        level = record['level']; origin = np.asarray(record['origin']); values = np.asarray(record['values'])
        if (type(level) is not int or not 2 <= level <= 30 or origin.shape != (3,)
                or origin.dtype.kind not in 'iu' or values.ndim != 3 or values.dtype.kind not in 'fiu'
                or any(n < 2 or n % 2 for n in values.shape) or any(n > 2**level for n in values.shape)):
            raise ValueError('Even aligned finite-volume grid geometry required')
        records.append({'level': level, 'origin': np.array([int(o) % (2**level) for o in origin],dtype=np.int64), 'values': values})
    levels = [r['level'] for r in records]
    if levels != list(range(levels[0], levels[-1] + 1)):
        raise ValueError('Hierarchy levels must be ordered and contiguous')
    base = records[0]; side = 2**levels[-1]
    if base['values'].shape != (2**levels[0],)*3 or np.any(base['origin'] != 0):
        raise ValueError('First hierarchy grid must cover the entire periodic box')
    if (not isinstance(memory_gib, (int,float,np.integer,np.floating)) or isinstance(memory_gib, bool) or not np.isfinite(memory_gib) or memory_gib <= 0
            or 8 * ((5 * side**3 if workspace_cells is None else workspace_cells) + 3 * sum(r['values'].size for r in records)) > memory_gib * 1024**3):
        raise ValueError('Hierarchy reference exceeds declared memory budget')
    for r in records:
        if not np.isfinite(r['values']).all():
            raise ValueError('Nonfinite hierarchy source')
        r['values'] = r['values'].astype(np.float64, copy=True)
    covered = {}
    for parent, child in zip(records, records[1:]):
        if np.any(child['origin'] % 2):
            raise ValueError('Child cells must align with complete parent cells')
        start = (child['origin']//2 - parent['origin']) % (2**parent['level'])
        shape = np.array(child['values'].shape)//2
        periodic_parent = parent['values'].shape == (2**parent['level'],)*3
        if not periodic_parent and np.any(start + shape > parent['values'].shape):
            raise ValueError('Child patch must be contained in its parent grid')
        covered[parent['level']] = np.ix_(*[(int(a)+np.arange(int(b))) % n
            for a,b,n in zip(start,shape,parent['values'].shape)])
    leaf_integral = 0.
    for r in records:
        mask = np.ones(r['values'].shape, dtype=bool)
        if r['level'] in covered: mask[covered[r['level']]] = False
        leaf_integral += float(np.sum(r['values'][mask])) / 2.**(3*r['level'])
    if sampling=='cell_average':
        for index in range(len(records)-1, 0, -1):
            child = records[index]; parent = records[index-1]; nx,ny,nz = child['values'].shape
            parent['values'][covered[parent['level']]] = child['values'].reshape(nx//2,2,ny//2,2,nz//2,2).mean(axis=(1,3,5))
    return records, levels, leaf_integral


def compose_hierarchy_density_reference(hierarchy, *, degree=3, memory_gib=1.0):
    """Reconstruct rectangular nested source-cell averages on a periodic grid.

    Records contain exactly level, origin (absolute cell indices), and values.
    Covered coarse cells are replaced by child averages before reconstruction;
    actual leaf-cell averages remain unchanged. The first grid covers the box.
    """
    records, levels, leaf_integral = _prepare_cell_average_hierarchy(hierarchy, degree, memory_gib)
    side = 2**levels[-1]
    field = records[0]['values']
    for r in records[1:]:
        field = conservative_cell_prolongation(field, 2, degree=degree, memory_gib=memory_gib)
        indices = [(int(o)+np.arange(n)) % (2**r['level']) for o,n in zip(r['origin'],r['values'].shape)]
        field[np.ix_(*indices)] = r['values']
    mean = float(field.mean())
    scale = max(float(np.max(np.abs(field))), 1.)
    if abs(mean-leaf_integral) > 2e-13*scale:
        raise RuntimeError('Hierarchy reconstruction changed integrated leaf source')
    return field, {'schema':'finite-volume-hierarchy-reference-v1','degree':degree,
        'levels':levels,'side':side,'grids':[{'level':r['level'],'origin':r['origin'].tolist(),'shape':list(r['values'].shape)} for r in records],'leaf_source_mean':leaf_integral,'reconstructed_mean':mean,
        'scope':'Independent reconstruction of measured source cells; not a noise/transfer or launch certificate',
        'science_acceptance':'NOT_ACCEPTED','launch_authorized':False}


def sample_hierarchy_density_reference(hierarchy, origin, shape, *, degree=3, memory_gib=1.0, sampling='cell_average'):
    """Evaluate only a bounded finest-level rectangle of the hierarchy model.

    The default is equivalent to the conservative full-grid model followed by
    periodic slicing. Cell-centred point sampling instead preserves measured
    parent nodes and interpolates at child centres with symmetric polynomial
    fits; it never replaces parent point samples by child averages. Native
    source conventions and physical acceptance remain separate requirements.
    """
    start = np.asarray(origin); extent = np.asarray(shape)
    if (start.shape != (3,) or extent.shape != (3,) or start.dtype.kind not in 'iu'
            or extent.dtype.kind not in 'iu' or any(int(n)<=0 or int(n)>2**30 for n in extent)
            or type(degree) is not int or degree not in (0,3,5)
            or not isinstance(hierarchy,(list,tuple)) or not hierarchy):
        raise ValueError('Declare integer origin/positive shape, hierarchy and degree 0/3/5')
    # Bound every recursive rectangle before allocating coordinate or work arrays.
    sizes = [int(n) for n in extent]; workspace_cells = 0
    for _ in hierarchy:
        workspace_cells += 6 * math.prod(sizes)
        sizes = [(n+1)//2 + degree+3 for n in sizes]
    records, levels, _ = _prepare_cell_average_hierarchy(hierarchy,degree,memory_gib,workspace_cells,sampling=sampling)
    side = 2**levels[-1]
    coordinates = [(int(o)%side + np.arange(int(n),dtype=np.int64))%side for o,n in zip(start,extent)]
    offsets = np.array([0],dtype=int) if degree==0 else np.arange(-(degree+1)//2,(degree+1)//2+1)
    weights = np.ones((2,1)) if degree==0 else np.zeros((2,degree+2))
    if degree:
        powers = np.arange(1,degree+2) if sampling=='cell_average' else np.arange(degree+1)
        for stencil_number,stencil in enumerate((offsets[:-1],offsets[1:])):
            moments = (((stencil[:,None]+.5)**powers-(stencil[:,None]-.5)**powers)/powers
                if sampling=='cell_average' else stencil[:,None]**powers)
            inverse = np.linalg.inv(moments)
            for child in (0,1):
                left=child/2-.5;right=(child+1)/2-.5
                values=(((right**powers-left**powers)/powers*2) if sampling=='cell_average'
                    else ((left+right)/2)**powers)@inverse*.5
                weights[child,stencil_number:stencil_number+degree+1] += values
    evaluated_cells = []
    def sample(index, coords):
        record = records[index]; local = [(c-int(o))%(2**record['level']) for c,o in zip(coords,record['origin'])]
        if index==0:
            result = record['values'][np.ix_(*local)]
        else:
            parent_side = 2**records[index-1]['level']
            parent_candidates = [(c[:,None]//2+offsets[None,:])%parent_side for c in coords]
            parents = [np.unique(c) for c in parent_candidates]
            lookup = [np.searchsorted(p,c) for p,c in zip(parents,parent_candidates)]
            result = sample(index-1,parents)
            for axis in range(3):
                dimensions=list(result.shape);dimensions[axis]=len(coords[axis]);refined=np.zeros(dimensions,dtype=np.float64)
                broadcast=[1,1,1];broadcast[axis]=len(coords[axis])
                for column in range(len(offsets)):
                    refined += np.take(result,lookup[axis][:,column],axis=axis) * weights[coords[axis]%2,column].reshape(broadcast)
                result=refined
            selections = [np.flatnonzero(c<n) for c,n in zip(local,record['values'].shape)]
            if all(len(s) for s in selections):
                result[np.ix_(*selections)] = record['values'][np.ix_(*[c[s] for c,s in zip(local,selections)])]
        evaluated_cells.append({'level':record['level'],'shape':list(result.shape),'cells':int(result.size)})
        return result
    values=sample(len(records)-1,coordinates)
    if not np.isfinite(values).all():raise RuntimeError('Nonfinite bounded hierarchy reference')
    return values, {'schema':'bounded-hierarchy-source-reference-v1' if sampling=='cell_average' else 'bounded-hierarchy-point-reference-v1',
        'sampling_convention':sampling,'degree':degree,'levels':levels,
        'origin':[int(x) for x in start],'shape':[int(x) for x in extent],
        'evaluated_rectangles':evaluated_cells,'estimated_workspace_bytes':8*workspace_cells,
        'global_finest_grid_allocated':False,'scope':'Bounded declared reconstruction of supplied hierarchy source values; native sampling/physics validation separate',
        'science_acceptance':'NOT_ACCEPTED','launch_authorized':False}


def periodic_hierarchy_displacement_reference(density, box_native, *, oversample=1, degree=3, memory_gib=1.0):
    """Solve a declared periodic source reconstruction with an independent FFT.

    Returns displacement averaged back into the input cells, in box_native
    length units. Oversampling/basis convergence must be measured separately.
    The continuum zero mode and derivative Nyquist planes are projected out.
    """
    values = np.asarray(density)
    if (values.ndim != 3 or len(set(values.shape)) != 1 or values.shape[0] < 4
            or values.shape[0] % 2 or values.dtype.kind not in 'fiu'
            or type(oversample) is not int or oversample not in (1,2,4) or type(degree) is not int or degree not in (0,3,5)):
        raise ValueError('Even cubic source, oversample 1/2/4 and reconstruction degree 0/3/5/5 required')
    if (not isinstance(box_native, (int,float,np.integer,np.floating)) or isinstance(box_native, bool) or not np.isfinite(box_native) or box_native <= 0
            or not isinstance(memory_gib, (int,float,np.integer,np.floating)) or isinstance(memory_gib, bool) or not np.isfinite(memory_gib) or memory_gib <= 0):
        raise ValueError('Positive finite box length and memory budget required')
    original = values.shape[0]; side = original*oversample
    if 12*8*side**3 > memory_gib*1024**3:
        raise ValueError('Periodic reference FFT exceeds declared memory budget')
    if not np.isfinite(values).all(): raise ValueError('Nonfinite periodic source')
    field = conservative_cell_prolongation(values,oversample,degree=degree,memory_gib=memory_gib)
    mean = float(field.mean()); spectrum = np.fft.rfftn(field,axes=(0,1,2)); del field
    frequencies = np.fft.fftfreq(side)*side; positive = np.fft.rfftfreq(side)*side
    waves = (frequencies[:,None,None], frequencies[None,:,None], positive[None,None,:])
    norm2 = waves[0]**2 + waves[1]**2 + waves[2]**2; norm2[0,0,0] = 1.
    spectrum /= norm2; spectrum[0,0,0] = 0.; del norm2
    output = np.empty((original,original,original,3),dtype=np.float64)
    for axis,wave in enumerate(waves):
        coefficient = np.where(np.abs(wave)==side/2,0.,wave) * (1j*box_native/(2*np.pi))
        work = spectrum * coefficient
        response = np.fft.irfftn(work,s=(side,)*3,axes=(0,1,2)); del work
        output[...,axis] = response.reshape(original,oversample,original,oversample,original,oversample).mean(axis=(1,3,5))
        del response
    if not np.isfinite(output).all(): raise RuntimeError('Nonfinite independent periodic response')
    return output, {'schema':'periodic-hierarchy-response-v1','oversample':oversample,'degree':degree,
        'source_side':original,'fft_side':side,'projected_source_mean':mean,'box_native':float(box_native),
        'response':'Cell-averaged displacement in declared box length units; independent continuum FFT',
        'scope':'Response to measured hierarchy reconstruction; approximation convergence and noise/transfer provenance separate',
        'science_acceptance':'NOT_ACCEPTED','launch_authorized':False}


# Zoom window
#
# Linear Lagrangian-window response of delivered species displacements.
#
# The observable is -i k dot <Psi exp(-i k q)> with a declared mass window.
# It includes the advected patch boundary and is not an Eulerian masked-density
# power spectrum. Model modes must be independent positive-half-space modes;
# omitted-mode, lattice/solver and start-epoch convergence remain separate gates.

def _modes(value):
    result = np.asarray(value)
    if (result.ndim != 2 or result.shape[1] != 3 or not len(result)
            or result.dtype.kind not in 'iuf' or np.any(~np.isfinite(result))
            or np.any(result != np.rint(result)) or np.any(np.all(result == 0, axis=1))
            or len(np.unique(result, axis=0)) != len(result)
            or any(next(x for x in row if x) < 0 for row in result)):
        raise ValueError('Declare unique nonzero integer positive-half-space Fourier modes')
    return result.astype(float)


def window_response(q_comoving_kpc_h, mass, box_comoving_kpc_h, observed_modes,
                    source_modes, *, memory_gib=1.):
    """Map independent real/imaginary full-box density modes to patch modes.

    For delta_m = r_m + i s_m, Psi_m = i k_m delta_m/k_m^2. Reality fixes
    the negative partner. Both partners contribute in a nonuniform window,
    so circular complex Gaussian assumptions would give the wrong covariance.
    Rows are all real outputs then all imaginary outputs; columns alternate
    source real and imaginary parts. No source mode is silently truncated.
    """
    observed, source = _modes(observed_modes), _modes(source_modes)
    q, mass = np.asarray(q_comoving_kpc_h), np.asarray(mass)
    box = box_comoving_kpc_h
    if (isinstance(box, bool) or not math.isfinite(box) or box <= 0
            or q.ndim != 2 or q.shape[1] != 3 or not len(q) or mass.shape != (len(q),)
            or np.any(~np.isfinite(q)) or np.any(q < 0) or np.any(q >= box)
            or np.any(~np.isfinite(mass)) or np.any(mass <= 0)):
        raise ValueError('Use finite periodic native q and positive aligned masses')
    if isinstance(memory_gib, bool) or not math.isfinite(memory_gib) or memory_gib <= 0:
        raise ValueError('Declare a positive window-response memory budget')
    m, n = len(observed), len(source)
    reserve = 5*q.nbytes + 2*mass.nbytes + 64*1024**2 + 96*m*n
    available = memory_gib*1024**3 - reserve
    if available < 96*m:
        raise MemoryError('Window-response input/matrix/workspace exceeds declared memory')
    chunk = max(1, min(8192, int(available//(96*m))))
    total = float(mass.sum())
    if not math.isfinite(total) or total <= 0:
        raise ValueError('Window normalization overflowed')
    response = np.empty((2*m, 2*n))
    cartesian = None
    if np.all(mass == mass[0]):
        axes = [np.unique(q[:,axis]) for axis in range(3)]
        if math.prod(len(axis) for axis in axes) == len(q) and len(np.unique(q,axis=0)) == len(q):
            cartesian = axes
    for index, wave in enumerate(source):
        dot = (observed @ wave) / (wave @ wave)
        minus, plus = np.zeros(m, complex), np.zeros(m, complex)
        if cartesian is not None:
            minus[:], plus[:] = 1, 1
            for axis,coordinates in enumerate(cartesian):
                phase = coordinates[:,None]*(2*np.pi/box)
                minus *= np.mean(np.exp(-1j*phase*(observed[:,axis]-wave[axis])),axis=0)
                plus *= np.mean(np.exp(-1j*phase*(observed[:,axis]+wave[axis])),axis=0)
        else:
            for first in range(0, len(q), chunk):
                points = q[first:first+chunk] * (2*np.pi/box)
                weights = mass[first:first+chunk]/total
                minus += weights @ np.exp(-1j*(points @ (observed-wave).T))
                plus += weights @ np.exp(-1j*(points @ (observed+wave).T))
        real_source = dot*(minus-plus)
        imaginary_source = 1j*dot*(minus+plus)
        response[:m,2*index], response[m:,2*index] = real_source.real, real_source.imag
        response[:m,2*index+1], response[m:,2*index+1] = imaginary_source.real, imaginary_source.imag
    return response


def source_mode_sphere(radius, *, memory_gib=1.):
    """Enumerate independent conjugate pairs inside an integer-mode sphere.

    This declares a continuum Fourier basis, not an even-grid Nyquist model.
    Native Nyquist/self-conjugate modes require separate discretization handling.
    """
    if type(radius) is not int or not 1 <= radius <= 128:
        raise ValueError('Declare an integer source radius in [1,128]')
    if (isinstance(memory_gib,bool) or not math.isfinite(memory_gib) or memory_gib<=0
            or 64*1024**2+128*(2*radius+1)**3 > memory_gib*1024**3):
        raise MemoryError('Source sphere enumeration exceeds its declared workspace budget')
    import itertools
    return np.array([wave for wave in itertools.product(range(-radius,radius+1),repeat=3)
                     if 0 < sum(x*x for x in wave) <= radius*radius
                     and next(x for x in wave if x) > 0],dtype=int)


def covariance_source_convergence(response, source_modes, source_power_mpc_h3,
                                  box_comoving_kpc_h, radii, *, maximum_fractional_change,
                                  maximum_condition_number=1e12):
    """Bound covariance changes in every whitened observed direction.

    Each spherical checkpoint must contain every integer mode in that sphere.
    Consecutive-shell stability is empirical; it is never a bound on the tail
    beyond the supplied basis or proof of native discretization accuracy.
    """
    source = _modes(source_modes)
    response, power = np.asarray(response), np.asarray(source_power_mpc_h3)
    if (not isinstance(radii,list) or len(radii)<3
            or any(type(r) is not int or not 1<=r<=128 for r in radii)
            or any(b<=a for a,b in zip(radii,radii[1:]))
            or isinstance(maximum_fractional_change,bool) or not math.isfinite(maximum_fractional_change)
            or not 0<maximum_fractional_change<1
            or isinstance(maximum_condition_number,bool) or not math.isfinite(maximum_condition_number)
            or maximum_condition_number<1
            or response.ndim!=2 or response.shape[1]!=2*len(source)
            or power.shape!=(len(source),)):
        raise ValueError('Declare aligned source basis and three increasing spherical convergence checkpoints')
    source_set = {tuple(map(int,row)) for row in source}
    for radius in radii:
        if not {tuple(row) for row in source_mode_sphere(radius)} <= source_set:
            raise ValueError('Source basis omits modes inside a declared convergence sphere')
    previous, records = None, []
    squared = np.sum(source*source,axis=1)
    for radius in radii:
        keep = squared <= radius*radius
        covariance = model_window_covariance(response[:,np.repeat(keep,2)],power[keep],box_comoving_kpc_h)
        eigen, vectors = np.linalg.eigh(covariance)
        record={'radius':radius,'source_modes':int(keep.sum()),'minimum_eigenvalue':float(eigen[0]),
                'maximum_eigenvalue':float(eigen[-1]),'maximum_fractional_change':None}
        if eigen[-1]<=0 or eigen[0]<=eigen[-1]/maximum_condition_number:
            record['status']='UNRESOLVED_COVARIANCE'
        elif previous is None:
            record['status']='BASELINE'
        else:
            # Whiten against the finer covariance, retaining weak directions.
            inverse_root = (vectors/np.sqrt(eigen)) @ vectors.T
            change = inverse_root @ (covariance-previous) @ inverse_root
            changes=np.linalg.eigvalsh((change+change.T)*.5)
            maximum=float(np.max(np.abs(changes)))
            record.update(maximum_fractional_change=maximum,
                          status='PASS' if maximum<=maximum_fractional_change else 'FAIL')
        previous=covariance
        records.append(record)
    return {'status':'STABLE_DECLARED_SHELLS' if all(r['status']=='PASS' for r in records[-2:]) else 'NOT_STABLE',
            'records':records,'maximum_fractional_change':maximum_fractional_change,
            'science_acceptance':'NOT_ACCEPTED','omitted_tail_status':'NOT_BOUNDED',
            'native_discretization_status':'NOT_VALIDATED',
            'scope':'Two consecutive complete spherical shell increments; no guarantee beyond the largest source radius'}


def model_window_covariance(response, source_power_mpc_h3, box_comoving_kpc_h):
    """Real covariance including the nonzero complex pseudo-covariance.

    For the convention delta(k)=box-volume^-1 integral delta exp(-ikx), each
    independent real/imaginary mode has variance P(k)/(2 L^3). P and L use
    Mpc/h; q and the box passed to the operator use native comoving kpc/h.
    """
    response, power = np.asarray(response), np.asarray(source_power_mpc_h3)
    box = box_comoving_kpc_h
    if (response.ndim != 2 or response.shape[0] % 2 or not response.size
            or power.ndim != 1 or response.shape[1] != 2*len(power)
            or np.any(~np.isfinite(response)) or np.any(~np.isfinite(power)) or np.any(power < 0)
            or isinstance(box, bool) or not math.isfinite(box) or box <= 0):
        raise ValueError('Invalid real window operator, species power or box')
    variance = np.repeat(power/(2*(box/1000)**3),2)
    return (response*variance) @ response.T


def window_gaussian_check(measured_modes, covariance, *, false_rejection_probability,
                          maximum_condition_number=1e12):
    """Correlated Gaussian consistency for the declared finite source basis.

    A singular/ill-conditioned covariance is unresolved, never regularized
    into a pass. This is not a transfer-amplitude or omitted-mode tolerance.
    """
    from scipy.stats import chi2
    measured, covariance = np.asarray(measured_modes), np.asarray(covariance)
    if (measured.ndim != 1 or not len(measured) or np.any(~np.isfinite(measured))
            or covariance.shape != (2*len(measured),2*len(measured))
            or np.any(~np.isfinite(covariance))
            or not np.allclose(covariance,covariance.T,rtol=1e-12,
                               atol=32*np.finfo(float).eps*np.max(np.abs(covariance)))
            or isinstance(false_rejection_probability,bool) or not 0 < false_rejection_probability < 1
            or not math.isfinite(maximum_condition_number) or maximum_condition_number < 1):
        raise ValueError('Declare valid measured modes, real covariance and test criteria')
    values, vectors = np.linalg.eigh(covariance)
    if values[-1] <= 0 or values[0] <= values[-1]/maximum_condition_number:
        return {'status':'UNRESOLVED_COVARIANCE','science_acceptance':'NOT_ACCEPTED',
                'reason':'Singular or ill-conditioned declared window covariance'}
    data = np.concatenate([measured.real,measured.imag])
    statistic = float(np.sum((vectors.T @ data)**2/values))
    cdf, tail = float(chi2.cdf(statistic,len(data))), float(chi2.sf(statistic,len(data)))
    p = min(1.,2*min(cdf,tail))
    return {'status':'PASS' if p >= false_rejection_probability else 'FAIL',
            'chi_squared':statistic,'degrees_of_freedom':len(data),'two_sided_p_value':p,
            'false_rejection_probability':false_rejection_probability,
            'condition_number':float(values[-1]/values[0]),'science_acceptance':'NOT_ACCEPTED',
            'scope':'Correlated Gaussian consistency for the declared source modes/window only; omitted modes and native discretization require separate convergence'}


@directory_owner('destination')
def plot_window_correlations(covariances, labels, destination, *, context=None):
    """Shared-scale correlation heatmaps expose the partial-volume coupling."""
    from pathlib import Path
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    destination = Path(destination)
    if not covariances or any((destination.with_suffix('.'+suffix)).exists() for suffix in ('png','pdf')):
        raise ValueError('Declare covariance panels and a fresh figure destination')
    figure, axes = plt.subplots(1,len(covariances),figsize=(5*len(covariances),4.6),squeeze=False,
                               layout='constrained')
    try:
        arrays={}; specifications={}
        for panel,(axis,(title,covariance)) in enumerate(zip(axes[0],covariances.items())):
            covariance=np.asarray(covariance)
            if covariance.shape!=(len(labels),len(labels)) or np.any(~np.isfinite(covariance)) or np.any(np.diag(covariance)<=0):
                raise ValueError('Correlation panels require finite covariance with positive variances')
            sigma=np.sqrt(np.diag(covariance))
            correlation=covariance/sigma[:,None]/sigma[None,:]
            if np.any(np.abs(correlation)>1+1e-10):
                raise ValueError('Covariance violates the correlation bound')
            for quantity,value in (('covariance',covariance),('correlation',correlation)):
                key=f'panel_{panel:03d}_{quantity}';arrays[key]=value
                specifications[key]={'units':'dimensionless','quantity':'finite-mode density '+quantity,
                    'annotations':{'panel':str(title),'mode_labels':list(labels)}}
            rendered=axis.imshow(correlation,cmap='coolwarm',vmin=-1,vmax=1)
            axis.set_title(title)
            axis.set_xticks(range(len(labels)),labels,rotation=70,ha='right')
            axis.set_yticks(range(len(labels)),labels)
        figure.colorbar(rendered,ax=axes[0].tolist(),label='Correlation',shrink=.8)
        figure.suptitle('Declared finite-mode Lagrangian-window model; science NOT_ACCEPTED',fontsize=11)
        destination.parent.mkdir(parents=True,exist_ok=True)
        inputs=export_model_arrays(destination,arrays,specifications,algorithm=__file__,context=context)
        return export_figure(figure,destination,formats=['png','pdf'],dpi=180,close=False,
            metadata={'science_acceptance':'NOT_ACCEPTED','numeric_inputs':inputs,'source_context':context or {}})
    finally:
        plt.close(figure)


def fourier_mode_labels(modes):
    """Human-readable labels, independent of NumPy scalar repr/version."""
    modes = _modes(modes)
    triples = ['('+','.join(str(int(value)) for value in row)+')' for row in modes]
    return ['Re '+value for value in triples] + ['Im '+value for value in triples]


@directory_owner('destination')
def plot_source_convergence(results, destination, *, context=None):
    """Show worst-direction changes and the stabilization of weak variance."""
    from pathlib import Path
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    destination=Path(destination)
    if not results or any(destination.with_suffix('.'+suffix).exists() for suffix in ('png','pdf')):
        raise ValueError('Declare convergence results and a fresh figure destination')
    figure,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    try:
        arrays={};specifications={}
        for panel,(name,result) in enumerate(results.items()):
            records=result['records']
            radius=np.array([row['radius'] for row in records])
            changes=np.array([np.nan if row['maximum_fractional_change'] is None else row['maximum_fractional_change'] for row in records])
            minimum=np.array([row['minimum_eigenvalue'] for row in records])
            if not len(radius):raise ValueError('Convergence records cannot be empty')
            if np.any(~np.isfinite(minimum)) or minimum[-1]<=0 or np.any(changes[np.isfinite(changes)]<0):
                raise ValueError('Invalid saved covariance convergence measurements')
            for quantity,value,unit in (('radius',radius,'integer Fourier units'),
                    ('fractional_change',changes,'dimensionless'),('minimum_eigenvalue',minimum,'dimensionless')):
                key=f'panel_{panel:03d}_{quantity}';arrays[key]=value
                specifications[key]={'units':unit,'quantity':'finite-mode shell convergence '+quantity,
                    'missing_policy':'NaN denotes unavailable first-shell comparison' if quantity=='fractional_change' else 'No missing values',
                    'annotations':{'panel':str(name),'declared_change_threshold':result['maximum_fractional_change']}}
            axes[0].plot(radius[1:],changes[1:]*100,'o-',label=name)
            axes[0].axhline(result['maximum_fractional_change']*100,color='grey',linestyle=':',alpha=.7)
            axes[1].plot(radius,minimum/minimum[-1],'o-',label=name)
        axes[0].set_yscale('log')
        axes[0].set_ylabel('Maximum whitened covariance change (%)')
        axes[0].set_title('Complete spherical shell increments')
        axes[1].set_ylabel('Smallest variance / value at final radius')
        axes[1].set_title('Weak covariance direction')
        for axis in axes:
            axis.set_xscale('log',base=2)
            axis.set_xlabel('Source radius (integer Fourier units)')
            axis.grid(alpha=.2)
            axis.legend()
        figure.suptitle('Measured shell stability; omitted tail and native grid accuracy remain unvalidated',fontsize=10)
        destination.parent.mkdir(parents=True,exist_ok=True)
        inputs=export_model_arrays(destination,arrays,specifications,algorithm=__file__,context=context)
        return export_figure(figure,destination,formats=['png','pdf'],dpi=180,close=False,
            metadata={'science_acceptance':'NOT_ACCEPTED','numeric_inputs':inputs,'source_context':context or {}})
    finally:
        plt.close(figure)
from core.native import native_run_owner
from core.foundation import atomic_copy
from core.native import run_native_command
import subprocess


# Zoom resolution
#
# Measured coarse-particle contamination within declared saved science regions.
#
# Centers are explicit saved-frame coordinates, never guessed from a camera or
# interpolated orbit. Passing contamination alone does not establish force,
# stream-width, stellar-population or mass-resolution convergence.

def radial_particle_census(paths, snapshot, *, center_comoving_kpc_h,
                          radial_edges_proper_kpc, fine_dm_mass_msun,
                          memory_gib, fine_dm_mass_range_msun=None):
    """Measure radial moments from complete native shards with bounded reads.

    Actual saved species/class softenings accompany unequal particle masses.
    The radial accumulator retains sum(m^2), so rare coarse particles cannot
    disappear into a count-based or mean-mass relaxation/heating calculation.
    """
    from analysis import validate_gadget_snapshot_set
    from halos import _unit_conversion
    from zoom import empty_census, accumulate_census, seal_census, positive, G
    validate_gadget_snapshot_set(paths, snapshot)
    _unit_conversion(paths)
    memory = positive(memory_gib, 'memory_gib') * 1024**3
    positive(fine_dm_mass_msun, 'fine_dm_mass_msun')
    fine_upper = _fine_mass_upper(fine_dm_mass_msun,fine_dm_mass_range_msun)
    chunk = min(100_000, int(memory // 256))
    if chunk < 1:
        raise MemoryError('Resolution census workspace cannot hold one particle')
    center = np.asarray(center_comoving_kpc_h, dtype=float)
    if center.shape != (3,) or not np.isfinite(center).all():
        raise ValueError('Census requires a finite saved-frame center')
    census = empty_census(radial_edges_proper_kpc)
    sources = {str(Path(p).resolve()): file_sha256(p) for p in paths}
    reference = None
    for path in paths:
        with h5py.File(path) as f:
            header, params = f['Header'].attrs, f['Parameters'].attrs
            a, h, box = float(header['Time']), float(params['HubbleParam']), float(header['BoxSize'])
            if not all(math.isfinite(x) and x > 0 for x in (a, h, box)) or census['edges_proper_kpc'][-1] >= box*a/h/2:
                raise ValueError('Census aperture must fit inside half the proper periodic box')
            g_override=float(params.get('GravityConstantInternal',0.))
            if not math.isfinite(g_override) or g_override<0:raise ValueError('Invalid native gravity constant')
            gravity=G if g_override==0 else g_override/1e10
            signature = (a, h, box, {k: float(v) for k, v in params.items() if k.startswith('Softening')},gravity)
            if reference is not None and signature != reference:
                raise ValueError('Census shards disagree about epoch/frame/softening')
            reference = signature
            for t in range(6):
                group = f.get(f'PartType{t}')
                if group is None:
                    continue
                count = len(group['Coordinates'])
                for first in range(0, count, chunk):
                    end = min(first+chunk, count)
                    pos = np.asarray(group['Coordinates'][first:end], dtype=float)
                    masses = (np.asarray(group['Masses'][first:end], dtype=float) if 'Masses' in group else np.full(end-first, float(header['MassTable'][t]))) * 1e10/h
                    classes = np.asarray(group['GravitySofteningClass'][first:end]) if 'GravitySofteningClass' in group else None
                    softening = proper_species_softening(params, t, a, h, classes)
                    if np.ndim(softening) == 0:
                        softening = np.full(end-first, softening)
                    radii = np.linalg.norm((pos-center+box/2)%box-box/2, axis=1) * a/h
                    coarse = np.full(end-first, t == 2)
                    if t == 1:
                        if np.any(masses < fine_dm_mass_msun*(1-1e-10)):
                            raise ValueError('Declared finest mass exceeds actual type-1 mass')
                        coarse |= masses > fine_upper*(1+1e-10)
                    accumulate_census(census, radii, masses, t, softening, coarse=coarse)
    if any(file_sha256(p) != checksum for p, checksum in sources.items()):
        raise ValueError('Native census input changed during measurement')
    return {**seal_census(census), 'source_hashes': sources, 'snapshot': int(snapshot),
            'gravity_constant_kpc_kms2_msun':reference[4],
            'fine_dm_mass_range_msun':fine_dm_mass_range_msun,
            'scale_factor': reference[0], 'center_comoving_kpc_h': center.tolist(),
            'read_workspace_bytes': chunk*256, 'measurement_origin': 'native_saved_snapshot'}


def _fine_mass_upper(fine_mass, mass_range):
    if mass_range is None:return fine_mass
    if not isinstance(mass_range,(list,tuple)) or len(mass_range)!=2 or not all(math.isfinite(x) and x>0 for x in mass_range) or mass_range[1]<mass_range[0] or not math.isclose(mass_range[0],fine_mass,rel_tol=1e-10):
        raise ValueError('Fine mass range must start at the declared actual minimum mass')
    return mass_range[1]


def contamination_epoch(
    paths,
    snapshot,
    *,
    center_comoving_kpc_h,
    radius_proper_kpc,
    fine_dm_mass_msun,
    maximum_coarse_mass_fraction,
    fine_dm_mass_range_msun=None,
):
    from analysis import validate_gadget_snapshot_set
    from halos import _unit_conversion

    source_records=[{'path':str(Path(path).resolve()),'sha256':file_sha256(path)} for path in paths]
    validate_gadget_snapshot_set(paths, snapshot)
    _unit_conversion(paths)
    center = np.asarray(center_comoving_kpc_h, dtype=float)
    if center.shape != (3,) or not np.all(np.isfinite(center)):
        raise ValueError(
            "Science center must be three finite comoving kpc/h coordinates"
        )
    if not all(
        math.isfinite(x) and x > 0 for x in (radius_proper_kpc, fine_dm_mass_msun)
    ):
        raise ValueError("Science radius and fine DM mass must be finite and positive")
    if (
        not math.isfinite(maximum_coarse_mass_fraction)
        or not 0 <= maximum_coarse_mass_fraction <= 1
    ):
        raise ValueError("Coarse mass fraction tolerance must be in [0,1]")
    total_inside, coarse_inside, coarse_count = 0.0, 0.0, 0
    fine_upper = _fine_mass_upper(fine_dm_mass_msun,fine_dm_mass_range_msun)
    nearest = math.inf
    star_count, fine_dm_count = 0, 0
    epoch = None
    species = {}
    for path in paths:
        with h5py.File(path) as source:
            attrs, params = source["Header"].attrs, source["Parameters"].attrs
            a = hdf5_scalar_attr(attrs["Time"], "Time")
            h = float(params["HubbleParam"])
            box = hdf5_scalar_attr(attrs["BoxSize"], "BoxSize")
            if not a > 0 or not h > 0 or radius_proper_kpc >= box * a / h / 2:
                raise ValueError(
                    "Science aperture must fit within half the proper periodic box"
                )
            if epoch is not None and not np.allclose(epoch, (a, h, box), rtol=1e-12):
                raise ValueError("Contamination shards disagree about frame")
            epoch = (a, h, box)
            for t in range(6):
                group = source.get(f"PartType{t}")
                if group is None:
                    continue
                count = len(group["Coordinates"])
                table_mass = float(attrs["MassTable"][t])
                if "Masses" not in group and table_mass <= 0:
                    raise ValueError(
                        "Contamination audit requires positive actual masses"
                    )
                bucket = species.setdefault(
                    str(t), {"count_inside": 0, "mass_inside_msun": 0.0}
                )
                for first in range(0, count, 100_000):
                    end = min(first + 100_000, count)
                    pos = np.asarray(group["Coordinates"][first:end], dtype=float)
                    mass = (
                        (
                            np.asarray(group["Masses"][first:end], dtype=float)
                            if "Masses" in group
                            else np.full(end - first, table_mass)
                        )
                        * 1e10
                        / h
                    )
                    if (
                        np.any(~np.isfinite(pos))
                        or np.any(~np.isfinite(mass))
                        or np.any(mass <= 0)
                    ):
                        raise ValueError(
                            "Contamination data must have finite positions and positive masses"
                        )
                    delta = (pos - center + box / 2) % box - box / 2
                    distance = np.linalg.norm(delta, axis=1) * a / h
                    inside = distance <= radius_proper_kpc
                    coarse = np.full(len(mass), t == 2)
                    if t == 1:
                        if np.any(mass < fine_dm_mass_msun * (1 - 1e-10)):
                            raise ValueError(
                                "DM masses are smaller than the declared finest resolution"
                            )
                        coarse |= mass > fine_upper * (1 + 1e-10)
                        fine_dm_count += int(np.count_nonzero(inside & ~coarse))
                    if t == 4:
                        star_count += int(np.count_nonzero(inside))
                    if np.any(coarse):
                        nearest = min(nearest, float(np.min(distance[coarse])))
                    coarse_count += int(np.count_nonzero(coarse & inside))
                    coarse_inside += float(mass[coarse & inside].sum())
                    subtotal = float(mass[inside].sum())
                    total_inside += subtotal
                    bucket["count_inside"] += int(np.count_nonzero(inside))
                    bucket["mass_inside_msun"] += subtotal
    fraction = coarse_inside / total_inside if total_inside > 0 else None
    if any(file_sha256(entry['path'])!=entry['sha256'] for entry in source_records):
        raise ValueError('Contamination snapshot changed during measurement')
    status = (
        "EMPTY_REGION"
        if fraction is None
        else "PASS"
        if fraction <= maximum_coarse_mass_fraction
        else "FAIL"
    )
    return {
        "snapshot": int(snapshot),
        "scale_factor": epoch[0],
        "h": epoch[1],
        "box_comoving_kpc_h": epoch[2],
        "center_comoving_kpc_h": center.tolist(),
        "radius_proper_kpc": radius_proper_kpc,
        "coarse_count_inside": coarse_count,
        "coarse_mass_inside_msun": coarse_inside,
        "total_mass_inside_msun": total_inside,
        "coarse_mass_fraction": fraction,
        "nearest_coarse_distance_proper_kpc": nearest
        if math.isfinite(nearest)
        else None,
        "fine_dm_count_inside": fine_dm_count,
        "stellar_count_inside": star_count,
        "species": species,
        "status": status,
        "source_files": [str(Path(p).resolve()) for p in paths],
        'source_sha256':source_records,
    }




@native_run_owner
@directory_owner('output_directory',default_relative='results/zoom_contamination')
def audit_zoom_regions(
    run_dir,
    *,
    regions=None,
    fine_dm_mass_msun,
    maximum_coarse_mass_fraction,
    output_directory=None,
    host_tracking=None,
    minimum_fine_dm_count=None,
    minimum_stellar_count=None,
    resolution_budget=None,
):
    from analysis import find_hdf5_files
    from core.data import run_epoch_contract,coordinate_contract,write_dataset_contract
    from core.foundation import product_provenance, verify_product_provenance, publish_directory, atomic_json, file_sha256 as digest

    run_dir = Path(run_dir).resolve()
    if (regions is None)==(host_tracking is None):
        raise ValueError('Supply exactly one of explicit regions or host_tracking')
    tracking,anchor_ids=None,None
    if host_tracking is not None:
        if not isinstance(host_tracking,dict):
            raise ValueError('host_tracking must contain explicit anchor/epoch/overlap criteria')
        if minimum_fine_dm_count is None or minimum_stellar_count is None:
            raise ValueError('Tracked host audit requires explicit minimum fine-DM and stellar counts')
        from zoom import recover_host_regions
        tracking,anchor_ids=recover_host_regions(run_dir,**host_tracking)
        if not math.isclose(fine_dm_mass_msun,tracking['run_lineage']['fine_dm_mass_msun'],rel_tol=1e-10):
            raise ValueError('Declared finest DM mass differs from the actual recorded IC inventory')
        regions=tracking['regions'] if tracking['status']=='PASS' else []
    minimum_fine_dm_count=0 if minimum_fine_dm_count is None else minimum_fine_dm_count
    minimum_stellar_count=0 if minimum_stellar_count is None else minimum_stellar_count
    if any(type(value) is not int or value<0 for value in (minimum_fine_dm_count,minimum_stellar_count)):
        raise ValueError('Sampling thresholds must be nonnegative integer counts')
    if tracking is None and (not isinstance(regions, list) or not regions):
        raise ValueError("Supply explicit saved science regions")
    records = []
    fine_range=None
    lineage_sources=[]
    if (run_dir/'manifest.json').is_file():
        native_manifest=json.loads((run_dir/'manifest.json').read_text())
        if native_manifest.get('gadget_handoff',{}).get('initial_conditions'):
            from zoom import _run_lineage
            native_lineage=_run_lineage(run_dir)
            if native_lineage['mixed_fluid_fine_masses']:
                fine_range=native_lineage['fine_dm_mass_range_msun']
                if not math.isclose(fine_dm_mass_msun,fine_range[0],rel_tol=1e-10):
                    raise ValueError('Mixed-fluid finest mass differs from recorded actual IC range')
                lineage_sources=[{'path':native_lineage['path'],'sha256':native_lineage['sha256']},*native_lineage['initial_conditions']]
    for region in regions:
        if set(region) != {"snapshot", "center_comoving_kpc_h", "radius_proper_kpc"}:
            raise ValueError(
                "Each region needs snapshot, center_comoving_kpc_h and radius_proper_kpc"
            )
        snap = region["snapshot"]
        if isinstance(snap, bool) or not isinstance(snap, int) or snap < 0:
            raise ValueError("Snapshot must be a nonnegative integer")
        paths = find_hdf5_files(run_dir / "data", f"snapshot_{snap:03d}")
        record=contamination_epoch(
                paths,
                **region,
                fine_dm_mass_msun=fine_dm_mass_msun,
                maximum_coarse_mass_fraction=maximum_coarse_mass_fraction,
                fine_dm_mass_range_msun=fine_range,
            )
        record['contamination_status']=record['status']
        record['epoch_contract']=run_epoch_contract(run_dir,snap,record['scale_factor'])
        record['sampling_status']='PASS' if (record['fine_dm_count_inside']>=minimum_fine_dm_count
            and record['stellar_count_inside']>=minimum_stellar_count) else 'FAIL'
        if record['status']=='PASS' and record['sampling_status']=='FAIL':
            record['status']='FAIL_SAMPLING'
        if resolution_budget is not None:
            from zoom import evaluate_resolution, validate_resolution_request
            if not isinstance(resolution_budget, dict) or set(resolution_budget) != {'request', 'targets_by_snapshot', 'memory_gib', 'convergence_by_snapshot'}:
                raise ValueError('Resolution audit needs request, targets_by_snapshot, memory_gib and convergence_by_snapshot')
            request = validate_resolution_request(resolution_budget['request'])
            if request['radial_edges_proper_kpc'][-1] != region['radius_proper_kpc']:
                raise ValueError('Resolution and contamination apertures must match')
            census = radial_particle_census(paths, snap, center_comoving_kpc_h=region['center_comoving_kpc_h'],
                radial_edges_proper_kpc=request['radial_edges_proper_kpc'], fine_dm_mass_msun=fine_dm_mass_msun,
                memory_gib=resolution_budget['memory_gib'],fine_dm_mass_range_msun=fine_range)
            census['source_hashes'].update({entry['path']:entry['sha256'] for entry in lineage_sources})
            record['resolution_budget'] = evaluate_resolution(resolution_budget['targets_by_snapshot'][str(snap)], census, request,
                convergence=resolution_budget['convergence_by_snapshot'].get(str(snap)))
            config_path = run_dir/'config.yaml'
            if config_path.is_file():
                import yaml
                from core.resources import resolution_followup_resources
                cfg = yaml.safe_load(config_path.read_text())
                generation_path=cfg.get('simulation',{}).get('native_generation_report')
                if generation_path:
                    from zoom import verify_generated_zoom
                    generated=verify_generated_zoom(generation_path)
                    record['resolution_budget']['refinement']['native_generation_source']={
                        'path':str(Path(generation_path).resolve()),'sha256':file_sha256(generation_path),
                        'source_configuration':generated['source_configuration']}
                    entry={'path':str(Path(generation_path).resolve()),'sha256':file_sha256(generation_path)}
                    record['source_sha256'].append(entry)
                    record['resolution_budget']['source_hashes'][entry['path']]=entry['sha256']
                # Saved aperture counts forecast local cost only. Full IC counts
                # and native mesh planning remain required before regeneration.
                counts = [int(sum(row)) for row in census['count']]
                if sum(counts):
                    record['resolution_budget']['refinement']['local_resource_forecast'] = resolution_followup_resources(
                        cfg, counts, len(regions), request['refinement'])
                    record['source_sha256'].append({'path':str(config_path),'sha256':file_sha256(config_path)})
            if record['resolution_budget']['preflight_status'] != 'PASS':
                record['status'] = 'FAIL_RESOLUTION'
        records.append(record)
    if any(
        records[i + 1]["scale_factor"] <= records[i]["scale_factor"]
        for i in range(len(records) - 1)
    ):
        raise ValueError(
            "Science regions must be in strictly increasing saved time order"
        )
    for key in ("h", "box_comoving_kpc_h"):
        if records and not np.allclose([r[key] for r in records], records[0][key], rtol=1e-12):
            raise ValueError("Science epochs disagree about " + key)
    report = {
        "schema": "zoom-contamination-v1",
        "records": records,
        "fine_dm_mass_msun": fine_dm_mass_msun,
        "maximum_coarse_mass_fraction": maximum_coarse_mass_fraction,
        "center_origin": ('Exact fixed-anchor DM overlap in saved FOF catalogues' if tracking is not None else
                          'User-declared saved-frame centers; host identity requires separate particle tracking'),
        'host_tracking':tracking,
        'sampling_criteria':{'minimum_fine_dm_count':minimum_fine_dm_count,'minimum_stellar_count':minimum_stellar_count},
        "coarse_definition": "All type-2 particles and type-1 masses above the constant fine mass or verified mixed-fluid fine IC range; type 5 is BH, not coarse DM",
        'verified_mixed_fluid_fine_mass_range_msun':fine_range,
        "status": "PASS" if records and all(r["status"] == "PASS" for r in records) else "FAIL",
        'science_acceptance':'NOT_ACCEPTED',
        "science_scope": ("Measured species/mass census, force/relaxation/heating preflight and independently verified observable convergence budgets. No automatic physical claim."
                          if resolution_budget is not None else "Contamination and declared particle-count gates only. Force/tracer quality and observable convergence remain separate."),
    }
    destination = (
        Path(output_directory).resolve()
        if output_directory
        else run_dir / "results/zoom_contamination"
    )
    if destination.exists():
        raise FileExistsError("Zoom audit output exists: " + str(destination))
    destination.parent.mkdir(parents=True, exist_ok=True)
    source_records=[entry for row in records for entry in row['source_sha256']]
    source_records.extend(lineage_sources)
    if tracking:
        lineage=tracking['run_lineage']
        source_records.extend([lineage,*lineage['initial_conditions'],*tracking['anchor_sources'],
            *[entry for row in tracking['records'] for entry in row['source_files']]])
    report['data_contract']=product_provenance(run_dir,algorithm=__file__,
        selection='Declared saved science regions and coarse/fine mass thresholds',source_files=source_records,
        coordinates=coordinate_contract(position_units='proper kpc',
            origin='declared saved-frame region center',periodic_wrapping='minimum image; aperture inside half-box',
            velocity_convention='no velocities in this contamination product'),
        helper_sources=[(Path(__file__).parent / name) for name in ('zoom.py','zoom.py','foundation.py')])
    stage=Path(tempfile.mkdtemp(prefix='.zoom-audit-',dir=destination.parent))
    try:
        for row in records:
            if 'resolution_budget' in row:
                from zoom import publish_resolution_budget
                filename=f'resolution_{row["snapshot"]:03d}.json'
                published=publish_resolution_budget(stage/filename,row['resolution_budget'],epoch=row['epoch_contract'],
                    coordinates=report['data_contract']['coordinates'])
                row['resolution_numeric_audit']={'path':filename,'sha256':digest(stage/filename),
                    'numeric_data':published['numeric_data']}
        if anchor_ids is not None:
            with h5py.File(stage/'anchor_particles.hdf5','w') as source:
                dataset=source.create_dataset('ParticleIDs',data=anchor_ids.astype(np.uint64),chunks=True,fletcher32=True)
                write_dataset_contract(dataset,units='identity',quantity='exact fixed-anchor dark-matter host particle IDs',
                    identity_namespace=report['data_contract']['identity_namespaces']['dark_matter'],
                    provenance=report['data_contract'],ancestry={'selection':'fixed saved anchor; overlaps measured at later saved epochs'})
            report['anchor_particles_sha256']=file_sha256(stage/'anchor_particles.hdf5')
        (stage / "provenance.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 4 if tracking else 3, figsize=(17 if tracking else 13, 4), layout="constrained")
        x = [r["scale_factor"] for r in records]
        axes[0].plot(x, [r["coarse_mass_fraction"] for r in records], "o-")
        axes[0].axhline(
            maximum_coarse_mass_fraction,
            color="r",
            linestyle="--",
            label="Declared tolerance",
        )
        axes[0].set_ylabel("Coarse mass / enclosed total mass")
        if tracking and tracking['status']!='PASS':
            axes[0].text(.5,.5,'Host identity unresolved\nNo aperture evaluated',ha='center',va='center',transform=axes[0].transAxes)
        axes[0].legend()
        axes[1].plot(
            x, [r["nearest_coarse_distance_proper_kpc"] for r in records], "o-"
        )
        axes[1].set_ylabel("Nearest coarse particle [proper kpc]")
        if all(r["nearest_coarse_distance_proper_kpc"] is None for r in records):
            axes[1].text(
                0.5,
                0.5,
                "No coarse particles\nin saved volume",
                ha="center",
                va="center",
                transform=axes[1].transAxes,
            )
        axes[2].plot(
            x, [r["stellar_count_inside"] for r in records], "o-", label="Stars"
        )
        axes[2].plot(
            x, [r["fine_dm_count_inside"] for r in records], "o-", label="Fine DM"
        )
        axes[2].set_ylabel("Particle count within science aperture")
        from matplotlib.ticker import MaxNLocator
        axes[2].yaxis.set_major_locator(MaxNLocator(integer=True))
        axes[2].legend()
        if tracking:
            epochs=[row['scale_factor'] for row in tracking['records']]
            axes[3].plot(epochs,[row['anchor_fraction'] for row in tracking['records']],'o-',label='Strongest host')
            axes[3].plot(epochs,[row['runner_up_anchor_fraction'] for row in tracking['records']],'s--',label='Runner up')
            axes[3].axhline(tracking['criteria']['minimum_anchor_fraction'],color='k',linestyle=':')
            axes[3].set_ylabel('Fraction of fixed anchor DM tracers')
            axes[3].set_ylim(0,1.05)
            axes[3].legend()
        for ax in axes:
            ax.set_xlabel("Saved scale factor a")
            ax.ticklabel_format(axis="x", style="plain", useOffset=False)
        epochs=x or ([row['scale_factor'] for row in tracking['records']] if tracking else [])
        if epochs:
            lo,hi=min(epochs),max(epochs)
            pad=max((hi-lo)*.05,.01)
            for ax in axes:
                ax.set_xlim(lo-pad,hi+pad)
        for ax in axes[:3]:
            ax.set_ylim(bottom=0)
        if not any(row['nearest_coarse_distance_proper_kpc'] is not None for row in records):
            axes[1].set_yticks([])
        fig.suptitle("Saved-region resolution and contamination audit")
        from figures import export_figure
        export_figure(fig,stage/'zoom_contamination',formats=('png','pdf'),dpi=180,close=False,
            metadata={'science_acceptance':'NOT_ACCEPTED','data_contract':report['data_contract'],
                'epochs':[row['epoch_contract'] for row in records],'x_units':'scale factor',
                'y_units':['fraction','proper kpc','particle count']+(['fraction'] if tracking else [])})
        plt.close(fig)
        if tracking:
            lineage=tracking['run_lineage']
            for entry in [lineage,*lineage['initial_conditions'],*tracking['anchor_sources'],
                          *[entry for row in tracking['records'] for entry in row['source_files']]]:
                if file_sha256(entry['path'])!=entry['sha256']:
                    raise ValueError('Tracked host source changed before contamination audit publication')
        (stage/'provenance.sha256').write_text(digest(stage/'provenance.json')+'\n')
        verify_product_provenance(run_dir,report['data_contract'])
        publish_directory(stage,destination)
    except BaseException as exc:
        if stage.exists():atomic_json(stage/'export_failure.json',{'status':'FAIL','science_acceptance':'NOT_ACCEPTED',
            'failure':{'type':type(exc).__name__,'message':str(exc)}})
        raise
    return {
        "directory": str(destination),
        "status": report["status"],
        "figure": str(destination / "zoom_contamination.png"),
    }


# Zoom softening
#
# Explicit mass-scaled coarse gravity resolution; not a convergence claim.

def softening_plan(simulation):
    soft = float(simulation["softening_kpc_h"])
    physical = float(simulation["softening_max_physical_kpc_h"])
    if (
        not math.isfinite(soft)
        or not math.isfinite(physical)
        or not 0 < physical <= soft
    ):
        raise ValueError(
            "Softening requires finite positive comoving and physical limits"
        )
    ratios = simulation.get("zoom_softening_mass_ratios")
    species = simulation.get('species_softening_kpc_h')
    comoving, maximum = [soft] * 6, [physical] * 6
    if species is not None:
        if not isinstance(species, dict) or set(species) != {str(t) for t in range(6)}:
            raise ValueError('species_softening_kpc_h must explicitly describe all six native particle types')
        for t in range(6):
            item = species[str(t)]
            if not isinstance(item, dict) or set(item) != {'comoving', 'maximum_physical'}:
                raise ValueError('Each species softening needs comoving and maximum_physical')
            values = (item['comoving'], item['maximum_physical'])
            if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in values) or not 0 < values[1] <= values[0]:
                raise ValueError('Species softening requires finite positive physical <= comoving limits')
            comoving[t], maximum[t] = values
    if ratios is None:
        return {
            "type_classes": list(range(6)),
            "comoving": comoving,
            "maximum_physical": maximum,
            "coarse_mass_ratios": [],
        }
    if not isinstance(ratios, list) or not 1 <= len(ratios) <= 121:
        raise ValueError(
            "zoom_softening_mass_ratios must list 1 to 121 distinct coarse/fine DM mass ratios"
        )
    if any(
        isinstance(v, bool)
        or not isinstance(v, (float, int))
        or not math.isfinite(v)
        or v <= 1
        for v in ratios
    ):
        raise ValueError("Coarse mass ratios must be finite numbers greater than one")
    if any(b <= a for a, b in zip(ratios, ratios[1:])):
        raise ValueError("Coarse mass ratios must be strictly increasing")
    comoving += [comoving[1] * v ** (1 / 3) for v in ratios]
    maximum += [maximum[1] * v ** (1 / 3) for v in ratios]
    if not all(math.isfinite(v) for v in comoving + maximum):
        raise ValueError("Mass-scaled softening overflows native precision")
    return {
        "type_classes": [0, 1, 6, 3, 4, 5],
        "comoving": comoving,
        "maximum_physical": maximum,
        "coarse_mass_ratios": ratios,
    }


def proper_species_softening(parameters, particle_type, scale_factor, h, classes=None):
    """Read the actual saved native class mapping, including mass-selected type 2.

    Softening is Plummer-equivalent epsilon. Native spline support is 2.8 epsilon.
    No fallback to a configuration scalar is allowed for measured budgets.
    """
    t = int(particle_type)
    if t not in range(6) or not math.isfinite(scale_factor) or not math.isfinite(h) or min(scale_factor, h) <= 0:
        raise ValueError('Invalid softening species or epoch')
    cls = int(parameters[f'SofteningClassOfPartType{t}'])
    def value(k):
        c = float(parameters[f'SofteningComovingClass{k}'])
        p = float(parameters[f'SofteningMaxPhysClass{k}'])
        if not all(math.isfinite(x) and x > 0 for x in (c, p)):
            raise ValueError('Invalid saved native softening')
        return min(c * scale_factor, p) / h
    if t == 2 and cls == 6:
        if classes is None:
            raise ValueError('Mass-selected coarse softening requires saved GravitySofteningClass')
    if classes is not None:
        indices = np.asarray(classes)
        if indices.ndim != 1 or indices.dtype.kind not in 'iu' or np.any(indices < 0):
            raise ValueError('Saved softening classes must be a vector of nonnegative integers')
        result = np.empty(len(indices))
        for k in np.unique(indices):
            result[indices == k] = value(int(k))
        return result
    return value(cls)


def validate_material_softening_masses(paths,simulation):
    """Check actual material masses against native nearest-log softening classes.

    Geometry supplies hierarchy levels; material masses supply the native
    requested cube-root epsilon. No octree level is inferred from a varying
    compensated mass. This is a technical assignment check, not convergence.
    """
    paths=[Path(path).resolve() for path in paths];plan=softening_plan(simulation)
    if not plan['coarse_mass_ratios']:raise ValueError('Nested material masses need explicit coarse softening ratios')
    if len(paths)!=1:raise ValueError('Material softening currently requires the complete single-file transformed generation')
    with h5py.File(paths[0],'r') as native:
        header=native['Header'].attrs
        if header.get('Research_twofluid_flow_schema')!=1 or header.get('Research_twofluid_projection_mode')!='finite_volume_material_cells' or header.get('Music_material_cell_schema')!=1:
            raise ValueError('Typed source-bound finite-volume material IC required for variable coarse mass classes')
        a=float(header['Time']);box=float(header['BoxSize']);high=int(header['Music_levelmax']);low=int(header['Music_levelmin'])
        if not 0<a<=1 or not 1<=low<high<=20:raise ValueError('Material softening requires positive epoch and nested dyadic levels')
        fine=native['PartType1/Masses'];count=len(fine);total=0.;minimum=math.inf;maximum=0.
        if not count:raise ValueError('Material softening has no fine-DM reference census')
        for first in range(0,count,100000):
            masses=np.asarray(fine[first:first+100000],float)
            if masses.ndim!=1 or not np.isfinite(masses).all() or np.any(masses<=0):raise ValueError('Invalid fine material masses')
            total+=float(masses.sum());minimum=min(minimum,float(masses.min()));maximum=max(maximum,float(masses.max()))
        reference=total/count;declared=float(header['Research_fine_dm_mean_mass_native'])
        if not math.isclose(reference,declared,rel_tol=1e-10):raise ValueError('Native fine material reference mean differs from its generation stamp')
        group=native['PartType2'];mass=group['Masses'];width=group['LagrangianCellWidths'];levels_present=set()
        if mass.shape!=width.shape or mass.ndim!=1 or not len(mass):raise ValueError('Material coarse mass and geometry census differs')
        for first in range(0,len(width),100000):
            widths=np.asarray(width[first:first+100000],float)
            if not np.isfinite(widths).all() or np.any(widths<=0):raise ValueError('Invalid coarse material softening cell widths')
            levels=np.rint(np.log2(box/widths)).astype(np.int64)
            if np.any(levels<low) or np.any(levels>=high) or not np.allclose(widths,box/np.exp2(levels),rtol=1e-12,atol=0):raise ValueError('Coarse material softening geometry is not a dyadic leaf hierarchy')
            levels_present.update(int(value) for value in levels)
        levels_order=sorted(levels_present,reverse=True)
        if len(levels_order)!=len(plan['coarse_mass_ratios']):raise ValueError('Declared coarse softening classes differ from represented material hierarchy levels')
        expected={level:6+index for index,level in enumerate(levels_order)}
        epsilon=np.minimum(np.asarray(plan['comoving'],float),np.asarray(plan['maximum_physical'],float)/a)
        observed_counts={str(level):0 for level in levels_order};ranges={str(level):[math.inf,0.] for level in levels_order};smallest_margin=math.inf
        for first in range(0,len(mass),100000):
            masses=np.asarray(mass[first:first+100000],float);widths=np.asarray(width[first:first+100000],float);levels=np.rint(np.log2(box/widths)).astype(np.int64)
            if masses.shape!=widths.shape or not np.isfinite(masses).all() or np.any(masses<=0):raise ValueError('Invalid coarse material masses')
            desired=epsilon[plan['type_classes'][1]]*np.power(masses/reference,1/3)
            # Native ties select the earliest index because its loop uses <.
            log_desired=np.log(desired);selected=np.full(len(masses),-1,np.int64);best=np.full(len(masses),math.inf);second=np.full(len(masses),math.inf)
            for index,value in enumerate(epsilon):
                distance=abs(log_desired-math.log(value));better=distance<best
                second=np.where(better,best,np.minimum(second,distance));best=np.minimum(best,distance);selected[better]=index
            margin=second-best;roundoff=128*np.finfo(float).eps*np.maximum(1,np.abs(log_desired))
            if np.any(margin<=roundoff):raise ValueError('Material softening assignment lies on an ambiguous native floating-point class boundary')
            smallest_margin=min(smallest_margin,float(margin.min()))
            for level in np.unique(levels):
                use=levels==level
                if np.any(selected[use]!=expected[int(level)]):raise ValueError('Actual compensated coarse masses cross their declared hierarchy softening class boundary')
                key=str(int(level));observed_counts[key]+=int(np.sum(use));ranges[key][0]=min(ranges[key][0],float(masses[use].min()));ranges[key][1]=max(ranges[key][1],float(masses[use].max()))
    return {'status':'PASS','fine_dm_mass_native':reference,'fine_dm_mass_range_native':[minimum,maximum],
        'material_mass_reference':True,'plan':plan,'native_mask':4,'epoch_scale_factor':a,'levels_to_classes':{str(level):value for level,value in expected.items()},
        'counts_by_level':observed_counts,'mass_ranges_native_by_level':ranges,
        'minimum_nearest_class_log_margin':smallest_margin,
        'selection':'actual cube-root mass scaling and nearest-log epsilon over all native classes, including physical caps',
        'convergence':'NOT_DEMONSTRATED'}


def validate_coarse_softening_masses(paths, simulation):
    """Check declared fine-DM mean mass and every coarse mass before native init.

    Chunked reads; complete shards and other IC invariants are checked by the
    calling IC validator. Ratios are relative to fine DM, including combined
    baryon+DM mass in MUSIC's coarse cells.
    """
    plan = softening_plan(simulation)
    if not plan["coarse_mass_ratios"]:
        return None
    paths=list(paths)
    if not paths:raise ValueError('Coarse softening needs complete IC paths')
    with h5py.File(paths[0],'r') as source:
        if source['Header'].attrs.get('Research_twofluid_flow_schema')==1 and source['Header'].attrs.get('Research_twofluid_projection_mode')=='finite_volume_material_cells':
            return validate_material_softening_masses(paths,simulation)
    fine_min, fine_max, fine_sum, fine_count = math.inf, 0.0, 0.0, 0
    coarse_seen = set()
    mixed_mean = None
    mixed_flags = set()
    for t in (1, 2):
        for path in paths:
            with h5py.File(path) as source:
                group = source.get(f"PartType{t}")
                if group is None:
                    continue
                if t == 1:
                    header = source['Header'].attrs
                    mixed = header.get('Music_mixed_fluid_schema') == 1
                    mixed_flags.add(bool(mixed))
                    if mixed:
                        means = np.asarray(header.get('Music_mean_species_masses_native',[]),dtype=float)
                        if means.shape != (6,) or not np.isfinite(means).all() or means[1] <= 0:
                            raise ValueError('Mixed-fluid softening requires valid declared mean species masses')
                        if mixed_mean is not None and not math.isclose(mixed_mean,float(means[1]),rel_tol=1e-12):
                            raise ValueError('Mixed-fluid fine mean mass differs across shards')
                        mixed_mean = float(means[1])
                n = len(group["Coordinates"])
                for start in range(0, n, 100_000):
                    size = min(100_000, n - start)
                    masses = (
                        np.asarray(group["Masses"][start : start + size], dtype=float)
                        if "Masses" in group
                        else np.full(size, source["Header"].attrs["MassTable"][t])
                    )
                    if (
                        masses.shape != (size,)
                        or np.any(~np.isfinite(masses))
                        or np.any(masses <= 0)
                    ):
                        raise ValueError("Invalid particle masses for zoom softening")
                    if t == 1:
                        fine_min = min(fine_min, float(masses.min()))
                        fine_max = max(fine_max, float(masses.max()))
                        fine_sum += float(masses.sum())
                        fine_count += size
                    else:
                        reference = fine_sum / fine_count
                        for mass in np.unique(masses):
                            ratio = float(mass / reference)
                            match = np.flatnonzero(
                                np.isclose(
                                    ratio, plan["coarse_mass_ratios"], rtol=1e-6, atol=0
                                )
                            )
                            if len(match) != 1:
                                raise ValueError(
                                    "Actual coarse mass has no unique declared softening class"
                                )
                            coarse_seen.add(int(match[0]))
        if t == 1 and (
            not fine_count
            or not math.isfinite(fine_sum)
            or len(mixed_flags) != 1
            or (mixed_mean is None and fine_max > fine_min * (1 + 1e-6))
            or (mixed_mean is not None and not math.isclose(fine_sum/fine_count,mixed_mean,rel_tol=1e-10))
        ):
            raise ValueError(
                "Mass-based softening requires constant fine DM or verified mixed-fluid mean mass"
            )
    if coarse_seen != set(range(len(plan["coarse_mass_ratios"]))):
        raise ValueError(
            "Declared coarse softening classes do not match the actual IC mass levels"
        )
    return {
        "status": "PASS",
        "fine_dm_mass_native": fine_sum / fine_count,
        "fine_dm_mass_range_native": [fine_min,fine_max],
        "mixed_fluid_mean_mass": mixed_mean is not None,
        "plan": plan,
        "native_mask": 4,
        "selection": "native cube-root mass scaling relative to fine type-1 DM",
        "convergence": "NOT_DEMONSTRATED",
    }


# Zoom resources
#
# Native MUSIC geometry planning and explicit estimated resource budgets.
#
# Actual hierarchy geometry comes from the generator before field allocation.
# Working-set coefficients are conservative planning assumptions, not measured
# RSS guarantees. Runtime must be calibrated on the intended computer.

def refinement_generation_limits(configuration, memory_gib, disk_gib):
    """Enforce the saved resolution follow-up cap in the real MUSIC planner."""
    import json
    from pathlib import Path
    from core.foundation import file_sha256
    stage = Path(configuration).resolve().parent
    pointer = stage.with_name(stage.name+'.refinement.json')
    if not pointer.is_file():
        return memory_gib, disk_gib, None
    if pointer.with_name(pointer.name+'.sha256').read_text().strip()!=file_sha256(pointer):
        raise ValueError('Resolution refinement lineage seal mismatch')
    lineage=json.loads(pointer.read_text())
    if lineage.get('schema')!='resolution-refinement-lineage-v1' or lineage.get('status')!='STAGED' or file_sha256(stage/'staging.json')!=lineage['child_staging_sha256']:
        raise ValueError('Resolution lineage does not identify this staged child')
    if file_sha256(lineage['budget_receipt'])!=lineage['budget_sha256']:
        raise ValueError('Resolution follow-up budget changed')
    memory=min(float(memory_gib),float(lineage['native_memory_gib']))
    disk=float(lineage['native_disk_gib']) if disk_gib is None else min(float(disk_gib),float(lineage['native_disk_gib']))
    return memory,disk,{'path':str(pointer),'sha256':file_sha256(pointer),'lineage':lineage}


def validate_bounded_source_settings(settings, *, lpt_order):
    """Validate the existing native extension options without changing defaults."""
    if settings is None: return None
    keys={'hybrid_source_extension','hybrid_extension_degree','hybrid_extension_boundary',
          'hybrid_extension_memory_mib','hybrid_extension_max_side'}
    if not isinstance(settings,dict) or set(settings)!=keys or settings['hybrid_source_extension']!='bounded_conservative':
        raise ValueError('Complete explicit bounded source settings required')
    if lpt_order!=2: raise ValueError('Bounded source settings require mixed 2LPT staging')
    for key in keys-{'hybrid_source_extension'}:
        if type(settings[key]) is not int: raise ValueError('Exact integer bounded source setting required: '+key)
    degree=settings['hybrid_extension_degree']; boundary=settings['hybrid_extension_boundary']
    memory=settings['hybrid_extension_memory_mib']; side=settings['hybrid_extension_max_side']
    if degree not in (3,5) or not 8<=side<=4096 or not 0<boundary<=side//2 or not 0<memory<=min(2**32-1,np.iinfo(np.uintp).max//(1024**2)):
        raise ValueError('Bounded source settings exceed native supported geometry/workspace')
    return dict(settings)


def estimate_mesh_resources(plan):
    if (plan.get('schema') != 'darkmattersim-music-mesh-v1'
            or type(plan.get('scalar_bytes')) is not int or plan['scalar_bytes'] not in (4, 8, 16)
            or plan.get('boundary_cells') != 4 or type(plan.get('convolution_margin')) is not int):
        raise ValueError('Unknown native MUSIC mesh-plan schema/storage')
    hierarchies = plan.get('hierarchies')
    if not isinstance(hierarchies, dict) or set(hierarchies) != {'poisson', 'convolution'}:
        raise ValueError('Native mesh plan must include both grid hierarchies')
    for rows in hierarchies.values():
        if not isinstance(rows, list) or not rows:
            raise ValueError('Native mesh plan has no grid levels')
        levels = []
        for row in rows:
            if (type(row.get('level')) is not int or not 1 <= row['level'] <= 20
                    or not isinstance(row.get('shape'), list) or len(row['shape']) != 3
                    or any(type(n) is not int or n < 1 for n in row['shape'])
                    or not isinstance(row.get('offset'), list) or len(row['offset']) != 3
                    or any(type(n) is not int for n in row['offset'])):
                raise ValueError('Native mesh plan has invalid grid dimensions')
            levels.append(row['level'])
        if levels != list(range(levels[0], levels[-1] + 1)):
            raise ValueError('Native mesh levels are missing, repeated or unordered')
    poisson, convolution = hierarchies['poisson'], hierarchies['convolution']
    if poisson[-1]['level'] != convolution[-1]['level']:
        raise ValueError('Poisson and convolution finest levels disagree')
    for rows in (poisson, convolution):
        if rows[0]['shape'] != [2**rows[0]['level']] * 3:
            raise ValueError('Native base grid must cover the complete periodic cube')
    scalar, boundary, margin = plan['scalar_bytes'], plan['boundary_cells'], plan['convolution_margin']
    poisson_scalar=plan.get('poisson_scalar_bytes',scalar)
    if type(poisson_scalar) is not int or poisson_scalar not in (scalar,16) or poisson_scalar<scalar:
        raise ValueError('Unsupported native Poisson potential scalar storage')
    # GridHierarchy includes multigrid levels below the full-domain base grid.
    poisson_cells = sum(math.prod(n + 2 * boundary for n in row['shape']) for row in poisson)
    poisson_cells += sum((2**level + 2 * boundary)**3 for level in range(poisson[0]['level']))
    fft_cells, work = 0, 0.
    for index, row in enumerate(convolution):
        dims = row['shape'] if index == 0 else [n + 2*margin if margin > 0 else 2*n for n in row['shape']]
        # DensityGrid uses an in-place real-to-complex FFT's padded last axis.
        cells = dims[0] * dims[1] * (2 * (dims[2] // 2 + 1))
        fft_cells += cells
        work += cells * math.log2(max(2, math.prod(dims)))
    raw_cells = sum(math.prod(row['shape']) for row in poisson)
    noise_cells = sum(math.prod(row['shape']) for row in convolution)
    # Two fine fluids and combined coarse matter cannot exceed two particles
    # per Poisson cell. This intentionally overcounts covered parent cells.
    particle_upper = 2 * raw_cells
    payload_upper = 216 * particle_upper  # Eight retained vector fields, uint64 IDs, widths and masses
    noise_bytes = scalar * noise_cells
    estimated_ram = 64*1024**2 + 12*poisson_scalar*poisson_cells + scalar*8*fft_cells + noise_bytes + payload_upper
    estimated_disk = 64*1024**2 + noise_bytes + 3*payload_upper
    archive_bytes=plan.get('source_archive_estimated_bytes',0)
    archive_potential_scalar=plan.get('source_archive_potential_scalar_bytes',8)
    if type(archive_potential_scalar) is not int or archive_potential_scalar not in (8,16):
        raise ValueError('Unsupported native archived potential scalar storage')
    if type(archive_bytes) is not int or archive_bytes<0:
        raise ValueError('Exact nonnegative native source archive size required')
    if archive_bytes:
        expected_archive=24+14*28+sum(14*52+8*8*math.prod(row['shape'])+
            archive_potential_scalar*6*math.prod(n+2*boundary for n in row['shape']) for row in poisson)
        if archive_bytes!=expected_archive:raise ValueError('Native source archive estimate differs from actual planned hierarchy')
        estimated_ram+=64*1024
        estimated_disk+=archive_bytes
    return {'estimated_memory_bytes': estimated_ram, 'estimated_disk_bytes': estimated_disk,
            'source_archive_estimated_bytes':archive_bytes,
            'source_archive_potential_scalar_bytes':archive_potential_scalar,
            'single_poisson_field_bytes': poisson_scalar*poisson_cells,
            'poisson_scalar_bytes':poisson_scalar,
            'single_convolution_field_bytes': scalar*fft_cells,
            'noise_cache_bytes': noise_bytes, 'particle_count_upper_bound': particle_upper,
            'fft_cell_log2_work': work,
            'coefficients': {'poisson_field_copies': 12, 'fft_field_copies': 8,
                             'particle_payload_bytes': 216, 'disk_particle_payload_copies': 3,
                             'fixed_overhead_bytes': 64*1024**2},
            'estimate_scope': 'geometry-based working-set estimate; allocator/FFT/native peak RSS needs measured calibration',
            'time_status': 'NOT_CALIBRATED', 'estimated_wall_seconds': None}


def _music_seed_stream_ranges(seeds, *, noise_levels=(), cube_size=32, coarse_ngenic=False):
    """Reject repeated raw mt19937 cube streams before hierarchy blending."""
    if type(cube_size) is not int or cube_size<1:
        raise ValueError('MUSIC random cube size must be positive')
    ranges=[]
    levels=sorted(map(int,seeds))
    for level in levels:
        if level in noise_levels or (coarse_ngenic and level==levels[0]):continue
        seed=seeds[str(level)]
        if type(seed) is not int or seed<=0:
            raise ValueError('MUSIC seeds must be positive exact integers')
        cubes=max(1,2**level//cube_size)
        last=seed+cubes**3-1
        if last>2**32-1:
            raise ValueError('MUSIC cube seed range exceeds unique mt19937 32-bit streams')
        if any(seed<=row['last_seed'] and row['first_seed']<=last for row in ranges):
            raise ValueError('MUSIC refinement levels reuse raw Gaussian cube seed streams')
        ranges.append({'level':level,'first_seed':seed,'last_seed':last,'raw_cube_count':cubes**3})
    return ranges


def _music_noise_files(cfg, configuration):
    files={}
    for level in range(cfg.getint('setup','levelmin'),cfg.getint('setup','levelmax')+1):
        value=cfg.get('random',f'seed[{level}]',fallback='1')
        try:int(value)
        except ValueError:
            path=Path(value).expanduser()
            files[level]=(Path(configuration).resolve().parent/path).resolve()
    return files


def _frozen_noise_workspace(cfg, files):
    low,high=cfg.getint('setup','levelmin'),cfg.getint('setup','levelmax')
    if not any(level>low for level in files):return 0
    highest_frozen=max(files)
    if set(files)!=set(range(low,highest_frozen+1)):
        raise ValueError('Frozen fine noise requires a contiguous file-backed parent hierarchy')
    if highest_frozen<high:
        # A retained globally constrained prefix can feed the existing local
        # seeded refinement constructor. Its full-field workspace stops at
        # the last file, rather than allocating the effective finest level.
        seeds={str(level):cfg.getint('random',f'seed[{level}]') for level in range(highest_frozen+1,high+1)}
        ranges=_music_seed_stream_ranges(seeds,cube_size=cfg.getint('random','cubesize',fallback=32))
        previous=[]
        for level in sorted(files):
            first=cfg.getint('random',f'frozen_stream_first[{level}]')
            last=cfg.getint('random',f'frozen_stream_last[{level}]')
            if not 0<first<=last<2**32:
                raise ValueError('Explicit retained frozen stream interval required')
            if any(first<=end and begin<=last for begin,end in previous):
                raise ValueError('Retained frozen noise stream intervals overlap')
            if any(first<=row['last_seed'] and row['first_seed']<=last for row in ranges):
                raise ValueError('Seeded refinement overlaps a retained frozen noise stream')
            previous.append((first,last))
    # Two full fields plus padded FFT workspaces and fixed overhead. Local
    # convolution meshes do not bound these full periodic file allocations.
    n=2**highest_frozen
    return 64*1024**2+32*(n**3+(n//2)**3)


@directory_owner('destination')
def plan_native_music(configuration, binary, destination, *, memory_gib, disk_gib=None, threads=2):
    destination = Path(destination).resolve()
    configuration, binary = Path(configuration).resolve(), Path(binary).resolve()
    if (isinstance(memory_gib, bool) or not math.isfinite(memory_gib) or memory_gib <= 0
            or disk_gib is not None and (isinstance(disk_gib, bool) or not math.isfinite(disk_gib) or disk_gib <= 0)
            or type(threads) is not int or threads < 1):
        raise ValueError('Declare positive native memory/disk budgets and integer threads')
    cfg=_configuration(configuration)
    noise_files=_music_noise_files(cfg,configuration)
    seeds={str(level):int(cfg.get('random',f'seed[{level}]'))
           for level in range(cfg.getint('setup','levelmin'),cfg.getint('setup','levelmax')+1)
           if level not in noise_files and cfg.has_option('random',f'seed[{level}]')}
    _music_seed_stream_ranges(seeds,noise_levels=noise_files,
        cube_size=cfg.getint('random','cubesize',fallback=32),
        coarse_ngenic=(str(cfg.getint('setup','levelmin')) in seeds
                       and cfg.getboolean('random','coarse_ngenic',fallback=False)))
    frozen_workspace=_frozen_noise_workspace(cfg,noise_files)
    if frozen_workspace>memory_gib*1024**3:
        raise MemoryError('Frozen full noise hierarchy/FFT exceeds declared memory budget')
    from evolution import _noise_shape
    for level,path in noise_files.items():_noise_shape(path,2**level)
    hashes = {str(p): file_sha256(p) for p in (configuration, binary,*noise_files.values())}
    plan_path, log_path = destination/'native-mesh.json', destination/'native-mesh.log'
    if plan_path.exists() or log_path.exists() or (destination/'resource-plan.json').exists():
        raise FileExistsError('Refusing to overwrite native resource planning evidence')
    destination.mkdir(parents=True, exist_ok=True)
    staged_configuration = destination/'planner.conf'
    atomic_copy(configuration, staged_configuration)
    if file_sha256(staged_configuration) != hashes[str(configuration)]:
        raise ValueError('Native mesh configuration changed during staging')
    with log_path.open('x') as output:
        result = run_native_command([str(binary), 'planner.conf', '--plan-mesh', 'native-mesh.json'],
                                cwd=destination, receipt_path=destination/'native-invocation.json',
                                check=False, timeout=60., env={**os.environ, 'OMP_NUM_THREADS': str(threads)},
                                stdout=output, stderr=subprocess.STDOUT)
    if result.returncode != 0 or not plan_path.is_file():
        raise RuntimeError('Native binary cannot provide a mesh plan; see '+str(log_path))
    plan = json.loads(plan_path.read_text())
    if cfg.getboolean('output','retain_source_hierarchy',fallback=False) and not plan.get('source_archive_estimated_bytes'):
        raise ValueError('Native planner must account for requested retained source hierarchy')
    from core.music.resources import rectangular_particle_census
    census=rectangular_particle_census(plan,configuration)
    header_counts=None
    if census['status']=='EXACT_GEOMETRY_CENSUS':
        from core.music.resources import native_header_count_requirements
        header_counts=native_header_count_requirements(census,configuration)
        hashes.update(header_counts['source_hashes'])
        if header_counts['status']!='SUPPORTED_ENCODING':raise ValueError('Native output header cannot represent the planned species counts')
    estimates = estimate_mesh_resources(plan)
    estimates['header_count_contract']=header_counts
    from core.music.resources import estimate_music_phase_resources
    phase_estimate=estimate_music_phase_resources(plan,configuration,threads=threads)
    estimates['phase_generation']=phase_estimate
    estimates['phase_generation_working_set_bytes']=phase_estimate['phase_peak_working_set_bytes']
    estimates['estimated_memory_bytes']+=phase_estimate['phase_peak_working_set_bytes']
    estimates['estimated_disk_bytes']+=max(0,phase_estimate['cache_storage_bytes']-estimates['noise_cache_bytes'])
    estimates['phase_allowance_policy']='Conservative addition to bulk mesh/FFT allowances; overlapping payloads are not subtracted'
    hashes.update(phase_estimate['source_hashes'])
    if cfg.get('poisson','hybrid_source_extension',fallback='none')!='none':
        settings={'hybrid_source_extension':cfg.get('poisson','hybrid_source_extension')}
        for key,default in [('hybrid_extension_degree',5),('hybrid_extension_boundary',32),
                            ('hybrid_extension_memory_mib',2048),('hybrid_extension_max_side',256)]:
            settings[key]=cfg.getint('poisson',key,fallback=default)
        validate_bounded_source_settings(settings,lpt_order=2 if cfg.getboolean('setup','use_2LPT',fallback=False) else 1)
        if not cfg.getboolean('cosmology','mixed_2lpt',fallback=False) or cfg.getboolean('poisson','kspace',fallback=False):
            raise ValueError('Bounded source planner requires finite-difference mixed 2LPT')
        from core.music.resources import hybrid_workspace_requirements
        hybrid=hybrid_workspace_requirements(plan,settings)
        estimates['bounded_source_requirements']=hybrid
        hashes.update(hybrid['source_hashes'])
        rows=plan['hierarchies']['poisson'];side=max(rows[-1]['shape'])+2*settings['hybrid_extension_boundary']
        workspace=settings['hybrid_extension_memory_mib']*1024**2
        if side>settings['hybrid_extension_max_side'] or side**3*64>workspace//2:
            raise MemoryError('Bounded hybrid FFT geometry exceeds declared workspace')
        if sum(math.prod(row['shape']) for row in rows)*24>workspace//6:
            raise MemoryError('Bounded hybrid source capture exceeds declared workspace')
        estimates['bounded_source_workspace_bytes']=workspace
        estimates['estimated_memory_bytes']+=workspace
    estimates['frozen_full_noise_workspace_bytes']=frozen_workspace
    estimates['estimated_memory_bytes']+=frozen_workspace
    free = shutil.disk_usage(destination).free
    disk_budget = min(free, int(disk_gib*1024**3)) if disk_gib is not None else free
    report = {**estimates, 'native_geometry': plan, 'particle_census':census, 'status': 'PASS',
              'budget_memory_bytes': int(memory_gib*1024**3), 'budget_disk_bytes': disk_budget,
              'available_disk_bytes': free, 'input_hashes': hashes,
              'estimator_source_sha256': file_sha256(Path(__file__)),
              'native_plan_sha256': file_sha256(plan_path), 'native_log_sha256': file_sha256(log_path),
              'science_acceptance': 'NOT_ACCEPTED'}
    if any(file_sha256(p) != expected for p, expected in hashes.items()):
        report.update(status='FAIL', reason='Native planner input changed')
    elif estimates['estimated_memory_bytes'] > report['budget_memory_bytes']:
        report.update(status='FAIL', reason='Estimated native MUSIC working set exceeds declared memory budget')
    elif estimates['estimated_disk_bytes'] > disk_budget:
        report.update(status='FAIL', reason='Estimated native MUSIC files exceed declared/available disk budget')
    atomic_json(destination/'resource-plan.json', report)
    if report['status'] != 'PASS':
        raise MemoryError(report['reason'])
    return report

# ========================================================================
# resolution_budget: Observable resolution budgets from measured mass moments in proper apertures.

from core.foundation import SECONDS_PER_YEAR

G = 1000 * G_MPC_KMS2_MSUN
KMS_MYR_KPC = 1e5 * SECONDS_PER_YEAR * 1e6 / KPC_CM
KINDS = {'halo_profile', 'cold_stream', 'first_stars', 'bh_spike', 'sidm_core'}


def positive(value, name, *, zero=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)) or not math.isfinite(float(value)) or (value < 0 if zero else value <= 0):
        raise ValueError(name + ' must be a finite ' + ('nonnegative' if zero else 'positive') + ' number')
    return float(value)


def radial_edges(values):
    edges = np.asarray(values, dtype=float)
    if edges.ndim != 1 or not 2 <= len(edges) <= 4097 or edges[0] != 0 or not np.isfinite(edges).all() or np.any(np.diff(edges) <= 0):
        raise ValueError('Radial edges must increase from zero, with 1 to 4096 bins')
    return edges


def validate_resolution_request(request):
    required = {'schema', 'radial_edges_proper_kpc', 'observables', 'maximum_coarse_mass_fraction', 'minimum_relaxation_age_ratio', 'maximum_timestep_crossing_fraction', 'maximum_force_support_fraction', 'refinement'}
    if not isinstance(request, dict) or not required <= set(request) or set(request)-required-{'sidm_limits','physical_support'} or request['schema'] != 'observable-resolution-request-v1':
        raise ValueError('Unsupported or incomplete observable resolution request')
    edges = radial_edges(request['radial_edges_proper_kpc'])
    for key in ('maximum_coarse_mass_fraction', 'maximum_timestep_crossing_fraction', 'maximum_force_support_fraction'):
        if positive(request[key], key, zero=key == 'maximum_coarse_mass_fraction') > 1:
            raise ValueError(key + ' must be <= 1')
    positive(request['minimum_relaxation_age_ratio'], 'minimum_relaxation_age_ratio')
    obs = request['observables']
    if not isinstance(obs, list) or not obs:
        raise ValueError('Declare at least one resolution observable')
    names = set()
    for item in obs:
        if not isinstance(item,dict):
            raise ValueError('Observable declaration must be an object')
        common = {'name', 'kind', 'minimum_radius_proper_kpc', 'minimum_members', 'species', 'maximum_relative_error', 'absolute_error_floor', 'units'}
        extra = {'width_proper_kpc', 'dispersion_km_s', 'duration_myr', 'maximum_heating_variance_fraction'} if item.get('kind') == 'cold_stream' else set()
        if item.get('kind') in {'first_stars','bh_spike'}:extra={'claim_scope'}
        if set(item) != common | extra or item['kind'] not in KINDS:
            raise ValueError('Malformed observable resolution declaration')
        if not isinstance(item['name'], str) or not item['name'] or item['name'] in names:
            raise ValueError('Observable names must be nonempty and unique')
        names.add(item['name'])
        if type(item['minimum_members']) is not int or item['minimum_members'] < 2 or type(item['species']) is not int or item['species'] not in range(6):
            raise ValueError('Declare a native species and minimum_members >= 2')
        if item['kind']=='first_stars' and (item['claim_scope']!='primordial_cooling_environment' or item['species']!=0):
            raise ValueError('First-cloud resolution applies to native gas and the declared primordial cooling environment')
        if item['kind']=='bh_spike' and (item['claim_scope'] not in {'resolved_dm_profile','collisionless_adiabatic_spike','weakly_collisional_sidm_cusp'} or item['species']!=1):
            raise ValueError('BH-profile resolution requires fine native DM and an explicit physical claim scope')
        if item['kind'] in {'first_stars','bh_spike'} and item['minimum_radius_proper_kpc'] not in edges[1:]:
            raise ValueError('Physical target resolution radius must be an actual measured radial edge')
        for key in ('minimum_radius_proper_kpc', 'maximum_relative_error', 'absolute_error_floor', *(extra-{'claim_scope'})):
            positive(item[key], key)
        if item['minimum_radius_proper_kpc'] > edges[-1] or not isinstance(item['units'], str) or not item['units']:
            raise ValueError('Observable radius must lie in measured aperture and units must be explicit')
    support=request.get('physical_support',{})
    if not isinstance(support,dict) or set(support)-names:
        raise ValueError('Physical support must name declared resolution observables')
    for name,reference in support.items():
        if set(reference)!={'path','sha256','memory_gib'} or not isinstance(reference['path'],str) or not isinstance(reference['sha256'],str) or len(reference['sha256'])!=64:
            raise ValueError('Physical support requires a retained product, SHA256 and bounded workspace')
        positive(reference['memory_gib'],'physical support intake workspace')
        if next(row for row in obs if row['name']==name)['kind'] not in {'first_stars','bh_spike'}:
            raise ValueError('Physical target evidence applies to first-cloud and BH-profile observables')
    refinement = request['refinement']
    if not isinstance(refinement, dict) or set(refinement) != {'maximum_levels', 'mass_reduction_factor', 'softening_reduction_factor', 'timestep_reduction_factor', 'memory_gib', 'disk_gib'}:
        raise ValueError('Declare bounded high-resolution follow-up limits')
    if type(refinement['maximum_levels']) is not int or not 0 <= refinement['maximum_levels'] <= 20:
        raise ValueError('maximum_levels must be 0 to 20')
    for key in ('mass_reduction_factor', 'softening_reduction_factor', 'timestep_reduction_factor'):
        if positive(refinement[key], key) <= 1:
            raise ValueError('Refinement reductions must be > 1')
    for key in ('memory_gib', 'disk_gib'):
        positive(refinement[key], key)
    return request


def empty_census(edges):
    edges = radial_edges(edges)
    n = len(edges) - 1
    return {'schema': 'radial-particle-census-v1', 'edges_proper_kpc': edges.tolist(),
            'count': np.zeros((6, n), dtype=np.int64), 'mass_msun': np.zeros((6, n)),
            'mass_squared_msun2': np.zeros((6, n)), 'minimum_mass_msun': np.full((6, n), np.inf),
            'maximum_mass_msun': np.zeros((6, n)), 'maximum_softening_proper_kpc': np.zeros((6, n)),
            'coarse_mass_msun': np.zeros(n)}


def accumulate_census(census, radii, masses, species, softening, *, coarse=None):
    """Bounded chunk accumulator; preserves unequal-mass second moments."""
    r, m, eps = (np.asarray(v, dtype=float) for v in (radii, masses, softening))
    if r.ndim != 1 or m.shape != r.shape or eps.shape != r.shape or type(species) is not int or species not in range(6):
        raise ValueError('Census chunk arrays must be matching one-dimensional vectors')
    if not all(np.isfinite(v).all() for v in (r, m, eps)) or np.any(r < 0) or np.any(m <= 0) or np.any(eps <= 0):
        raise ValueError('Census needs finite radii and positive masses/softenings')
    edges = np.asarray(census['edges_proper_kpc'])
    bins = np.searchsorted(edges, r, side='left') - 1
    bins[r == 0] = 0
    keep = (bins >= 0) & (bins < len(edges) - 1)
    b, m, eps = bins[keep], m[keep], eps[keep]
    for key, weights in [('count', np.ones(len(m), dtype=np.int64)), ('mass_msun', m), ('mass_squared_msun2', m*m)]:
        np.add.at(census[key][species], b, weights)
    np.minimum.at(census['minimum_mass_msun'][species], b, m)
    np.maximum.at(census['maximum_mass_msun'][species], b, m)
    np.maximum.at(census['maximum_softening_proper_kpc'][species], b, eps)
    if coarse is not None:
        mask = np.asarray(coarse)
        if mask.dtype.kind != 'b' or mask.shape != r.shape:
            raise ValueError('Coarse membership must be a matching boolean mask')
        np.add.at(census['coarse_mass_msun'], b[mask[keep]], m[mask[keep]])


def seal_census(census):
    result = dict(census)
    for key, value in result.items():
        if isinstance(value, np.ndarray):
            result[key] = [[None if not math.isfinite(float(x)) else x.item() for x in row] for row in value] if value.ndim == 2 else value.tolist()
    return result


def measure_actual_mass_levels(particle_census):
    if particle_census.get('schema') != 'radial-particle-census-v1':
        raise ValueError('Actual radial mass census is required; particle counts alone are insufficient')
    edges = radial_edges(particle_census['edges_proper_kpc'])
    shape = (6, len(edges)-1)
    arrays = {}
    for key in ('count', 'mass_msun', 'mass_squared_msun2', 'maximum_mass_msun', 'maximum_softening_proper_kpc'):
        v = np.asarray(particle_census[key])
        if v.shape != shape or not np.isfinite(v).all() or np.any(v < 0):
            raise ValueError('Invalid radial census ' + key)
        if key == 'count' and (v.dtype.kind not in 'iu'):
            raise ValueError('Census counts must be exact integers')
        arrays[key] = v
    count, mass, square = (arrays[k] for k in ('count', 'mass_msun', 'mass_squared_msun2'))
    present = count > 0
    if np.any((mass > 0) != present) or np.any((square > 0) != present) or np.any(present & (arrays['maximum_softening_proper_kpc'] <= 0)):
        raise ValueError('Mass census has inconsistent occupied bins')
    if np.any(square * count + 1e-10 * mass**2 < mass**2) or np.any(square > mass * arrays['maximum_mass_msun'] * (1+1e-10)):
        raise ValueError('Mass moments violate positive-mass bounds')
    total = mass.sum(axis=0).cumsum()
    squared = square.sum(axis=0).cumsum()
    effective = np.divide(total**2, squared, out=np.zeros_like(total), where=squared > 0)
    coarse = np.asarray(particle_census['coarse_mass_msun'], dtype=float)
    if coarse.shape != (shape[1],) or not np.isfinite(coarse).all() or np.any(coarse < 0) or np.any(coarse > mass.sum(axis=0)*(1+1e-10)):
        raise ValueError('Invalid coarse mass census')
    gravity=positive(particle_census.get('gravity_constant_kpc_kms2_msun',G),'gravity_constant')
    return {'edges': edges, **arrays, 'enclosed_mass': total, 'enclosed_effective_count': effective,'gravity_constant':gravity,
            'enclosed_coarse_fraction': np.divide(coarse.cumsum(), total, out=np.ones_like(total), where=total > 0)}


def estimate_relaxation_and_force_limits(target, mass_levels):
    age = positive(target['age_myr'], 'age_myr')
    step = positive(target['maximum_proper_timestep_myr'], 'maximum_proper_timestep_myr')
    r, mass, n = mass_levels['edges'][1:], mass_levels['enclosed_mass'], mass_levels['enclosed_effective_count']
    vc = np.sqrt(mass_levels['gravity_constant'] * mass / r)
    crossing = np.divide(r, vc * KMS_MYR_KPC, out=np.full_like(r, np.inf), where=vc > 0)
    relaxation = np.zeros_like(r)
    ok = n > 1
    relaxation[ok] = n[ok] / (8*np.log(n[ok])) * crossing[ok]
    support = 2.8 * np.maximum.accumulate(mass_levels['maximum_softening_proper_kpc'].max(axis=0))
    return {'radius_proper_kpc': r.tolist(), 'effective_count': n.tolist(),
            'relaxation_age_ratio_estimate': (relaxation / age).tolist(),
            'timestep_crossing_fraction': (step / crossing).tolist(),
            'force_support_proper_kpc': support.tolist(),
            'scope': 'Power et al. Eq.20 equal-mass scaling with mass-moment effective N; mixed-mass substitution is a preflight approximation, not empirical convergence'}


def estimate_then_calibrate_stream_heating(target, mass_levels):
    """Weak-encounter impulse diffusion, with explicit finite impact limits.

    Sum rho_j*m_j rather than mean particle mass. Uses total three-dimensional
    velocity variance; no arbitrary division by three for anisotropic streams.
    b_min=max(native spline support, G*m_max/v^2), b_max=local shell width.
    This homogeneous, straight-encounter approximation excludes collective
    noise and physical substructure. Measured null-stream controls are mandatory.
    """
    speed = positive(target['relative_speed_km_s'], 'relative_speed_km_s')
    edges = mass_levels['edges']
    volume = 4*np.pi/3 * np.diff(edges**3)
    gravity=mass_levels['gravity_constant']
    bmin = np.maximum(2.8*mass_levels['maximum_softening_proper_kpc'], gravity*mass_levels['maximum_mass_msun']/speed**2)
    bmax = np.diff(edges)[None, :]
    log_lambda = np.log(np.maximum(1, np.divide(bmax, bmin, out=np.ones_like(bmin), where=bmin > 0)))
    rate = (8*np.pi*gravity**2/speed * mass_levels['mass_squared_msun2']/volume * log_lambda).sum(axis=0) * KMS_MYR_KPC
    return {'variance_rate_km2_s2_per_myr_estimate': rate.tolist(), 'impact_logarithm': log_lambda.tolist(),
            'scope': 'Local weak-encounter impulse estimate; actual heating acceptance requires matched null-stream resolution controls'}


def stream_heating_statistics(granular_velocity, smooth_velocity, masses_msun):
    """Measure matched cumulative kicks in peculiar km/s, preserving bulk drift.

    The mass-weighted population covariance has no sample Bessel correction.
    The total second moment includes coherent drift, so subtraction of a mean
    cannot hide a systematic acceleration error from the heating budget.
    This measures a paired control discrepancy; it is not a heat-flow estimator.
    """
    granular = np.asarray(granular_velocity, dtype=float)
    smooth = np.asarray(smooth_velocity, dtype=float)
    mass = np.asarray(masses_msun, dtype=float)
    if (granular.ndim != 3 or granular.shape != smooth.shape or granular.shape[2] != 3
            or granular.shape[0] < 2 or granular.shape[1] < 3
            or mass.shape != (granular.shape[1],) or not np.isfinite(granular).all()
            or not np.isfinite(smooth).all() or not np.isfinite(mass).all() or np.any(mass <= 0)):
        raise ValueError('Heating requires finite matched epoch/particle/3 velocities and positive masses')
    weights = mass / mass.sum()
    if not np.isfinite(weights).all() or weights.sum() <= 0:
        raise ValueError('Heating mass normalization overflowed')
    difference = granular - smooth
    kicks = difference - difference[:1]
    drift = np.einsum('n,tnj->tj', weights, kicks)
    centered = kicks - drift[:, None, :]
    covariance = np.einsum('n,tni,tnj->tij', weights, centered, centered)
    incoherent = np.trace(covariance, axis1=1, axis2=2)
    total = incoherent + np.sum(drift**2, axis=1)
    dispersion = []; means=[]; velocity_covariances=[]
    for velocity in (granular, smooth):
        mean = np.einsum('n,tnj->tj', weights, velocity)
        deviation = velocity-mean[:, None]
        means.append(mean)
        velocity_covariances.append(np.einsum('n,tni,tnj->tij',weights,deviation,deviation))
        dispersion.append(np.einsum('n,tnj,tnj->t', weights, deviation, deviation))
        del deviation
    excess = (dispersion[0]-dispersion[0][0]) - (dispersion[1]-dispersion[1][0])
    if not all(np.isfinite(value).all() for value in (kicks, drift, covariance, total, excess,*means,*velocity_covariances)):
        raise ValueError('Heating statistics overflowed')
    return {'matched_kicks_km_s': kicks, 'mean_kick_km_s': drift,
            'kick_covariance_km2_s2': covariance, 'incoherent_variance_km2_s2': incoherent,
            'heating_variance_km2_s2': total, 'signed_dispersion_excess_km2_s2': excess,
            'granular_mean_velocity_km_s':means[0], 'smooth_mean_velocity_km_s':means[1],
            'granular_velocity_covariance_km2_s2':velocity_covariances[0],
            'smooth_velocity_covariance_km2_s2':velocity_covariances[1]}


def _read_stream_control_population(files, selected_ids, species, namespace, memory_gib):
    """Read frozen exact identities in bounded chunks; reordering has no meaning."""
    from core.data import exact_uint64_ids, native_unit_factor, dataset_to_cgs_factor, snapshot_total_counts
    from core.data import verify_native_hdf5_set
    paths = [Path(p).resolve() for p in files]
    publication = verify_native_hdf5_set(paths, require_publication=True)
    _unit_conversion(paths)
    sources = {str(p): file_sha256(p) for p in paths}
    ids = exact_uint64_ids(selected_ids, allow_empty=False)
    budget = positive(memory_gib, 'stream memory_gib') * 1024**3
    if len(ids)*160 + 16*1024**2 >= budget:
        raise MemoryError('Frozen stream population exceeds declared memory budget')
    chunk = max(1, min(65536, int((budget-len(ids)*160-16*1024**2)/128)))
    order = np.argsort(ids); ordered = ids[order]
    velocity = np.zeros((len(ids), 3)); position = np.zeros((len(ids), 3))
    mass = np.zeros(len(ids)); seen = np.zeros(len(ids), dtype=bool)
    reference = None; census = np.zeros(6, dtype=object)
    for path in paths:
        with h5py.File(path, 'r') as native:
            header = native['Header'].attrs; params = native['Parameters'].attrs
            a = positive(header['Time'], 'actual stream epoch'); h = positive(params['HubbleParam'], 'HubbleParam')
            if a > 1 or not math.isclose(float(header['Redshift']), 1/a-1, rel_tol=0, abs_tol=1e-8):
                raise ValueError('Stream snapshot redshift/epoch mismatch')
            counts = np.asarray(header['NumPart_ThisFile'])
            totals = tuple(map(int, snapshot_total_counts(header)))
            if counts.shape != (6,) or counts.dtype.kind not in 'iu' or np.any(counts < 0):
                raise ValueError('Stream snapshot requires exact native particle counts')
            current = (a, h, positive(header['BoxSize'], 'BoxSize'), totals,
                       tuple(float(params[k]) for k in ('Omega0', 'OmegaLambda', 'OmegaBaryon')))
            if reference is not None and current != reference:
                raise ValueError('Stream shards differ in actual epoch, cosmology, box or census')
            reference = current; census += counts.astype(object)
            group = native.get(f'PartType{species}')
            count = int(counts[species])
            if not count:
                if group is not None and len(group['ParticleIDs']):
                    raise ValueError('Stream group disagrees with zero native census')
                continue
            if group is None or group['ParticleIDs'].shape != (count,) or group['Velocities'].shape != (count, 3) or group['Coordinates'].shape != (count, 3):
                raise ValueError('Stream group lengths differ from the native census')
            saved_namespace = group['ParticleIDs'].attrs.get('identity_namespace')
            if isinstance(saved_namespace, bytes): saved_namespace = saved_namespace.decode()
            if saved_namespace is not None and saved_namespace != namespace:
                raise ValueError('Stream identity namespace differs from the frozen population')
            factor = native_unit_factor('particle_velocity_peculiar_kms', scale_factor=a, hubble_param=h)
            saved_factor = dataset_to_cgs_factor(group['Velocities'].attrs, a, h)
            if saved_factor is None or not math.isclose(saved_factor, factor*1e5, rel_tol=1e-10):
                raise ValueError('Stream peculiar velocity convention differs from the pinned native convention')
            saved_length = dataset_to_cgs_factor(group['Coordinates'].attrs, a, h)
            if saved_length is None or not math.isclose(saved_length, KPC_CM*a/h, rel_tol=1e-10):
                raise ValueError('Stream coordinate convention differs from the pinned native convention')
            constant = float(header['MassTable'][species])
            if constant == 0 and group['Masses'].shape != (count,):
                raise ValueError('Stream masses differ from the native census')
            if constant == 0:
                saved_mass = dataset_to_cgs_factor(group['Masses'].attrs, a, h)
                if saved_mass is None or not math.isclose(saved_mass, 1e10*MSUN_G/h, rel_tol=1e-10):
                    raise ValueError('Stream mass convention differs from the pinned native convention')
            for start in range(0, count, chunk):
                values = exact_uint64_ids(group['ParticleIDs'][start:start+chunk])
                slots = np.searchsorted(ordered, values)
                present = slots < len(ordered)
                present[present] &= ordered[slots[present]] == values[present]
                rows = np.flatnonzero(present); targets = order[slots[present]]
                if np.any(seen[targets]):
                    raise ValueError('Duplicate stream identity across chunks or shards')
                if not len(rows): continue
                v = np.asarray(group['Velocities'][start:start+chunk], dtype=float)[rows]*factor
                xyz = np.asarray(group['Coordinates'][start:start+chunk], dtype=float)[rows]
                m = (np.asarray(group['Masses'][start:start+chunk], dtype=float)[rows] if constant == 0
                     else np.full(len(rows), constant))*native_unit_factor('mass_msun', scale_factor=a, hubble_param=h)
                if not np.isfinite(v).all() or not np.isfinite(xyz).all() or not np.isfinite(m).all() or np.any(m <= 0):
                    raise ValueError('Stream population has invalid velocity or mass')
                velocity[targets] = v; position[targets] = xyz; mass[targets] = m; seen[targets] = True
    if tuple(map(int, census)) != reference[3] or not seen.all():
        raise ValueError('Incomplete stream shard census or missing frozen stream identities')
    for path in paths:
        marker = path.parent / (path.name.split('.')[0]+'.publication.json')
        sources[str(marker)] = file_sha256(marker)
    verify_native_hdf5_set(paths, require_publication=True)
    if any(file_sha256(p) != value for p, value in sources.items()):
        raise ValueError('Stream sources changed during population measurement')
    return velocity, mass, reference, sources, publication, position


@directory_owner('destination')
def publish_stream_heating_calibration(request, destination):
    """Produce the measured null-stream product consumed by resolution intake.

    A frozen population must survive every snapshot in both native runs. Actual
    epochs and a hashed expansion clock determine elapsed proper time. Native
    invocation receipts must bind the shared declared smooth-potential asset.
    The result remains NOT_ACCEPTED until independent numerical comparisons pass.
    """
    from core.data import exact_uint64_ids, export_numeric_audit
    from core.cosmology import run_proper_time_clock, interpolate_proper_time
    from core.native import verify_invocation_inputs
    required = {'schema', 'observable', 'selection', 'identity_namespace', 'particle_ids', 'species',
                'bin_coordinate', 'physics_fingerprint', 'smooth_potential', 'granular', 'smooth',
                'maximum_initial_velocity_difference_km_s', 'maximum_initial_position_difference_proper_kpc', 'memory_gib'}
    if not isinstance(request, dict) or set(request) != required or request['schema'] != 'stream-heating-calibration-v1':
        raise ValueError('Declare the complete versioned matched native stream control')
    for key in ('observable', 'selection', 'identity_namespace', 'physics_fingerprint'):
        if not isinstance(request[key], str) or not request[key].strip(): raise ValueError('Missing stream '+key)
    species = request['species']
    if type(species) is not int or not 0 <= species < 6: raise ValueError('Invalid exact stream species')
    initial_limit = positive(request['maximum_initial_velocity_difference_km_s'], 'initial velocity difference', zero=True)
    initial_position_limit = positive(request['maximum_initial_position_difference_proper_kpc'], 'initial position difference', zero=True)
    bin_coordinate = positive(request['bin_coordinate'], 'bin coordinate', zero=True)
    ids = exact_uint64_ids(request['particle_ids'], allow_empty=False)
    if len(ids) < 3: raise ValueError('At least three frozen stream particles are required')
    memory = positive(request['memory_gib'], 'memory_gib')
    potential = request['smooth_potential']; potential_path = Path(potential['path']).resolve()
    if file_sha256(potential_path) != potential['sha256']: raise ValueError('Smooth-potential asset changed')
    sources = {str(potential_path): potential['sha256'], str(Path(__file__).resolve()): file_sha256(__file__)}
    histories = []; clocks = []; snapshots = []; initial_positions = []; synthetic = False
    for kind in ('granular', 'smooth'):
        run = request[kind]
        if set(run) != {'run_directory', 'snapshot_sets', 'execution_receipt'}:
            raise ValueError('Stream runs require complete epochs, execution receipt and run directory')
        sets = run['snapshot_sets']
        if not isinstance(sets, list) or len(sets) < 2 or any(not isinstance(s, list) or not s for s in sets):
            raise ValueError('Stream controls require at least two complete native epoch sets')
        if len(ids)*len(sets)*192 + 32*1024**2 > memory*1024**3:
            raise MemoryError('Stream trajectory statistics exceed declared memory budget')
        receipt = Path(run['execution_receipt']).resolve(); record = json.loads(receipt.read_text())
        if record.get('schema') != 'native-command-invocation-v1' or record.get('status') != 'RETURNED' or record.get('returncode') != 0:
            raise ValueError('Stream calibration requires successful native invocations')
        if Path(record['cwd']).resolve() != Path(run['run_directory']).resolve() or record['input_hashes'].get(str(potential_path)) != potential['sha256']:
            raise ValueError('Stream native invocation does not bind its run and shared smooth potential')
        verify_invocation_inputs(record); sources.update(record['input_hashes']); sources[str(receipt)] = file_sha256(receipt)
        clock = run_proper_time_clock(run['run_directory'])
        if clock[0] is None: raise ValueError('Stream calibration requires a verified expansion-based proper-time clock')
        manifest = Path(run['run_directory']).resolve()/'manifest.json'
        manifest_data = json.loads(manifest.read_text())
        synthetic = synthetic or bool(manifest_data.get('fixture') or manifest_data.get('synthetic_inputs'))
        sources[str(manifest)] = file_sha256(manifest); sources[clock[2]['path']] = clock[2]['sha256']
        values = []; references = []; masses = []
        for files in sets:
            velocity, mass, reference, hashes, _, position = _read_stream_control_population(files, ids, species, request['identity_namespace'], memory)
            if not values: initial_positions.append(position)
            sources.update(hashes); values.append(velocity); masses.append(mass); references.append(reference)
            snapshots.extend({'path': str(Path(p).resolve()), 'sha256': hashes[str(Path(p).resolve())]} for p in files)
        epochs = [row[0] for row in references]
        if any(b <= a for a, b in zip(epochs, epochs[1:])): raise ValueError('Stream actual epochs must increase strictly')
        if any((row[1], row[2], row[4]) != (references[0][1], references[0][2], references[0][4]) for row in references):
            raise ValueError('Stream history changes its cosmology or simulation box')
        if any(not np.array_equal(masses[0], m) for m in masses[1:]): raise ValueError('Stream tracer mass changes require a separately modeled lifecycle control')
        times = [1000*interpolate_proper_time(clock[0], clock[1], a) for a in epochs]
        histories.append((np.asarray(values), masses[0], references)); clocks.append(times)
        del values
    paired_references = [[(r[0], r[1], r[2], r[4]) for r in history[2]] for history in histories]
    if paired_references[0] != paired_references[1] or not np.allclose(clocks[0], clocks[1], rtol=1e-12, atol=0):
        raise ValueError('Granular/smooth controls differ in actual epochs, cosmology, box or clock')
    a, h, box = histories[0][2][0][:3]
    offset = (initial_positions[0]-initial_positions[1]+box/2) % box-box/2
    if np.max(np.linalg.norm(offset*a/h, axis=1)) > initial_position_limit:
        raise ValueError('Granular/smooth initial stream positions exceed the declared matching tolerance')
    if not np.array_equal(histories[0][1], histories[1][1]): raise ValueError('Granular/smooth tracer masses differ')
    if np.max(np.linalg.norm(histories[0][0][0]-histories[1][0][0], axis=1)) > initial_limit:
        raise ValueError('Granular/smooth initial stream velocities exceed the declared matching tolerance')
    statistics = stream_heating_statistics(histories[0][0], histories[1][0], histories[0][1])
    epochs = [row[0] for row in histories[0][2]]
    arrays = {**statistics, 'particle_ids': ids, 'mass_msun': histories[0][1],
              'actual_scale_factors': np.asarray(epochs), 'elapsed_proper_time_myr': np.asarray(clocks[0])-clocks[0][0]}
    specifications = {key: {'units': 'km/s' if key.endswith('km_s') else 'km^2/s^2',
                           'quantity': key.replace('_', ' ')} for key in statistics}
    specifications.update({'particle_ids': {'units': 'identity', 'quantity': 'frozen matched stream population', 'identity_namespace': request['identity_namespace']},
        'mass_msun': {'units': 'Msun', 'quantity': 'constant matched tracer masses'},
        'actual_scale_factors': {'units': 'dimensionless', 'quantity': 'actual saved native epochs'},
        'elapsed_proper_time_myr': {'units': 'Myr', 'quantity': 'elapsed verified expansion-based proper time'}})
    meta = {'origin': 'synthetic_control' if synthetic else 'native_simulation', 'control': 'matched_smooth_potential_null_stream', 'observable': request['observable'],
            'units': 'km^2/s^2', 'selection': request['selection'], 'actual_scale_factor': epochs[-1],
            'physics_fingerprint': request['physics_fingerprint'], 'duration_myr': float(arrays['elapsed_proper_time_myr'][-1]),
            'bin_coordinates': [bin_coordinate], 'all_requested_bins_valid': True, 'source_snapshots':
                [{'path': str(Path(p).resolve()), 'sha256': sources[str(Path(p).resolve())]} for p in request['granular']['snapshot_sets'][-1]]}
    # Resolution intake expects a bin vector at the final actual epoch.
    arrays['null_stream_heating'] = statistics['heating_variance_km2_s2'][-1:]
    specifications['null_stream_heating'] = {'units': 'km^2/s^2', 'quantity': 'final matched kick second moment including coherent drift'}
    history_epoch = {'representation': 'row-aligned actual native epochs', 'axis': 0,
                     'rows': [epoch_contract(a) for a in epochs]}
    for key in (*statistics, 'actual_scale_factors', 'elapsed_proper_time_myr'):
        specifications[key]['epoch'] = history_epoch
    report = {'schema': 'measured-stream-heating-v1', 'status': 'MEASURED', 'request': request,
        'resolution_measurement': meta, 'source_hashes': sources, 'all_control_snapshots': snapshots, 'synthetic_inputs': synthetic,
        'sampling_uncertainty': 'NOT_ESTIMATED; repeat independent matched realizations for calibration uncertainty',
        'dispersion_convention':'Mass-weighted 3D population covariance about the instantaneous bulk velocity; its trace includes velocity gradients across the selected population and is not an intrinsic transverse stream dispersion',
        'definition': 'mass-weighted cumulative granular-minus-smooth peculiar velocity kick second moment; signed dispersion growth retained separately'}
    return export_numeric_audit(destination, report, arrays, specifications, algorithm=__file__, source_hashes=sources,
        context={'epoch': epoch_contract(epochs[-1]), 'selection': request['selection'],
                 'coordinates': coordinate_contract(position_units='proper kpc', origin='matched control frame',
                    periodic_wrapping='native periodic box', velocity_convention='peculiar km/s; native sqrt(a) conversion')})


def verify_stream_heating_calibration(path):
    """Recheck both histories, source freshness and the measured kick budget."""
    from core.data import verify_numeric_audit
    path = Path(path).resolve(); report = verify_numeric_audit(path)
    if report.get('schema') != 'measured-stream-heating-v1' or report.get('status') != 'MEASURED':
        raise ValueError('A measured stream-heating producer product is required')
    for source, checksum in report['source_hashes'].items():
        if file_sha256(source) != checksum: raise ValueError('Stream calibration source changed: '+source)
    with np.load(path.parent/report['numeric_data']['directory']/'arrays.npz', allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    request = report['request']; meta = report['resolution_measurement']
    if (meta['observable'] != request['observable'] or meta['selection'] != request['selection']
            or meta['physics_fingerprint'] != request['physics_fingerprint']
            or arrays['particle_ids'].tolist() != request['particle_ids']
            or meta['bin_coordinates'] != [request['bin_coordinate']]):
        raise ValueError('Stream calibration changed its frozen selection or physics')
    histories = []; epochs = None
    for kind in ('granular', 'smooth'):
        values = []
        actual = []
        for files in request[kind]['snapshot_sets']:
            velocity, mass, reference, _, _, _ = _read_stream_control_population(
                files, arrays['particle_ids'], request['species'], request['identity_namespace'], request['memory_gib'])
            if not np.array_equal(mass, arrays['mass_msun']): raise ValueError('Stream calibration mass continuity differs')
            values.append(velocity); actual.append(reference[0])
        if epochs is not None and epochs != actual: raise ValueError('Stream calibration actual epochs differ')
        epochs = actual; histories.append(np.asarray(values))
    if not np.array_equal(arrays['actual_scale_factors'], epochs) or meta['actual_scale_factor'] != epochs[-1]:
        raise ValueError('Stream calibration metadata differs from saved actual epochs')
    from core.cosmology import run_proper_time_clock, interpolate_proper_time
    for kind in ('granular', 'smooth'):
        clock = run_proper_time_clock(request[kind]['run_directory'])
        times = np.asarray([1000*interpolate_proper_time(clock[0], clock[1], a) for a in epochs])
        if not np.array_equal(arrays['elapsed_proper_time_myr'], times-times[0]):
            raise ValueError('Stream calibration duration differs from the bound clock')
    if meta['duration_myr'] != arrays['elapsed_proper_time_myr'][-1]: raise ValueError('Stream calibration duration metadata changed')
    measured = stream_heating_statistics(histories[0], histories[1], arrays['mass_msun'])
    if any(not np.array_equal(arrays[key], value) for key, value in measured.items()):
        raise ValueError('Stream calibration statistics differ from actual native velocities')
    if not np.array_equal(arrays['null_stream_heating'], measured['heating_variance_km2_s2'][-1:]):
        raise ValueError('Stream heating budget differs from matched kicks')
    return report


def wdm_fragmentation_mask(wdm, object_masses):
    required = {'mean_density_msun_comoving_mpc3', 'spacing_comoving_mpc', 'k_peak_per_comoving_mpc', 'safety_factor'}
    if set(wdm) != required:
        raise ValueError('WDM needs density, local initial fine spacing, peak k and safety factor in unscaled comoving units')
    for k in required:
        positive(wdm[k], k)
    if wdm['safety_factor'] < 1:
        raise ValueError('WDM safety factor must be >=1')
    mass = np.asarray(object_masses, dtype=float)
    if not np.isfinite(mass).all() or np.any(mass <= 0):
        raise ValueError('WDM object masses must be finite and positive')
    limit = 10.1 * wdm['mean_density_msun_comoving_mpc3'] * wdm['spacing_comoving_mpc'] / wdm['k_peak_per_comoving_mpc']**2
    return {'estimated_fragmentation_mass_msun': limit, 'masked': (mass <= limit*wdm['safety_factor']).tolist(),
            'scope': 'Wang & White empirical mass floor; unmasked objects still require morphology and resolution convergence'}


def measure_sidm_support(target, limits=None):
    data = target.get('sidm')
    if data is None:
        return {'status': 'MISSING', 'reason': 'Native SIDM kernel, hazard and collapse diagnostics unavailable'}
    required = {'kernel_length_proper_kpc', 'minimum_fine_neighbors', 'coarse_overlap_fraction', 'omitted_hazard_bound', 'maximum_pair_probability', 'maximum_total_hazard', 'collapse_radius_proper_kpc', 'operator_campaign_verified'}
    if set(data) != required:
        raise ValueError('Incomplete measured SIDM support diagnostics')
    for key in required-{'operator_campaign_verified'}:
        positive(data[key], key, zero=key in {'coarse_overlap_fraction', 'omitted_hazard_bound', 'maximum_pair_probability', 'maximum_total_hazard'})
    if limits is None:
        return {'status':'MISSING_LIMITS','measurements':data,'reason':'Declare SIDM kernel, hazard, mass-boundary and collapse limits'}
    expected = {'maximum_kernel_collapse_ratio','minimum_fine_neighbors','maximum_coarse_overlap_fraction',
                'maximum_omitted_hazard_bound','maximum_pair_probability','maximum_total_hazard'}
    if set(limits)!=expected:
        raise ValueError('Incomplete SIDM numerical limits')
    for key,value in limits.items():positive(value,key,zero=key in {'maximum_coarse_overlap_fraction','maximum_omitted_hazard_bound'})
    checks = {}
    pairs = {'kernel_collapse_ratio':(data['kernel_length_proper_kpc']/data['collapse_radius_proper_kpc'],limits['maximum_kernel_collapse_ratio']),
        'coarse_overlap_fraction':(data['coarse_overlap_fraction'],limits['maximum_coarse_overlap_fraction']),
        'omitted_hazard_bound':(data['omitted_hazard_bound'],limits['maximum_omitted_hazard_bound']),
        'pair_probability':(data['maximum_pair_probability'],limits['maximum_pair_probability']),
        'total_hazard':(data['maximum_total_hazard'],limits['maximum_total_hazard'])}
    for key,(value,limit) in pairs.items():checks[key]={'value':value,'limit':limit,'passed':value<=limit}
    checks['fine_neighbors']={'value':data['minimum_fine_neighbors'],'limit':limits['minimum_fine_neighbors'],
                              'passed':data['minimum_fine_neighbors']>=limits['minimum_fine_neighbors']}
    # A user boolean never substitutes for the separately implemented operator
    # and native kernel/timestep/collapse campaign in Block 47.
    return {'status':'UNVALIDATED_OPERATOR' if all(c['passed'] for c in checks.values()) else 'FAIL_SUPPORT',
            'checks':checks,'measurements':data,'reason':'Native SIDM operator campaign must be independently verified before acceptance'}


def validate_physical_target_request(parameters):
    request=parameters['request']
    if request.get('schema')!='physical-target-support-request-v1' or request.get('kind') not in {'first_stars','bh_spike'}:
        raise ValueError('Native first-cloud or BH-profile support request required')
    radial_edges(request['radial_edges_proper_kpc'])
    positive(request['memory_gib'],'physical support memory')
    for name in ('maximum_particles','maximum_neighbor_links','maximum_pair_evaluations','minimum_effective_particles'):
        if type(request[name])is not int or request[name]<1:raise ValueError('Positive exact physical support bound required: '+name)
    identity=request['target_identity']
    if not isinstance(identity,dict) or set(identity)!={'namespace','object_id'} or not isinstance(identity['namespace'],str) or not identity['namespace']:
        raise ValueError('Exact source-linked physical target identity required')
    from core.data import exact_uint64_ids
    exact_uint64_ids([identity['object_id']])
    if type(request['snapshot'])is not int or request['snapshot']<0 or not isinstance(request['particle_namespace'],str) or not request['particle_namespace']:
        raise ValueError('Exact nonnegative snapshot label and native particle namespace required')
    if request['minimum_effective_particles']<2:raise ValueError('Physical support requires at least two effective particles')
    if request['kind']=='first_stars' and request['claim_scope']!='primordial_cooling_environment':
        raise ValueError('Resolved cooling environments do not implement individual primordial-star fragmentation or IMF physics')
    if request['kind']=='bh_spike' and request['claim_scope'] not in {'resolved_dm_profile','collisionless_adiabatic_spike','weakly_collisional_sidm_cusp'}:
        raise ValueError('Explicit BH-profile versus theoretical-cusp claim scope required')
    limits=request['limits']
    for name,value in limits.items():positive(value,'physical support limit '+name,zero=True)
    if request['kind']=='first_stars':
        required={'minimum_jeans_kernel_masses','minimum_jeans_smoothing_lengths','maximum_softening_jeans_fraction','maximum_unsupported_mass_fraction','minimum_cloud_gas_mass_msun','maximum_cooling_freefall_ratio'}
    else:
        required={'maximum_softening_radius_fraction','maximum_influence_radius_fraction','maximum_schwarzschild_radius_fraction','maximum_coarse_mass_fraction'}
    if set(limits)!=required:raise ValueError('Complete frozen physical target limits required')
    for name,value in limits.items():
        if name.startswith('maximum_') and name!='maximum_cooling_freefall_ratio' and value>1:raise ValueError('Physical fraction limit must not exceed one: '+name)
        if name!='maximum_coarse_mass_fraction' and name!='maximum_unsupported_mass_fraction' and value<=0:
            raise ValueError('Physical length/mass support limits must be positive')
    controls=request.get('model_assumption_controls',[])
    if not isinstance(controls,list):raise ValueError('Measured model controls must be a retained-product list')
    if controls and (request['kind']!='bh_spike' or request['claim_scope']=='resolved_dm_profile'):
        raise ValueError('Model controls apply only to explicitly declared theoretical BH cusps')
    return request


def physical_target_input_bindings(parameters):
    from core.sidm_analysis import _load_product,_stage_native_bindings
    request=validate_physical_target_request(parameters)
    sources=_stage_native_bindings(parameters)
    report,_,binding=_load_product(request['center_product'],parameters['request_base'],request['memory_gib']*1024**3)
    if report.get('schema')!='native-halo-membership-measurement-v1' or report['snapshot']!=request['snapshot']:
        raise ValueError('Physical target center must come from the actual measured native aperture at the same epoch')
    original=report.get('native_particle_source_hashes',{})
    if not original or any(sources.get(path)!=sha for path,sha in original.items()):
        raise ValueError('Physical target center is outside the actual owned native evolution')
    sources.update(binding)
    for reference in request.get('model_assumption_controls',[]):
        control,_,binding=_load_product(reference,parameters['request_base'],request['memory_gib']*1024**3)
        if control.get('schema')!='physical-target-model-control-v1':
            raise ValueError('Theoretical cusp assumptions need measured model-control products')
        sources.update(binding)
    return sources


def publish_physical_target_support(parameters,destination):
    """Measure actual native Jeans/force support and BH-profile prerequisites.

    Successful execution is separate from physical support and from numerical
    convergence. The first-star scope is a cooling environment; no individual
    star, IMF or opaque-cloud radiation model is inferred from SSP particles.
    """
    import h5py
    from core.sidm_analysis import _load_product,_stage_native_bindings
    from core.observation_geometry import read_native_gravitating_particles
    from core.data import export_numeric_audit,dataset_to_cgs_factor,exact_uint64_ids
    from core.foundation import MSUN_G
    request=validate_physical_target_request(parameters);memory=request['memory_gib']*1024**3
    sources=physical_target_input_bindings(parameters);owned=_stage_native_bindings(parameters)
    center_report,center_arrays,_=_load_product(request['center_product'],parameters['request_base'],memory)
    center=np.asarray(center_arrays['center_proper_kpc'],float)
    box=float(center_report['periodic_box_proper_kpc'])
    edges=radial_edges(request['radial_edges_proper_kpc'])
    if center.shape!=(3,) or not np.isfinite(center).all() or not 0<edges[-1]<box/2:
        raise ValueError('Measured proper target center and aperture inside half the periodic box required')
    root=Path(parameters['native_run_directory']).resolve();snapshot=request['snapshot']
    files=_snapshot_files(root,snapshot)
    with h5py.File(files[0],'r') as f:
        comoving=bool(int(f['Parameters'].attrs['ComovingIntegrationOn']))
    if not comoving:raise ValueError('This target producer uses the declared cosmological periodic aperture; isolated controls have separate producers')
    population,particles=read_native_gravitating_particles(root,snapshot,species=list(range(6)),
        maximum_particles=request['maximum_particles'],memory_gib=request['memory_gib'],namespace=request['particle_namespace'])
    if any(owned.get(path)!=sha for path,sha in population['source_hashes'].items()):
        raise ValueError('Physical target particles are outside the completed native evolution')
    sources.update(population['source_hashes']);ids=particles['particle_ids'];n=len(ids)
    if n*2560+32*1024**2>memory:raise MemoryError('Combined complete physical particle/census workspace exceeds allocation')
    softening=np.zeros(n);smoothing=np.full(n,np.nan);physical_bh_mass={};lookup={int(value):i for i,value in enumerate(ids)}
    for file in files:
        with h5py.File(file,'r') as f:
            p=f['Parameters'].attrs;a=float(f['Header'].attrs['Time']);h=float(p['HubbleParam'])
            for kind in range(6):
                group=f.get('PartType'+str(kind))
                if group is None:continue
                native_ids=exact_uint64_ids(group['ParticleIDs'][:],unique=False)
                rows=np.asarray([lookup.get(int(value),-1) for value in native_ids]);keep=rows>=0
                classes=group['GravitySofteningClass'][:] if 'GravitySofteningClass' in group else None
                eps=proper_species_softening(p,kind,a,h,classes)
                softening[rows[keep]]=eps if np.ndim(eps)==0 else np.asarray(eps)[keep]
                if kind==0 and request['kind']=='first_stars':
                    if 'SmoothingLength' not in group:raise ValueError('First-cloud support requires actual saved native smoothing lengths')
                    length=group['SmoothingLength'][:]*dataset_to_cgs_factor(group['SmoothingLength'].attrs,a,h)/KPC_CM
                    smoothing[rows[keep]]=length[keep]
                if kind==5 and 'BH_Mass' in group:
                    mass=group['BH_Mass'][:]*dataset_to_cgs_factor(group['BH_Mass'].attrs,a,h)/MSUN_G
                    for value,m in zip(native_ids[keep],mass[keep]):physical_bh_mass[int(value)]=float(m)
    if np.any(softening<=0):raise ValueError('Actual native force resolution is incomplete')
    if request['kind']=='bh_spike':
        if request['target_identity']['namespace']!=request['particle_namespace']:
            raise ValueError('BH target identity must use the original native particle namespace')
        bh_id=request['target_identity']['object_id'];row=lookup.get(bh_id)
        if row is None or particles['species'][row]!=5 or bh_id not in physical_bh_mass:
            raise ValueError('BH support requires the actual live BH identity and physical mass')
        center=particles['position_proper_kpc'][row].copy()
        bh_mass=physical_bh_mass[bh_id];bh_gravity_mass=float(particles['mass_msun'][row])
        if bh_mass<=0 or bh_mass>bh_gravity_mass*(1+1e-12):raise ValueError('Invalid native physical/composite BH mass')
    else:
        expected=center_report['matching']
        if request['target_identity']!={'namespace':expected['halo_match_namespace'],'object_id':expected['halo_match_id']}:
            raise ValueError('First-cloud support changes its measured host identity')
    offsets=particles['position_proper_kpc']-center;offsets-=box*np.rint(offsets/box)
    radius=np.linalg.norm(offsets,axis=1);shell=np.searchsorted(edges,radius,side='left')-1
    shell[radius==0]=0;nb=len(edges)-1
    census=empty_census(edges)
    for kind in range(6):
        use=particles['species']==kind
        accumulate_census(census,radius[use],particles['mass_msun'][use],kind,softening[use],coarse=np.full(int(use.sum()),kind==2))
    census=seal_census(census)
    census.update(target_identity=request['target_identity'],actual_scale_factor=population['epoch']['actual_scale_factor'],
        center_proper_kpc=center.tolist(),source_hashes=population['source_hashes'])
    arrays={'radial_edges_proper_kpc':edges,'radius_proper_kpc':((edges[:-1]**3+edges[1:]**3)/2)**(1/3),
        'physical_supported':np.zeros(nb,bool),'effective_particle_count':np.zeros(nb),'population_mass_msun':np.zeros(nb),
        'maximum_force_support_proper_kpc':np.full(nb,np.nan),'excluded_mass_fraction':np.ones(nb)}
    rows=[];limits=request['limits'];model={'claim_scope':request['claim_scope']}
    if request['kind']=='first_stars':
        from stars import read_native_hydro_census
        from core.first_stars import gas_resolution_diagnostics
        census,meta,binding,_=read_native_hydro_census(root,snapshot,memory_bytes=int(memory))
        if any(owned.get(path)!=sha for path,sha in binding.items()):raise ValueError('Cloud chemistry census escapes actual native evolution')
        sources.update(binding);gas_mask=census['species']==0
        gas={key:value[gas_mask] for key,value in census.items() if key.startswith('gas_') or key in {'particle_ids','mass_msun'}}
        gas_rows=np.asarray([lookup[int(value)] for value in gas['particle_ids']]);gas['mean_molecular_weight']=gas['gas_mean_molecular_weight']
        gas['smoothing_length_proper_kpc']=smoothing[gas_rows]
        tree=cKDTree(np.mod(particles['position_proper_kpc'][gas_rows],box),boxsize=box)
        kernel_mass=np.zeros(len(gas_rows));links=0
        for i in range(len(gas_rows)):
            neighbors=tree.query_ball_point(np.mod(particles['position_proper_kpc'][gas_rows[i]],box),gas['smoothing_length_proper_kpc'][i])
            links+=len(neighbors)
            if links>request['maximum_neighbor_links']:raise MemoryError('Actual gas kernel census exceeds the explicit link budget')
            kernel_mass[i]=gas['mass_msun'][neighbors].sum()
        diagnostic=gas_resolution_diagnostics(gas,{'adiabatic_index':meta['gas_adiabatic_index'],'kernel_neighbors':1,
            'minimum_jeans_kernel_masses':limits['minimum_jeans_kernel_masses'],'minimum_jeans_smoothing_lengths':limits['minimum_jeans_smoothing_lengths']})
        thermal=gas['gas_temperature_available']&gas['gas_native_eos_available']&gas['gas_molecular_species_available']
        thermal&=gas['gas_chemistry_state_version']==4
        physical=thermal&diagnostic['thermodynamic_supported']
        from core.foundation import G_CGS
        density=gas['gas_density_g_cm3'];inverse=gas['gas_inverse_cooling_time_per_s']
        freefall=np.full(density.shape,np.nan);cooling=np.full(density.shape,np.nan)
        valid_density=gas['gas_density_g_cm3_available']&(density>0)
        freefall[valid_density]=np.sqrt(3*np.pi/(32*G_CGS*density[valid_density]))/(SECONDS_PER_YEAR*1e6)
        cooling_available=gas['gas_cooling_time_available']&(inverse>0)
        cooling[cooling_available]=1/inverse[cooling_available]/(SECONDS_PER_YEAR*1e6)
        physical&=cooling_available&(cooling<=limits['maximum_cooling_freefall_ratio']*freefall)
        physical&=diagnostic['jeans_mass_msun']>=limits['minimum_jeans_kernel_masses']*kernel_mass
        physical&=diagnostic['jeans_length_kpc']>=limits['minimum_jeans_smoothing_lengths']*gas['smoothing_length_proper_kpc']
        physical&=2.8*softening[gas_rows]<=limits['maximum_softening_jeans_fraction']*diagnostic['jeans_length_kpc']
        arrays.update(jeans_length_minimum_proper_kpc=np.full(nb,np.nan),jeans_mass_minimum_msun=np.full(nb,np.nan),
            maximum_cooling_freefall_ratio=np.full(nb,np.nan),minimum_freefall_time_myr=np.full(nb,np.nan))
        for b in range(nb):
            use=shell[gas_rows]==b;mass=gas['mass_msun'][use];total=float(mass.sum());valid=use&physical
            count=int(use.sum());neff=total**2/float(np.dot(mass,mass)) if count else 0.
            excluded=1-float(gas['mass_msun'][valid].sum())/total if total else 1.
            support=count>0 and neff>=request['minimum_effective_particles'] and excluded<=limits['maximum_unsupported_mass_fraction'] and total>=limits['minimum_cloud_gas_mass_msun']
            arrays['effective_particle_count'][b]=neff;arrays['population_mass_msun'][b]=total;arrays['excluded_mass_fraction'][b]=excluded
            arrays['physical_supported'][b]=support
            if count:arrays['maximum_force_support_proper_kpc'][b]=float((2.8*softening[gas_rows][use]).max())
            if valid.any():
                arrays['jeans_length_minimum_proper_kpc'][b]=float(diagnostic['jeans_length_kpc'][valid].min())
                arrays['jeans_mass_minimum_msun'][b]=float(diagnostic['jeans_mass_msun'][valid].min())
                arrays['maximum_cooling_freefall_ratio'][b]=float((cooling[valid]/freefall[valid]).max())
                arrays['minimum_freefall_time_myr'][b]=float(freefall[valid].min())
            rows.append({'radial_interval_proper_kpc':edges[b:b+2].tolist(),'status':'SUPPORTED_PREREQUISITES' if support else 'UNRESOLVED',
                'supported':bool(support),'count':count,'effective_count':neff,'excluded_gas_mass_fraction':excluded})
        model.update(network=meta['parameters']['grackle_network'],eos='actual native single-phase molecular EOS',
            cooling_definition='inverse of signed pinned Grackle local_calculate_cooling_time at actual native chemical/thermal state; net cooling required',
            kernel_mass_definition='complete actual gas mass within saved compact kernel support; self particle included',
            exclusions=['individual-star fragmentation','PopIII IMF','SH03 effective-EOS temperature','opaque-cloud radiation transfer'])
    else:
        import yaml
        cfg=yaml.safe_load((root/'config.yaml').read_text());sidm=cfg.get('sidm',{}).get('enabled',False)
        if sidm and request['claim_scope']=='collisionless_adiabatic_spike' or not sidm and request['claim_scope']=='weakly_collisional_sidm_cusp':
            raise ValueError('Theoretical BH cusp claim conflicts with the actual native interaction model')
        arrays.update(density_msun_kpc3=np.zeros(nb),velocity_dispersion_squared_km2_s2=np.full(nb,np.nan),
            influence_radius_proper_kpc=np.full(nb,np.nan),coarse_mass_fraction=np.ones(nb),
            enclosed_background_mass_msun=np.zeros(nb),orbital_time_myr=np.full(nb,np.nan),
            sidm_collision_time_myr=np.full(nb,np.nan),sidm_mean_free_path_proper_kpc=np.full(nb,np.nan))
        cross_section=None
        if sidm:
            from core.sidm_physics import sidm_physical_model
            physics_source=Path(__file__).with_name('sidm_physics.py').resolve()
            sources[str(physics_source)]=file_sha256(physics_source)
            from core.sidm_physics import read_native_differential_model
            law,asset=sidm_physical_model(cfg['sidm']);sources[asset['path']]=asset['sha256']
            normalization=float(cfg['sidm']['cross_section_normalization_cm2_per_g'])
            if law['schema']=='sidm-elastic-model-v2':
                with h5py.File(files[0],'r') as snapshot_file:
                    native=snapshot_file['Parameters'].attrs
                    def text_attribute(name):
                        value=native[name]
                        return value.decode('ascii') if isinstance(value,bytes) else str(value)
                    table_path=Path(text_attribute('SIDMNativeTablePath')).expanduser().resolve()
                    table_hash=text_attribute('SIDMNativeTableSHA256');model_hash=text_attribute('SIDMAngularModelHash')
                if owned.get(str(table_path))!=table_hash or model_hash!=canonical_digest(law):
                    raise ValueError('BH collision-time diagnostic lost the actual evolved SIDM interpolant')
                sources[str(table_path)]=table_hash
                elastic=read_native_differential_model(table_path,model_sha256=model_hash,table_sha256=table_hash,
                    particle_convention=cfg['sidm']['particle_convention'],memory_bytes=int(memory-n*2560-32*1024**2))
                cross_section=lambda speed:np.asarray(elastic.total_cross_section_per_mass(speed))*(0. if normalization==0 else 1.)
            elif law['kind']=='isotropic':cross_section=lambda speed:np.full(np.shape(speed),normalization)
            elif law['kind']=='screened_rutherford':cross_section=lambda speed:normalization/(1+(np.asarray(speed)/law['velocity_scale_kms'])**2)
            else:
                from core.sidm_physics import moller_primitive as primitive
                cross_section=lambda speed:normalization*np.asarray([primitive(1.,.5*(float(v)/law['velocity_scale_kms'])**2) for v in np.asarray(speed).flat]).reshape(np.shape(speed))
        pair_visits=0
        schwarzschild=2*G*bh_mass/299792.458**2
        for b in range(nb):
            use=(shell==b)&np.isin(particles['species'],(1,2));fine=use&(particles['species']==1)
            mass=particles['mass_msun'][fine];total=float(mass.sum());count=len(mass)
            neff=total**2/float(np.dot(mass,mass)) if count else 0.;coarse=float(particles['mass_msun'][use&~fine].sum())
            fraction=coarse/(total+coarse) if total+coarse else 1.
            arrays['effective_particle_count'][b]=neff;arrays['population_mass_msun'][b]=total;arrays['coarse_mass_fraction'][b]=fraction
            support=False
            if count:
                v=particles['velocity_peculiar_kms'][fine];mean=np.average(v,axis=0,weights=mass)
                sigma2=float(np.sum(mass[:,None]*(v-mean)**2)/total/3)
                influence=G*bh_gravity_mass/sigma2 if sigma2>0 else None
                force=float(2.8*np.max(softening[fine]));rmin=float(edges[b])
                background=float(particles['mass_msun'][(radius<=edges[b+1])&(ids!=bh_id)].sum())
                arrays['enclosed_background_mass_msun'][b]=background
                arrays['orbital_time_myr'][b]=2*np.pi*np.sqrt(arrays['radius_proper_kpc'][b]**3/(G*(bh_gravity_mass+background)))/KMS_MYR_KPC
                arrays['density_msun_kpc3'][b]=total/(4*np.pi/3*(edges[b+1]**3-edges[b]**3))
                arrays['velocity_dispersion_squared_km2_s2'][b]=sigma2;arrays['maximum_force_support_proper_kpc'][b]=force
                if influence is not None:arrays['influence_radius_proper_kpc'][b]=influence
                if cross_section is not None and count>=2:
                    # Empirical unordered-pair expectation, excluding self pairs.
                    # This is a measured local molecular-chaos rate estimate,
                    # not a count of native Monte Carlo events or a heat flux.
                    pair_visits+=count*(count-1)//2
                    if pair_visits>request['maximum_pair_evaluations']:raise MemoryError('BH local collision-rate census exceeds its explicit pair budget')
                    weighted_rate=0.;weighted_speed=0.;pair_weight=0.
                    for i in range(count-1):
                        speed=np.linalg.norm(v[i+1:]-v[i],axis=1);weight=mass[i]*mass[i+1:]
                        weighted_rate+=float(np.dot(weight,cross_section(speed)*speed))
                        weighted_speed+=float(np.dot(weight,speed));pair_weight+=float(weight.sum())
                    if pair_weight>0 and weighted_rate>0:
                        mean_rate=weighted_rate/pair_weight
                        rho_cgs=arrays['density_msun_kpc3'][b]*MSUN_G/KPC_CM**3
                        collision_s=1/(rho_cgs*mean_rate*1e5)
                        arrays['sidm_collision_time_myr'][b]=collision_s/(SECONDS_PER_YEAR*1e6)
                        arrays['sidm_mean_free_path_proper_kpc'][b]=weighted_speed/pair_weight*1e5*collision_s/KPC_CM
                influence_supported=request['claim_scope']=='resolved_dm_profile' or influence is not None and edges[b+1]/influence<=limits['maximum_influence_radius_fraction']
                support=(rmin>0 and neff>=request['minimum_effective_particles'] and influence_supported
                    and force/rmin<=limits['maximum_softening_radius_fraction']
                    and schwarzschild/rmin<=limits['maximum_schwarzschild_radius_fraction'] and fraction<=limits['maximum_coarse_mass_fraction'])
            arrays['physical_supported'][b]=support;arrays['excluded_mass_fraction'][b]=fraction
            rows.append({'radial_interval_proper_kpc':edges[b:b+2].tolist(),'supported':bool(support),
                'status':'SUPPORTED_PREREQUISITES' if support else 'UNRESOLVED','count':count,'effective_count':neff})
        model.update(bh_physical_mass_msun=bh_mass,bh_composite_gravity_mass_msun=bh_gravity_mass,
            schwarzschild_radius_proper_kpc=schwarzschild,interaction_model='elastic_sidm' if sidm else 'collisionless',
            theory_status='requires independent measured growth/collisionality/steady-state model controls' if request['claim_scope']!='resolved_dm_profile' else 'no theoretical origin inferred')
        if request['claim_scope']!='resolved_dm_profile':
            controls=request.get('model_assumption_controls',[])
            if not controls:arrays['physical_supported'][:]=False
            for reference in controls:
                report,values,_=_load_product(reference,parameters['request_base'],memory)
                if (report['target_identity']!=request['target_identity'] or report['claim_scope']!=request['claim_scope'] or report['epoch']!=population['epoch']
                        or report['native_run_directory']!=str(root) or report['final_native_particle_source_hashes']!=population['source_hashes']):
                    raise ValueError('BH model controls change actual target, epoch or physical assumptions')
                if not np.array_equal(values['radial_edges_proper_kpc'],edges):raise ValueError('BH model controls change measured radial bins')
                flags=np.asarray(values['model_supported'])
                if flags.dtype.kind!='b' or flags.shape!=(nb,):raise ValueError('BH model support must use actual aligned boolean control arrays')
                arrays['physical_supported']&=flags
            for i,row in enumerate(rows):row['supported']=bool(arrays['physical_supported'][i]);row['status']='SUPPORTED_PREREQUISITES' if row['supported'] else 'UNRESOLVED_MODEL_ASSUMPTIONS'
    source=Path(__file__).resolve();sources[str(source)]=file_sha256(source)
    retained_request=json.loads(json.dumps(request))
    for reference in [retained_request['center_product'],*retained_request.get('model_assumption_controls',[])]:
        path=Path(reference['path']).expanduser()
        reference['path']=str((Path(parameters['request_base'])/path if not path.is_absolute() else path).resolve())
    report={'schema':'physical-target-support-v1','execution_status':'COMPLETE','request':retained_request,'kind':request['kind'],
        'claim_scope':request['claim_scope'],'target_identity':request['target_identity'],'epoch':population['epoch'],'model':model,
        'particle_census':census,'native_particle_source_hashes':population['source_hashes'],
        'source_snapshots':[{'path':str(path.resolve()),'sha256':sources[str(path.resolve())]} for path in files],
        'native_run_directory':str(root),'particle_namespace':request['particle_namespace'],
        'native_stage_receipt':str(Path(parameters['native_stage_receipt']).resolve()),'source_hashes':sources,'radial_cells':rows,
        'support_scope':'measured physical prerequisites only; current independent numerical convergence is separately required',
        'science_acceptance':'NOT_ESTABLISHED'}
    units={name:('Msun / proper kpc^3' if name=='density_msun_kpc3' else 'proper kpc' if 'kpc' in name else '(km / s)^2' if 'km2_s2' in name else 'Msun' if 'msun' in name else 'Myr' if 'myr' in name else 'dimensionless') for name in arrays}
    return export_numeric_audit(destination,report,arrays,{name:{'units':units[name],'quantity':'measured physical support '+name.replace('_',' ')} for name in arrays},
        algorithm=source,context={'epoch':population['epoch'],'coordinates':{'frame':'exact native measured host/BH proper center'}},source_hashes=sources)


def validate_target_model_control_request(parameters):
    request=parameters['request']
    if request.get('schema')!='physical-target-model-control-request-v1' or request.get('claim_scope') not in {'collisionless_adiabatic_spike','weakly_collisional_sidm_cusp'}:
        raise ValueError('Measured BH-growth or SIDM-relaxation history request required')
    positive(request['memory_gib'],'BH model history workspace')
    if type(request['maximum_epochs'])is not int or not 3<=len(request['histories'])<=request['maximum_epochs']:
        raise ValueError('At least three bounded actual BH-profile epochs required')
    identity=request['target_identity']
    from core.data import exact_uint64_ids
    if not isinstance(identity,dict) or set(identity)!={'namespace','object_id'} or not isinstance(identity['namespace'],str) or not identity['namespace']:
        raise ValueError('Exact native BH model-control identity required')
    exact_uint64_ids([identity['object_id']]);radial_edges(request['radial_edges_proper_kpc'])
    shared={'maximum_background_bh_mass_ratio','maximum_output_orbit_fraction'}
    fields=shared|({'minimum_growth_orbit_ratio','minimum_bh_growth_factor','maximum_negative_growth_fraction'} if request['claim_scope']=='collisionless_adiabatic_spike'
        else {'minimum_collision_orbit_ratio','minimum_elapsed_collision_times','minimum_mean_free_path_radius_ratio','maximum_density_change_per_collision_time','maximum_dispersion_change_per_collision_time'})
    if set(request['limits'])!=fields:raise ValueError('Complete frozen model-history assumptions required')
    for name,value in request['limits'].items():positive(value,name,zero=name=='maximum_negative_growth_fraction')
    if request['limits']['maximum_output_orbit_fraction']>1:raise ValueError('History cadence must resolve orbital time')
    if request['claim_scope']=='collisionless_adiabatic_spike':
        if request['limits']['minimum_growth_orbit_ratio']<=1 or request['limits']['minimum_bh_growth_factor']<=1 or request['limits']['maximum_negative_growth_fraction']>=1:
            raise ValueError('Adiabatic controls require resolved slow positive growth')
    elif request['limits']['minimum_collision_orbit_ratio']<=1 or request['limits']['minimum_elapsed_collision_times']<1 or request['limits']['minimum_mean_free_path_radius_ratio']<=1:
        raise ValueError('Weakly collisional cusp controls require many orbits between collisions, relaxation and a long mean free path')
    return request


def target_model_control_input_bindings(parameters):
    from core.sidm_analysis import _load_product
    request=validate_target_model_control_request(parameters);sources={}
    for reference in request['histories']:
        report,_,binding=_load_product(reference,parameters['request_base'],request['memory_gib']*1024**3)
        if report.get('schema')!='physical-target-support-v1' or report['kind']!='bh_spike' or report['claim_scope']!='resolved_dm_profile':
            raise ValueError('Model history requires independently produced raw native BH-profile epochs')
        sources.update(binding)
    return sources


def measure_target_model_controls(parameters):
    """Compute sampled physical prerequisites; never synthesize theoretical cusps.

    The adiabatic route measures BH growth against orbital times. The SIDM
    route measures a local molecular-chaos collision-time estimate and its
    stationarity over a resolved native history. Unobserved formation stages,
    relativistic capture, and individual phase-space occupations are excluded.
    """
    from core.sidm_analysis import _load_product
    request=validate_target_model_control_request(parameters);edges=radial_edges(request['radial_edges_proper_kpc'])
    if len(request['histories'])*(len(edges)-1)*2048>request['memory_gib']*1024**3:
        raise MemoryError('Complete BH model history exceeds its workspace before loading')
    sources=target_model_control_input_bindings(parameters);profiles=[];reports=[];times=[];run=None;physics=None
    import yaml
    for reference in request['histories']:
        report,arrays,_=_load_product(reference,parameters['request_base'],request['memory_gib']*1024**3)
        if report['target_identity']!=request['target_identity'] or not np.array_equal(arrays['radial_edges_proper_kpc'],edges):
            raise ValueError('BH model history changes its exact native target or radial aperture')
        if run is None:run=report['native_run_directory']
        if report['native_run_directory']!=run:raise ValueError('BH model history must follow one actual native trajectory')
        expected='collisionless' if request['claim_scope']=='collisionless_adiabatic_spike' else 'elastic_sidm'
        if report['model']['interaction_model']!=expected:raise ValueError('BH model assumptions conflict with actual native interaction physics')
        epoch=report['epoch'];time=epoch.get('cosmic_time_gyr')
        if time is None:raise ValueError('BH model history needs the source-bound actual proper cosmological clock')
        times.append(1000*positive(time,'actual BH model time'));reports.append(report);profiles.append(arrays)
        configuration=Path(report['native_run_directory'])/'config.yaml'
        if report['source_hashes'].get(str(configuration.resolve()))!=file_sha256(configuration):raise ValueError('BH model history lost native configuration binding')
        value=yaml.safe_load(configuration.read_text());signature=canonical_digest({key:value.get(key) for key in ('cosmology','ede','wdm','sidm','chemistry','black_holes','stellar_evolution','hydro')})
        if physics is not None and signature!=physics:raise ValueError('BH model history changes its physical model')
        physics=signature
    time=np.asarray(times);dt=np.diff(time)
    if np.any(dt<=0):raise ValueError('BH model epochs must increase strictly in measured physical time')
    nb=len(edges)-1;nt=len(time)
    if nt*nb*1024>request['memory_gib']*1024**3:raise MemoryError('BH model history exceeds declared workspace')
    def history(name):
        result=np.stack([np.asarray(row[name]) for row in profiles])
        if result.shape!=(nt,nb):raise ValueError('BH model history field has changed its radial dimensions: '+name)
        return result
    orbit=history('orbital_time_myr');background=history('enclosed_background_mass_msun')
    support=history('physical_supported')
    if support.dtype.kind!='b':raise ValueError('BH model history masks must retain exact booleans')
    physical_mass=np.asarray([report['model']['bh_physical_mass_msun'] for report in reports])
    gravity_mass=np.asarray([report['model']['bh_composite_gravity_mass_msun'] for report in reports])
    valid=np.all(support,axis=0)&np.all(np.isfinite(orbit)&(orbit>0),axis=0)
    with np.errstate(divide='ignore',invalid='ignore'):
        cadence=dt[:,None]/np.minimum(orbit[:-1],orbit[1:])
        contamination=background/gravity_mass[:,None]
    limits=request['limits']
    valid&=np.all(cadence<=limits['maximum_output_orbit_fraction'],axis=0)
    valid&=np.all(contamination<=limits['maximum_background_bh_mass_ratio'],axis=0)
    arrays={'radial_edges_proper_kpc':edges,'actual_times_myr':time,'bh_physical_mass_msun':physical_mass,
        'bh_gravity_mass_msun':gravity_mass,'orbital_time_myr':orbit,'output_orbit_fraction':cadence,
        'background_bh_mass_ratio':contamination}
    if request['claim_scope']=='collisionless_adiabatic_spike':
        rate=np.diff(np.log(physical_mass))/dt
        negative=(physical_mass[:-1]-physical_mass[1:])/physical_mass[:-1]
        with np.errstate(divide='ignore',invalid='ignore'):
            ratio=1/(np.abs(rate[:,None])*np.maximum(orbit[:-1],orbit[1:]))
        # Constant mass is infinitely slow growth; the full history must still
        # contain the explicitly requested positive total growth factor.
        ratio[rate==0]=np.inf
        valid&=np.all(ratio>=limits['minimum_growth_orbit_ratio'],axis=0)
        valid&=bool(np.all(negative<=limits['maximum_negative_growth_fraction']) and physical_mass[-1]/physical_mass[0]>=limits['minimum_bh_growth_factor'])
        arrays.update(log_bh_growth_rate_per_myr=rate,growth_orbit_ratio=ratio)
        exclusions=['unobserved earlier BH formation','individual orbital-action invariance','relativistic capture','annihilation physics']
    else:
        collision=history('sidm_collision_time_myr');path=history('sidm_mean_free_path_proper_kpc')
        density=history('density_msun_kpc3');dispersion=history('velocity_dispersion_squared_km2_s2')
        valid&=np.all(np.isfinite(collision)&(collision>0)&np.isfinite(path)&(path>0)&(density>0)&(dispersion>0),axis=0)
        with np.errstate(divide='ignore',invalid='ignore'):
            ratio=collision/orbit;relaxed=(time[-1]-time[0])/np.max(collision,axis=0)
            density_change=np.abs(np.diff(np.log(density),axis=0))/dt[:,None]*np.maximum(collision[:-1],collision[1:])
            dispersion_change=np.abs(np.diff(np.log(dispersion),axis=0))/dt[:,None]*np.maximum(collision[:-1],collision[1:])
        valid&=np.all(ratio>=limits['minimum_collision_orbit_ratio'],axis=0)&(relaxed>=limits['minimum_elapsed_collision_times'])
        valid&=np.all(path/edges[None,1:]>=limits['minimum_mean_free_path_radius_ratio'],axis=0)
        valid&=np.all(density_change<=limits['maximum_density_change_per_collision_time'],axis=0)
        valid&=np.all(dispersion_change<=limits['maximum_dispersion_change_per_collision_time'],axis=0)
        arrays.update(collision_time_myr=collision,mean_free_path_proper_kpc=path,collision_orbit_ratio=ratio,
            elapsed_collision_times=relaxed,density_change_per_collision_time=density_change,dispersion_change_per_collision_time=dispersion_change)
        exclusions=['short-mean-free-path cusp theory','unobserved relaxation before supplied history','relativistic capture','unique cross-section inference']
    arrays['model_supported']=valid
    retained_request=json.loads(json.dumps(request))
    for reference in retained_request['histories']:
        path=Path(reference['path']).expanduser()
        reference['path']=str((Path(parameters['request_base'])/path if not path.is_absolute() else path).resolve())
    report={'schema':'physical-target-model-control-v1','execution_status':'COMPLETE','request':retained_request,
        'target_identity':request['target_identity'],'claim_scope':request['claim_scope'],'epoch':reports[-1]['epoch'],
        'native_run_directory':run,'physics_fingerprint':physics,'source_hashes':sources,
        'final_native_particle_source_hashes':reports[-1]['native_particle_source_hashes'],'history_epochs':[row['epoch'] for row in reports],
        'supported_scope':'sampled growth/relaxation physical prerequisites, separately from measured numerical convergence',
        'exclusions':exclusions,'science_acceptance':'NOT_ESTABLISHED'}
    return report,arrays,sources


def publish_target_model_controls(parameters,destination):
    from core.data import export_numeric_audit
    report,arrays,sources=measure_target_model_controls(parameters)
    units={name:('Msun' if 'mass_msun' in name else 'proper kpc' if 'kpc' in name else 'Myr^-1' if 'per_myr' in name else 'Myr' if 'myr' in name else 'dimensionless') for name in arrays}
    specifications={name:{'units':units[name],'quantity':name.replace('_',' ')} for name in arrays}
    history_fields={'actual_times_myr','bh_physical_mass_msun','bh_gravity_mass_msun','orbital_time_myr','background_bh_mass_ratio',
        'collision_time_myr','mean_free_path_proper_kpc','collision_orbit_ratio'}
    for name in history_fields&set(arrays):
        specifications[name]['epoch']={'representation':'row-aligned actual native epochs','axis':0,'rows':report['history_epochs']}
    interval_fields={'output_orbit_fraction','log_bh_growth_rate_per_myr','growth_orbit_ratio','density_change_per_collision_time','dispersion_change_per_collision_time'}
    for name in interval_fields&set(arrays):
        specifications[name]['annotations']={'interval_start_epochs':report['history_epochs'][:-1],'interval_end_epochs':report['history_epochs'][1:]}
    return export_numeric_audit(destination,report,arrays,specifications,
        algorithm=Path(__file__).resolve(),context={'epoch':report['epoch'],'coordinates':{'frame':'measured BH-centered proper radial history'}},source_hashes=sources)


def validate_target_measurement_request(parameters):
    request=parameters['request']
    if request.get('schema')!='physical-target-resolution-measurement-request-v1':raise ValueError('Native physical-target convergence-vector request required')
    positive(request['memory_gib'],'physical convergence-vector workspace')
    bins=request['radial_bin_indices']
    if not isinstance(bins,list) or not bins or any(type(i)is not int or i<0 for i in bins) or any(b<=a for a,b in zip(bins,bins[1:])):
        raise ValueError('Explicit ordered distinct native radial-bin indices required')
    if not isinstance(request['observable'],str) or not request['observable']:raise ValueError('Named physical convergence observable required')
    fields={'density_msun_kpc3','velocity_dispersion_squared_km2_s2','jeans_mass_minimum_msun','jeans_length_minimum_proper_kpc','population_mass_msun'}
    if request['dataset'] not in fields:raise ValueError('Select a measured physical profile, not a support/count flag, for convergence')
    return request


def target_measurement_input_bindings(parameters):
    from core.sidm_analysis import _load_product
    request=validate_target_measurement_request(parameters)
    report,_,sources=_load_product(request['support_product'],parameters['request_base'],request['memory_gib']*1024**3)
    if report.get('schema')!='physical-target-support-v1':raise ValueError('Physical convergence needs an actual native support product')
    return sources


def publish_target_resolution_measurement(parameters,destination):
    """Publish actual radial values with native axis metadata and failed bins.

    These products feed the existing mass/softening/timestep comparison; no
    supplied PASS, synthetic vector, or analytic profile grants convergence.
    """
    import yaml
    from core.sidm_analysis import _load_product
    from core.data import export_numeric_audit
    request=validate_target_measurement_request(parameters);sources=target_measurement_input_bindings(parameters)
    parent,measured,_=_load_product(request['support_product'],parameters['request_base'],request['memory_gib']*1024**3)
    index=np.asarray(request['radial_bin_indices'],dtype=int);name=request['dataset']
    if np.any(index>=len(measured['physical_supported'])) or name not in measured:raise ValueError('Requested native physical measurement bins/field are absent')
    if parent['kind']=='first_stars' and name in {'density_msun_kpc3','velocity_dispersion_squared_km2_s2'}:
        raise ValueError('DM profile fields cannot be relabeled as first-cloud measurements')
    species=0 if parent['kind']=='first_stars' else 1
    values=np.asarray(measured[name][index],float);mask=np.asarray(measured['physical_supported'][index],bool)&np.isfinite(values)
    root=Path(parent['native_run_directory']);cfg_path=(root/'config.yaml').resolve()
    if sources.get(str(cfg_path))!=file_sha256(cfg_path):raise ValueError('Physical convergence product lost actual native physics')
    config=yaml.safe_load(cfg_path.read_text())
    # Species, chemistry, BH and interaction physics are fixed; the three
    # measured numerical axes come directly from retained native snapshots.
    physics={key:config.get(key) for key in ('cosmology','ede','wdm','sidm','chemistry','black_holes','stellar_evolution','hydro')}
    physics['box_mpc_h']=config['simulation']['box_mpc_h'];physics['start_redshift']=config['simulation']['start_redshift']
    physics['random_seed']=config['simulation']['random_seed'];physics['physics']=config['simulation']['physics']
    actual_a=parent['epoch']['actual_scale_factor'];units=parent['numeric_data']['array_contract']['datasets'][name]['units']
    edges=np.asarray(measured['radial_edges_proper_kpc']);selection={'target_identity':parent['target_identity'],
        'claim_scope':parent['claim_scope'],'particle_namespace':parent['particle_namespace'],
        'radial_intervals_proper_kpc':[[float(edges[i]),float(edges[i+1])] for i in index]}
    meta={'origin':'native_simulation','observable':request['observable'],'units':units,'selection':selection,
        'actual_scale_factor':actual_a,'physics_fingerprint':canonical_digest(physics),'bin_coordinates':measured['radius_proper_kpc'][index].tolist(),
        'all_requested_bins_valid':bool(mask.all()),'numerical_settings':measure_native_resolution_settings(parent['source_snapshots'],species,actual_a),
        'source_snapshots':parent['source_snapshots'],'species':species}
    report={'schema':'physical-target-resolution-measurement-v1','execution_status':'COMPLETE','request':request,
        'target_identity':parent['target_identity'],'epoch':parent['epoch'],'resolution_measurement':meta,
        'source_hashes':sources,'science_acceptance':'NOT_ESTABLISHED'}
    arrays={name:values,'measurement_supported':mask,'radius_proper_kpc':measured['radius_proper_kpc'][index]}
    return export_numeric_audit(destination,report,arrays,{name:{'units':units,'quantity':'actual native '+name.replace('_',' ')},
        'measurement_supported':{'units':'dimensionless','quantity':'actual physical and numerical bin prerequisites'},
        'radius_proper_kpc':{'units':'proper kpc','quantity':'measured radial shell volume midpoint'}},
        algorithm=Path(__file__).resolve(),context={'epoch':parent['epoch'],'coordinates':{'frame':'actual native target-centered proper radial shells'}},source_hashes=sources)


def _resolution_physical_support(target,census,request):
    from core.sidm_analysis import _load_product
    products={};sources={}
    for name,reference in request.get('physical_support',{}).items():
        report,arrays,binding=_load_product({key:reference[key] for key in ('path','sha256')},'.',reference['memory_gib']*1024**3)
        obs=next(row for row in request['observables'] if row['name']==name)
        if report.get('schema')!='physical-target-support-v1' or report['kind']!=obs['kind'] or report['claim_scope']!=obs['claim_scope'] or report.get('execution_status')!='COMPLETE':
            raise ValueError('Resolution physical prerequisite is not the requested completed native target measurement')
        if report['target_identity']!=target.get('target_identity') or report['epoch']['actual_scale_factor']!=target.get('actual_scale_factor'):
            raise ValueError('Resolution physical evidence changes exact target or actual epoch')
        if report['particle_census']!=census or not np.array_equal(arrays['radial_edges_proper_kpc'],request['radial_edges_proper_kpc']):
            raise ValueError('Resolution physical support differs from the actual native particle census or radial aperture')
        flags=np.asarray(arrays['physical_supported'])
        if flags.dtype.kind!='b' or flags.shape!=(len(request['radial_edges_proper_kpc'])-1,):raise ValueError('Physical support mask changed shape/type')
        for control in report['request'].get('model_assumption_controls',[]):
            saved,values,_=_load_product(control,'.',reference['memory_gib']*1024**3)
            recomputed,measured,_=measure_target_model_controls({'request':saved['request'],'request_base':'.'})
            if saved['epoch']!=recomputed['epoch'] or not np.array_equal(values['model_supported'],measured['model_supported']):
                raise ValueError('BH model-control support differs from retained native history')
        products[name]=(report,flags);sources.update(binding)
    return products,sources


def evaluate_resolution(target, particle_census, request, *, convergence=None):
    validate_resolution_request(request)
    levels = measure_actual_mass_levels(particle_census)
    if not np.array_equal(levels['edges'], request['radial_edges_proper_kpc']):
        raise ValueError('Request/census radial apertures differ')
    limits = estimate_relaxation_and_force_limits(target, levels)
    heating = estimate_then_calibrate_stream_heating(target, levels)
    physical,physical_sources=_resolution_physical_support(target,particle_census,request)
    fingerprint = canonical_digest({'target': target, 'census': particle_census, 'request': request,'physical_source_hashes':physical_sources})
    # Evidence is source-bound and recomputed by the comparison intake below.
    measured = verify_resolution_convergence(convergence, fingerprint) if convergence else None
    rows = []
    r = levels['edges'][1:]
    for obs in request['observables']:
        # Use the cumulative outer edge at/inside the requested scale, never a larger
        # cumulative aperture to manufacture sufficient counts for a small scale.
        candidates = np.flatnonzero(r <= obs['minimum_radius_proper_kpc'])
        b = int(candidates[-1]) if len(candidates) else None
        checks = {}
        def check(name, value, limit, direction='maximum'):
            passed = value is not None and (value <= limit if direction == 'maximum' else value >= limit)
            checks[name] = {'value': value, 'limit': limit, 'direction': direction, 'passed': bool(passed)}
        check('aperture_sampled', b is not None, True, 'minimum')
        if b is not None:
            check('members', int(levels['count'][obs['species'], :b+1].sum()), obs['minimum_members'], 'minimum')
            check('coarse_mass_fraction', float(levels['enclosed_coarse_fraction'][b]), request['maximum_coarse_mass_fraction'])
            check('relaxation_age_ratio_estimate', limits['relaxation_age_ratio_estimate'][b], request['minimum_relaxation_age_ratio'], 'minimum')
            scale = obs.get('width_proper_kpc', obs['minimum_radius_proper_kpc'])
            check('force_support_fraction', limits['force_support_proper_kpc'][b]/scale, request['maximum_force_support_fraction'])
            check('timestep_crossing_fraction', limits['timestep_crossing_fraction'][b], request['maximum_timestep_crossing_fraction'])
            if obs['kind'] == 'cold_stream':
                variance = heating['variance_rate_km2_s2_per_myr_estimate'][b]*obs['duration_myr']
                check('heating_variance_fraction_estimate', variance/obs['dispersion_km_s']**2, obs['maximum_heating_variance_fraction'])
                crossing = obs['width_proper_kpc']/(target['relative_speed_km_s']*KMS_MYR_KPC)
                check('stream_timestep_crossing_fraction', target['maximum_proper_timestep_myr']/crossing, request['maximum_timestep_crossing_fraction'])
        if obs['kind'] in {'first_stars','bh_spike'}:
            evidence=physical.get(obs['name'])
            # Shell evidence must actually cover the requested radius; never
            # use an earlier occupied bin to certify an unsampled gap.
            edges=np.asarray(request['radial_edges_proper_kpc'])
            cell=int(np.searchsorted(edges,obs['minimum_radius_proper_kpc'],side='left')-1)
            supported=evidence is not None and 0<=cell<len(edges)-1 and bool(evidence[1][cell])
            check('measured_physical_target_support',supported,True,'minimum')
        if obs['kind'] == 'sidm_core':
            support = measure_sidm_support(target, request.get('sidm_limits'))
            checks['sidm_measured_support'] = {'value':support,'limit':'all declared limits and native operator validation','passed':support['status']=='PASS'}
            check('native_sidm_support_verified', support['status'] == 'PASS', True, 'minimum')
        if target.get('wdm'):
            mask = wdm_fragmentation_mask(target['wdm'], [target['object_mass_msun']])
            check('wdm_fragmentation_unmasked', not mask['masked'][0], True, 'minimum')
        preflight = all(c['passed'] for c in checks.values())
        accepted = preflight and measured is not None and measured['observables'].get(obs['name'], {}).get('status') == 'PASS'
        rows.append({'name': obs['name'], 'kind': obs['kind'], 'bin': b, 'checks': checks,
                     'preflight_status': 'PASS' if preflight else 'FAIL', 'science_acceptance': 'ACCEPTED' if accepted else 'NOT_ACCEPTED'})
    return {'schema': 'observable-resolution-budget-v1', 'fingerprint': fingerprint,
            'target': target, 'request': request, 'census': particle_census, 'relaxation_force': limits, 'heating': heating,
            'observables': rows, 'preflight_status': 'PASS' if all(x['preflight_status']=='PASS' for x in rows) else 'FAIL',
            'science_acceptance': 'ACCEPTED' if all(x['science_acceptance']=='ACCEPTED' for x in rows) else 'NOT_ACCEPTED',
            'refinement': {'status': 'REQUIRES_REGENERATION_AND_REMEASUREMENT', 'limits': request['refinement'],
                           'route': 'commands.py with expanded fine region, reduced mass and species softening; regenerate model/phase-matched ICs and audit saved host apertures'},
            'convergence': measured, 'source_hashes':{str(Path(__file__).resolve()):file_sha256(__file__),
                **particle_census.get('source_hashes',{}),**physical_sources}}


def verify_resolution_convergence(path, fingerprint):
    """Strict retained-evidence intake. Never accept a manually supplied PASS."""
    import json
    path = Path(path).resolve()
    if path.with_name(path.name+'.sha256').read_text().strip() != file_sha256(path):
        raise ValueError('Resolution convergence seal mismatch')
    report = json.loads(path.read_text())
    if report.get('schema') != 'measured-resolution-convergence-v1' or report.get('fingerprint') != fingerprint:
        raise ValueError('Resolution convergence target/config mismatch')
    for source, checksum in report['source_hashes'].items():
        if file_sha256(source) != checksum:
            raise ValueError('Resolution convergence source changed')
    if not report['source_hashes'] or not report.get('comparisons'):
        raise ValueError('Convergence requires retained measurements')
    recomputed = compare_resolution_measurements(report['request'], report['comparisons'])
    if recomputed != report['observables']:
        raise ValueError('Resolution convergence decisions differ from measured comparisons')
    return {**report, 'receipt_path':str(path)}


def publish_resolution_budget(destination, budget, *, epoch, coordinates):
    """Seal measured census and preflight arrays through the shared contracts."""
    from core.data import export_numeric_audit
    c=budget['census'];force=budget['relaxation_force'];heat=budget['heating']
    arrays={'radius_proper_kpc':np.asarray(c['edges_proper_kpc'][1:],dtype=float),
        'species_count':np.asarray(c['count'],dtype=np.int64),
        'species_mass_msun':np.asarray(c['mass_msun'],dtype=float),
        'species_mass_squared_msun2':np.asarray(c['mass_squared_msun2'],dtype=float),
        'species_maximum_softening_proper_kpc':np.asarray(c['maximum_softening_proper_kpc'],dtype=float),
        'effective_count':np.asarray(force['effective_count'],dtype=float),
        'relaxation_age_ratio_estimate':np.asarray(force['relaxation_age_ratio_estimate'],dtype=float),
        'force_support_proper_kpc':np.asarray(force['force_support_proper_kpc'],dtype=float),
        'timestep_crossing_fraction':np.asarray(force['timestep_crossing_fraction'],dtype=float),
        'heating_variance_rate_estimate':np.asarray(heat['variance_rate_km2_s2_per_myr_estimate'],dtype=float)}
    units={'radius_proper_kpc':'proper kpc','species_count':'particle count','species_mass_msun':'Msun',
        'species_mass_squared_msun2':'Msun^2','species_maximum_softening_proper_kpc':'proper kpc',
        'effective_count':'dimensionless','relaxation_age_ratio_estimate':'dimensionless',
        'force_support_proper_kpc':'proper kpc','timestep_crossing_fraction':'dimensionless',
        'heating_variance_rate_estimate':'km^2/s^2/Myr'}
    specifications={k:{'units':units[k],'quantity':k.replace('_',' '),
        'annotations':{'method':'measured radial census and declared preflight estimates; no convergence inferred',
                       'species_axis':'native particle types 0 through 5' if v.ndim==2 else 'radial outer edges'}} for k,v in arrays.items()}
    # Numeric products do not grant global simulation acceptance. Preserve the
    # scoped resolution decision explicitly beside the generic product status.
    budget={**budget,'resolution_acceptance':{'status':budget['science_acceptance'],'scope':'only the declared observables/target/epoch',
        'requires':'measured physical prerequisites and mass/softening/timestep convergence'}}
    return export_numeric_audit(destination,budget,arrays,specifications,algorithm=__file__,
        context={'epoch':epoch,'coordinates':coordinates,'selection':'saved target-centered radial apertures',
                 'interpretation':'Analytic estimates are preflight; science acceptance requires independent native convergence'},
        source_hashes=budget['source_hashes'])


def compare_resolution_measurements(request, comparisons):
    """Compare actual producer vectors; every observable needs all three axes.

    Each axis is a pair of verified numeric-audit products. Their reports must
    declare observable units, same physical selection/epoch/model, and differing
    numerical mass/softening/timestep. Synthetic fixtures cannot grant science.
    """
    from core.data import verify_numeric_audit
    import json
    validate_resolution_request(request)
    decisions = {}
    for obs in request['observables']:
        tests = []
        for axis in ('mass', 'softening', 'timestep'):
            pair = comparisons.get(obs['name'], {}).get(axis)
            reason = None
            residual = None
            try:
                if not isinstance(pair, dict) or set(pair) != {'coarse', 'fine', 'dataset'}:
                    raise ValueError('Missing paired '+axis+' convergence measurement')
                products = [verify_numeric_audit(pair[k]) for k in ('coarse', 'fine')]
                if any(p['numeric_data']['array_contract']['datasets'][pair['dataset']]['units']!=obs['units'] for p in products):
                    raise ValueError('Convergence dataset units differ from the requested observable')
                provenance = [p['resolution_measurement'] for p in products]
                for meta in provenance:
                    if meta['origin'] != 'native_simulation' or meta['units'] != obs['units'] or not meta['source_snapshots']:
                        raise ValueError('Only native source-bound measurements with matching units establish convergence')
                    for s in meta['source_snapshots']:
                        if file_sha256(s['path']) != s['sha256']:
                            raise ValueError('Native measurement snapshot changed')
                        import h5py
                        with h5py.File(s['path']) as native:
                            if not math.isclose(float(native['Header'].attrs['Time']),meta['actual_scale_factor'],rel_tol=1e-12):
                                raise ValueError('Observable epoch differs from its native source')
                    from core.data import verify_native_hdf5_set
                    verify_native_hdf5_set([s['path'] for s in meta['source_snapshots']],require_publication=True)
                    actual_settings=measure_native_resolution_settings(meta['source_snapshots'],obs['species'],meta['actual_scale_factor'])
                    if any(not math.isclose(actual_settings[key],positive(meta['numerical_settings'][key],key),rel_tol=1e-10) for key in actual_settings):
                        raise ValueError('Declared numerical settings differ from actual native snapshots')
                for field in ('selection', 'actual_scale_factor', 'physics_fingerprint', 'observable', 'bin_coordinates'):
                    if provenance[0][field] != provenance[1][field]:
                        raise ValueError('Convergence measurement '+field+' mismatch')
                if provenance[0]['observable'] != obs['name']:
                    raise ValueError('Convergence observable mismatch')
                bins=np.asarray(provenance[0]['bin_coordinates'],dtype=float)
                if bins.ndim!=1 or not len(bins) or not np.isfinite(bins).all():
                    raise ValueError('Convergence requires explicit finite measurement bins')
                if any(meta.get('all_requested_bins_valid') is not True for meta in provenance):
                    raise ValueError('Unresolved or failed observable bins cannot grant convergence')
                coarse_settings, fine_settings = (p['numerical_settings'] for p in provenance)
                if positive(coarse_settings[axis], axis) <= positive(fine_settings[axis], axis):
                    raise ValueError('Fine setting must improve requested numerical axis')
                if any(coarse_settings[k] != fine_settings[k] for k in ('mass', 'softening', 'timestep') if k != axis):
                    raise ValueError('Convergence comparison changes unrelated numerical axes')
                values = []
                for key, p in zip(('coarse', 'fine'), products):
                    archive = Path(pair[key]).parent / p['numeric_data']['directory'] / 'arrays.npz'
                    with np.load(archive, allow_pickle=False) as data:
                        values.append(np.asarray(data[pair['dataset']], dtype=float))
                a, b = values
                if a.shape != b.shape or not a.size or a.ndim==0 or a.shape[0]!=len(bins) or not np.isfinite(a).all() or not np.isfinite(b).all():
                    raise ValueError('Empty, failed or unmatched observable vectors')
                residual = float(np.max(np.abs(a-b)/np.maximum(np.abs(b), obs['absolute_error_floor'])))
                if residual > obs['maximum_relative_error']:
                    raise ValueError('Measured residual exceeds declared tolerance')
            except (ValueError, KeyError, OSError, TypeError) as exc:
                reason = str(exc)
            tests.append({'axis': axis, 'maximum_normalized_residual': residual, 'status': 'PASS' if reason is None else 'FAIL', 'reason': reason})
        if obs['kind']=='cold_stream':
            reason=None;variance=None
            try:
                control=comparisons[obs['name']]['null_stream_heating']
                product=verify_stream_heating_calibration(control['report'])
                meta=product['resolution_measurement']
                if meta['origin']!='native_simulation' or meta.get('control')!='matched_smooth_potential_null_stream' or meta['units']!='km^2/s^2' or meta['observable']!=obs['name'] or not meta['source_snapshots']:
                    raise ValueError('Cold-stream acceptance needs a native matched null-stream heating control')
                reference=verify_numeric_audit(comparisons[obs['name']]['timestep']['fine'])['resolution_measurement']
                if any(meta[field]!=reference[field] for field in ('selection','actual_scale_factor','physics_fingerprint')):
                    raise ValueError('Null-stream control differs from the requested population, epoch or model')
                if meta['bin_coordinates'] != reference['bin_coordinates']:
                    raise ValueError('Null-stream control bins differ from the requested population')
                if meta.get('all_requested_bins_valid') is not True or positive(meta['duration_myr'],'null stream duration')<obs['duration_myr']:
                    raise ValueError('Null-stream control must cover the requested duration with valid bins')
                if product['numeric_data']['array_contract']['datasets'][control['dataset']]['units']!='km^2/s^2':
                    raise ValueError('Null-stream dataset variance units mismatch')
                for source in meta['source_snapshots']:
                    if file_sha256(source['path'])!=source['sha256']:raise ValueError('Null-stream native source changed')
                from core.data import verify_native_hdf5_set
                verify_native_hdf5_set([s['path'] for s in meta['source_snapshots']],require_publication=True)
                settings = measure_native_resolution_settings(meta['source_snapshots'], obs['species'], meta['actual_scale_factor'])
                if settings != reference['numerical_settings']:
                    raise ValueError('Null-stream control numerical settings differ from the accepted fine comparison')
                archive=Path(control['report']).parent/product['numeric_data']['directory']/'arrays.npz'
                with np.load(archive,allow_pickle=False) as data:
                    v=np.asarray(data[control['dataset']],dtype=float)
                if v.shape != (len(meta['bin_coordinates']),) or not v.size or not np.isfinite(v).all() or np.any(v<0):raise ValueError('Null-stream variance must be a finite nonnegative matched bin vector')
                variance=float(v.max())
                if variance>obs['dispersion_km_s']**2*obs['maximum_heating_variance_fraction']:
                    raise ValueError('Measured null-stream heating exceeds dispersion budget')
            except (ValueError,KeyError,TypeError,OSError) as exc:
                reason=str(exc)
            tests.append({'axis':'null_stream_heating','maximum_variance_km2_s2':variance,'status':'PASS' if reason is None else 'FAIL','reason':reason})
        decisions[obs['name']] = {'status': 'PASS' if all(t['status']=='PASS' for t in tests) else 'FAIL', 'tests': tests}
    return decisions


def measure_native_resolution_settings(sources, species, scale_factor):
    """Verify convergence axes against saved masses, force classes and timestep.

    Mass is the maximum actual representative mass of the requested species;
    epsilon is its maximum proper Plummer-equivalent softening. Timestep is the
    native maximum Delta ln(a), not an invented physical-time measurement.
    """
    import h5py
    from halos import _unit_conversion
    from core.data import dataset_to_cgs_factor
    from zoom import proper_species_softening
    paths=[Path(s['path']).resolve() for s in sources]
    _unit_conversion(paths)
    maximum_mass=0.;maximum_softening=0.;timestep=None
    for path in paths:
        with h5py.File(path) as native:
            params=native['Parameters'].attrs
            step=positive(params['MaxSizeTimestep'],'native maximum timestep')
            if timestep is not None and timestep!=step:raise ValueError('Convergence shard timesteps differ')
            timestep=step
            h=positive(params['HubbleParam'],'native Hubble parameter')
            group=native.get(f'PartType{species}')
            if group is None:continue
            for start in range(0,len(group['Coordinates']),100_000):
                end=min(start+100_000,len(group['Coordinates']))
                mass=np.asarray(group['Masses'][start:end],dtype=float) if 'Masses' in group else np.full(end-start,float(native['Header'].attrs['MassTable'][species]))
                if not np.isfinite(mass).all() or np.any(mass<=0):raise ValueError('Invalid convergence source masses')
                conversion=dataset_to_cgs_factor(group['Masses'].attrs,scale_factor,h)/MSUN_G if 'Masses' in group else positive(params['UnitMass_in_g'],'native mass unit')/h/MSUN_G
                maximum_mass=max(maximum_mass,float(mass.max())*conversion)
                classes=group['GravitySofteningClass'][start:end] if 'GravitySofteningClass' in group else None
                softening=proper_species_softening(params,species,scale_factor,h,classes)
                maximum_softening=max(maximum_softening,float(np.max(softening)))
    if maximum_mass<=0 or maximum_softening<=0:raise ValueError('Requested convergence species is empty')
    return {'mass':maximum_mass,'softening':maximum_softening,'timestep':timestep}


@directory_owner('destination')
def publish_resolution_convergence(destination, fingerprint, request, comparisons):
    """Durable immutable comparison receipt; producers retain numeric contracts."""
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    decisions = compare_resolution_measurements(request, comparisons)
    sources = {str(Path(__file__).resolve()): file_sha256(__file__)}
    from core.data import verify_numeric_audit
    for group in comparisons.values():
        for axis,pair in group.items():
            for key in (('report',) if axis=='null_stream_heating' else ('coarse', 'fine')):
                p = Path(pair[key]).resolve()
                report = verify_numeric_audit(p)
                sources[str(p)] = file_sha256(p)
                for s in report['resolution_measurement']['source_snapshots']:
                    sources[str(Path(s['path']).resolve())] = s['sha256']
    result = {'schema': 'measured-resolution-convergence-v1', 'fingerprint': fingerprint, 'request': request,
              'comparisons': comparisons, 'observables': decisions, 'source_hashes': sources,
              'status': 'PASS' if all(x['status']=='PASS' for x in decisions.values()) else 'FAIL'}
    atomic_json(destination, result)
    atomic_text(destination.with_name(destination.name+'.sha256'), file_sha256(destination)+'\n')
    return result


def validate_target_budget_request(parameters):
    validate_resolution_request(parameters['request'])
    positive(parameters['memory_gib'],'physical resolution budget workspace')
    if not parameters['request'].get('physical_support'):raise ValueError('Physical-target budget needs measured target support products')
    for name in ('age_myr','maximum_proper_timestep_myr','relative_speed_km_s'):
        positive(parameters['target'][name],name)
    return parameters


def target_budget_input_bindings(parameters):
    from core.sidm_analysis import _load_product
    validate_target_budget_request(parameters);memory=parameters['memory_gib']*1024**3
    parent,_,sources=_load_product(parameters['primary_support'],parameters['request_base'],memory)
    if parent.get('schema')!='physical-target-support-v1':raise ValueError('Physical budget needs a measured full native particle census')
    target={**parameters['target'],'target_identity':parent['target_identity'],'actual_scale_factor':parent['epoch']['actual_scale_factor']}
    request=_target_budget_bound_request(parameters)
    _,bindings=_resolution_physical_support(target,parent['particle_census'],request);sources.update(bindings)
    if parameters.get('convergence'):
        path=Path(parameters['convergence']);path=(Path(parameters['request_base'])/path if not path.is_absolute() else path).resolve()
        raw=json.loads(path.read_text());sources.update(raw['source_hashes']);sources[str(path)]=file_sha256(path)
        sources[str(path.with_name(path.name+'.sha256'))]=file_sha256(path.with_name(path.name+'.sha256'))
    return sources


def _target_budget_bound_request(parameters):
    request=json.loads(json.dumps(parameters['request']))
    for reference in request['physical_support'].values():
        path=Path(reference['path']).expanduser()
        reference['path']=str((Path(parameters['request_base'])/path if not path.is_absolute() else path).resolve())
        if reference['memory_gib']>parameters['memory_gib']:raise MemoryError('Physical evidence intake exceeds budget allocation')
    return request


def publish_target_resolution_budget(parameters,destination):
    from core.sidm_analysis import _load_product
    sources=target_budget_input_bindings(parameters)
    parent,_,_=_load_product(parameters['primary_support'],parameters['request_base'],parameters['memory_gib']*1024**3)
    target={**parameters['target'],'target_identity':parent['target_identity'],'actual_scale_factor':parent['epoch']['actual_scale_factor']}
    request=_target_budget_bound_request(parameters);convergence=parameters.get('convergence')
    if convergence:
        path=Path(convergence);convergence=(Path(parameters['request_base'])/path if not path.is_absolute() else path).resolve()
    budget=evaluate_resolution(target,parent['particle_census'],request,convergence=convergence)
    budget['source_hashes'].update(sources)
    return publish_resolution_budget(destination,budget,epoch=parent['epoch'],coordinates={'frame':'actual native physical target center','center_proper_kpc':parent['particle_census']['center_proper_kpc']})


def _physical_target_stage(function,parameters,context):
    p=dict(parameters);p.setdefault('request_base',str(context.base))
    memory=p.get('memory_gib',p['request'].get('memory_gib',p['request'].get('refinement',{}).get('memory_gib')))
    if positive(memory,'physical target stage workspace')>context.resources['memory_gib']:
        raise MemoryError('Physical target stage exceeds its workflow memory allocation')
    destination=Path(p['destination']);destination=(context.generation/destination if not destination.is_absolute() else destination).resolve()
    if not destination.is_relative_to(context.generation):raise ValueError('Physical target publication escapes its owned workflow generation')
    return function(p,destination)


def physical_target_support_runner(parameters,context):
    return _physical_target_stage(publish_physical_target_support,parameters,context)


def physical_target_model_runner(parameters,context):
    return _physical_target_stage(publish_target_model_controls,parameters,context)


def physical_target_measurement_runner(parameters,context):
    return _physical_target_stage(publish_target_resolution_measurement,parameters,context)


def physical_target_budget_runner(parameters,context):
    return _physical_target_stage(publish_target_resolution_budget,parameters,context)


def validate_target_convergence_request(parameters):
    validate_resolution_request(parameters['request']);positive(parameters['memory_gib'],'physical convergence workspace')
    if not isinstance(parameters['fingerprint'],str) or len(parameters['fingerprint'])!=64:raise ValueError('Exact physical resolution budget fingerprint required')
    return parameters


def target_convergence_input_bindings(parameters):
    from core.sidm_analysis import _load_product
    validate_target_convergence_request(parameters);sources={}
    for group in parameters['comparisons'].values():
        for axis,pair in group.items():
            for name in (('report',) if axis=='null_stream_heating' else ('coarse','fine')):
                path=Path(pair[name]);path=(Path(parameters['request_base'])/path if not path.is_absolute() else path).resolve()
                _,_,bindings=_load_product({'path':str(path),'sha256':file_sha256(path)},'.',parameters['memory_gib']*1024**3)
                sources.update(bindings)
    return sources


def _publish_target_convergence(parameters,destination):
    target_convergence_input_bindings(parameters)
    comparisons=json.loads(json.dumps(parameters['comparisons']))
    for group in comparisons.values():
        for axis,pair in group.items():
            for name in (('report',) if axis=='null_stream_heating' else ('coarse','fine')):
                path=Path(pair[name]);pair[name]=str((Path(parameters['request_base'])/path if not path.is_absolute() else path).resolve())
    return publish_resolution_convergence(destination,parameters['fingerprint'],parameters['request'],comparisons)


def physical_target_convergence_runner(parameters,context):
    return _physical_target_stage(_publish_target_convergence,parameters,context)


def physical_target_stage_input_bindings(kind,parameters):
    functions={'physical_target_support':physical_target_input_bindings,'physical_target_model_control':target_model_control_input_bindings,
        'physical_target_measurement':target_measurement_input_bindings,'physical_target_budget':target_budget_input_bindings,
        'physical_target_convergence':target_convergence_input_bindings}
    return functions[kind](parameters)


def physical_target_main():
    import argparse
    parser=argparse.ArgumentParser(description='Produce source-bound first-cloud/BH support, measured growth/relaxation controls and resolution evidence')
    parser.add_argument('mode',choices=('support','model-control','measurement','budget','convergence'))
    parser.add_argument('--parameters',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();parameters=json.loads(args.parameters.read_text())
    parameters.setdefault('request_base',str(args.parameters.resolve().parent))
    if args.mode=='convergence':
        return _publish_target_convergence(parameters,args.output)
    functions={'support':publish_physical_target_support,'model-control':publish_target_model_controls,
        'measurement':publish_target_resolution_measurement,'budget':publish_target_resolution_budget}
    return functions[args.mode](parameters,args.output)


@directory_owner('destination')
def stage_resolution_refinement(destination, *, parent_staging_directory,
                                budget_receipt, base_request, refinement_levels,
                                new_seeds_by_level, padding_cells):
    """Execute real model/phase-preserving staging at a higher MUSIC level.

    Each new level reduces fine cell masses by eight; species velocities,
    transfers, parent phases and LPT prescription use the existing generator
    route. Native mesh/resource and target coverage gates are still enforced by
    run_zoom.py. This step cannot certify regenerated/evolved convergence.
    """
    import json
    from zoom import verify_staged_zoom_inputs,stage_zoom_inputs
    parent=Path(parent_staging_directory).resolve()
    original=verify_staged_zoom_inputs(parent)
    budget_path=Path(budget_receipt).resolve()
    if budget_path.with_name('provenance.sha256').read_text().strip()!=file_sha256(budget_path):
        raise ValueError('Resolution budget receipt seal mismatch')
    budget=json.loads(budget_path.read_text())
    records=budget.get('records',[])
    if not records or any('resolution_budget' not in row for row in records):
        raise ValueError('Refinement requires measured saved-aperture budgets')
    limits=records[0]['resolution_budget']['request']['refinement']
    if type(refinement_levels) is not int or not 1<=refinement_levels<=limits['maximum_levels']:
        raise ValueError('Refinement exceeds the declared follow-up limit')
    if limits['mass_reduction_factor']!=8:
        raise ValueError('MUSIC refinement uses an eightfold cell-mass reduction per added level')
    for row in records:
        measured=row['resolution_budget']
        generation=measured['refinement'].get('native_generation_source')
        if generation is None:
            raise ValueError('Resolution refinement requires verified MUSIC generation lineage for the measured target')
        from zoom import verify_generated_zoom
        if file_sha256(generation['path'])!=generation['sha256']:
            raise ValueError('Measured target generation source changed')
        generated=verify_generated_zoom(generation['path'])
        config=generated['source_configuration']
        if Path(config['path']).resolve().parent!=parent or file_sha256(config['path'])!=config['sha256']:
            raise ValueError('Requested parent staging did not generate the measured target ICs')
        if measured['request']['refinement']!=limits:raise ValueError('Epoch refinement limits differ')
        for p,h in measured['source_hashes'].items():
            if file_sha256(p)!=h:raise ValueError('Resolution census source changed')
    for key in ('box_mpc_h','parent_level','refinement_level','seeds_by_level','phase_noise_by_level'):
        value=base_request.get(key,{} if key=='phase_noise_by_level' else None)
        if value!=original[key]:raise ValueError('Refinement changes parent '+key)
    if str(Path(base_request['spectrum_manifest']).resolve())!=original['solver_manifest']['path']:
        raise ValueError('Refinement changes the model/epoch solver bundle')
    if base_request.get('lpt_order',1)!=original['perturbation_order'] or base_request.get('lpt_seed_kind','radiation_regular')!=original['lpt_seed_kind'] or base_request.get('lpt_seed')!=original['lpt_seed']:
        raise ValueError('Refinement changes the LPT prescription')
    high=original['refinement_level']+refinement_levels
    required={str(i) for i in range(original['refinement_level']+1,high+1)}
    if not isinstance(new_seeds_by_level,dict) or set(new_seeds_by_level)!=required:
        raise ValueError('Choose explicit seeds for every added refinement level')
    if type(padding_cells) is not int or padding_cells<base_request['padding_cells']:
        raise ValueError('Fine-region guard cannot shrink during refinement')
    child={**base_request,'destination':str(Path(destination).resolve()),'refinement_level':high,
           'seeds_by_level':{**original['seeds_by_level'],**new_seeds_by_level},'padding_cells':padding_cells}
    staged=stage_zoom_inputs(**child)
    lineage={'schema':'resolution-refinement-lineage-v1','status':'STAGED','parent_staging':str(parent),
             'parent_staging_sha256':file_sha256(parent/'staging.json'),'budget_receipt':str(budget_path),
             'budget_sha256':file_sha256(budget_path),'child_staging_sha256':file_sha256(Path(destination)/'staging.json'),
             'added_levels':refinement_levels,'fine_mass_reduction':8**refinement_levels,
             'native_memory_gib':limits['memory_gib'],'native_disk_gib':limits['disk_gib'],
             'required_next_stage':'commands.py with parent IC/noise, traced region and native mesh/resource validation',
             'science_acceptance':'NOT_ACCEPTED'}
    pointer=Path(destination).with_name(Path(destination).name+'.refinement.json')
    if pointer.exists():raise FileExistsError(pointer)
    atomic_json(pointer,lineage)
    atomic_text(pointer.with_name(pointer.name+'.sha256'),file_sha256(pointer)+'\n')
    return {'staging':staged,'lineage':lineage}

# ========================================================================
# zoom_generation: Native zoom input staging, generation and GADGET handoff.
from core.resources import initial_condition_census
from core.foundation import fsync_directory
from core.data import copy_numeric_table
from core.cosmology import write_species_transfer_validation_table, write_monofonic_ic_transfer_table, ede_parameters
from core.foundation import physical_constants_contract
from core.data import verify_numeric_audit
import time
import io


# Zoom generation
#
# Run the pinned native zoom generator and retain explicit acceptance state.
#
# This stage supplies audited ICs, not science acceptance or particle evolution.
# Model/phase matching and target-halo coverage remain separate required gates.

def validate_native_music_regression(binary, native_validation):
    """The existing generator admission gate, shared with workflow preflight."""
    binary=Path(binary).resolve();native_validation=Path(native_validation).resolve()
    proof = json.loads(native_validation.read_text())
    driver = Path(__file__).resolve().parents[1] / 'core/research/generator_controls.py'
    if (
        proof.get("status") != "PASS"
        or proof.get("execution", {}).get("binary_sha256") != file_sha256(binary)
        or proof.get("regression_driver_sha256") != file_sha256(driver)
        or proof.get("nested", {}).get("status") != "PASS"
        or proof.get("periodic", {}).get("status") != "PASS"
    ):
        raise ValueError(
            "The current native binary must pass the current nested transfer regression suite"
        )
    rejections = proof.get("execution", {}).get("rejections", {})
    required = {
        "reject-second-order",
        "reject-backscaling",
        "reject-gauge",
        "reject-model",
        "reject-redshift",
        "reject-mode-coverage",
        "reject-poisson",
    }
    if set(rejections) != required or any(
        row.get("exit_code") != 1 for row in rejections.values()
    ):
        raise ValueError(
            "Native validation is missing the required configuration/Poisson failure checks"
        )
    return proof


@directory_owner('destination')
def generate_zoom(
    configuration,
    binary,
    destination,
    *,
    native_validation,
    traced_region=None,
    threads=2,
    memory_gib=1.0,
    parent_ic=None,
    parent_noise=None,
    native_memory_gib=4.0,
    native_disk_gib=None,
    native_wall_hours=None,
):
    configuration = Path(configuration).expanduser().resolve()
    binary = Path(binary).expanduser().resolve()
    destination = Path(destination).expanduser().resolve()
    native_memory_gib,native_disk_gib,resolution_lineage=refinement_generation_limits(configuration,native_memory_gib,native_disk_gib)
    if isinstance(native_memory_gib,bool) or not isinstance(native_memory_gib,(int,float)) or not math.isfinite(native_memory_gib) or native_memory_gib<=0:
        raise ValueError('Positive finite actual native generation allocation required')
    from core.resources import memory_limit_gb
    assigned_memory_gib=memory_limit_gb({'hardware':{'memory_limit_gb':'auto'}})
    if native_memory_gib>assigned_memory_gib:
        raise MemoryError('Actual native generation allocation exceeds current assigned host memory; planning larger hardware does not allocate it')
    if type(threads) is not int or not 1 <= threads <= 1024:
        raise ValueError("Generator thread count must be an integer in [1,1024]")
    if not math.isfinite(memory_gib) or memory_gib <= 0:
        raise ValueError("Audit memory budget must be finite and positive")
    if native_wall_hours is not None and (isinstance(native_wall_hours,bool)
            or not math.isfinite(native_wall_hours) or native_wall_hours <= 0):
        raise ValueError('Native wall-time budget must be finite and positive')
    if not configuration.is_file() or not binary.is_file():
        raise FileNotFoundError(
            "A saved MUSIC configuration and built native binary are required"
        )
    if destination.exists():
        raise FileExistsError(
            "Refusing to overwrite zoom generation artifacts: " + str(destination)
        )
    native_validation = Path(native_validation).expanduser().resolve()
    proof=validate_native_music_regression(binary,native_validation)
    cfg = configparser.ConfigParser(interpolation=None)
    # MUSIC names are case sensitive (Omega_m, A_s, H0, use_2LPT).
    # ConfigParser's default lowercasing would silently select native defaults.
    cfg.optionxform = str
    cfg.read(configuration)
    low, high = cfg.getint('setup','levelmin'), cfg.getint('setup','levelmax')
    if not 1 <= low <= high <= 20:
        raise ValueError('Native MUSIC generation requires bounded parent/refinement levels')
    if cfg.getboolean('darkmattersim','requires_traced_region',fallback=False) and traced_region is None:
        raise ValueError('Model-matched zoom template requires an actual traced catalogue/aperture region')
    region_record = None
    if traced_region is not None:
        from zoom import apply_traced_region

        region_record = apply_traced_region(cfg, traced_region)
    if cfg.get("cosmology", "transfer") != "DMS_file":
        raise ValueError(
            "This generation route requires explicit N-body-gauge species transfers"
        )
    if (
        not cfg.getboolean("setup", "baryons")
        or (cfg.getboolean("setup", "use_2LPT", fallback=False)
            and not cfg.getboolean("cosmology","mixed_2lpt",fallback=False))
        or cfg.getboolean("setup", "use_LLA", fallback=False)
    ):
        raise ValueError(
            "The validated DMS_file route requires two-fluid 1LPT or the explicit mixed 2LPT profile"
        )
    if (
        cfg.get("output", "format") != "arepo"
        or cfg.getint("output", "arepo_num_files") != 1
    ):
        raise ValueError("Zoom generation requires a single HDF5 arepo output")
    if cfg.getint("output", "arepo_coarsetype") != 2:
        raise ValueError(
            "Coarse matter must use type 2; type 5 is reserved for black holes"
        )
    if not cfg.getboolean("output", "arepo_doubleprec") or not cfg.getboolean(
        "output", "arepo_longids"
    ):
        raise ValueError("Zoom generation requires double coordinates and 64-bit IDs")
    staging = None
    if (configuration.parent/'staging.json').is_file():
        staging = verify_staged_zoom_inputs(configuration.parent)
    source_table = Path(cfg.get("cosmology", "transfer_file")).expanduser()
    if not source_table.is_absolute():
        source_table = configuration.parent / source_table
    source_table = source_table.resolve()
    if not source_table.is_file():
        raise FileNotFoundError("Native species-transfer table does not exist")
    verify_numeric_table(source_table)
    order = 2 if cfg.getboolean('setup','use_2LPT',fallback=False) else 1
    if order == 2:
        require_mixed_lpt_validation(proof,binary)
        if staging is None or staging.get('perturbation_order') != 2:
            raise ValueError('Mixed-fluid 2LPT generation requires sealed background-matched staging')
    elif cfg.getboolean('cosmology','mixed_2lpt',fallback=False):
        raise ValueError('Mixed-fluid kernel profile requires declared LPT order 2')
    input_model = cfg.get('cosmology', 'input_model', fallback=None)
    if input_model not in {'lcdm', 'ede', 'wdm'}:
        raise ValueError('Declare a supported cosmology.input_model for native velocity validation')
    if input_model == 'wdm' and (staging is None or staging.get('wdm') is None):
        raise ValueError('WDM generation requires sealed physical relic staging and resolution diagnostics')
    if (parent_ic is None) != (parent_noise is None):
        raise ValueError('Phase matching requires both parent_ic and parent_noise')
    parent_paths = None
    if parent_ic is not None:
        parent_paths = [Path(p).expanduser().resolve() for p in (parent_ic, parent_noise)]
        if any(not p.is_file() for p in parent_paths):
            raise FileNotFoundError('Parent IC and parent coarse-noise file must exist before generation')
    # Native mesh planning occurs before the generator allocates field grids.
    destination.mkdir(parents=True, exist_ok=False)
    cfg.set("cosmology", "transfer_file", "transfers.txt")
    cfg.set("output", "filename", "ics_gadget.hdf5")
    noise_files=_music_noise_files(cfg,configuration)
    for level,path in noise_files.items():cfg['random'][f'seed[{level}]']=str(path)
    if _frozen_noise_workspace(cfg,noise_files):
        cfg['random']['frozen_noise_memory_mib']=format(native_memory_gib*1024,'.17g')
    staged_cfg = destination / "music.conf"
    log_path = destination / "generation.log"
    report_path = destination / "generation.json"
    report = {
        "schema_version": 1,
        'resolution_refinement':resolution_lineage,
        "status": "IN_PROGRESS",
        "source_configuration": {
            "path": str(configuration),
            "sha256": file_sha256(configuration),
        },
        "source_transfer_table": {
            "path": str(source_table),
            "sha256": file_sha256(source_table),
        },
        "binary": {"path": str(binary), "sha256": file_sha256(binary)},
        "native_validation": {
            "path": str(native_validation),
            "sha256": file_sha256(native_validation),
        },
        "threads": threads,
        "perturbation_order":order,
        "traced_region": region_record,
        'parent_phase_inputs': None if parent_paths is None else {
            str(p): file_sha256(p) for p in parent_paths},
        "audit_memory_gib": memory_gib,
        'native_wall_hours': native_wall_hours,
        "native_mesh_memory_gate": "PENDING",
        "science_acceptance": "NOT_ACCEPTED",
        "remaining_science_gates": [
            "CLASS/AxiCLASS source and cosmology identity",
            "primordial spectrum and parent phase agreement",
            "traced target Lagrangian coverage",
            "mass-dependent force resolution",
            "per-epoch contamination",
            "start-redshift and resolution convergence",
        ],
    }
    try:
        copy_numeric_table(source_table, destination / "transfers.txt")
        if file_sha256(destination / 'transfers.txt') != report['source_transfer_table']['sha256']:
            raise ValueError('Transfer input changed during staging')
        # Retain supplied lineage; copying it does not independently validate
        # the solver or grant acceptance to a file claiming to be model matched.
        source_sidecar = Path(str(source_table) + ".json")
        if source_sidecar.is_file():
            atomic_copy(source_sidecar, destination / "transfers.txt.json")
            report["transfer_sidecar_sha256"] = file_sha256(source_sidecar)
        stream=io.StringIO()
        cfg.write(stream)
        atomic_text(staged_cfg,stream.getvalue())
        atomic_json(report_path, report)
        report['native_resource_plan'] = plan_native_music(staged_cfg, binary, destination/'planning',
            memory_gib=native_memory_gib, disk_gib=native_disk_gib, threads=threads)
        report['native_mesh_memory_gate'] = 'ESTIMATE_PASS'
        atomic_json(report_path, report)
        with log_path.open("w") as stream:
            started = time.monotonic()
            try:
                completed = run_native_command(
                    [str(binary), "music.conf"],
                    cwd=destination,
                    receipt_path=destination/"native-invocation.json",
                    inputs=[destination/"transfers.txt",native_validation,*noise_files.values()],check=False,
                    env={**os.environ, "OMP_NUM_THREADS": str(threads)},
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    timeout=None if native_wall_hours is None else native_wall_hours*3600,
                )
            finally:
                report['native_execution_wall_seconds'] = time.monotonic()-started
        report["native_exit_code"] = completed.returncode
        if completed.returncode != 0:
            raise RuntimeError(
                "Native zoom generation failed; partial files are not accepted. See "
                + str(log_path)
            )
        ic = destination / "ics_gadget.hdf5"
        if cfg.getboolean('output','retain_source_hierarchy',fallback=False):
            from core.music.archive import index_music_source_archive
            report['source_archive_index']=index_music_source_archive(destination/'source-hierarchy.bin',
                destination/'source-hierarchy-index.json',memory_gib=memory_gib)
        if input_model == 'wdm':
            from initial_conditions import apply_native_wdm_thermal
            report['wdm_thermal'] = apply_native_wdm_thermal(ic,staging['wdm']['configuration'])
            report['wdm_staging'] = staging['wdm']
        input_kind = 'parent' if cfg.getint('setup','levelmin') == cfg.getint('setup','levelmax') else 'zoom'
        audit = audit_music_ic(ic, staged_cfg, memory_gib=memory_gib, input_kind=input_kind)
        report["structural_audit"] = audit
        if audit.get('status') != 'STRUCTURAL_PASS':
            raise ValueError('Structural audit did not pass')
        if input_kind=='zoom' and cfg.getboolean('output','retain_source_hierarchy',fallback=False):
            from core.music.replay import audit_music_archive_transfer
            report['source_transfer_replay']=audit_music_archive_transfer(
                destination/'source-hierarchy-index.json',destination/'music.conf',ic,
                {'schema':'native-music-transfer-replay-request-v1',
                 'mesh':str(destination/'planning/native-mesh.json'),'noise_directory':str(destination)},
                destination/'source-transfer-replay.json',memory_gib=memory_gib)
            if report['source_transfer_replay']['status']!='PASS':
                raise ValueError('Generated original source noise/transfer replay failed')
        atomic_json(report_path, report)
        if staging is not None:
            from initial_conditions import validate_ic_realization
            physical = validate_ic_realization(ic,staged_cfg,destination/'physical-audit.json',
                memory_gib=memory_gib,input_kind=input_kind)
            report['physical_audit'] = physical
            if physical['status'] == 'FAIL':
                raise ValueError('Delivered native IC conservation audit failed; see physical-audit.json')
        velocity = audit_zoom_velocity(ic, destination / 'transfers.txt', destination / 'velocity-audit.json',
                                       model=input_model, memory_gib=memory_gib)
        report['velocity_audit'] = velocity
        atomic_json(report_path, report)
        if velocity.get('status') != 'PASS':
            raise ValueError('Delivered native IC velocity audit failed; see velocity-audit.json')
        if parent_paths is not None:
            level = cfg.getint('setup', 'levelmin')
            noise = destination / f'wnoise_{level:04d}.bin'
            phase = compare_density_phases(parent_paths[0], ic, parent_paths[1], noise,
                                           destination / 'phase-audit.json', memory_gib=memory_gib)
            report['phase_audit'] = phase
            atomic_json(report_path, report)
            if phase.get('status') != 'PASS':
                raise ValueError('Delivered native IC parent phases failed; see phase-audit.json')
            if any(file_sha256(p) != report['parent_phase_inputs'][str(p)] for p in parent_paths):
                raise ValueError('Parent phase inputs changed during generation/auditing')
        report["status"] = "GENERATED_STRUCTURAL_PASS"
        report['linear_audit_status'] = 'PASS' if parent_paths is not None else 'INCOMPLETE_NO_PARENT_PHASES'
        report["artifacts"] = {
            name: file_sha256(destination / name)
            for name in (
                "music.conf",
                "transfers.txt",
                "ics_gadget.hdf5",
                "generation.log",
            )
        }
        report['artifacts'].update({str(path.relative_to(destination)):file_sha256(path)
            for path in sorted((destination/'planning').rglob('*')) if path.is_file()})
        if traced_region is not None:
            # Coverage binds to an immutable structural-stage record. Keep it
            # separate from the final report so its hash cannot refer to itself.
            structural_path = destination / 'generation-structural.json'
            atomic_json(structural_path, report)
            coverage_path = destination / 'target-coverage.json'
            coverage = audit_target_coverage(structural_path, traced_region, coverage_path,
                                             memory_gib=memory_gib)
            report['static_target_coverage'] = coverage
            report['artifacts']['generation-structural.json'] = file_sha256(structural_path)
            report['artifacts']['target-coverage.json'] = file_sha256(coverage_path)
            if coverage.get('status') != 'PASS':
                raise ValueError('Generated coarse cells intersect traced target guards; see target-coverage.json')
    except BaseException as error:
        report["status"] = 'INTERRUPTED' if isinstance(error, (KeyboardInterrupt, SystemExit)) else "FAILED"
        report["error"] = str(error)
        if log_path.is_file():
            report["generation_log_sha256"] = file_sha256(log_path)
        atomic_json(report_path, report)
        raise
    for name in ('source_configuration','source_transfer_table','binary','native_validation'):
        entry=report[name]
        if file_sha256(entry['path'])!=entry['sha256']:
            report.update(status='FAILED',error='Native generation input changed: '+name)
            atomic_json(report_path,report)
            raise ValueError(report['error'])
    report['publication_contract']='native-ic-generation-v1'
    report['artifacts']={path.relative_to(destination).as_posix():file_sha256(path)
        for path in destination.rglob('*') if path.is_file()
        and '.native-run-locks' not in path.relative_to(destination).parts
        and path.name not in ('generation.json','generation.sha256')}
    for name in report['artifacts']:
        with (destination/name).open('rb') as stream:os.fsync(stream.fileno())
    for directory in sorted((p for p in destination.rglob('*') if p.is_dir()),key=lambda p:len(p.parts),reverse=True):
        fsync_directory(directory)
    fsync_directory(destination)
    atomic_json(report_path, report)
    atomic_text(destination/'generation.sha256',file_sha256(report_path)+'\n')
    return report


@directory_owner('report_path')
def verify_generated_zoom(report_path):
    """Verify immutable generated bytes before handoff or coverage consumption."""
    from core.data import verify_numeric_audit
    path=Path(report_path).resolve();root=path.parent
    before=file_sha256(path)
    report=json.loads(path.read_text())
    if report.get('resolution_refinement'):
        entry=report['resolution_refinement']
        if file_sha256(entry['path'])!=entry['sha256']:
            raise ValueError('Resolution refinement lineage changed after generation')
    if report.get('status')!='GENERATED_STRUCTURAL_PASS':
        raise ValueError('Native generation is incomplete or failed')
    contracted=report.get('publication_contract')=='native-ic-generation-v1'
    seal=path.with_name(path.stem+'.sha256')
    if contracted and (not seal.is_file() or seal.read_text().strip()!=before):
        raise ValueError('Native generation report seal is missing or changed')
    artifacts=report.get('artifacts',{})
    required={'music.conf','transfers.txt','ics_gadget.hdf5','generation.log'}
    if not required<=set(artifacts):raise ValueError('Native generation lacks required artifact hashes')
    if contracted:
        actual={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()
            and '.native-run-locks' not in p.relative_to(root).parts
            and p.name not in (path.name,seal.name)}
        if actual!=set(artifacts):raise ValueError('Native generation artifact inventory changed')
    for name,expected in artifacts.items():
        target=(root/name).resolve()
        if not target.is_relative_to(root) or not target.is_file() or file_sha256(target)!=expected:
            raise ValueError('Delivered native artifact changed: '+name)
    verify_numeric_table(root/'transfers.txt')
    native_contract=report.get('structural_audit',{}).get('native_dataset_contract')
    if native_contract is not None:
        from core.data import verify_external_hdf5_contract
        verify_external_hdf5_contract(root/'ics_gadget.hdf5',native_contract)
    for key,name in (('velocity_audit','velocity-audit.json'),('phase_audit','phase-audit.json')):
        if key in report and (contracted or 'numeric_data' in report[key]):
            if verify_numeric_audit(root/name)!=report[key]:
                raise ValueError('Native generation numeric audit differs from saved report: '+key)
    if file_sha256(path)!=before:raise RuntimeError('Native generation report changed while reading')
    return report


def require_mixed_lpt_validation(proof,binary):
    """Require current native second-order controls and actual monofonIC evidence."""
    entry = proof.get('mixed_fluid_2lpt',{})
    driver = Path(__file__).resolve().parents[1]/'core/research/generator_controls.py'
    if entry.get('status') != 'PASS' or entry.get('regression_driver_sha256') != file_sha256(driver):
        raise ValueError('Native mixed-fluid 2LPT requires the current second-order regression suite')
    record = entry.get('receipt',{})
    path = Path(record.get('path','')).expanduser().resolve()
    if not path.is_file() or file_sha256(path) != record.get('sha256'):
        raise ValueError('Native mixed-fluid validation receipt changed')
    receipt = json.loads(path.read_text())
    if (receipt.get('status') != 'PASS' or receipt.get('binary_sha256') != file_sha256(binary)
            or receipt.get('regression_driver_sha256') != file_sha256(driver)
            or receipt.get('monofonic',{}).get('status') != 'PASS'
            or receipt.get('phases',{}).get('status') != 'PASS'
            or any(receipt.get(name,{}).get('velocity',{}).get('status') != 'PASS' for name in ('uniform','nested'))):
        raise ValueError('Native second-order validation lacks uniform/nested/monofonIC controls')
    if (any(receipt.get(name,{}).get('status') != 'PASS' for name in
            ('quadratic_controls','nested_quadratic_controls','refinement_continuity'))
            or set(receipt.get('selected_epochs',{})) != {'49','199'}
            or any(control.get('velocity',{}).get('status') != 'PASS'
                   for epoch in receipt.get('selected_epochs',{}).values()
                   for control in (epoch.get('uniform',{}),epoch.get('nested',{})))
            or receipt.get('ede_expansion',{}).get('nested',{}).get('velocity',{}).get('status') != 'PASS'):
        raise ValueError('Native second-order validation lacks zero/relative/epoch/expansion controls')
    from core.data import verify_numeric_audit
    for key,relative in (('monofonic','primary/monofonic-comparison.json'),
                         ('refinement_continuity','primary/smooth-interface/continuity.json')):
        numeric = verify_numeric_audit(path.parent/relative)
        if numeric != receipt[key]:
            raise ValueError('Native second-order numeric proof differs from its receipt: '+key)
        for source,checksum in numeric.get('sources',{}).items():
            if not Path(source).is_file() or file_sha256(source) != checksum:
                raise ValueError('Native second-order measured source changed: '+source)
    for source,checksum in receipt.get('source_hashes',{}).items():
        if not Path(source).is_file() or file_sha256(source) != checksum:
            raise ValueError('Native mixed-fluid regression source changed: '+source)
    if not receipt.get('source_hashes'):
        raise ValueError('Native mixed-fluid regression has no source-bound evidence')
    return receipt


# Zoom staging
#
# Stage model-matched native MUSIC parent/zoom inputs from solver provenance.
#
# Uses the accepted schema-2 Boltzmann stage, N-body species density/velocity
# tables and native pre-displacement-coordinate parent route. No scientific
# evolution or convergence is inferred from staging these inputs.

def validate_solver_input(path, declared):
    """Check the saved executable input, including undeclared/duplicate options.

    Hashes identify bytes; this check additionally binds those bytes to the
    settings used to interpret the output tables. It is not solver validation.
    """
    actual = {}
    for line in Path(path).read_text().splitlines():
        line = line.split('#', 1)[0].strip()
        if not line:
            continue
        if '=' not in line:
            raise ValueError('Malformed saved solver input')
        key, value = [part.strip() for part in line.split('=', 1)]
        if not key or key in actual:
            raise ValueError('Duplicate or empty saved solver option: ' + key)
        actual[key] = value
    if set(actual) != set(declared):
        raise ValueError('Saved solver input options differ from manifest')
    for key, expected in declared.items():
        value = actual[key]
        if value == str(expected).strip():
            continue
        # Permit numeric spelling differences only, never differing strings
        # such as a gauge, species prescription, root or output switch.
        try:
            first, second = float(value), float(expected)
        except (TypeError, ValueError):
            raise ValueError('Saved solver input differs from manifest: ' + key) from None
        if not (math.isfinite(first) and math.isfinite(second)
                and math.isclose(first, second, rel_tol=1e-14, abs_tol=0)):
            raise ValueError('Saved solver input differs from manifest: ' + key)


@directory_owner('destination')
def stage_zoom_inputs(spectrum_manifest, destination, *, box_mpc_h, parent_level,
                      refinement_level, seeds_by_level, padding_cells, poisson_accuracy, wdm_request=None,
                      lpt_order=1, lpt_seed_kind="radiation_regular", lpt_seed=None, phase_noise_by_level=None,
                      bounded_source_settings=None, source_archive_max_mib=None):
    spectrum_manifest, destination = [Path(p).expanduser().resolve() for p in (spectrum_manifest, destination)]
    if type(lpt_order) is not int or lpt_order not in (1,2):
        raise ValueError('Native zoom LPT order must be 1 or 2')
    bounded_source_settings=validate_bounded_source_settings(bounded_source_settings,lpt_order=lpt_order)
    if source_archive_max_mib is not None and (type(source_archive_max_mib) is not int or not 0<source_archive_max_mib<=(2**64-1)//(1024**2) or lpt_order!=2):
        raise ValueError('Native source archive requires mixed 2LPT and an exact positive representable disk budget in MiB')
    if lpt_order == 1 and (lpt_seed_kind != 'radiation_regular' or lpt_seed is not None):
        raise ValueError('Higher-order seed settings require LPT order 2')
    if destination.exists():
        raise FileExistsError('Refusing to overwrite model-matched zoom staging')
    if (isinstance(box_mpc_h, bool) or not math.isfinite(box_mpc_h) or box_mpc_h <= 0
            or type(parent_level) is not int or type(refinement_level) is not int
            or not 5 <= parent_level <= refinement_level <= 20
            or type(padding_cells) is not int or padding_cells < 2
            or isinstance(poisson_accuracy, bool) or not math.isfinite(poisson_accuracy)
            or not 0 < poisson_accuracy <= 1e-8):
        raise ValueError('Declare positive box, 5<=parent/refinement levels<=20, padding>=2 and Poisson accuracy<=1e-8')
    if not isinstance(seeds_by_level, dict) or set(seeds_by_level) != {str(level) for level in range(parent_level, refinement_level+1)}:
        raise ValueError('Declare one exact seed per level, including the shared parent level')
    if any(type(v) is not int or not 1 <= v <= 2147483647 for v in seeds_by_level.values()):
        raise ValueError('MUSIC seeds must be integers in [1,2147483647]')
    info = json.loads(spectrum_manifest.read_text())
    if info.get('schema_version') != 2 or info.get('status') != 'complete' or info.get('model') not in {'lcdm','ede','wdm'}:
        raise ValueError('Staging requires a complete schema-2 model-matched Boltzmann manifest')
    model, parameters = info['model'], info['parameters']
    solver = info['class_parameters']['start']
    if (parameters.get('model') != model or str(solver.get('nbody_gauge_transfer_functions','')).lower() != 'yes'
            or not {'mTk','vTk'} <= set(str(solver.get('output','')).split(','))):
        raise ValueError('Density and velocity species transfers must be declared in N-body gauge')
    expected_solver = 'AxiCLASS' if model == 'ede' else 'CLASS'
    if info.get('solver') != expected_solver:
        raise ValueError('Boltzmann engine does not match requested cosmological model')
    zstart = float(parameters['z_start'])
    c = {key: float(solver[key]) for key in ('H0','omega_b','omega_cdm','A_s','n_s')}
    if not all(math.isfinite(v) and v > 0 for v in c.values()) or not math.isfinite(zstart) or zstart <= 0:
        raise ValueError('Invalid physical cosmology or start epoch')
    if not math.isclose(float(solver['z_pk']),zstart,rel_tol=0,abs_tol=1e-8):
        raise ValueError('Boltzmann transfer epoch differs from declared parent start')
    h = c['H0']/100.
    if model == 'wdm':
        c['omega_cdm'] = float(parameters['Omega_cdm'])*h**2
    for key, value in (('H0',c['H0']),('Omega_b',c['omega_b']/h**2),('Omega_cdm',c['omega_cdm']/h**2),
                       ('A_s',c['A_s']),('n_s',c['n_s'])):
        if not math.isclose(float(parameters[key]),value,rel_tol=1e-12,abs_tol=0):
            raise ValueError('Solver/requested cosmology normalization mismatch: '+key)
    c.update(T_cmb=float(solver['T_cmb']),N_ur=float(solver['N_ur']))
    wdm_record = None
    if model == 'wdm':
        wdm_record = _stage_wdm_contract(info, spectrum_manifest, wdm_request, c,
            box_mpc_h, parent_level, refinement_level, seeds_by_level, lpt_order)
    elif wdm_request is not None:
        raise ValueError('A thermal relic request cannot accompany a cold model')
    if model != 'wdm' and (float(solver.get('N_ncdm',0)) != 0
            or float(solver.get('Omega_ncdm',0)) != 0
            or float(solver.get('omega_ncdm',0)) != 0):
        # A cold MUSIC transfer route does not add required relic thermal kicks.
        raise ValueError('Massive-species/WDM zoom staging needs a separately validated thermal/species route')
    if (float(solver.get('Omega_k',0)) != 0
            or float(solver.get('Omega_fld',0)) != 0
            or float(solver.get('w0_fld',-1)) != -1
            or float(solver.get('wa_fld',0)) != 0):
        raise ValueError('Native staging currently supports flat Lambda/EDE backgrounds only')
    if any(not math.isfinite(float(solver[key])) or float(solver[key]) <= 0 for key in ('T_cmb','N_ur')):
        raise ValueError('Declare finite positive radiation settings for the native input')
    if model == 'ede':
        expected_ede = ede_parameters({'ede':{'parameterization':'axion',
            'f_ede':parameters['f_ede'],'z_c':10.**float(parameters['log10_zc']),
            'theta_i':parameters['theta_i'],'n_axion':parameters['n_axion'],'extra_axiclass':{}}})
        for key, expected in expected_ede.items():
            value = solver.get(key)
            if isinstance(expected,(int,float)):
                matched = value is not None and math.isclose(float(value),float(expected),rel_tol=1e-12,abs_tol=0)
            else:
                matched = str(value) == str(expected)
            if not matched:
                raise ValueError('Requested EDE parameters differ from saved solver settings: '+key)
    elif 'scf_potential' in solver or float(solver.get('Omega_scf',0)) != 0:
        raise ValueError('LCDM staging cannot accept an undeclared scalar-field background')
    sources = {}
    for kind in ('transfer','background','ini'):
        relative = info['source_files']['start'][kind]
        path = (spectrum_manifest.parent/relative).resolve()
        if (not path.is_relative_to(spectrum_manifest.parent) or not path.is_file()
                or info['checksums_sha256'].get(relative) != file_sha256(path)):
            raise ValueError('Changed or missing solver source: '+kind)
        sources[kind] = path
    validate_solver_input(sources['ini'], solver)
    from initial_conditions import validate_linear_bundle
    linear_validation = validate_linear_bundle(spectrum_manifest, model=model, redshift=zstart,
        box_mpc_h=box_mpc_h, refinement_level=refinement_level)
    config = {'cosmology':dict(c, model=model), 'simulation':{'box_mpc_h':box_mpc_h,
              'particles_per_dimension':2**refinement_level}}
    noise_inputs = {} if phase_noise_by_level is None else phase_noise_by_level
    if not isinstance(noise_inputs,dict) or set(noise_inputs)-set(seeds_by_level):
        raise ValueError('Frozen phase noise must name declared MUSIC levels')
    frozen_prefix_extension=False
    frozen_streams=[]
    if set(noise_inputs)-{str(parent_level)}:
        highest_frozen=max(map(int,noise_inputs))
        if set(noise_inputs)!={str(level) for level in range(parent_level,highest_frozen+1)}:
            raise ValueError('Frozen fine noise requires files at every parent/refinement level in a contiguous retained prefix')
        frozen_prefix_extension=highest_frozen<refinement_level
        if frozen_prefix_extension:
            # The declared original seeds describe the hashed frozen prefix.
            # Keep these intervals in the staging receipt and native config;
            # numerical provenance/low-mode acceptance is still a separate gate.
            all_streams=_music_seed_stream_ranges(seeds_by_level)
            frozen_streams=[row for row in all_streams if str(row['level']) in noise_inputs]
    seed_streams=_music_seed_stream_ranges(seeds_by_level,noise_levels=set(map(int,noise_inputs)))
    if noise_inputs:
        from evolution import _checked,_noise_shape
        for level,entry in noise_inputs.items():
            _noise_shape(_checked(entry),2**int(level))
    destination.mkdir(parents=True,exist_ok=False)
    report = {'schema_version':1,'status':'STAGING','science_acceptance':'NOT_ACCEPTED',
              'solver_manifest':{'path':str(spectrum_manifest),'sha256':file_sha256(spectrum_manifest)},
              'source_files':{name:{'path':str(p),'sha256':file_sha256(p)} for name,p in sources.items()},
              'epoch':epoch_contract(1/(1+zstart)),
              'adopted_physical_constants':physical_constants_contract(),
              'algorithm_sha256':file_sha256(__file__),
              'box_mpc_h':box_mpc_h,'parent_level':parent_level,'refinement_level':refinement_level,
              'seeds_by_level':seeds_by_level,'perturbation_order':lpt_order,
              'phase_noise_by_level':noise_inputs,
              'bounded_source_settings':bounded_source_settings,
              'source_archive_max_mib':source_archive_max_mib,
              'frozen_prefix_extension':frozen_prefix_extension,
              'declared_frozen_raw_gaussian_seed_streams':frozen_streams,
              'raw_gaussian_seed_streams':seed_streams,
              'lpt_seed_kind':lpt_seed_kind,'lpt_seed':lpt_seed,
              'linear_bundle_validation':linear_validation,'wdm':wdm_record,
              'required_epoch_convergence':'NOT_VALIDATED',
              'parent_lagrangian_route':'native MUSIC schema 1 before displacements; never reconstructed from final positions'}
    try:
        atomic_json(destination/'staging.json',report)
        if wdm_record is not None:
            atomic_json(destination/'wdm-prescription.json',wdm_record)
        reference = destination/'species-reference.txt'
        write_species_transfer_validation_table(sources['transfer'],reference,config,model,zstart)
        table = destination/'transfers.txt'
        metadata = write_monofonic_ic_transfer_table(reference,table,model,zstart,config,sources['background'])
        # Same table is used at every level; its coverage gate includes the
        # highest refinement mesh diagonal, not merely the parent Nyquist.
        report['transfer_staging'] = metadata
        from core.cosmology import omega_radiation_from_background
        omega_r = omega_radiation_from_background(sources['background'])
        omega_m = (c['omega_b']+c['omega_cdm'])/h**2
        if not math.isfinite(omega_r) or omega_r < 0 or omega_m+omega_r >= 1:
            raise ValueError('This stage requires a flat positive-dark-energy cosmology')
        growth2 = None
        if lpt_order == 2:
            from core.cosmology import write_monofonic_background_table
            from initial_conditions import second_order_growth,radiation_growing_history
            from dataclasses import asdict
            import numpy as np
            background = destination/'lpt-background.txt'
            write_monofonic_background_table(sources['background'],background,config,model)
            rows = np.loadtxt(background)
            h0 = c['H0']/299792.458
            linear_d,linear_f=rows[:,2],rows[:,3]
            if lpt_seed_kind=='radiation_regular':
                linear_d,linear_f=radiation_growing_history(rows[:,0],rows[:,1]*h0,
                    omega_m0=omega_m,omega_r0=omega_r,hubble0=h0)
                # Keep the original solver export intact and publish the
                # explicitly derived common-flow history as a separate table.
                growing_background=destination/'lpt-growing-background.txt'
                from core.data import publish_numeric_table
                publish_numeric_table(growing_background,
                    np.column_stack((rows[:,0],rows[:,1],linear_d,linear_f)),
                    {'a':{'units':'dimensionless','quantity':'Scale-factor coordinate'},
                     'H_over_H0':{'units':'dimensionless','quantity':'Original solver expansion rate divided by H0'},
                     'D_regular':{'units':'dimensionless','quantity':'Regular pressureless growth normalized at last supplied epoch'},
                     'f_regular':{'units':'dimensionless','quantity':'d ln D_regular / d ln a'}},
                    context={'definition':'Regular pressureless common-flow growth solved in supplied H(a); original CLASS dust-growth column is not its seed',
                        'epoch':{'representation':'scale-factor-indexed history','scale_factor_range':[float(rows[0,0]),float(rows[-1,0])]}},
                    source_hashes={str(background.resolve()):file_sha256(background),
                        str(Path(__file__).parent/'initial_conditions.py'):file_sha256(Path(__file__).parent/'initial_conditions.py')},algorithm=__file__)
                report['second_order_linear_growth_source']={
                    'path':str(growing_background),'sha256':file_sha256(growing_background),
                    'scope':'Regular pressureless common-flow model in actual expansion; exact linear species transfers remain separate',
                    'solver_background_growth_used_as_seed':False}
            growth2 = asdict(second_order_growth(rows[:,0],rows[:,1]*h0,linear_d,linear_f,
                omega_m0=omega_m,omega_r0=omega_r,hubble0=h0,target_a=1/(1+zstart),
                seed_kind=lpt_seed_kind,initial_second_order=lpt_seed))
            report['second_order_growth'] = growth2
            report['mixed_fluid_formulation'] = {
                'reference':'https://arxiv.org/abs/2008.09124',
                'density':'common growing matter displacement; compensated relative density in species masses',
                'velocity':'background-dependent common second-order term plus exact species linear velocities',
                'relative_velocity_order':1,
                'nonlinear_scalar_field_perturbations':'NOT_REPRESENTED',
                'finite_refinement_window':'explicit species compensation zero modes retained in native header'}
            atomic_json(destination/'lpt-growth.json',growth2)
        # MUSIC reads the gas displacement switch before constructing its
        # output plugin; the plugin's later default cannot enable gas positions.
        common = {'setup':{'boxlength':format(box_mpc_h,'.17g'),'zstart':format(zstart,'.17g'),
                  'levelmin':str(parent_level),'levelmin_TF':str(parent_level),'padding':str(padding_cells),
                  'baryons':'yes','do_SPH':'yes','use_2LPT':'no','use_LLA':'no'},
                  'cosmology':{'Omega_m':format(omega_m,'.17g'),'Omega_b':format(c['omega_b']/h**2,'.17g'),
                  'Omega_L':format(1-omega_m-omega_r,'.17g'),'H0':format(c['H0'],'.17g'),
                  'A_s':format(c['A_s'],'.17g'),'n_s':format(c['n_s'],'.17g'),
                  'k_p':format(float(solver.get('k_pivot',.05)),'.17g'),
                  'Tcmb':format(float(solver['T_cmb']),'.17g'),'N_ur':format(float(solver['N_ur']),'.17g'),
                  'ZeroRadiation':'no',
                  'm_nu1':'0','m_nu2':'0','m_nu3':'0','transfer':'DMS_file','input_model':model,
                  'ztarget':format(zstart,'.17g'),'transfer_file':str(table)},
                  'output':{'format':'arepo','filename':'ics_gadget.hdf5','arepo_coarsetype':'2',
                            'arepo_doubleprec':'yes','arepo_longids':'yes','arepo_num_files':'1','shift_back':'yes'},
                  'poisson':{'accuracy':format(poisson_accuracy,'.17g')}}
        if growth2 is not None:
            common['setup'].update(use_2LPT='yes',do_SPH='yes')
            common['cosmology'].update(mixed_2lpt='yes',
                lpt2_displacement=format(growth2['displacement_coefficient'],'.17g'),
                lpt2_velocity=format(growth2['velocity_coefficient'],'.17g'),
                lpt2_aH_km_s_Mpc=format(metadata['conformal_hubble_rate_start_per_Mpc']*299792.458,'.17g'))
        for name, maxlevel in (('parent',parent_level),('zoom_template',refinement_level)):
            cfg=configparser.ConfigParser(interpolation=None)
            cfg.optionxform=str
            cfg.read_dict(common)
            cfg.set('setup','levelmax',str(maxlevel))
            if name=='zoom_template' and bounded_source_settings is not None:
                cfg['poisson'].update({key:str(value) for key,value in bounded_source_settings.items()})
            if name=='zoom_template' and source_archive_max_mib is not None:
                cfg['output'].update(retain_source_hierarchy='yes',source_hierarchy_max_mib=str(source_archive_max_mib))
            cfg['random']={f'seed[{level}]':str(seeds_by_level[str(level)]) for level in range(parent_level,maxlevel+1)}
            for level,entry in noise_inputs.items():
                if int(level)<=maxlevel:
                    cfg['random'][f'seed[{level}]']=entry['path']
                    cfg['random']['grafic_sign']='yes'
            if name=='zoom_template' and frozen_prefix_extension:
                for row in frozen_streams:
                    cfg['random'][f"frozen_stream_first[{row['level']}]"]=str(row['first_seed'])
                    cfg['random'][f"frozen_stream_last[{row['level']}]"]=str(row['last_seed'])
            if name == 'zoom_template':
                # These files deliberately have no invented refinement centre.
                # generate_zoom receives the catalogue-linked traced region.
                cfg.set('setup','region','box')
                cfg['darkmattersim']={'requires_traced_region':'yes'}
            stream=io.StringIO()
            cfg.write(stream)
            atomic_text(destination/(name+'.conf'),stream.getvalue())
        for name,entry in report['source_files'].items():
            if file_sha256(entry['path']) != entry['sha256']:
                raise ValueError('Solver source changed during staging: '+name)
        if file_sha256(spectrum_manifest) != report['solver_manifest']['sha256']:
            raise ValueError('Solver manifest changed during staging')
        for entry in noise_inputs.values():
            _checked(entry)
        verify_numeric_table(reference)
        verify_numeric_table(table)
        report['status']='STAGED_MODEL_MATCHED_INPUTS'
        report['artifacts']={p.name:file_sha256(p) for p in destination.iterdir() if p.is_file() and p.name!='staging.json'}
        report['sealed_assets']={p.relative_to(destination).as_posix():file_sha256(p)
            for p in destination.rglob('*') if p.is_file()
            and '.native-run-locks' not in p.relative_to(destination).parts
            and p.name not in ('staging.json','staging.sha256')}
    except BaseException as exc:
        report['status']='INTERRUPTED' if isinstance(exc,(KeyboardInterrupt,SystemExit)) else 'FAILED'
        report['failure']={'type':type(exc).__name__,'message':str(exc)}
        atomic_json(destination/'staging.json',report)
        raise
    atomic_json(destination/'staging.json',report)
    atomic_text(destination/'staging.sha256',file_sha256(destination/'staging.json')+'\n')
    return report


@directory_owner('directory')
def verify_staged_zoom_inputs(directory):
    """Verify completed recipes and contracted tables before native consumption."""
    root=Path(directory).resolve()
    path=root/'staging.json'
    before=file_sha256(path)
    report=json.loads(path.read_text())
    if report.get('status')!='STAGED_MODEL_MATCHED_INPUTS':
        raise ValueError('Native zoom input staging is incomplete or failed')
    seal=root/'staging.sha256'
    if 'sealed_assets' in report and not seal.is_file():
        raise ValueError('Native zoom staging report seal is missing')
    if seal.exists() and seal.read_text().strip()!=before:
        raise ValueError('Native zoom staging report checksum mismatch')
    assets=report.get('sealed_assets',report.get('artifacts',{}))
    if 'sealed_assets' in report:
        actual={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()
            and '.native-run-locks' not in p.relative_to(root).parts
            and p.name not in ('staging.json','staging.sha256')}
        if actual!=set(assets):raise ValueError('Native zoom staging asset inventory changed')
    for name,expected in assets.items():
        target=(root/name).resolve()
        if not target.is_relative_to(root) or not target.is_file() or file_sha256(target)!=expected:
            raise ValueError('Native zoom staging asset changed: '+name)
    for entry in report.get('phase_noise_by_level',{}).values():
        from evolution import _checked
        _checked(entry)
    verify_numeric_table(root/'transfers.txt')
    verify_numeric_table(root/'species-reference.txt')
    if report.get('wdm') is not None:
        wdm = report['wdm']
        records = list(wdm['resolution_spectra'].values())
        control = wdm['lcdm_control_validation']
        records += [control['manifest'],*control['sources'].values()]
        for entry in records:
            if not Path(entry['path']).is_file() or file_sha256(entry['path']) != entry['sha256']:
                raise ValueError('Native WDM resolution/control input changed')
    if file_sha256(path)!=before:
        raise RuntimeError('Native zoom staging report changed while reading')
    return report


def _stage_wdm_contract(info, manifest, request, cosmology, box, low, high, seeds, lpt_order):
    """Bind one physical relic and a matched LCDM control to all mesh levels."""
    from core.cosmology import wdm_parameters, wdm_resolution_report, validate_wdm_resolution_gate
    from initial_conditions import validate_linear_bundle
    if not isinstance(request,dict) or set(request) != {'wdm','thermal_seed','lcdm_manifest'}:
        raise ValueError('WDM staging requires wdm, thermal_seed and lcdm_manifest in wdm_request')
    w = request['wdm']
    if not isinstance(w,dict) or w.get('model') != 'wdm_thermal_relic' or w.get('include_thermal_velocities',True) is not True:
        raise ValueError('WDM zoom staging requires a physical thermal relic with thermal velocities enabled')
    seed = request['thermal_seed']
    if type(seed) is not int or not 0 <= seed <= 2**64-1:
        raise ValueError('Declare an exact unsigned 64-bit thermal seed')
    cfg = {'cosmology':dict(cosmology,model='wdm'), 'wdm':w,
           'spectrum':{'source':'boltzmann'},
           'simulation':{'physics':'hydro','box_mpc_h':box,'particles_per_dimension':2**high,'random_seed':seed,'lpt_order_hydro':lpt_order}}
    relic = wdm_parameters(cfg)
    solver = info['class_parameters']['start']
    for key,value in relic.items():
        if key not in solver:
            raise ValueError('Saved solver lacks normalized relic option: '+key)
        try: matched = math.isclose(float(solver[key]),float(value),rel_tol=1e-12,abs_tol=0)
        except (TypeError,ValueError): matched = str(solver[key]) == str(value)
        if not matched:
            raise ValueError('Saved solver differs from physical thermal relic: '+key)
    lcdm = Path(request['lcdm_manifest']).expanduser().resolve()
    control_validation = validate_linear_bundle(lcdm,model='lcdm',redshift=info['parameters']['z_start'])
    control = json.loads(lcdm.read_text())
    for key in ('H0','Omega_b','Omega_cdm','A_s','n_s','z_start'):
        if not math.isclose(float(control['parameters'][key]),float(info['parameters'][key]),rel_tol=1e-12):
            raise ValueError('WDM/LCDM resolution control mismatch: '+key)
    inputs = {}
    for name,source,record in (('wdm',manifest,info),('lcdm',lcdm,control)):
        relative = record['source_files']['start']['pk']
        path = (source.parent/relative).resolve()
        if not path.is_relative_to(source.parent) or not path.is_file() or file_sha256(path) != record['checksums_sha256'].get(relative):
            raise ValueError('WDM resolution spectrum changed: '+name)
        from core.cosmology import class_redshift_matches, pk_output_redshift
        actual_z = pk_output_redshift(path)
        if not class_redshift_matches(actual_z, info['parameters']['z_start']):
            raise ValueError('WDM resolution spectrum epoch mismatch: '+name)
        if path.stat().st_size*12+64*1024**2 > 1024**3:
            raise MemoryError('WDM resolution spectrum exceeds staging workspace')
        inputs[name] = {'path':str(path),'sha256':file_sha256(path)}
    levels = {}
    for level in range(low,high+1):
        per_level = dict(cfg,simulation=dict(cfg['simulation'],particles_per_dimension=2**level))
        diagnostic = wdm_resolution_report(inputs['wdm']['path'],inputs['lcdm']['path'],per_level)
        diagnostic['second_order_lpt_velocity'] = ('common growing mixed 2LPT plus first-order species bulk and physical relic thermal velocity'
            if lpt_order == 2 else 'NOT_USED; native species 1LPT bulk plus physical relic thermal velocity')
        levels[str(level)] = diagnostic
    gate = validate_wdm_resolution_gate(levels[str(high)],cfg)
    for entry in inputs.values():
        if file_sha256(entry['path']) != entry['sha256']:
            raise ValueError('WDM resolution spectrum changed during staging')
    return {'schema':'native-wdm-staging-v1','configuration':cfg,'request':request,
            'relic_solver_parameters':relic,'lcdm_control_validation':control_validation,
            'resolution_spectra':inputs,'resolution_by_level':levels,
            'finest_resolution_gate':gate,'seeds_by_level':seeds,'thermal_seed':seed,
            'coarse_prescription':'combined-matter COM with relic-fraction thermal velocities; unresolved phase space',
            'science_acceptance':'NOT_ACCEPTED'}


# Zoom handoff
#
# Revalidate delivered native ICs against solver inputs before particle handoff.
#
# Technical handoff validation remains distinct from resolved science acceptance.
# No saved PASS string substitutes for remeasuring the delivered IC audits.

def _configuration(path):
    cfg=configparser.ConfigParser(interpolation=None)
    cfg.optionxform=str
    with Path(path).open() as stream:
        cfg.read_file(stream)
    return cfg


def relocate_music_configuration(source, destination, *, overrides=None):
    """Copy an IC recipe while preserving the meaning of file-backed inputs.

    Native transfer/noise paths are relative to the original recipe directory.
    Resolve them before running in a new campaign working directory. Output
    paths stay local to the new recipe unless an explicit override is supplied.
    """
    source=Path(source).expanduser().resolve();destination=Path(destination).expanduser().resolve()
    if destination.exists():raise FileExistsError('Keep previous native configuration')
    cfg=_configuration(source)
    bindings={str(source):file_sha256(source)}
    for section,key in [('cosmology','transfer_file'),('cosmology','background_file')]:
        if not cfg.has_option(section,key):continue
        asset=Path(cfg.get(section,key)).expanduser()
        if not asset.is_absolute():asset=source.parent/asset
        asset=asset.resolve()
        if not asset.is_file():raise FileNotFoundError('Native recipe input is missing: '+str(asset))
        cfg.set(section,key,str(asset));bindings[str(asset)]=file_sha256(asset)
    if cfg.has_section('random'):
        for key,value in list(cfg.items('random')):
            if not key.startswith('seed['):continue
            try:int(value);continue
            except ValueError:pass
            asset=Path(value).expanduser()
            if not asset.is_absolute():asset=source.parent/asset
            asset=asset.resolve()
            if not asset.is_file():raise FileNotFoundError('Frozen native noise is missing: '+str(asset))
            cfg.set('random',key,str(asset));bindings[str(asset)]=file_sha256(asset)
    for section,values in (overrides or {}).items():
        if not cfg.has_section(section):cfg.add_section(section)
        for key,value in values.items():cfg.set(section,key,str(value))
    destination.parent.mkdir(parents=True,exist_ok=True)
    with destination.open('x') as stream:cfg.write(stream)
    return {'configuration':str(destination),'configuration_sha256':file_sha256(destination),'source_hashes':bindings}


def _checked_record(entry):
    path=Path(entry['path']).expanduser().resolve()
    if not path.is_file() or file_sha256(path)!=entry['sha256']:
        raise ValueError('Native handoff source changed: '+str(path))
    return path


@directory_owner('destination')
def verify_native_handoff(ic, generation_report, cfg, destination, *, memory_gib=1):
    ic,generation_report,destination=[Path(p).expanduser().resolve() for p in (ic,generation_report,destination)]
    if destination.exists():
        raise FileExistsError('Refusing to overwrite native handoff audit')
    if not math.isfinite(memory_gib) or memory_gib<=0 or generation_report.stat().st_size*8>memory_gib*1024**3:
        raise MemoryError('Native handoff generation metadata exceeds its allocation')
    if json.loads(generation_report.read_text()).get('schema')=='twofluid-native-ic-generation-v1':
        from initial_conditions import verify_twofluid_native_handoff
        return verify_twofluid_native_handoff(ic,generation_report,cfg,destination,memory_gib=memory_gib)
    generation=verify_generated_zoom(generation_report)
    if generation.get('status')!='GENERATED_STRUCTURAL_PASS' or ic!=generation_report.parent/'ics_gadget.hdf5':
        raise ValueError('Native handoff requires the exact completed generation IC')
    if (cfg['simulation']['physics']!='hydro' or type(cfg['simulation']['lpt_order_hydro']) is not int
            or cfg['simulation']['lpt_order_hydro'] not in (1,2)):
        raise ValueError('Native two-fluid IC handoff requires hydro and declared supported LPT order')
    source_cfg=_checked_record(generation['source_configuration'])
    _checked_record(generation['source_transfer_table'])
    binary=_checked_record(generation['binary'])
    proof_path=_checked_record(generation['native_validation'])
    proof=json.loads(proof_path.read_text())
    driver=Path(__file__).resolve().parents[1]/'core/research/generator_controls.py'
    if (proof.get('status')!='PASS' or proof.get('nested',{}).get('status')!='PASS'
            or proof.get('periodic',{}).get('status')!='PASS'
            or proof.get('execution',{}).get('binary_sha256')!=file_sha256(binary)
            or proof.get('regression_driver_sha256')!=file_sha256(driver)):
        raise ValueError('Native binary has no passing current transfer regression')
    if cfg['simulation']['lpt_order_hydro'] == 2:
        require_mixed_lpt_validation(proof,binary)
    rejections=proof.get('execution',{}).get('rejections',{})
    required={'reject-second-order','reject-backscaling','reject-gauge','reject-model',
              'reject-redshift','reject-mode-coverage','reject-poisson'}
    if set(rejections)!=required or any(row.get('exit_code')!=1 for row in rejections.values()):
        raise ValueError('Native transfer proof lacks required failure controls')
    for name,checksum in generation['artifacts'].items():
        path=(generation_report.parent/name).resolve()
        if not path.is_relative_to(generation_report.parent) or file_sha256(path)!=checksum:
            raise ValueError('Delivered native artifact changed: '+name)
    if not {'music.conf','transfers.txt','ics_gadget.hdf5','generation.log'}<=set(generation['artifacts']):
        raise ValueError('Native handoff lacks required artifact hashes')
    staging_path=source_cfg.parent/'staging.json'
    staging=verify_staged_zoom_inputs(staging_path.parent)
    if staging.get('status')!='STAGED_MODEL_MATCHED_INPUTS':
        raise ValueError('Native handoff requires model-matched staging')
    solver_manifest=_checked_record(staging['solver_manifest'])
    if cfg['spectrum']['source']!='modular' or Path(cfg['spectrum']['file']).expanduser().resolve()!=solver_manifest:
        raise ValueError('Native handoff must reuse the exact source Boltzmann manifest')
    info=json.loads(solver_manifest.read_text())
    model=info['model']
    if model!=cfg['cosmology']['model']:
        raise ValueError('Native handoff cosmological model mismatch')
    h=cfg['cosmology']['H0']/100.
    matching={'H0':cfg['cosmology']['H0'],'Omega_b':cfg['cosmology']['omega_b']/h**2,
              'Omega_cdm':cfg['cosmology']['omega_cdm']/h**2,'A_s':cfg['cosmology']['A_s'],
              'n_s':cfg['cosmology']['n_s'],'z_start':cfg['simulation']['start_redshift']}
    if model=='ede':
        matching.update(f_ede=cfg['ede']['f_ede'],log10_zc=math.log10(cfg['ede']['z_c']),
                        theta_i=cfg['ede']['theta_i'],n_axion=cfg['ede']['n_axion'])
    for key,value in matching.items():
        if not math.isclose(float(value),float(info['parameters'][key]),rel_tol=1e-12,abs_tol=0):
            raise ValueError('Native handoff requested cosmology/epoch mismatch: '+key)
    for key in ('T_cmb','N_ur'):
        if not math.isclose(float(cfg['cosmology'][key]),float(info['class_parameters']['start'][key]),rel_tol=1e-12):
            raise ValueError('Native handoff radiation mismatch: '+key)
    native=_configuration(generation_report.parent/'music.conf')
    original=_configuration(source_cfg)
    actual_order=2 if native.getboolean('setup','use_2LPT',fallback=False) else 1
    if actual_order != cfg['simulation']['lpt_order_hydro'] or actual_order != staging.get('perturbation_order',1):
        raise ValueError('Native handoff perturbation order mismatch')
    low,high=native.getint('setup','levelmin'),native.getint('setup','levelmax')
    kind='parent' if low==high else 'zoom'
    expected_name='parent.conf' if kind=='parent' else 'zoom_template.conf'
    if source_cfg.name!=expected_name or staging['artifacts'].get(expected_name)!=file_sha256(source_cfg):
        raise ValueError('Native handoff configuration is not the staged parent/zoom recipe')
    if (cfg['simulation']['particles_per_dimension']!=2**high
            or not math.isclose(cfg['simulation']['box_mpc_h'],native.getfloat('setup','boxlength'),rel_tol=1e-12)):
        raise ValueError('Native handoff finest resolution/box mismatch')
    destination.mkdir(parents=True,exist_ok=False)
    report={'schema_version':1,'status':'IN_PROGRESS','input_kind':kind,'science_acceptance':'NOT_ACCEPTED',
            'generation_report':{'path':str(generation_report),'sha256':file_sha256(generation_report)},
            'staging_report':{'path':str(staging_path),'sha256':file_sha256(staging_path)}}
    try:
        atomic_json(destination/'handoff.json',report)
        if 'native_resource_plan' in generation:
            saved_plan = generation['native_resource_plan']
            if saved_plan.get('status') != 'PASS' or generation.get('native_mesh_memory_gate') != 'ESTIMATE_PASS':
                raise ValueError('Native generation resource plan did not pass')
            report['native_resource_plan'] = plan_native_music(
                generation_report.parent/'music.conf', binary, destination/'resource-planning',
                memory_gib=saved_plan['budget_memory_bytes']/1024**3,
                disk_gib=saved_plan['budget_disk_bytes']/1024**3,
                threads=generation['threads'])
            if report['native_resource_plan']['native_geometry'] != saved_plan['native_geometry']:
                raise ValueError('Native generation mesh geometry changed before handoff')
            atomic_json(destination/'handoff.json',report)
        # Regenerate references from the hashed actual solver tables. This also
        # checks saved input options, epoch, species, normalization and coverage.
        stage_zoom_inputs(solver_manifest,destination/'reference',box_mpc_h=staging['box_mpc_h'],
            parent_level=staging['parent_level'],refinement_level=staging['refinement_level'],
            seeds_by_level=staging['seeds_by_level'],padding_cells=original.getint('setup','padding'),
            poisson_accuracy=original.getfloat('poisson','accuracy'),
            wdm_request=None if staging.get('wdm') is None else staging['wdm']['request'],
            lpt_order=actual_order,lpt_seed_kind=staging.get('lpt_seed_kind','radiation_regular'),
            lpt_seed=staging.get('lpt_seed'),phase_noise_by_level=staging.get('phase_noise_by_level'),
            bounded_source_settings=staging.get('bounded_source_settings'),source_archive_max_mib=staging.get('source_archive_max_mib'))
        expected=_configuration(destination/'reference'/expected_name)
        region=generation.get('traced_region')
        if kind=='zoom' and region is None:
            raise ValueError('Nested native handoff requires a traced target region')
        if region is not None:
            region_path=_checked_record(region)
            apply_traced_region(expected,region_path)
        expected.set('cosmology','transfer_file','transfers.txt')
        expected.set('output','filename','ics_gadget.hdf5')
        if {s:dict(expected[s]) for s in expected.sections()}!={s:dict(native[s]) for s in native.sections()}:
            raise ValueError('Delivered native configuration differs from regenerated solver-matched recipe')
        if (not math.isfinite(memory_gib) or memory_gib<=0
                or 16*(generation_report.parent/'transfers.txt').stat().st_size+64*1024**2>memory_gib*1024**3):
            raise MemoryError('Native handoff tables exceed the declared comparison workspace')
        verify_numeric_table(generation_report.parent/'transfers.txt')
        verify_numeric_table(destination/'reference/transfers.txt')
        actual_table=np.loadtxt(generation_report.parent/'transfers.txt')
        reference_table=np.loadtxt(destination/'reference/transfers.txt')
        if actual_table.shape!=reference_table.shape or not np.array_equal(actual_table,reference_table):
            raise ValueError('Delivered native species table differs from regenerated solver inputs')
        report['structural_audit']=audit_music_ic(ic,generation_report.parent/'music.conf',memory_gib=memory_gib,input_kind=kind)
        if 'linear_bundle_validation' in staging:
            from initial_conditions import validate_ic_realization
            report['physical_audit'] = validate_ic_realization(ic,generation_report.parent/'music.conf',
                destination/'physical.json',memory_gib=memory_gib,input_kind=kind)
            if report['physical_audit']['status'] == 'FAIL':
                raise ValueError('Native handoff conservation audit failed')
        report['resource_census']=initial_condition_census([ic])
        if model == 'wdm':
            from initial_conditions import verify_native_wdm_thermal
            if staging.get('wdm') is None or generation.get('wdm_staging') != staging['wdm']:
                raise ValueError('Native WDM handoff lacks the sealed relic and resolution contract')
            sample_cfg = staging['wdm']['configuration']
            for key in ('model','mode','thermal_relic_mass_keV','include_thermal_velocities'):
                if cfg['wdm'].get(key,True if key=='include_thermal_velocities' else None) != sample_cfg['wdm'].get(key,True if key=='include_thermal_velocities' else None):
                    raise ValueError('Native WDM handoff relic request mismatch: '+key)
            if cfg['simulation']['random_seed'] != sample_cfg['simulation']['random_seed']:
                raise ValueError('Native WDM handoff thermal seed mismatch')
            report['wdm_thermal'] = verify_native_wdm_thermal(ic,sample_cfg)
            if report['wdm_thermal'] != generation.get('wdm_thermal'):
                raise ValueError('Native WDM handoff thermal report differs from generation')
            report['wdm_resolution_by_level'] = staging['wdm']['resolution_by_level']
        report['velocity_audit']=audit_zoom_velocity(ic,generation_report.parent/'transfers.txt',
            destination/'velocity.json',model=model,memory_gib=memory_gib)
        report['velocity_audit']=verify_numeric_audit(destination/'velocity.json')
        if report['velocity_audit']['status']!='PASS':
            raise ValueError('Native handoff velocity audit failed')
        if native.getboolean('output','retain_source_hierarchy',fallback=False):
            from core.music.archive import index_music_source_archive
            captured=index_music_source_archive(generation_report.parent/'source-hierarchy.bin',
                destination/'source-hierarchy-index.json',memory_gib=memory_gib)
            if captured!=generation.get('source_archive_index'):
                raise ValueError('Native retained source hierarchy differs from generated evidence')
            with h5py.File(ic) as native_ic:
                header=native_ic['Header'].attrs
                frame={'actual_scale_factor':float(header['Time']),
                    'box_comoving_kpc_h':float(header['BoxSize']),
                    'output_coordinate_shift_box_fraction':np.asarray(header['Music_output_coordinate_shift_box_fraction']).tolist(),
                    'baryon_material_offset_native':np.asarray(header['Music_baryon_material_offset_native']).tolist()}
            report['source_archive_binding']={'index':captured,'actual_ic_sha256':file_sha256(ic),
                'configuration_sha256':file_sha256(generation_report.parent/'music.conf'),
                'binary':generation['binary'],'frame':frame,
                'operator_configuration':dict(native.items('poisson')),
                'transfer_table_sha256':file_sha256(generation_report.parent/'transfers.txt'),
                'reference_acceptance':'NOT_ESTABLISHED'}
            from core.music.archive import audit_music_archive_quadratic_source
            audit_music_archive_quadratic_source(destination/'source-hierarchy-index.json',
                generation_report.parent/'music.conf',ic,destination/'quadratic-source-operator.json',memory_gib=memory_gib)
            report['quadratic_source_operator']=verify_numeric_audit(destination/'quadratic-source-operator.json')
            if report['quadratic_source_operator']['status']!='PASS':
                raise ValueError('Retained native quadratic source operator is failed or unresolved')
            if kind=='zoom':
                from core.music.replay import audit_music_archive_transfer
                audit_music_archive_transfer(destination/'source-hierarchy-index.json',
                    generation_report.parent/'music.conf',ic,
                    {'schema':'native-music-transfer-replay-request-v1',
                     'mesh':str(generation_report.parent/'planning/native-mesh.json'),
                     'noise_directory':str(generation_report.parent)},
                    destination/'source-transfer-replay.json',memory_gib=memory_gib)
                report['source_transfer_replay']=verify_numeric_audit(destination/'source-transfer-replay.json')
                if report['source_transfer_replay']['status']!='PASS':
                    raise ValueError('Original native source noise/transfer replay failed')
                from core.music.windows import measure_native_compensated_windows
                measure_native_compensated_windows(destination/'source-hierarchy-index.json',
                    generation_report.parent/'music.conf',ic,
                    {'schema':'native-music-compensated-window-request-v1',
                     'mesh':str(generation_report.parent/'planning/native-mesh.json'),
                     'noise_directory':str(generation_report.parent),
                     'modes':source_mode_sphere(2,memory_gib=memory_gib).tolist(),
                     'windows':[{'id':'whole-fine','kind':'all_fine'}]},
                    destination/'compensated-window-response.json',memory_gib=memory_gib)
                report['compensated_window_response']=verify_numeric_audit(destination/'compensated-window-response.json')
                if report['compensated_window_response']['status']!='PASS':
                    raise ValueError('Original native compensated-mass window response failed')
                from core.music.archive import audit_music_archive_poisson
                audit_music_archive_poisson(destination/'source-hierarchy-index.json',
                    generation_report.parent/'music.conf',ic,destination/'original-poisson-equations.json',memory_gib=memory_gib)
                report['original_poisson_equations']=verify_numeric_audit(destination/'original-poisson-equations.json')
                if report['original_poisson_equations']['status']!='PASS':
                    raise ValueError('Original prepared Poisson source equations fail the declared physical-leaf tolerance')
                from core.music.replay import audit_music_archive_gradients
                audit_music_archive_gradients(destination/'source-hierarchy-index.json',
                    generation_report.parent/'music.conf',ic,destination/'gradient-operators.json',memory_gib=memory_gib)
                report['gradient_operators']=verify_numeric_audit(destination/'gradient-operators.json')
                if report['gradient_operators']['status']!='PASS':
                    raise ValueError('Native particle gradient/hybrid operator handoff failed')
        if kind=='zoom':
            # Preserve each original species/hierarchy window in the handoff.
            # This is delivered-field evidence; transfer/operator references
            # and quadratic acceptance remain independent required gates.
            measure_native_species_window_modes(ic,source_mode_sphere(2,memory_gib=memory_gib),
                destination/'species-window-modes.json',memory_gib=memory_gib)
            report['species_window_measurement']=verify_numeric_audit(destination/'species-window-modes.json')
            if actual_order==2:
                from core.music.archive import audit_native_quadratic_velocity
                audit_native_quadratic_velocity(ic,generation_report.parent/'music.conf',
                    destination/'quadratic-assembly.json',memory_gib=memory_gib)
                report['quadratic_assembly_audit']=verify_numeric_audit(destination/'quadratic-assembly.json')
                if report['quadratic_assembly_audit']['status']!='PASS':
                    raise ValueError('Native quadratic output component assembly is failed or unresolved')
            parent_inputs=generation.get('parent_phase_inputs') or {}
            if len(parent_inputs)!=2:
                raise ValueError('Nested native handoff lacks exact parent IC/noise inputs')
            paths=[_checked_record({'path':path,'sha256':checksum}) for path,checksum in parent_inputs.items()]
            parents=[path for path in paths if h5py.is_hdf5(path)]
            noise=[path for path in paths if not h5py.is_hdf5(path)]
            if len(parents)!=1 or len(noise)!=1:
                raise ValueError('Nested native handoff needs one parent HDF5 and one noise file')
            region_info=json.loads(region_path.read_text())
            if parents[0]!=Path(region_info['initial_ic']['path']).resolve():
                raise ValueError('Native phase parent differs from traced target parent')
            report['phase_audit']=compare_density_phases(parents[0],ic,noise[0],
                generation_report.parent/f'wnoise_{low:04d}.bin',destination/'phases.json',memory_gib=memory_gib)
            report['phase_audit']=verify_numeric_audit(destination/'phases.json')
            if report['phase_audit']['status']!='PASS':
                raise ValueError('Native handoff parent phase audit failed')
            report['coverage_audit']=audit_target_coverage(generation_report,region_path,
                destination/'coverage.json',memory_gib=memory_gib)
            if report['coverage_audit']['status']!='PASS':
                raise ValueError('Native handoff static target coverage failed')
            if not cfg['simulation'].get('zoom_softening_mass_ratios'):
                raise ValueError('Nested native handoff requires explicit mass-dependent softening classes')
            report['softening_audit']=validate_coarse_softening_masses([ic],cfg['simulation'])
        elif cfg['simulation'].get('zoom_softening_mass_ratios') is not None:
            raise ValueError('Uniform native parent cannot declare coarse softening classes')
        if file_sha256(generation_report)!=report['generation_report']['sha256'] or file_sha256(staging_path)!=report['staging_report']['sha256']:
            raise ValueError('Native source reports changed during handoff audit')
        for name,checksum in generation['artifacts'].items():
            if file_sha256(generation_report.parent/name)!=checksum:
                raise ValueError('Native artifact changed during handoff audit: '+name)
        report['status']='TECHNICAL_PASS'
        report['remaining_science_gates']=['species/refinement-scale transfer controls','LPT order/start-epoch convergence',
            'nonlinear target recovery','evolving contamination','resolved tracers','observable convergence']
    except BaseException as error:
        report['status']='INTERRUPTED' if isinstance(error,(KeyboardInterrupt,SystemExit)) else 'FAILED'
        report['failure']={'type':type(error).__name__,'message':str(error)}
        atomic_json(destination/'handoff.json',report)
        raise
    atomic_json(destination/'handoff.json',report)
    return report
