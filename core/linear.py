"""Standalone CLASS/AxiCLASS linear bundles and their public command.

Saved linear spectra, transfer functions and background tables are reusable
inputs for particle preparation. Plotting loads only when a figure is requested.
"""
from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
import re
import subprocess
import numpy as np
from core.foundation import C_SI, atomic_json, atomic_text, fsync_directory, file_sha256 as checksum
from core.native import directory_owner, native_run_lock, run_native_command
from core.data import epoch_contract, publish_numeric_table, verify_numeric_table
from core.resources import runtime_paths

C_KM_S = C_SI / 1000.0


def solver_parameters(cfg: dict, root: Path, redshift: float) -> dict:
    """Translate the user's density fractions into CLASS physical densities.

    CLASS's omega_b and omega_cdm are Omega*h^2. Its P(k) output with
    ``format=class`` is k [h/Mpc], P [(Mpc/h)^3].
    """
    h = cfg["H0"] / 100.0
    params = {
        "H0": cfg["H0"],
        "omega_b": cfg["Omega_b"] * h * h,
        "omega_cdm": cfg["Omega_cdm"] * h * h,
        "A_s": cfg["A_s"],
        "n_s": cfg["n_s"],
        "tau_reio": cfg["tau_reio"],
        "T_cmb": 2.7255,
        "N_ur": 3.046,
        "output": "mPk,mTk,vTk",
        "gauge": "synchronous",
        "nbody_gauge_transfer_functions": "yes",
        "P_k_max_h/Mpc": cfg["k_max_h_mpc"],
        "z_pk": redshift,
        "root": str(root),
        "write background": "yes",
        "write parameters": "yes",
        "headers": "yes",
        "format": "class",
    }
    if cfg["model"] == "ede":
        from core.cosmology import ede_parameters
        params.update(ede_parameters({"ede": {
            "parameterization": "axion", "f_ede": cfg["f_ede"],
            "z_c": 10.0 ** cfg["log10_zc"], "theta_i": cfg["theta_i"],
            "n_axion": cfg["n_axion"], "extra_axiclass": {},
        }}))
    return params



def read_background(path: Path) -> tuple[dict, np.ndarray]:
    """Read CLASS's numbered background header instead of assuming offsets."""
    from core.cosmology import class_background_table
    columns, data = class_background_table(path)
    if not {'z', 'H', 'growth_D', 'growth_f'} <= columns.keys():
        raise ValueError(f"Malformed CLASS background table: {path}")
    return {'z': columns['z'], 'H': columns['H'],
            'D': columns['growth_D'], 'f': columns['growth_f']}, data



@directory_owner('destination')
def export_background(source: Path, destination: Path, z_start: float, *, particle_inputs: bool = True,
                      expected_h0_km_s_mpc=None) -> None:
    """Write physical H and normalized GADGET growth histories.

    CLASS reports H in 1/Mpc (c=1 convention). Multiplying by c yields
    km/s/Mpc. GADGET's patched background reader instead needs E=H/H0,
    D/D(1), and f=dlnD/dlna, sampled from a_start through a=1.
    """
    source_before=checksum(source)
    columns, raw = read_background(source)
    a = 1.0 / (1.0 + raw[:, columns["z"]])
    order = np.argsort(a)
    a = a[order]
    H = raw[order, columns["H"]]
    D = raw[order, columns["D"]]
    f = raw[order, columns["f"]]
    if np.any(~np.isfinite(a)) or np.any(a<=0) or np.any(np.diff(a) <= 0) or np.any(~np.isfinite(H)) or np.any(H <= 0) or np.any(~np.isfinite(D)) or np.any(D <= 0) or np.any(~np.isfinite(f)) or np.any(f <= 0):
        raise ValueError("Invalid CLASS background or growth history")
    a_start = 1.0 / (1.0 + z_start)
    if a[0] > a_start or a[-1] < 1.0:
        raise ValueError("CLASS background does not span simulation start through today")
    # Interpolate positive amplitudes in log space and the dimensionless rate
    # linearly in ln(a). Exact end rows make the integration range explicit.
    table_start = min(a_start, 1.0e-3) if particle_inputs else a_start
    if a[0]>table_start:raise ValueError('CLASS background does not cover the declared earliest interpolation epoch')
    background_context={'canonical_time':'scale_factor','epoch_column':'a',
        'interpolation':'ln(H), ln(D) linearly in ln(a); f linearly in ln(a)',
        'normalization':'D(a)/D(1); source amplitudes retained'}
    selected = np.unique(np.r_[table_start, a[(a > table_start) & (a < 1)], 1.0])
    ln_a = np.log(a)
    H_out = np.exp(np.interp(np.log(selected), ln_a, np.log(H)))
    if expected_h0_km_s_mpc is not None:
        expected_h0 = float(expected_h0_km_s_mpc)
        if (isinstance(expected_h0_km_s_mpc, (bool, np.bool_)) or not math.isfinite(expected_h0)
                or expected_h0 <= 0 or not math.isclose(float(H_out[-1]) * C_KM_S, expected_h0, rel_tol=1.e-6, abs_tol=1.e-8)):
            raise ValueError('CLASS/AxiCLASS H0 differs from the configured linear bundle')
    if not particle_inputs:
        # EDE exports its exact solver expansion, without pressureless growth
        # columns or a table that could be mistaken for a GADGET EDE input.
        publish_numeric_table(destination/'background_expansion.txt',np.column_stack((selected,H_out*C_KM_S)),
            {'a':{'units':'dimensionless','quantity':'Canonical scale factor'},'H':{'units':'km/s/Mpc','quantity':'CLASS physical expansion rate'}},
            algorithm=__file__,context=background_context,source_hashes={str(source):source_before},
            header='a H[km/s/Mpc]; AxiCLASS scalar-field background')
        return
    D_out = np.exp(np.interp(np.log(selected), ln_a, np.log(D)))
    f_out = np.interp(np.log(selected), ln_a, f)
    H0 = float(H_out[-1])
    D0 = float(D_out[-1])
    physical = np.column_stack((selected, H_out * C_KM_S, D_out / D0, f_out))
    gadget = np.column_stack((selected, H_out / H0, D_out / D0, f_out))
    for name,table,h_unit,h_quantity,header in (('background_expansion.txt',physical,'km/s/Mpc','CLASS physical expansion rate','a H[km/s/Mpc] D(a)/D(1) f=dlnD/dlna'),
            ('gadget_background.txt',gadget,'dimensionless','CLASS H(a)/H0','a E(a)=H/H0 D(a)/D(1) f=dlnD/dlna')):
        publish_numeric_table(destination/name,table,{'a':{'units':'dimensionless','quantity':'Canonical scale factor'},
            'H':{'units':h_unit,'quantity':h_quantity},'D':{'units':'dimensionless','quantity':'D(a)/D(1)'},
            'f':{'units':'dimensionless','quantity':'dlnD/dln(a)'}},
            algorithm=__file__,context=background_context,source_hashes={str(source):source_before},header=header)



@directory_owner('destination')
def export_power(source: Path, destination: Path, reference: bool, *, particle_inputs: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Export conventional P(k) and numeric-only N-GenIC dimensionless power.

    The local GADGET-4 convention is 4*pi*k^3*P_gadget where
    P_gadget=P_CLASS/(2*pi)^3. Thus the actual numeric column is the
    conventional Delta^2=k^3*P_CLASS/(2*pi^2), not 4*pi*k^3*P_CLASS.
    """
    source_before=checksum(source)
    verify_numeric_table(source)
    table = np.loadtxt(source, comments="#", ndmin=2)
    k, power = table[:, 0], table[:, 1]
    if np.any(~np.isfinite(table)) or len(k) < 2 or np.any(k <= 0) or np.any(power <= 0) or np.any(np.diff(k) <= 0):
        raise ValueError(f"Invalid linear P(k): {source}")
    name = "pk_class_z0.txt" if reference else "pk_class.txt"
    from core.cosmology import pk_output_redshift
    actual_z=pk_output_redshift(source)
    context={'epoch':epoch_contract(1/(1+actual_z)) if actual_z is not None else
        {'canonical_time':'scale_factor','actual_epoch_status':'UNDECLARED'},
        'Fourier_convention':'Delta^2 = k^3 P_CLASS / (2 pi^2)','amplitude_rescaled':False}
    publish_numeric_table(destination/name,np.column_stack((k,power)),
        {'k':{'units':'h/Mpc','quantity':'Comoving Fourier wavenumber'},
         'P':{'units':'(Mpc/h)^3','quantity':'Linear CLASS matter power spectrum'}},
        algorithm=__file__,context=context,source_hashes={str(source):source_before},fmt='%.12e',
        header='k[h/Mpc] P_class[(Mpc/h)^3]')
    if not reference and particle_inputs:
        delta2 = k ** 3 * power / (2.0 * math.pi ** 2)
        # No comment, blank line, or header: N-GenIC scans numeric pairs.
        if np.any(~np.isfinite(delta2)) or np.any(delta2<=0):raise ValueError('Dimensionless power exceeds numerical range')
        publish_numeric_table(destination/'pk_gadget.txt',np.column_stack((np.log10(k),np.log10(delta2))),
            {'log10_k':{'units':'dimensionless','quantity':'log10(k / (h/Mpc))'},
             'log10_delta2':{'units':'dimensionless','quantity':'log10(k^3 P_CLASS / (2 pi^2))'}},
            algorithm=__file__,context=context,source_hashes={str(source):source_before},fmt='%.12e')
    return k, power



def check_pk_redshift(path: Path, expected: float) -> None:
    """Verify the solver's printed epoch before using an IC power spectrum."""
    from core.cosmology import class_redshift_matches, pk_output_redshift
    if not class_redshift_matches(pk_output_redshift(path), expected):
        raise ValueError(f"P(k) epoch in {path} does not match requested z={expected:g}")



def plot_power(start: tuple, today: tuple, z_start: float, destination: Path) -> None:
    """Render a readable comparison in physical CLASS P(k) units."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from core.figures import apply_plot_style
    apply_plot_style()
    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#ffffff")
    ax.loglog(*today, color="#172c61", linewidth=2.4, label="Today · z = 0")
    ax.loglog(*start, color="#a65332", linewidth=2.4, label=f"Initial conditions · z = {z_start:g}")
    ax.set(xlabel=r"Wavenumber $k$ [$h\,\mathrm{Mpc}^{-1}$]",
           ylabel=r"Linear matter power $P(k)$ [$(\mathrm{Mpc}/h)^3$]",
           title="Linear matter power across cosmic time")
    ax.grid(which="major", color="#dce2eb", linewidth=0.8)
    ax.grid(which="minor", color="#edf0f5", linewidth=0.4)
    ax.legend(frameon=False, loc="best")
    ax.text(0.03, 0.04, f"z = {z_start:g}  •  a = {1/(1+z_start):.5f}\nreference z = 0",
            transform=ax.transAxes, va="bottom", color="#172c61",
            bbox={"boxstyle": "round,pad=0.55", "facecolor": "#f1eee6", "edgecolor": "none"})
    from core.figures import export_figure
    export_figure(fig, destination, metadata={'source':'CLASS/AxiCLASS','epochs':[epoch_contract(1/(1+z_start)),epoch_contract(1.)],
        'science_acceptance':'NOT_ACCEPTED','source_products':['pk_class.txt','pk_class_z0.txt']})
    plt.close(fig)



@directory_owner('output_root')
def run(cfg: dict, output_root: Path, binary: Path) -> Path:
    """Execute isolated start and reference epochs; publish after validation."""
    from core.cosmology import class_output_files, solver_execution_provenance
    validate(cfg)
    target = output_root / cfg["name"]
    if target.exists():
        raise FileExistsError(f"Run already exists: {target}")
    target.mkdir(parents=True)
    atomic_json(target/'execution_status.json',{'status':'PENDING','science_acceptance':'NOT_ACCEPTED'})
    sources = {}
    executed_parameters = {}
    try:
        with native_run_lock(Path(binary).expanduser().resolve().parent, operation='linear solver and provenance'):
            for epoch, redshift in (("start", cfg["z_start"]), ("z0", 0.0)):
                work = target / "solver" / epoch
                work.mkdir(parents=True)
                # CLASS appends run suffixes inside a fixed-size root buffer.
                # Keep the native prefix short even for deep workflow generations.
                params = solver_parameters(cfg, Path("class_"), redshift)
                executed_parameters[epoch] = dict(params)
                ini = work / "input.ini"
                atomic_text(ini,"".join(f"{key} = {value}\n" for key, value in params.items()))
                log = work / "solver.log"
                with log.open("w", encoding="utf-8") as stream:
                    result = run_native_command([str(binary.resolve()),ini.name],
                        cwd=work.resolve(),receipt_path=work/'invocation.json',stdout=stream,stderr=subprocess.STDOUT)
                if result.returncode:
                    raise RuntimeError(f"{cfg['model']} solver failed at z={redshift:g}; see {log}")
                products = class_output_files(work/'class_', redshift)
                pk, tk, bg = products['pk'], products['transfer'], products['background']
                execution_provenance=solver_execution_provenance('axiclass' if cfg['model']=='ede' else 'class',
                    binary,ini,log,{'power_spectrum':pk,'transfer':tk,'background':bg},redshift)
                atomic_json(work/'solver_provenance.json',execution_provenance)
                sources[epoch] = {"pk": pk, "transfer": tk, "background": bg, "ini": ini, "log": log}
            # EDE sets the linear initial conditions and smooth expansion table.
            # Gadget evolves matter particles, not nonlinear scalar-field modes.
            particle_inputs = True
            start = export_power(sources["start"]["pk"], target, reference=False, particle_inputs=particle_inputs)
            today = export_power(sources["z0"]["pk"], target, reference=True)
            if start[0][-1] < cfg["k_max_h_mpc"] * 0.98:
                raise ValueError("CLASS P(k) grid does not reach the requested k maximum")
            export_background(sources["start"]["background"], target, cfg["z_start"], particle_inputs=particle_inputs,
                              expected_h0_km_s_mpc=cfg['H0'])
            plot_power(start, today, cfg["z_start"], target / "matter_power.png")
            metadata = {
                "schema_version": 2, "status": "complete", "model": cfg["model"],
                "calculation_scope": "linear_boltzmann_and_particle_inputs",
                "matter_definition": "baryons plus collisionless matter; scalar dark energy excluded from delta_m",
                "solver": "AxiCLASS" if cfg["model"] == "ede" else "CLASS",
                "solver_binary": {"path": str(binary.resolve()), "sha256": checksum(binary.resolve())},
                "parameters": cfg, "h": cfg["H0"] / 100.0,
                'solver_provenance':{epoch:{'path':str((files['ini'].parent/'solver_provenance.json').relative_to(target)),
                    'sha256':checksum(files['ini'].parent/'solver_provenance.json')} for epoch,files in sources.items()},
                "class_parameters": executed_parameters,
                "units": {"k": "h/Mpc", "P": "(Mpc/h)^3", "H": "km/s/Mpc", "a": "dimensionless"},
                "gadget_pk_convention": "log10(k[h/Mpc]), log10(k^3 P_CLASS/(2 pi^2)) at z_start",
                "gadget_background_convention": "a, H(a)/H0, D(a)/D(1), f(a); includes a=0.001 for Gadget table interpolation",
                "source_files": {epoch: {kind: str(path.relative_to(target)) for kind, path in files.items()}
                                 for epoch, files in sources.items()},
                "checksums_sha256": {str(path.relative_to(target)): checksum(path)
                                     for path in sorted(target.rglob("*")) if path.is_file() and path.name!="execution_status.json" and ".native-run-locks" not in path.relative_to(target).parts},
            }
            metadata['science_acceptance']='NOT_ACCEPTED'
            metadata['algorithm_sha256']=checksum(Path(__file__))
            atomic_json(target/'spectrum.json',metadata)
            atomic_text(target/'spectrum.sha256',checksum(target/'spectrum.json')+'\n')
            for path in target.rglob('*'):
                if path.is_file():
                    with path.open('rb') as stream:os.fsync(stream.fileno())
            for path in sorted([p for p in target.rglob('*') if p.is_dir()],key=lambda p:len(p.parts),reverse=True):fsync_directory(path)
            atomic_json(target/'execution_status.json',{'status':'COMPLETE','science_acceptance':'NOT_ACCEPTED',
                'manifest_sha256':checksum(target/'spectrum.json')})
            return target
    except BaseException as exc:
        atomic_json(target/'execution_status.json',{'status':'FAILED','science_acceptance':'NOT_ACCEPTED',
            'failure':{'type':type(exc).__name__,'message':str(exc)}})
        raise



def verify_spectrum_directory(directory):
    """Verify published modular solver artifacts before IC preparation reuses them."""
    root=Path(directory).resolve();manifest=root/'spectrum.json'
    status_path=root/'execution_status.json';seal=root/'spectrum.sha256'
    if status_path.exists():
        status=json.loads(status_path.read_text())
        if status.get('status')!='COMPLETE':raise ValueError('Modular solver publication is incomplete')
        if status.get('manifest_sha256')!=checksum(manifest):raise ValueError('Modular solver manifest changed')
    if seal.exists() and seal.read_text().strip()!=checksum(manifest):raise ValueError('Modular solver manifest checksum mismatch')
    record=json.loads(manifest.read_text())
    if record.get('status')!='complete':raise ValueError('Modular solver manifest is incomplete')
    for relative,expected in record['checksums_sha256'].items():
        name=Path(relative);path=(root/name).resolve()
        if name.is_absolute() or '..' in name.parts or not path.is_relative_to(root):
            raise ValueError('Modular solver artifact escapes saved run')
        if not path.is_file() or checksum(path)!=expected:raise ValueError('Modular solver artifact changed: '+relative)
    for name in ('pk_class.txt','pk_class_z0.txt','pk_gadget.txt','background_expansion.txt','gadget_background.txt'):
        if (root/name).exists():verify_numeric_table(root/name)
    return dict(record,integrity_status='SEALED_ARTIFACTS_VERIFIED' if seal.exists() and status_path.exists() else 'LEGACY_UNSEALED')



def ask_number(label: str, low: float, high: float) -> float:
    """Read a finite physical parameter within its declared inclusive range."""
    while True:
        answer = input(f"{label}: ").strip()
        try:
            value = float(answer)
            if math.isfinite(value) and low <= value <= high:
                return value
        except ValueError:
            pass
        print(f"Enter a finite number from {low:g} to {high:g}.")



def prompt_config() -> dict:
    """Collect every user-facing model choice and simulation-relevant unit."""
    while True:
        name = input("Run name [identifier; letters, digits, underscores, hyphens]: ").strip()
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", name):
            break
        print("Use 1–64 letters, digits, underscores, or hyphens.")
    while True:
        model = input("Model (lcdm/ede): ").strip().lower()
        if model in {"lcdm", "ede"}:
            break
        print("Choose lcdm or ede.")
    cfg = {"name": name, "model": model}
    from core.cli_prompts import required_values
    if model == "ede":
        cfg["f_ede"] = ask_number("Peak early-dark-energy fraction f_EDE [dimensionless]", 0.000001, 0.5)
        cfg["log10_zc"] = ask_number("log10(z_c), redshift of peak EDE fraction [dimensionless]", 2.0, 6.0)
        cfg["theta_i"] = ask_number("Initial axion angle theta_i [radians]", 0.01, 3.13)
        cfg["n_axion"] = ask_number("Axion potential exponent n [dimensionless]", 2.0, 10.0)
    cfg.update(required_values("Background cosmology (Omega values, not omega=Omega*h^2)", [
        ("H0", "Hubble constant [km/s/Mpc]", float, 40., 100.),
        ("Omega_b", "present baryon density / critical density", float, .001, .2),
        ("Omega_cdm", "present CDM density / critical density", float, .01, .8)]))
    cfg.update(required_values("Primordial spectrum and reionization", [
        ("A_s", "scalar power amplitude", float, 1e-10, 1e-8),
        ("n_s", "scalar spectral index", float, .8, 1.2),
        ("tau_reio", "reionization optical depth", float, .001, .2)]))
    cfg.update(required_values("Initial epoch and spectrum coverage", [
        ("z_start", "initial-condition redshift", float, 5., 1000.),
        ("k_max_h_mpc", "maximum wavenumber [h/Mpc]", float, 1., 10000.)]))
    validate(cfg)
    return cfg



def validate(cfg: dict) -> None:
    """Apply the same bounds to interactive and JSON configurations."""
    bounds = {"H0": (40, 100), "Omega_b": (0.001, 0.2), "Omega_cdm": (0.01, 0.8),
              "A_s": (1e-10, 1e-8), "n_s": (0.8, 1.2), "tau_reio": (0.001, 0.2),
              "z_start": (5, 1000), "k_max_h_mpc": (1, 10000)}
    if cfg.get("model") not in {"lcdm", "ede"}:
        raise ValueError("model must be lcdm or ede")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", str(cfg.get("name", ""))):
        raise ValueError("Invalid run name")
    if cfg["model"] == "ede":
        bounds.update({"f_ede": (0.000001, 0.5), "log10_zc": (2, 6),
                       "theta_i": (0.01, 3.13), "n_axion": (2, 10)})
    for key, (low, high) in bounds.items():
        value = float(cfg[key])
        if not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f"{key} must be from {low:g} to {high:g}")
        cfg[key] = value
    if cfg["Omega_b"] + cfg["Omega_cdm"] >= 0.95:
        raise ValueError("Omega_b + Omega_cdm must be below 0.95")



def run_class_main() -> None:
    """Resolve the compiled solver and generate self-contained run products."""
    parser = argparse.ArgumentParser(description="Small, reusable CLASS/AxiCLASS runner for the modular suite.\n\nThe parameter and N-GenIC conventions follow the validated implementations in\nthe repository's original ``cosmology.py``. Pinned solver source and patches are bundled in core/software.")
    parser.add_argument("--config", type=Path, help="JSON object with the interactive prompt keys")
    parser.add_argument('--output',type=Path,help='Explicit parent output directory for the isolated named cosmology run')
    parser.add_argument('--binary',type=Path,help='Explicit compiled CLASS/AxiCLASS executable; native provenance remains mandatory')
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8")) if args.config else prompt_config()
    validate(cfg)
    from core.native import native_engine_directory
    solver = args.binary.expanduser().resolve() if args.binary else native_engine_directory('axiclass' if cfg['model']=='ede' else 'class')/'class'
    if not solver.is_file():
        raise FileNotFoundError(f"Compiled solver is missing: {solver}")
    output = run(cfg, args.output.expanduser().resolve() if args.output else runtime_paths().runs_root / "linear", solver)
    print(f"Complete: {output}")
    print(f"Linear spectrum: {output / 'pk_class.txt'}")
    print(f"Expansion history: {output / 'background_expansion.txt'}")
    if cfg["model"] != "ede":
        print(f"Particle IC input: {output / 'pk_gadget.txt'}")

