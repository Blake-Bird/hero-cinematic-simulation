"""Explicit numerical controls for the native MUSIC transfer interface.

These deliberately synthetic inputs check species velocities, normalization,
periodic coordinates, mixed-fluid 2LPT and refinement continuity. They support
admission of a specific generator executable; they are not production ICs or
astrophysical results. Preparation and inspection launch no native work.
"""
from __future__ import annotations

import argparse
import configparser
import json
import math
import os
import subprocess
from pathlib import Path
import h5py
import numpy as np
from scipy.integrate import solve_ivp
from scipy.spatial import cKDTree
from core.cosmology import ede_parameters
from core.foundation import atomic_json, file_sha256, KPC_CM, C_SI
from core.native import run_native_command
from core.data import export_numeric_audit
from core.initial_conditions import validate_ic_realization, compare_ic_generators, validate_refinement_continuity
from core.zoom import stage_zoom_inputs, compare_density_phases, linear_density_modes, velocity_divergence_modes, audit_music_ic, audit_zoom_velocity

def solver_bundle(root, *, redshift=99., model='lcdm', amplitude=2.1e-9, ede_bump=0.):
    if isinstance(redshift,bool) or not isinstance(redshift,(int,float)) or not math.isfinite(redshift) or not 0 <= redshift <= 1e10-1:
        raise ValueError('Control redshift lies outside the retained growth grid')
    if model not in {'lcdm','ede'} or not math.isfinite(amplitude) or amplitude <= 0 or not math.isfinite(ede_bump) or ede_bump < 0:
        raise ValueError('Control needs a supported model, positive amplitude and nonnegative expansion bump')
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=False)
    om,ob,orr,h = .3,.05,9e-5,.7
    a=np.geomspace(1e-10,1.,2200)
    def expansion(x):
        av=math.exp(x)
        base=orr/av**4+om/av**3+1-om-orr
        bump=ede_bump*math.exp(-((x-math.log(.001))/.7)**2)
        dl=(-4*orr/av**4-3*om/av**3)/base
        return math.sqrt(base*(1+bump)),.5*(dl-2*(x-math.log(.001))/.7**2*bump/(1+bump))
    seed=om*a[0]/orr
    def rhs(x,state):
        e,derivative=expansion(x)
        return [state[1],-(2+derivative)*state[1]+1.5*om/math.exp(3*x)/e**2*state[0]]
    solution=solve_ivp(rhs,(math.log(a[0]),0),(1+1.5*seed,1.5*seed),method='DOP853',
                       dense_output=True,rtol=1e-11,atol=1e-12,max_step=.03)
    if not solution.success:raise RuntimeError(solution.message)
    d,dx=solution.sol(np.log(a));d0=d[-1]
    H=np.array([expansion(x)[0] for x in np.log(a)])*70/(C_SI/1000)
    np.savetxt(root/'background.dat',np.column_stack((1/a-1,H,d/d0,dx/d,orr/a**4/(H/(70/(C_SI/1000)))**2)),
        header='1:z 2:H [1/Mpc] 3:gr.fac. D 4:gr.fac. f 5:Omega_r(z)')
    k=np.geomspace(1e-5,1000,256)
    dc=-1e5*k*k/(1+(k/.1)**2)
    db=.9*dc
    dm=(1-ob/om)*dc+ob/om*db
    ast=1/(1+redshift);state=solution.sol(math.log(ast))
    f=state[1]/state[0]
    aH=ast*expansion(math.log(ast))[0]*70/(C_SI/1000)
    theta=-f*aH*dm
    np.savetxt(root/'transfer.dat',np.column_stack((k,db,dc,theta,theta,np.zeros_like(k),np.zeros_like(k))),
        header=f'transfer functions at redshift z = {redshift}\n1:k 2:d_b 3:d_cdm 4:t_b 5:t_cdm 6:h_t_nb_prime 7:k2gamma_nb')
    solver={'H0':70.,'omega_b':ob*h*h,'omega_cdm':(om-ob)*h*h,'A_s':amplitude,'n_s':.96,
        'T_cmb':2.7255,'N_ur':3.046,'z_pk':redshift,'output':'mPk,mTk,vTk','nbody_gauge_transfer_functions':'yes'}
    params={'model':model,'z_start':redshift,'H0':70.,'Omega_b':ob,'Omega_cdm':om-ob,'A_s':amplitude,'n_s':.96}
    if model == 'ede':
        params.update(f_ede=.1,log10_zc=math.log10(3500.),theta_i=2.8,n_axion=3.)
        solver.update(ede_parameters({'ede':{'parameterization':'axion','f_ede':.1,'z_c':3500.,
                       'theta_i':2.8,'n_axion':3.,'extra_axiclass':{}}}))
    (root/'input.ini').write_text(''.join(f'{key} = {value}\n' for key,value in solver.items()))
    info={'schema_version':2,'status':'complete','model':model,'solver':'AxiCLASS' if model=='ede' else 'CLASS',
        'parameters':params,'class_parameters':{'start':solver},
        'source_files':{'start':{'transfer':'transfer.dat','background':'background.dat','ini':'input.ini'}},
        'checksums_sha256':{name:file_sha256(root/name) for name in ('transfer.dat','background.dat','input.ini')},
        'fixture':'SYNTHETIC pressureless growth control; no Boltzmann solver execution or science acceptance'}
    (root/'spectrum.json').write_text(json.dumps(info,indent=2)+'\n')
    return root/'spectrum.json'


def grafic_noise(path, n=32):
    """Persist one band-limited realization in the native shared GRAFIC format."""
    raw=np.random.default_rng(123456).normal(size=(n,)*3)
    k=np.meshgrid(*([np.fft.fftfreq(n)*n]*3),indexing='ij',sparse=True)
    radius=sum(x*x for x in k)
    modes=np.fft.fftn(raw);modes[(radius>9)|(radius==0)]=0
    values=np.fft.ifftn(modes).real;values/=values.std();values*=.01
    with Path(path).open('wb') as stream:
        np.array([16,n,n,n,123456,16],dtype='=i4').tofile(stream)
        for z in range(n):
            np.array([n*n*8],dtype='=i4').tofile(stream)
            values[:,:,z].T.astype('=f8').tofile(stream)
            np.array([n*n*8],dtype='=i4').tofile(stream)
    return values


CASES = ("uniform-reference", "uniform-velocity-double", "uniform-zero")


NESTED_CASES = (
    "dms-reference",
    "dms-velocity-double",
    "dms-zero",
    "nested-common-growth",
)


PERIODIC_CASES = ("periodic-common-growth", "periodic-zero", "periodic-shifted-zero")


REJECTIONS = {
    "reject-second-order": ("use_2LPT=no", "use_2LPT=yes", "two-fluid 1LPT"),
    "reject-backscaling": ("ztarget=99", "ztarget=0", "ztarget=zstart"),
    "reject-gauge": ("gauge: N-body", "gauge: synchronous", "schema, N-body gauge"),
    "reject-model": ("Model: lcdm;", "Model: ede;", "schema, N-body gauge"),
    "reject-redshift": ("redshift: 99", "redshift: 98", "schema, N-body gauge"),
    "reject-mode-coverage": (
        "boxlength=10",
        "boxlength=0.0001",
        "does not cover zoom Fourier modes",
    ),
    "reject-poisson": (
        "accuracy=1e-14",
        "accuracy=1e-14\nmaximum_iterations=2",
        "Poisson residual failed declared tolerance",
    ),
}


def native_music_transfer_prepare(root, nested=False):
    root = Path(root).resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError('Use a fresh generator-control directory')
    common = """[setup]
boxlength=10
zstart=99
levelmin=4
levelmin_TF=4
levelmax=4
padding=2
baryons=yes
do_SPH=yes
use_LLA=no
use_2LPT=no
[cosmology]
Omega_m=0.3
Omega_b=0.05
Omega_L=0.7
H0=70
n_s=0.96
A_s=2.1e-9
k_p=0.05
m_nu1=0
m_nu2=0
m_nu3=0
transfer=DMS_file
input_model=lcdm
ztarget=99
transfer_file=transfers.txt
[random]
seed[4]=140
[output]
format=arepo
filename=zoom_fixture.hdf5
arepo_coarsetype=2
arepo_doubleprec=yes
arepo_longids=yes
arepo_num_files=1
"""
    k = np.geomspace(1e-5, 1000, 128)
    density = -1e5 * k * k / (1 + (k / 0.1) ** 2)
    for case, scale in zip(CASES, (1.0, 2.0, 0.0)):
        destination = root / case
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "fixture.conf").write_text(common)
        table = np.column_stack(
            (
                k,
                density,
                0.9 * density,
                -scale * 0.001 * density,
                -scale * 0.0011 * density,
            )
        )
        if scale == 0:
            table[:, 1:] = 0
        np.savetxt(
            destination / "transfers.txt",
            table,
            header="DarkMatterSim monofonIC transfer table v1\n gauge: N-body\n Model: lcdm; redshift: 99\n growth_f_start: 1\n aH_start_Mpc^-1: 0.001\n Synthetic normalization control; not science ICs",
        )
    if nested:
        nested_config = (
            common.replace("levelmax=4", "levelmax=6") + "\n[poisson]\naccuracy=1e-14\n"
        )
        nested_config = nested_config.replace(
            "padding=2",
            "padding=2\nref_center=0.5,0.5,0.5\nref_extent=0.2,0.2,0.2\nalign_top=no",
        )
        nested_config = nested_config.replace(
            "seed[4]=140", "seed[4]=140\nseed[5]=150\nseed[6]=160"
        )
        for case, scale in zip(NESTED_CASES, (1, 2, 0, 1)):
            destination = root / case
            destination.mkdir(parents=True, exist_ok=True)
            (destination / "fixture.conf").write_text(nested_config)
            density = -10 * k * k / (1 + (k / 0.1) ** 2)
            if case == "nested-common-growth":
                density *= 1e4
            theta_b = (
                -0.0009 * density
                if case == "nested-common-growth"
                else -0.0011 * density
            )
            table = np.column_stack(
                (k, density, 0.9 * density, -scale * 0.001 * density, scale * theta_b)
            )
            if scale == 0:
                table[:, 1:] = 0
            np.savetxt(
                destination / "transfers.txt",
                table,
                header="DarkMatterSim monofonIC transfer table v1\ngauge: N-body\nModel: lcdm; redshift: 99\ngrowth_f_start: 1\naH_start_Mpc^-1: 0.001\nSynthetic nested normalization control; not science ICs",
            )
        for case, source_case in zip(
            PERIODIC_CASES, ("nested-common-growth", "dms-zero", "dms-zero")
        ):
            destination = root / case
            destination.mkdir(parents=True, exist_ok=True)
            cfg = nested_config.replace("ref_center=0.5,0.5,0.5", "ref_center=0,0,0")
            if case == "periodic-shifted-zero":
                cfg = cfg.replace(
                    "arepo_num_files=1", "arepo_num_files=1\nshift_back=no"
                )
            (destination / "fixture.conf").write_text(cfg)
            (destination / "transfers.txt").write_bytes(
                (root / source_case / "transfers.txt").read_bytes()
            )
        for case, (old, new, _) in REJECTIONS.items():
            destination = root / case
            destination.mkdir(parents=True, exist_ok=True)
            cfg = (root / "dms-reference" / "fixture.conf").read_text()
            table = (root / "dms-reference" / "transfers.txt").read_text()
            if old in cfg:
                cfg = cfg.replace(old, new)
            elif old in table:
                table = table.replace(old, new)
            else:
                raise AssertionError(
                    "Rejection fixture substitution did not match: " + case
                )
            (destination / "fixture.conf").write_text(cfg)
            (destination / "transfers.txt").write_text(table)


def verify_lagrangian(first, controls, zero, key):
    q = first[key]["LagrangianCoordinates"][:]
    if (
        first["Header"].attrs.get("Music_lagrangian_schema") != 1
        or q.dtype != np.float64
    ):
        raise AssertionError(
            "Native Lagrangian coordinates need schema 1 and double precision"
        )
    if not all(np.array_equal(q, f[key]["LagrangianCoordinates"][:]) for f in controls):
        raise AssertionError("Perturbations altered unperturbed particle coordinates")
    if not np.allclose(q, zero[key]["Coordinates"][:], rtol=0, atol=1e-10):
        raise AssertionError(
            "Native q disagrees with the zero-displacement particle load"
        )


def verify(root):
    root = Path(root)
    report = {
        "fixture": "Synthetic uniform two-fluid 1LPT normalization control; not zoom acceptance",
        "species": {},
        "inputs": {},
    }
    for case in CASES:
        report["inputs"][case] = {
            name: file_sha256(root / case / name)
            for name in ("fixture.conf", "transfers.txt", "zoom_fixture.hdf5")
        }
    with (
        h5py.File(root / CASES[0] / "zoom_fixture.hdf5") as first,
        h5py.File(root / CASES[1] / "zoom_fixture.hdf5") as doubled,
        h5py.File(root / CASES[2] / "zoom_fixture.hdf5") as zero,
    ):
        for t in (0, 1):
            key = f"PartType{t}"
            verify_lagrangian(first, (doubled, zero), zero, key)
            ids = first[key]["ParticleIDs"][:]
            if not all(
                np.array_equal(ids, f[key]["ParticleIDs"][:]) for f in (doubled, zero)
            ):
                raise AssertionError("Control particle ordering differs")
            x = first[key]["Coordinates"][:]
            v = first[key]["Velocities"][:]
            if not np.array_equal(x, doubled[key]["Coordinates"][:]):
                raise AssertionError("Velocity transfer altered density positions")
            if not np.array_equal(2 * v, doubled[key]["Velocities"][:]):
                raise AssertionError("Native velocities did not scale linearly")
            if np.any(zero[key]["Velocities"][:]):
                raise AssertionError("Zero transfer generated nonzero velocities")
            # Independently delta=-div(Psi), CLASS theta=div(v)/c, hence
            # v_pec=-(theta/delta)*c/h * displacement[Mpc/h]. The HDF5
            # velocity divides by sqrt(a), coordinates use native kpc/h.
            ratio = 0.001 if t == 1 else 0.0011 / 0.9
            coefficient = ratio * (C_SI/1000) / 0.7 / 1000 / 0.1
            delta = (x - zero[key]["Coordinates"][:] + 5000) % 10000 - 5000
            residual = float(
                np.linalg.norm(v - coefficient * delta) / np.linalg.norm(v)
            )
            if not np.isfinite(residual) or residual > 1e-9:
                raise AssertionError(f"Absolute velocity units fail: {residual}")
            report["species"][str(t)] = {
                "positions_bitwise_equal": True,
                "velocity_doubling_bitwise_exact": True,
                "zero_transfer_velocity_exactly_zero": True,
                "absolute_velocity_displacement_relative_residual": residual,
            }
    (root / "uniform-normalization-audit.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    return report


def verify_nested(root):
    root = Path(root)
    report = {
        "status": "PASS",
        "fixture": "Synthetic nested two-fluid 1LPT controls; not target-halo or spectrum acceptance",
        "species": {},
    }
    with (
        h5py.File(root / NESTED_CASES[0] / "zoom_fixture.hdf5") as first,
        h5py.File(root / NESTED_CASES[1] / "zoom_fixture.hdf5") as doubled,
        h5py.File(root / NESTED_CASES[2] / "zoom_fixture.hdf5") as zero,
        h5py.File(root / NESTED_CASES[3] / "zoom_fixture.hdf5") as common,
    ):
        for t in (0, 1, 2):
            key = f"PartType{t}"
            verify_lagrangian(first, (doubled, zero, common), zero, key)
            ids = first[key]["ParticleIDs"][:]
            if not all(
                np.array_equal(ids, f[key]["ParticleIDs"][:])
                for f in (doubled, zero, common)
            ):
                raise AssertionError("Nested particle ordering differs")
            x, v = first[key]["Coordinates"][:], first[key]["Velocities"][:]
            if not np.array_equal(x, doubled[key]["Coordinates"][:]):
                raise AssertionError(
                    "Nested velocity transfer altered density positions"
                )
            scaling = float(
                np.linalg.norm(doubled[key]["Velocities"][:] - 2 * v)
                / np.linalg.norm(2 * v)
            )
            if not np.isfinite(scaling) or scaling > 1e-10:
                raise AssertionError(f"Nested velocity doubling failed: {scaling}")
            if np.any(zero[key]["Velocities"][:]):
                raise AssertionError(
                    "Zero nested transfer generated nonzero velocities"
                )
            displacement = (
                common[key]["Coordinates"][:] - zero[key]["Coordinates"][:] + 5000
            ) % 10000 - 5000
            velocity = common[key]["Velocities"][:] + common["Header"].attrs["Music_removed_bulk_velocity_gadget"]
            # Both species have theta/delta=-.001, so combined coarse matter
            # has the same independent continuity relation as gas and fine DM.
            coefficient = 0.001 * (C_SI/1000) / 0.7 / 1000 / 0.1
            continuity = float(
                np.linalg.norm(velocity - coefficient * displacement)
                / np.linalg.norm(velocity)
            )
            if not np.isfinite(continuity) or continuity > 1e-9:
                raise AssertionError(f"Nested absolute continuity failed: {continuity}")
            report["species"][str(t)] = {
                "positions_bitwise_equal": True,
                "zero_transfer_velocity_exactly_zero": True,
                "velocity_doubling_relative_residual": scaling,
                "absolute_velocity_displacement_relative_residual": continuity,
            }
    report["inputs"] = {
        case: {
            name: file_sha256(root / case / name)
            for name in (
                "fixture.conf",
                "transfers.txt",
                "zoom_fixture.hdf5",
                "native-ic-generation.log",
            )
        }
        for case in NESTED_CASES
    }
    (root / "nested-normalization-audit.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    return report


def verify_periodic(root):
    root = Path(root)
    report = {
        "status": "PASS",
        "fixture": "Synthetic periodic corner, original-frame q/x and physical continuity",
        "species": {},
    }
    with (
        h5py.File(root / PERIODIC_CASES[0] / "zoom_fixture.hdf5") as first,
        h5py.File(root / PERIODIC_CASES[1] / "zoom_fixture.hdf5") as zero,
        h5py.File(root / PERIODIC_CASES[2] / "zoom_fixture.hdf5") as shifted,
    ):
        if (
            first["Header"].attrs["Music_shift_back"] != 1
            or shifted["Header"].attrs["Music_shift_back"] != 0
        ):
            raise AssertionError("Periodic output frame metadata is incorrect")
        np.testing.assert_allclose(
            first["Header"].attrs["Music_output_coordinate_shift_box_fraction"],
            [-0.5] * 3,
            rtol=0,
            atol=1e-14,
        )
        for t in (0, 1, 2):
            key = f"PartType{t}"
            verify_lagrangian(first, (zero,), zero, key)
            for other in (zero, shifted):
                if not np.array_equal(
                    first[key]["ParticleIDs"][:], other[key]["ParticleIDs"][:]
                ):
                    raise AssertionError("Periodic control ordering differs")
            q = first[key]["LagrangianCoordinates"][:]
            translated = (shifted[key]["Coordinates"][:] - 5000) % 10000
            # 0 and L are the same point on the torus; roundoff at a boundary
            # must be compared using periodic distance, not absolute x.
            frame_delta = (q - translated + 5000) % 10000 - 5000
            if np.max(np.abs(frame_delta)) > 1e-10:
                raise AssertionError(
                    "Output failed to restore the requested parent frame"
                )
            displacement = (first[key]["Coordinates"][:] - q + 5000) % 10000 - 5000
            velocity = first[key]["Velocities"][:] + first["Header"].attrs["Music_removed_bulk_velocity_gadget"]
            coefficient = 0.001 * (C_SI/1000) / 0.7 / 1000 / 0.1
            residual = float(
                np.linalg.norm(velocity - coefficient * displacement)
                / np.linalg.norm(velocity)
            )
            if not np.isfinite(residual) or residual > 1e-9:
                raise AssertionError(f"Periodic physical continuity failed: {residual}")
            if np.any(zero[key]["Velocities"][:]):
                raise AssertionError("Periodic zero transfer has nonzero velocity")
            report["species"][str(t)] = {
                "original_frame_restored": True,
                "absolute_velocity_displacement_relative_residual": residual,
            }
    report["inputs"] = {
        case: {
            name: file_sha256(root / case / name)
            for name in (
                "fixture.conf",
                "transfers.txt",
                "zoom_fixture.hdf5",
                "native-ic-generation.log",
            )
        }
        for case in PERIODIC_CASES
    }
    return report


def native_music_transfer_main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--nested", action="store_true")
    args = parser.parse_args()
    args.root = args.root.resolve()
    if args.prepare:
        native_music_transfer_prepare(args.root, args.nested)
    execution = {}
    if args.binary:
        binary = args.binary.resolve()
        execution["binary_sha256"] = file_sha256(binary)
        for case in CASES + (NESTED_CASES + PERIODIC_CASES if args.nested else ()):
            with (args.root / case / "native-ic-generation.log").open("w") as log:
                subprocess.run(
                    [str(binary), "fixture.conf"],
                    cwd=args.root / case,
                    env={**os.environ, "OMP_NUM_THREADS": "2"},
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
        if args.nested:
            execution["rejections"] = {}
            for case, (_, _, expected) in REJECTIONS.items():
                destination = args.root / case
                with (destination / "native-ic-generation.log").open("w") as log:
                    result = subprocess.run(
                        [str(binary), "fixture.conf"],
                        cwd=destination,
                        env={**os.environ, "OMP_NUM_THREADS": "2"},
                        stdout=log,
                        stderr=subprocess.STDOUT,
                    )
                content = (destination / "native-ic-generation.log").read_text()
                if result.returncode == 0 or expected not in content:
                    raise AssertionError(
                        f"Native rejection did not fail for its declared reason: {case}, exit={result.returncode}"
                    )
                execution["rejections"][case] = {
                    "exit_code": result.returncode,
                    "expected_diagnostic": expected,
                    "input_hashes": {
                        name: file_sha256(destination / name)
                        for name in (
                            "fixture.conf",
                            "transfers.txt",
                            "native-ic-generation.log",
                        )
                    },
                }
    # This PASS is a numerical generator-control result. No native
    # evolution or astrophysical acceptance follows from these toy transfers.
    report = {"status": "PASS", "uniform": verify(args.root), "execution": execution}
    conservation = {}
    for case in CASES + (NESTED_CASES + PERIODIC_CASES if args.nested else ()):
        result = validate_ic_realization(args.root/case/'zoom_fixture.hdf5',args.root/case/'fixture.conf')
        conservation[case] = {name:result['checks'][name] for name in ('mass','exact_ids','momentum','barycenter')}
        if any(row['status'] != 'PASS' for row in conservation[case].values()):
            raise AssertionError('Native delivered conservation failed: '+case)
    report['conservation'] = conservation
    report["regression_driver_sha256"] = file_sha256(Path(__file__))
    if args.nested:
        report["nested"] = verify_nested(args.root)
        report["periodic"] = verify_periodic(args.root)
    (args.root / "native-transfer-suite.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


def config(path):
    result=configparser.ConfigParser(interpolation=None);result.optionxform=str;result.read(path);return result


def native_music_mixed2lpt_run(binary,directory,recipe,*,threads=2):
    path=directory/'fixture.conf'
    with path.open('w') as stream:recipe.write(stream)
    with (directory/'generation.log').open('w') as log:
        run_native_command([str(binary),'fixture.conf'],cwd=directory,receipt_path=directory/'invocation.json',
            inputs=[path,Path(recipe['cosmology']['transfer_file'])],env=dict(os.environ,OMP_NUM_THREADS=str(threads)),
            stdout=log,stderr=subprocess.STDOUT)


def prepare_native(root,binary,z=99.,model='lcdm',bump=0.):
    manifest=solver_bundle(root/'source',redshift=z,model=model,ede_bump=bump)
    staged=stage_zoom_inputs(manifest,root/'staged',box_mpc_h=10.,parent_level=5,refinement_level=6,
        seeds_by_level={'5':150,'6':160},padding_cells=2,poisson_accuracy=1e-14,lpt_order=2)
    noise=root/'shared-noise.grafic';grafic_noise(noise)
    results={}
    for name,zoom in (('uniform',False),('nested',True)):
        directory=root/name;directory.mkdir()
        recipe=config(root/'staged'/('zoom_template.conf' if zoom else 'parent.conf'))
        recipe['random']['seed[5]']=str(noise)
        recipe['random']['grafic_sign']='yes'
        recipe['setup'].update(ref_center='0.5,0.5,0.5',ref_extent='0.2,0.2,0.2',align_top='no') if zoom else None
        recipe['output']['filename']='ics_gadget.hdf5'
        native_music_mixed2lpt_run(binary,directory,recipe)
        ic=directory/'ics_gadget.hdf5'
        results[name]={'conservation':validate_ic_realization(ic,directory/'fixture.conf',directory/'physical.json'),
            'velocity':audit_zoom_velocity(ic,root/'staged/transfers.txt',directory/'velocity.json',model=model)}
        if results[name]['conservation']['status']=='FAIL' or results[name]['velocity']['status']!='PASS':
            raise AssertionError('Mixed native conservation/velocity failed: '+name)
    results['phases']=compare_density_phases(root/'uniform/ics_gadget.hdf5',root/'nested/ics_gadget.hdf5',
        root/'uniform/wnoise_0005.bin',root/'nested/wnoise_0005.bin',root/'phases.json')
    if results['phases']['status']!='PASS':raise AssertionError('Mixed native parent phases failed')
    results['growth']=staged['second_order_growth']
    return results


def monofonic_control(root,binary,staged,noise):
    directory=root/'monofonic';directory.mkdir()
    native=config(staged/'parent.conf')
    cosm=native['cosmology'];z=native.getfloat('setup','zstart')
    # The same frozen GRAFIC field drives both engines. A separate zero-
    # transfer realization supplies original q; measured x is never inverted.
    common=f'''[setup]
GridRes = 32
BoxLength = 10
zstart = {z}
LPTorder = 2
DoBaryons = yes
DoBaryonVrel = yes
DoFixing = no
DoInversion = no
DoRemoveCornerModes = no
ParticleLoad = sc
[cosmology]
ParameterSet = none
Omega_m = {cosm['Omega_m']}
Omega_b = {cosm['Omega_b']}
Omega_L = {cosm['Omega_L']}
Omega_r = 0.00009
H0 = 70
Tcmb = 2.7255
n_s = 0.96
A_s = 2.1e-9
N_ur = 3.046
m_nu1 = 0
ZeroRadiation = false
transfer = DMS_file
ztarget = {z}
TransferComponent = baryonpluscdm
input_model = lcdm
transfer_file = {staged/'transfers.txt'}
background_table = {staged/'lpt-background.txt'}
[random]
generator = MUSIC1
seed[5] = {noise}
grafic_sign = yes
[execution]
NumThreads = 2
[output]
relative_to_config = yes
format = gadget_hdf5_double
UseLongids = yes
filename = ics_gadget.hdf5
'''
    path=directory/'fixture.conf';path.write_text(common)
    with (directory/'generation.log').open('w') as log:
        run_native_command([str(binary),str(path)],cwd=directory,receipt_path=directory/'invocation.json',
            inputs=[path,noise,staged/'transfers.txt',staged/'lpt-background.txt'],
            env=dict(os.environ,OMP_NUM_THREADS='2'),stdout=log,stderr=subprocess.STDOUT)
    # Obtain native original q from an independent zero-transfer realization,
    # never invert the nonzero particle positions. Exact IDs must align.
    zero=directory/'zero';zero.mkdir()
    transfer=np.loadtxt(staged/'transfers.txt');transfer[:,1:]=0
    comments='\n'.join(line for line in (staged/'transfers.txt').read_text().splitlines() if line.startswith('#'))
    np.savetxt(zero/'transfers.txt',transfer,header=comments.replace('# ',''))
    zero_cfg=common.replace(str(staged/'transfers.txt'),str(zero/'transfers.txt'))
    zero_path=zero/'fixture.conf';zero_path.write_text(zero_cfg)
    with (zero/'generation.log').open('w') as log:
        run_native_command([str(binary),str(zero_path)],cwd=zero,receipt_path=zero/'invocation.json',
            inputs=[zero_path,noise,zero/'transfers.txt',staged/'lpt-background.txt'],
            env=dict(os.environ,OMP_NUM_THREADS='2'),stdout=log,stderr=subprocess.STDOUT)
    ic=directory/'ics_gadget.hdf5'
    from core.evolution import rescale_monofonic_ic
    rescale_monofonic_ic(ic,10.)
    rescale_monofonic_ic(zero/'ics_gadget.hdf5',10.)
    from core.data import transactional_ic_transform
    native_ic=root/'uniform/ics_gadget.hdf5'
    def attach_original_q(paths):
        with h5py.File(paths[0],'r+') as actual,h5py.File(zero/'ics_gadget.hdf5') as reference, h5py.File(native_ic) as music:
            shift = None
            for t in (0,1):
                key='PartType'+str(t)
                if not np.array_equal(actual[key+'/ParticleIDs'][:],reference[key+'/ParticleIDs'][:]):
                    raise AssertionError('monofonIC zero realization changed particle identity/order')
                q=reference[key+'/Coordinates'][:]
                native_q=music[key+'/LagrangianCoordinates'][:]
                candidate=(native_q[0]-q[0]+5000)%10000-5000
                if shift is None:shift=candidate
                if np.max(abs((native_q-q-shift+5000)%10000-5000)) > 1e-9:
                    raise AssertionError('Uniform native q frames differ beyond a single declared lattice-origin shift')
                actual[key]['LagrangianCoordinates']=(q+shift)%10000
                actual[key+'/Coordinates'][:]=(actual[key+'/Coordinates'][:]+shift)%10000
            actual['Header'].attrs.update(Music_lagrangian_schema=1,Music_lagrangian_unitlength_cm=KPC_CM,
                                         Omega0=.3,OmegaBaryon=.05)
            actual.attrs['comparison_q_source_sha256']=file_sha256(zero/'ics_gadget.hdf5')
            actual.attrs['comparison_coordinate_shift_kpc_h']=shift
        return {'status':'PASS','scope':'native zero-realization q and independently verified lattice-origin translation',
                'zero_realization_sha256':file_sha256(zero/'ics_gadget.hdf5'),
                'music_original_q_sha256':file_sha256(native_ic),'coordinate_shift_kpc_h':shift.tolist()}
    transactional_ic_transform([ic],attach_original_q,operation='comparison-native-original-q-v1',
        configuration={'zero_realization_sha256':file_sha256(zero/'ics_gadget.hdf5'),
                       'music_original_q_sha256':file_sha256(native_ic)})
    # Low modes are insensitive to differing small-scale derivative filters.
    modes=[[1,0,0],[0,1,0],[0,0,1],[1,1,0],[1,0,1],[0,1,1]]
    comparison=compare_ic_generators(root/'uniform/ics_gadget.hdf5',ic,modes,root/'monofonic-comparison.json',
                    maximum_mode_error=.03)
    if comparison['status']!='PASS':raise AssertionError('Uniform mixed 2LPT differs from monofonIC')
    return comparison


def quadratic_and_zero_controls(root,binary,kind="uniform"):
    original=root/kind
    controls=root/(kind+"-controls");controls.mkdir()
    recipe=config(original/'fixture.conf')
    serial=controls/'serial-repeat';serial.mkdir()
    native_music_mixed2lpt_run(binary,serial,recipe,threads=1)
    doubled=controls/'quadratic-doubled';doubled.mkdir()
    for key in ('lpt2_displacement','lpt2_velocity'):
        recipe['cosmology'][key]=format(2*recipe.getfloat('cosmology',key),'.17g')
    native_music_mixed2lpt_run(binary,doubled,recipe)
    zero=controls/'zero';zero.mkdir()
    table=Path(recipe['cosmology']['transfer_file'])
    data=np.loadtxt(table);data[:,1:]=0
    comments='\n'.join(line[2:] for line in table.read_text().splitlines() if line.startswith('# '))
    np.savetxt(zero/'transfers.txt',data,header=comments)
    recipe['cosmology']['transfer_file']=str(zero/'transfers.txt')
    native_music_mixed2lpt_run(binary,zero,recipe)
    relative=controls/'linear-relative-velocity';relative.mkdir()
    relative_recipe=config(original/'fixture.conf')
    relative_table=np.loadtxt(relative_recipe['cosmology']['transfer_file'])
    relative_table[:,3]*=.9;relative_table[:,4]*=1.5
    np.savetxt(relative/'transfers.txt',relative_table,header=comments)
    relative_recipe['cosmology']['transfer_file']=str(relative/'transfers.txt')
    native_music_mixed2lpt_run(binary,relative,relative_recipe)
    measurements={}
    with h5py.File(original/'ics_gadget.hdf5') as first,h5py.File(doubled/'ics_gadget.hdf5') as twice,h5py.File(zero/'ics_gadget.hdf5') as absent,h5py.File(serial/'ics_gadget.hdf5') as repeated:
        for t in (0,1):
            key='PartType'+str(t)
            q=first[key+'/LagrangianCoordinates'][:]
            for other in (twice,absent):
                if not np.array_equal(first[key+'/ParticleIDs'][:],other[key+'/ParticleIDs'][:]):
                    raise AssertionError('Quadratic control altered exact particle identities')
            if not np.array_equal(absent[key+'/Coordinates'][:],absent[key+'/LagrangianCoordinates'][:]) or np.any(absent[key+'/Velocities'][:]):
                raise AssertionError('Zero transfer generated nonzero mixed-fluid flow')
            if np.any(absent[key+'/FirstOrderDisplacements'][:]) or np.any(absent[key+'/FirstOrderVelocities'][:]):
                raise AssertionError('Zero transfer generated nonzero retained first-order components')
            psi=[];vel=[]
            for source in (first,twice):
                psi.append((source[key+'/Coordinates'][:]-q+5000)%10000-5000-source[key+'/FirstOrderDisplacements'][:])
                vel.append(source[key+'/Velocities'][:]+source['Header'].attrs['Music_removed_bulk_velocity_gadget']-source[key+'/FirstOrderVelocities'][:])
            pn=np.linalg.norm(psi[0]);vn=np.linalg.norm(vel[0])
            if not np.isfinite([pn,vn]).all() or pn <= 1e-6 or vn <= 1e-6:raise AssertionError('No measurable second-order native signal')
            serial_psi=(repeated[key+'/Coordinates'][:]-q+5000)%10000-5000-repeated[key+'/FirstOrderDisplacements'][:]
            thread_error=float(np.linalg.norm(serial_psi-psi[0])/pn)
            pe=float(np.linalg.norm(psi[1]-2*psi[0])/pn)
            ve=float(np.linalg.norm(vel[1]-2*vel[0])/vn)
            h=first['Header'].attrs
            coefficient=(float(h['Music_2lpt_velocity_coefficient'])/float(h['Music_2lpt_displacement_coefficient'])*
                recipe.getfloat('cosmology','lpt2_aH_km_s_Mpc')/.7/1000/math.sqrt(float(h['Time'])))
            units=float(np.linalg.norm(vel[0]-coefficient*psi[0])/vn)
            if not np.isfinite([pe,ve,units,thread_error]).all() or max(pe,ve,units,thread_error)>1e-7:
                raise AssertionError(f'Native quadratic/unit/thread control failed: displacement={pe}, velocity={ve}, units={units}, threads={thread_error}')
            measurements[key]={'status':'PASS','quadratic_displacement_doubling_error':pe,
                'quadratic_velocity_doubling_error':ve,'absolute_second_order_velocity_unit_error':units,
                'zero_transfer_exact':True,'one_vs_two_thread_second_order_error':thread_error,'second_order_displacement_l2_kpc_h':float(pn)}
    with h5py.File(original/'ics_gadget.hdf5') as first,h5py.File(relative/'ics_gadget.hdf5') as shifted:
        for t,factor in ((1,.9),(0,1.5)):
            key='PartType'+str(t)
            if not np.allclose(first[key+'/Coordinates'][:],shifted[key+'/Coordinates'][:],rtol=0,atol=1e-10):
                raise AssertionError('Relative velocity changed the common density displacement')
            original_v=first[key+'/FirstOrderVelocities'][:]
            changed_v=shifted[key+'/FirstOrderVelocities'][:]
            error=float(np.linalg.norm(changed_v-factor*original_v)/np.linalg.norm(original_v))
            if not math.isfinite(error) or error>1e-10:raise AssertionError('Native linear relative species velocity failed')
            a=first[key+'/Velocities'][:]+first['Header'].attrs['Music_removed_bulk_velocity_gadget']-original_v
            b=shifted[key+'/Velocities'][:]+shifted['Header'].attrs['Music_removed_bulk_velocity_gadget']-changed_v
            denominator = np.linalg.norm(a)
            residual = np.linalg.norm(a-b)/denominator if denominator > 0 else math.inf
            if not math.isfinite(residual) or residual>1e-9:
                raise AssertionError('Relative velocity contaminated the common second-order kernel')
            measurements[key]['linear_relative_velocity_error']=error
    return {'status':'PASS','species':measurements}


def native_music_mixed2lpt_main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--binary',type=Path,required=True)
    parser.add_argument('--monofonic',type=Path,required=True);parser.add_argument('--legacy-proof',type=Path,required=True)
    args=parser.parse_args();root=args.output.resolve();root.mkdir(parents=True,exist_ok=False)
    binary=args.binary.resolve();mono=args.monofonic.resolve()
    primary=root/'primary';primary.mkdir();results=prepare_native(primary,binary)
    comparison=monofonic_control(primary,mono,primary/'staged',primary/'shared-noise.grafic')
    nonlinear=quadratic_and_zero_controls(primary,binary)
    nested_nonlinear=quadratic_and_zero_controls(primary,binary,'nested')
    continuity=boundary_control(primary,binary)
    epochs={}
    for z in (49.,199.):
        branch=root/('epoch-'+str(int(z)));branch.mkdir()
        epochs[str(int(z))]=prepare_native(branch,binary,z=z)
    ede=root/'ede-expansion';ede.mkdir()
    ede_results=prepare_native(ede,binary,model='ede',bump=.3)
    if abs(ede_results['growth']['displacement_coefficient']-results['growth']['displacement_coefficient'])<1e-4:
        raise AssertionError('EDE expansion did not change the native higher-order kernel')

    receipt={'status':'PASS','scope':'common growing mixed-fluid 2LPT with compensated density and first-order relative velocity',
        'uniform':results['uniform'],'nested':results['nested'],'phases':results['phases'],'monofonic':comparison,
        'growth':results['growth'],'refinement_continuity':continuity,'quadratic_controls':nonlinear,'nested_quadratic_controls':nested_nonlinear,'selected_epochs':epochs,
        'ede_expansion':ede_results,'binary_sha256':file_sha256(binary),'monofonic_binary_sha256':file_sha256(mono),
        'regression_driver_sha256':file_sha256(__file__),'science_acceptance':'NOT_ACCEPTED',
        'source_hashes':{str(path.resolve()):file_sha256(path) for path in [
            mono,Path(__file__),
            *[Path(__file__).resolve().parents[1]/name for name in
              ('initial_conditions.py','zoom.py','data.py','foundation.py','native.py')]]}}
    atomic_json(root/'mixed-2lpt-receipt.json',receipt)
    proof=json.loads(args.legacy_proof.read_text())
    if proof['execution']['binary_sha256']!=file_sha256(binary):raise AssertionError('Legacy proof is for another binary')
    proof['mixed_fluid_2lpt']={'status':'PASS','receipt':{'path':str(root/'mixed-2lpt-receipt.json'),
        'sha256':file_sha256(root/'mixed-2lpt-receipt.json')},'regression_driver_sha256':file_sha256(__file__)}
    atomic_json(root/'native-transfer-suite.json',proof)
    print(json.dumps(receipt,indent=2))


def boundary_control(root,binary):
    directory=root/'smooth-interface';directory.mkdir()
    recipe=config(root/'nested/fixture.conf')
    table=Path(recipe['cosmology']['transfer_file'])
    rows=np.loadtxt(table)
    rows[:,1:]*=np.exp(-(rows[:,0]/.4)**4)[:,None]
    comments='\n'.join(line[2:] for line in table.read_text().splitlines() if line.startswith('# '))
    transfer=directory/'transfers.txt';np.savetxt(transfer,rows,header=comments)
    recipe['cosmology']['transfer_file']=str(transfer)
    native_music_mixed2lpt_run(binary,directory,recipe)
    ic=directory/'ics_gadget.hdf5'
    with h5py.File(ic) as source:
        box=float(source['Header'].attrs['BoxSize'])
        spacing=box/2**int(source['Header'].attrs['Music_levelmax'])
        qfine=source['PartType1/LagrangianCoordinates'][:]
        qcoarse=source['PartType2/LagrangianCoordinates'][:]
        lo=qfine.min(0)-spacing/2;hi=qfine.max(0)+spacing/2
        points=[]
        for axis in range(3):
            other=[j for j in range(3) if j!=axis]
            for end in (lo[axis],hi[axis]):
                for u in (.25,.5,.75):
                    for v in (.25,.5,.75):
                        p=(lo+hi)/2;p[axis]=end
                        p[other]=lo[other]+np.array([u,v])*(hi-lo)[other]
                        points.append(p)
        points=np.asarray(points)
        def reconstruct(q,values):
            if len(q) < 48:
                raise ValueError('Boundary reconstruction requires all 48 requested neighbors')
            # Fit a quadratic in local cell-spacing units. Squared weights
            # enter least squares because both the design and values are scaled.
            _,neighbors=cKDTree(q).query(points,k=48)
            output=[]
            for p,ids in zip(points,neighbors):
                dx=(q[ids]-p)/(2*spacing)
                x,y,z=dx.T
                design=np.column_stack((np.ones(len(ids)),x,y,z,x*x,y*y,z*z,x*y,x*z,y*z))
                weight=1/(1+np.sum(dx*dx,axis=1))
                fit,_,rank,_=np.linalg.lstsq(design*weight[:,None],values[ids]*weight[:,None],rcond=None)
                if rank != 10:raise AssertionError('Boundary reconstruction is underdetermined')
                output.append(fit[0])
            return np.asarray(output)
        fields={}
        for name in ('FirstOrderDisplacements','total_displacement','total_velocity'):
            def read(t):
                group=source['PartType'+str(t)]
                if name=='total_velocity':
                    return (group['Velocities'][:]+source['Header'].attrs['Music_removed_bulk_velocity_gadget'])*np.sqrt(float(source['Header'].attrs['Time']))
                if name=='total_displacement':
                    return (group['Coordinates'][:]-group['LagrangianCoordinates'][:]+box/2)%box-box/2
                return group[name][:]
            coarse_values,fine_values=read(2),read(1)
            coarse,fine=reconstruct(qcoarse,coarse_values),reconstruct(qfine,fine_values)
            scale=float(np.sqrt(np.mean(np.sum(coarse_values**2,axis=1))))
            if scale <= 1e-12:raise AssertionError('Smooth boundary control has no measurable signal')
            fields[name]={'coarse':coarse,'fine':fine,'scale':scale}
    arrays={'interface_q':points};specs={'interface_q':{'units':'comoving kpc/h','quantity':'declared shared interface samples'}}
    checks={}
    for name,value in fields.items():
        checks[name]=validate_refinement_continuity(points,value['coarse'],points,value['fine'],
            box=box,tolerance=.1*value['scale'])
        checks[name]['coarse_vector_rms']=value['scale']
        for side in ('coarse','fine'):
            key=name+'_'+side;arrays[key]=value[side]
            specs[key]={'units':'km s^-1' if name == 'total_velocity' else 'comoving kpc/h',
                'quantity':'local quadratic reconstruction of actual native flow'}
    report={'status':'PASS' if all(v['status']=='PASS' for v in checks.values()) else 'FAIL',
        'checks':checks,'scope':__doc__,'science_acceptance':'NOT_ACCEPTED',
        'criteria':{'maximum_jump_fraction_of_coarse_vector_rms':.1,'neighbors':48,'polynomial_degree':2,
                    'samples':len(points),'transfer_smoothing_k_h_Mpc':.4}}
    sources={str(p.resolve()):file_sha256(p) for p in (ic,transfer,Path(__file__))}
    report['sources']=sources
    result=export_numeric_audit(directory/'continuity.json',report,arrays,specs,algorithm=__file__,
        source_hashes=sources,context={'definition':'Native smooth-limit refinement interface continuity'})
    if result['status']!='PASS':raise AssertionError('Native smooth interface continuity failed')
    return result

