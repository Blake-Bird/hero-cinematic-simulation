"""Fixed native-package calculations for the optional lab, never installers.

CLASS/AxiCLASS use the production reusable-bundle driver. Blender uses the
production material, geometry-node and camera helpers on 32 synthetic points.
monofonIC runs fresh CLASS transfers and the production 128-particle IC route.
Grackle compiles a tiny 32-cell control against its installed shared library.
MUSIC runs uniform, nested and periodic normalization/rejection controls.
Rockstar and VELOCIraptor inspect explicit 32-DM-particle snapshot fixtures.
Consistent Trees uses 128-object catalogues; DisPerSE uses 64 DM samples.
OpenGadget3 requires an explicit existing Linux reference build.
Every output stays beneath the requested new external directory; no movies.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import traceback

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CHECKS = ('class', 'axiclass', 'blender', 'monofonic', 'grackle', 'music2', 'rockstar', 'velociraptor', 'consistent-trees', 'disperse', 'opengadget3')


def calculate(name, output, seed):
    """Execute real bounded native work and return its measured scope."""
    import numpy as np
    from core.foundation import file_sha256
    output = Path(output).resolve()
    if name == 'opengadget3':
        from optional_tests.native_sidm import calculate as independent_control
        return independent_control(output, seed)
    if name in {'consistent-trees', 'disperse'}:
        from optional_tests.native_structure import trees, disperse
        return (trees if name == 'consistent-trees' else disperse)(output, seed)
    if name in {'rockstar', 'velociraptor'}:
        from optional_tests.native_finders import calculate as finder_control
        return finder_control(name, output, seed)
    if name == 'music2':
        from optional_tests.native_music import calculate as music_control
        return music_control(output)
    if name == 'grackle':
        from optional_tests.native_grackle import calculate as cooling_control
        return cooling_control(output)
    if name == 'monofonic':
        from core.native import native_engine_directory, run_native_command
        directory = native_engine_directory('monofonic')
        candidates = [directory/'build/monofonIC', *sorted(directory.glob('build*/monofonIC'))]
        if not any(path.is_file() for path in candidates):
            raise ImportError('Build monofonIC with scripts/setup.sh --engines monofonic; CLASS and MPI are also required.')
        if not (native_engine_directory('class')/'class').is_file():
            raise ImportError('Build CLASS for the fresh monofonIC transfer calculation.')
        helper = Path(__file__).with_name('native_monofonic.py')
        with (output/'generation.log').open('w') as log:
            run_native_command([sys.executable, '-B', str(helper), '--output', str(output), '--seed', str(seed)],
                cwd=output, inputs=[helper, ROOT/'core/config.yaml', ROOT/'core/evolution.py', ROOT/'core/cosmology.py'],
                receipt_path=output/'generation-invocation.json', timeout=150,
                stdout=log, stderr=subprocess.STDOUT)
        return json.loads((output/'inspection.json').read_text())
    if name in {'class', 'axiclass'}:
        from core.native import native_engine_directory
        from core.linear import run, verify_spectrum_directory
        binary = native_engine_directory(name) / 'class'
        if not binary.is_file():
            raise ImportError('Build the selected native engine with scripts/setup.sh --engines ' + name)
        h = .674
        config = dict(name='lab-' + name, model='ede' if name == 'axiclass' else 'lcdm',
            H0=67.4, Omega_b=.0224/h**2, Omega_cdm=.12/h**2, A_s=2.1e-9,
            n_s=.965, tau_reio=.054, z_start=49., k_max_h_mpc=2.)
        if name == 'axiclass':
            config.update(f_ede=.05, log10_zc=float(np.log10(3500.)), theta_i=2.8, n_axion=3.)
        (output/'request.json').write_text(json.dumps(config, indent=2))
        bundle = run(config, output, binary)
        report = verify_spectrum_directory(bundle)
        start = np.loadtxt(bundle/'pk_class.txt')
        today = np.loadtxt(bundle/'pk_class_z0.txt')
        assert start.shape[1] == today.shape[1] == 2
        assert np.isfinite(start).all() and np.isfinite(today).all()
        assert (start > 0).all() and (today > 0).all()
        arrays = {'start_k_h_mpc': start[:, 0], 'start_power_mpc_h_cubed': start[:, 1],
                  'today_k_h_mpc': today[:, 0], 'today_power_mpc_h_cubed': today[:, 1]}
        np.savez_compressed(output/'arrays.npz', **arrays)
        return dict(backend=name, epochs=[49., 0.], modes=[len(start), len(today)],
            units=report['units'], binary_sha256=file_sha256(binary),
            bundle_manifest=str((bundle/'spectrum.json').relative_to(output)),
            integrity=report['integrity_status'],
            implementation='core/linear.py:run → native solver → export_power/export_background → plot_power',
            scope='Two actual linear Boltzmann executions and sealed numerical/figure exports. These solvers do not evolve particles; this does not establish nonlinear cosmological convergence.')
    if name == 'blender':
        from core.visualization import _find_blender_executable
        from core.native import run_native_command
        binary = _find_blender_executable()
        if binary is None:
            raise ImportError('Install Blender separately and expose its executable on PATH.')
        points = np.clip(.5 + np.random.default_rng(seed).normal(size=(32, 3))*.14, .08, .92)
        request = output/'points.json'
        request.write_text(json.dumps(dict(origin='synthetic_visual_control',
            units='display units, not physical distances', points=points.tolist()), indent=2))
        helper = Path(__file__).with_name('native_blender_scene.py')
        for mode in ('build', 'verify'):
            with (output/(mode+'.log')).open('w') as log:
                run_native_command([binary, '--background', '--factory-startup', '--threads', '2', '--python-exit-code', '1',
                    '--python', str(helper), '--', mode, str(request)], cwd=output,
                    inputs=[helper, request, ROOT/'core/blender.py'],
                    receipt_path=output/(mode+'-invocation.json'), timeout=90,
                    stdout=log, stderr=subprocess.STDOUT)
        verified = json.loads((output/'verified-scene.json').read_text())
        assert verified['particles'] == 32 and verified['positions_match']
        from PIL import Image
        with Image.open(output/'preview.png') as image:
            image.load();pixels=np.asarray(image.convert('RGB'))
            assert image.size == (256, 192)
            assert np.std(pixels.reshape(-1, 3), axis=0).max() > .5
        np.savez_compressed(output/'arrays.npz', position_display_units=points)
        return dict(backend=name, particles=32, image_pixels=[256, 192], samples=4,
            binary_sha256=file_sha256(binary), fresh_load_verification=verified,
            implementation='core/blender.py:_make_material/_make_geometry_nodes/_add_lighting_and_camera → Blender save/render → independent fresh Blender load',
            scope='Actual scene construction, still rendering and fresh-process verification of 32 synthetic points. Glyph radius and light are display settings; no stellar formation, luminosity calibration or native particle evolution is claimed.')
    raise ValueError('Unknown fixed native check')


def main():
    """Save a finite worker receipt; stop owned native work on timeout/cancel."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('backend', choices=CHECKS)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, required=True)
    args=parser.parse_args();args.output.mkdir(parents=True, exist_ok=False)
    def timeout(signum, frame):
        raise TimeoutError('Native package check exceeded 180-second budget')
    def cancelled(signum, frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, cancelled)
    signal.signal(signal.SIGALRM, timeout);signal.alarm(180)
    try:
        record=dict(status='PASS', backend=args.backend, result=calculate(args.backend,args.output,args.seed))
    except ImportError as error:
        record=dict(status='UNAVAILABLE', backend=args.backend, reason=str(error))
    except Exception as error:
        record=dict(status='FAIL', backend=args.backend, reason=str(error), traceback=traceback.format_exc())
    finally:
        signal.alarm(0)
    record['worker_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (args.output/'receipt.json').write_text(json.dumps(record, indent=2, allow_nan=False))
    print(json.dumps({k:v for k,v in record.items() if k!='result'}), flush=True)
    return int(record['status']=='FAIL')


if __name__=='__main__':
    raise SystemExit(main())
