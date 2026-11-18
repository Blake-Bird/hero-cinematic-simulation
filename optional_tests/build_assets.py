"""Build the optional notebook, portable source inventory and archived figure atlas.

The atlas imports actual sealed figure outputs from a retained campaign. It
copies no movies, native binaries, solver builds or raw research snapshots.
Missing recipes remain missing. Run without --campaign to rebuild the notebook
and inventory using already-bundled examples.
"""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from optional_tests.catalogue import HERE,EQUATIONS,sha,build_catalogue,source_inventory


def write(path,data):
    """Write finite portable JSON with stable readable formatting."""
    Path(path).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')


def import_campaign(evidence):
    """Copy one verified recorded plot per recipe; keep masks, arrays and original scope."""
    evidence=Path(evidence).resolve();coverage=json.loads((evidence/'graph-coverage.json').read_text())
    output=HERE/'reference_gallery';output.mkdir(exist_ok=True);figures=[]
    for row in coverage['entries']:
        candidates=[]
        for binding in row.get('figure_support',[]):
            path=Path(binding['manifest'])
            if path.is_file():candidates.append((binding['supported_series_entries'],path.stat().st_mtime_ns,path))
        if not candidates:continue
        support,_,path=max(candidates)
        manifest=json.loads(path.read_text());stem=row['family']+'__'+row['variant']
        # Manifests live below gallery/products/scientific_figures/<id>/.
        gallery=next(parent for parent in path.parents if (parent/'gallery.json').is_file())
        entries=json.loads((gallery/'gallery.json').read_text())['figures']
        entry=next(f for f in entries if (gallery/f['manifest']).resolve()==path.resolve())
        image=(gallery/entry['png']).resolve();arrays=(gallery/entry['physical_arrays']).resolve()
        if not image.is_file() or not arrays.is_file() or sha(arrays)!=entry['physical_arrays_sha256']:
            raise ValueError('Incomplete or changed original figure '+str(path))
        if max(image.stat().st_size,arrays.stat().st_size)>32*1024**2:
            raise ValueError('Choose an explicitly budgeted smaller reference, not a truncated payload')
        image_target=output/(stem+'.png');array_target=output/(stem+'.npz');contract_target=output/(stem+'.json')
        shutil.copy2(image,image_target);shutil.copy2(arrays,array_target);shutil.copy2(path,contract_target)
        figures.append({'family':row['family'],'variant':row['variant'],'image':image_target.name,
          'arrays':array_target.name,'manifest':contract_target.name,'supported_entries':support,
          'status':row['full_scientific_recipe_execution'],'original_manifest_sha256':sha(path),
          'image_sha256':sha(image_target),'arrays_sha256':sha(array_target),
          'axes':manifest.get('axes',[]),
          'selections':manifest.get('scientific_metadata',{}).get('selections',[]),
          'scope':'Archived original campaign output. Synthetic controls and unsupported samples retain their original contracts. This is not a fresh run on the recipient machine.',
          'original_manifest':str(path),'science_status':manifest.get('science_status',entry.get('science_status','NOT_ACCEPTED'))})
    write(output/'catalogue.json',{'schema':'research2-archived-atlas-v1','figures':figures,
      'source_campaign':'retained smoke campaign; see docs/smoke-test.md','movies_included':False,
      'scope':'Read-only examples, not accepted science or portable native producer inputs.'})
    summaries=[]
    for path in sorted(evidence.glob('*summary.json')):
        raw=path.read_text().replace(str(ROOT.parent/'research2-runtime'),'<RUNTIME_ROOT>').replace(str(ROOT),'<SOURCE_ROOT>')
        summaries.append({'title':path.stem.replace('-',' '),'scope':'Archived original execution summary; inspect component status, scope and missing inputs.',
                          'summary':json.loads(raw),'original_summary_sha256':sha(path)})
    write(output/'software.json',summaries)
    # Prefer a completed merged suite over an older refresh or a single-suite
    # trace. Its summary supplies the exact implementation revisions tested.
    merged = []
    for candidate in evidence.glob('python-execution-*.json'):
        suffix = candidate.stem.removeprefix('python-execution-')
        summary = evidence / ('retained-tests-' + suffix + '-summary.json')
        if not summary.is_file():
            continue
        record = json.loads(summary.read_text())
        if (record.get('status') in {'PASS', 'PASS_WITH_SKIPS'}
                and record.get('suites') and isinstance(record.get('source_sha256'), dict)):
            merged.append((summary.stat().st_mtime_ns, candidate, summary, record['source_sha256']))
    summary_path = None
    if merged:
        _, trace, summary_path, revisions = max(merged)
    else:
        trace = evidence / 'python-execution-refresh.json'
        if not trace.is_file():
            trace = evidence / 'python-execution.json'
        revisions_path = evidence / 'regression-refresh-source-hashes.json'
        revisions = json.loads(revisions_path.read_text()) if trace.name == 'python-execution-refresh.json' else {}
    if trace.is_file():
        observed = json.loads(trace.read_text())['files']
        source_paths = {entry['path'] for entry in source_inventory()}
        write(output / 'regression-calls.json', {
            'schema': 'research2-archived-regression-calls-v1',
            'original_sha256': sha(trace),
            'execution_summary_sha256': sha(summary_path) if summary_path else None,
            'scope': 'Recorded regression calls. Fixtures and mocks may apply. Where available, execution_source_sha256 identifies the source revision recorded for this run. Call counts include failing tests.',
            'files': {path: {**observed[path], 'execution_source_sha256': revisions.get(path)}
                      for path in sorted(source_paths & observed.keys())},
        })
    return len(figures)


def markdown(text):return {'cell_type':'markdown','metadata':{},'source':text.splitlines(keepends=True)}
def code(text):return {'cell_type':'code','metadata':{},'execution_count':None,'outputs':[],'source':text.splitlines(keepends=True)}


def build_notebook():
    """Author an executable teaching notebook, including an honest entry for all 102 recipes."""
    catalogue=build_catalogue()
    cells=[markdown('''# Research2: from particles to a figure

A local, optional teaching notebook. **Synthetic controls are explicit.** A plot is
not proof of a resolved galaxy, a converged estimator or an accepted posterior.
This notebook explains the quick lab, reads its actual saved arrays and exposes
all registered scientific recipes with their original input contracts.

Start the dashboard with `python optional_tests/lab.py serve`, run the 32-particle
lab, and then run these cells. No cell downloads data or creates a video.
'''),code('''from pathlib import Path
import os, sys, json
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, Markdown, Image
base = Path.cwd().resolve()
ROOT = next(p for p in [base, *base.parents] if (p / 'configs/research/graph_producers.json').is_file())
sys.path.insert(0, str(ROOT))
from optional_tests.catalogue import build_catalogue
catalogue = build_catalogue()
print(catalogue['counts'])
'''),markdown('''## Select a fresh lab run

Set `RESEARCH2_LAB_RUN` to an explicit results directory, or let the cell select
the newest `run.json` below the optional-lab runtime root. A missing run leaves
an actionable message; archived gallery examples can still be inspected below.
'''),code('''specified = os.environ.get('RESEARCH2_LAB_RUN')
choices = list((ROOT.parent / 'research2-runtime/optional-lab').rglob('run.json')) if not specified else []
RUN = Path(specified).resolve() if specified else (max(choices, key=lambda p: p.stat().st_mtime).parent if choices else None)
if RUN is None:
    print('No fresh lab output. Run the dashboard, or call run_lab below with a NEW directory.')
else:
    receipt = json.loads((RUN / 'run.json').read_text())
    print(RUN, receipt['status'], receipt['particles'], 'particles')
    for step in receipt['steps']:
        print(step['id'], step['status'], step.get('seconds'), 'seconds')
'''),markdown('''To create a new run directly, use a fresh destination:

```python
from optional_tests.lab import run_lab
run_lab(ROOT.parent / 'research2-runtime/optional-lab/my-new-32', particles=32)
```

`32` is the **total** count in this lab. In cosmological IC configuration,
`particles_per_dimension=32` instead means $32^3$ particles per generated species.
''')]
    cells.extend([markdown('''## Which Python files actually ran?

Fresh runs include a main-thread call record. It separates function calls from
module imports and class definitions, and binds the source hashes captured at
the start of the measurements. These counts are not full branch coverage.
Native code and the isolated library workers have separate receipts. The next
cell checks the recorded trace against the run receipt, then lists the actual
observed functions and whether their captured source still matches this checkout.
'''),code('''trace_path = RUN / 'python-calls.json' if RUN is not None else None
if trace_path is None or not trace_path.is_file():
    print('This run predates per-file observation. Run a fresh lab profile for observed calls.')
else:
    from hashlib import sha256
    recorded = {row['path']: row for row in receipt['artifacts']}
    assert sha256(trace_path.read_bytes()).hexdigest() == recorded['python-calls.json']['sha256']
    observed = json.loads(trace_path.read_text())
    assert observed == receipt['python_execution']
    print(observed['scope'])
    print(observed['files_with_calls'], 'files with calls;', observed['call_events'], 'function call events')
    for name, entry in sorted(observed['files'].items()):
        path = (ROOT / name).resolve()
        matches = path.is_relative_to(ROOT) and path.is_file() and sha256(path.read_bytes()).hexdigest() == entry['source_sha256']
        print(name, '| calls:', sum(entry['calls'].values()), '| imports:', entry['module_events'],
              '| class definitions:', entry['definition_events'], '| current source matches:', matches)
        for function, count in sorted(entry['calls'].items()):
            print('  ', count, function)
''')])
    for key,e in EQUATIONS.items():
        cells.append(markdown('## '+e['title']+'\n\n`'+e['formula']+'`\n\n**Units.** '+e['units']+'\n\n**Reading the result.** '+e['read']+'\n\nImplementation: `'+e['implementation']+'`, `'+e['function']+'`.'))
        if key=='density':
            cells.extend([markdown(r'''For shell $j$ with edges $r_j,r_{j+1}$,
$$V_j=\frac{4\pi}{3}(r_{j+1}^3-r_j^3),\qquad
\rho_j=\frac{\sum_{i\in j}m_i}{V_j}.$$
The plotted radius is the arithmetic midpoint; it is not a new measurement
of a pointwise density. The surface map divides pixel mass by pixel area.
The mask used on the logarithmic density axis is saved alongside the counts.
'''),code('''if RUN is not None:
    with np.load(RUN / 'density_arrays.npz') as data:
        rho = data['shell_mass_msun'] / data['shell_volume_kpc3']
        np.testing.assert_allclose(rho, data['density_msun_kpc3'])
        print('Measured total mass [Msun]:', data['shell_mass_msun'].sum())
        print('Shell counts:', data['shell_count'])
        support = data['supported']
        plt.loglog(data['radius_proper_kpc'][support], rho[support], 'o-')
        plt.xlabel('Shell midpoint [proper kpc]'); plt.ylabel('Density [Msun / proper kpc³]')
        plt.show()
    display(Image(filename=str(RUN / 'density.png')))
''')])
        elif key=='power':cells.append(code('''if RUN is not None:
    with np.load(RUN / 'power_arrays.npz') as data:
        plt.loglog(data['k_inverse_proper_kpc'], data['power_proper_kpc3'], 'o-')
        plt.xlabel('k [1 / proper kpc]'); plt.ylabel('P(k) [proper kpc³]'); plt.show()
    # Read the implementation to see rFFT multiplicity and normalization.
    from core.analysis import fft_power_spectrum
    import inspect
    print(inspect.getsource(fft_power_spectrum))
'''))
        elif key=='orbit':cells.append(code('''if RUN is not None:
    with np.load(RUN / 'orbit_arrays.npz') as data:
        E = data['specific_energy_kms2']
        drift = np.max(np.abs(E-E[0]) / np.maximum(np.abs(E[0]), 1.0))
        print('Max relative finite-tracer energy drift:', drift)
        print('Actual saved Myr:', data['times_myr'])
    display(Image(filename=str(RUN / 'orbits.png')))
'''))
        elif key=='covariance':cells.append(code('''if RUN is not None:
    with np.load(RUN / 'velocity_arrays.npz') as data:
        print('Mean [km/s]:', data['mean_velocity_kms'])
        C = data['population_covariance_kms2']
        print('Covariance [(km/s)²]:', C)
        print('Eigenvalues:', np.linalg.eigvalsh(C))
'''))
        elif key=='units':cells.append(code('''if RUN is not None:
    import h5py
    with h5py.File(RUN / 'synthetic_snapshot.hdf5') as f:
        print('Origin:', f.attrs['source_origin'])
        print('Clock:', f['Parameters'].attrs['ComovingIntegrationOn'], '(0 = isolated)')
        for name in ['PartType0','PartType1','PartType4','PartType5']:
            print(name, len(f[name]['ParticleIDs']), dict(f[name]['Coordinates'].attrs))
'''))
    cells.extend([markdown('''## A viewing zoom is not IC refinement

The lab measures nested 100, 20 and 5 proper-kpc projected views of the same
synthetic HDF5 particles. Pixel areas and selected mass are recalculated at
each view. No particles are added. MUSIC refinement, contamination controls
and native zoom evolution are separate workflows in the main repository.
'''),code('''if RUN is not None and (RUN / 'zoom_arrays.npz').exists():
    with np.load(RUN / 'zoom_arrays.npz') as data:
        for i in range(3):
            selected = data[f'view_{i}_selected']
            edges = data[f'view_{i}_x_edges_proper_kpc']
            pixel_area = (edges[1] - edges[0])**2
            mass = data[f'view_{i}_surface_density_msun_kpc2'].sum() * pixel_area
            print(i, 'selected particles:', selected.sum(), 'projected mass [Msun]:', mass)
    display(Image(filename=str(RUN / 'zoom.png')))
''')])
    if (HERE/'reference_gallery/native_zoom.json').exists():
        cells.append(markdown('''## Native MUSIC refinement

MUSIC initialized 2,744 refined gas particles, 2,744 refined DM particles and
5,265 coarse boundary particles. GADGET evolved the load and saved the epoch
shown below. Each panel crops the same projected coordinates in comoving
Mpc/h. To convert a plotted length to proper Mpc, multiply by the saved
scale factor and divide by h. The zero-transfer control exposes the
refinement geometry and tests the handoff between programs.

![Native zoom geometry](native_zoom.png)

The plotted coordinates are in `native_zoom.npz`. The snapshot hash, species
counts and view limits are in `native_zoom.json`. See
`optional_tests/zoom_reference.py` for the rendering code.
'''))
    cells.extend([markdown('''## Inspect fresh optional-library calculations

Run the `libraries` profile to produce these files locally. Each backend has
its own process log, status, numerical assertions, units and scope. The cell
verifies the published file hashes before reading arrays. A chain or trained
network is an execution artifact; its existence does not establish calibration.
The FSPS spectrum belongs to a separately calculated 1-Gyr, 1-Msun-formed SSP,
not the stellar marker particles in the synthetic HDF5 fixture.
'''),code('''if RUN is not None and (RUN / 'libraries').is_dir():
    import hashlib
    published = {a['path']: a for a in receipt['artifacts']}
    for path in sorted((RUN / 'libraries').glob('*/receipt.json')):
        key = path.relative_to(RUN).as_posix()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == published[key]['sha256']
        backend = json.loads(path.read_text())
        print(path.parent.name, backend['status'])
        print(backend.get('result', {}).get('scope', backend.get('reason', '')))
        arrays = path.parent / 'arrays.npz'
        if arrays.is_file():
            key = arrays.relative_to(RUN).as_posix()
            assert hashlib.sha256(arrays.read_bytes()).hexdigest() == published[key]['sha256']
            with np.load(arrays, allow_pickle=False) as values:
                print({name: (values[name].shape, str(values[name].dtype)) for name in values.files})
    spectrum = RUN / 'libraries/fsps/spectrum.png'
    if spectrum.is_file():
        key = spectrum.relative_to(RUN).as_posix()
        assert hashlib.sha256(spectrum.read_bytes()).hexdigest() == published[key]['sha256']
        display(Image(filename=str(spectrum)))
else:
    print('Select a completed libraries profile run to inspect its fresh backend outputs.')
''')])
    cells.extend([markdown(r'''## Inspect native solvers, scenes, ICs, cooling and catalogues

The `engines` profile adds actual CLASS/AxiCLASS calculations, a Blender
still scene, a fresh CLASS-to-monofonIC IC control and a 32-cell Grackle
atomic cooling control. MUSIC adds ten uniform/nested/periodic analytic-transfer
realizations and seven expected rejections, with 10,753 particles in the nested
reference. These are normalization controls, not a target-halo zoom. The profile
requires their installed native software. Rockstar and VELOCIraptor use separate 32-DM-particle synthetic
fixtures with the same seed. Their two saved epochs are repeated samples, not
an evolution. The catalogue counts may be zero; different settings and mass
definitions do not establish finder agreement or halo accuracy.
The next cell verifies worker receipts and arrays against the run artifact hashes, checks each
sealed Boltzmann bundle, and replays the dimensionless handoff:
$$\Delta^2(k,z)=\frac{k^3P_{\rm CLASS}(k,z)}{2\pi^2}.$$
Here $k$ is in $h/{\rm Mpc}$ and $P$ is in $({\rm Mpc}/h)^3$. These are linear
solver outputs, not measured particle mesh powers. The particle handoff stores
the logarithms at z=49. The Blender picture uses 32 arbitrary display points
and enlarged display glyphs. Its separate native verification reopens the
saved `.blend`; reading this cell does not launch another renderer or a movie.
The monofonIC control uses a 4³ grid per species (128 total), distinct from the
32-particle quick fixture. Its original HDF5 positions are comoving kpc/h;
stored GADGET velocities become peculiar velocities after multiplying by sqrt(a).
The cell rechecks IDs and the cosmological gas mass fraction from the arrays.
Grackle evolves 32 independent constant-density cells, not particles. Its
atomic H/He network has no UV, metals, dust or molecules. Species fractions
are mass divided by total gas mass, so HI+HII=0.76 and HeI+HeII+HeIII=0.24.
The signed instantaneous cooling time is u/(du/dt), negative for net cooling;
its magnitude is a timescale. Temperatures are native calculations in kelvin.
'''),code('''engine_root = RUN / 'engines' if RUN is not None else None
if engine_root is None or not engine_root.is_dir():
    print('Select a completed engines-profile run to inspect native package outputs.')
else:
    from hashlib import sha256
    from core.linear import verify_spectrum_directory
    published = {row['path']: row for row in receipt['artifacts']}
    def check_published(path):
        key = path.relative_to(RUN).as_posix()
        assert sha256(path.read_bytes()).hexdigest() == published[key]['sha256'], path
    for backend in ('class', 'axiclass', 'blender', 'monofonic', 'grackle', 'music2', 'rockstar', 'velociraptor'):
        directory = engine_root / backend
        result_path = directory / 'receipt.json'
        if not result_path.is_file():
            print(backend, 'has no receipt in this run')
            continue
        check_published(result_path)
        worker = json.loads(result_path.read_text())
        print(backend, worker['status'])
        if worker['status'] != 'PASS':
            print(worker.get('reason'))
            continue
        print(worker['result']['implementation'])
        print(worker['result']['scope'])
        check_published(directory / 'arrays.npz')
        if backend in ('class', 'axiclass'):
            manifest = directory / worker['result']['bundle_manifest']
            check_published(manifest)
            bundle = verify_spectrum_directory(manifest.parent)
            handoff = np.loadtxt(manifest.parent / 'pk_gadget.txt')
            with np.load(directory / 'arrays.npz', allow_pickle=False) as data:
                k, power = data['start_k_h_mpc'], data['start_power_mpc_h_cubed']
                delta_squared = k**3 * power / (2 * np.pi**2)
                np.testing.assert_allclose(handoff[:, 0], np.log10(k), rtol=0, atol=1e-9)
                np.testing.assert_allclose(handoff[:, 1], np.log10(delta_squared), rtol=0, atol=1e-9)
                print('Original start modes:', len(k), '| units:', bundle['units'])
            figure = manifest.parent / 'matter_power.png'
        elif backend == 'monofonic':
            check_published(directory / 'inspection.json')
            check_published(directory / worker['result']['hdf5'])
            with np.load(directory / 'arrays.npz', allow_pickle=False) as data:
                ids = np.concatenate([data['ids_0'], data['ids_1']])
                assert len(ids) == len(np.unique(ids)) == 128
                for species in (0, 1):
                    assert data[f'position_{species}_kpc_h'].shape == (64, 3)
                gas = data['mass_0_1e10_msun_h'].sum()
                dm = data['mass_1_1e10_msun_h'].sum()
                np.testing.assert_allclose(gas / (gas + dm), .0224 / (.0224 + .12), rtol=0, atol=1e-6)
                print('128 IC particles; original gas mass fraction:', gas / (gas + dm))
            figure = directory / 'positions.png'
        elif backend == 'grackle':
            check_published(directory / 'cooling.csv')
            with np.load(directory / 'arrays.npz', allow_pickle=False) as data:
                assert data['temperature_K'].shape == (2, 32)
                np.testing.assert_allclose(data['HI_mass_fraction'] + data['HII_mass_fraction'], .76, rtol=0, atol=1e-10)
                helium = sum(data[key] for key in ('HeI_mass_fraction', 'HeII_mass_fraction', 'HeIII_mass_fraction'))
                np.testing.assert_allclose(helium, .24, rtol=0, atol=1e-10)
                assert (data['cooling_time_s'] < 0).all()
                assert (data['specific_energy_erg_g'][1] < data['specific_energy_erg_g'][0]).all()
                print('Native 32-cell atomic cooling; temperatures [K]:', data['temperature_K'].min(), data['temperature_K'].max())
            figure = directory / 'cooling.png'
        elif backend == 'music2':
            report_path = directory / worker['result']['report']
            check_published(report_path)
            music = json.loads(report_path.read_text())
            assert music['status'] == music['nested']['status'] == music['periodic']['status'] == 'PASS'
            assert len(music['execution']['rejections']) == 7
            assert all(row['exit_code'] != 0 for row in music['execution']['rejections'].values())
            for family in ('uniform', 'nested', 'periodic'):
                for case, inputs in music[family]['inputs'].items():
                    for filename, expected in inputs.items():
                        original = directory / 'controls' / case / filename
                        check_published(original)
                        assert sha256(original.read_bytes()).hexdigest() == expected
            import h5py
            with h5py.File(directory / 'controls/dms-reference/zoom_fixture.hdf5') as native, np.load(directory / 'arrays.npz', allow_pickle=False) as data:
                count = 0
                for species in (0, 1, 2):
                    np.testing.assert_array_equal(native[f'PartType{species}/Coordinates'][:], data[f'Coordinates_{species}'])
                    count += len(data[f'ParticleIDs_{species}'])
                assert count == worker['result']['reference_particles'] == 10753
                print('Nested reference particles:', count, '| velocity/displacement residuals:', worker['result']['velocity_displacement_relative_residuals'])
            figure = directory / 'nested.png'
        elif backend in ('rockstar', 'velociraptor'):
            check_published(directory / 'catalogue.json')
            check_published(directory / worker['result']['native_report'])
            with np.load(directory / 'arrays.npz', allow_pickle=False) as data:
                assert data['input_position_comoving_kpc_h'].shape == (32, 3)
                assert len(np.unique(data['input_ids'])) == 32
                np.testing.assert_array_equal(data['object_count'], worker['result']['object_counts'])
                catalogue_output = json.loads((directory / 'catalogue.json').read_text())
                assert len(catalogue_output['objects']) == len(data['object_mass_msun']) == data['object_count'][-1]
                print('Native object counts:', data['object_count'], '| definition:', worker['result']['mass_definition'])
            if backend == 'rockstar':
                for snapshot, count in zip(worker['result']['catalogue_snapshots'], worker['result']['object_counts']):
                    path = directory / f'catalogue-{snapshot:03}.json'
                    check_published(path)
                    assert len(json.loads(path.read_text())['objects']) == count
            else:
                report = json.loads((directory / 'velociraptor/report.json').read_text())
                check_published(directory / 'velociraptor/report.sha256')
                assert sha256((directory / 'velociraptor/report.json').read_bytes()).hexdigest() == (directory / 'velociraptor/report.sha256').read_text().strip()
                for name, expected in report['artifacts'].items():
                    path = directory / 'velociraptor' / name
                    check_published(path)
                    assert sha256(path.read_bytes()).hexdigest() == expected
            figure = directory / 'catalogue.png'
        else:
            check_published(directory / 'particles.blend')
            check_published(directory / 'verified-scene.json')
            print('Fresh Blender load:', json.loads((directory / 'verified-scene.json').read_text()))
            figure = directory / 'preview.png'
        check_published(figure)
        display(Image(filename=str(figure)))
''')])
    cells.extend([markdown(r'''## Native tree links and filament topology

The tree control supplies **128 synthetic catalogue objects per epoch**, at
a=0.99 and a=1. It creates no particles; the input Np=1000 is a declared toy
catalogue property. The production Consistent Trees route writes actual
hlists. Their integer IDs, descendant IDs and descendant scale factors define
the links. The next cell reads these original tables, compares the saved
arrays and checks that every link advances to its recorded later epoch.
Positions remain comoving Mpc/h and masses Msun/h. This is not a physically
evolved merger history or an accuracy benchmark.

DisPerSE uses **64 DM samples**, its production minimum, on a 16³ periodic
CIC mass mesh. With one-cell Gaussian density smoothing it processes
F=log10(1 + smoothed rho / mean rho), a dimensionless scalar. The density
smoothing scale is 0.1/16=0.00625 comoving Mpc/h. Native mse finds the ascending
persistence skeleton at cut=0.1; skelconv exports topology and display geometry.
One display-smoothing iteration does not replace the separately retained
unsmoothed topology. The XY picture is **boolean skeleton occupancy**, not
projected density or filament thickness. The cell reconstructs that occupancy
from the saved periodic segments and checks the original sealed artifacts.
Implementation: `core/halos.py:run_consistent_trees` and
`core/visualization.py:_load_disperse_skeleton/_disperse_filament_projection_map`.
'''),code('''if engine_root is None or not engine_root.is_dir():
    print('Select a completed engines run for tree and filament controls.')
else:
    from core.analysis import rockstar_table_from_path
    import h5py
    for backend in ('consistent-trees', 'disperse'):
        directory = engine_root / backend
        receipt_path = directory / 'receipt.json'
        if not receipt_path.is_file():
            print(backend, 'has no receipt in this run')
            continue
        check_published(receipt_path)
        worker = json.loads(receipt_path.read_text())
        print(backend, worker['status'])
        if worker['status'] != 'PASS':
            print(worker.get('reason'))
            continue
        check_published(directory / 'arrays.npz')
        print(worker['result']['scope'])
        with np.load(directory / 'arrays.npz', allow_pickle=False) as data:
            if backend == 'consistent-trees':
                native = []
                for name in worker['result']['native_catalogues']:
                    path = directory / name
                    check_published(path)
                    native.append(rockstar_table_from_path(path))
                for key in ('id','scale','desc_id','desc_scale','x','y','z','mvir','orig_halo_id'):
                    np.testing.assert_array_equal(np.concatenate([table[key] for table in native]), data[key])
                assert len(np.unique(data['id'])) == 256
                links = data['link_rows']
                assert links.shape == (128, 2) and ((links >= 0) & (links < 256)).all()
                source, target = links.T
                np.testing.assert_array_equal(data['desc_id'][source], data['id'][target])
                np.testing.assert_allclose(data['desc_scale'][source], data['scale'][target], rtol=0, atol=1e-12)
                assert (data['scale'][target] > data['scale'][source]).all()
                print('Original tree:', len(data['id']), 'nodes;', len(links), 'forward links')
                figure = directory / 'trees.png'
            else:
                manifest = directory / worker['result']['manifest']
                check_published(manifest)
                contract = json.loads(manifest.read_text())
                for item in contract['assets']:
                    path = manifest.parent / item['path']
                    check_published(path)
                    assert sha256(path.read_bytes()).hexdigest() == item['sha256']
                with h5py.File(manifest.parent / 'skeleton.hdf5') as original:
                    np.testing.assert_array_equal(original['points_comoving_mpc_h'][:], data['points_comoving_mpc_h'])
                    np.testing.assert_array_equal(original['vtk_polyline_connectivity'][:], data['vtk_polyline_connectivity'])
                points, lines = data['points_comoving_mpc_h'], data['vtk_polyline_connectivity']
                occupancy = np.zeros((16,16), dtype=bool)
                cursor = 0
                while cursor < len(lines):
                    count = int(lines[cursor]); rows = lines[cursor+1:cursor+1+count]
                    cursor += count+1
                    assert count >= 2 and len(rows) == count and ((rows >= 0) & (rows < len(points))).all()
                    for start, finish in zip(points[rows[:-1]], points[rows[1:]]):
                        delta = finish-start; delta -= .1*np.round(delta/.1)
                        samples = max(2, int(np.ceil(np.max(np.abs(delta))*16/.1))+1)
                        xyz = np.mod(start + np.linspace(0,1,samples)[:,None]*delta, .1)
                        bins = np.floor(xyz[:,:2]*16/.1).astype(int) % 16
                        occupancy[bins[:,0],bins[:,1]] = True
                np.testing.assert_array_equal(occupancy, data['projection_occupancy'])
                print('Original skeleton:', len(points), 'points;', np.count_nonzero(occupancy), 'occupied XY cells')
                figure = directory / 'filaments.png'
        check_published(figure)
        display(Image(filename=str(figure)))
''')])
    cells.extend([markdown(r'''## Independent OpenGadget3 scattering

This optional control requires an existing Linux rare/directed reference build.
Without it the checklist records UNAVAILABLE. It evolves 32 beam particles at
three saved epochs with gravity disabled and a deliberately high toy total
cross section of 100 cm²/g. The checks use exact IDs, fixed macro masses,
P=sum(m v) and K=sum(m |v|²)/2. Momentum residuals divide by the initial
sum(m |v|); energy residuals divide by initial K. The threshold is 1e-10.
K uses Msun (km/s)² and P uses Msun km/s. Static proper elapsed seconds equal
native time multiplied by L/(V h), as in the production control reader.
Changed velocities demonstrate scattering; they are not an event count or a
calibrated cross-section measurement.

Set RESEARCH2_LAB_SIDM to a standalone completed worker directory to inspect
Linux outputs from another environment. Otherwise the cell uses this run's
OpenGadget3 worker. It verifies retained output hashes and replays conservation
without executing the native binary.
'''),code('''sidm_specified = os.environ.get('RESEARCH2_LAB_SIDM')
sidm_directory = Path(sidm_specified).resolve() if sidm_specified else (engine_root / 'opengadget3' if engine_root is not None else None)
if sidm_directory is None or not (sidm_directory / 'receipt.json').is_file():
    print('No independent worker output selected. Set RESEARCH2_LAB_SIDM to its completed directory.')
else:
    sidm_receipt = json.loads((sidm_directory / 'receipt.json').read_text())
    print('OpenGadget3:', sidm_receipt['status'])
    if sidm_receipt['status'] != 'PASS':
        print(sidm_receipt.get('reason'))
    else:
        from hashlib import sha256
        for name, expected in sidm_receipt['result']['artifact_sha256'].items():
            path = (sidm_directory / name).resolve()
            assert path.is_relative_to(sidm_directory)
            assert sha256(path.read_bytes()).hexdigest() == expected
        with np.load(sidm_directory / 'arrays.npz', allow_pickle=False) as data:
            mass, velocity = data['mass_msun'], data['velocity_peculiar_kms']
            assert velocity.shape == (3,32,3)
            np.testing.assert_array_equal(data['particle_ids'], np.broadcast_to(data['particle_ids'][0], (3,32)))
            momentum = np.sum(mass[:,:,None] * velocity, axis=1)
            energy = .5*np.sum(mass*np.sum(velocity**2, axis=2), axis=1)
            np.testing.assert_array_equal(momentum, data['momentum_msun_kms'])
            np.testing.assert_array_equal(energy, data['kinetic_energy_msun_kms2'])
            momentum_scale = np.sum(mass[0]*np.linalg.norm(velocity[0], axis=1))
            assert np.max(np.linalg.norm(momentum-momentum[0],axis=1)/momentum_scale) < 1e-10
            assert np.max(abs(energy/energy[0]-1)) < 1e-10
            changed = np.count_nonzero(np.linalg.norm(velocity-velocity[0], axis=2)>1e-9,axis=1)
            np.testing.assert_array_equal(changed, data['changed_velocity_particles'])
            assert changed[-1] > 0
            print('Changed-velocity particles:', changed)
        display(Image(filename=str(sidm_directory / 'scattering.png')))
''')])
    cells.extend([markdown(r'''## Native gas, stellar births and black-hole sources

The `physics` profile adds three actual GADGET source controls to the seven
quick measurements; `complete` also includes the libraries and native engines.
Each source control uses a 4³ gas grid, **64 gas particles**. Stellar particles
are absent initially. Native SH03 conversions carry SSPBirthMode=1 and their
parent IDs, birth times and source journal are retained. The stellar seed is
the selected StellarRandomSeed, separate from continuous hydrodynamics.

The gas run shows the mass-weighted internal energy after hydrodynamics and
implicit atomic cooling. The star-forming run uses an effective multiphase
EOS, so its increasing gas energy is not a single-phase temperature estimate.
Its production history audit reconciles six-species masses, momentum and
native birth events with the frozen dimensional tolerances. Delayed stellar
returns and SPS light synthesis are separate controls.

The BH run contains one explicitly seeded 10-Msun sink. It starts from a
separately declared compressed gas IC; the original sealed grid is retained.
The toy background contains gas/BH baryonic matter. Saved BH_Mass is the
physical mass and BH_InflowRate is converted through its own cgs attributes.
Header/Time is scale factor a in this run, and physical native L/V time in the
other two. Positive mass growth/inflow is an execution check; full BH source
budget reconciliation, formation, galaxy accuracy and convergence are not
inferred from this example.

The next cell checks original HDF5 counts/energies against the saved arrays,
verifies the stellar source-history audit, and displays the original figures.
Implementation: `core/research/controls.py:prepare_native_source_control` →
`core/evolution.py:build_gadget/gadget_param_text` → native GADGET →
`core/stars.py:audit_native_hydro_history` for stellar reconciliation.
'''),code('''physics_root = RUN / 'physics' if RUN is not None else None
if physics_root is None or not physics_root.is_dir():
    print('Select a completed physics or complete-profile run for native source controls.')
else:
    from hashlib import sha256
    from core.data import dataset_to_cgs_factor
    from core.stars import verify_native_hydro_history
    import h5py
    published = {item['path']: item for item in receipt['artifacts']}
    def check_physics_file(path):
        assert sha256(path.read_bytes()).hexdigest() == published[path.relative_to(RUN).as_posix()]['sha256']
    for backend in ('gas','stars','black-holes'):
        directory = physics_root / backend
        path = directory / 'receipt.json'
        check_physics_file(path)
        worker = json.loads(path.read_text())
        print(backend, worker['status'])
        if worker['status'] != 'PASS':
            print(worker.get('reason')); continue
        print('Recorded source matches current checkout:', all(sha256((ROOT/name).read_bytes()).hexdigest()==expected for name,expected in worker['source_sha256'].items()))
        check_physics_file(directory / 'arrays.npz')
        with np.load(directory / 'arrays.npz',allow_pickle=False) as data:
            np.testing.assert_array_equal(data['particle_counts'],worker['result']['particle_counts'])
            for epoch,snapshot in enumerate(sorted((directory / 'data').glob('snapshot_*.hdf5'))):
                check_physics_file(snapshot)
                with h5py.File(snapshot) as native:
                    np.testing.assert_array_equal(native['Header'].attrs['NumPart_Total'],data['particle_counts'][epoch])
                    actual = float(native['Header'].attrs['Time'])
                    assert abs(actual-data['native_time'][epoch])<1e-12
                    a = actual if worker['result']['comoving'] else 1.
                    h = .7 if worker['result']['comoving'] else 1.
                    gas = native['PartType0']
                    u = gas['InternalEnergy'][:]*dataset_to_cgs_factor(gas['InternalEnergy'].attrs,a,h)
                    np.testing.assert_allclose(u,data[f'gas_specific_energy_erg_g_{epoch}'],rtol=1e-14,atol=0)
                    np.testing.assert_allclose(np.average(u,weights=gas['Masses'][:]),data['gas_mass_weighted_specific_energy_erg_g'][epoch],rtol=1e-14,atol=0)
            if backend=='gas':
                assert data['gas_mass_weighted_specific_energy_erg_g'][-1]<data['gas_mass_weighted_specific_energy_erg_g'][0]
            elif backend=='stars':
                assert data['particle_counts'][0,4]==0 and data['particle_counts'][-1,4]>0
                assert (data['SSPBirthMode_2']==1).all()
                check_physics_file(directory / 'history.json')
                history = verify_native_hydro_history(directory / 'history.json')
                assert history['status']=='PASS'
                print('Native stellar history:',history['status'],'| births:',data['particle_counts'][-1,4])
            else:
                assert data['bh_physical_mass_msun'][-1]>data['bh_physical_mass_msun'][0]
                assert data['bh_inflow_msun_per_year'].max()>0
                print('BH physical mass gain [Msun]:',np.diff(data['bh_physical_mass_msun']).sum())
        check_physics_file(directory / 'physics.png')
        display(Image(filename=str(directory / 'physics.png')))
''')])
    cells.extend([markdown(r'''## Fresh known-input shape and density controls

Run the `figures` profile, or `complete`, to regenerate these examples from the
shipped `advanced_controls.py`. The worker creates its own synthetic HDF5 and
measures it with the production readers; it does not need the author's campaign
folders. Shapes uses 6000 particles, and the density-valley control uses 32768.
Neither is native solver evolution. The next cell verifies the worker and
parent hashes, replays the physical coordinate/mass conversion, independently
recomputes shell tensor ratios, and checks the linear derivative/covariance
identities against the original saved arrays. It reads the original images
without rerunning AGAMA or selecting a different derivative window.
'''),code('''control_root = RUN / 'figure-controls' if RUN is not None else None
if control_root is None or not control_root.is_dir():
    print('Select a completed figures or complete profile for fresh known-input controls.')
else:
    import h5py
    from hashlib import sha256
    from core.data import dataset_to_cgs_factor
    from core.foundation import KPC_CM, MSUN_G
    published = {r['path']: r['sha256'] for r in receipt['artifacts']}
    for control in ('shapes','splashback'):
        directory = control_root / control
        path = directory / 'receipt.json'
        assert sha256(path.read_bytes()).hexdigest() == published[path.relative_to(RUN).as_posix()]
        worker = json.loads(path.read_text())
        if worker['status'] == 'UNAVAILABLE':
            print(control, 'UNAVAILABLE:', worker['reason'])
            continue
        assert worker['status'] == 'PASS'
        for name,digest in worker['artifacts_sha256'].items():
            assert sha256((directory / name).read_bytes()).hexdigest() == digest
        print(control, '| particles:',worker['result']['particles'], '| native evolution:',worker['result']['native_solver_executed'])
        with np.load(directory / 'authored-geometry.npz') as geometry:
            original = geometry['offsets_proper_kpc']
        with h5py.File(directory / 'data/snapshot_000.hdf5') as source:
            gas = source['PartType1']
            a = float(source['Header'].attrs['Time'])
            h = float(source['Parameters'].attrs['HubbleParam'])
            x = (gas['Coordinates'][:] - 50) * dataset_to_cgs_factor(gas['Coordinates'].attrs,a,h) / KPC_CM
            mass = gas['Masses'][:] * dataset_to_cgs_factor(gas['Masses'].attrs,a,h) / MSUN_G
        np.testing.assert_allclose(x,original,rtol=1e-12,atol=1e-12)
        np.testing.assert_allclose(mass,1.,rtol=1e-12,atol=0)
        structure = json.loads((directory / 'structure.json').read_text())
        if control == 'shapes':
            for shell in structure['density_shapes']['dm']['shells']:
                metric = np.asarray(shell['selection_metric'])
                r2 = np.einsum('ni,ij,nj->n',x,metric,x)
                keep = (r2 >= shell['lower_radius_kpc']**2) & (r2 < shell['upper_radius_kpc']**2)
                assert keep.sum() == shell['particle_count']
                tensor = np.einsum('n,ni,nj->ij',mass[keep],x[keep],x[keep]) / mass[keep].sum()
                eigen = np.linalg.eigvalsh(tensor)[::-1]
                ratios = np.sqrt(eigen[1:] / eigen[0])
                np.testing.assert_allclose(ratios,[shell['b_over_a'],shell['c_over_a']],rtol=0,atol=1e-4)
            images = sorted(directory.glob('figures/gallery/figures/png/structure/*axis_orientation_twist.png'))
            print('Original ellipsoidal-window tensor ratios replayed.')
        else:
            with np.load(directory / 'structure.json.data/arrays.npz') as raw:
                rho = raw['profile_dm_density_msun_kpc3']
                error = raw['profile_dm_density_shot_noise_msun_kpc3']
            with np.load(directory / 'environment.json.data/arrays.npz') as raw:
                prefix = 'smoothing_00_halo_00_splashback_'
                operator = raw[prefix+'derivative_operator']
                support = raw[prefix+'supported']
                logarithm = np.zeros_like(rho); logarithm[rho>0] = np.log(rho[rho>0])
                np.testing.assert_allclose((operator @ logarithm)[support],raw[prefix+'logarithmic_slope'][support],rtol=1e-10,atol=1e-10)
                covariance = np.diag(np.divide(error,rho,out=np.zeros_like(rho),where=rho>0)**2)
                np.testing.assert_allclose(operator @ covariance @ operator.T,raw[prefix+'slope_covariance'],rtol=1e-10,atol=1e-10)
            images = sorted(directory.glob('figures/gallery/figures/png/structure/*splashback.png'))
            print('Original derivative and covariance replayed; feature:',worker['result']['measurements']['feature']['status'])
        assert images
        display(Image(filename=str(images[0])))
''')])
    cells.extend([markdown(r'''## Replay the portable analytic flyby

The shipped `orbits` control creates 32 particles in two authored apertures at
three epochs. It calls the production graph, periodic-image, clock and Hermite
trajectory/event pipeline. This is a known straight-line geometry, not evolved
halos, finder detections or certified physical ancestry. Its host boundary is
a fixed 1.5-kpc test aperture, not a measured R200. The next cell derives the
analytic infall/pericentre times from the original saved relative states and
boundary, checks the reported candidates and displays the original figure.
The interpolation-uncertified status and saved brackets remain explicit.
'''),code('''orbit_directory = RUN / 'figure-controls/orbits' if RUN is not None else None
if orbit_directory is None or not (orbit_directory / 'receipt.json').is_file():
    print('Select a completed figures or complete profile with the portable orbits control.')
else:
    from hashlib import sha256
    import h5py
    from core.foundation import KPC_CM, SECONDS_PER_YEAR
    worker_path = orbit_directory / 'receipt.json'
    published = {r['path']: r['sha256'] for r in receipt['artifacts']}
    assert sha256(worker_path.read_bytes()).hexdigest() == published[worker_path.relative_to(RUN).as_posix()]
    worker = json.loads(worker_path.read_text())
    if worker['status'] == 'UNAVAILABLE':
        print('orbits UNAVAILABLE:',worker['reason'])
    else:
        assert worker['status'] == 'PASS' and not worker['result']['native_solver_executed']
        assert not worker['result']['trajectory_certified']
        for name,digest in worker['artifacts_sha256'].items():
            assert sha256((orbit_directory / name).read_bytes()).hexdigest() == digest
        with np.load(orbit_directory / 'events.json.data/arrays.npz') as data:
            time = data['times_myr']
            separation = data['object_position_proper_kpc'] - data['host_position_proper_kpc']
            relative_derivative = data['object_velocity_derivative_kms'] - data['host_velocity_derivative_kms']
            assert data['object_valid_intervals'].all() and data['host_valid_intervals'].all()
            speed = (separation[-1,0]-separation[0,0])/(time[-1]-time[0])
            velocity_kpc_myr = relative_derivative * 1e5 * (1e6*SECONDS_PER_YEAR) / KPC_CM
            np.testing.assert_allclose(velocity_kpc_myr,np.tile([speed,0.,0.],(3,1)),rtol=1e-10,atol=1e-12)
            t_peri = time[0] - separation[0,0]/speed
            impact = np.linalg.norm(separation[:,1:],axis=1)
            np.testing.assert_allclose(impact,impact[0],rtol=0,atol=1e-12)
            with h5py.File(orbit_directory / 'data/fof_subhalo_tab_000.hdf5') as source:
                boundary = source['Group/Group_R_Crit200'][1] * source['Header'].attrs['Time'] / source['Parameters'].attrs['HubbleParam']
            t_infall = t_peri - np.sqrt(boundary**2-impact[0]**2)/speed
            for code,expected in ((0,t_peri),(1,t_infall)):
                values = data['event_time_myr'][data['event_kind_code']==code]
                assert len(values)==1
                np.testing.assert_allclose(values,[expected],rtol=0,atol=worker['result']['time_absolute_tolerance_myr'])
            assert len(data['merger_time_myr'])==0
        print('Derived infall/pericentre [Myr]:',t_infall,t_peri,'| native evolution: False | interpolation: NOT_CERTIFIED')
        image = next(orbit_directory.glob('figures/gallery/figures/png/assembly/*assembly-events.png'))
        display(Image(filename=str(image)))
''')])
    cells.extend([markdown(r'''## Read and replay the native GADGET control

Select a `native` or `combined` run to execute this cell. The isolated
self-gravity control uses native units with $G=1$ and a length/velocity time
unit; its output clock is not a cosmological scale factor or a number of Myr.
For the saved masses, velocities and self-gravity potentials,
$$K=\frac12\sum_i m_i|v_i|^2,\quad
W=\frac12\sum_i m_i\Phi_i,\quad E=K+W,\quad
L=\sum_i m_i(x_i\times v_i).$$
The factor one half in $W$ prevents counting each pair twice. This equation
applies to this isolated self-gravity case. A prescribed external potential
would have a different potential-energy accounting rule. The cell recomputes
the original energy and angular-momentum residuals, checks their numerical
archives and displays the measured energy drift. The saved thresholds and
reference floors remain visible; a tiny short run does not establish general
force accuracy or physical convergence.
Implementation: `optional_tests/lab.py:native_control` →
`core/research/controls.py:run_control_campaign` → native GADGET →
`core/research/controls.py:_analyze_case` → saved report and NPZ.
'''),code('''native_path = RUN / 'native-control/plummer-tree/analysis.json' if RUN is not None else None
if native_path is None or not native_path.is_file():
    print('Select a completed native or combined run to replay actual native diagnostics.')
else:
    from hashlib import sha256
    from core.data import verify_numeric_audit
    native = verify_numeric_audit(native_path)
    published = {row['path']: row for row in receipt['artifacts']}
    arrays_path = native_path.parent / native['numeric_data']['directory'] / native['numeric_data']['array_contract']['path']
    for path in (native_path, arrays_path):
        key = path.relative_to(RUN).as_posix()
        assert sha256(path.read_bytes()).hexdigest() == published[key]['sha256']
    print('Clock:', native['clock'], '| numerical status:', native['status'])
    for finding in native['findings']:
        print(finding['metric'], finding['status'], 'threshold', finding.get('threshold'))
    with np.load(arrays_path, allow_pickle=False) as data:
        times = data['actual_times']
        mass, position, velocity, potential = [data[key] for key in ('mass', 'position', 'velocity', 'potential')]
        kinetic = .5 * np.sum(mass * np.sum(velocity**2, axis=-1), axis=1)
        binding = .5 * np.sum(mass * potential, axis=1)
        energy = kinetic + binding
        angular = np.sum(mass[..., None] * np.cross(position, velocity), axis=1)
        energy_drift = energy - energy[0]
        np.testing.assert_allclose(energy_drift, data['isolated_system_energy_residual'], rtol=1e-12, atol=1e-15)
        np.testing.assert_allclose(angular - angular[0], data['isolated_system_angular_residual'], rtol=1e-12, atol=1e-15)
        print('Actual times:', times, '| particles per epoch:', mass.shape[1])
        print('K, W, E in native M V^2:', np.column_stack((kinetic, binding, energy)))
        fig, ax = plt.subplots(figsize=(7, 4), layout='constrained')
        ax.plot(times, energy_drift, 'o-', color='#126b73')
        ax.set(xlabel='Actual isolated time [native L/V]', ylabel='E(t) - E(0) [native M V²]',
               title='Native GADGET energy drift · original saved values')
        ax.grid(alpha=.2)
        display(fig)
        plt.close(fig)
''')])
    cells.append(markdown('''## All registered recipes: provenance before interpretation

Every entry below is generated from `configs/research/graph_producers.json`.
Archived PNGs show actual retained products; their NPZ files contain the plotted
numerical sidecars. **No retained example** means execution is still unproven.
A zero-support example proves rendering with masks, not a supported measurement.
The portable reference contracts retain original workstation paths for lineage;
those original native inputs are not all part of this optional download.
'''))
    for graph in catalogue['graphs']:
        ref=graph.get('reference');text='### '+graph['family']+' / '+graph['variant']+'\n\n'+graph['reading_guide']+'\n\n'
        text+='Required contract: `'+str(graph.get('input_contract','See the registered route'))+'`\n\n'
        text+='```json\n'+json.dumps({k:v for k,v in graph.items() if k not in {'reference','reading_guide'}},indent=2)+'\n```\n'
        if ref:
            text+='\nArchived example: '+str(ref['supported_entries'])+' supported plotted entries.\n\n![Original plot]('+ref['image']+')\n\nArrays: `'+ref['arrays']+'`; figure contract: `'+ref['manifest']+'`.\n'
            text+='\n'+ '\n'.join('- '+axis.get('x_label','')+' versus '+axis.get('y_label','') for axis in ref.get('axes',[]))+'\n'
        else:text+='\n**No retained example.** Supply the named physical inputs and execute the original producer.\n'
        cells.append(markdown(text))
    cells.append(markdown("""## Replay a selected archived figure

Set `RESEARCH2_LAB_FIGURE` to a `family/variant` pair before starting the kernel,
or edit `selected` in the next cell. The default opens the certified
impact-geometry control when present. The cell verifies the archive hashes,
displays the figure and axis contracts, and prints each measured array's shape,
type and finite numeric range. The implementation chain identifies the
producer and renderer source locations in your checkout.
"""))
    cells.append(code('''from hashlib import sha256
selected = os.environ.get('RESEARCH2_LAB_FIGURE', 'encounters/impact_velocity_geometry')
recipes = {g['family'] + '/' + g['variant']: g for g in catalogue['graphs']}
recipe = recipes[selected]
example = recipe['reference']
for step in recipe['implementation_route']:
    print(step['role'], step['name'], step['path'], 'line', step['line'])
if example is None:
    print('No retained example. Required input:', recipe['input_contract'])
else:
    directory = ROOT / 'optional_tests/reference_gallery'
    for key, field in [('arrays', 'arrays_sha256'), ('image', 'image_sha256'),
                       ('manifest', 'original_manifest_sha256')]:
        path = directory / example[key]
        assert sha256(path.read_bytes()).hexdigest() == example[field], path
    display(Image(filename=str(directory / example['image'])))
    for axis in example['axes']:
        print(axis.get('x_label', ''), '|', axis.get('y_label', ''), '|', axis.get('z_label', ''))
    print(example['supported_entries'], 'supported entries across plotted series')
    with np.load(directory / example['arrays'], allow_pickle=False) as arrays:
        for name in arrays.files:
            value = arrays[name]
            print(name, value.shape, value.dtype, end=' ')
            if value.dtype.kind == 'b':
                print(int(value.sum()), 'true')
            elif value.dtype.kind in 'iuf':
                finite = value[np.isfinite(value)]
                print((finite.min().item(), finite.max().item()) if finite.size else 'no finite values')
            else:
                print()
    print('Full figure contract:', directory / example['manifest'])
'''))
    notebook={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python 3 (Research2 environment)','language':'python','name':'python3'},
      'language_info':{'name':'python','version':'3'}},'cells':cells}
    for i,cell in enumerate(cells):cell['id']='lab-'+str(i)
    write(HERE/'reference_gallery/Walkthrough.ipynb',notebook)
    return len(cells)


def main():
    """Optionally import campaign examples, then rebuild notebook and source inventory."""
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--campaign',type=Path)
    args=parser.parse_args()
    count=import_campaign(args.campaign) if args.campaign else None
    cells=build_notebook();write(HERE/'reference_gallery/source-inventory.json',source_inventory())
    print(json.dumps({'imported_recipes':count,'notebook_cells':cells}))

if __name__=='__main__':main()
