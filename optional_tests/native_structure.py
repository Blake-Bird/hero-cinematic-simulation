"""Native tree and filament controls with original outputs and explicit scope.

Consistent Trees receives 128 analytic catalogue objects at two epochs, not
particle detections or an evolved halo history. DisPerSE receives 64 synthetic
DM samples, the production CIC/log-density field and a 16³ periodic mesh.
Both require installed native engines; no dependencies are built or installed.
All inputs, native products, numerical arrays and figures stay in the lab run.
"""
from pathlib import Path
import json
import shutil


def trees(output, seed):
    """Run production tree construction; read actual hlists and validate links."""
    import numpy as np
    import yaml
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from core.native import native_engine_directory
    from core.evolution import ensure_run_layout
    from core.halos import run_consistent_trees
    from core.analysis import rockstar_table_from_path
    from core.foundation import file_sha256
    engine = native_engine_directory('consistent-trees')
    rockstar = native_engine_directory('rockstar')
    if not all(p.is_file() for p in [engine/'gravitational_consistency', engine/'do_merger_tree.pl',
            engine/'halo_trees_to_catalog.pl', rockstar/'scripts/gen_merger_cfg.pl']) or not shutil.which('perl'):
        raise ImportError('Build consistent-trees and rockstar with setup; Perl is also required.')
    root = Path(__file__).resolve().parents[1]
    output = Path(output).resolve(); ensure_run_layout(output)
    directory = output/'data/rockstar'; directory.mkdir()
    cfg = yaml.safe_load((root/'core/config.yaml').read_text())
    cfg['cosmology'].update(model='lcdm', H0=70., omega_b=.0245, omega_cdm=.1225)
    cfg['simulation'].update(physics='dm', box_mpc_h=100., particles_per_dimension=None)
    cfg['hardware'].update(cpu_threads=2, omp_threads=2, mpi_ranks=1, memory_limit_gb=2.)
    cfg['halos']['min_particles'] = 16
    (output/'config.yaml').write_text(yaml.safe_dump(cfg))
    columns = ('ID DescID Mvir Vmax Vrms Rvir Rs Np X Y Z VX VY VZ JX JY JZ Spin rs_klypin '
        'Mvir_all M200c M200b M500c M2500c Xoff Voff spin_bullock b_to_a c_to_a A[x] A[y] A[z] '
        'b_to_a(500c) c_to_a(500c) A[x](500c) A[y](500c) A[z](500c) T/|U| M_pe_Behroozi M_pe_Diemer Halfmass_Radius').split()
    rng = np.random.default_rng(seed)
    positions = rng.uniform(5., 95., (128, 3))
    for epoch, scale in enumerate((.99, 1.)):
        header = ['#'+' '.join(columns), f'#a = {scale:.6f}',
            '#Om = 0.300000; Ol = 0.700000; h = 0.700000',
            '#Particle mass: 5.00000e+07 Msun/h', '#Box size: 100.000000 Mpc/h',
            '#Units: Masses in Msun / h', '#Units: Positions in Mpc / h (comoving)',
            '#Units: Velocities in km / s (physical, peculiar)',
            '#Units: Halo Distances, Lengths, and Radii in kpc / h (comoving)',
            '#Synthetic catalogue control; no particle finding or dynamics']
        for identity, position in enumerate(positions):
            values = dict.fromkeys(columns, 0.)
            position = position + rng.normal(0., .002, 3)*epoch
            values.update(ID=identity, DescID=identity if epoch == 0 else -1,
                Mvir=1e12, Vmax=150., Vrms=200., Rvir=200., Rs=20., Np=1000,
                rs_klypin=20., Mvir_all=1e12, M200c=1e12, M200b=1e12,
                M500c=1e12, M2500c=1e12, Halfmass_Radius=100.)
            values.update(zip(('X','Y','Z'), position))
            values.update(zip(('VX','VY','VZ'), rng.normal(0., 2., 3)))
            for key in ('b_to_a','c_to_a','b_to_a(500c)','c_to_a(500c)','A[x]','A[x](500c)'): values[key] = 1.
            values['T/|U|'] = .5
            header.append(' '.join(str(int(values[key])) if key in {'ID','DescID','Np'} else format(values[key], '.17e') for key in columns))
        (directory/f'out_{epoch}.list').write_text('\n'.join(header)+'\n')
    (output/'input/rockstar.cfg').write_text(f'OUTBASE = "{directory}"\nNUM_SNAPS = 2\n'
        'MASS_DEFINITION = "vir"\nMASS_DEFINITION2 = "200c"\nMASS_DEFINITION3 = "200b"\n'
        'MASS_DEFINITION4 = "500c"\nMASS_DEFINITION5 = "2500c"\n')
    status = run_consistent_trees(output)
    assert status['status'] == 'COMPLETE', status
    bundle = Path(status['bundle_directory'])
    paths = sorted(bundle.rglob('hlist_*.list'))
    assert len(paths) == 2
    tables = [rockstar_table_from_path(path) for path in paths]
    assert all(table is not None and len(table['id']) == 128 for table in tables)
    arrays = {key:np.concatenate([table[key] for table in tables]) for key in
        ('id','scale','desc_id','desc_scale','x','y','z','mvir','orig_halo_id')}
    ids = arrays['id']; scales = arrays['scale']
    assert ids.dtype.kind in 'iu' and len(np.unique(ids)) == len(ids) == 256
    by_id = {int(value):index for index, value in enumerate(ids)}
    links = []
    for index, descendant in enumerate(arrays['desc_id']):
        if descendant < 0: continue
        assert int(descendant) in by_id
        target = by_id[int(descendant)]
        assert scales[target] > scales[index]
        assert np.isclose(arrays['desc_scale'][index], scales[target], rtol=0, atol=1e-12)
        links.append((index, target))
    assert len(links) == 128
    arrays['link_rows'] = np.asarray(links, dtype=np.int64)
    np.savez_compressed(output/'arrays.npz', **arrays)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for row, target in links[:16]:
        axes[0].plot(scales[[row,target]], arrays['x'][[row,target]], marker='.', alpha=.6)
    axes[0].set(xlabel='Scale factor a', ylabel='x [comoving Mpc/h]', title='16 native descendant links')
    axes[1].bar([.99,1.], [128,128], width=.003)
    axes[1].set(xlabel='Scale factor a', ylabel='Native catalogue rows', title='128 synthetic objects per epoch')
    fig.savefig(output/'trees.png', dpi=140); plt.close(fig)
    return dict(input_objects_per_epoch=128, epochs=2, native_nodes=256, native_links=128,
        native_catalogues=[p.relative_to(output).as_posix() for p in paths],
        bundle=bundle.relative_to(output).as_posix(), binary_sha256=file_sha256(engine/'gravitational_consistency'),
        units={'position':'comoving Mpc/h','mass':'Msun/h','time':'scale factor a','identity':'exact native integer IDs'},
        implementation='core/halos.py:run_consistent_trees → Rockstar gen_merger_cfg.pl → Consistent Trees native executables/Perl → core/analysis.py:rockstar_table_from_path',
        scope='Actual native construction and intake of 256 nodes/128 forward links from two analytic 128-object catalogues. No particles generated, halo detection, physical merger history, matching accuracy or convergence claimed. The declared Np field is a synthetic catalogue property, not a particle count executed here.')


def disperse(output, seed):
    """Run the production periodic CIC-density/skeleton route on 64 DM samples."""
    import numpy as np
    import yaml
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from core.native import native_engine_directory
    from core.foundation import file_sha256
    from optional_tests.native_finders import fixture
    from core.visualization import _load_disperse_skeleton, _disperse_filament_projection_map
    output = Path(output).resolve()
    engine = native_engine_directory('disperse')/'build/src/mse'
    if not (engine/'mse').is_file() or not (engine/'skelconv').is_file():
        raise ImportError('Build DisPerSE with scripts/setup.sh --engines disperse.')
    cfg, position, _, ids = fixture(output, seed, particles=64)
    cfg['visualization'].update(disperse_executable=str(engine/'mse'), skelconv_executable=str(engine/'skelconv'),
        filament_grid_size=16, disperse_persistence_cut=.1, disperse_density_smoothing_cells=1.,
        disperse_smoothing_cells=1, disperse_output_coordinate_representation='grid_vertex_coordinates',
        disperse_timeout_s=45, disperse_memory_gib=.125)
    (output/'config.yaml').write_text(yaml.safe_dump(cfg))
    skeleton = _load_disperse_skeleton(output, 1, 16, .1)
    centers, occupancy, metadata = _disperse_filament_projection_map(output, 1, (0,1), grid_size=16)
    assert metadata['density_particle_count'] == 64 and metadata['grid_size'] == 16
    assert occupancy.shape == (16,16) and occupancy.any()
    points, lines = skeleton['points_mpc_h'], skeleton['lines']
    assert points.shape[1] == 3 and np.isfinite(points).all()
    assert ((points >= 0) & (points < .1)).all()
    np.savez_compressed(output/'arrays.npz', input_position_comoving_kpc_h=position, input_ids=ids,
        points_comoving_mpc_h=points, vtk_polyline_connectivity=lines,
        projection_centers_comoving_mpc_h=centers, projection_occupancy=occupancy)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axes[0].scatter(position[:,0]/1000., position[:,1]/1000., s=12)
    axes[0].set(title='64 synthetic DM samples', xlabel='x [comoving Mpc/h]', ylabel='y [comoving Mpc/h]')
    axes[1].imshow(occupancy.T, origin='lower', extent=(0,.1,0,.1), interpolation='nearest', cmap='magma')
    axes[1].set(title='Native skeleton XY occupancy', xlabel='x [comoving Mpc/h]', ylabel='y [comoving Mpc/h]')
    for axis in axes: axis.set_aspect('equal')
    fig.savefig(output/'filaments.png', dpi=140); plt.close(fig)
    bundle = Path(metadata['bundle_directory'])
    return dict(input_particles=64, grid_size=16, persistence_cut=.1,
        skeleton_points=len(points), polylines=int(metadata['filament_polyline_count']),
        projected_cells=int(np.count_nonzero(occupancy)), bundle=bundle.relative_to(output).as_posix(),
        manifest=(bundle/'filament_contract.json').relative_to(output).as_posix(),
        binary_sha256={name:file_sha256(engine/name) for name in ('mse','skelconv')},
        units={'position':'comoving Mpc/h','occupancy':'boolean projected grid cells','field':'dimensionless log10(1 + smoothed rho / mean rho)'},
        implementation='core/visualization.py:_load_disperse_skeleton → CIC field/FITS → native mse/skelconv → topology/geometry intake → _disperse_filament_projection_map',
        scope='Actual native periodic topology extraction and projection from 64 synthetic DM samples on a 16³ mesh. One-cell density smoothing and one display-smoothing iteration are explicit. Native unsmoothed topology is retained separately; no resolved cosmic-web, persistence calibration or convergence acceptance claimed.')
