"""Finder-sensitive catalogue mass profiles with fixed spatial jackknife bins.

This measures summed spherical-overdensity host masses per aperture volume.
Overlapping host masses are retained: this is a catalogue statistic, not a
measurement of the underlying matter density or a cross-finder identity match.
"""
from pathlib import Path
import json
import numpy as np
from .foundation import atomic_json,file_sha256,canonical_digest
from .foundation import merge_source_bindings as _merge


def catalogue_profile_inputs(parameters):
    from .sidm_analysis import _stage_native_bindings
    from .halos import _group_files,_snapshot_files,_unit_conversion
    from .data import native_publication_markers
    request=parameters['request'];memory=request['memory_gib']
    if type(memory) not in (int,float) or not np.isfinite(memory) or memory<=0:raise ValueError('Positive catalogue convergence allocation required')
    pins=_stage_native_bindings(parameters)
    run=Path(parameters['native_run_directory']).resolve()
    files=[*_group_files(run,request['snapshot']),*_snapshot_files(run,request['snapshot'])]
    if not files:raise ValueError('Actual saved catalogue and snapshot required')
    for path in files:
        if pins.get(str(path.resolve()))!=file_sha256(path):raise ValueError('Finder convergence sources must be owned by completed native evolution')
    _unit_conversion(files)
    _merge(pins,{str(p.resolve()):file_sha256(p) for p in native_publication_markers(files)})
    reference=parameters.get('paired_reference')
    if reference:
        path=Path(reference).resolve();seal=path.with_name(path.name+'.sha256')
        if seal.read_text().strip()!=file_sha256(path):raise ValueError('Catalogue convergence baseline measurement seal changed')
        baseline=json.loads(path.read_text());_merge(pins,{str(path):file_sha256(path),str(seal):file_sha256(seal),**baseline['source_hashes']})
    return pins


def catalogue_profile(parameters):
    import h5py
    from .analysis import load_cfg,_read_cross_run_halo_catalogue,group_field
    from .halos import _group_files,_snapshot_files
    from .data import run_epoch_contract
    from core.research.measurements import _estimate_profile
    request=parameters['request'];definition=request['observable_definition'];selection=parameters['selection']
    pins=catalogue_profile_inputs(parameters);run=Path(parameters['native_run_directory']).resolve();snapshot=request['snapshot']
    if definition.get('observable')!='catalogued_host_m200c_mass_per_shell_volume' or definition['units']!='Msun / proper kpc^3':raise ValueError('Explicit catalogue M200c statistic and units required')
    if set(selection)!={'minimum_fine_dm_particles','minimum_objects_per_bin','minimum_effective_objects','minimum_objects_per_octant'}:raise ValueError('Complete fixed catalogue support criteria required')
    if any(type(v) is not int or v<1 for v in selection.values()) or selection['minimum_fine_dm_particles']<100:raise ValueError('Retain at least100 fine DM particles per selected host and positive catalogue support')
    edges=np.asarray(definition['radius_edges_proper_kpc'],float);center=np.asarray(definition['center_proper_kpc'],float)
    if center.shape!=(3,) or not np.isfinite(center).all() or edges.ndim!=1 or len(edges)!=len(definition['bins'])+1 or not np.isfinite(edges).all() or edges[0]<0 or np.any(np.diff(edges)<=0):raise ValueError('Fixed physical catalogue profile domain required')
    if definition.get('jackknife_partition')!='fixed_center_octants':raise ValueError('Common fixed catalogue octant partition required')
    for i,row in enumerate(definition['bins']):
        if row!={'radius_proper_kpc':edges[i:i+2].tolist()}:raise ValueError('Catalogue profile bins must retain actual physical bounds')
    # Bound the original reader's metadata and exact-ID workspace before intake.
    rows=0;ids=0
    for path in _group_files(run,snapshot):
        with h5py.File(path) as f:
            if 'Group/GroupLenType' in f:rows+=len(f['Group/GroupLenType'])
            if 'IDs/ParticleIDs' in f:ids+=len(f['IDs/ParticleIDs'])
    if rows*4096+ids*64+64*1024**2>request['memory_gib']*1024**3:raise MemoryError('Native finder catalogue exceeds allocated workspace')
    cfg=load_cfg(run);h=float(cfg['cosmology']['H0'])/100.
    first=_snapshot_files(run,snapshot)[0]
    with h5py.File(first) as f:a=float(f['Header'].attrs['Time']);box=float(f['Header'].attrs['BoxSize'])*a/h
    declared_box=definition.get('periodic_box_proper_kpc')
    if type(declared_box) not in (int,float) or not np.isfinite(declared_box) or not np.isclose(declared_box,box,rtol=1e-12,atol=0) or edges[-1]>box/2:raise ValueError('Catalogue convergence box/frame differs from actual native snapshot')
    catalogue=_read_cross_run_halo_catalogue(run,snapshot,'m200c',selection['minimum_fine_dm_particles'],request['maximum_particles'])
    if catalogue['redshift'] is None or not np.isclose(1/(1+catalogue['redshift']),a,rtol=1e-12,atol=0):raise ValueError('Native catalogue and saved snapshot epochs differ')
    lengths=group_field(run,snapshot,['GroupLenType'])
    if lengths is None:raise ValueError('Actual coarse/fine catalogue census required')
    selected=lengths[catalogue['group_indices']]
    keep=(catalogue['n_dm']>=selection['minimum_fine_dm_particles'])&(selected[:,2]==0)&(selected[:,3]==0)
    displacement=np.asarray(catalogue['position_mpc_h'])[keep]*1000*a/h-center
    displacement=(displacement+box/2)%box-box/2
    mass=np.asarray(catalogue['mass_msun_h'])[keep]/h;velocity=np.zeros_like(displacement)
    value,counts,effective=_estimate_profile(displacement,velocity,mass,edges,'density')
    octants=np.sum((displacement>=0)*np.array([1,2,4]),axis=1);replicates=[];masks=[];populations=[]
    for label in range(8):
        chosen=octants!=label
        v,n,neff=_estimate_profile(displacement[chosen],velocity[chosen],mass[chosen],edges,'density',8/7)
        replicates.append(v);masks.append((n>=selection['minimum_objects_per_bin'])&(neff>=selection['minimum_effective_objects']))
        populations.append(int(np.sum((octants==label)&(np.linalg.norm(displacement,axis=1)<=edges[-1]))))
    replicates=np.asarray(replicates);residual=replicates-replicates.mean(axis=0);covariance=7/8*residual.T@residual
    supported=(counts>=selection['minimum_objects_per_bin'])&(effective>=selection['minimum_effective_objects'])&np.logical_and.reduce(masks)
    supported&=min(populations)>=selection['minimum_objects_per_octant']
    uncertainty={'method':'delete-one fixed spatial octant catalogue jackknife','scope':'conditional catalogue spatial sampling; excludes cosmic variance and cross-finder object identities',
        'replicate_values':replicates.tolist(),'replicate_labels':list(range(8)),'covariance_factor':7/8,
        'partition_definition_hash':canonical_digest(definition),'bin_object_count':counts.tolist(),'effective_object_count':effective.tolist(),'octant_object_count':populations}
    reference=parameters.get('paired_reference')
    if reference:
        baseline=json.loads(Path(reference).read_text());previous=baseline['measurement_uncertainty']
        if baseline['observable_definition_hash']!=canonical_digest(definition):raise ValueError('Catalogue convergence baseline observable changed')
        for key in ('method','replicate_labels','covariance_factor','partition_definition_hash'):
            if uncertainty[key]!=previous[key]:raise ValueError('Catalogue paired spatial partition changed')
        old=np.asarray(previous['replicate_values'],float)
        if old.shape!=replicates.shape:raise ValueError('Catalogue baseline jackknife dimensions differ')
        uncertainty.update(paired_cross_covariance=(7/8*residual.T@(old-old.mean(axis=0))).tolist(),paired_reference_sha256=file_sha256(reference))
    pins[str(Path(__file__).resolve())]=file_sha256(__file__)
    return {'schema':'native-catalogue-convergence-profile-v1','observable':definition['observable'],'observable_definition_hash':canonical_digest(definition),
        'units_of_observable':definition['units'],'bins':definition['bins'],'epoch':run_epoch_contract(run,snapshot,a),
        'measurement_uncertainty':uncertainty,'selection':selection,'catalogue_scope':'Fixed proper aperture; clean fine-DM FOF-selected hosts with common M200c mass definition; summed overlapping host masses retained'}, {'value':np.where(supported,value,np.nan),'covariance':np.where(supported[:,None]&supported[None,:],covariance,np.nan),'supported':supported},pins


def catalogue_profile_runner(parameters,context):
    from .data import export_numeric_audit
    if parameters['request']['memory_gib']>context.resources['memory_gib']:raise MemoryError('Catalogue profile exceeds stage allocation')
    report,arrays,pins=catalogue_profile(parameters)
    export_numeric_audit(context.generation/'catalogue-profile.json',report,arrays,{
        'value':{'units':'Msun / proper kpc^3','quantity':'Sum of catalogue host M200c per shell volume'},
        'covariance':{'units':'(Msun / proper kpc^3)^2','quantity':'Conditional catalogue jackknife covariance'},
        'supported':{'units':'dimensionless','quantity':'Measured catalogue sampling support'}},algorithm=__file__,context={'epoch':report['epoch']},source_hashes=pins)
    return {'measurements':{'supported_bins':int(arrays['supported'].sum()),'science_acceptance':'NOT_ACCEPTED'}}
