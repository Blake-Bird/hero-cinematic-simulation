"""Exercise graph/HDF5 orbital-event intake with an authored straight-line flyby.

32 particles, two explicit aperture centres and three epochs have known proper
positions/derivatives. These are synthetic catalogues, not finder detections,
GADGET evolution, native ancestry, certified interpolation or physical mergers.
The host boundary is a declared 1.5-kpc test aperture, not a measured R200.
"""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def _parameters(run):
    from core.foundation import KPC_CM, MSUN_G
    return dict(UnitLength_in_cm=KPC_CM,UnitMass_in_g=1e10*MSUN_G,
        UnitVelocity_in_cm_per_s=1e5,HubbleParam=.7,Omega0=.3,OmegaLambda=.7,
        OmegaBaryon=.05,ComovingIntegrationOn=1,InitCondFile=str(run/'initial_source'))


def _write_snapshot(run,epoch,a,centres,velocities,offsets):
    import h5py
    import numpy as np
    from core.foundation import KPC_CM, MSUN_G
    with h5py.File(run/'data'/f'snapshot_{epoch:03d}.hdf5','w') as f:
        f.attrs.update(synthetic_fixture=True,purpose=__doc__)
        counts=np.array([0,32,0,0,0,0],dtype=np.uint32)
        f.create_group('Header').attrs.update(Time=a,Redshift=1/a-1,BoxSize=100.,
            MassTable=np.zeros(6),NumFilesPerSnapshot=1,NumPart_Total=counts,NumPart_ThisFile=counts,
            NumPart_Total_HighWord=np.zeros(6,dtype=np.uint32),HubbleParam=.7,
            Omega0=.3,OmegaBaryon=.05,OmegaLambda=.7,
            Research2Generation=f'analytic-flyby-{epoch}',Research2ShardCount=1)
        f.create_group('Parameters').attrs.update(_parameters(run));g=f.create_group('PartType1')
        for name,values,unit,aa,hh in (
            ('Coordinates',np.r_[centres[0]+offsets,centres[1]+offsets]*.7/a,KPC_CM,1.,-1.),
            ('Velocities',np.repeat(velocities,16,axis=0)/np.sqrt(a),1e5,.5,0.),
            ('Masses',np.full(32,.7/1e10),1e10*MSUN_G,0.,-1.)):
            g[name]=values;g[name].attrs.update(to_cgs=unit,a_scaling=aa,h_scaling=hh)
        g['ParticleIDs']=np.arange(1,33,dtype=np.uint64)


def _write_catalogue(run,epoch,a,centres):
    import h5py
    import numpy as np
    with h5py.File(run/'data'/f'fof_subhalo_tab_{epoch:03d}.hdf5','w') as f:
        f.attrs.update(synthetic_fixture=True,purpose='Authored test centres and boundary; no finder or R200 measurement')
        f.create_group('Header').attrs.update(Time=a,BoxSize=100.,NumFiles=1,Ngroups_Total=2,Ngroups_ThisFile=2)
        f.create_group('Parameters').attrs.update(_parameters(run));g=f.create_group('Group')
        g['GroupPos']=centres*.7/a;g['Group_R_Crit200']=np.array([.05,1.5])*.7/a
        g['GroupLenType']=np.array([[0,16,0,0,0,0]]*2,dtype=np.uint64)
        g['GroupOffsetType']=np.array([[0,0,0,0,0,0],[0,16,0,0,0,0]],dtype=np.uint64)


def calculate(output, *, seed, memory_gib):
    """Create three epochs, run the real event producer, and check analytic times.

    Returns the actual event list and gallery location. All catalogues and
    identity correspondences are explicitly authored, with 16 particles per
    aperture. The 1.5-kpc test boundary does not represent a measured R200.
    """
    import json, shutil
    import numpy as np, h5py, yaml
    from core.foundation import atomic_json,file_sha256,KPC_CM,SECONDS_PER_YEAR,C_SI
    from core.halos import IdentityGraph
    from core.data import run_epoch_contract
    from core.stream_common import publish_stream_product,read_product
    from core.flagship_orbits import measure_native_assembly_orbits,export_native_assembly_event_figures
    run=Path(output).resolve();(run/'data').mkdir()
    scales=np.array([.5,.5000025,.500005]);times=1000+2e6*(scales-.5)
    center=np.full(3,50*.5/.7);positions=[];velocities=[];pins={};cfg=yaml.safe_load((ROOT/'core/config.yaml').read_text());cfg['cosmology'].update(H0=70.,omega_b=.0245,omega_cdm=.1225);(run/'config.yaml').write_text(yaml.safe_dump(cfg))
    rng=np.random.default_rng(seed);q=rng.normal(size=(8,3));q=.01*q/np.linalg.norm(q,axis=1)[:,None];q=np.r_[q,-q]
    # Proper derivative: 0.4 kpc/Myr -> km/s; c is not an input to this trajectory.
    speed=.4*KPC_CM/(1e6*SECONDS_PER_YEAR)/1e5
    for i,(a,t) in enumerate(zip(scales,times)):
     obj=center+np.array([.4*(t-1005),.5,0.]);host=center.copy();pos=np.vstack([obj,host])
     Hs=1/(2000*a*1e9*SECONDS_PER_YEAR);v=np.array([[speed,0.,0.],[0.,0.,0.]])-Hs*(pos*KPC_CM)/1e5
     positions.append(pos);velocities.append(v)
     _write_snapshot(run,i,float(a),pos,v,q)
     atomic_json(run/'data'/f'snapshot_{i:03d}.publication.json',{'schema':'native-hdf5-set-publication-v1','state':'COMPLETE','generation':f'analytic-flyby-{i}','shard_count':1})
     _write_catalogue(run,i,float(a),pos)
    shutil.copyfile(run/'data/snapshot_000.hdf5',run/'initial_source.hdf5')
    a=np.linspace(scales[0],scales[-1],21);t=1+2000*(a-.5);H=1000*KPC_CM/(C_SI*100*2000e9*SECONDS_PER_YEAR*a);clock=run/'background.dat'
    clock.write_text('# 1:z 2:proper time [Gyr] 3:Omega_r(z) 4:H [1/Mpc]\n'+''.join(f'{1/x-1:.17g} {y:.17g} 0 {z:.17g}\n' for x,y,z in zip(a,t,H)))
    atomic_json(run/'manifest.json',{'fixture':True,'scope':__doc__,'gadget_handoff':{'initial_conditions':[{'path':str(run/'initial_source.hdf5'),'sha256':file_sha256(run/'initial_source.hdf5')}]},'input_artifacts':{'boltzmann_background':{'path':str(clock),'sha256':file_sha256(clock),'origin':'Authored analytic clock t(a)=1+2000(a-.5) Gyr; no Boltzmann solver'}}})
    graph=IdentityGraph(maximum_nodes=16,maximum_edges=16);keys=[[],[]];radius_definition='synthetic fixed proper aperture radius 1.5 kpc; no R200 claim'
    for p in (run/'data').glob('*.hdf5'):pins[str(p)]=file_sha256(p)
    pins[str(Path(__file__).resolve())]=file_sha256(__file__)
    for i,a in enumerate(scales):
     epoch=run_epoch_contract(run,i,float(a));graph.add_epoch(i,epoch)
     for kind in (0,1):
      key=('analytic-flyby-control',kind,i);keys[kind].append(list(key));attrs={'id':kind,'position_proper_kpc':positions[i][kind].tolist(),'velocity_peculiar_kms':velocities[i][kind].tolist(),'mass_msun':16.,'radius_proper_kpc':1.5 if kind else .05,'radius_definition':radius_definition,'measurement_quality_passed':True,'measurement_quality_scope':'Authored complete 16-particle aperture; no finder resolution acceptance','synthetic_control':True}
      graph.add_node(*key,kind='halo',attributes=attrs)
      if i:graph.add_edge(('analytic-flyby-control',kind,i-1),key,'descendant',evidence={'origin':'native_ancestry','fixture':True,'scope':'Authored identity correspondence','source_hashes':pins.copy()})
    g=graph.publish_indexed_queries(run/'graph',source_hashes=pins);image=run/'images.json'
    publish_stream_product(image,{'schema':'native-catalogue-orbit-images-v1','graph_fingerprint':g['fingerprint'],'object_keys':keys[0],'host_keys':keys[1],'source_hashes':pins},{'image_offsets':np.zeros((3,2,3),dtype=np.int64)},{'image_offsets':'dimensionless'},sources=pins,namespace='analytic-flyby-control')
    request={'schema':'flagship-native-assembly-orbits-v1','run_directory':str(run),'maximum_saved_epochs':3,'maximum_interval_myr':5.1,'object_keys':keys[0],'host_keys':keys[1],'host_radius_definition':radius_definition,'object_valid_intervals':[True,True],'host_valid_intervals':[True,True],'recorded_images':{'path':str(image),'sha256':file_sha256(image)},'controls':{'maximum_orbital_radius_kpc':5.,'boundary_extremum_step_myr':.001,'event_time_deduplication_myr':.001,'include_apocenters':True}}
    atomic_json(run/'predeclared.json',dict(request=request,expected_pericenter_myr=1005.,expected_infall_myr=float(1005-np.sqrt(2)/.4),time_absolute_tolerance_myr=1e-6,scope=__doc__));product=run/'events.json';measure_native_assembly_orbits(run/'graph',request,product,memory_gib=memory_gib)
    report,arrays,_=read_product(product,memory_gib=memory_gib)
    for kind,expected in [('pericenter',1005.),('infall',1005-np.sqrt(1.5**2-.5**2)/.4)]:
     found=[r for r in report['events'] if r['kind']==kind];assert len(found)==1,(kind,report['events']);assert abs(found[0]['time_myr']-expected)<1e-6,(kind,found,expected)
    assert not report['native_mergers']
    gallery=export_native_assembly_event_figures(product,run/'figures/gallery',memory_gib=memory_gib)
    return dict(control='orbits',particles=32,seed=seed,events=report['events'],
        trajectory_certified=report.get('trajectory_certified',False),
        expected_pericenter_myr=1005.,expected_infall_myr=float(1005-np.sqrt(2)/.4),
        time_absolute_tolerance_myr=1e-6,figures=len(gallery['figures']),
        gallery='figures/gallery/gallery.json',native_solver_executed=False,
        product='events.json',scope=__doc__)
