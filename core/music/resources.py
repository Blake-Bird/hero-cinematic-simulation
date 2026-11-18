"""MUSIC hierarchy census and phase, response and solver allocation plans.

Cell/particle counts use the declared native geometry. Memory/disk figures use
bytes (MiB = 2**20 bytes), with explicit allocator/FFT allowances; estimates do
not establish measured peak memory or scientific resolution. Native boundary
operation counts are queried only when the selected caller requests them.
"""
import configparser
import ctypes
import json
import math
import re
from pathlib import Path
from core.foundation import file_sha256, canonical_digest, atomic_json

_CORE_ROOT = Path(__file__).resolve().parent.parent


def native_boundary_count_profile(build_receipt,order):
    from core.music.boundary import verify_boundary_adapter_build
    if type(order) is not int or order not in (2,4,6):
        raise ValueError('Native boundary order2/4/6 required')
    binary,record=verify_boundary_adapter_build(build_receipt)
    library=ctypes.CDLL(str(binary))
    method=library.music_boundary_count_profile
    method.argtypes=[ctypes.c_int,ctypes.POINTER(ctypes.c_uint64),ctypes.c_void_p,ctypes.c_uint64]
    method.restype=ctypes.c_int
    output=(ctypes.c_uint64*4)()
    error=ctypes.create_string_buffer(2048)
    if method(order,output,error,len(error)):
        raise ValueError(error.value.decode())
    return {'schema':'native-boundary-operation-count-profile-v1','order':order,'cubic_nodes_per_ghost_cell':int(output[0]),
        'face_pair_nodes':[int(output[i]) for i in range(1,4)],'face_endpoint_increment':0 if order==2 else 1,
        'derivation':'Actual native symbolic arithmetic; all8cubic parities and rectangular face-count certificates checked by the compiled counter',
        'build_receipt':{'path':str(Path(build_receipt).resolve()),'sha256':file_sha256(build_receipt)},'binary':record['binary'],
        'source_sha256':{str(Path(__file__).resolve()):file_sha256(__file__)}}


def boundary_workspace(coarse_shape,fine_shape,parent_offset,profile):
    """Count the native boundary tape and its existing allocation allowances.

    Each padded grid has four ghost cells per face, hence product(n+8).
    The compiled operation profile supplies arithmetic nodes per ghost/face;
    allocator allowances remain provisioning estimates, not measured RSS.
    """
    order=profile['order']
    shapes=(coarse_shape,fine_shape)
    if (type(order) is not int or order not in (2,4,6)
            or any(len(s)!=3 or any(type(n) is not int or not 4<=n<=2**20 or n%2 for n in s) for s in shapes)
            or len(parent_offset)!=3 or any(type(o) is not int or o<0 for o in parent_offset)
            or any(o+n//2>p for o,n,p in zip(parent_offset,fine_shape,coarse_shape))):
        raise ValueError('Actual even bounded native coarse/fine geometry required')
    cubic=profile['cubic_nodes_per_ghost_cell']
    faces=profile['face_pair_nodes']
    endpoint=0 if order==2 else 1
    if (type(cubic) is not int or cubic<=0 or len(faces)!=3 or any(type(n) is not int or n<0 for n in faces)
            or profile['face_endpoint_increment']!=endpoint):raise ValueError('Native arithmetic count profile required')
    cp=math.prod(n+8 for n in coarse_shape)
    fp=math.prod(n+8 for n in fine_shape)
    inputs=cp+fp
    ghost=fp-math.prod(fine_shape)
    nodes=cubic*ghost
    if any(parent_offset):
        nodes+=sum(faces[a]*(fine_shape[(a+1)%3]//2+endpoint)*(fine_shape[(a+2)%3]//2+endpoint) for a in range(3))
    if inputs>(2**63-1)//2 or inputs+nodes>2**63-1:
        raise ValueError('Native boundary node identity overflow')
    band=fp-math.prod(max(0,n-8) for n in fine_shape)
    outputs=order*sum(fine_shape[(a+1)%3]*fine_shape[(a+2)%3] for a in range(3))
    reserve=64*1024**2+128*band+32*inputs
    construction=reserve+96*max(nodes,1024)+1
    application=64*1024**2+96*nodes+40*inputs+24*outputs
    minimum=max(construction,application)
    return {'schema':'native-boundary-allocation-guard-v1','order':order,'linear_tape_nodes':nodes,
        'input_cells':inputs,'axis_stencil_ghost_rows':outputs,'construction_guard_bytes':construction,
        'application_guard_bytes':application,'minimum_workspace_bytes':minimum,
        'allocated_workspace_bytes':math.ceil(minimum/1024**2)*1024**2,
        'coarse_shape':list(coarse_shape),'fine_shape':list(fine_shape),'parent_offset':list(parent_offset),
        'scope':'Exact native operation count and existing conservative allocator guards; not measured peak RSS'}


def coupled_geometry(hierarchy,order):
    if not isinstance(hierarchy,(list,tuple)) or not hierarchy or type(order) is not int or order not in (2,4,6):
        raise ValueError('Explicit supported nested native geometry required')
    levels=[]
    shapes=[]
    origins=[]
    offsets=[]
    for row in hierarchy:
        level=row['level']
        shape=row['interior_shape']
        origin=row['origin']
        if (type(level) is not int or not 2<=level<=20 or len(shape)!=3 or len(origin)!=3
                or any(type(n) is not int or n<4 or n%2 or n>2**level for n in shape)
                or any(type(n) is not int or not 0<=n<2**level for n in origin)):
            raise ValueError('Even integer native interior geometry required')
        levels.append(level)
        shapes.append(tuple(shape))
        origins.append(tuple(origin))
        if len(levels)>1:
            if level!=levels[-2]+1 or any(o%2 for o in origin):
                raise ValueError('Consecutive aligned native refinement levels required')
            offset=tuple((o//2-p)%2**levels[-2] for o,p in zip(origin,origins[-2]))
            if any(o<4 or o+n//2+4>p for o,n,p in zip(offset,shape,shapes[-2])):
                raise ValueError('Native boundary coupling requires four parent interior cells of padding')
            offsets.append(offset)
    if shapes[0]!=(2**levels[0],)*3 or origins[0]!=(0,0,0):
        raise ValueError('Whole periodic native base required')
    return levels,shapes,origins,offsets


def coupled_workspace_plan(hierarchy,build_receipt,*,order=6,restart=32,max_iterations=128):
    levels,shapes,origins,offsets=coupled_geometry(hierarchy,order)
    if type(restart) is not int or not 2<=restart<=256 or type(max_iterations) is not int or not 1<=max_iterations<=100000:
        raise ValueError('Declared bounded Krylov restart/iterations required')
    profile=native_boundary_count_profile(build_receipt,order) if offsets else None
    interfaces=[boundary_workspace(shapes[i-1],shapes[i],offset,profile) for i,offset in enumerate(offsets,1)]
    cells=sum(math.prod(s) for s in shapes)
    base=64*1024**2+320*sum(math.prod(n+8 for n in s) for s in shapes)
    krylov=8*cells*(min(restart,max_iterations)+16)
    construction=base+sum(row['allocated_workspace_bytes'] for row in interfaces)
    application=sum(96*r['linear_tape_nodes']+40*r['input_cells']+24*r['axis_stencil_ghost_rows'] for r in interfaces)
    solve=base+krylov+application
    return {'schema':'native-coupled-poisson-allocation-plan-v1','levels':levels,'shapes':[list(s) for s in shapes],
        'origins':[list(o) for o in origins],'parent_offsets':[list(o) for o in offsets],'order':order,
        'interior_cells':cells,'base_workspace_bytes':base,'krylov_workspace_bytes':krylov,'restart':restart,'max_iterations':max_iterations,
        'constructor_workspace_bytes':construction,'solve_workspace_bytes':solve,'minimum_operator_workspace_bytes':max(construction,solve),
        'interfaces':interfaces,'operation_count_profile':profile,'interface_allocation':'Individually counted native guards; no equal split',
        'scope':'Source-derived native tape counts and current operator/Krylov guards; no field allocation or solve; allocator/FFTW RSS remains uncalibrated',
        'source_sha256':{str(Path(__file__).resolve()):file_sha256(__file__)},'science_acceptance':'NOT_ACCEPTED'}


def source_response_resources(mesh,counts,windows,*,mode_count,quantity_count,boundary):
    """Budget real response channels, retained source grids, IDs and masks.

    Each mode/window/quantity pair has two real channels. Potential scalars
    retain their declared 8- or 16-byte precision; no downcast saves memory.
    """
    from core.numeric import response_disk_bound
    if (len(counts)!=6 or any(type(n) is not int or n<0 for n in counts) or any(counts[3:])
            or type(mode_count) is not int or mode_count<1 or type(quantity_count) is not int or quantity_count<1
            or type(boundary) is not int or boundary<0 or not windows):
        raise ValueError('Exact original matter census, windows and mode/quantity counts required')
    physical=mesh['hierarchies']['poisson']
    interior=sum(math.prod(r['shape']) for r in physical)
    scalar=mesh.get('source_archive_potential_scalar_bytes',8)
    if type(scalar) is not int or scalar not in (8,16):
        raise ValueError('Explicit supported captured potential precision required')
    potential=sum(math.prod(n+8 for n in r['shape']) for r in physical)
    fine=math.prod(physical[-1]['shape'])
    wider_resident=(scalar-8)*6*potential
    raw=8*interior+6*potential
    pairs=len(windows)*quantity_count*mode_count
    channels=2*pairs
    masks=sum(counts[s] for w in windows for s in w['particle_types'])
    identity=8*sum(counts)
    metadata_arrays=sum(n>0 for n in counts[:3])+sum(len(w['particle_types']) for w in windows)
    payload=channels*2*8*(interior+potential+fine)+identity+masks
    arrays=channels*2*(2*len(physical)+1)+metadata_arrays
    side=max(physical[-1]['shape'])+2*boundary
    common=64*1024**2+960*side**3+256*sum(counts)
    return {'schema':'native-source-response-resources-v1','real_channels':channels,'raw_source_cells':raw,
        'streamed_memory_guard_bytes':common+96*raw+wider_resident,'legacy_memory_guard_bytes':common+(32+16*pairs)*raw+wider_resident,
        'captured_potential_scalar_bytes':scalar,'additional_wide_potential_resident_bytes':wider_resident,
        'payload_upper_bound_bytes':payload,'array_count_upper_bound':arrays,'disk_upper_bound_bytes':response_disk_bound(payload,arrays),
        'source_channel_payload_upper_bound_bytes':2*8*(interior+potential+fine),
        'copied_identity_mask_bytes':identity+masks,'copied_identity_mask_arrays':metadata_arrays,
        'scope':'Current allocation guards and worst-case source-coefficient publication; no measured RSS or physical acceptance'}


def noise_response_resources(mesh,channels,*,source_channel_bytes,metadata_bytes,metadata_arrays):
    from core.numeric import response_disk_bound
    if (type(channels) is not int or channels<=0 or any(type(n) is not int or n<0 for n in (source_channel_bytes,metadata_bytes,metadata_arrays))):
        raise ValueError('Positive response count and exact source/metadata byte bounds required')
    rows=mesh['hierarchies']['convolution']
    margin=mesh['convolution_margin']
    if type(margin) is not int or margin<=0 or margin%2:
        raise ValueError('Full-flow reference requires a positive even convolution margin')
    shapes={r['level']:tuple(n+(0 if i==0 else 2*margin) for n in r['shape']) for i,r in enumerate(rows)}
    cells=sum(math.prod(s) for s in shapes.values())
    payload=8*channels*cells+16*channels+metadata_bytes
    arrays=len(shapes)+2+metadata_arrays
    return {'schema':'native-noise-response-resources-v1','cache_shapes':shapes,
        'input_convolution_resident_bytes':64*1024**2+2*source_channel_bytes+64*cells,
        'payload_upper_bound_bytes':payload,'array_count_upper_bound':arrays,'disk_upper_bound_bytes':response_disk_bound(payload,arrays),
        'scope':'One source channel plus original cached noise/convolution and disk-backed responses; coupled operator allocation additional'}


def hybrid_workspace_requirements(plan,settings):
    from core.zoom import validate_bounded_source_settings
    settings=validate_bounded_source_settings(settings,lpt_order=2)
    if plan.get('schema')!='darkmattersim-music-mesh-v1' or plan.get('scalar_bytes')!=8:
        raise ValueError('Actual binary64 native hierarchy required')
    rows=plan['hierarchies']['poisson']
    if not rows or any(len(r['shape'])!=3 or any(type(n) is not int or n<2 or n%2 or n>2**r['level'] for n in r['shape']) for r in rows):
        raise ValueError('Even bounded native source hierarchy required')
    if [r['level'] for r in rows]!=list(range(rows[0]['level'],rows[-1]['level']+1)):
        raise ValueError('Complete consecutive native source hierarchy required')
    if rows[0]['shape']!=[2**rows[0]['level']]*3 or 2**rows[0]['level']<settings['hybrid_extension_degree']+1:
        raise ValueError('Supported full periodic interpolation base required')
    cells=sum(math.prod(r['shape']) for r in rows)
    side=max(rows[-1]['shape'])+2*settings['hybrid_extension_boundary']
    capture=144*cells
    fft=128*side**3
    minimum=max(capture,fft)
    supplied=settings['hybrid_extension_memory_mib']*1024**2
    feasible=side<=settings['hybrid_extension_max_side'] and supplied>=minimum
    root=_CORE_ROOT
    return {'schema':'native-bounded-hybrid-resource-estimate-v1','status':'WORKSPACE_GEOMETRY_PASS' if feasible else 'INSUFFICIENT_WORKSPACE_OR_SIDE',
        'native_geometry_digest':canonical_digest(plan),'extended_fft_side':side,'captured_source_cells':cells,
        'source_capture_required_bytes':capture,'extended_fft_required_bytes':fft,'minimum_workspace_bytes':minimum,
        'minimum_workspace_mib':(minimum+1024**2-1)//1024**2,'declared_workspace_bytes':supplied,'settings':settings,
        'requirements':'Same native three-capture/half-FFT workspace inequalities; no numerical tolerance changes',
        'source_hashes':{str(p):file_sha256(p) for p in (Path(__file__).resolve(),root/'zoom.py',root/'software/music2/src/main.cc')},
        'scope':'Native resource inequalities only; total generation allocation and measured peak memory must be checked separately',
        'science_acceptance':'NOT_ACCEPTED'}


def propose_hybrid_workspace(plan,*,degree,boundary,maximum_side):
    """Derive the minimum declared workspace from unchanged native bounds."""
    settings={'hybrid_source_extension':'bounded_conservative','hybrid_extension_degree':degree,
        'hybrid_extension_boundary':boundary,'hybrid_extension_memory_mib':1,'hybrid_extension_max_side':maximum_side}
    requirement=hybrid_workspace_requirements(plan,settings)
    settings['hybrid_extension_memory_mib']=requirement['minimum_workspace_mib']
    return settings,hybrid_workspace_requirements(plan,settings)


def estimate_music_phase_resources(plan,configuration,*,threads=1):
    """Account for actual full-prefix and bounded seeded-subvolume paths.

    Allocation payloads follow native shapes and tile visitation. Container,
    GSL and FFT-internal overheads are explicit provisioning assumptions, not
    measured peak RSS. No effective-finest full cube is allocated or needed
    by this estimator or by the supported seeded-subvolume strategy.
    """
    from core.zoom import _configuration,_music_noise_files,_music_seed_stream_ranges,_frozen_noise_workspace
    path=Path(configuration).resolve()
    cfg=_configuration(path)
    if (plan.get('schema')!='darkmattersim-music-mesh-v1' or plan.get('scalar_bytes')!=8
            or type(threads) is not int or not 1<=threads<=1024):
        raise ValueError('Actual binary64 native phase geometry and bounded threads required')
    rows=plan['hierarchies']['convolution']
    low=cfg.getint('setup','levelmin')
    high=cfg.getint('setup','levelmax')
    if [r['level'] for r in rows]!=list(range(low,high+1)) or not 1<=low<=high<=20:
        raise ValueError('Actual contiguous native convolution hierarchy required')
    cube=cfg.getint('random','cubesize',fallback=32)
    margin=plan['convolution_margin']
    if (type(cube) is not int or not 1<=cube<=2**20 or cube&(cube-1)
            or type(margin) is not int or margin< -1):
        raise ValueError('Power-of-two native raw-cube size and native convolution margin (-1, zero or positive) required')
    for row in rows:
        if (len(row['shape'])!=3 or len(row['offset'])!=3
                or any(type(n) is not int or not 2<=n<=2**row['level'] or n%2 for n in row['shape'])
                or any(type(o) is not int for o in row['offset'])):
            raise ValueError('Even bounded native phase grid dimensions required')
    if rows[0]['shape']!=[2**low]*3 or rows[0]['offset']!=[0]*3:
        raise ValueError('Complete periodic native phase base required')
    files=_music_noise_files(cfg,path)
    seeds={}
    declared=[int(match.group(1)) for key in cfg['random'] if (match:=re.fullmatch(r'seed\[(\d+)\]',key))]
    first_seed=min(declared) if declared else low
    if first_seed<0 or first_seed>low:
        raise ValueError('Explicit native generation seed base required')
    for level in range(first_seed,high+1):
        if not cfg.has_option('random',f'seed[{level}]'):
            raise ValueError('Explicit seed or retained noise file required at every native level')
        if level not in files:
            try:seeds[str(level)]=cfg.getint('random',f'seed[{level}]')
            except ValueError:
                if level>=low:
                    raise
                # Native full-prefix file loading below the grid base.
                value=Path(cfg.get('random',f'seed[{level}]')).expanduser()
                files[level]=(path.parent/value).resolve()
    if first_seed<low and any(level>first_seed for level in files):
        raise ValueError('Native full-prefix generation cannot insert a frozen file above its original seed base')
    ngenic=cfg.getboolean('random','coarse_ngenic',fallback=False)
    streams=_music_seed_stream_ranges(seeds,noise_levels=files,cube_size=cube,coarse_ngenic=ngenic)
    frozen_validation=_frozen_noise_workspace(cfg,{level:p for level,p in files.items() if level>=low})
    disk=cfg.getboolean('random','disk_cached',fallback=True)
    row_records=[]
    cache_cells=0
    cache_disk=0
    cache_peak=0
    previous_tiles=0
    working_peak=0
    for ordinal,row in enumerate(rows):
        level=row['level']
        side=2**level
        full=ordinal==0 or level in files
        shape=tuple(row['shape']) if ordinal==0 else tuple(n+2*margin if margin>0 else 2*n for n in row['shape'])
        if any(n%2 for n in shape):
            raise ValueError('Whole-parent aligned even native phase FFT sizes required')
        cells=math.prod(shape)
        cache_cells+=cells
        cache_disk+=12+8*cells
        # fill_subvolume registers floor(lx/cubesize)+2 visits on each axis;
        # periodic aliases are scheduled once by the native visits map.
        tile_side=min(cube,side)
        global_tiles=side//tile_side
        unique_axes=[global_tiles]*3 if full else [min(global_tiles,n//tile_side+2) for n in shape]
        tiles=math.prod(unique_axes)
        tile_cells=side**3 if full else tiles*tile_side**3
        # File-backed generators store one full Meshvar, whereas seeded
        # generators keep one Meshvar per unique registered raw cube.
        objects=1 if level in files or ordinal==0 and ngenic else tiles
        tile_bytes=8*tile_cells+256*objects
        # Cubemap, visit map, jobs and vector growth reserve: explicit fixed
        # per-object allowance; not a claim about a particular STL allocator.
        container_bytes=512*objects
        fine_fft=coarse_fft=0
        if not full:
            fine_fft=8*shape[0]*shape[1]*(shape[2]+2)
            half=tuple(n//2 for n in shape)
            coarse_fft=8*half[0]*half[1]*(half[2]+2)
        elif ordinal==0 and ngenic:
            fine_fft=8*side*side*(side+2)
            coarse_fft=4*side*side
        fft_auxiliary=fine_fft+coarse_fft
        cache_payload=8*cells
        storage_workspace=8*shape[1]*shape[2] if disk else 8*cache_cells
        # Disk reload currently reads a complete ny*nz slab. In-memory
        # caches retain all levels until the density convolution consumes them.
        cache_peak=max(cache_peak,storage_workspace)
        gsl_workspace=threads*65536
        prefix_workspace=frozen_validation if level in files else 0
        payload=previous_tiles+tile_bytes+container_bytes+fine_fft+coarse_fft+fft_auxiliary+storage_workspace+gsl_workspace+prefix_workspace
        working_peak=max(working_peak,payload)
        row_records.append({'level':level,'strategy':'FULL_RETAINED_FILE' if level in files else 'FULL_BASE_NGENIC' if ordinal==0 and ngenic else 'FULL_BASE_TILED' if ordinal==0 else 'SEEDED_SUBVOLUME',
            'effective_periodic_side':side,'cache_shape':list(shape),'cache_payload_bytes':cache_payload,'cache_file_bytes':12+cache_payload,
            'allocated_noise_cells':tile_cells,'unique_raw_tiles':0 if level in files or ordinal==0 and ngenic else tiles,'raw_seed_tile_side':tile_side,'tile_side':side if level in files or ordinal==0 and ngenic else tile_side,
            'allocated_mesh_objects':objects,'raw_tile_payload_bytes':8*tile_cells,'tile_object_allowance_bytes':256*objects,'container_allowance_bytes':container_bytes,
            'fine_fft_payload_bytes':fine_fft,'coarse_fft_payload_bytes':coarse_fft,'fft_auxiliary_allowance_bytes':fft_auxiliary,
            'retained_parent_tile_bytes':previous_tiles,'cache_workspace_bytes':storage_workspace,'gsl_thread_allowance_bytes':gsl_workspace,
            'frozen_prefix_validation_allowance_bytes':prefix_workspace,'estimated_phase_working_set_bytes':payload,
            'full_effective_finest_field_constructed':full and level==high})
        previous_tiles=tile_bytes+container_bytes
    # Native coarse-seed upsampling fills whole fields only through the
    # actual generation base. Its temporary full-prefix peak is separate
    # from the fine local-subvolume refinement peak.
    lower_prefix_peak=0
    for level in range(first_seed,low+1) if first_seed<low else ():
        n=2**level
        t=min(cube,n)
        objects=(n//t)**3
        current=8*n**3+768*objects
        parent=0 if level==first_seed else 8*(n//2)**3+768*(max(1,n//2//cube))**3
        fft=0 if level==first_seed else 8*n*n*(n+2)+8*(n//2)**2*(n//2+2)
        if level==first_seed and ngenic:
            fft=8*n*n*(n+2)+4*n*n
        lower_prefix_peak=max(lower_prefix_peak,current+parent+2*fft+threads*65536)
    working_peak=max(working_peak,lower_prefix_peak)
    root=_CORE_ROOT
    sources=[Path(__file__).resolve(),path,root/'zoom.py',root/'software/music2/src/plugins/random_music.cc',
        root/'software/music2/src/plugins/random_music_wnoise_generator.cc',root/'software/music2/src/plugins/random_music_wnoise_generator.hh',root/'software/music2/src/mesh.hh']
    highest_file=max(files) if files else None
    return {'schema':'native-music-phase-resource-estimate-v1','status':'ESTIMATED','levels':row_records,
        'native_geometry_digest':canonical_digest(plan),'generation_base_level':low,'raw_seed_base_level':first_seed,'lower_seed_full_prefix_peak_bytes':lower_prefix_peak,'effective_level':high,'highest_full_file_level':highest_file,
        'strategy':'Explicit full generation base/retained prefix followed by seeded local subvolume FFT refinement',
        'disk_cached':disk,'phase_peak_working_set_bytes':64*1024**2+working_peak,'cache_storage_bytes':cache_disk if disk else 0,
        'cache_all_level_payload_bytes':8*cache_cells,'cache_peak_workspace_bytes':cache_peak,'raw_seed_streams':streams,
        'uniform_effective_finest_field_required':bool(high in files or low==high),
        'source_hashes':{str(p):file_sha256(p) for p in sources},
        'provisioning_assumptions':{'tile_object_bytes':256,'container_bytes_per_unique_tile':512,'gsl_bytes_per_thread':65536,'fft_auxiliary_allowance':'One additional fine/coarse in-place FFT payload; measured FFTW peak remains required'},
        'scope':'Allocation-payload accounting plus explicit uncalibrated container/GSL/FFT overhead allowances; no fields generated and no measured runtime/science acceptance',
        'science_acceptance':'NOT_ACCEPTED'}


def publish_music_phase_resources(request,destination):
    """Size pinned retained native geometry without generating any field."""
    import json
    from core.foundation import atomic_json
    from core.zoom import estimate_mesh_resources
    if (not isinstance(request,dict) or set(request)!={'schema','cases','memory_gib','threads'}
            or request['schema']!='native-phase-resource-request-v1'
            or isinstance(request['memory_gib'],bool) or not math.isfinite(request['memory_gib']) or request['memory_gib']<=0
            or type(request['threads']) is not int or not 1<=request['threads']<=1024
            or not isinstance(request['cases'],list) or not request['cases']):
        raise ValueError('Explicit native phase resource request required')
    destination=Path(destination).resolve()
    if destination.exists():
        raise FileExistsError('Keep previous native phase planning')
    pins={}
    rows=[]
    identities=set()
    def bound(ref):
        if not isinstance(ref,dict) or set(ref)!={'path','sha256'}:
            raise ValueError('Pinned phase configuration/mesh required')
        p=Path(ref['path']).resolve()
        if file_sha256(p)!=ref['sha256']:
            raise ValueError('Native phase planning input changed')
        pins[str(p)]=ref['sha256']
        return p
    for case in request['cases']:
        if set(case)!={'id','configuration','mesh'} or not isinstance(case['id'],str) or not case['id'] or case['id'] in identities:
            raise ValueError('Unique explicit phase-planning case identity required')
        identities.add(case['id'])
        configuration=bound(case['configuration'])
        mesh_path=bound(case['mesh'])
        mesh=json.loads(mesh_path.read_text())
        phase=estimate_music_phase_resources(mesh,configuration,threads=request['threads'])
        bulk=estimate_mesh_resources(mesh)
        pins.update(phase['source_hashes'])
        combined=bulk['estimated_memory_bytes']+phase['phase_peak_working_set_bytes']
        rows.append({'id':case['id'],'configuration':case['configuration'],'mesh':case['mesh'],'phase':phase,
            'bulk_mesh_memory_estimate_bytes':bulk['estimated_memory_bytes'],'combined_mesh_phase_memory_estimate_bytes':combined,
            'combined_mesh_phase_storage_estimate_bytes':bulk['estimated_disk_bytes']+max(0,phase['cache_storage_bytes']-bulk['noise_cache_bytes']),
            'allocation_status':'EXCEEDS_DECLARED_MEMORY' if combined>request['memory_gib']*1024**3 else 'ESTIMATE_WITHIN_DECLARED_MEMORY'})
    for p,digest in pins.items():
        if file_sha256(p)!=digest:
            raise ValueError('Phase planning dependency changed during sizing')
    result={'schema':'native-phase-resource-report-v1','status':'EXCEEDS_DECLARED_MEMORY' if any(r['allocation_status']=='EXCEEDS_DECLARED_MEMORY' for r in rows) else 'ESTIMATED',
        'cases':rows,'declared_memory_gib':request['memory_gib'],'threads':request['threads'],'source_hashes':pins,
        'scope':'Source-bound phase allocation estimates on pinned retained native geometry. Additive bulk mesh allowances can overlap phase payloads. Hybrid correction/output/sharding/analysis costs and measured RSS remain separate; no fresh native planning, fields or evolution executed.',
        'actual_hardware_allocation':None,'launch_authorized':False,'science_acceptance':'NOT_ACCEPTED'}
    atomic_json(destination,result)
    return result


def phase_resources_main():
    """Plan native tiled phase memory and cache storage from retained geometry."""
    import argparse,json
    parser=argparse.ArgumentParser(description=phase_resources_main.__doc__)
    parser.add_argument('request',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=publish_music_phase_resources(json.loads(args.request.read_text()),args.output)
    print(json.dumps(result,indent=2))
    return 2 if result['status']=='EXCEEDS_DECLARED_MEMORY' else 0


def rectangular_particle_census(plan, configuration):
    path=Path(configuration).resolve()
    cfg=configparser.ConfigParser(interpolation=None)
    cfg.optionxform=str
    if not cfg.read(path):
        raise FileNotFoundError(path)
    if cfg.get('setup','region',fallback='box')!='box' or cfg.get('output','format')!='arepo':
        return {'status':'UNSUPPORTED','reason':'Exact census requires the native all-points box mask and Arepo output; retain conservative bounds'}
    coarse=cfg.getint('output','arepo_coarsetype',fallback=2)
    if coarse!=2:
        raise ValueError('Selected census contract reserves type 2 for combined coarse matter')
    if plan.get('schema')!='darkmattersim-music-mesh-v1':
        raise ValueError('Exact census requires actual native mesh geometry')
    rows=plan['hierarchies']['poisson']
    low=cfg.getint('setup','levelmin')
    high=cfg.getint('setup','levelmax')
    if not 1<=low<=high<=20 or [row.get('level') for row in rows]!=list(range(low,high+1)):
        raise ValueError('Native census levels disagree with the actual configuration')
    for row in rows:
        for key in ('shape','offset'):
            value=row.get(key)
            if not isinstance(value,list) or len(value)!=3 or any(type(n) is not int for n in value):
                raise ValueError('Native census requires exact integer geometry')
        if any(n<=0 or n%2 or n>2**row['level'] for n in row['shape']):
            raise ValueError('Native census requires bounded even grid dimensions')
    if rows[0]['shape']!=[2**low]*3 or rows[0]['offset']!=[0,0,0]:
        raise ValueError('Native census base must cover the complete periodic cube')
    def product(values):
        result=1
        for n in values:
            result*=n
        return result
    leaves=[]
    for index,row in enumerate(rows):
        cells=product(row['shape'])
        covered=0
        if index+1<len(rows):
            child=rows[index+1]
            relative=[]
            for axis in range(3):
                delta=child['offset'][axis]-2*row['offset'][axis]
                if delta%2:
                    raise ValueError('Native child grid is not aligned to whole parent cells')
                start=delta//2
                length=child['shape'][axis]//2
                if start<0 or start+length>row['shape'][axis]:
                    raise ValueError('Native child grid is not contained in its parent; exact census refused')
                relative.append(length)
            covered=product(relative)
        leaves.append({'level':row['level'],'grid_cells':cells,'covered_parent_cells':covered,'leaf_cells':cells-covered})
    # Independent integer volume closure at the finest-grid lattice. This is a
    # partition check, not a floating-point tolerance or a particle-mass fit.
    volume=sum(row['leaf_cells']*8**(high-row['level']) for row in leaves)
    if volume!=8**high:
        raise ValueError('Native census does not partition the periodic material volume')
    fine=leaves[-1]['leaf_cells']
    coarse_cells=sum(row['leaf_cells'] for row in leaves[:-1])
    counts={str(i):0 for i in range(6)}
    counts['1']=fine
    counts['2']=coarse_cells
    if cfg.getboolean('setup','baryons'):
        counts['0']=fine
    total=sum(counts.values())
    if total>=2**64:
        raise OverflowError('Native particle census exceeds exact uint64 identities')
    source=_CORE_ROOT/'software/music2/src'
    algorithm_sources={str(p):file_sha256(p) for p in (Path(__file__).resolve(),source/'mesh.hh',source/'region_generator.cc',source/'plugins/output_arepo.cc')}
    return {'schema':'music-rectangular-particle-census-v1','status':'EXACT_GEOMETRY_CENSUS',
        'species_counts':counts,'total_particle_count':total,'leaf_cells_by_level':leaves,
        'periodic_volume_in_finest_cells':volume,'configuration_sha256':file_sha256(path),
        'native_geometry_digest':canonical_digest(plan),'algorithm_sources':algorithm_sources,
        'scope':'Exact count implied by the actual native unmasked box/Arepo hierarchy; emitted IC headers remain the runtime census authority',
        'native_fields_generated':False,'science_acceptance':'NOT_ACCEPTED'}


def native_header_count_requirements(census,configuration):
    """Bind the exact header encoding to native writer/reader source contracts."""
    cfg=configparser.ConfigParser(interpolation=None)
    cfg.optionxform=str
    path=Path(configuration).resolve()
    if not cfg.read(path):
        raise FileNotFoundError(path)
    if census.get('status')!='EXACT_GEOMETRY_CENSUS':
        raise ValueError('Exact native census required for header capacity')
    counts=[census['species_counts'][str(i)] for i in range(6)]
    if any(type(n) is not int or n<0 for n in counts) or sum(counts)>2**63-1:
        raise OverflowError('Native global particle count exceeds signed 64-bit storage')
    modern=cfg.get('cosmology','transfer',fallback='')=='DMS_file'
    files=cfg.getint('output','arepo_num_files',fallback=1)
    if files<1:
        raise ValueError('Positive native IC file count required')
    if modern and files!=1:
        raise ValueError('Native mixed-fluid mass/momentum path currently requires one full-count file')
    local=[n//files+(n%files if files>1 else 0) for n in counts]
    root=_CORE_ROOT
    sources=[Path(__file__).resolve(),root/'software/music2/src/particle_count_header.hh',root/'software/music2/src/plugins/output_arepo.cc',
        root/'software/gadget4/src/io/snap_io.cc',root/'software/gadget4/src/io/snap_io.h']
    reader=sources[-2].read_text()
    if ('"NumPart_ThisFile", header.npart, H5T_NATIVE_UINT64' not in reader
            or '"NumPart_Total", header.npartTotal, H5T_NATIVE_UINT64' not in reader):
        raise ValueError('Native GADGET-4 full-count header reader changed')
    supported=modern or max(local)<=2**32-1
    return {'schema':'native-music-header-count-capacity-v1','status':'SUPPORTED_ENCODING' if supported else 'UNSUPPORTED_LEGACY_PER_FILE_COUNTS',
        'encoding':'FULL_UINT64_COUNTS_ZERO_HIGH_WORDS' if modern else 'CLASSIC_UINT32_LOCAL_SPLIT_TOTALS',
        'per_file_count_bits':64 if modern else 32,'species_total_counts':counts,'number_of_files':files,
        'maximum_local_species_count':max(local),'requires_uint64_local_counts':max(local)>2**32-1,
        'reader_requirement':'GADGET-4 HDF5 build with GADGET2_HEADER disabled' if modern else 'Compatible legacy split-total header reader',
        'source_hashes':{str(p):file_sha256(p) for p in sources},
        'scope':'Exact count representability and source-level header contract; no native IC created or GADGET read executed',
        'science_acceptance':'NOT_ACCEPTED'}


def estimate_generation_handoff(mesh,configuration,boundary_build,*,modes,solve,threads,provisioning_factor=1.25):
    """Combine geometry-based generation and response allocations in bytes.

    Stages run sequentially, so memory is their maximum. Their retained files
    coexist, so disk is their sum. Proposed GiB allocations round upward after
    the explicit provisioning factor; they are estimates, not measured RSS.
    Evolution and saved snapshots use separate caller-supplied assumptions.
    """
    from core.zoom import _configuration,_modes,_music_noise_files,_frozen_noise_workspace,estimate_mesh_resources
    from core.music.seeded import patch_geometry, estimate_seeded_moments
    cfg=_configuration(configuration)
    wave=_modes(modes)
    if (not isinstance(solve,dict) or set(solve)!={'restart','max_iterations'} or type(threads) is not int or not 1<=threads<=1024
            or isinstance(provisioning_factor,bool) or not isinstance(provisioning_factor,(float,int)) or not math.isfinite(provisioning_factor) or provisioning_factor<1):
        raise ValueError('Explicit solver/thread settings and provisioning factor at least one required')
    import numpy as np
    if np.linalg.matrix_rank(wave.astype(float))!=3:
        raise ValueError('Handoff modes must span all three spatial axes')
    low=cfg.getint('setup','levelmin')
    high=cfg.getint('setup','levelmax')
    if (not low<high or cfg.getint('setup','levelmin_TF',fallback=low)!=low or not cfg.getboolean('setup','baryons')
            or cfg.get('cosmology','transfer')!='DMS_file' or not cfg.getboolean('setup','use_2LPT')
            or not cfg.getboolean('cosmology','mixed_2lpt') or not cfg.getboolean('output','retain_source_hierarchy',fallback=False)
            or not cfg.getboolean('output','arepo_doubleprec',fallback=True) or not cfg.getboolean('output','arepo_longids',fallback=True)
            or cfg.getint('poisson','grad_order',fallback=6) not in (2,4,6)):
        raise ValueError('Original nested DMS mixed2LPT gas/DM route with retained source hierarchy required')
    if not mesh.get('source_archive_estimated_bytes'):
        raise ValueError('Actual retained source-archive estimate required in native mesh')
    census=rectangular_particle_census(mesh,configuration)
    if census['status']!='EXACT_GEOMETRY_CENSUS':
        raise ValueError('Exact actual native particle census required')
    counts=[census['species_counts'][str(i)] for i in range(6)]
    header=native_header_count_requirements(census,configuration)
    if header['status']!='SUPPORTED_ENCODING':
        raise ValueError('Native output counts exceed supported header contract')
    coarse=[r['level'] for r in census['leaf_cells_by_level'][:-1] if r['leaf_cells']]
    windows=[{'id':'species0-level'+str(high),'particle_types':[0],'refinement_levels':[high]},
        {'id':'species1-level'+str(high),'particle_types':[1],'refinement_levels':[high]}]
    windows += [{'id':'species2-level'+str(level),'particle_types':[2],'refinement_levels':[level]} for level in coarse]
    physical=mesh['hierarchies']['poisson']
    boundary=cfg.getint('poisson','hybrid_extension_boundary',fallback=32)
    source=source_response_resources(mesh,counts,windows,mode_count=len(wave),quantity_count=2,boundary=boundary)
    noise=noise_response_resources(mesh,source['real_channels'],source_channel_bytes=source['source_channel_payload_upper_bound_bytes'],
        metadata_bytes=source['copied_identity_mask_bytes'],metadata_arrays=source['copied_identity_mask_arrays'])
    hierarchy=[{'level':r['level'],'origin':r['offset'],'interior_shape':r['shape']} for r in physical]
    operator=coupled_workspace_plan(hierarchy,boundary_build,order=cfg.getint('poisson','laplace_order',fallback=6),**solve)
    phase=estimate_music_phase_resources(mesh,configuration,threads=threads)
    bulk=estimate_mesh_resources(mesh)
    files=_music_noise_files(cfg,configuration)
    frozen=_frozen_noise_workspace(cfg,files)
    settings={'hybrid_source_extension':cfg.get('poisson','hybrid_source_extension')}
    for name,default in (('hybrid_extension_degree',5),('hybrid_extension_boundary',32),('hybrid_extension_memory_mib',2048),('hybrid_extension_max_side',256)):
        settings[name]=cfg.getint('poisson',name,fallback=default)
    hybrid=hybrid_workspace_requirements(mesh,settings)
    if hybrid['status']!='WORKSPACE_GEOMETRY_PASS':
        raise ValueError('Native hybrid workspace or side is insufficient')
    generation=bulk['estimated_memory_bytes']+phase['phase_peak_working_set_bytes']+settings['hybrid_extension_memory_mib']*1024**2+frozen
    strategies=[r['strategy'] for r in phase['levels']]
    if strategies[0]=='FULL_RETAINED_FILE' and all(s=='SEEDED_SUBVOLUME' for s in strategies[1:]):
        # Translation is irrelevant to resource counts. These are not original
        # phase identities: native cache origins must be bound after generation.
        geometry=patch_geometry(mesh,[0.,0.,0.])
        covariance=estimate_seeded_moments(geometry,source['real_channels'])
        covariance_memory=covariance['peak_workspace_bytes']+24*covariance['local_cache_cells']
        covariance_disk=covariance['innovation_disk_bytes']
        conditioning='fixed_prefix'
    elif all(s=='FULL_RETAINED_FILE' for s in strategies):
        full_cells=sum((2**r['level'])**3 for r in mesh['hierarchies']['convolution'])
        covariance_memory=64*1024**2+(32*source['real_channels']+640)*full_cells
        covariance_disk=0
        conditioning='fixed'
        covariance={'strategy':'Verified full-frozen ensemble with explicit fixed base','full_cube_levels':[r['level'] for r in mesh['hierarchies']['convolution']]}
    else:raise ValueError('Prepare a verified complete full-frozen or full-base-file/local-seeded phase strategy')
    stages={'native_generation':generation,'source_response':source['streamed_memory_guard_bytes'],
        'coupled_noise_response':noise['input_convolution_resident_bytes']+operator['minimum_operator_workspace_bytes'],
        'window_covariance':covariance_memory}
    # Eight binary64 vector datasets plus widths/masses/uint64 IDs in the actual
    # mixed-fluid writer. Headers/chunk indices use an explicit uncalibrated
    # allowance. Three copies retain the existing conservative output policy.
    particle_payload=216*sum(counts)
    ic_file=math.ceil(particle_payload*1.02)+128*1024**2
    generation_disk=max(bulk['estimated_disk_bytes']+max(0,phase['cache_storage_bytes']-bulk['noise_cache_bytes']),
        3*ic_file+phase['cache_storage_bytes']+mesh['source_archive_estimated_bytes'])
    retained_disk=generation_disk+source['disk_upper_bound_bytes']+noise['disk_upper_bound_bytes']+covariance_disk
    proposed={name:math.ceil(value/1024**3*provisioning_factor) for name,value in stages.items()}
    return {'schema':'native-generation-handoff-capacity-v1','generation_base_level':low,'effective_level':high,'particle_census':census,'header_count_contract':header,
        'windows':windows,'modes':wave.tolist(),'real_channels':source['real_channels'],'parent_conditioning':conditioning,
        'phase':phase,'hybrid':hybrid,'source_response':source,'noise_response':noise,'coupled_operator':operator,'covariance':covariance,
        'stage_memory_bytes':stages,'stage_proposed_memory_gib':proposed,'sequential_peak_memory_bytes':max(stages.values()),
        'proposed_shared_host_application_memory_gib':max(proposed.values()),'retained_pre_evolution_disk_bytes':retained_disk,
        'proposed_pre_evolution_disk_gib':math.ceil(retained_disk/1024**3*provisioning_factor),
        'generation_disk_bytes':generation_disk,'native_ic_payload_bytes':particle_payload,'native_ic_file_allowance_bytes':ic_file,
        'response_workspace':{'source_disk_gib':math.ceil(source['disk_upper_bound_bytes']/1024**3*provisioning_factor),
            'noise_disk_gib':math.ceil(noise['disk_upper_bound_bytes']/1024**3*provisioning_factor)},
        'seeded_covariance_disk_gib':math.ceil(covariance_disk/1024**3*provisioning_factor) if covariance_disk else None,
        'provisioning_assumptions':{'factor':provisioning_factor,'ic_chunk_header_fraction':.02,'ic_fixed_header_bytes':128*1024**2,
            'memory':'Maximum of sequential stages; application allocation on one shared-memory host, not aggregate MPI RAM',
            'disk':'Additive retained workspaces/publications plus generation outputs; evolution/catalogues/rendering additional'},
        'phase_origins':'NOT_BOUND; unshifted computational coordinates used only for resource counts',
        'actual_hardware_allocation':None,'measured_peak_rss':None,'estimated_wall_seconds':None,
        'scope':'Supported code-path and allocation-guard estimates on actual geometry; no native fields, handoff acceptance, evolution or science validation',
        'launch_authorized':False,'science_acceptance':'NOT_ACCEPTED'}


def publish_generation_capacity(request,destination):
    from core.music.boundary import verify_boundary_adapter_build
    from core.zoom import _configuration,_music_noise_files
    if (not isinstance(request,dict) or set(request)-{'evolution_resources'}!={'schema','cases','boundary_adapter_build','memory_gib','disk_gib','threads','provisioning_factor'}
            or request['schema']!='native-generation-handoff-capacity-request-v1' or not isinstance(request['cases'],list) or not request['cases']
            or any(isinstance(request[k],bool) or not isinstance(request[k],(int,float)) or not math.isfinite(request[k]) or request[k]<=0 for k in ('memory_gib','disk_gib'))):
        raise ValueError('Explicit pinned generation/handoff resource request required')
    evolution=request.get('evolution_resources')
    if evolution is not None:
        if (not isinstance(evolution,dict) or set(evolution)!={'evolution_bytes_per_particle','snapshot_bytes_per_particle','saved_epochs','checkpoint_copies','aggregate_memory_gib'}
                or any(type(evolution[k]) is not int or evolution[k]<=0 for k in ('evolution_bytes_per_particle','snapshot_bytes_per_particle','saved_epochs','checkpoint_copies'))
                or isinstance(evolution['aggregate_memory_gib'],bool) or not isinstance(evolution['aggregate_memory_gib'],(int,float))
                or not math.isfinite(evolution['aggregate_memory_gib']) or evolution['aggregate_memory_gib']<=0):
            raise ValueError('Explicit evolution payload/epoch/checkpoint and aggregate-memory assumptions required')
    target=Path(destination).resolve()
    if target.exists():
        raise FileExistsError('Preserve earlier production capacity plans')
    pins={}
    def bound(ref):
        if not isinstance(ref,dict) or set(ref)!={'path','sha256'}:
            raise ValueError('Exact capacity input binding required')
        path=Path(ref['path']).resolve()
        if file_sha256(path)!=ref['sha256']:
            raise ValueError('Capacity planning input changed')
        pins[str(path)]=ref['sha256']
        return path
    build=bound(request['boundary_adapter_build'])
    binary,compiled=verify_boundary_adapter_build(build)
    pins.update(compiled['source_sha256'])
    pins[str(binary)]=file_sha256(binary)
    cases=[]
    names=set()
    for case in request['cases']:
        if (not isinstance(case,dict) or set(case)!={'id','configuration','mesh','modes','solve'} or not isinstance(case['id'],str)
                or not case['id'] or case['id'] in names):raise ValueError('Unique exact capacity case and actual modes/solver required')
        names.add(case['id'])
        configuration=bound(case['configuration'])
        mesh_path=bound(case['mesh'])
        mesh=json.loads(mesh_path.read_text())
        estimate=estimate_generation_handoff(mesh,configuration,build,modes=case['modes'],solve=case['solve'],threads=request['threads'],provisioning_factor=request['provisioning_factor'])
        if evolution is not None:
            from core.resources import periodic_image_resource_allowance
            from core.native import native_particle_index_requirement
            count=estimate['particle_census']['total_particle_count']
            images=periodic_image_resource_allowance(count)
            memory=count*evolution['evolution_bytes_per_particle']+images
            saved=(count*evolution['snapshot_bytes_per_particle']+images)*(evolution['saved_epochs']+evolution['checkpoint_copies'])
            indexing=native_particle_index_requirement(count)
            pins.update(indexing['source_hashes'])
            estimate['evolution_requirements']={'aggregate_memory_estimate_bytes':memory,
                'proposed_aggregate_memory_gib':math.ceil(memory/1024**3*request['provisioning_factor']),
                'retained_saved_checkpoint_bytes':saved,'particle_indexing_necessary_floor':indexing,
                'assumptions':evolution,'scope':'Aggregate evolution memory is separate from the single-host handoff allocation; rank/node placement and actual species census remain required'}
        from core.evolution import _noise_shape
        for level,path in _music_noise_files(_configuration(configuration),configuration).items():
            _noise_shape(path,2**level)
            pins[str(path)]=file_sha256(path)
        pins.update(estimate['phase']['source_hashes'])
        pins.update(estimate['hybrid']['source_hashes'])
        pins.update(estimate['header_count_contract']['source_hashes'])
        pins.update(estimate['particle_census']['algorithm_sources'])
        estimate.update(id=case['id'],configuration=case['configuration'],mesh=case['mesh'],
            allocation_status='EXCEEDS_DECLARED_BUDGETS' if estimate['sequential_peak_memory_bytes']>request['memory_gib']*1024**3
                or estimate['retained_pre_evolution_disk_bytes']>request['disk_gib']*1024**3 else 'ESTIMATE_WITHIN_DECLARED_BUDGETS')
        cases.append(estimate)
    for name in (
        'music/resources.py',
        'music/boundary.py',
        'music/poisson.py',
        'music/windows.py',
        'music/flow.py',
        'music/covariance.py',
        'music/seeded.py',
        'numeric.py',
        'music/replay.py',
        'data.py',
        'foundation.py',
    ):
        path=(_CORE_ROOT / name)
        pins[str(path)]=file_sha256(path)
    for path,checksum in pins.items():
        if file_sha256(path)!=checksum:
            raise ValueError('Capacity dependency changed during planning')
    disk=sum(c['retained_pre_evolution_disk_bytes'] for c in cases)
    report={'schema':'native-generation-handoff-capacity-report-v1','status':'REQUIRES_LARGER_ALLOCATION' if any(c['allocation_status']=='EXCEEDS_DECLARED_BUDGETS' for c in cases)
        or disk>request['disk_gib']*1024**3 else 'ESTIMATED','cases':cases,'source_hashes':pins,
        'sequential_case_peak_proposed_memory_gib':max(c['proposed_shared_host_application_memory_gib'] for c in cases),
        'campaign_retained_pre_evolution_disk_bytes':disk,'campaign_proposed_pre_evolution_disk_gib':math.ceil(disk/1024**3*request['provisioning_factor']),
        'declared_memory_gib':request['memory_gib'],'declared_disk_gib':request['disk_gib'],'case_schedule':'Sequential; retaining all case products',
        'scope':'Generation plus original-window handoff planning. Evolved outputs, catalogues, all102figures and rendering are additional campaign stages; estimates do not certify numerical or scientific acceptance.',
        'actual_hardware_allocation':None,'science_acceptance':'NOT_ACCEPTED','launch_authorized':False}
    if evolution is not None:
        saved=sum(c['evolution_requirements']['retained_saved_checkpoint_bytes'] for c in cases)
        report['evolution_resources']=evolution
        report['campaign_retained_saved_checkpoint_bytes']=saved
        report['campaign_generation_handoff_evolution_disk_bytes']=disk+saved
        report['campaign_proposed_generation_handoff_evolution_disk_gib']=math.ceil((disk+saved)/1024**3*request['provisioning_factor'])
        report['proposed_aggregate_evolution_memory_gib']=max(c['evolution_requirements']['proposed_aggregate_memory_gib'] for c in cases)
        report['scope']='Generation/handoff and explicit aggregate-evolution/saved-output resource assumptions. Native catalogues, downstream measurements,102graph exports and rendering remain additional campaign products; no execution or science acceptance.'
        if disk+saved>request['disk_gib']*1024**3 or any(c['evolution_requirements']['aggregate_memory_estimate_bytes']>evolution['aggregate_memory_gib']*1024**3 for c in cases):
            report['status']='REQUIRES_LARGER_ALLOCATION'
    atomic_json(target,report)
    return report
