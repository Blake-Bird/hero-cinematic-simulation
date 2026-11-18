"""Read native image counters and bind them to exact measured worldline rows."""
from pathlib import Path
import numpy as np
from .foundation import file_sha256

FIELDS = {'PeriodicImageVersion':('version',np.dtype('uint64'),1),
          'PeriodicImageStatus':('status',np.dtype('uint64'),1),
          'PeriodicImageAnchorNativeTime':('anchor_native_time',np.dtype('float64'),1),
          'PeriodicImageOffsets':('image_offsets',np.dtype('int64'),3)}


def read_native_image_state(group, *, start, stop, memory_bytes):
    """Missing legacy state is unavailable; partial/lossy state is rejected."""
    if type(start) is not int or type(stop) is not int or not 0<=start<=stop<=len(group['ParticleIDs']):
        raise ValueError('Bounded exact native image-state slice required')
    if type(memory_bytes) is not int or memory_bytes<4096+(stop-start)*128:
        raise MemoryError('Native image-state slice exceeds its workspace')
    present=set(FIELDS).intersection(group)
    if not present:return None
    if present!=set(FIELDS):raise ValueError('Partial native periodic image state')
    count=len(group['ParticleIDs']);arrays={}
    for name,(key,dtype,width) in FIELDS.items():
        dataset=group[name];shape=(count,width) if width>1 else (count,)
        if dataset.dtype!=dtype or dataset.shape!=shape:
            raise ValueError('Native image state dtype/shape mismatch: '+name)
        expected={'to_cgs':1.,'a_scaling':0.,'h_scaling':0.,'length_scaling':0.,'mass_scaling':0.,'velocity_scaling':0.}
        if any(attr not in dataset.attrs or float(dataset.attrs[attr])!=value for attr,value in expected.items()):
            raise ValueError('Native image state must use dimensionless integer/native-clock metadata: '+name)
        arrays[key]=dataset[start:stop]
    if np.any(arrays['version']>1) or np.any(arrays['status']>4) or not np.isfinite(arrays['anchor_native_time']).all():
        raise ValueError('Unsupported native image state schema, status or anchor')
    inactive=arrays['version']==0
    if any(np.any(value[inactive]!=0) for key,value in arrays.items() if key!='version'):
        raise ValueError('Unversioned native image state carries recorded values')
    arrays['supported']=(arrays['version']==1)&(arrays['status']==1)
    return arrays


def publish_native_worldline_images(history, destination, *, memory_gib):
    """Join actual per-drift counters to a sealed history, without unwrapping."""
    import h5py
    from core.stream_common import read_product, publish_stream_product
    from .data import exact_uint64_ids,numeric_audit_source_bindings,native_box_proper_kpc
    from .halos import _snapshot_files
    from .flagship_products import require_native_worldline_units
    report,values,sources=read_product(history,memory_gib=memory_gib/4)
    if report.get('schema')!='measured-native-stream-worldlines-v1' or report.get('periodic') is not True:
        raise ValueError('Measured original periodic native worldlines required')
    flags=report.get('native_build_flags',[])
    if any(str(flag).split('=',1)[0] in {'LONG_X_BITS','LONG_Y_BITS','LONG_Z_BITS','GRAVITY_TALLBOX'} for flag in flags):
        raise ValueError('Encounter image sidecar requires a cubic fully periodic box; native open/stretched axes need an explicit axis-length frame adapter')
    require_native_worldline_units(report)
    sources={str(Path(name).resolve()):digest for name,digest in sources.items()}
    for name,digest in numeric_audit_source_bindings(report).items():
        name=str(Path(name).resolve())
        if name in sources and sources[name]!=digest:raise ValueError('Conflicting native image history source bindings')
        sources[name]=digest
    snapshots=report['request']['snapshots'];species=report['request']['species']
    ids=exact_uint64_ids(values['particle_ids']);times=values['times_myr'];boxes=values['periodic_box_proper_kpc']
    if (not len(ids) or len(np.unique(ids))!=len(ids) or times.ndim!=1 or len(times)<2
            or not np.isfinite(times).all() or np.any(np.diff(times)<=0) or boxes.shape!=times.shape
            or len(report['epochs'])!=len(times)):
        raise ValueError('Aligned unique identities and strictly ordered native image epochs required')
    if (not isinstance(snapshots,list) or len(snapshots)!=len(times) or len(set(snapshots))!=len(snapshots)
            or any(type(s) is not int or s<0 for s in snapshots) or type(species) is not int or not 0<=species<=5):
        raise ValueError('Exact saved labels/species must match native history epochs')
    if len(ids)*len(times)*160>memory_gib*1024**3/2:
        raise MemoryError('Native image sidecar exceeds declared allocation')
    order=np.argsort(ids);sorted_ids=ids[order];shape=(len(times),len(ids))
    images=np.zeros(shape+(3,),np.int64);version=np.zeros(shape,np.uint64);status=np.zeros(shape,np.uint64)
    anchors=np.zeros(shape);supported=np.zeros(shape,bool)
    chunk=max(1,min(100000,int(memory_gib*1024**3/1024)))
    for epoch,snapshot in enumerate(snapshots):
        seen=np.zeros(len(ids),bool);files=_snapshot_files(Path(report['native_run_directory']),snapshot)
        if not files:raise FileNotFoundError('Native image source snapshot is missing')
        for path in files:
            name=str(path.resolve());digest=file_sha256(path)
            if sources.get(name)!=digest:raise ValueError('Image counters are outside the history native source binding')
            with h5py.File(path) as native:
                declared=report['epochs'][epoch];clock='actual_scale_factor' if int(native['Parameters'].attrs['ComovingIntegrationOn']) else 'native_time_code'
                if declared.get(clock)!=float(native['Header'].attrs['Time']):
                    raise ValueError('Native image snapshot epoch differs from bound history')
                box=native_box_proper_kpc(native['Header'].attrs,native['Parameters'].attrs)
                if not np.isclose(box,boxes[epoch],rtol=1e-12,atol=0):raise ValueError('Native image box differs from its history frame')
                group=native.get('PartType'+str(species))
                if group is None:continue
                for first in range(0,len(group['ParticleIDs']),chunk):
                    stop=min(first+chunk,len(group['ParticleIDs']))
                    candidates=exact_uint64_ids(group['ParticleIDs'][first:stop],unique=False)
                    at=np.searchsorted(sorted_ids,candidates);valid=at<len(ids)
                    valid[valid]=sorted_ids[at[valid]]==candidates[valid]
                    rows=order[at[valid]]
                    if len(np.unique(rows))!=len(rows) or seen[rows].any():raise ValueError('Duplicate native image identity')
                    seen[rows]=True
                    state=read_native_image_state(group,start=first,stop=stop,memory_bytes=int(memory_gib*1024**3/4))
                    if state is not None:
                        images[epoch,rows]=state['image_offsets'][valid];version[epoch,rows]=state['version'][valid]
                        status[epoch,rows]=state['status'][valid];anchors[epoch,rows]=state['anchor_native_time'][valid]
                        supported[epoch,rows]=state['supported'][valid]
            if file_sha256(path)!=digest:raise ValueError('Native image source changed during measurement')
        if not seen.all():raise ValueError('A selected exact history identity is absent from native image sources')
    # A newly initialized native run cannot masquerade as continuation of an
    # earlier winding origin even if its particle IDs and output times match.
    consistent=np.ones(len(ids),bool)
    for column in range(len(ids)):
        recorded=anchors[version[:,column]==1,column]
        if len(recorded)>1:consistent[column]=bool(np.all(recorded==recorded[0]))
    supported[:,~consistent]=False
    arrays={'particle_ids':ids,'times_myr':times,'periodic_box_proper_kpc':boxes,'image_offsets':images,
        'image_supported':supported,'image_version':version,'image_status':status,'anchor_native_time':anchors}
    units={key:'dimensionless' for key in arrays};units.update(particle_ids='identity',times_myr='Myr',periodic_box_proper_kpc='proper kpc')
    sources[str(Path(__file__).resolve())]=file_sha256(__file__)
    for path,digest in sources.items():
        if file_sha256(path)!=digest:raise ValueError('Native image/history source changed')
    result={'schema':'native-worldline-periodic-images-v1','history_sha256':file_sha256(history),'namespace':report['namespace'],
        'execution_status':'COMPLETE','status':'PASS' if supported.all() else 'UNRESOLVED',
        'recording':'native per-drift integer counters; legacy absent fields remain unsupported',
        'anchor_clock':'native scale factor or native elapsed-time code according to original epoch contracts',
        'epochs':report['epochs'],'unsupported_rows':int((~supported).sum()),
        'scope':'Recorded native spatial images only; no temporal interpolation or encounter science acceptance',
        'science_acceptance':'NOT_ACCEPTED'}
    return publish_stream_product(destination,result,arrays,units,sources=sources,namespace=report['namespace'],epoch={'epochs':report['epochs']})
