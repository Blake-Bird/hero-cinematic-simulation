"""Saved SUBFIND ownership and recorded SSP births for native cooling joins."""
from pathlib import Path
import numpy as np
from .foundation import file_sha256
from .foundation import merge_source_bindings as _merge


def build_native_birth_graph(run,snapshots,destination,*,final_host_row,namespace,native_sources,limits,memory_gib,epoch_contracts):
    """Retain SUBFIND bound mass/owners; never promote FOF membership to binding."""
    from .halos import gadget_catalogue,graph_from_catalogues,add_native_ssp_birth_audit
    from .halos import read_native_subhalo_spans
    from .native_gadget_ancestry import native_link_table,recorded_link_sources
    from .stars import audit_native_ssp_birth_state
    from core.stream_common import read_product
    from . import source_path
    if not snapshots or snapshots!=sorted(set(snapshots)) or any(b!=a+1 for a,b in zip(snapshots,snapshots[1:])):
        raise ValueError('Birth ownership requires every actual consecutive saved epoch')
    if set(epoch_contracts)!=set(snapshots):raise ValueError('Birth graph must retain exact owned hydro epoch set')
    out=Path(destination);out.mkdir();catalogues=[];fields_by_epoch={};pins=recorded_link_sources();census=0;unresolved=[]
    if any(native_sources.get(path)!=sha for path,sha in pins.items()):raise ValueError('Birth ancestry needs original owned native tree writer')
    for epoch in snapshots:
        fields,_,_=read_native_subhalo_spans(run,epoch,native_sources=native_sources,maximum_subhalos=limits['maximum_subhalos'],
            maximum_particles=limits['maximum_particles'],memory_gib=memory_gib)
        count=len(fields['SubhaloGroupNr']);fields_by_epoch[epoch]=fields
        members=sum(int(x) for x in fields['SubhaloLenType'][:,[0,4]].flat);census+=count+members
        if members>limits['maximum_particles'] or census>limits['maximum_nodes'] or census*2048>memory_gib*1024**3/3:
            raise MemoryError('Native bound gas/SSP graph exceeds selected census/workspace')
        if not count:continue
        for species in (0,4):
            catalogue=gadget_catalogue(run,epoch,finder='subfind',particle_type=species,particle_namespace=namespace,
                memory_gib=memory_gib/3,selected_rows=list(range(count)))
            if any(native_sources.get(path)!=sha for path,sha in catalogue['source_hashes'].items() if path!=str(source_path('halos'))):
                raise ValueError('Bound birth catalogue is outside original owned native evolution')
            _merge(pins,catalogue['source_hashes']);catalogues.append(catalogue)
    graph,sources=graph_from_catalogues(catalogues,maximum_nodes=limits['maximum_nodes'],maximum_edges=limits['maximum_edges']);_merge(pins,sources)
    owners=str(Path(run).resolve())+':subfind:catalogue-row'
    for epoch in snapshots:
        fields=fields_by_epoch[epoch]
        if epoch not in graph.epochs:graph.add_epoch(epoch,epoch_contracts[epoch])
        if graph.epochs[epoch]!=epoch_contracts[epoch]:raise ValueError('Bound ownership and original hydro actual epoch contracts differ')
        for row in range(len(fields['SubhaloGroupNr'])):
            node=graph.nodes[(owners,row,epoch)]
            node['attributes']['measurement_quality_passed']=sum(map(int,fields['SubhaloLenType'][row]))>=limits['minimum_host_particles']
    for before,after in zip(snapshots[:-1],snapshots[1:]):
        left=fields_by_epoch[before];right=fields_by_epoch[after]
        desc,present,ds=native_link_table(run,before,'desc',expected_rows=len(left['SubhaloGroupNr']),maximum_rows=limits['maximum_subhalos'],memory_gib=memory_gib,native_sources=native_sources)
        prog,ppresent,ps=native_link_table(run,after,'prog',expected_rows=len(right['SubhaloGroupNr']),maximum_rows=limits['maximum_subhalos'],memory_gib=memory_gib,native_sources=native_sources)
        _merge(pins,ds);_merge(pins,ps)
        if np.any(desc[present]>=len(right['SubhaloGroupNr'])) or np.any(prog[ppresent]>=len(left['SubhaloGroupNr'])):
            raise ValueError('Recorded birth-host ancestry points outside original catalogue')
        for row in range(len(left['SubhaloGroupNr'])):
            if not present[row]:
                unresolved.append({'snapshot':before,'subhalo_row':row,'reason':'NO_RECORDED_NATIVE_DESCENDANT'});continue
            target=int(desc[row]);evidence={'origin':'native_ancestry','basis':'Recorded GADGET primary SUBFIND descendant; no row-continuity or FOF-binding assumption','source_hashes':{**ds,**ps,**recorded_link_sources()}}
            if ppresent[target] and int(prog[target])==row:evidence['main_progenitor']=True
            graph.add_edge((owners,row,before),(owners,target,after),'descendant',evidence=evidence)
    fields=fields_by_epoch[snapshots[-1]]
    final=np.flatnonzero((fields['SubhaloGroupNr']==final_host_row)&(fields['SubhaloRankInGr']==0))
    if len(final)!=1:
        return None,None,[*unresolved,{'reason':'Recovered final FOF host has no unique recorded rank-zero SUBFIND central'}]
    for epoch in snapshots:
        audit=out/('ssp-birth-'+str(epoch)+'.json')
        audit_native_ssp_birth_state({'run_directory':str(Path(run).resolve()),'snapshot':epoch,'memory_gib':memory_gib/3},audit)
        _merge(pins,add_native_ssp_birth_audit(graph,audit,particle_namespace=namespace,maximum_bytes=int(memory_gib*1024**3/3)))
        report,arrays,apins=read_product(audit,memory_gib=memory_gib/3);_merge(pins,apins)
        for pid,mass in zip(arrays['ParticleIDs'],arrays['SSPCurrentMass']):
            key=(namespace,int(pid),epoch)
            if key in graph.nodes and graph.nodes[key]['kind']=='ssp':
                graph.nodes[key]['attributes']['mass_msun']=float(mass)
    _merge(pins,{str(Path(__file__).resolve()):file_sha256(__file__)})
    reader=Path(__file__).with_name('halos.py').resolve()
    _merge(pins,{str(reader):file_sha256(reader)})
    for path,sha in pins.items():
        if file_sha256(path)!=sha:raise ValueError('Native birth graph source changed during intake')
    graph.publish(out/'identity-graph',source_hashes=pins)
    return out/'identity-graph',[owners,int(final[0]),snapshots[-1]],unresolved
