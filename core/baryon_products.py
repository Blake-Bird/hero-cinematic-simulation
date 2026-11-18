"""Measured native gas, stellar-return, metal and population products.

All products reuse the saved hydro census and original physical units. Gas
histograms sum mass per physical bin; SFR is a whole-snapshot sum in Msun/yr.
Stellar source losses and gas recipient gains remain separate signed ledgers.
Stellar birth-metal bookkeeping is distinct from current gas metal inventory.
Population colors use summed spectral luminosities and the pinned SSP model;
incomplete coverage stays masked rather than becoming a whole-census total.
"""
from pathlib import Path
import numpy as np
from .foundation import file_sha256
from .data import export_numeric_audit, exact_uint64_ids
from .stream_common import read_product
from .research.display import export_scientific_suite


def hydro_baryon_inputs(product,*,memory_gib):
    from .flagship_bh import bh_event_input_bindings
    report,sources=bh_event_input_bindings(product,memory_gib=memory_gib)
    reread,arrays,pins=read_product(product,memory_gib=memory_gib/3)
    if reread!=report:raise ValueError('Native hydro census changed during intake')
    return report,arrays,{**sources,**pins}



def native_sfr_history_arrays(report,arrays):
    """Use the existing whole-snapshot estimator, preserving every saved gap."""
    from .analysis import build_star_formation_history
    summary={};snapshots=[];scale=[]
    declarations=report['numeric_data']['array_contract']['datasets']
    for ordinal,meta in enumerate(report['epochs']):
        if not meta['parameters']['comoving']:raise ValueError('Cosmological SFR history requires an actual scale-factor clock')
        snapshot=meta['snapshot'];a=float(meta['epoch']['actual_scale_factor'])
        if type(snapshot) is not int or snapshot in snapshots or not np.isfinite(a) or a<=0:
            raise ValueError('Unique saved snapshots and positive actual scale factors required')
        z=float(meta['epoch']['redshift'])
        if not np.isclose(z,1/a-1,rtol=1.e-13,atol=1.e-13):raise ValueError('Native SFR epoch clock differs')
        prefix=f'epoch_{ordinal:03d}_';values={}
        for name,unit in {'species':'dimensionless','gas_sfr_msun_per_year':'Msun/yr',
                'gas_sfr_msun_per_year_available':'dimensionless'}.items():
            key=prefix+name
            if declarations[key]['units']!=unit:raise ValueError('Native SFR unit mismatch')
            values[name]=np.asarray(arrays[key])
        species=values['species'];rate=values['gas_sfr_msun_per_year'];available=values['gas_sfr_msun_per_year_available']
        if species.ndim!=1 or rate.shape!=species.shape or available.shape!=species.shape or available.dtype!=np.dtype(bool):
            raise ValueError('Native SFR requires aligned boolean coverage')
        gas=species==0;complete=bool(available[gas].all())
        if np.any(~np.isfinite(rate[gas&available])) or np.any(rate[gas&available]<0):raise ValueError('Invalid original measured gas SFR')
        measured=float(rate[gas].sum()) if complete else None
        if measured is not None and not np.isfinite(measured):
            raise ValueError('Whole-snapshot native SFR overflowed')
        recorded=meta['summary']['instantaneous_sfr_msun_per_year']
        if (measured is None)!=(recorded is None) or measured is not None and not np.isclose(measured,recorded,rtol=1.e-13,atol=0):
            raise ValueError('Original SFR summary differs from its complete census')
        summary[str(snapshot)]={'actual_redshift':z,'star_formation':{
            'sfr_physical_status':'COMPLETE' if complete else 'UNAVAILABLE',
            'sfr_total_msun_per_year':measured,'sfr_physical_reason':'Incomplete native gas SFR coverage' if not complete else None}}
        snapshots.append(snapshot);scale.append(a)
    scale=np.asarray(scale,float)
    if np.any(np.diff(scale)<=0):raise ValueError('Saved SFR history must have strictly increasing actual epochs')
    history=build_star_formation_history(summary)
    accepted=dict(zip(history['snapshot'],history['total_sfr_msun_per_year']))
    rate=np.asarray([accepted.get(s,np.nan) for s in snapshots],float)
    return history,{'snapshot':np.asarray(snapshots,np.int64),'scale_factor':scale,
        'instantaneous_sfr':rate,'supported':np.isfinite(rate)}



def export_native_sfr_history(product,destination,*,memory_gib=1.):
    report,arrays,sources=hydro_baryon_inputs(product,memory_gib=memory_gib)
    history,values=native_sfr_history_arrays(report,arrays)
    out=Path(destination).resolve()
    if out.exists():raise FileExistsError('Keep previous native SFR history')
    epoch={'canonical_time':'saved actual scale factor','saved_epoch_contracts':[m['epoch'] for m in report['epochs']]}
    units={'snapshot':'dimensionless','scale_factor':'dimensionless','instantaneous_sfr':'Msun/yr','supported':'dimensionless'}
    sources[str(Path(__file__).resolve())]=file_sha256(__file__)
    # The reused estimator lives in a consolidated logical module.
    from . import source_path
    algorithm=source_path('analysis');sources[str(algorithm)]=file_sha256(algorithm)
    out.mkdir(parents=True);target=out/'star-formation-history.json'
    export_numeric_audit(target,{'schema':'native-star-formation-history-v1','epoch':epoch,
        'original_estimator':history,'native_accounting_status':report['status'],'science_acceptance':'NOT_ACCEPTED',
        'quantity':'whole native snapshot instantaneous sum of gas StarFormationRate',
        'comparison_scope':'same frozen run; not a volume-normalized cosmic SFR density',
        'uncertainty':'native deterministic sum; resolution/sampling convergence not established',
        'missing_policy':'every saved epoch retained; unsupported epochs NaN and false; lines break across gaps'},
        values,{k:{'units':u,'quantity':k,'missing_policy':'explicit saved-epoch support mask'} for k,u in units.items()},
        algorithm=__file__,context={'epoch':epoch},source_hashes=sources)
    if any(file_sha256(p)!=h for p,h in sources.items()):raise ValueError('Native SFR sources changed during export')
    return export_scientific_suite({'sources':{'history':str(target)},'dpi':160,'maximum_panels_per_figure':1,
        'organized_outputs':True,'figures':[{'figure_id':'native-star-formation-history','family':'baryons',
            'variant':'star_formation_history','title':'Native instantaneous star-formation history',
            'science_status':'NOT_ACCEPTED','epoch':epoch,'panels':[{'panel_id':'history',
                'title':'Complete saved gas censuses; zero rates retained','quality_status':'PENDING_CONVERGENCE',
                'selection':'all native gas in each saved snapshot; incomplete SFR coverage creates gaps',
                'xlabel':'Actual scale factor [dimensionless]','ylabel':'Whole snapshot instantaneous SFR [Msun/yr]',
                'x_units':'dimensionless','y_units':'Msun/yr','series':[
                    {'source':'history','kind':kind,'x':'scale_factor','y':'instantaneous_sfr','support':'supported',
                        'label':label,'marker_size':10} for kind,label in [('line','native SFR; missing epochs break line'),('scatter','saved measurements')]]}]}]},
        out/'gallery',base=out,memory_gib=memory_gib)



def native_return_history_arrays(report):
    """Aggregate the original source/recipient ledger separately, without repair."""
    epochs=report['epochs'];intervals=report['intervals']
    if len(intervals)!=len(epochs)-1:raise ValueError('Complete saved return interval census required')
    scale=np.asarray([m['epoch']['actual_scale_factor'] for m in epochs],float)
    if any(not m['parameters']['comoving'] for m in epochs) or not np.isfinite(scale).all() or np.any(scale<=0) or np.any(np.diff(scale)<=0):
        raise ValueError('Actual increasing cosmological return epochs required')
    released=[0.];deposited=[0.];counts=[0]
    for i,interval in enumerate(intervals):
        before,after=epochs[i:i+2]
        if interval['before_snapshot']!=before['snapshot'] or interval['after_snapshot']!=after['snapshot']:
            raise ValueError('Native return interval snapshot joins differ')
        rows=interval['stellar_return_rows']
        if after['return_event_count']-before['return_event_count']!=len(rows):raise ValueError('Incomplete native return row counter census')
        loss=0.;gain=0.
        for row in rows:
            exact_uint64_ids([row['particle_id'],row['source_id']],unique=False)
            if row['species'] not in (0,4) or not before['native_time']<=row['native_time']<=after['native_time']:
                raise ValueError('Native return row outside source interval')
            delta=row['mass_after_msun']-row['mass_before_msun']
            if not np.isfinite(delta):raise ValueError('Nonfinite native return mass ledger')
            if row['species']==4:loss-=delta
            else:gain+=delta
        released.append(released[-1]+loss);deposited.append(deposited[-1]+gain);counts.append(counts[-1]+len(rows))
    return {'scale_factor':scale,'released_since_first_saved':np.asarray(released),
        'deposited_since_first_saved':np.asarray(deposited),'cumulative_journal_rows':np.asarray(counts,np.int64),
        'supported':np.ones(len(epochs),bool)}



def export_native_mass_return(product,destination,*,memory_gib=1.):
    report,_,sources=hydro_baryon_inputs(product,memory_gib=memory_gib)
    values=native_return_history_arrays(report);out=Path(destination).resolve()
    if out.exists():raise FileExistsError('Keep previous native stellar return history')
    epoch={'canonical_time':'saved actual scale factor','saved_epoch_contracts':[m['epoch'] for m in report['epochs']]}
    units={k:'Msun' if k.endswith('_saved') else 'dimensionless' for k in values}
    sources[str(Path(__file__).resolve())]=file_sha256(__file__)
    out.mkdir(parents=True);target=out/'mass-return.json'
    export_numeric_audit(target,{'schema':'native-stellar-return-history-v1','epoch':epoch,
        'native_accounting_status':report['status'],'science_acceptance':'NOT_ACCEPTED',
        'quantity':'committed delayed-return source mass loss and gas recipient mass gain; independently retained ledgers',
        'baseline':'zero at first selected saved snapshot; prior returns are unknown, not zero',
        'prompt_return':'already recycled SH03 birth bookkeeping excluded; never deposited again',
        'native_return_build_fingerprints':[i['stellar_return_build_fingerprints'] for i in report['intervals']],
        'signed_values':'original differences retained without clipping or closure repair',
        'uncertainty':'native operator ledger; yield-source validity and resolution acceptance remain separate'},
        values,{k:{'units':u,'quantity':k,'missing_policy':'complete verified journal counters; no interpolation or extrapolation'} for k,u in units.items()},
        algorithm=__file__,context={'epoch':epoch},source_hashes=sources)
    if any(file_sha256(p)!=h for p,h in sources.items()):raise ValueError('Native return source changed during export')
    return export_scientific_suite({'sources':{'returns':str(target)},'dpi':160,'maximum_panels_per_figure':1,
        'organized_outputs':True,'figures':[{'figure_id':'native-mass-return','family':'baryons','variant':'mass_return',
            'title':'Committed native delayed stellar returns','science_status':'NOT_ACCEPTED','epoch':epoch,
            'panels':[{'panel_id':'returns','title':'Separate source and recipient ledgers; first saved epoch is baseline',
                'quality_status':'NATIVE_ACCOUNTING_'+report['status'],'selection':'complete committed delayed-return journal only; prompt bookkeeping excluded',
                'xlabel':'Actual scale factor [dimensionless]','ylabel':'Mass transferred since first saved epoch [Msun]',
                'x_units':'dimensionless','y_units':'Msun','series':[
                    {'source':'returns','kind':kind,'x':'scale_factor','y':key,'support':'supported','label':label,'marker_size':10}
                    for key,label in [('released_since_first_saved','stellar source loss'),('deposited_since_first_saved','gas recipient gain')]
                    for kind in ('line','scatter')]}]}]},out/'gallery',base=out,memory_gib=memory_gib)



def gas_phase_arrays(report, arrays, ordinal):
    prefix=f'epoch_{ordinal:03d}_'
    contracts=report['numeric_data']['array_contract']['datasets']
    units={'particle_ids':'identity','species':'dimensionless','mass_msun':'Msun',
        'gas_density_g_cm3':'g/cm^3','gas_temperature_kelvin':'K',
        'gas_sfr_msun_per_year':'Msun/yr'}
    values={}
    for name,unit in units.items():
        key=prefix+name
        if contracts[key]['units']!=unit:raise ValueError('Native gas unit mismatch: '+name)
        values[name]=np.asarray(arrays[key])
    ids=exact_uint64_ids(values['particle_ids']);n=len(ids)
    if any(v.shape!=(n,) for v in values.values()):raise ValueError('Aligned native census required')
    gas=values['species']==0
    flags={}
    for name in ('gas_density_g_cm3','gas_temperature','gas_sfr_msun_per_year'):
        key=prefix+name+'_available';flag=np.asarray(arrays[key])
        if flag.dtype!=np.dtype(bool) or flag.shape!=(n,) or contracts[key]['units']!='dimensionless':
            raise ValueError('Native gas availability must be a boolean census')
        flags[name]=flag
    sfr=values['gas_sfr_msun_per_year']
    measured_sfr=gas&flags['gas_sfr_msun_per_year']&np.isfinite(sfr)&(sfr>=0)
    eos=measured_sfr&(sfr>0)
    support=measured_sfr&(sfr==0)&flags['gas_density_g_cm3']&flags['gas_temperature']
    for name in ('gas_density_g_cm3','gas_temperature_kelvin','mass_msun'):
        support &= np.isfinite(values[name])&(values[name]>0)
    return {'particle_ids':ids,'density':values['gas_density_g_cm3'],
        'temperature':values['gas_temperature_kelvin'],'mass':values['mass_msun'],
        'gas':gas,'effective_eos':eos,'supported':support}



def gas_phase_inputs(request,*,memory_gib):
    if request.get('schema')!='native-gas-phase-request-v1':raise ValueError('Explicit gas phase request required')
    edges={}
    for name in ('density_edges_g_cm3','temperature_edges_kelvin'):
        edge=np.asarray(request[name],float)
        if edge.ndim!=1 or not 2<=len(edge)<=4097 or not np.isfinite(edge).all() or np.any(edge<=0) or np.any(np.diff(edge)<=0):
            raise ValueError('Positive ordered bounded physical phase edges required')
        edges[name]=edge
    if np.prod([len(v)-1 for v in edges.values()])*64>memory_gib*1024**3/4:
        raise MemoryError('Phase histogram workspace exceeds allocation')
    binding=request['hydro_history'];path=Path(binding['path']).resolve()
    if file_sha256(path)!=binding['sha256']:raise ValueError('Native hydro history binding changed')
    report,arrays,sources=hydro_baryon_inputs(path,memory_gib=memory_gib)
    return report,arrays,sources,edges



def export_native_gas_phase(request,destination,*,memory_gib=1.):
    report,arrays,sources,edges=gas_phase_inputs(request,memory_gib=memory_gib)
    out=Path(destination).resolve()
    if out.exists():raise FileExistsError('Keep previous native gas exports')
    out.mkdir(parents=True);products={};figures=[]
    namespace=report['provenance']['identity_namespaces']['native_particle']
    for ordinal,meta in enumerate(report['epochs']):
        values=gas_phase_arrays(report,arrays,ordinal);epoch=meta['epoch'];name=f'gas-{ordinal:03d}'
        physical=values['supported'];inrange=physical.copy()
        for key,edgekey in (('density','density_edges_g_cm3'),('temperature','temperature_edges_kelvin')):
            edge=edges[edgekey];inrange&=(values[key]>=edge[0])&(values[key]<=edge[-1])
        values['in_display_range']=inrange
        units={'particle_ids':'identity','density':'g/cm^3','temperature':'K','mass':'Msun',
            'gas':'dimensionless','effective_eos':'dimensionless','supported':'dimensionless','in_display_range':'dimensionless'}
        specs={k:{'units':u,'quantity':k,'missing_policy':'original census retained; explicit physical and display support'} for k,u in units.items()}
        specs['particle_ids']['identity_namespace']=namespace
        path=out/(name+'.json');products[name]=str(path)
        export_numeric_audit(path,{'schema':'native-gas-phase-v1','epoch':epoch,'snapshot':meta['snapshot'],
            'native_accounting_status':report['status'],'science_acceptance':'NOT_ACCEPTED',
            'temperature_convention':meta['summary']['temperature_convention'],
            'selection':'native gas with measured non-star-forming SFR, supported single-phase temperature and positive proper density/mass',
            'histogram_definition':'sum of original gas mass per declared physical bin; no probability normalization',
            'supported_count':int(physical.sum()),'outside_display_count':int((physical&~inrange).sum()),
            'edges':{k:v.tolist() for k,v in edges.items()}},values,specs,algorithm=__file__,
            context={'epoch':epoch},source_hashes={**sources,str(Path(__file__).resolve()):file_sha256(__file__)})
        figures.append({'figure_id':name+'-phase','family':'baryons','variant':'gas_phase_diagram',
            'title':'Native single-phase gas','science_status':'NOT_ACCEPTED','epoch':epoch,'panels':[{
                'panel_id':'phase','title':f'Mass per physical bin; {int(inrange.sum())} supported gas particles',
                'quality_status':'CONDITIONAL_NATIVE_SUPPORT' if inrange.any() else 'NO_SUPPORTED_GAS_IN_DISPLAY',
                'selection':'single-phase native gas; effective EOS excluded; out-of-range census recorded',
                'xlabel':'Proper density [g/cm^3]','ylabel':'Temperature [K]','x_units':'g/cm^3','y_units':'K',
                'xscale':'log','yscale':'log','series':[{'source':name,'kind':'histogram2d','label':'native gas mass',
                    'array_bindings':{'x':'density','y':'temperature','weight':'mass','support':'in_display_range'},
                    'binding_units':{'x':'g/cm^3','y':'K','weight':'Msun','support':'dimensionless'},
                    'output_units':{'x_edges':'g/cm^3','y_edges':'K','weighted_sum_xy':'Msun','counts_xy':'dimensionless','sample_support':'dimensionless'},
                    'options':{'x_edges':edges['density_edges_g_cm3'].tolist(),'y_edges':edges['temperature_edges_kelvin'].tolist(),'color_label':'Gas mass per bin [Msun]'}}]}]})
    if any(file_sha256(p)!=h for p,h in sources.items()):raise ValueError('Native gas sources changed during export')
    return export_scientific_suite({'sources':products,'figures':figures,'dpi':160,
        'maximum_panels_per_figure':1,'organized_outputs':True},out/'gallery',base=out,memory_gib=memory_gib)



ELEMENTS=['H','He','C','N','O','Ne','Mg','Si','S','Ca','Fe']



def native_metal_history_arrays(report,arrays):
    contracts=report['numeric_data']['array_contract']['datasets'];rows=[];times=[]
    emitted=np.zeros(11);deposited=np.zeros(11)
    if len(report['intervals'])!=len(report['epochs'])-1:raise ValueError('Complete native metal-history interval census required')
    for ordinal,meta in enumerate(report['epochs']):
        if meta['element_order']!=ELEMENTS:raise ValueError('Native metal element order differs')
        if not meta['parameters']['comoving']:raise ValueError('Native metal history requires actual cosmological epochs')
        times.append(float(meta['epoch']['actual_scale_factor']));prefix=f'epoch_{ordinal:03d}_'
        values={}
        for name,unit in {'species':'dimensionless','mass_msun':'Msun','gas_element_mass_fraction':'dimensionless',
                'gas_elements_available':'dimensionless','SSPBirthStateVersion':'dimensionless',
                'SSPInitialLockedMass':'Msun','SSPBirthMetallicity':'dimensionless'}.items():
            key=prefix+name
            if contracts[key]['units']!=unit:raise ValueError('Native metal boundary unit mismatch: '+name)
            values[name]=np.asarray(arrays[key])
        mass=values['mass_msun'];n=len(mass);fractions=values['gas_element_mass_fraction'];available=values['gas_elements_available']
        if mass.ndim!=1 or fractions.shape!=(n,11) or available.dtype!=np.dtype(bool) or any(v.shape!=(n,) for k,v in values.items() if k!='gas_element_mass_fraction'):
            raise ValueError('Aligned native metal census and boolean support required')
        gas=values['species']==0;stars=values['species']==4;measured=gas&available
        if not np.isfinite(mass).all() or np.any(mass<=0) or not np.isfinite(fractions[measured]).all() or np.any(fractions[measured]<0):
            raise ValueError('Invalid original native gas element mass')
        if np.any(abs(fractions[measured].sum(axis=1)-1)>1.e-12):raise ValueError('Original gas composition closure failed')
        subtotal=np.sum(mass[measured,None]*fractions[measured],axis=0);complete=bool(available[gas].all())
        birth=stars&(values['SSPBirthStateVersion']==1);birth_complete=bool(birth.sum()==stars.sum())
        locked=values['SSPInitialLockedMass'];z=values['SSPBirthMetallicity']
        if not np.isfinite(locked[birth]).all() or np.any(locked[birth]<=0) or not np.isfinite(z[birth]).all() or np.any((z[birth]<0)|(z[birth]>1)):
            raise ValueError('Invalid measured native stellar birth metal bookkeeping')
        if ordinal:
            before=report['epochs'][ordinal-1];interval=report['intervals'][ordinal-1]
            if interval['before_snapshot']!=before['snapshot'] or interval['after_snapshot']!=meta['snapshot']:
                raise ValueError('Metal return journal epoch join differs')
            native_rows=interval['stellar_return_rows']
            if meta['return_event_count']-before['return_event_count']!=len(native_rows):raise ValueError('Incomplete metal return journal row census')
            for row in native_rows:
                eb=np.asarray(row['elements_before_msun'],float);ea=np.asarray(row['elements_after_msun'],float)
                if eb.shape!=(11,) or ea.shape!=(11,) or not np.isfinite(eb).all() or not np.isfinite(ea).all():raise ValueError('Invalid original metal journal arrays')
                if row['species']==4:emitted+=ea
                elif row['species']==0:deposited+=ea-eb
                else:raise ValueError('Unknown native stellar-return metal ledger species')
        row={'gas_supported':complete,'stellar_birth_supported':birth_complete,
            'stellar_current_supported':not bool(stars.any()),'stellar_current_metals':0. if not stars.any() else np.nan,
            'stellar_birth_locked_metals':float(np.sum(locked[birth]*z[birth])) if birth_complete else np.nan,
            'gas_supported_metals_subtotal':float(subtotal[2:].sum()),'gas_metals':float(subtotal[2:].sum()) if complete else np.nan,
            'emitted_metals_since_first_saved':float(emitted[2:].sum()),'deposited_metals_since_first_saved':float(deposited[2:].sum()),
            'gas_unavailable_count':int((gas&~available).sum()),'stellar_birth_unavailable_count':int((stars&~birth).sum())}
        for i,element in enumerate(ELEMENTS):
            row['gas_'+element]=float(subtotal[i]) if complete else np.nan
            row['emitted_'+element+'_since_first_saved']=float(emitted[i]);row['deposited_'+element+'_since_first_saved']=float(deposited[i])
        rows.append(row)
    times=np.asarray(times)
    if not np.isfinite(times).all() or np.any(times<=0) or np.any(np.diff(times)<=0):raise ValueError('Increasing actual metal-history epochs required')
    values={k:np.asarray([row[k] for row in rows]) for k in rows[0]};values['scale_factor']=times
    return values



def export_native_metal_history(product,destination,*,memory_gib=2.):
    report,arrays,sources=hydro_baryon_inputs(product,memory_gib=memory_gib);values=native_metal_history_arrays(report,arrays)
    out=Path(destination).resolve()
    if out.exists():raise FileExistsError('Keep previous native metal histories')
    epoch={'canonical_time':'actual saved scale factor','saved_epoch_contracts':[m['epoch'] for m in report['epochs']]}
    units={k:'dimensionless' if k.endswith('_supported') or k.endswith('_count') or k=='scale_factor' else 'Msun' for k in values}
    sources[str(Path(__file__).resolve())]=file_sha256(__file__);out.mkdir(parents=True);target=out/'metals-history.json'
    export_numeric_audit(target,{'schema':'native-metal-history-v1','epoch':epoch,'element_order':ELEMENTS,
        'native_accounting_status':report['status'],'science_acceptance':'NOT_ACCEPTED',
        'gas_definition':'sum of current gas mass times original native element fractions; complete gas coverage required for totals; metals exclude H and He',
        'stellar_birth_definition':'initial locked birth mass times original birth metallicity, for presently retained SSPs; birth bookkeeping, not current stellar chemical mass',
        'stellar_current_definition':'current stellar chemical inventory unavailable for nonempty SSP census; explicit NaN/false retained',
        'return_definition':'original committed source emitted elements and gas recipient element gains, separately accumulated since first selected saved epoch; no prompt redeposition',
        'uncertainty':'native inventories and operator ledgers; original accounting FAIL, yield validity and convergence remain separate',
        'closure_policy':'no normalization, clipping or inferred global gas-plus-star metal conservation'},values,
        {k:{'units':u,'quantity':k,'missing_policy':'explicit completeness support; original signed ledgers retained'} for k,u in units.items()},
        algorithm=__file__,context={'epoch':epoch},source_hashes=sources)
    if any(file_sha256(p)!=h for p,h in sources.items()):raise ValueError('Native metal sources changed during export')
    panels=[]
    for name,title,series in [
        ('gas','Current gas metal inventory',[('gas_metals','gas_supported','current gas metals')]),
        ('stellar','Stellar birth bookkeeping; current stellar chemistry unavailable',[
            ('stellar_birth_locked_metals','stellar_birth_supported','retained SSP initial locked birth metals'),
            ('stellar_current_metals','stellar_current_supported','current stellar chemical inventory')]),
        ('returns','Separate delayed-return ledgers since first selected snapshot',[
            ('emitted_metals_since_first_saved',None,'committed stellar ejecta metals'),
            ('deposited_metals_since_first_saved',None,'committed gas recipient metal gain')])]:
        panels.append({'panel_id':name,'title':title,'quality_status':'NATIVE_ACCOUNTING_'+report['status'],
            'selection':'original gas composition, SSP birth state and complete committed return journal; scopes kept distinct',
            'xlabel':'Actual scale factor [dimensionless]','ylabel':'Metal mass [Msun]','x_units':'dimensionless','y_units':'Msun',
            'series':[{'source':'history','kind':kind,'x':'scale_factor','y':key,'label':label,'marker_size':10,
                **({'support':support} if support else {})} for key,support,label in series for kind in ('line','scatter')]})
    return export_scientific_suite({'sources':{'history':str(target)},'dpi':160,'maximum_panels_per_figure':3,
        'organized_outputs':True,'figures':[{'figure_id':'native-metals-history','family':'baryons','variant':'metals_history',
            'title':'Native gas, stellar birth and ejecta metal history','science_status':'NOT_ACCEPTED','epoch':epoch,'panels':panels}]},
        out/'gallery',base=out,memory_gib=memory_gib)



def integrated_stellar_light(report,arrays,color_filters):
    """Whole-census luminosity; incomplete SSP coverage cannot become a total."""
    supported=np.asarray(arrays['sps_supported']);bol=np.asarray(arrays['bolometric_luminosity_lsun'])
    bands=np.asarray(arrays['band_luminosity_erg_s_hz']);n=len(bol)
    if supported.dtype!=np.dtype(bool) or supported.shape!=(n,) or bands.shape!=(n,len(report['filters'])):
        raise ValueError('Aligned measured SSP luminosity support required')
    if np.any(~np.isfinite(bol[supported])) or np.any(bol[supported]<0) or np.any(~np.isfinite(bands[supported])) or np.any(bands[supported]<0):
        raise ValueError('Invalid supported SSP luminosity')
    if not isinstance(color_filters,list) or len(color_filters)!=2 or color_filters[0]==color_filters[1] or any(f not in report['filters'] for f in color_filters):
        raise ValueError('Two distinct pinned color filters required')
    complete=bool(supported.all());subtotal=float(bol[supported].sum())
    lnu=np.sum(bands[supported],axis=0);i,j=[report['filters'].index(f) for f in color_filters]
    if not np.isfinite(subtotal) or not np.isfinite(lnu).all():raise ValueError('Integrated SSP luminosity overflowed')
    color_supported=complete and lnu[i]>0 and lnu[j]>0
    return {'bolometric_total':subtotal if complete else np.nan,'supported_bolometric_subtotal':subtotal,
        # AB zero points cancel between bands. Subtract logs of their summed
        # luminosities to avoid overflow/underflow in an intermediate ratio.
        'color_ab':float(-2.5*(np.log10(lnu[i])-np.log10(lnu[j]))) if color_supported else np.nan,
        'luminosity_supported':complete,'color_supported':bool(color_supported),
        'stellar_count':n,'unsupported_stellar_count':int((~supported).sum())}



def population_inputs(request,*,memory_gib):
    from .stellar_population import pinned_fsps
    if request.get('schema')!='native-population-history-request-v1':raise ValueError('Explicit native population history request required')
    binding=request['hydro_history'];path=Path(binding['path']).resolve()
    if file_sha256(path)!=binding['sha256']:raise ValueError('Native population hydro history changed')
    report,_,sources=hydro_baryon_inputs(path,memory_gib=memory_gib)
    _,model=pinned_fsps(request['synthesis']['backend'])
    colors=request['color_filters']
    if not isinstance(colors,list) or len(colors)!=2 or colors[0]==colors[1] or any(f not in model['filters'] for f in colors):
        raise ValueError('Requested color filters differ from pinned synthesis')
    return report,{**sources,**model['source_hashes']},model



def export_native_population_history(request,destination,*,memory_gib=2.):
    from .stellar_population import stellar_sps_truth
    from .stellar_population import synthesize_stellar_light
    from . import source_path
    report,sources,model=population_inputs(request,memory_gib=memory_gib)
    helper=source_path('stellar_population');sources[str(helper)]=file_sha256(helper)
    sources[str(Path(__file__).resolve())]=file_sha256(__file__)
    out=Path(destination).resolve()
    if out.exists():raise FileExistsError('Keep previous native population history')
    out.mkdir(parents=True);rows=[];products=[];times=[]
    namespace=report['provenance']['identity_namespaces']['native_particle']
    for ordinal,meta in enumerate(report['epochs']):
        truth,pins=stellar_sps_truth(report['request']['run_directory'],meta['snapshot'],
            memory_bytes=int(memory_gib*1024**3/3),namespace=namespace)
        sources.update(pins)
        sps,arrays=synthesize_stellar_light(truth,{**request['synthesis'],'memory_gib':memory_gib/3})
        if sps['model']!=model:raise ValueError('Pinned SSP model changed across saved history')
        arrays={**arrays,'native_age_myr':np.asarray(truth['age_myr']),
            'native_birth_metallicity':np.asarray(truth['metallicity_mass_fraction'])}
        timekey='cosmic_time_gyr' if meta['parameters']['comoving'] else 'elapsed_time_gyr'
        time=float(meta['epoch'][timekey])*1000
        if not np.isfinite(time):raise ValueError('Actual saved population proper time required')
        times.append(time);rows.append(integrated_stellar_light(sps,arrays,request['color_filters']))
        units={'particle_ids':'identity','native_current_mass_msun':'Msun','sps_surviving_mass_msun':'Msun',
            'initial_formed_mass_msun':'Msun','sps_supported':'dimensionless','sps_age_supported':'dimensionless',
            'sps_metallicity_supported':'dimensionless','native_age_myr':'Myr',
            'native_birth_metallicity':'dimensionless',**sps['units']}
        specs={k:{'units':units[k],'quantity':k,'missing_policy':'original FSPS SSP domain support retained'} for k in arrays}
        specs['particle_ids']['identity_namespace']=namespace
        target=out/f'sps-{ordinal:03d}.json';products.append(str(target))
        export_numeric_audit(target,{**sps,'epoch':meta['epoch'],'snapshot':meta['snapshot'],
            'native_accounting_status':report['status'],'birth_source_hashes':truth['birth_source_hashes']},arrays,specs,
            algorithm=__file__,context={'epoch':meta['epoch']},source_hashes=sources)
    if np.any(np.diff(times)<=0):raise ValueError('Population history saved proper times must increase')
    values={k:np.asarray([row[k] for row in rows]) for k in rows[0]};values['saved_time_myr']=np.asarray(times)
    epoch={'canonical_time':'actual saved proper time','saved_epoch_contracts':[m['epoch'] for m in report['epochs']]}
    units={k:'Lsun' if 'bolometric' in k else 'AB mag' if k=='color_ab' else 'Myr' if k=='saved_time_myr' else 'dimensionless' for k in values}
    sources[str(Path(__file__).resolve())]=file_sha256(__file__)
    target=out/'population-history.json'
    export_numeric_audit(target,{'schema':'native-population-history-v1','epoch':epoch,'model':model,
        'sps_products':products,'native_accounting_status':report['status'],'science_acceptance':'NOT_ACCEPTED',
        'selection':'all original native species-4 SSPs in each saved census; no partial-domain total',
        'color_filters':request['color_filters'],'color_convention':'intrinsic dust-free rest-frame integrated AB color: -2.5 log10(sum Lnu_first / sum Lnu_second)',
        'empty_census':'zero bolometric luminosity; color undefined without positive band luminosities',
        'unsupported_census':'supported subtotal retained separately; whole-census luminosity/color NaN and masked',
        'uncertainty':'conditional pinned synthesis model; model systematics and numerical resolution remain unaccepted'},values,
        {k:{'units':u,'quantity':k,'missing_policy':'saved-epoch support; lines break across unavailable totals'} for k,u in units.items()},
        algorithm=__file__,context={'epoch':epoch},source_hashes=sources)
    if any(file_sha256(p)!=h for p,h in sources.items()):raise ValueError('Native population sources changed during export')
    figures=[]
    for variant,key,support,unit in [('luminosity_history','bolometric_total','luminosity_supported','Lsun'),
            ('color_history','color_ab','color_supported','AB mag')]:
        figures.append({'figure_id':'native-'+variant.replace('_','-'),'family':'baryons','variant':variant,
            'title':'Native '+variant.replace('_',' '),'science_status':'NOT_ACCEPTED','epoch':epoch,'panels':[{
                'panel_id':'history','title':'Whole native SSP census; pinned intrinsic synthesis','quality_status':'CONDITIONAL_SPS_MODEL',
                'selection':'all native SSPs; incomplete model coverage masks total; zero photon color undefined',
                'xlabel':'Actual saved proper time [Myr]','ylabel':key+' ['+unit+']','x_units':'Myr','y_units':unit,
                'series':[{'source':'history','kind':kind,'x':'saved_time_myr','y':key,'support':support,
                    'label':label,'marker_size':10} for kind,label in [('line','saved history; gaps retained'),('scatter','saved measurements')]]}]})
    return export_scientific_suite({'sources':{'history':str(target)},'figures':figures,'dpi':160,
        'maximum_panels_per_figure':1,'organized_outputs':True},out/'gallery',base=out,memory_gib=memory_gib)

