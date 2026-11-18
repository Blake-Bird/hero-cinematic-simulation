"""Native stellar/gas history intake and pinned FSPS population synthesis.

Native clocks determine ages in Myr; synthesis takes ages in Gyr. Luminosities
scale with gross initially formed mass in Msun, not present surviving mass.
FSPS surviving mass and the simulation's locked/current masses remain distinct.
Unavailable model-domain coverage stays masked; no age/metallicity clipping.
"""
from pathlib import Path
import importlib
import importlib.metadata
import json
import os
import numpy as np
from .stream_common import finite, positive, digest


def gas_epoch(root,snapshot,*,memory_bytes,namespace,host_identity,resolution_fields):
    from .stars import read_native_hydro_census
    from .data import exact_uint64_ids
    census,meta,sources,files=read_native_hydro_census(root,snapshot,memory_bytes=memory_bytes)
    use=census['species']==0
    gas={k:v[use] for k,v in census.items() if k.startswith('gas_') or k in {'particle_ids','mass_msun','positions_proper_kpc','velocities_peculiar_km_s'}}
    composition=gas['gas_element_mass_fraction'];inverse_mu=(composition/np.array([1,4,12,14,16,20,24,28,32,40,56])).sum(axis=1)+gas['gas_electron_abundance']*composition[:,0]
    gas['mean_molecular_weight']=np.divide(1.,inverse_mu,out=np.full(len(composition),np.nan),where=inverse_mu>0)
    native=gas.get('gas_native_eos_available',np.zeros(len(composition),dtype=bool))
    gas['mean_molecular_weight'][native]=gas['gas_mean_molecular_weight'][native]
    # Smoothing must be measured, never inferred from the particle separation.
    ids=exact_uint64_ids(resolution_fields['particle_ids'],allow_empty=True)
    if len(np.unique(ids))!=len(ids):raise ValueError('Duplicate gas resolution identities')
    length=finite(resolution_fields['smoothing_length_proper_kpc'],ids.shape)
    lookup={int(q):i for i,q in enumerate(ids)}
    if any(int(q) not in lookup for q in gas['particle_ids']):raise ValueError('Incomplete measured gas smoothing census')
    gas['smoothing_length_proper_kpc']=length[[lookup[int(q)] for q in gas['particle_ids']]]
    if np.any(gas['smoothing_length_proper_kpc']<=0):raise ValueError('Positive measured gas smoothing required')
    epoch = meta['epoch']
    clock_field = 'cosmic_time_gyr' if meta['parameters']['comoving'] else 'elapsed_time_gyr'
    time = epoch.get(clock_field)
    if time is None:raise ValueError('Native gas analysis requires hashed cosmological clock or isolated physical time')
    gas['epoch']=epoch;gas['namespace']=namespace
    network=meta['parameters']['grackle_network']
    physics={'non_equilibrium_primordial_chemistry':network>=1,'h2_cooling':network>=2,'hd_cooling':network==3,
        'resolved_single_phase_temperature':bool(gas['gas_temperature_available'].any())}
    return {'time_myr':time*1000,'gas':gas,'gas_namespace':namespace,'host_identity':host_identity,'host_key_by_gas_id':resolution_fields['host_key_by_gas_id'],
        'source_hashes':sources,'native_metadata':meta,'actual_native_physics':physics},sources



def native_births(root,before,after,*,journal_directory,namespace,epoch_by_id,memory_bytes,observation_epoch=None):
    from .stars import read_native_ssp_birth_journal,native_ssp_event_physical_arrays
    from .cosmology import run_proper_time_clock,interpolate_proper_time
    if before['parameters']!=after['parameters']:raise ValueError('Birth interval changes native source model')
    p=before['parameters']
    arrays,commits,sources=read_native_ssp_birth_journal(journal_directory,start_commit=before['commit_count'],stop_commit=after['commit_count'],
        start_count=before['event_count'],stop_count=after['event_count'],native_time_interval=[before['native_time'],after['native_time']],
        expected_parameters={key:p[key] for key in ('unit_mass_g','unit_time_s','unit_velocity_cm_s','unit_length_cm','hubble_param','factor_sn','comoving','seed')},memory_bytes=memory_bytes)
    physical=native_ssp_event_physical_arrays(arrays,p)
    if p['comoving']:
        a,t,clock=run_proper_time_clock(root)
        if a is None:raise ValueError('Missing cosmological birth clock')
        times=np.array([interpolate_proper_time(a,t,q)*1000 for q in physical['birth_time']]);sources[clock['path']]=clock['sha256']
    else:
        from .foundation import SECONDS_PER_YEAR
        times=physical['birth_time']*p['unit_time_s']/SECONDS_PER_YEAR/1e6
    result=[]
    for i,identifier in enumerate(physical['ssp_id']):
        recorded_epoch=epoch_by_id.get(str(int(identifier)),observation_epoch)
        if recorded_epoch is None:raise ValueError('Birth observation epoch mapping incomplete')
        if type(recorded_epoch) is not int or recorded_epoch<0:raise ValueError('Exact actual birth observation epoch required')
        result.append({'stellar_key':[namespace,int(identifier),recorded_epoch],
            'observation_epoch_origin':'native_graph' if str(int(identifier)) in epoch_by_id else 'actual_interval_end_no_graph_node','parent_id':str(int(physical['parent_id'][i])),
            'applied_time_myr':float(times[i]),'gross_initial_mass_msun':float(physical['gross_mass'][i]),'locked_birth_mass_msun':float(physical['locked_mass'][i]),
            'metallicity_mass_fraction':float(physical['metallicity'][i]),'source_hashes':sources})
    return result,sources



def stellar_sps_truth(root,snapshot,*,memory_bytes,namespace):
    """Read native SSP birth masses and derive ages from the saved run clock.

    Cosmological ages use the background-derived proper clock. Isolated ages
    convert the native L/V time unit to Myr. Gross initially formed mass
    normalizes synthesis; native current mass stays independent of FSPS mass.
    """
    from .stars import read_native_hydro_census
    from .cosmology import run_proper_time_clock,interpolate_proper_time
    from .foundation import SECONDS_PER_YEAR
    arrays,meta,sources,files=read_native_hydro_census(root,snapshot,memory_bytes=memory_bytes)
    use=arrays['species']==4
    if np.any(arrays['SSPBirthStateVersion'][use]!=1):raise ValueError('SPS requires committed native SSP birth states')
    birth=arrays['SSPBirthNativeTime'][use];p=meta['parameters']
    if p['comoving']:
        a,t,clock=run_proper_time_clock(root)
        if a is None:raise ValueError('SPS requires the actual cosmological proper-time clock')
        now=interpolate_proper_time(a,t,meta['native_time'])*1000
        born=np.array([interpolate_proper_time(a,t,q)*1000 for q in birth]);sources[clock['path']]=clock['sha256']
    else:
        now=meta['native_time']*p['unit_time_s']/SECONDS_PER_YEAR/1e6;born=birth*p['unit_time_s']/SECONDS_PER_YEAR/1e6
    return {'particle_ids':arrays['particle_ids'][use],'species':arrays['species'][use],'gross_initial_mass_msun':arrays['SSPInitialMass'][use],
        'current_mass_msun':arrays['mass_msun'][use],'age_myr':now-born,'metallicity_mass_fraction':arrays['SSPBirthMetallicity'][use],
        'position_kpc':arrays['positions_proper_kpc'][use],'namespace':namespace,'initial_mass_convention':'native_gross_formed_mass_before_prompt_return',
        'birth_source_hashes':sources},sources



LSUN_ERG_S=3.839e33  # pinned FSPS sps_vars.f90 luminosity unit



C_ANGSTROM_S=2.99792458e18



TEN_PC_CM=3.085677581491367e19



PIN=Path(__file__).resolve().parents[1]/'configs/sps.upstream.json'



def pinned_fsps(request):
    pin=json.loads(PIN.read_text())
    for record in pin['repositories']:
        directory=PIN.parents[1]/record['directory']
        for name,expected in record['source_sha256'].items():
            if digest(directory/name)!=expected:raise ValueError('Pinned SPS source/table changed: '+name)
    distribution=importlib.metadata.distribution('fsps')
    from core.native import verified_python_backend
    wrapper = next(record for record in pin['repositories'] if record['name']=='python-fsps')
    installation = verified_python_backend('fsps', wrapper, distribution=distribution)
    fsps=importlib.import_module('fsps')
    if str(Path(fsps.__file__).resolve()) not in installation['installed_assets']:
        raise ValueError('Imported FSPS module is outside its verified distribution')
    if wrapper['revision'] != pin['python_fsps']['revision']:
        raise ValueError('FSPS wrapper/source pins disagree')
    home=Path(os.environ.get('SPS_HOME','')).resolve()
    if not os.environ.get('SPS_HOME') or home!=Path(request['sps_home']).resolve():raise ValueError('SPS_HOME must equal the declared hashed FSPS data directory')
    assets=request['data_source_sha256']
    if not assets or any(digest(home/name)!=checksum for name,checksum in assets.items()):raise ValueError('Complete SPS table source checksums required')
    # All read tables are bound, including IMF, filters and isochrones.
    for subdirectory in ('data','ISOCHRONES','SPECTRA','dust','nebular'):
        files=[p for p in (home/subdirectory).rglob('*') if p.is_file()]
        from .release import nonruntime_document
        files=[p for p in files if not nonruntime_document(p.relative_to(home))]
        if not files or any(str(p.relative_to(home)) not in assets for p in files):raise ValueError('SPS data directory contains undeclared model assets: '+subdirectory)
    parameters=dict(request['fsps_parameters'])
    for key,value in {'sfh':0,'zcontinuous':1,'dust1':0.,'dust2':0.,'add_dust_emission':False,'add_neb_emission':False,'compute_vega_mags':False}.items():
        if key in parameters and parameters[key]!=value:raise ValueError('Intrinsic SSP synthesis requires '+key+'='+str(value))
        parameters[key]=value
    if 'imf_type' not in parameters:raise ValueError('Explicit pinned IMF required')
    population=fsps.StellarPopulation(**parameters)
    libraries=[q.decode() if isinstance(q,bytes) else str(q) for q in population.libraries]
    if libraries!=request['expected_libraries']:raise ValueError('Compiled FSPS isochrone/spectral/dust libraries differ from requested model')
    sources={**installation['source_hashes'],str(PIN):digest(PIN),**{str(home/name):checksum for name,checksum in assets.items()}}
    for record in pin['repositories']:
        directory=PIN.parents[1]/record['directory'];sources.update({str(directory/name):checksum for name,checksum in record['source_sha256'].items()})
    for file in distribution.files or []:
        path=Path(distribution.locate_file(file)).resolve()
        if path.is_file() and path.suffix in ('.py','.so'):sources[str(path)]=digest(path)
    return population,{'model':'FSPS','libraries':libraries,'imf':parameters['imf_type'],'parameters':parameters,'source_hashes':sources,'normalization':'one solar mass initially formed','filters':request['filters']}



def synthesize_stellar_light(stars,request):
    from .data import exact_uint64_ids
    identifiers=exact_uint64_ids(stars['particle_ids'],allow_empty=True)
    if len(np.unique(identifiers))!=len(identifiers):raise ValueError('Duplicate native stellar IDs')
    n=len(identifiers);species=np.asarray(stars['species'])
    if species.shape!=(n,) or np.any(species!=4):raise ValueError('SPS accepts actual native SSP species 4 only')
    initial=finite(stars['gross_initial_mass_msun'],(n,));current=finite(stars['current_mass_msun'],(n,))
    age=finite(stars['age_myr'],(n,))/1000;z=finite(stars['metallicity_mass_fraction'],(n,))
    if np.any(initial<=0) or np.any(current<0):raise ValueError('Positive formed mass and nonnegative present stellar mass required')
    if stars['initial_mass_convention']!='native_gross_formed_mass_before_prompt_return':raise ValueError('SSP mass normalization must use committed native gross birth mass')
    if not stars['birth_source_hashes']:raise ValueError('SPS requires native birth mass provenance')
    lo,hi=request['age_domain_gyr'];zlo,zhi=request['metallicity_domain_mass_fraction']
    if not np.isfinite([lo,hi,zlo,zhi]).all() or not 0<lo<hi or not 0<zlo<zhi:raise ValueError('Positive ordered explicit SSP model domains required')
    age_supported=(age>=lo)&(age<=hi);metallicity_supported=(z>=zlo)&(z<=zhi)&(z>0)
    supported=age_supported&metallicity_supported
    population,model=pinned_fsps(request['backend']);filters=request['backend']['filters'];nf=len(filters)
    if not nf or len(filters)!=len(set(filters)):raise ValueError('Unique nonempty FSPS filter list required')
    population.params['add_stellar_remnants']=True
    wave=np.asarray(population.wavelengths,float)
    if n*len(wave)*8>positive(request['memory_gib'],'memory_gib')*1024**3:raise MemoryError('Requested SSP spectra exceed workspace')
    spectra=np.full((n,len(wave)),np.nan);lbol=np.full(n,np.nan);mass_model=np.full(n,np.nan);magnitudes=np.full((n,nf),np.nan)
    grid=10**np.asarray(population.ssp_ages)/1e9
    if lo<grid.min() or hi>grid.max():raise ValueError('Requested age support exceeds actual FSPS SSP tables')
    zsolar=positive(request['solar_metallicity_mass_fraction'],'solar metallicity')
    if not np.isclose(zsolar,float(population.solar_metallicity),rtol=1e-12):raise ValueError('Solar metallicity must match compiled FSPS isochrone convention')
    actual_metallicities=np.asarray(population.zlegend,float)
    if zlo<actual_metallicities.min() or zhi>actual_metallicities.max():raise ValueError('Requested metallicity support exceeds FSPS isochrone tables')
    for i in np.flatnonzero(supported):
        population.params['logzsol']=np.log10(z[i]/zsolar)
        wavelength,spectrum=population.get_spectrum(tage=float(age[i]),peraa=True)
        if not np.array_equal(wave,wavelength):raise ValueError('FSPS wavelength grid changed across SSPs')
        formed=float(population.formed_mass)
        if not np.isclose(formed,1.,rtol=1e-6):raise ValueError('FSPS SSP differs from one solar mass formed normalization')
        spectra[i]=finite(spectrum,wave.shape)*initial[i]
        lbol[i]=10**float(population.log_lbol)*initial[i];mass_model[i]=float(population.stellar_mass)*initial[i]
        magnitudes[i]=population.get_mags(tage=float(age[i]),bands=filters)-2.5*np.log10(initial[i])
    # Absolute AB magnitude converts to rest-frame spectral luminosity, not Lsun bolometric.
    band_lnu=4*np.pi*TEN_PC_CM**2*10**(-.4*(magnitudes+48.6))
    colors=magnitudes[:,:,None]-magnitudes[:,None,:]
    ml=np.divide(current[:,None],band_lnu,out=np.full_like(band_lnu,np.nan),where=band_lnu>0)
    report={'schema':'native-stellar-sps-v1','namespace':stars['namespace'],'stellar_namespace':stars['namespace'],'model':model,'filters':filters,
        'age_domain_gyr':[lo,hi],'metallicity_domain_mass_fraction':[zlo,zhi],'dust':'intrinsic_dust_free',
        'supported_count':int(supported.sum()),'unsupported_count':int((~supported).sum()),'unsupported_age_count':int((~age_supported).sum()),
        'unsupported_metallicity_count':int((~metallicity_supported).sum()),'support_definition':'intersection of unchanged requested and actual model domains; no clipping, replacement, or extrapolation','initial_mass_convention':stars['initial_mass_convention'],
        'native_vs_sps_mass_note':'FSPS return fractions are model predictions; native present mass is preserved separately',
        'science_acceptance':'NOT_ACCEPTED','units':{'wavelength_angstrom':'Angstrom','spectrum_lsun_per_angstrom':'Lsun / Angstrom','bolometric_luminosity_lsun':'Lsun',
        'band_luminosity_erg_s_hz':'erg s^-1 Hz^-1','mass_to_light_msun_per_erg_s_hz':'Msun / (erg s^-1 Hz^-1)','absolute_ab_magnitude':'AB mag','colors_ab_magnitude':'AB mag'}}
    return report,{'particle_ids':identifiers,'wavelength_angstrom':wave,'spectrum_lsun_per_angstrom':spectra,'bolometric_luminosity_lsun':lbol,
        'band_luminosity_erg_s_hz':band_lnu,'absolute_ab_magnitude':magnitudes,'colors_ab_magnitude':colors,'mass_to_light_msun_per_erg_s_hz':ml,
        'native_current_mass_msun':current,'sps_surviving_mass_msun':mass_model,'initial_formed_mass_msun':initial,'sps_supported':supported,'sps_age_supported':age_supported,'sps_metallicity_supported':metallicity_supported}

