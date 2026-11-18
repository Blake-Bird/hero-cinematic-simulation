"""Stellar observation models: spectral-age chronometers and distance ladders.

Native SSP particles provide host populations, not resolved Cepheid pulsations
or SN explosions. Indicator prescriptions and generating/fitting spectral assets
remain explicit. Ages are Gyr, wavelengths Angstrom, magnitudes in the authored
photometric system; distances are physical Mpc without h. Estimators do not read
native formation-age or H0 truth. References: Moresco arXiv:2003.07362; SH0ES
arXiv:2112.04510; Pantheon+ arXiv:2202.04077.
"""
from collections import Counter
from dataclasses import dataclass, asdict
from functools import cached_property
from pathlib import Path
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.linalg import solve_triangular
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import minimize
from scipy.special import log_ndtr, logsumexp
from .foundation import file_sha256
from .stream_common import finite, positive, integer
from .observation_data import (array, covariance, exact_ids, observed_catalogue, read_observations,
    gls, freeze_result, gaussian_observable, HUBBLE_GYR, C_KMS, canonical)


@dataclass(frozen=True)
class SpectralLibrary:
    """Immutable SSP grid: age Gyr, metallicity mass fraction Z, wavelength Angstrom.

    Spectral normalization belongs to the pinned asset. Generation supplies an
    explicit host flux scale; age fitting profiles its continuum amplitude.
    """
    wavelength_angstrom: np.ndarray
    age_gyr: np.ndarray
    metallicity: np.ndarray
    luminosity: np.ndarray
    model_id: str
    revision: str
    asset_hash: str
    imf: str
    intrinsic_resolution_sigma_angstrom: float

    def __post_init__(self):
        w=array(self.wavelength_angstrom,'SPS wavelength',1,positive=True)
        a=array(self.age_gyr,'SPS ages',1,positive=True)
        m=array(self.metallicity,'SPS metallicity',1,positive=True)
        f=array(self.luminosity,'SPS luminosity',3)
        if f.shape!=(len(a),len(m),len(w)) or min(len(w),len(a),len(m))<2:
            raise ValueError('SPS library shape must be (age,metallicity,wavelength)')
        if np.any(m>1):raise ValueError('SSP metallicity axis must be a mass fraction Z in (0,1]')
        if np.any(f<0) or any(np.any(np.diff(x)<=0) for x in (w,a,m)):
            raise ValueError('SPS axes must increase and luminosities be nonnegative')
        if not all((self.model_id,self.revision,self.asset_hash,self.imf)):
            raise ValueError('SPS library requires model/revision/content hash/IMF')
        resolution=float(array(self.intrinsic_resolution_sigma_angstrom,'SPS resolution sigma Angstrom',0))
        if resolution<0:
            raise ValueError('SPS resolution sigma cannot be negative')
        object.__setattr__(self,'intrinsic_resolution_sigma_angstrom',resolution)
        for name,value in [('wavelength_angstrom',w),('age_gyr',a),('metallicity',m),('luminosity',f)]:
            value=value.copy();value.flags.writeable=False;object.__setattr__(self,name,value)

    @cached_property
    def _interpolator(self):
        # The model arrays are immutable. Construct their interpolator once,
        # rather than rebuilding the same grid for every SFH component.
        return RegularGridInterpolator((self.age_gyr,self.metallicity),self.luminosity,bounds_error=True)

    def spectrum(self, age, metallicity):
        if not self.age_gyr[0]<=age<=self.age_gyr[-1] or not self.metallicity[0]<=metallicity<=self.metallicity[-1]:
            raise ValueError('SPS interpolation requested outside pinned library coverage')
        return self._interpolator([[age,metallicity]])[0]

    def record(self):
        return {k:getattr(self,k) for k in ('model_id','revision','asset_hash','imf',
                                         'intrinsic_resolution_sigma_angstrom')}

    @classmethod
    def from_npz(cls,path,metadata):
        path=Path(path)
        if file_sha256(path)!=metadata['asset_hash']:raise ValueError('SPS asset hash mismatch')
        with np.load(path,allow_pickle=False) as source:
            return cls(source['wavelength_angstrom'],source['age_gyr'],source['metallicity'],
                       source['luminosity'],**metadata)


def composite_spectrum(library, ages_gyr, formed_mass_weights, metallicities, *, dust_ebv,
                       attenuation_curve):
    ages=array(ages_gyr,'SFH ages',1,positive=True)
    weights=array(formed_mass_weights,'formed mass weights',1)
    metals=array(metallicities,'SFH metallicities',1,positive=True)
    if ages.shape!=weights.shape or metals.shape!=ages.shape or np.any(weights<0) or weights.sum()<=0:
        raise ValueError('SFH ages/metals/positive total formed mass must match')
    dust_ebv=float(array(dust_ebv,'dust color excess',0))
    if dust_ebv<0:raise ValueError('Dust color excess must be nonnegative')
    k=array(attenuation_curve,'pinned attenuation curve',1)
    if k.shape!=library.wavelength_angstrom.shape or np.any(k<0):
        raise ValueError('Attenuation curve must cover all library wavelengths')
    # A zero-formed-mass component contributes no spectrum and needs no
    # interpolation outside the library's age/metallicity support.
    spectrum=sum(w*library.spectrum(a,z) for a,w,z in zip(ages,weights,metals) if w>0)
    return array(spectrum*10**(-.4*dust_ebv*k),'attenuated composite spectrum',1)


def render_passive_spectra(hosts, library, instrument, *, seed):
    """Generate observed spectra from declared SFHs; retain truth outside observation product."""
    ids=exact_ids(hosts['object_id']);rng=np.random.default_rng(integer(seed,'spectral generation seed',0))
    w=array(instrument['rest_wavelength_angstrom'],'observed rest wavelength',1,positive=True)
    if np.any(np.diff(w)<=0):raise ValueError('Observed wavelength must increase')
    if w[0]<library.wavelength_angstrom[0] or w[-1]>library.wavelength_angstrom[-1]:
        raise ValueError('Observed spectrum exceeds SPS wavelength support')
    sigma=float(array(instrument['resolution_sigma_angstrom'],'instrument resolution sigma Angstrom',0))
    if sigma<library.intrinsic_resolution_sigma_angstrom:
        raise ValueError('Instrument cannot sharpen an intrinsically lower-resolution library')
    gridstep=np.diff(library.wavelength_angstrom)
    if not np.allclose(gridstep,gridstep[0],rtol=1e-4):
        raise ValueError('Gaussian LSF convolution requires a uniform library wavelength grid')
    smooth=np.sqrt(sigma*sigma-library.intrinsic_resolution_sigma_angstrom**2)/gridstep[0]
    flux_scale=finite(hosts['flux_scale'],(len(ids),),'host flux scale')
    if np.any(flux_scale<=0) or len(hosts['sfh'])!=len(ids):raise ValueError('Positive aligned host flux scale and SFH required')
    signal_to_noise=positive(instrument['signal_to_noise'],'signal to noise')
    error_floor=positive(instrument['flux_error_floor'],'spectral flux error floor')
    spectra=[];covariances=[]
    for j,ident in enumerate(ids):
        sfh=hosts['sfh'][j]
        f=composite_spectrum(library,sfh['ages_gyr'],sfh['formed_mass_weights'],sfh['metallicities'],
            dust_ebv=sfh['dust_ebv'],attenuation_curve=instrument['attenuation_curve'])
        if smooth>0:f=gaussian_filter1d(f,smooth,mode='nearest')
        f=np.interp(w,library.wavelength_angstrom,f)*flux_scale[j]
        sig=np.maximum(np.abs(f)/signal_to_noise,error_floor)
        cc=np.diag(sig**2)
        for systematic in instrument.get('correlated_flux_modes',[]):
            mode=array(systematic['fractional_pattern'],'spectral systematic',1)*f
            if mode.shape!=f.shape:raise ValueError('Spectral calibration mode shape differs')
            mode_sigma=float(array(systematic['sigma'],'spectral systematic sigma',0))
            if mode_sigma<0:raise ValueError('Spectral systematic sigma must be nonnegative')
            cc+=mode_sigma**2*np.outer(mode,mode)
        spectra.append(f+rng.multivariate_normal(np.zeros(len(w)),cc));covariances.append(cc)
    # Passive selection diagnostics are measured from the noisy observed spectrum.
    diagnostics=[measure_passive_spectral_indices(w,f) for f in spectra]
    columns={'object_id':ids,'redshift':hosts['redshift'],'flux':spectra,
             'dn4000':[d['dn4000'] for d in diagnostics],
             'equivalent_width_angstrom':[d['hdelta_equivalent_width_angstrom'] for d in diagnostics],
             # Passive cuts use the measured spectrum and quoted noise, not
             # the requested S/N target that may be limited by the error floor.
             'signal_to_noise':[float(np.median(f/np.sqrt(np.diag(cc)))) for f,cc in zip(spectra,covariances)]}
    return observed_catalogue(columns,units={'object_id':'1','redshift':'1','flux':instrument['flux_units'],
        'dn4000':'1','equivalent_width_angstrom':'angstrom','signal_to_noise':'1'},namespace=hosts['namespace'],
        selection={'seed':seed,'generating_sps':library.record()},calibration={'wavelength_angstrom':w,
        'flux_covariances':covariances,'instrument_resolution_sigma_angstrom':sigma,
        'signal_to_noise_definition':'median observed flux / per-pixel quoted standard deviation',
        'instrument_target_signal_to_noise':signal_to_noise})


def select_from_observed_spectra_and_catalogues(observations, selection):
    """Select measured passive diagnostics while preserving exact source IDs."""
    observed = read_observations(observations)
    table = observed['columns']
    ids = exact_ids(table['object_id'])
    count = len(ids)
    mask = np.ones(count, dtype=bool)
    for field, limits in selection['observed_cuts'].items():
        values = finite(table[field], (count,), field)
        low, high = finite(limits, (2,), field+' selection bounds')
        if low > high:
            raise ValueError('Passive selection bounds must be ordered')
        mask &= (values >= low) & (values <= high)
    if mask.sum() < integer(selection['minimum_objects'], 'minimum passive objects'):
        raise ValueError('Observed passive sample too small; no formation-age truth substitution')
    columns = {}
    for key, values in table.items():
        if len(values) != count:
            raise ValueError('Observed spectral column length differs: '+key)
        # NumPy dtype inference on a Python integer list can round uint64 IDs.
        columns[key] = (ids[mask] if key == 'object_id' else np.asarray(values)[mask]).tolist()
    calibration = dict(observed['calibration'])
    calibration['flux_covariances'] = np.asarray(calibration['flux_covariances'])[mask].tolist()
    return observed_catalogue(columns, units=observed['units'], namespace=observed['namespace'],
        selection=dict(observed['selection'], passive_selection=selection, input_count=count, selected_count=int(mask.sum())),
        calibration=calibration)


def _template_bank(library, wavelength, spec, resolution_sigma):
    """Explicit old+young SFH grid, variable metallicity/dust and recorded priors."""
    step=np.diff(library.wavelength_angstrom)
    if not np.allclose(step,step[0],rtol=1e-4):raise ValueError('Library wavelength grid must be uniform for LSF convolution')
    resolution_sigma=float(array(resolution_sigma,'fitting resolution sigma Angstrom',0))
    if resolution_sigma<library.intrinsic_resolution_sigma_angstrom:
        raise ValueError('Fitting library resolution exceeds observed resolution')
    smooth=np.sqrt(resolution_sigma**2-library.intrinsic_resolution_sigma_angstrom**2)/step[0]
    rows=[];templates=[]
    for entry in spec['sfh_grid']:
        age=float(entry['old_age_gyr']); young=float(entry['young_age_gyr']);fraction=float(entry['young_mass_fraction'])
        if not 0<=fraction<=1:raise ValueError('Rejuvenation mass fraction outside [0,1]')
        flux=composite_spectrum(library,[age,young],[1-fraction,fraction],
            [entry['metallicity'],entry.get('young_metallicity',entry['metallicity'])],
            dust_ebv=entry['dust_ebv'],attenuation_curve=spec['attenuation_curve'])
        if smooth>0:flux=gaussian_filter1d(flux,smooth,mode='nearest')
        interp=np.interp(wavelength,library.wavelength_angstrom,flux)
        if np.max(interp)<=0:raise ValueError('SPS template has zero luminosity')
        templates.append(interp/np.max(interp));rows.append(entry)
    if not rows:raise ValueError('Spectral-age fitting grid is empty')
    return rows,np.asarray(templates)


def fit_spectral_ages(observations, fitting_sps_model, spec):
    """Profile continuum/SFH templates against measured spectra and covariance.

    Grid weights combine profiled chi-square with the authored prior/cell weights;
    they are not a fully marginalized spectral posterior. One covariance factor
    per object serves every template. Age variances include the declared grid
    floor; stored grid-quantile intervals exclude that floor and shared modes.
    """
    observed = read_observations(observations)
    calibration, columns = observed['calibration'], observed['columns']
    ids = exact_ids(columns['object_id'])
    if not len(ids):
        raise ValueError('Spectral-age sample is empty')
    generating = observed['selection'].get('generating_sps')
    distinct = spec.get('require_distinct_sps', False)
    if type(distinct) is not bool:
        raise ValueError('Distinct-SPS experiment setting must be Boolean')
    if distinct and (not generating or not generating.get('asset_hash')):
        raise ValueError('Distinct-SPS experiment requires the generating asset identity')
    if distinct and generating['asset_hash'] == fitting_sps_model.asset_hash:
        raise ValueError('Model-mismatch experiment requires distinct generating/fitting SPS assets')
    wavelength = array(calibration['wavelength_angstrom'], 'wavelength Angstrom', 1, positive=True)
    if len(wavelength) < 2 or np.any(np.diff(wavelength) <= 0):
        raise ValueError('Spectral fitting wavelengths must increase')
    if wavelength[0] < fitting_sps_model.wavelength_angstrom[0] or wavelength[-1] > fitting_sps_model.wavelength_angstrom[-1]:
        raise ValueError('Fitting SPS library lacks observed wavelength coverage')
    entries, templates = _template_bank(fitting_sps_model, wavelength, spec, calibration['instrument_resolution_sigma_angstrom'])
    keep = np.ones(len(wavelength), dtype=bool)
    for limits in spec.get('masked_wavelength_ranges', []):
        low, high = finite(limits, (2,), 'masked wavelength range')
        if low > high:
            raise ValueError('Masked wavelength range must be ordered')
        keep &= ~((wavelength >= low) & (wavelength <= high))
    minimum = integer(spec['minimum_spectral_pixels'], 'minimum spectral pixels', 2)
    order = integer(spec['continuum_order'], 'continuum polynomial order', 0)
    if keep.sum() < minimum or keep.sum() <= order+1:
        raise ValueError('Too few unmasked pixels to constrain spectral shape beyond the continuum')
    coordinate = 2*(wavelength[keep]-wavelength[keep].min())/np.ptp(wavelength[keep])-1
    from numpy.polynomial.legendre import legvander
    continuum = legvander(coordinate, order)
    age_grid = array([entry['old_age_gyr'] for entry in entries], 'template ages Gyr', 1, positive=True)
    young = array([entry['young_mass_fraction'] for entry in entries], 'young mass fractions', 1)
    metals = array([entry['metallicity'] for entry in entries], 'template metallicities', 1, positive=True)
    priors = array([entry['prior_weight'] for entry in entries], 'SFH prior/cell weights', 1, positive=True)
    floor = positive(spec['age_grid_uncertainty_floor_gyr'], 'age grid uncertainty floor Gyr')
    redshift = finite(columns['redshift'], (len(ids),), 'spectral redshifts')
    fluxes = finite(columns['flux'], (len(ids), len(wavelength)), 'observed spectra')
    if len(calibration['flux_covariances']) != len(ids):
        raise ValueError('Spectrum covariance count differs from source identities')
    rows = []
    age_order = np.argsort(age_grid)
    for j, identity in enumerate(ids):
        full_covariance = covariance(calibration['flux_covariances'][j], len(wavelength))
        cholesky = np.linalg.cholesky(full_covariance[np.ix_(keep, keep)])
        whitened_flux = solve_triangular(cholesky, fluxes[j, keep], lower=True)
        scores, amplitudes = [], []
        for template in templates:
            design = template[keep, None]*continuum
            whitened = solve_triangular(cholesky, design, lower=True)
            coefficients, _, rank, _ = np.linalg.lstsq(whitened, whitened_flux, rcond=None)
            if rank != design.shape[1]:
                raise ValueError('Spectral continuum/template coordinates are not identifiable')
            residual = whitened_flux-whitened@coefficients
            chi2 = float(residual@residual)
            if not np.isfinite(chi2):
                raise ValueError('Spectral residual exceeds the representable numeric range')
            scores.append(-.5*chi2 if coefficients[0] > 0 else -np.inf)
            amplitudes.append(coefficients)
        if not np.isfinite(scores).any():
            raise ValueError('No positive-amplitude spectral template fits the sample')
        log_weights = np.asarray(scores)+np.log(priors)
        probability = np.exp(log_weights-logsumexp(log_weights))
        mean = float(probability@age_grid)
        variance = float(probability@((age_grid-mean)**2))+floor**2
        best = int(np.argmax(log_weights))
        cdf = np.cumsum(probability[age_order])
        cdf /= cdf[-1]
        interval = np.interp([.16, .84], cdf, age_grid[age_order])
        rows.append({
            'object_id': str(identity), 'redshift': float(redshift[j]), 'age_gyr': mean,
            'age_variance_gyr2': variance, 'age_interval_68': interval.tolist(),
            'age_interval_68_scope': 'profile-grid quantiles only; excludes grid floor and shared age modes',
            'young_fraction_mean': float(probability@young), 'metallicity_mean': float(probability@metals),
            'age_metallicity_covariance': float(probability@((age_grid-mean)*(metals-probability@metals))),
            'best_template_index': best, 'best_continuum_coefficients': amplitudes[best].tolist(),
            'grid_boundary_probability': float(probability[(age_grid==age_grid.min()) | (age_grid==age_grid.max())].sum()),
            'template_weights': probability.tolist(),
        })
    age_covariance = np.diag([row['age_variance_gyr2'] for row in rows])
    for systematic in spec.get('correlated_age_systematics', []):
        pattern = finite(systematic['response_gyr'], (len(rows),), 'correlated age response Gyr')
        sigma = float(finite(systematic['sigma'], (), 'age systematic amplitude'))
        if sigma < 0:
            raise ValueError('Age-systematic sigma must be nonnegative')
        age_covariance += sigma**2*np.outer(pattern, pattern)
    covariance(age_covariance, len(rows), 'spectral age covariance Gyr^2')
    return {
        'role': 'estimator_observations', 'schema': 'spectral-age-observations-v1',
        'namespace': observed['namespace'], 'rows': rows, 'age_covariance_gyr2': age_covariance.tolist(),
        'selection': observed['selection'], 'fitting_sps': fitting_sps_model.record(), 'spectral_fitting': spec,
        'uncertainty': 'SFH grid profile-likelihood weights with explicit priors, age floor and declared shared modes',
    }


def fit_differential_ages_with_covariance(age_observations, spec, *, realization_id=None):
    """Estimate H(z) from a resolved negative age-versus-redshift slope.

    For ages in Gyr, H=-HUBBLE_GYR/[(1+z)*dt/dz] in km/s/Mpc. The response
    dH/dage propagates the full age covariance, including correlations between
    overlapping windows. Unsupported slopes, age-grid boundaries and lever arms
    retain explicit unresolved statuses. No background H0 is read by this fit.
    """
    obs = read_observations(age_observations)
    rows = obs['rows']
    z = array([r['redshift'] for r in rows], 'redshifts', 1)
    ages = array([r['age_gyr'] for r in rows], 'estimated spectral ages', 1, positive=True)
    c = covariance(obs['age_covariance_gyr2'], len(rows))
    if 'progenitor_age_correction_gyr' in spec:
        correction = array(spec['progenitor_age_correction_gyr'], 'observational progenitor correction', 1)
        if correction.shape != ages.shape:
            raise ValueError('Progenitor correction must match sample')
        ages = ages - correction
        if np.any(ages <= 0):
            raise ValueError('Progenitor-corrected stellar ages must remain positive')
    if 'progenitor_covariance_gyr2' in spec:
        c = c + covariance(spec['progenitor_covariance_gyr2'], len(rows), semidefinite=True)
    integer(spec['polynomial_order'], 'differential-age polynomial order', 1)
    integer(spec['minimum_objects_per_window'], 'minimum chronometer objects', 2)
    positive(spec['minimum_redshift_span'], 'minimum redshift lever arm')
    positive(spec['minimum_negative_slope_significance'], 'negative-slope significance')
    boundary = finite([r['grid_boundary_probability'] for r in rows], (len(rows),), 'SPS boundary probabilities')
    maximum = float(array(spec['maximum_grid_boundary_probability'], 'maximum boundary probability', 0))
    if not 0 <= maximum <= 1 or np.any((boundary < 0) | (boundary > 1)):
        raise ValueError('Grid boundary probabilities must lie in [0,1]')
    exact_ids([row['object_id'] for row in rows])
    fit_rows = []
    influences = []
    values = {}
    for i, window in enumerate(spec['redshift_windows']):
        lo, hi = finite(window['range'], (2,), 'chronometer redshift window')
        center = float(array(window['center'], 'chronometer center', 0))
        if lo < 0 or hi <= lo:
            raise ValueError('Chronometer windows require ordered nonnegative redshifts')
        keep = (z >= lo) & (z <= hi)
        if not lo <= center <= hi:
            raise ValueError('Chronometer window center outside its range')
        if keep.sum() < spec['minimum_objects_per_window']:
            fit_rows.append({'window': window, 'status': 'INSUFFICIENT_PASSIVE_SAMPLE'})
            continue
        if np.ptp(z[keep]) < spec['minimum_redshift_span']:
            fit_rows.append({'window': window, 'status': 'INSUFFICIENT_REDSHIFT_LEVER_ARM'})
            continue
        if np.any(boundary[keep] > maximum):
            fit_rows.append({'window': window, 'status': 'SPS_AGE_COVERAGE_UNRESOLVED'})
            continue
        x = z[keep] - center
        a = np.column_stack([x ** power for power in range(spec['polynomial_order'] + 1)])
        if spec['polynomial_order'] < 1:
            raise ValueError('Differential age relation needs at least a linear term')
        fit = gls(a, ages[keep], c[np.ix_(keep, keep)], parameter_names=['age_intercept', 'dt_dz'] + [f'age_z_order_{j}' for j in range(2, spec['polynomial_order'] + 1)])
        fit['parameter_units'] = ['Gyr'] * len(fit['parameter_names'])
        slope = float(fit['mean'][1])
        var = float(fit['covariance'][1, 1])
        if slope >= 0 or -slope < spec['minimum_negative_slope_significance'] * np.sqrt(var):
            fit_rows.append({'window': window, 'status': 'AGE_SLOPE_UNRESOLVED', 'fit': canonical(fit)})
            continue
        h = -HUBBLE_GYR / ((1 + center) * slope)
        derivative = HUBBLE_GYR / ((1 + center) * slope * slope)
        value = gaussian_observable(h, derivative ** 2 * var, 'km/s/Mpc', center)
        if slope + np.sqrt(var) < 0:
            value['interval_68'] = sorted([-HUBBLE_GYR / ((1 + center) * (slope - np.sqrt(var))), -HUBBLE_GYR / ((1 + center) * (slope + np.sqrt(var)))])
        values[f'H_z:{i}'] = value
        response = np.zeros(len(z))
        response[keep] = derivative * fit['influence'][1]
        influences.append(response)
        fit_rows.append({'window': window, 'status': 'MEASURED', 'fit': canonical(fit), 'observable_key': f'H_z:{i}', 'object_count': int(keep.sum())})
    if not values:
        raise ValueError('No chronometer window resolves a negative differential-age slope')
    response = np.asarray(influences)
    hcov = response @ c @ response.T
    return freeze_result('cosmic_chronometer', {'windows': fit_rows, 'H_z_covariance': hcov, 'observable_order': list(values), 'age_to_H_response': response}, observables=values, selection=obs['selection'], object_ids=[r['object_id'] for r in rows], namespace=obs['namespace'], model={'differential_age': spec, 'fitting_sps': obs.get('fitting_sps'), 'spectral_fitting': obs.get('spectral_fitting')}, realization_id=realization_id, nuisance={'progenitor_bias': 'explicit correction/covariance required when requested', 'fundamental_H0_claim': False})




def measure_passive_spectral_indices(wavelength_angstrom, flux):
    """Narrow D4000 from mean F_nu and Hdelta absorption from local continuum.

    Definitions: blue 3850–3950 A, red 4000–4100 A; Hdelta feature
    4083.5–4122.25 A with continua 4041.6–4079.75/4128.5–4161 A.
    Positive equivalent width denotes absorption. Coverage is explicit.
    """
    w=array(wavelength_angstrom,'spectral-index wavelengths',1,positive=True)
    f=array(flux,'spectral-index flux',1)
    if f.shape!=w.shape or np.any(np.diff(w)<=0):raise ValueError('Spectral-index axes invalid')
    def band(lo,hi):
        if w[0]>lo or w[-1]<hi:raise ValueError('Passive spectral index lacks wavelength coverage')
        use=(w>lo)&(w<hi)
        ww=np.r_[lo,w[use],hi];ff=np.interp(ww,w,f)
        return ww,ff
    wb,fb=band(3850,3950);wr,fr=band(4000,4100)
    blue=np.trapezoid(fb*wb**2,x=wb)/100
    red=np.trapezoid(fr*wr**2,x=wr)/100
    if blue<=0 or red<=0:raise ValueError('Observed D4000 band flux is nonpositive')
    wa,fa=band(4041.6,4079.75);wc,fc=band(4128.5,4161.)
    left=np.trapezoid(fa,x=wa)/(wa[-1]-wa[0]);right=np.trapezoid(fc,x=wc)/(wc[-1]-wc[0])
    wl,fl=band(4083.5,4122.25)
    continuum=np.interp(wl,[(wa[0]+wa[-1])/2,(wc[0]+wc[-1])/2],[left,right])
    if np.any(continuum<=0):raise ValueError('Hdelta continuum is nonpositive')
    return {'dn4000':float(red/blue),
            'hdelta_equivalent_width_angstrom':float(np.trapezoid(1-fl/continuum,x=wl))}


def distance_modulus(distance_mpc):
    d = array(distance_mpc, 'luminosity distance Mpc', positive=True)
    return 5*np.log10(d)+25


@dataclass(frozen=True)
class IndicatorPrescription:
    """Empirical magnitude relations; Cepheid abundances/pivot are in dex.

    This logarithmic indicator abundance is distinct from the SSP grid's total
    metal mass fraction Z. Photometric colors use their declared catalogue units.
    """
    model_id: str
    revision: str
    magnitude_system: str
    cepheid_absolute_magnitude: float
    cepheid_period_slope: float
    cepheid_metallicity_slope: float
    cepheid_scatter_mag: float
    trgb_absolute_magnitude: float
    trgb_color_slope: float
    trgb_scatter_mag: float
    sn_absolute_magnitude: float
    sn_stretch_slope: float
    sn_color_slope: float
    sn_mass_step_mag: float
    sn_scatter_mag: float
    metallicity_pivot: float
    trgb_color_pivot: float

    def __post_init__(self):
        if not self.model_id or not self.revision or not self.magnitude_system:
            raise ValueError('Indicator model must name a pinned revision and photometric system')
        for key, value in asdict(self).items():
            if key in {'model_id','revision','magnitude_system'}:
                if not isinstance(value,str) or not value.strip():raise ValueError('Named indicator model metadata required')
            else:
                object.__setattr__(self,key,float(array(value,'indicator '+key,0)))
        if min(self.cepheid_scatter_mag,self.trgb_scatter_mag,self.sn_scatter_mag) < 0:
            raise ValueError('Intrinsic scatters must be nonnegative')


def sample_indicator_populations(hosts, model, population_spec, *, seed, maximum_objects=None):
    """Seeded physical host-conditioned populations; all prescriptions are recorded.

    Host metallicity, extinction and crowding enter each resolved *mock* indicator.
    Rates/counts are supplied explicitly; they are never deduced as pulsations
    or explosions from native SSP particle number.
    """
    rng = np.random.default_rng(integer(seed,'indicator seed',0))
    ids = exact_ids(hosts['host_id'], 'host_id')
    distance = array(hosts['luminosity_distance_mpc'], 'host distances', 1, positive=True)
    if distance.shape != ids.shape:
        raise ValueError('Host identity and distance shapes differ')
    kinds=('cepheid','trgb','sn_calibrator','sn_hubble')
    if set(population_spec['counts'])-set(kinds):raise ValueError('Unsupported indicator population kind')
    plan={}
    for kind in kinds:
        counts=population_spec['counts'].get(kind,0)
        if isinstance(counts,list):
            if len(counts)!=len(ids):raise ValueError('Indicator count vectors must align with hosts')
            plan[kind]=[integer(value,'indicator count',0) for value in counts]
        else:
            plan[kind]=[integer(counts,'indicator count',0)]*len(ids)
    total=sum(sum(values) for values in plan.values())
    if maximum_objects is not None and total>integer(maximum_objects,'maximum indicator objects'):
        raise MemoryError(f'Authored indicator count {total} exceeds the fit workspace/object limit {maximum_objects}')
    rows = []
    for j, host in enumerate(ids):
        for kind in kinds:
            count = plan[kind][j]
            if kind == 'sn_hubble' and count and hosts.get('anchor_mu', [None]*len(ids))[j] is not None:
                raise ValueError('An anchor host cannot also be treated as a Hubble-flow host')
            for i in range(count):
                lp = rng.uniform(*population_spec['log_period_range'])
                metal = float(hosts['metallicity'][j]) + rng.normal(0,population_spec['metallicity_scatter'])
                color = rng.normal(*population_spec['trgb_color_normal'])
                stretch = rng.normal(*population_spec['sn_stretch_normal'])
                sncolor = rng.normal(*population_spec['sn_color_normal'])
                rows.append({'object_id':f'{host}:{kind}:{i}', 'host_id':str(host), 'kind':kind,
                    'distance_mpc':float(distance[j]), 'redshift':float(hosts['redshift'][j]),
                    'log_period':lp, 'metallicity':metal, 'trgb_color':color,
                    'stretch':stretch, 'sn_color':sncolor,
                    'mass_step':float(hosts['stellar_mass_msun'][j] >= population_spec['mass_step_threshold_msun']),
                    'extinction_mag':float(hosts['extinction_mag'][j]),
                    'crowding_mag':float(hosts['crowding_mag'][j]),
                    'instrument':str(hosts['instrument'][j])})
    if not rows:
        raise ValueError('Indicator population is physically empty')
    return {'role':'indicator_population_truth','rows':rows, 'prescription':asdict(model),
            'population_spec':population_spec,'seed':seed,'namespace':hosts['namespace']}


def render_indicator_measurements(populations, anchors, measurement_spec, *, seed):
    """Correlated instrument shifts are drawn once and kept out of fit inputs.

    Extinction/crowding estimates are observations with configured measurement
    errors. Magnitude limits are recorded for a truncated-likelihood fit.
    Anchor covariance may include cross-anchor calibration correlations.
    """
    model = IndicatorPrescription(**populations['prescription'])
    rng = np.random.default_rng(integer(seed,'indicator seed',0))
    instruments = measurement_spec['instrument_ids']
    if not instruments or any(not isinstance(name,str) or not name.strip() for name in instruments) or len(set(instruments))!=len(instruments):raise ValueError('Distinct named instrument calibration axes required')
    zpcov = covariance(measurement_spec['zeropoint_covariance'],len(instruments))
    zp = dict(zip(instruments,rng.multivariate_normal(np.zeros(len(instruments)),zpcov)))
    rows=[]
    for source in populations['rows']:
        row=dict(source)
        kind=row['kind']
        if kind=='cepheid':
            absolute=(model.cepheid_absolute_magnitude+model.cepheid_period_slope*(row['log_period']-1)
                      +model.cepheid_metallicity_slope*(row['metallicity']-model.metallicity_pivot))
            scatter=model.cepheid_scatter_mag
        elif kind=='trgb':
            absolute=model.trgb_absolute_magnitude+model.trgb_color_slope*(row['trgb_color']-model.trgb_color_pivot)
            scatter=model.trgb_scatter_mag
        else:
            absolute=(model.sn_absolute_magnitude-model.sn_stretch_slope*row['stretch']
                      +model.sn_color_slope*row['sn_color']+model.sn_mass_step_mag*row['mass_step'])
            scatter=model.sn_scatter_mag
        error=positive(measurement_spec['photometric_sigma_mag'][kind],'photometric sigma mag')
        mag=(float(distance_modulus(row.pop('distance_mpc')))+absolute+row['extinction_mag']
             +row['crowding_mag']+zp[row['instrument']]+rng.normal(0,np.hypot(error,scatter)))
        limit=float(array(measurement_spec['magnitude_limits'][kind],'magnitude limit',0))
        if mag > limit:
            continue
        row['magnitude']=mag
        row['sigma_mag']=error
        row['intrinsic_sigma_mag']=scatter
        row['magnitude_limit']=limit
        for term in ('extinction','crowding'):
            sig=float(array(measurement_spec[term+'_sigma_mag'],term+' sigma mag',0))
            if sig<0:raise ValueError('Extinction/crowding sigma cannot be negative')
            row[term+'_mag'] += rng.normal(0,sig)
            row[term+'_sigma_mag']=sig
        row['metallicity'] += rng.normal(0,measurement_spec.get('metallicity_sigma',0.))
        rows.append(row)
    if not rows:
        raise ValueError('Photometric selection detected no indicators')
    columns={k:[r[k] for r in rows] for k in rows[0]}
    units={k:('mag' if 'mag' in k or k in ('magnitude','magnitude_limit','trgb_color','sn_color') else '1') for k in columns}
    units.update(log_period='log10(day)',metallicity='dex')
    calibration={'anchors':anchors, 'instrument_ids':instruments,
        'zeropoint_covariance':zpcov.tolist(), 'magnitude_system':model.magnitude_system,
        'metallicity_pivot':model.metallicity_pivot,'trgb_color_pivot':model.trgb_color_pivot}
    return observed_catalogue(columns,units=units,namespace=populations['namespace'],
        selection={'seed':seed,'magnitude_limits':measurement_spec['magnitude_limits'],
                   'detection':'upper magnitude truncation','generator_model_id':model.model_id,
                   'generator_revision':model.revision},calibration=calibration)


def _ladder_design(obs, fit_spec):
    """Build the joint linear magnitude model and its calibration blocks.

    Host distance moduli are shared by their local indicators. Statistical
    photometry uses either the full supplied covariance or sigma_mag's diagonal,
    never both. Intrinsic/correction/flow variances are separate; instrument
    zeropoints use their own correlated prior. q0, j0 and residual-flow dispersion
    are explicit model inputs. Cepheid log-periods are log10(day), metals dex.
    """
    from .hubble_observers import cosmographic_distance_shape
    t = obs['columns']
    cal = obs['calibration']
    n = len(exact_ids(t['object_id']))
    if not n:
        raise ValueError('Distance-ladder sample is empty')
    if 'cepheid' in t['kind'] and (obs['units'].get('log_period') != 'log10(day)' or obs['units'].get('metallicity') != 'dex'):
        raise ValueError('Cepheid period/metallicity must be explicitly log10(day)/dex')
    redshift_range = finite(fit_spec['hubble_redshift_range'], (2,), 'Hubble-flow redshift range')
    if redshift_range[0] <= 0 or redshift_range[1] <= redshift_range[0]:
        raise ValueError('Positive ordered Hubble-flow redshift domain required')
    flow = float(array(fit_spec['flow_dispersion_kms'], 'flow dispersion km/s', 0))
    if flow < 0:
        raise ValueError('Flow dispersion cannot be negative')
    host_keys = []
    for host in t['host_id']:
        if isinstance(host, (bool, np.bool_)) or not isinstance(host, (int, np.integer, str)):
            raise ValueError('Ladder hosts require exact integer or text identities')
        host_keys.append(str(host))
    local_hosts = sorted({host_keys[i] for i in range(n) if t['kind'][i] != 'sn_hubble'})
    names = ['mu:' + h for h in local_hosts] + ['M_cepheid', 'b_period', 'gamma_metallicity', 'M_trgb', 'b_trgb_color', 'M_sn', 'alpha_stretch', 'beta_color', 'mass_step', 'minus5log10H0']
    instruments = cal.get('instrument_ids', [])
    names += ['zeropoint:' + i for i in instruments]
    index = {name: i for i, name in enumerate(names)}
    errors = {key: finite(t[key], (n,), key + ' mag') for key in ('sigma_mag', 'intrinsic_sigma_mag', 'extinction_sigma_mag', 'crowding_sigma_mag')}
    if np.any(errors['sigma_mag'] <= 0) or any((np.any(errors[key] < 0) for key in errors if key != 'sigma_mag')):
        raise ValueError('Photometric sigma must be positive; intrinsic/correction sigmas nonnegative')
    design = np.zeros((n, len(names)))
    y = np.zeros(n)
    var = np.zeros(n)
    for i, kind in enumerate(t['kind']):
        y[i] = t['magnitude'][i] - t['extinction_mag'][i] - t['crowding_mag'][i]
        var[i] = sum((errors[key][i] ** 2 for key in ('intrinsic_sigma_mag', 'extinction_sigma_mag', 'crowding_sigma_mag')))
        if kind != 'sn_hubble':
            design[i, index['mu:' + host_keys[i]]] = 1
        else:
            z = float(t['redshift'][i])
            if not fit_spec['hubble_redshift_range'][0] <= z <= fit_spec['hubble_redshift_range'][1]:
                raise ValueError('Hubble SN outside declared fitting redshift domain')
            shape = float(cosmographic_distance_shape([z], fit_spec['q0'], fit_spec['j0'])[0])
            y[i] -= 5 * np.log10(shape) + 25
            design[i, index['minus5log10H0']] = 1
            var[i] += (5 / np.log(10) * flow / (C_KMS * z)) ** 2
        if kind == 'cepheid':
            design[i, index['M_cepheid']] = 1
            design[i, index['b_period']] = t['log_period'][i] - 1
            design[i, index['gamma_metallicity']] = t['metallicity'][i] - cal['metallicity_pivot']
        elif kind == 'trgb':
            design[i, index['M_trgb']] = 1
            design[i, index['b_trgb_color']] = t['trgb_color'][i] - cal['trgb_color_pivot']
        elif kind in ('sn_calibrator', 'sn_hubble'):
            design[i, index['M_sn']] = 1
            design[i, index['alpha_stretch']] = -t['stretch'][i]
            design[i, index['beta_color']] = t['sn_color'][i]
            design[i, index['mass_step']] = t['mass_step'][i]
        else:
            raise ValueError('Unsupported distance indicator: ' + kind)
        if instruments:
            design[i, index['zeropoint:' + t['instrument'][i]]] = 1
    photometric = covariance(cal['photometric_covariance'], n, semidefinite=True) if 'photometric_covariance' in cal else np.diag(errors['sigma_mag'] ** 2)
    data_cov = photometric + np.diag(var)
    blocks = [(design, y, data_cov, 'indicators')]
    anchors = cal['anchors']
    anchor_hosts = list(map(str, exact_ids(anchors['host_ids'], 'anchor host IDs')))
    if set(local_hosts) & {host_keys[i] for i in range(n) if t['kind'][i] == 'sn_hubble'}:
        raise ValueError('This ladder model requires disjoint calibrated and Hubble-flow host cohorts')
    anchor_design = np.zeros((len(anchor_hosts), len(names)))
    for i, host in enumerate(anchor_hosts):
        if host not in local_hosts:
            raise ValueError('Anchor has no indicator host connection: ' + host)
        anchor_design[i, index['mu:' + host]] = 1
    blocks.append((anchor_design, array(anchors['distance_moduli'], 'anchor moduli', 1), covariance(anchors['covariance'], len(anchor_hosts)), 'anchors'))
    if instruments:
        prior = np.zeros((len(instruments), len(names)))
        for i, ins in enumerate(instruments):
            prior[i, index['zeropoint:' + ins]] = 1
        blocks.append((prior, np.zeros(len(instruments)), covariance(cal['zeropoint_covariance'], len(instruments)), 'zeropoints'))
    for prior in fit_spec.get('parameter_priors', []):
        a = np.zeros((len(prior['names']), len(names)))
        for i, name in enumerate(prior['names']):
            a[i, index[name]] = 1
        blocks.append((a, array(prior['means'], 'prior means', 1), covariance(prior['covariance'], len(prior['names'])), 'parameter_priors'))
    from scipy.linalg import block_diag
    combined = np.vstack([b[0] for b in blocks])
    active = np.any(combined != 0, axis=0)
    if not active[index['minus5log10H0']]:
        raise ValueError('A Hubble-flow SN sample is required for H0 inference')
    return (combined[:, active], np.concatenate([b[1] for b in blocks]), block_diag(*[b[2] for b in blocks]), [name for name, use in zip(names, active) if use], n, design[:, active], y, data_cov)


def jointly_fit_anchors_and_indicator_relations(observations, fit_spec, *, realization_id=None):
    """Fit the shared host, indicator and calibration coordinates jointly.

    Host moduli and indicator/zeropoint intercepts are magnitudes. The Hubble
    coordinate is -5*log10(H0/[km/s/Mpc]), so H0=exp[-ln(10)*q/5]. A detected
    sample uses log Phi(threshold) normalization, conditional on its fitted
    shared zeropoints. Correlated detection residuals need a multivariate model.
    """
    observed = read_observations(observations)
    design, data, measured_covariance, names, count, indicator_design, indicator_data, indicator_covariance = _ladder_design(observed, fit_spec)
    fit = gls(design, data, measured_covariance, parameter_names=names)
    mode = fit_spec['selection_likelihood']
    if mode == 'magnitude_truncated':
        off_diagonal = indicator_covariance-np.diag(np.diag(indicator_covariance))
        scale = np.max(np.abs(indicator_covariance))
        if np.any(np.abs(off_diagonal) > 1e-14*scale):
            raise ValueError('Correlated photometric detection needs a multivariate selection model')
        columns = observed['columns']
        # indicator_data already removed the exact extinction/crowding and
        # Hubble cosmographic offset. Replace only its observed magnitude with
        # the detection limit; no second cosmographic calculation is needed.
        limits = indicator_data + finite(columns['magnitude_limit'], (count,), 'magnitude limits') - finite(columns['magnitude'], (count,), 'observed magnitudes')
        sigma = np.sqrt(np.diag(indicator_covariance))
        cholesky = np.linalg.cholesky(measured_covariance)

        def objective(parameters):
            residual = solve_triangular(cholesky, data-design@parameters, lower=True)
            return .5*float(residual@residual) + float(np.sum(log_ndtr((limits-indicator_design@parameters)/sigma)))

        result = minimize(objective, fit['mean'], method='BFGS', options={'maxiter':10000, 'gtol':1e-7})
        parameters = finite(result.x, (len(names),), 'truncated ladder parameters')
        gradient = finite(result.jac, (len(names),), 'truncated ladder gradient')
        if not np.isfinite(result.fun) or (not result.success and np.linalg.norm(gradient)>1e-4):
            raise RuntimeError('Truncated distance-ladder fit failed: '+result.message)
        # Finite differences of this selected objective, not the optimizer's
        # approximate inverse Hessian. Curvature remains a local approximation.
        step = 1e-4*np.maximum(1, np.abs(parameters))
        hessian = np.empty((len(parameters), len(parameters)))
        center = objective(parameters)
        for i in range(len(parameters)):
            first = np.zeros(len(parameters)); first[i] = step[i]
            hessian[i,i] = (objective(parameters+first)-2*center+objective(parameters-first))/step[i]**2
            for j in range(i):
                second = np.zeros(len(parameters)); second[j] = step[j]
                hessian[i,j] = hessian[j,i] = (objective(parameters+first+second)-objective(parameters+first-second)-objective(parameters-first+second)+objective(parameters-first-second))/(4*step[i]*step[j])
        curvature = np.linalg.cholesky(covariance(hessian, len(parameters), 'selected ladder curvature'))
        inverse_factor = solve_triangular(curvature, np.eye(len(parameters)), lower=True)
        parameter_covariance = covariance(inverse_factor.T@inverse_factor, len(parameters), 'selected ladder parameter covariance')
        residual = finite(data-design@parameters, data.shape, 'selected ladder residual')
        whitened = solve_triangular(cholesky, residual, lower=True)
        fit.update(mean=parameters, covariance=parameter_covariance, residual=residual,
                   chi2=float(whitened@whitened), negative_log_likelihood=float(result.fun),
                   influence=None, uncertainty='local selected-likelihood curvature', optimizer_message=result.message)
    elif mode != 'complete':
        raise ValueError('Declare complete or magnitude_truncated selection likelihood')
    units = observed['units']
    slopes = {'b_period':'mag/'+units.get('log_period','log10(day)'),
              'gamma_metallicity':'mag/'+units.get('metallicity','dex'),
              'b_trgb_color':'mag/'+units.get('trgb_color','mag'),
              'alpha_stretch':'mag/'+units.get('stretch','1'),
              'beta_color':'mag/'+units.get('sn_color','mag')}
    fit['parameter_units'] = [slopes.get(name,'mag') for name in names]
    fit['covariate_uncertainty'] = 'Period, metallicity and colors are fixed measured covariates; their measurement-error distribution is not marginalized'
    index = names.index('minus5log10H0')
    exponent = -np.log(10)/5
    h0 = float(np.exp(exponent*fit['mean'][index]))
    variance = h0*h0*exponent**2*fit['covariance'][index,index]
    value = gaussian_observable(h0, variance, 'km/s/Mpc')
    width = np.sqrt(fit['covariance'][index,index])
    value['interval_68'] = [float(np.exp(exponent*(fit['mean'][index]+width))), float(np.exp(exponent*(fit['mean'][index]-width)))]
    return freeze_result('distance_ladder', fit, observables={'H0':value}, selection=observed['selection'],
        object_ids=observed['columns']['object_id'], namespace=observed['namespace'], model=fit_spec,
        nuisance={'indicator_counts':dict(Counter(observed['columns']['kind'])), 'host_shared_moduli':True},
        realization_id=realization_id, calibration_ids=list(map(str,exact_ids(observed['calibration']['anchors']['host_ids'],'anchor host IDs'))))



