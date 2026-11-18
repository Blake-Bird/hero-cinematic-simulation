"""Native BH parameter contracts, unit-explicit models and saved-source readers.

Accretion references use physical cgs quantities and distinguish inflow,
retained subgrid mass, gravity-carrier mass, radiation and feedback energy.
History exports consume measured type-5 snapshot state; they do not reconstruct
self-consistent growth by applying the analytic oracle to sparse snapshots.
Optional dynamics, campaign and scene helpers retain their declared contracts.
"""
from __future__ import annotations

# ========================================================================
# black_holes

# ========================================================================
# black_hole_model: Unit-explicit reference equations for a Bondi/Eddington thermal BH model.

from dataclasses import dataclass
import math

from core.foundation import G_CGS, C_SI, MSUN_G, SECONDS_PER_YEAR

C_CGS = C_SI * 100
PROTON_MASS_G = 1.67262192595e-24
THOMSON_CROSS_SECTION_CM2 = 6.6524587051e-25

NATIVE_PARAMETERS = {
    "radiative_efficiency": "BHRadiativeEfficiency", "feedback_coupling": "BHFeedbackCoupling",
    "bondi_boost": "BHBondiBoost", "eddington_factor": "BHEddingtonFactor",
    "kernel_radius_proper_kpc": "BHKernelRadiusProperKpc", "seed_mass_msun": "BHSeedMassMsun",
    "source_fraction": "BHSourceFraction", "maximum_count": "BHMaximumCount",
    "seed_density_hydrogen_cm3": "BHSeedDensityHydrogenCm3",
    "seed_exclusion_proper_kpc": "BHSeedExclusionProperKpc",
}


def native_parameters(settings):
    """Validate every native model choice before building or launching."""
    if not isinstance(settings, dict) or not isinstance(settings.get("enabled", False), bool):
        raise ValueError("black_holes must be a mapping with boolean enabled")
    if not settings.get("enabled", False):
        if set(settings) - {"enabled"}:
            raise ValueError("Disabled black_holes accepts only enabled=false")
        return {}
    if set(settings) - {"dynamics","angular_momentum","feedback","wake_diagnostics"} != {"enabled", "seeding_mode", *NATIVE_PARAMETERS}:
        raise ValueError("Enabled black_holes requires exactly: enabled, " + ", ".join(NATIVE_PARAMETERS))
    ThermalModel(*(settings[k] for k in ("radiative_efficiency", "feedback_coupling", "bondi_boost", "eddington_factor")))
    for key in ("kernel_radius_proper_kpc", "seed_mass_msun", "source_fraction"):
        finite_number(settings[key], key, strict=True)
    if settings["source_fraction"] > .1:
        raise ValueError("BH source_fraction must be at most 0.1")
    count = settings["maximum_count"]
    if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= 100000:
        raise ValueError("BH maximum_count must be an integer in [1,100000]")
    mode=settings["seeding_mode"]
    modes={"ic_particles":0,"density_exclusion":1,"density_exclusion_reservoir":2}
    if mode not in modes:
        raise ValueError("BH seeding_mode must be ic_particles, density_exclusion or density_exclusion_reservoir")
    for key in ("seed_density_hydrogen_cm3", "seed_exclusion_proper_kpc"):
        finite_number(settings[key], key, strict=mode!="ic_particles")
    if 'wake_diagnostics' in settings:wake_diagnostic_settings(settings['wake_diagnostics'])
    return {**{native: settings[key] for key, native in NATIVE_PARAMETERS.items()},
            "BHSeedingMode": modes[mode], **dynamics_native_parameters(settings.get("dynamics", {"enabled":False})),
            **accretion_feedback_native_parameters(settings.get("angular_momentum",{"model":"none"}),settings.get("feedback",{"mode":"thermal"}))}


def wake_diagnostic_settings(settings):
    from pathlib import Path
    from core.foundation import file_sha256
    if not isinstance(settings,dict) or set(settings)!={'schedule','maximum_bytes','maximum_rows'}:
        raise ValueError('Complete pinned native wake schedule and parse limits required')
    if not isinstance(settings['schedule'],dict) or set(settings['schedule'])!={'path','sha256'}:raise ValueError('Pin the native wake schedule bytes')
    for name in ('maximum_bytes','maximum_rows'):
        if type(settings[name])is not int or not 1<=settings[name]<=2**31-1:raise ValueError('Exact positive native wake schedule bounds required')
    requested=Path(settings['schedule']['path']).expanduser();path=requested.resolve();expected=settings['schedule']['sha256']
    if requested.is_symlink() or not isinstance(expected,str) or len(expected)!=64 or any(c not in '0123456789abcdef' for c in expected) or file_sha256(path)!=expected:
        raise ValueError('Native wake schedule is not its pinned original regular source')
    if path.stat().st_size>settings['maximum_bytes']:raise MemoryError('Native wake schedule exceeds its byte allocation')
    return path


def publish_dm_wake_schedule(request,destination):
    """Author the immutable native schedule; never invoke a diagnostic/run."""
    import json
    from pathlib import Path
    from core.foundation import atomic_text,atomic_json,file_sha256
    keys={'schema','physics_signature','rows','maximum_particles','maximum_work','maximum_workspace_bytes',
        'maximum_publication_bytes','epoch_tolerance_native','maximum_bytes','maximum_rows'}
    if not isinstance(request,dict) or set(request)!=keys or request['schema']!='native-wake-schedule-request-v1':raise ValueError('Complete bounded native diagnostic schedule request required')
    signature=request['physics_signature']
    if not isinstance(signature,str) or len(signature)!=64 or any(c not in '0123456789abcdef' for c in signature):raise ValueError('Exact original native physical signature required')
    for name in ('maximum_particles','maximum_work','maximum_workspace_bytes','maximum_publication_bytes','maximum_bytes','maximum_rows'):
        if type(request[name])is not int or not 1<=request[name]<2**64:raise ValueError('Exact positive native schedule/source allocation required')
    if max(request['maximum_bytes'],request['maximum_rows'])>2**31-1:raise ValueError('Schedule compile allocation exceeds the native supported integer domain')
    if request['maximum_workspace_bytes']<1024 or request['maximum_publication_bytes']<8192:raise ValueError('Native diagnostic workspace/publication ceilings are too small')
    tolerance=finite_number(request['epoch_tolerance_native'],'wake epoch tolerance')
    rows=request['rows']
    if not isinstance(rows,list) or not 1<=len(rows)<=request['maximum_rows']:raise ValueError('Complete bounded nonempty scheduled measurements required')
    previous=None;identities=set();lines=['RESEARCH2_WAKE_SCHEDULE_V1',signature,
        ' '.join(str(request[name]) for name in ('maximum_particles','maximum_work','maximum_workspace_bytes','maximum_publication_bytes')),
        format(tolerance,'.17g'),str(len(rows))]
    for row in rows:
        if not isinstance(row,dict) or set(row)!={'native_time','carrier_id','measurement_id','radial_edges_proper_kpc'}:raise ValueError('Complete scheduled native epoch/probe/shell definition required')
        time=finite_number(row['native_time'],'scheduled native time',strict=True)
        if previous is not None and time-previous<=2*tolerance:raise ValueError('Scheduled native epochs must be increasing and distinguishable')
        previous=time
        for name in ('carrier_id','measurement_id'):
            if type(row[name])is not int or not 0<row[name]<2**64:raise ValueError('Exact nonzero uint64 scheduled measurement/probe identity required')
        identity=(row['carrier_id'],row['measurement_id'])
        if identity in identities:raise ValueError('Schedule repeats a measurement identity')
        identities.add(identity);edges=row['radial_edges_proper_kpc']
        if not isinstance(edges,list) or not 2<=len(edges)<=4096:raise ValueError('Bounded native diagnostic shell edges required')
        for value in edges:finite_number(value,'native shell edge')
        if any(b<=a for a,b in zip(edges,edges[1:])):raise ValueError('Ordered distinct native shell edges required')
        if 8192+2048*len(edges)>request['maximum_publication_bytes']:raise MemoryError('Native shell serialization exceeds publication allocation')
        lines.append(' '.join([format(time,'.17g'),str(row['carrier_id']),str(row['measurement_id']),str(len(edges)),*[format(value,'.17g') for value in edges]]))
    payload='\n'.join(lines)+'\n'
    if len(payload.encode('ascii'))>request['maximum_bytes']:raise MemoryError('Serialized schedule exceeds its byte allocation')
    destination=Path(destination).resolve();report_path=destination.with_name(destination.name+'.json')
    if destination.exists() or report_path.exists():raise FileExistsError('Retain existing native diagnostic schedule')
    atomic_text(destination,payload);seal=destination.with_name(destination.name+'.sha256');atomic_text(seal,file_sha256(destination)+'\n')
    settings={'schedule':{'path':str(destination),'sha256':file_sha256(destination)},'maximum_bytes':request['maximum_bytes'],'maximum_rows':request['maximum_rows']}
    report={'schema':'native-wake-schedule-v1','request':request,'wake_diagnostics':settings,
        'execution_status':'AUTHORED_NOT_NATIVE_EXECUTED','scope':'read-only native monopoles at synchronized noninitial output epochs; physics signature must match actual native config',
        'source_sha256':file_sha256(__file__),'science_acceptance':'NOT_ESTABLISHED'}
    atomic_json(report_path,report);return report


def dm_wake_schedule_main():
    import argparse,json
    from pathlib import Path
    parser=argparse.ArgumentParser(description='Materialize a bounded SHA-pinned native wake diagnostic schedule')
    parser.add_argument('request',type=Path);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();publish_dm_wake_schedule(json.loads(args.request.read_text()),args.output);return 0


def accretion_feedback_native_parameters(angular,feedback):
    """Declare rotational suppression and the feedback channel separately."""
    if not isinstance(angular,dict) or angular.get('model') not in {'none','rosas2015'}:raise ValueError('Declare none or rosas2015 angular suppression')
    if angular['model']=='none':
        if set(angular)!={'model'}:raise ValueError('Unsuppressed Bondi accepts only model=none')
        viscosity=1.
    else:
        if set(angular)!={'model','viscosity_parameter'}:raise ValueError('Rotational suppression requires a declared viscosity parameter')
        viscosity=finite_number(angular['viscosity_parameter'],'viscosity parameter',strict=True)
    if not isinstance(feedback,dict) or set(feedback)!={'mode'} or feedback['mode'] not in {'thermal','kinetic'}:raise ValueError('Declare thermal or conservative zero-net-momentum kinetic feedback')
    return {'BHAngularMomentumModel':int(angular['model']=='rosas2015'),'BHViscosityParameter':viscosity,
        'BHFeedbackMode':int(feedback['mode']=='kinetic')}


def angular_momentum_suppression(sound_speed_cm_s,specific_angular_momentum_cm2_s,support_radius_cm,viscosity_parameter):
    """Rosas-Guevara et al. 2015 rotational Bondi suppression, explicit C_visc."""
    sound=finite_number(sound_speed_cm_s,'sound speed');j=finite_number(specific_angular_momentum_cm2_s,'specific angular momentum')
    radius=finite_number(support_radius_cm,'support radius',strict=True);viscosity=finite_number(viscosity_parameter,'viscosity parameter',strict=True)
    rotation=j/radius
    return 1. if rotation==0 else min(1.,(sound/rotation)**3/viscosity)


def dynamics_native_parameters(settings):
    """Explicit restricted native binary/friction profile; no implicit SIDM model."""
    if not isinstance(settings,dict) or not isinstance(settings.get('enabled',False),bool):
        raise ValueError('BH dynamics requires a mapping with boolean enabled')
    if not settings.get('enabled',False):
        if set(settings)!={'enabled'}:raise ValueError('Disabled BH dynamics accepts only enabled=false')
        return {'BHDynamicsEnabled':0,'BHBackgroundRadiusProperKpc':0,'BHMinimumImpactProperKpc':0,
            'BHStellarHardeningCoefficient':0,'BHBinaryMaximumEccentricity':0,'BHDynamicsFraction':0,
            'BHDynamicsMaximumSubcycles':0,'BHRecoilSeed':'0','BHSeedSpinX':0,'BHSeedSpinY':0,'BHSeedSpinZ':0}
    keys={'enabled','background_model','background_radius_proper_kpc','minimum_impact_proper_kpc',
        'stellar_hardening_coefficient','maximum_capture_eccentricity','fraction','maximum_subcycles','seed_spin','recoil_seed'}
    if set(settings)!=keys:raise ValueError('Enabled BH dynamics requires exactly '+', '.join(sorted(keys)))
    if settings['background_model']!='collisionless':
        raise ValueError('Native BH friction supports collisionless backgrounds; SIDM needs a separate physical calibration and operator')
    numeric={'background_radius_proper_kpc':'BHBackgroundRadiusProperKpc',
        'minimum_impact_proper_kpc':'BHMinimumImpactProperKpc','stellar_hardening_coefficient':'BHStellarHardeningCoefficient',
        'maximum_capture_eccentricity':'BHBinaryMaximumEccentricity','fraction':'BHDynamicsFraction'}
    for key in numeric:finite_number(settings[key],key,strict=key not in {'stellar_hardening_coefficient','maximum_capture_eccentricity'})
    if settings['maximum_capture_eccentricity']>.05 or settings['fraction']>.01:
        raise ValueError('Quasi-circular native profile requires capture e<=.05 and source fraction<=.01')
    if settings['minimum_impact_proper_kpc']>=settings['background_radius_proper_kpc']:
        raise ValueError('BH minimum impact must be below the measured background support')
    count=settings['maximum_subcycles'];seed=settings['recoil_seed'];spin=settings['seed_spin']
    if isinstance(count,bool) or not isinstance(count,int) or not 1<=count<=1000000:raise ValueError('BH dynamics subcycle budget must be integer [1,1000000]')
    if isinstance(seed,bool) or not isinstance(seed,int) or not 0<=seed<2**64:raise ValueError('BH recoil seed must be an exact uint64')
    if not isinstance(spin,list) or len(spin)!=3 or any(isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) for x in spin) or sum(x*x for x in spin)>.99**2:
        raise ValueError('BH seed spin must be a finite dimensionless three-vector with magnitude<=.99')
    return {'BHDynamicsEnabled':1,**{v:settings[k] for k,v in numeric.items()},'BHDynamicsMaximumSubcycles':count,
        'BHRecoilSeed':str(seed),**dict(zip(('BHSeedSpinX','BHSeedSpinY','BHSeedSpinZ'),spin))}


def finite_number(value, name, *, minimum=0., strict=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(name + " must be a finite number")
    if value < minimum or strict and value == minimum:
        raise ValueError(name + " is outside the physical range")
    return float(value)


@dataclass(frozen=True)
class ThermalModel:
    """Explicit research choices; no unresolved density boost unless declared."""
    radiative_efficiency: float
    feedback_coupling: float
    bondi_boost: float
    eddington_factor: float

    def __post_init__(self):
        finite_number(self.radiative_efficiency, "radiative_efficiency", strict=True)
        finite_number(self.feedback_coupling, "feedback_coupling")
        finite_number(self.bondi_boost, "bondi_boost", strict=True)
        finite_number(self.eddington_factor, "eddington_factor", strict=True)
        if self.radiative_efficiency >= 1 or self.feedback_coupling > 1:
            raise ValueError("Efficiency must be in (0,1), coupling in [0,1]")


def accretion_rates(mass_msun, density_g_cm3, sound_speed_cm_s,
                    relative_speed_cm_s, model):
    """Return gas inflow, mass-growth and thermal-feedback rates.

    Bondi-Hoyle-Lyttleton: 4*pi*alpha*G^2*M^2*rho/(cs^2+vrel^2)^1.5.
    Eddington inflow: f_Edd*4*pi*G*M*m_p/(epsilon_r*sigma_T*c).
    L_bol = epsilon_r * inflow * c^2; thermal power = epsilon_f * L_bol.
    """
    mass = finite_number(mass_msun, "mass_msun", strict=True)*MSUN_G
    rho = finite_number(density_g_cm3, "density_g_cm3")
    sound = finite_number(sound_speed_cm_s, "sound_speed_cm_s")
    speed = finite_number(relative_speed_cm_s, "relative_speed_cm_s")
    denominator = math.hypot(sound, speed)**3
    if denominator == 0:
        raise ValueError("A finite Bondi estimator requires nonzero sound or relative speed")
    bondi = 4*math.pi*model.bondi_boost*G_CGS**2*mass**2*rho/denominator
    eddington = model.eddington_factor*4*math.pi*G_CGS*mass*PROTON_MASS_G/(model.radiative_efficiency*THOMSON_CROSS_SECTION_CM2*C_CGS)
    inflow = min(bondi, eddington)
    conversion = SECONDS_PER_YEAR/MSUN_G
    result = {"bondi_inflow_msun_year": bondi*conversion,
              "eddington_inflow_msun_year": eddington*conversion,
              "inflow_msun_year": inflow*conversion,
              "bh_growth_msun_year": (1-model.radiative_efficiency)*inflow*conversion,
              "bolometric_luminosity_erg_s": model.radiative_efficiency*inflow*C_CGS**2,
              "thermal_feedback_power_erg_s": model.feedback_coupling*model.radiative_efficiency*inflow*C_CGS**2}
    if any(not math.isfinite(value) for value in result.values()):
        raise ValueError("Accretion estimator overflow; check inputs and units")
    return result


def salpeter_time_years(model):
    """BH mass e-folding time under the selected constant Eddington cap."""
    return (model.radiative_efficiency*THOMSON_CROSS_SECTION_CM2*C_CGS /
            ((1-model.radiative_efficiency)*model.eddington_factor*4*math.pi*G_CGS*PROTON_MASS_G) / SECONDS_PER_YEAR)


def constant_environment_growth(mass_msun, duration_years, density_g_cm3,
                                sound_speed_cm_s, relative_speed_cm_s, model):
    """Exact constant-environment solution, including Bondi-to-Eddington switch.

    This is an analytic test problem with unlimited gas and no response to
    feedback. It cannot be used to post-process a self-consistent growth curve
    from cosmological snapshots. Native tests should converge to this solution
    when gas depletion, gravity, star formation and feedback are disabled.
    """
    mass = finite_number(mass_msun, "mass_msun", strict=True)
    duration = finite_number(duration_years, "duration_years")
    rates = accretion_rates(mass, density_g_cm3, sound_speed_cm_s, relative_speed_cm_s, model)
    edd_coefficient = 1/salpeter_time_years(model)
    bondi_coefficient = (1-model.radiative_efficiency)*rates["bondi_inflow_msun_year"]/mass**2
    if duration == 0 or bondi_coefficient == 0:
        return mass
    transition_mass = edd_coefficient/bondi_coefficient
    try:
        if mass >= transition_mass:
            result = mass*math.exp(edd_coefficient*duration)
        else:
            transition_time = (1/mass - 1/transition_mass)/bondi_coefficient
            if duration <= transition_time:
                result = mass/(1-bondi_coefficient*mass*duration)
            else:
                result = transition_mass*math.exp(edd_coefficient*(duration-transition_time))
    except OverflowError as exc:
        raise ValueError("Analytic growth overflows; reduce the test duration") from exc
    if not math.isfinite(result):
        raise ValueError("Analytic growth overflows; reduce the test duration")
    return result


def energy_mass_budget(inflow_msun, model):
    """Accounting oracle; includes escaping radiation as a mass equivalent."""
    inflow = finite_number(inflow_msun, "inflow_msun")
    energy = model.radiative_efficiency*inflow*MSUN_G*C_CGS**2
    return {"gas_mass_removed_msun": inflow,
            "bh_mass_gained_msun": (1-model.radiative_efficiency)*inflow,
            "radiated_mass_equivalent_msun": model.radiative_efficiency*inflow,
            "radiated_energy_erg": energy,
            "coupled_thermal_energy_erg": model.feedback_coupling*energy,
            "escaping_radiation_energy_erg": (1-model.feedback_coupling)*energy}

# ========================================================================
# black_hole_analysis: Measured native BH histories, physical luminosities and accounting figures.

def peters_binary_rates(mass1_msun,mass2_msun,semimajor_axis_cm,eccentricity):
    """Orbit-averaged leading GW loss rates, in physical seconds (Peters 1964)."""
    m1=finite_number(mass1_msun,'primary mass',strict=True)*MSUN_G
    m2=finite_number(mass2_msun,'secondary mass',strict=True)*MSUN_G
    a=finite_number(semimajor_axis_cm,'semimajor axis',strict=True)
    e=finite_number(eccentricity,'eccentricity')
    if e>=1:raise ValueError('GW hardening requires a bound orbit with e<1')
    coefficient=G_CGS**3*m1*m2*(m1+m2)/C_CGS**5
    da=-64/5*coefficient/a**3*(1+73/24*e*e+37/96*e**4)/(1-e*e)**3.5
    de=-304/15*coefficient/a**4*e*(1+121/304*e*e)/(1-e*e)**2.5
    luminosity=G_CGS*m1*m2/(2*a*a)*(-da)
    return {'da_cm_s':da,'de_per_s':de,'gw_luminosity_erg_s':luminosity,
        'circular_coalescence_time_s':a**4/(4*(64/5)*coefficient) if e==0 else None}


def bound_binary_elements(relative_position_cm,relative_velocity_cm_s,mass1_msun,mass2_msun):
    """Physical two-body osculating elements; closeness never implies binding."""
    import numpy as np
    r=np.asarray(relative_position_cm,dtype=float);v=np.asarray(relative_velocity_cm_s,dtype=float)
    if r.shape!=(3,) or v.shape!=(3,) or not np.isfinite(r).all() or not np.isfinite(v).all():raise ValueError('Finite three-vector binary state required')
    mass=finite_number(mass1_msun,'mass1',strict=True)+finite_number(mass2_msun,'mass2',strict=True)
    radius=float(np.linalg.norm(r))
    if radius<=0:raise ValueError('Resolved binary elements require nonzero separation')
    mu=G_CGS*mass*MSUN_G;energy=float(np.dot(v,v)/2-mu/radius);angular=np.cross(r,v)
    if energy>=0:return {'bound':False,'specific_orbital_energy_cm2_s2':energy,'semimajor_axis_cm':None,'eccentricity':None}
    eccentric_vector=np.cross(v,angular)/mu-r/radius;e=float(np.linalg.norm(eccentric_vector))
    if e>=1:raise ValueError('Inconsistent bound binary orbital invariants')
    norm=float(np.linalg.norm(angular))
    return {'bound':True,'specific_orbital_energy_cm2_s2':energy,'semimajor_axis_cm':-mu/(2*energy),
        'eccentricity':e,'angular_momentum_unit':(angular/norm).tolist() if norm>0 else None,
        'radial_unit':(r/radius).tolist()}


def unresolved_friction_acceleration(mass_msun,relative_velocity_cm_s,slow_density_g_cm3,
    *,force_support_cm,minimum_impact_cm,background_kind='collisionless',sidm_calibration=None):
    """Tremmel-style unresolved Chandrasekhar correction below force support.

    The slow-particle density is a measured local velocity-distribution integral,
    not a Maxwellian substitution. Resolved larger impact parameters are excluded.
    The caller must retain the momentum/energy exchanged with the unresolved bath.
    """
    import numpy as np
    mass=finite_number(mass_msun,'physical BH mass',strict=True)*MSUN_G
    rho=finite_number(slow_density_g_cm3,'slow background density')
    support=finite_number(force_support_cm,'force support',strict=True)
    minimum=finite_number(minimum_impact_cm,'minimum impact',strict=True)
    velocity=np.asarray(relative_velocity_cm_s,dtype=float)
    if velocity.shape!=(3,) or not np.isfinite(velocity).all():raise ValueError('Finite friction velocity required')
    if background_kind not in {'collisionless','sidm'}:raise ValueError('Declare background interaction model')
    if background_kind=='sidm':
        # A truthy mapping is not a measured SIDM response. Until the actual
        # calibrated unresolved-wake evaluator is connected, never silently
        # return the collisionless coefficient under an SIDM label.
        raise ValueError('SIDM friction requires the measured unresolved-wake calibration evaluator; collisionless friction cannot consume sidm_calibration')
    if sidm_calibration is not None:
        raise ValueError('Collisionless friction cannot consume an SIDM calibration')
    speed=float(np.linalg.norm(velocity))
    if speed==0 or rho==0 or minimum>=support:return np.zeros(3)
    return -4*math.pi*G_CGS**2*mass*rho*math.log(support/minimum)*velocity/speed**3


def conservative_bh_drag_step(carrier,bath,acceleration_cm_s2,proper_dt_seconds,*,seed,counter,
    maximum_particles,memory_gib,maximum_fractional_relative_velocity_change,minimum_inertia_eigenvalue_ratio,
    conservation_relative_tolerance,momentum_absolute_floor_msun_km_s,energy_absolute_floor_msun_km2_s2,
    angular_absolute_floor_msun_kpc_km_s):
    """Conservative fixed-position drag kick with an explicitly resolved bath.

    Masses are gravitating Msun, positions proper kpc and velocities peculiar
    km/s. The calibrated force must supply a gravity-carrier acceleration;
    no physical-BH/composite-mass conversion is guessed here. Heat becomes
    zero-momentum, zero-angular-momentum bath velocity degrees of freedom.
    """
    import hashlib
    import numpy as np
    from core.data import exact_uint64_ids
    mass=finite_number(carrier['gravity_mass_msun'],'BH gravity carrier mass',strict=True)
    dt=finite_number(proper_dt_seconds,'proper source duration',strict=True)
    workspace=finite_number(memory_gib,'source workspace',strict=True)*1024**3
    for name,value in (('seed',seed),('counter',counter)):
        if type(value)is not int or not 0<=value<2**64:raise ValueError('Exact uint64 source '+name+' required')
    if type(maximum_particles)is not int or maximum_particles<3:raise ValueError('Bounded complete bath of at least three particles required')
    fraction=finite_number(maximum_fractional_relative_velocity_change,'source fractional change',strict=True)
    eigen_floor=finite_number(minimum_inertia_eigenvalue_ratio,'inertia eigenvalue ratio',strict=True)
    tolerance=finite_number(conservation_relative_tolerance,'conservation tolerance',strict=True)
    if fraction>=1 or eigen_floor>=1 or tolerance>=1:raise ValueError('Source fractional/domain/conservation bounds must lie below one')
    p_floor=finite_number(momentum_absolute_floor_msun_km_s,'momentum floor',strict=True)
    e_floor=finite_number(energy_absolute_floor_msun_km2_s2,'energy floor',strict=True)
    l_floor=finite_number(angular_absolute_floor_msun_kpc_km_s,'angular momentum floor',strict=True)
    ids=exact_uint64_ids(bath['particle_ids']);weights=np.asarray(bath['mass_msun'],float)
    carrier_id=int(exact_uint64_ids([carrier['particle_id']])[0])
    if carrier_id==0 or np.any(ids==carrier_id):raise ValueError('A live BH carrier must have a distinct nonzero native identity outside its bath')
    x=np.asarray(bath['position_proper_kpc'],float);v=np.asarray(bath['velocity_peculiar_km_s'],float)
    bx=np.asarray(carrier['position_proper_kpc'],float);bv=np.asarray(carrier['velocity_peculiar_km_s'],float);a=np.asarray(acceleration_cm_s2,float)
    if not 3<=len(ids)<=maximum_particles or weights.shape!=(len(ids),) or x.shape!=(len(ids),3) or v.shape!=x.shape or any(value.shape!=(3,) for value in (bx,bv,a)):
        raise ValueError('Aligned complete source carrier and exact-ID bath state required')
    if len(ids)*2048>workspace:raise MemoryError('Complete constrained bath source arrays exceed allocation')
    if not all(np.isfinite(value).all() for value in (weights,x,v,bx,bv,a)) or np.any(weights<=0) or len(np.unique(ids))!=len(ids):
        raise ValueError('Finite positive source masses and distinct bath identities required')
    # Stable ID ordering fixes reduction and keyed-direction order.
    order=np.argsort(ids,kind='stable');inverse=np.argsort(order);ids=ids[order];weights=weights[order];x=x[order];v=v[order]
    total=float(weights.sum());mean=np.sum(weights[:,None]*v,axis=0)/total;relative=bv-mean
    delta_bh=a*dt/1e5;speed=float(np.linalg.norm(relative))
    if speed==0 and np.any(delta_bh) or np.dot(delta_bh,relative)>0:
        raise ValueError('Drag source accelerates the carrier relative to its measured bath')
    if speed>0 and np.linalg.norm(delta_bh)*(1+mass/total)>fraction*speed:
        raise ValueError('Conservative drag kick exceeds the frozen source fraction; reduce the proper timestep')
    impulse=mass*delta_bh;r=x-bx;center=np.sum(weights[:,None]*r,axis=0)/total;R=r-center
    inertia=np.eye(3)*np.sum(weights*np.sum(R*R,axis=1))-np.einsum('n,ni,nj->ij',weights,R,R)
    eigen=np.linalg.eigvalsh(inertia)
    if not np.isfinite(eigen).all() or eigen[-1]<=0 or eigen[0]<=eigen_floor*eigen[-1]:
        raise ValueError('Bath geometry cannot support the declared angular-momentum conserving source')
    rotation=np.linalg.solve(inertia,np.cross(center,impulse))
    delta_bath=-impulse/total+np.cross(rotation,R)
    com=(mass*bv+total*mean)/(mass+total)
    work=mass*np.dot(bv-com,delta_bh)+np.sum(weights*np.einsum('ni,ni->n',v-com,delta_bath))
    quadratic=.5*mass*np.dot(delta_bh,delta_bh)+.5*np.sum(weights*np.sum(delta_bath*delta_bath,axis=1))
    heat=-float(work+quadratic)
    if not math.isfinite(heat) or heat<0:raise ValueError('Torque-balanced source is not dissipative at this duration/state; reduce the timestep or reject this bath')
    new_bh=bv+delta_bh;new_bath=v+delta_bath
    if heat>0:
        raw=np.empty_like(v)
        for row,identity in enumerate(ids):
            for axis in range(3):
                key=f'research2-bh-heat-v1:{seed}:{carrier_id}:{int(identity)}:{counter}:{axis}'.encode('ascii')
                bits=int.from_bytes(hashlib.sha256(key).digest()[:8],'little')
                raw[row,axis]=2*((bits>>11)*2.**-53)-1
        raw-=np.sum(weights[:,None]*raw,axis=0)/total
        spin=np.linalg.solve(inertia,np.sum(weights[:,None]*np.cross(R,raw),axis=0))
        kicks=raw-np.cross(spin,R)
        A=.5*float(np.sum(weights*np.sum(kicks*kicks,axis=1)))
        B=float(np.sum(weights*np.einsum('ni,ni->n',new_bath-com,kicks)))
        if not math.isfinite(A) or not math.isfinite(B) or A<=0:raise ValueError('Bath lacks resolved zero-momentum/torque heat degrees of freedom')
        radical=math.hypot(B,2*math.sqrt(A)*math.sqrt(heat))
        amplitude=2*heat/(radical+B) if B>=0 else (radical-B)/(2*A)
        new_bath+=amplitude*kicks
    actual_bh=new_bh-bv;actual_bath=new_bath-v
    momentum_residual=mass*actual_bh+np.sum(weights[:,None]*actual_bath,axis=0)
    angular_residual=np.sum(weights[:,None]*np.cross(r,actual_bath),axis=0)
    energy_residual=float(mass*np.dot(bv-com,actual_bh)+.5*mass*np.dot(actual_bh,actual_bh)
        +np.sum(weights*np.einsum('ni,ni->n',v-com,actual_bath))+.5*np.sum(weights*np.sum(actual_bath*actual_bath,axis=1)))
    p_scale=max(p_floor,float(np.linalg.norm(impulse)),float(np.sum(weights*np.linalg.norm(actual_bath,axis=1))))
    l_scale=max(l_floor,float(np.sum(weights*np.linalg.norm(np.cross(r,actual_bath),axis=1))))
    e_scale=max(e_floor,heat,abs(float(work)),float(quadratic))
    if not all(np.isfinite(value).all() for value in (new_bh,new_bath,momentum_residual,angular_residual)) or not math.isfinite(energy_residual):
        raise ArithmeticError('Conservative bath source overflows')
    if np.linalg.norm(momentum_residual)>tolerance*p_scale or np.linalg.norm(angular_residual)>tolerance*l_scale or abs(energy_residual)>tolerance*e_scale:
        raise ArithmeticError('Actual rounded source updates exceed their frozen momentum/angular/energy budgets')
    return {'carrier_velocity_peculiar_km_s':new_bh,'bath_velocity_peculiar_km_s':new_bath[inverse],
        'bh_impulse_msun_km_s':impulse,'bath_heat_msun_km2_s2':heat,'momentum_residual_msun_km_s':momentum_residual,
        'angular_residual_msun_kpc_km_s':angular_residual,'energy_residual_msun_km2_s2':energy_residual,
        'source_definition':'fixed-position gravity-carrier kick; torque-balanced bath recoil and ID-keyed constrained heat; no hidden momentum/energy reservoir'}


def validate_dm_wake_force_request(parameters):
    request=parameters['request']
    keys={'schema','snapshot','particle_namespace',
        'radial_edges_proper_kpc','force_support_proper_kpc','maximum_particles','memory_gib','chunk_rows'}
    if not isinstance(request,dict):raise ValueError('Native wake-force request must be an object')
    version=request.get('schema')
    if version=='native-dm-wake-force-request-v1':keys|={'probe_position_proper_kpc','probe_velocity_peculiar_km_s'}
    elif version in ('native-dm-wake-force-request-v2','native-dm-wake-force-request-v3'):
        keys|={'probe_particle_id'}
        if version.endswith('v3'):keys.add('maximum_reference_operator_difference_cm_s2')
    else:raise ValueError('Unsupported native wake-force request version')
    if set(request)!=keys:
        raise ValueError('Complete original native DM wake-force measurement request required')
    for key in ('snapshot','maximum_particles','chunk_rows'):
        if type(request[key])is not int or request[key]<(0 if key=='snapshot' else 1):raise ValueError('Exact bounded wake-force work required: '+key)
    for key in ('memory_gib','force_support_proper_kpc'):finite_number(request[key],key,strict=True)
    if version.endswith('v3'):finite_number(request['maximum_reference_operator_difference_cm_s2'],'reference/native pair support discrepancy limit')
    if not isinstance(request['particle_namespace'],str) or not request['particle_namespace']:raise ValueError('Explicit original DM namespace required')
    import numpy as np
    if version=='native-dm-wake-force-request-v1':
        for key in ('probe_position_proper_kpc','probe_velocity_peculiar_km_s'):
            value=np.asarray(request[key],float)
            if value.shape!=(3,) or not np.isfinite(value).all():raise ValueError('Finite physical wake probe state required')
    elif type(request['probe_particle_id'])is not int or not 0<request['probe_particle_id']<2**64:
        raise ValueError('Exact nonzero uint64 original native BH probe ID required')
    edges=np.asarray(request['radial_edges_proper_kpc'],float)
    if edges.ndim!=1 or not 2<=len(edges)<=4096 or not np.isfinite(edges).all() or edges[0]<0 or np.any(np.diff(edges)<=0):
        raise ValueError('Bounded ordered nonnegative wake-force shell edges required')
    if request['maximum_particles']*800+request['chunk_rows']*512+len(edges)*1024>request['memory_gib']*1024**3:
        raise MemoryError('Declared complete DM census and force measurement workspace exceeds allocation')
    return request


def _read_native_wake_bh_probe(files,request,sources):
    """Select a living BH without loading an unbounded BH census.

    Gravity-carrier mass and physical BH mass are distinct native quantities.
    No caller-provided position, velocity, mass or softening substitutes for
    the retained particle state. The requested measurement spline remains a
    reference operator; this reader does not establish native force equality.
    """
    import h5py
    import numpy as np
    from core.data import dataset_to_cgs_factor,exact_uint64_ids,snapshot_total_counts
    from core.foundation import file_sha256,KPC_CM,MSUN_G
    probe_id=request['probe_particle_id'];found=None;declared=None;rows=0
    for path in files:
        with h5py.File(path,'r') as native:
            header=native['Header'].attrs;p=native['Parameters'].attrs
            total=int(snapshot_total_counts(header)[5])
            if declared is not None and declared!=total:raise ValueError('Native BH census totals change across shards')
            declared=total
            if total>request['maximum_particles']:raise MemoryError('Native BH identity selection exceeds declared census bound')
            n=int(header['NumPart_ThisFile'][5]);rows+=n
            if not n:continue
            group=native['PartType5'];a=float(header['Time']) if int(p['ComovingIntegrationOn']) else 1.;h=float(p['HubbleParam'])
            if 'BH_Mass' not in group:raise ValueError('Native BH probe requires separately stored physical BH_Mass')
            for name in ('ParticleIDs','Coordinates','Velocities','BH_Mass'):
                expected=(n,3) if name in ('Coordinates','Velocities') else (n,)
                if group[name].shape!=expected:raise ValueError('Incomplete native BH state field: '+name)
            if 'Masses' in group and group['Masses'].shape!=(n,):raise ValueError('Incomplete BH gravity-carrier masses')
            for begin in range(0,n,request['chunk_rows']):
                ids=exact_uint64_ids(group['ParticleIDs'][begin:min(n,begin+request['chunk_rows'])],unique=False,allow_empty=True)
                locations=np.flatnonzero(ids==np.uint64(probe_id))
                if len(locations)>1 or (found is not None and len(locations)):raise ValueError('Native BH probe identity appears more than once')
                if not len(locations):continue
                index=begin+int(locations[0])
                position=np.asarray(group['Coordinates'][index],float)*dataset_to_cgs_factor(group['Coordinates'].attrs,a,h)/KPC_CM
                velocity=np.asarray(group['Velocities'][index],float)*dataset_to_cgs_factor(group['Velocities'].attrs,a,h)/1e5
                physical=float(group['BH_Mass'][index])*dataset_to_cgs_factor(group['BH_Mass'].attrs,a,h)/MSUN_G
                carrier=(float(group['Masses'][index])*dataset_to_cgs_factor(group['Masses'].attrs,a,h)/MSUN_G
                    if 'Masses' in group else float(header['MassTable'][5])*float(p['UnitMass_in_g'])/h/MSUN_G)
                if not np.isfinite(position).all() or not np.isfinite(velocity).all() or not np.isfinite([physical,carrier]).all() or min(physical,carrier)<=0:
                    raise ValueError('Living native BH probe requires finite phase state and positive physical/carrier masses')
                found={'kind':'original_native_bh_particle','particle_id':probe_id,'particle_namespace':request['particle_namespace'],
                    'position_proper_kpc':position.tolist(),'velocity_peculiar_km_s':velocity.tolist(),
                    'physical_bh_mass_msun':physical,'gravity_carrier_mass_msun':carrier,
                    'snapshot_file':str(path.resolve()),'snapshot_sha256':sources[str(path.resolve())],
                    'native_row':index,'native_force_support_equivalence':'NOT_ESTABLISHED'}
                if request['schema']=='native-dm-wake-force-request-v3':
                    if 'GravitySoftening' not in group or group['GravitySoftening'].shape!=(n,) or group['GravitySoftening'].dtype.kind!='f':
                        raise ValueError('Actual saved per-particle BH softening required for native pair support measurement')
                    softening=float(group['GravitySoftening'][index])*dataset_to_cgs_factor(group['GravitySoftening'].attrs,a,h)/KPC_CM
                    if not math.isfinite(softening) or softening<=0:raise ValueError('Native BH Plummer-equivalent softening must be positive/finite')
                    found['native_plummer_equivalent_softening_proper_kpc']=softening
                    found['native_spline_support_proper_kpc']=2.8*softening
                    if not math.isfinite(found['native_spline_support_proper_kpc']):raise ArithmeticError('Native BH spline support overflows')
        if sources.get(str(path.resolve()))!=file_sha256(path):raise ValueError('Native BH state source changed during selection')
    if rows!=declared:raise ValueError('Native BH census is incomplete')
    if found is None:raise ValueError('Requested living BH ID is absent from the original native snapshot')
    return found


def _read_native_wake_dm_support(files,request,particle_ids,sources):
    """Align actual saved DM softenings to the complete exact original census."""
    import h5py,numpy as np
    from core.data import dataset_to_cgs_factor,exact_uint64_ids
    from core.foundation import KPC_CM,file_sha256
    ids=exact_uint64_ids(particle_ids,allow_empty=False);order=np.argsort(ids);ordered=ids[order]
    support=np.empty(len(ids));seen=np.zeros(len(ids),bool)
    for path in files:
        with h5py.File(path,'r') as native:
            n=int(native['Header'].attrs['NumPart_ThisFile'][1])
            if not n:continue
            group=native['PartType1'];p=native['Parameters'].attrs
            a=float(native['Header'].attrs['Time']) if int(p['ComovingIntegrationOn']) else 1.;h=float(p['HubbleParam'])
            if 'GravitySoftening' not in group or group['GravitySoftening'].shape!=(n,) or group['GravitySoftening'].dtype.kind!='f':
                raise ValueError('Actual saved per-particle DM softening required for native pair support measurement')
            factor=2.8*dataset_to_cgs_factor(group['GravitySoftening'].attrs,a,h)/KPC_CM
            for begin in range(0,n,request['chunk_rows']):
                end=min(n,begin+request['chunk_rows']);current=exact_uint64_ids(group['ParticleIDs'][begin:end],allow_empty=False)
                at=np.searchsorted(ordered,current)
                if np.any(at>=len(ids)) or not np.array_equal(ordered[at],current):raise ValueError('Native DM softening rows change the original particle census')
                slots=order[at]
                if np.any(seen[slots]):raise ValueError('Native DM softening rows duplicate original IDs')
                values=np.asarray(group['GravitySoftening'][begin:end],float)*factor
                if not np.isfinite(values).all() or np.any(values<=0):raise ValueError('Positive finite native DM spline supports required')
                support[slots]=values;seen[slots]=True
        if sources.get(str(path.resolve()))!=file_sha256(path):raise ValueError('Native softening snapshot changes during measurement')
    if not seen.all():raise ValueError('Native softening shard list omits original DM identities')
    return support


def _wake_spline_acceleration_factor(radius,support):
    """Compact-spline pair factor with support h=max(h_BH,h_DM)."""
    import numpy as np
    radius=np.asarray(radius,float);support=np.broadcast_to(np.asarray(support,float),radius.shape)
    if not np.isfinite(radius).all() or not np.isfinite(support).all() or np.any(radius<0) or np.any(support<=0):
        raise ValueError('Finite radii and positive spline supports required')
    u=radius/support;factor=np.empty_like(radius);inner=u<.5;outer=(u>=.5)&(u<1);far=u>=1
    factor[inner]=(32/3+u[inner]**2*(32*u[inner]-192/5))/support[inner]**3
    factor[outer]=(64/3-48*u[outer]+192/5*u[outer]**2-32/3*u[outer]**3-1/(15*u[outer]**3))/support[outer]**3
    factor[far]=1/radius[far]**3
    if not np.isfinite(factor).all():raise ArithmeticError('Native compact-spline force factor overflows')
    return factor


def dm_wake_force_input_bindings(parameters):
    from core.sidm_analysis import _stage_native_bindings
    validate_dm_wake_force_request(parameters)
    return _stage_native_bindings(parameters)


def publish_dm_wake_force(parameters,destination):
    """Measure aperture DM forces and bath moments from original snapshots.

    The probe and spline support define a common reference force operator,
    rather than claiming the native tree actually used this requested support.
    Perturbed-minus-baseline and fine-minus-coarse comparisons must preserve
    that operator/probe definition and remain separate calibration products.
    """
    import h5py
    import numpy as np
    from pathlib import Path
    from core.observation_geometry import read_native_gravitating_particles
    from core.halos import _snapshot_files
    from core.data import export_numeric_audit
    from core.foundation import file_sha256,KPC_CM,MSUN_G,G_CGS
    request=validate_dm_wake_force_request(parameters);sources=dm_wake_force_input_bindings(parameters)
    root=Path(parameters['native_run_directory']).resolve();files=_snapshot_files(root,request['snapshot'])
    if not files:raise FileNotFoundError('Wake-force original native snapshot set is missing')
    configuration=root/'input/Config.sh'
    if sources.get(str(configuration))!=file_sha256(configuration):raise ValueError('Wake measurement lacks its original native compile configuration')
    flags={line.split('#',1)[0].strip().split('=',1)[0] for line in configuration.read_text().splitlines()};periodic='PERIODIC' in flags
    common=None
    for path in files:
        if sources.get(str(path.resolve()))!=file_sha256(path):raise ValueError('Wake snapshot is outside the completed native generation')
        with h5py.File(path,'r') as native:
            p=native['Parameters'].attrs;a=float(native['Header'].attrs['Time']) if int(p['ComovingIntegrationOn']) else 1.
            current=(a,float(p['HubbleParam']),float(p['UnitLength_in_cm']),float(p['BoxSize']))
            if common is not None and current!=common:raise ValueError('Wake snapshot shards change their physical box/epoch')
            common=current
    a,h,L,box_code=common
    if not np.isfinite(common).all() or min(common)<=0:raise ValueError('Positive original wake epoch/box/units required')
    box=box_code*L*a/h/KPC_CM;edges=np.asarray(request['radial_edges_proper_kpc'],float);bins=len(edges)-1
    if periodic and edges[-1]>=box/2:raise ValueError('Wake aperture exceeds unique periodic nearest-image support')
    population,particles=read_native_gravitating_particles(root,request['snapshot'],species=[1],maximum_particles=request['maximum_particles'],
        memory_gib=request['memory_gib']/4,namespace=request['particle_namespace'])
    import yaml
    config_path=root/'config.yaml'
    if sources.get(str(config_path))!=file_sha256(config_path):raise ValueError('Wake measurement requires its original native physical configuration')
    config=yaml.safe_load(config_path.read_text())
    from core.foundation import canonical_digest
    interaction=config.get('sidm',{'enabled':False})
    physical_interaction={key:interaction[key] for key in ('enabled','physical_model','cross_section_normalization_cm2_per_g','particle_convention') if key in interaction}
    if isinstance(physical_interaction.get('physical_model'),dict):
        physical_interaction['physical_model']={'sha256':physical_interaction['physical_model']['sha256']}
    physics={'cosmology':config['cosmology'],'ede':config.get('ede',{}),'wdm':config.get('wdm',{}),'sidm':physical_interaction,
        'native_physics':config['simulation']['physics'],'hydro':config.get('hydro',{}),'chemistry':config.get('chemistry',{}),
        'stellar_evolution':config.get('stellar_evolution',{})}
    sources.update(population['source_hashes'])
    native_support=request['schema']=='native-dm-wake-force-request-v3'
    if request['schema'] in ('native-dm-wake-force-request-v2','native-dm-wake-force-request-v3'):
        probe_state=_read_native_wake_bh_probe(files,request,sources)
        probe=np.asarray(probe_state['position_proper_kpc'],float)
        reference_velocity=np.asarray(probe_state['velocity_peculiar_km_s'],float)
    else:
        probe=np.asarray(request['probe_position_proper_kpc'],float)
        reference_velocity=np.asarray(request['probe_velocity_peculiar_km_s'],float)
        probe_state={'kind':'declared_reference_probe','native_particle_binding':'NOT_ESTABLISHED',
            'position_proper_kpc':probe.tolist(),'velocity_peculiar_km_s':reference_velocity.tolist()}
    import hashlib
    census_hash=hashlib.sha256();order=np.argsort(particles['particle_ids'],kind='stable')
    for key in ('particle_ids','position_proper_kpc','velocity_peculiar_kms','mass_msun'):
        value=particles[key];census_hash.update(key.encode('ascii'))
        for begin in range(0,len(order),request['chunk_rows']):
            census_hash.update(np.ascontiguousarray(value[order[begin:begin+request['chunk_rows']]]).tobytes())
    support=request['force_support_proper_kpc']
    pair_support=None;native_force=np.zeros((bins,3));support_min=np.full(bins,np.inf);support_max=np.zeros(bins)
    if native_support:
        if 'INDIVIDUAL_GRAVITY_SOFTENING' not in flags:raise ValueError('Native pair support measurement needs actual saved individual gravity softening output')
        pair_support=np.maximum(_read_native_wake_dm_support(files,request,particles['particle_ids'],sources),probe_state['native_spline_support_proper_kpc'])
    counts=np.zeros(bins,np.uint64);mass=np.zeros(bins);momentum=np.zeros((bins,3));force=np.zeros((bins,3));covariance=np.zeros((bins,3,3))
    def samples():
        for begin in range(0,len(particles['particle_ids']),request['chunk_rows']):
            sl=slice(begin,min(len(particles['particle_ids']),begin+request['chunk_rows']))
            delta=particles['position_proper_kpc'][sl]-probe
            if periodic:delta-=box*np.rint(delta/box)
            radius=np.linalg.norm(delta,axis=1);at=np.searchsorted(edges,radius,side='right')-1
            keep=(at>=0)&(at<bins)
            yield at[keep],delta[keep],radius[keep],particles['mass_msun'][sl][keep],particles['velocity_peculiar_kms'][sl][keep],None if pair_support is None else pair_support[sl][keep]
    for at,delta,radius,weights,velocity,actual_support in samples():
        factor=_wake_spline_acceleration_factor(radius,support)
        np.add.at(counts,at,np.uint64(1));np.add.at(mass,at,weights)
        np.add.at(momentum,at,weights[:,None]*velocity)
        np.add.at(force,at,(G_CGS*MSUN_G/KPC_CM**2)*weights[:,None]*delta*factor[:,None])
        if actual_support is not None:
            actual_factor=_wake_spline_acceleration_factor(radius,actual_support)
            np.add.at(native_force,at,(G_CGS*MSUN_G/KPC_CM**2)*weights[:,None]*delta*actual_factor[:,None])
            np.minimum.at(support_min,at,actual_support);np.maximum.at(support_max,at,actual_support)
    available=mass>0;mean=np.divide(momentum,mass[:,None],out=np.zeros_like(momentum),where=available[:,None])
    for at,delta,radius,weights,velocity,actual_support in samples():
        residual=velocity-mean[at];np.add.at(covariance,at,weights[:,None,None]*residual[:,:,None]*residual[:,None,:])
    covariance=np.divide(covariance,mass[:,None,None],out=np.zeros_like(covariance),where=available[:,None,None])
    density=mass/(4*math.pi/3*(edges[1:]**3-edges[:-1]**3))
    aperture_mass=float(mass.sum());aperture_available=aperture_mass>0
    aperture_mean=momentum.sum(axis=0)/aperture_mass if aperture_available else np.zeros(3)
    # Combine extensive within-shell moments and between-shell mean offsets.
    # Averaging shell dispersions alone would lose resolved bulk gradients.
    offsets=mean-aperture_mean
    aperture_covariance=(np.sum(mass[:,None,None]*(covariance+offsets[:,:,None]*offsets[:,None,:]),axis=0)/aperture_mass
        if aperture_available else np.zeros((3,3)))
    arrays={'shell_acceleration_cm_s2':force,'aperture_acceleration_cm_s2':force.sum(axis=0),'shell_mass_msun':mass,
        'shell_particle_count':counts,'shell_density_msun_kpc3':density,'shell_mean_velocity_km_s':mean,
        'shell_velocity_covariance_km2_s2':covariance,'shell_velocity_moments_available':available,
        'aperture_mass_msun':np.asarray([aperture_mass]),'aperture_velocity_moments_available':np.asarray([aperture_available]),
        'aperture_mean_velocity_km_s':aperture_mean,'aperture_velocity_covariance_km2_s2':aperture_covariance,
        'aperture_probe_relative_velocity_km_s':reference_velocity-aperture_mean if aperture_available else np.zeros(3),
        'shell_probe_relative_velocity_km_s':np.subtract(reference_velocity,mean,out=np.zeros_like(mean),where=available[:,None]),'radial_edges_proper_kpc':edges}
    if native_support:
        difference=native_force-force
        support_min[~available]=0
        arrays.update(native_pair_shell_acceleration_cm_s2=native_force,native_pair_aperture_acceleration_cm_s2=native_force.sum(axis=0),
            native_pair_minus_reference_shell_acceleration_cm_s2=difference,
            native_pair_shell_support_min_proper_kpc=support_min,native_pair_shell_support_max_proper_kpc=support_max,
            native_pair_shell_support_available=available.copy())
    if any(not np.isfinite(value).all() for value in arrays.values()):raise ArithmeticError('Measured wake force or bath moments overflow')
    source=Path(__file__).resolve();sources[str(source)]=file_sha256(source)
    report={'schema':'measured-native-dm-wake-force-v1','execution_status':'COMPLETE','request':request,'epoch':population['epoch'],
        'operator_definition':{'kernel':'gadget_compact_spline_reference','force_support_proper_kpc':support,
            'probe_position_proper_kpc':probe.tolist(),'probe_velocity_peculiar_km_s':reference_velocity.tolist(),
            'radial_edges_proper_kpc':edges.tolist(),'periodic_nearest_image':periodic},
        'particle_namespace':request['particle_namespace'],'source_hashes':sources,'science_acceptance':'NOT_ESTABLISHED',
        'probe_state':probe_state,
        'native_phase_seed':config['simulation']['random_seed'],'native_operator_seed':interaction.get('operator_seed'),
        'native_dm_physics_signature':canonical_digest(physics),'native_physical_interaction':physical_interaction,
        'native_proper_box_kpc':box,
        'native_dm_census_sha256':census_hash.hexdigest(),
        'native_dm_particle_count':len(particles['particle_ids']),'native_dm_total_mass_msun':float(particles['mass_msun'].sum()),
        'native_dm_mass_range_msun':[float(particles['mass_msun'].min()),float(particles['mass_msun'].max())] if len(particles['particle_ids']) else None,
        'scope':'original DM-only aperture force and resolved bath moments at an explicit probe; no background subtraction, total periodic force, drag calibration or native friction acceptance'}
    if native_support:
        discrepancy=float(np.sum(np.linalg.norm(difference,axis=1)))
        report['native_pair_support_measurement']={'schema':'measured-native-pair-softening-support-v1',
            'kernel':'gadget_compact_spline_monopole','support_rule':'max(2.8*saved_BH_GravitySoftening,2.8*saved_DM_GravitySoftening)',
            'snapshot_softening_definition':'saved SofteningTable value, Plummer-equivalent, converted to proper kpc',
            'reference_difference_shell_norm_sum_cm_s2':discrepancy,
            'status':'WITHIN_DECLARED_SUPPORT_DIFFERENCE' if discrepancy<=request['maximum_reference_operator_difference_cm_s2'] else 'REFERENCE_SUPPORT_MISMATCH',
            'force_scope':'reconstructed saved-state nearest-image DM aperture monopoles; excludes native arithmetic differences, tree approximation, PM splitting and Ewald images',
            'native_total_force_equivalence':'NOT_ESTABLISHED'}
    units={'shell_acceleration_cm_s2':'cm / s^2','aperture_acceleration_cm_s2':'cm / s^2','shell_mass_msun':'Msun',
        'aperture_mass_msun':'Msun','aperture_mean_velocity_km_s':'km / s','aperture_probe_relative_velocity_km_s':'km / s',
        'aperture_velocity_covariance_km2_s2':'(km / s)^2',
        'shell_density_msun_kpc3':'Msun / kpc^3','shell_mean_velocity_km_s':'km / s','shell_probe_relative_velocity_km_s':'km / s',
        'shell_velocity_covariance_km2_s2':'(km / s)^2','radial_edges_proper_kpc':'proper kpc'}
    if native_support:
        for name in ('native_pair_shell_acceleration_cm_s2','native_pair_aperture_acceleration_cm_s2','native_pair_minus_reference_shell_acceleration_cm_s2'):units[name]='cm / s^2'
        for name in ('native_pair_shell_support_min_proper_kpc','native_pair_shell_support_max_proper_kpc'):units[name]='proper kpc'
    return export_numeric_audit(destination,report,arrays,{name:{'units':units.get(name,'dimensionless'),'quantity':'original native DM wake '+name} for name in arrays},
        algorithm=source,source_hashes=sources,context={'epoch':population['epoch'],'coordinates':{'frame':'inertial proper probe-centred nearest-image aperture; peculiar velocity'}})


def validate_dm_wake_kernel_comparison_request(parameters):
    request=parameters['request'];keys={'schema','native_diagnostic','reference_force','memory_gib','maximum_bins','maximum_diagnostic_bytes',
        'maximum_native_epoch_error','maximum_probe_position_error_kpc','relative_mass_tolerance','maximum_shell_acceleration_error_cm_s2'}
    if isinstance(request,dict) and 'maximum_closed_total_force_error_cm_s2' in request:keys.add('maximum_closed_total_force_error_cm_s2')
    if not isinstance(request,dict) or set(request)!=keys or request['schema']!='native-dm-wake-kernel-comparison-request-v1':
        raise ValueError('Complete native kernel/reference comparison and frozen limits required')
    for name in ('native_diagnostic','reference_force'):
        if not isinstance(request[name],dict) or set(request[name])!={'path','sha256'}:raise ValueError('Exact source-bound kernel comparison inputs required')
    for name in ('maximum_bins','maximum_diagnostic_bytes'):
        if type(request[name])is not int or request[name]<1:raise ValueError('Exact positive diagnostic allocation bound required')
    if request['maximum_bins']>4095:raise ValueError('Native diagnostic shell allocation exceeds its supported bound')
    for name in ('memory_gib','maximum_native_epoch_error','maximum_probe_position_error_kpc','relative_mass_tolerance','maximum_shell_acceleration_error_cm_s2'):
        finite_number(request[name],name,strict=name in ('memory_gib','relative_mass_tolerance'))
    if request['relative_mass_tolerance']>=1:raise ValueError('Native mass comparison tolerance must be below one')
    if 'maximum_closed_total_force_error_cm_s2' in request:finite_number(request['maximum_closed_total_force_error_cm_s2'],'closed native total force error')
    if request['maximum_diagnostic_bytes']*16+request['maximum_bins']*8192>request['memory_gib']*1024**3:
        raise MemoryError('Native kernel metadata/array allocation exceeds workspace')
    return request


def _wake_kernel_comparison_inputs(parameters):
    import json
    from pathlib import Path
    from core.sidm_analysis import _load_product
    from core.foundation import file_sha256
    request=validate_dm_wake_kernel_comparison_request(parameters);base=Path(parameters['request_base']).resolve()
    reference,arrays,sources=_load_product(request['reference_force'],base,request['memory_gib']*1024**3/2)
    if reference.get('schema')!='measured-native-dm-wake-force-v1' or reference.get('execution_status')!='COMPLETE':raise ValueError('Original complete native wake reference measurement required')
    path=(base/request['native_diagnostic']['path']).resolve();seal=path.with_name(path.name+'.sha256')
    if path.is_symlink() or seal.is_symlink() or path.stat().st_size>request['maximum_diagnostic_bytes'] or sources.get(str(path))!=request['native_diagnostic']['sha256'] or file_sha256(path)!=request['native_diagnostic']['sha256']:
        raise ValueError('Native diagnostic is outside the actual original completed native source generation')
    if sources.get(str(seal))!=file_sha256(seal) or seal.read_text().strip()!=request['native_diagnostic']['sha256']:
        raise ValueError('Native diagnostic publication seal changed')
    def unique(items):
        record={}
        for key,value in items:
            if key in record:raise ValueError('Duplicate native diagnostic JSON key')
            record[key]=value
        return record
    diagnostic=json.loads(path.read_text(),object_pairs_hook=unique,parse_constant=lambda value:(_ for _ in ()).throw(ValueError('Nonfinite native diagnostic JSON')))
    if diagnostic.get('schema')!='native-bh-wake-monopole-diagnostic-v1' or diagnostic.get('force_scope')!='native_kernel_aperture_monopoles_excludes_tree_pm_ewald':
        raise ValueError('Actual native aperture-kernel diagnostic scope required')
    if diagnostic['native_physics_signature']!=reference['native_dm_physics_signature']:raise ValueError('Native diagnostic changes the actual reference physical configuration')
    sources[str(path)]=file_sha256(path);sources[str(seal)]=file_sha256(seal)
    return request,reference,arrays,diagnostic,sources


def dm_wake_kernel_comparison_input_bindings(parameters):
    return _wake_kernel_comparison_inputs(parameters)[-1]


def publish_dm_wake_kernel_comparison(parameters,destination):
    import numpy as np
    from pathlib import Path
    from core.foundation import file_sha256
    from core.data import export_numeric_audit
    request,reference,rv,diagnostic,sources=_wake_kernel_comparison_inputs(parameters)
    def integer(name,value):
        if not isinstance(value,str) or not value.isascii() or not value.isdecimal() or not 0<=int(value)<2**64:raise ValueError('Exact native diagnostic uint64 decimal required: '+name)
        return int(value)
    count=integer('complete_dm_count',diagnostic['complete_dm_count_decimal']);probe=reference['probe_state']
    if probe.get('kind')!='original_native_bh_particle' or integer('carrier',diagnostic['carrier_id_decimal'])!=probe['particle_id'] or count!=reference['native_dm_particle_count']:
        raise ValueError('Native diagnostic changes its original living BH/fine-DM census')
    integer('timeline',diagnostic['native_timeline_decimal']);integer('measurement_id',diagnostic['measurement_id_decimal'])
    epoch=reference['epoch'];expected_epoch=epoch.get('actual_scale_factor') if epoch['canonical_time']=='scale_factor' else epoch.get('native_time_code')
    actual=finite_number(diagnostic['native_time'],'actual diagnostic native time')
    if expected_epoch is None or abs(actual-expected_epoch)>request['maximum_native_epoch_error']:raise ValueError('Native diagnostic misses the actual saved reference epoch')
    position=np.asarray(diagnostic['probe_position_proper_kpc'],float);expected=np.asarray(probe['position_proper_kpc'],float)
    if position.shape!=(3,) or not np.isfinite(position).all():raise ValueError('Complete native diagnostic probe position required')
    difference=position-expected
    if reference['operator_definition']['periodic_nearest_image']:
        box=reference['native_proper_box_kpc'];difference-=box*np.rint(difference/box)
    if np.linalg.norm(difference)>request['maximum_probe_position_error_kpc']:raise ValueError('Native diagnostic probe phase differs beyond its frozen budget')
    for name in ('physical_bh_mass_msun','gravity_carrier_mass_msun'):
        value=finite_number(diagnostic[name],name,strict=True)
        if abs(value/probe[name]-1)>request['relative_mass_tolerance']:raise ValueError('Native diagnostic changes its original probe masses')
    edges=np.asarray(diagnostic['radial_edges_proper_kpc'],float)
    if not np.array_equal(edges,rv['radial_edges_proper_kpc']) or not 1<=len(edges)-1<=request['maximum_bins'] or len(diagnostic['shells'])!=len(edges)-1:
        raise ValueError('Native diagnostic changes its original aperture shell definition')
    shells=diagnostic['shells'];counts=np.asarray([integer('shell count',row['particle_count_decimal']) for row in shells],np.uint64)
    if not np.array_equal(counts,rv['shell_particle_count']) or integer('pair evaluations',diagnostic['pair_evaluations_decimal'])!=sum(int(value) for value in counts):
        raise ValueError('Native diagnostic shell census differs from the original saved reference')
    force=np.asarray([row['acceleration_cm_s2'] for row in shells],float);mass=np.asarray([row['mass_msun'] for row in shells],float)
    lower=np.asarray([row['minimum_support_proper_kpc'] for row in shells],float);upper=np.asarray([row['maximum_support_proper_kpc'] for row in shells],float)
    if force.shape!=(len(shells),3) or not all(np.isfinite(value).all() for value in (force,mass,lower,upper)) or np.any(mass<0) or np.any(lower<0) or np.any(upper<lower):
        raise ValueError('Native kernel diagnostic shell state is invalid')
    populated=counts>0
    if np.any(lower[populated]<=0) or np.any(mass[populated]<=0) or np.any(lower[~populated]!=0) or np.any(upper[~populated]!=0) or np.any(mass[~populated]!=0):
        raise ValueError('Native kernel support/mass availability contradicts its shell census')
    mass_ok=np.abs(mass-rv['shell_mass_msun'])<=request['relative_mass_tolerance']*rv['shell_mass_msun']
    reference_field='native_pair_shell_acceleration_cm_s2' if 'native_pair_support_measurement' in reference else 'shell_acceleration_cm_s2'
    acceleration_difference=force-rv[reference_field];force_ok=np.linalg.norm(acceleration_difference,axis=1)<=request['maximum_shell_acceleration_error_cm_s2']
    output={'shell_particle_count':counts,'native_shell_acceleration_cm_s2':force,'reference_shell_acceleration_cm_s2':rv[reference_field],
        'shell_acceleration_difference_cm_s2':acceleration_difference,'shell_mass_msun':mass,'shell_mass_within_limits':mass_ok,
        'shell_force_within_limits':force_ok,'minimum_support_proper_kpc':lower,'maximum_support_proper_kpc':upper}
    periodic_available=None
    closed_ok=None
    if 'maximum_closed_total_force_error_cm_s2' in request:
        if diagnostic.get('closed_total_comparison_available')is not True or diagnostic.get('closed_total_comparison_scope')!='tracked_gravity_frame_periodic_single_bh_fine_dm_only_no_pm_external_or_hierarchy':
            raise ValueError('Closed total-force comparison requires an actual unchanged captured gravity frame and complete supported native census')
        species=[integer('species count',value) for value in diagnostic['source_species_counts_decimal']]
        if len(species)!=7 or species[1]!=count or species[5]!=1 or any(species[index] for index in (0,2,3,4,6)):
            raise ValueError('Closed native force comparison has unaccounted source species')
        if diagnostic.get('periodic_correction_available')is not True or diagnostic.get('pm_short_range_available')is not False:
            raise ValueError('Closed native force comparison changes its supported periodic/mesh operator')
        monopoles=np.asarray(diagnostic['full_dm_monopoles_cm_s2'],float);correction=np.asarray(diagnostic['full_dm_periodic_correction_cm_s2'],float)
        stored=np.asarray(diagnostic['stored_grav_accel_cm_s2'],float)
        if any(value.shape!=(3,) or not np.isfinite(value).all() for value in (monopoles,correction,stored)):raise ValueError('Complete closed native force vectors required')
        direct=monopoles+correction;residual=stored-direct
        if not np.isfinite(residual).all():raise ArithmeticError('Closed native force composition overflows')
        closed_ok=float(np.linalg.norm(residual))<=request['maximum_closed_total_force_error_cm_s2']
        output.update(closed_direct_periodic_acceleration_cm_s2=direct,closed_stored_gravity_acceleration_cm_s2=stored,
            closed_force_difference_cm_s2=residual,closed_force_within_limits=np.asarray([closed_ok],bool))
    pm_available=None
    if 'pm_short_range_available' in diagnostic:
        pm_available=diagnostic['pm_short_range_available']
        if type(pm_available)is not bool or diagnostic.get('pm_short_range_scope')!='actual_native_cutoff_monopoles; excludes_long_range_mesh_and_tree_approximation':
            raise ValueError('Native PM short-range availability/scope is ambiguous')
        short=np.asarray([row['pm_short_range_cm_s2'] for row in shells],float)
        if short.shape!=force.shape or not np.isfinite(short).all() or not pm_available and np.any(short!=0):raise ValueError('Native PM short-range rows are invalid or unavailable')
        output.update(native_pm_short_range_cm_s2=short,native_pm_short_range_available=np.full(len(shells),pm_available,bool))
    if 'periodic_correction_available' in diagnostic:
        periodic_available=diagnostic['periodic_correction_available']
        if type(periodic_available)is not bool or diagnostic.get('periodic_correction_scope')!='actual_native_ewald_aperture_sources; excludes_pm_tree_and_unselected_matter':
            raise ValueError('Native periodic correction scope/availability is ambiguous')
        correction=np.asarray([row['periodic_correction_cm_s2'] for row in shells],float)
        if correction.shape!=force.shape or not np.isfinite(correction).all() or not periodic_available and np.any(correction!=0):
            raise ValueError('Native periodic correction rows are invalid or relabel unavailable force')
        composed=force+correction
        if not np.isfinite(composed).all():raise ArithmeticError('Native kernel/periodic composition overflows')
        output.update(native_periodic_correction_cm_s2=correction,native_periodic_correction_available=np.full(len(shells),periodic_available,bool),
            native_kernel_plus_periodic_correction_cm_s2=composed if periodic_available else np.zeros_like(composed),
            native_kernel_plus_periodic_correction_available=np.full(len(shells),periodic_available,bool))
    report={'schema':'measured-native-wake-kernel-comparison-v1','execution_status':'COMPLETE','request':request,'request_base':str(Path(parameters['request_base']).resolve()),
        'epoch':epoch,'probe_particle_id_decimal':str(probe['particle_id']),'reference_field':reference_field,
        'status':'WITHIN_DECLARED_KERNEL_LIMITS' if mass_ok.all() and force_ok.all() and (closed_ok is None or closed_ok) else 'NATIVE_KERNEL_UNRESOLVED',
        'scope':'actual native aperture monopole arithmetic versus saved-state reference; not total tree/PM/Ewald force equivalence or unresolved drag acceptance',
        'source_hashes':sources,'science_acceptance':'NOT_ESTABLISHED'}
    if closed_ok is not None:report['closed_total_force_scope']='tracked unchanged force frame; periodic single BH and all fine-DM sources; direct native monopoles/Ewald versus stored tree force under a frozen dimensional limit; no arbitrary hydro/PM or global force accuracy'
    if periodic_available is not None:report['periodic_correction_scope']={'available':periodic_available,
        'definition':'actual native Ewald lookup for selected fine-DM aperture sources only; separately retained from the compared monopole field',
        'total_native_force_equivalence':'NOT_ESTABLISHED'}
    if pm_available is not None:report['pm_short_range_scope']={'available':pm_available,'definition':diagnostic['pm_short_range_scope'],
        'long_range_mesh_and_total_force_equivalence':'NOT_ESTABLISHED'}
    source=Path(__file__).resolve();sources[str(source)]=file_sha256(source)
    specs={name:{'units':'cm / s^2' if name.endswith('cm_s2') else 'Msun' if name.endswith('msun') else 'proper kpc' if name.endswith('kpc') else 'dimensionless','quantity':'native wake kernel '+name} for name in output}
    if periodic_available is not None:
        specs['native_periodic_correction_cm_s2']['missing_policy']='numeric zero when native_periodic_correction_available is false; unavailable is not measured zero Ewald correction'
        specs['native_kernel_plus_periodic_correction_cm_s2']['missing_policy']='numeric zero when native_kernel_plus_periodic_correction_available is false; unavailable composition is not total native force'
    if pm_available is not None:specs['native_pm_short_range_cm_s2']['missing_policy']='numeric zero when native_pm_short_range_available is false; unavailable is not measured zero short-range force'
    return export_numeric_audit(destination,report,output,specs,algorithm=source,source_hashes=sources,context={'epoch':epoch})


def dm_wake_kernel_comparison_runner(parameters,context):
    from pathlib import Path
    if parameters['request']['memory_gib']>context.resources['memory_gib']:raise MemoryError('Kernel comparison exceeds allocated memory')
    destination=Path(parameters['destination']).resolve()
    if not destination.is_relative_to(context.generation):raise ValueError('Kernel comparison escapes its owned generation')
    return publish_dm_wake_kernel_comparison(parameters,destination)


def dm_wake_kernel_comparison_main():
    import argparse,json
    from pathlib import Path
    parser=argparse.ArgumentParser(description='Compare actual native BH aperture-kernel diagnostics with original saved-state force products')
    parser.add_argument('parameters',type=Path);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();path=args.parameters.resolve();parameters=json.loads(path.read_text());parameters.setdefault('request_base',str(path.parent))
    publish_dm_wake_kernel_comparison(parameters,args.output);return 0


def validate_dm_wake_contrast_request(parameters):
    request=parameters['request']
    if not isinstance(request,dict):raise ValueError('Wake contrast request must be an object')
    keys={'schema','resolution_pairs','memory_gib','maximum_resolution_levels','maximum_reference_residual_cm_s2'}
    version=request.get('schema')
    if version=='native-dm-wake-resolution-contrast-request-v1':keys|={'probe_relative_velocity_km_s'}
    elif version=='native-dm-wake-resolution-contrast-request-v2':keys|={'maximum_bath_velocity_mismatch_km_s','minimum_probe_relative_speed_km_s'}
    else:raise ValueError('Unsupported wake contrast request version')
    if 'maximum_native_pair_support_error_cm_s2' in request:keys.add('maximum_native_pair_support_error_cm_s2')
    if set(request)!=keys:
        raise ValueError('Complete matched three-or-more-level wake resolution request required')
    if type(request['maximum_resolution_levels'])is not int or request['maximum_resolution_levels']<3:raise ValueError('At least three bounded actual native resolutions required')
    pairs=request['resolution_pairs']
    if not isinstance(pairs,list) or not 3<=len(pairs)<=request['maximum_resolution_levels']:raise ValueError('Bounded complete perturbed/background resolution pairs required')
    for pair in pairs:
        if not isinstance(pair,dict) or set(pair)!={'perturbed','background'}:raise ValueError('Each resolution needs its measured perturbed and unperturbed wake products')
    finite_number(request['memory_gib'],'wake contrast memory',strict=True)
    finite_number(request['maximum_reference_residual_cm_s2'],'wake reference convergence bound',strict=True)
    if 'maximum_native_pair_support_error_cm_s2' in request:finite_number(request['maximum_native_pair_support_error_cm_s2'],'native pair/reference support budget')
    if request['maximum_resolution_levels']*4095*3*8*12>request['memory_gib']*1024**3/2:
        raise MemoryError('Complete bounded wake resolution vectors and contrast workspace exceed allocation')
    import numpy as np
    if version=='native-dm-wake-resolution-contrast-request-v1':
        velocity=np.asarray(request['probe_relative_velocity_km_s'],float)
        if velocity.shape!=(3,) or not np.isfinite(velocity).all() or np.linalg.norm(velocity)==0:raise ValueError('Finite nonzero declared probe/bath relative velocity required')
    else:
        finite_number(request['maximum_bath_velocity_mismatch_km_s'],'background velocity refinement budget')
        finite_number(request['minimum_probe_relative_speed_km_s'],'minimum measured relative speed',strict=True)
    return request


def dm_wake_contrast_input_bindings(parameters):
    from core.sidm_analysis import _load_product
    request=validate_dm_wake_contrast_request(parameters);sources={}
    for pair in request['resolution_pairs']:
        for reference in pair.values():
            _,_,binding=_load_product(reference,parameters['request_base'],request['memory_gib']*1024**3/4);sources.update(binding)
    return sources


def publish_dm_wake_contrast(parameters,destination):
    """Subtract backgrounds before forming signed fine-minus-coarse forces.

    The two finest levels measure reference stability under the same probe
    operator. Version 2 derives the relative velocity from the actual BH
    and finest background aperture; background/perturbed role labels still
    require physical experiment evidence outside this force measurement.
    """
    import numpy as np
    from pathlib import Path
    from core.sidm_analysis import _load_product
    from core.data import export_numeric_audit
    from core.foundation import file_sha256
    request=validate_dm_wake_contrast_request(parameters);memory=request['memory_gib']*1024**3;forces=[];masses=[];sources={}
    operator=None;epoch=None;physics=None;phase=None;operator_seed=None;counts=[];box=None;total_mass=None;census_pairs=[]
    perturber_definition=None;probe_states=[];bath_states=[];pair_support_errors=[]
    measured_axis=request['schema']=='native-dm-wake-resolution-contrast-request-v2'
    for pair in request['resolution_pairs']:
        products=[];support_errors=[]
        for side in ('perturbed','background'):
            report,arrays,binding=_load_product(pair[side],parameters['request_base'],memory/4);sources.update(binding)
            if report.get('schema')!='measured-native-dm-wake-force-v1' or report.get('execution_status')!='COMPLETE':raise ValueError('Wake contrast needs original complete force measurements')
            if operator is not None and report['operator_definition']!=operator:raise ValueError('Wake contrast changes its common physical probe/aperture/kernel')
            current_epoch={key:report['epoch'].get(key) for key in ('schema','canonical_time','actual_scale_factor','redshift',
                'cosmic_time_gyr','native_time_code','time_unit_gyr','elapsed_time_gyr')}
            if epoch is not None and current_epoch!=epoch:raise ValueError('Wake contrast changes its actual physical epoch')
            if physics is not None and report['native_dm_physics_signature']!=physics:raise ValueError('Wake contrast changes the actual cosmology/DM interaction law')
            if box is not None and report['native_proper_box_kpc']!=box:raise ValueError('Wake refinement changes its actual physical boundary box')
            if total_mass is not None and not np.isclose(report['native_dm_total_mass_msun'],total_mass,rtol=1e-10,atol=0.):raise ValueError('Wake refinement changes the actual complete bath mass')
            if phase is not None and (report['native_phase_seed']!=phase or report['native_operator_seed']!=operator_seed):raise ValueError('Matched wake contrasts change actual phase/operator seed axes')
            operator=report['operator_definition'];epoch=current_epoch;physics=report['native_dm_physics_signature'];phase=report['native_phase_seed'];operator_seed=report['native_operator_seed']
            box=report['native_proper_box_kpc'];total_mass=report['native_dm_total_mass_msun']
            mass_range=report['native_dm_mass_range_msun']
            if mass_range is None or not np.isclose(mass_range[0],mass_range[1],rtol=1e-12,atol=0.):raise ValueError('Homogeneous wake refinement requires one actual DM mass level per resolution')
            force=np.asarray(arrays['shell_acceleration_cm_s2'],float)
            if force.shape!=(len(report['operator_definition']['radial_edges_proper_kpc'])-1,3) or not np.isfinite(force).all():raise ValueError('Complete finite physical force shell vectors required')
            if 'maximum_native_pair_support_error_cm_s2' in request:
                binding=report.get('native_pair_support_measurement')
                actual=np.asarray(arrays.get('native_pair_shell_acceleration_cm_s2',[]),float)
                discrepancy=np.asarray(arrays.get('native_pair_minus_reference_shell_acceleration_cm_s2',[]),float)
                if not isinstance(binding,dict) or binding.get('schema')!='measured-native-pair-softening-support-v1' or actual.shape!=force.shape or discrepancy.shape!=force.shape or not np.isfinite(actual).all() or not np.isfinite(discrepancy).all():
                    raise ValueError('Native pair support budget requires complete version-3 force measurements')
                if not np.array_equal(discrepancy,actual-force):raise ValueError('Native pair support discrepancy changes its original force arrays')
                error=float(np.sum(np.linalg.norm(discrepancy,axis=1)))
                if not math.isfinite(error) or error!=binding['reference_difference_shell_norm_sum_cm_s2']:
                    raise ValueError('Native pair support error disagrees with the saved dimensional bound')
                support_errors.append(error)
            if measured_axis and side=='background':
                available=np.asarray(arrays.get('aperture_velocity_moments_available',[]))
                velocity=np.asarray(arrays.get('aperture_mean_velocity_km_s',[]),float)
                covariance=np.asarray(arrays.get('aperture_velocity_covariance_km2_s2',[]),float)
                bath_mass=np.asarray(arrays.get('aperture_mass_msun',[]),float)
                if available.shape!=(1,) or available.dtype.kind!='b' or not available[0]:raise ValueError('Measured-axis calibration requires a nonempty original background aperture')
                if velocity.shape!=(3,) or covariance.shape!=(3,3) or bath_mass.shape!=(1,) or not np.isfinite(velocity).all() or not np.isfinite(covariance).all() or not np.isfinite(bath_mass).all() or bath_mass[0]<=0:
                    raise ValueError('Complete finite original background aperture moments required')
                if not np.allclose(covariance,covariance.T,rtol=1e-12,atol=0.):raise ValueError('Background velocity covariance is not symmetric')
                if np.linalg.eigvalsh(covariance).min()<-1e-12*max(float(np.max(np.abs(covariance))),1.):raise ValueError('Background velocity covariance is not positive semidefinite')
                edges=np.asarray(report['operator_definition']['radial_edges_proper_kpc'],float)
                bath_states.append({'mean_velocity_km_s':velocity.tolist(),'velocity_covariance_km2_s2':covariance.tolist(),
                    'aperture_mass_msun':float(bath_mass[0]),'aperture_mean_density_msun_kpc3':float(bath_mass[0]/(4*math.pi/3*(edges[-1]**3-edges[0]**3))),
                    'product_sha256':pair[side]['sha256'],'native_dm_census_sha256':report['native_dm_census_sha256']})
            products.append((report,force))
        left,right=products
        state=left[0].get('probe_state',{'kind':'declared_reference_probe'})
        if measured_axis and state['kind']!='original_native_bh_particle':raise ValueError('Measured-axis wake correction requires an actual native BH perturbed probe')
        if state['kind']=='original_native_bh_particle':
            current_perturber={key:state[key] for key in ('kind','particle_id','particle_namespace','physical_bh_mass_msun','gravity_carrier_mass_msun')}
        else:current_perturber={'kind':'declared_reference_probe','native_particle_binding':'NOT_ESTABLISHED'}
        if perturber_definition is not None and current_perturber!=perturber_definition:
            raise ValueError('Wake resolution calibration changes native BH identity/mass or mixes declared and native probes')
        perturber_definition=current_perturber;probe_states.append(state)
        if pair['perturbed']['sha256']==pair['background']['sha256'] or left[0]['native_dm_mass_range_msun']!=right[0]['native_dm_mass_range_msun'] or left[0]['native_dm_particle_count']!=right[0]['native_dm_particle_count']:
            raise ValueError('Within-resolution wake backgrounds must be distinct products of the same actual bath resolution')
        forces.append(left[1]-right[1]);masses.append(left[0]['native_dm_mass_range_msun'][0]);counts.append(left[0]['native_dm_particle_count'])
        census_pairs.append({'perturbed':left[0]['native_dm_census_sha256'],'background':right[0]['native_dm_census_sha256']})
        if support_errors:pair_support_errors.append(sum(support_errors))
    if any(right>=left for left,right in zip(masses,masses[1:])):raise ValueError('Wake resolution levels must strictly decrease the actual DM macro mass')
    values=np.asarray(forces);aperture=values.sum(axis=1);reference_residual=values[-1]-values[-2]
    # Summing vector norms prevents cancellation of opposite shell errors from
    # manufacturing an apparently converged total aperture force.
    residual_bound=float(np.sum(np.linalg.norm(reference_residual,axis=1)))
    pair_support_limit=True
    if pair_support_errors:
        correction_support=np.asarray(pair_support_errors[:-1])+pair_support_errors[-1]
        finest_support=pair_support_errors[-1]+pair_support_errors[-2]
        residual_bound+=finest_support
        pair_support_limit=bool(np.all(correction_support<=request['maximum_native_pair_support_error_cm_s2']))
    if measured_axis:
        bath_velocity=np.asarray([state['mean_velocity_km_s'] for state in bath_states],float)
        mismatch=np.linalg.norm(bath_velocity-bath_velocity[-1],axis=1)
        if np.any(mismatch>request['maximum_bath_velocity_mismatch_km_s']):raise ValueError('Measured background velocity changes beyond its frozen resolution budget')
        velocity=np.asarray(probe_states[-1]['velocity_peculiar_km_s'],float)-bath_velocity[-1]
        if np.linalg.norm(velocity)<request['minimum_probe_relative_speed_km_s']:raise ValueError('Measured native BH/background relative speed is below the supported floor')
        axis_definition='native BH peculiar velocity minus finest measured background aperture mass-weighted mean; no background-role proof inferred'
    else:
        velocity=np.asarray(request['probe_relative_velocity_km_s'],float)
        axis_definition='declared peculiar probe/background relative velocity'
    speed=float(np.linalg.norm(velocity))
    if not np.isfinite(velocity).all() or not math.isfinite(speed) or speed<=0:raise ValueError('Finite representable nonzero wake projection speed required')
    correction=values[-1][None]-values[:-1];axis=velocity/speed
    projected=np.einsum('lbi,i->lb',correction,axis);transverse=correction-projected[:,:,None]*axis
    arrays={'wake_shell_acceleration_cm_s2':values,'wake_aperture_acceleration_cm_s2':aperture,'reference_shell_residual_cm_s2':reference_residual,
        'unresolved_shell_force_contrast_cm_s2':correction,'unresolved_aperture_force_contrast_cm_s2':correction.sum(axis=1),
        'signed_parallel_force_contrast_cm_s2':projected,'transverse_force_contrast_cm_s2':transverse,'dm_macro_mass_msun':np.asarray(masses)}
    if pair_support_errors:
        arrays['native_pair_reference_correction_error_bound_cm_s2']=correction_support
        arrays['native_pair_reference_finest_error_bound_cm_s2']=np.asarray([finest_support])
        residual_bound+=float(np.max(correction_support))
    if any(not np.isfinite(value).all() for value in arrays.values()):raise ArithmeticError('Wake resolution contrast overflows')
    source=Path(__file__).resolve();sources[str(source)]=file_sha256(source)
    report={'schema':'measured-native-dm-wake-resolution-contrast-v1','execution_status':'COMPLETE','request':request,'epoch':epoch,
        'request_base':str(Path(parameters['request_base']).resolve()),
        'native_dm_physics_signature':physics,'native_phase_seed':phase,'native_operator_seed':operator_seed,'operator_definition':operator,
        'native_dm_particle_counts':counts,'reference_residual_shell_norm_sum_cm_s2':residual_bound,
        'native_dm_census_pairs':census_pairs,
        'perturber_definition':perturber_definition,'resolution_perturbed_probe_states':probe_states,
        'probe_relative_velocity_km_s':velocity.tolist(),'relative_velocity_definition':axis_definition,
        'resolution_background_bath_states':bath_states,
        'reference_status':'WITHIN_DECLARED_FORCE_LIMIT' if residual_bound<=request['maximum_reference_residual_cm_s2'] and pair_support_limit else 'UNRESOLVED_REFERENCE',
        'sign_definition':'negative parallel acceleration is deceleration along the recorded relative velocity; negative/positive corrections are preserved',
        'scope':'single-realization matched background-subtracted reference-operator contrasts; not an executable calibrated drag law, independent ensemble result or actual native-force support equivalence',
        'science_acceptance':'NOT_ESTABLISHED','source_hashes':sources}
    if pair_support_errors:
        report['native_pair_support_budget']={'status':'WITHIN_DECLARED_PAIR_SUPPORT_LIMIT' if pair_support_limit else 'REFERENCE_SUPPORT_MISMATCH',
            'definition':'shell-norm triangle bounds through background and finest-coarse subtraction; calibration reference bound includes pair mismatch',
            'scope':'saved native pair softening only; tree/PM/Ewald and force-domain equivalence remain unestablished'}
    return export_numeric_audit(destination,report,arrays,{name:{'units':'Msun' if name=='dm_macro_mass_msun' else 'cm / s^2','quantity':'measured matched wake '+name} for name in arrays},
        algorithm=source,source_hashes=sources,context={'epoch':epoch,'coordinates':{'frame':'common inertial physical probe','relative_velocity_definition':axis_definition}})


def validate_dm_wake_ensemble_request(parameters):
    request=parameters['request']
    keys={'schema','contrasts','minimum_realizations','maximum_realizations','maximum_observables','coarse_resolution_index',
        'memory_gib','mean_error_multiplier','maximum_reference_residual_cm_s2','maximum_transverse_fraction','minimum_parallel_deceleration_cm_s2'}
    if not isinstance(request,dict):raise ValueError('Wake ensemble request must be an object')
    version=request.get('schema')
    if version=='native-dm-wake-ensemble-request-v2':
        keys|={'maximum_velocity_deviation_km_s','maximum_density_relative_deviation','maximum_covariance_relative_deviation',
            'density_floor_msun_kpc3','covariance_floor_km2_s2'}
        for name in ('maximum_velocity_deviation_km_s','maximum_density_relative_deviation','maximum_covariance_relative_deviation'):
            finite_number(request.get(name),name)
        for name in ('density_floor_msun_kpc3','covariance_floor_km2_s2'):finite_number(request.get(name),name,strict=True)
    elif version!='native-dm-wake-ensemble-request-v1':raise ValueError('Unsupported wake ensemble request version')
    if set(request)!=keys:
        raise ValueError('Complete independent-realization matched wake ensemble request required')
    for key in ('minimum_realizations','maximum_realizations','maximum_observables','coarse_resolution_index'):
        if type(request[key])is not int or request[key]<(0 if key=='coarse_resolution_index' else 1):raise ValueError('Exact positive wake ensemble work bounds required')
    if request['minimum_realizations']<2 or not isinstance(request['contrasts'],list) or not request['minimum_realizations']<=len(request['contrasts'])<=request['maximum_realizations']:
        raise ValueError('At least two bounded independent measured wake realizations required')
    for key in ('memory_gib','maximum_reference_residual_cm_s2'):finite_number(request[key],key,strict=True)
    for key in ('mean_error_multiplier','maximum_transverse_fraction','minimum_parallel_deceleration_cm_s2'):finite_number(request[key],key)
    if request['maximum_transverse_fraction']>=1:raise ValueError('A scalar drag response requires a declared transverse fraction below one')
    bound=request['maximum_observables']**2*64+request['maximum_realizations']*(request['maximum_observables']*128+2048)+8192
    if bound>request['memory_gib']*1024**3:raise MemoryError('Declared complete wake covariance and paired vectors exceed allocation')
    return request


def dm_wake_ensemble_input_bindings(parameters):
    from core.sidm_analysis import _load_product
    request=validate_dm_wake_ensemble_request(parameters);sources={}
    for reference in request['contrasts']:
        _,_,binding=_load_product(reference,parameters['request_base'],request['memory_gib']*1024**3/4);sources.update(binding)
    return sources


def publish_dm_wake_ensemble(parameters,destination):
    """Measure independent paired wake means and full shell covariance.

    A mean scalar-drag support decision uses explicitly frozen operational
    sampling/reference/transverse budgets, not an asserted confidence level.
    All signed observations and failed reference realizations are retained.
    """
    import numpy as np
    from pathlib import Path
    from core.sidm_analysis import _load_product
    from core.data import export_numeric_audit
    from core.foundation import canonical_digest,file_sha256
    request=validate_dm_wake_ensemble_request(parameters);sources={};vectors=[];reference_residuals=[];seed_records=[];state_measurements=[];pair_support_decisions=[]
    phases=set();operators=set();censuses=set();definition=None;layout=None;prototype=None
    varying_state=request['schema']=='native-dm-wake-ensemble-request-v2';velocities=[];densities=[];bath_covariances=[]
    for reference in request['contrasts']:
        report,arrays,binding=_load_product(reference,parameters['request_base'],request['memory_gib']*1024**3/4);sources.update(binding)
        if report.get('schema')!='measured-native-dm-wake-resolution-contrast-v1' or report.get('execution_status')!='COMPLETE':
            raise ValueError('Wake ensemble requires complete actual matched resolution contrasts')
        phase=report['native_phase_seed'];operator=report['native_operator_seed']
        if isinstance(phase,bool) or not isinstance(phase,int) or phase<0 or phase in phases:
            raise ValueError('Independent wake realizations require distinct actual native phase seeds')
        if operator is not None:
            if isinstance(operator,bool) or not isinstance(operator,int) or not 0<=operator<2**64 or operator in operators:
                raise ValueError('Independent SIDM wake realizations require distinct actual uint64 operator seeds')
            operators.add(operator)
        phases.add(phase);seed_records.append({'phase_seed':phase,'operator_seed':operator})
        state_measurements.append({'contrast_product_sha256':reference['sha256'],'phase_seed':phase,'operator_seed':operator,
            'resolution_perturbed_probe_states':report.get('resolution_perturbed_probe_states',[]),
            'resolution_background_bath_states':report.get('resolution_background_bath_states',[])})
        census=canonical_digest(report['native_dm_census_pairs'])
        if census in censuses:raise ValueError('Independent wake realizations reuse the exact original DM census pairs')
        censuses.add(census)
        values=np.asarray(arrays['unresolved_shell_force_contrast_cm_s2'],float);index=request['coarse_resolution_index']
        if values.ndim!=3 or values.shape[-1]!=3 or index>=len(values) or not np.isfinite(values).all():raise ValueError('Selected original coarse wake correction is unavailable')
        selected=values[index];dimension=selected.size
        if dimension>request['maximum_observables']:raise MemoryError('Actual shell force vector exceeds covariance ceiling')
        current={'operator':report['operator_definition'],'epoch':report['epoch'],'physics':report['native_dm_physics_signature'],
            'macro_masses':np.asarray(arrays['dm_macro_mass_msun']).tolist(),'counts':report['native_dm_particle_counts'],
            'perturber':report.get('perturber_definition',{'kind':'declared_reference_probe','native_particle_binding':'NOT_ESTABLISHED'}),
            'relative_velocity':report.get('probe_relative_velocity_km_s',report['request'].get('probe_relative_velocity_km_s')),
            'relative_velocity_definition':report.get('relative_velocity_definition','declared peculiar probe/background relative velocity'),
            'coarse_resolution_index':index}
        if 'native_pair_support_budget' in report:
            current['native_pair_support_contract']={'maximum_error_cm_s2':report['request']['maximum_native_pair_support_error_cm_s2'],
                'scope':report['native_pair_support_budget']['scope']}
        actual_velocity=np.asarray(current['relative_velocity'],float)
        if actual_velocity.shape!=(3,) or not np.isfinite(actual_velocity).all() or not 0<float(np.linalg.norm(actual_velocity))<math.inf:
            raise ValueError('Wake ensemble requires an original finite nonzero projection velocity')
        velocities.append(actual_velocity)
        signature_definition=dict(current)
        if varying_state:
            if report['request']['schema']!='native-dm-wake-resolution-contrast-request-v2' or current['perturber']['kind']!='original_native_bh_particle':
                raise ValueError('State-varying ensemble requires actual native BH and measured-background contrasts')
            bath=report['resolution_background_bath_states'][-1]
            rho=float(bath['aperture_mean_density_msun_kpc3']);bath_covariance=np.asarray(bath['velocity_covariance_km2_s2'],float)
            if not math.isfinite(rho) or rho<=0 or bath_covariance.shape!=(3,3) or not np.isfinite(bath_covariance).all():
                raise ValueError('Original complete finest bath density and covariance required')
            densities.append(rho);bath_covariances.append(bath_covariance)
            signature_definition.pop('relative_velocity')
            signature_definition['state_variation_contract']={name:request[name] for name in (
                'maximum_velocity_deviation_km_s','maximum_density_relative_deviation','maximum_covariance_relative_deviation',
                'density_floor_msun_kpc3','covariance_floor_km2_s2')}
        signature=canonical_digest(signature_definition)
        if definition is not None and signature!=definition:raise ValueError('Wake ensemble changes actual physics, probe, epoch, resolutions or physical axis')
        if layout is not None and selected.shape!=layout:raise ValueError('Wake ensemble changes complete shell coordinates')
        definition=signature;layout=selected.shape;prototype=current
        residual=np.asarray(arrays['reference_shell_residual_cm_s2'],float)
        actual_reference_error=float(np.sum(np.linalg.norm(residual,axis=1)))
        if 'native_pair_support_budget' in report:
            correction_bounds=np.asarray(arrays['native_pair_reference_correction_error_bound_cm_s2'],float)
            finest_bound=np.asarray(arrays['native_pair_reference_finest_error_bound_cm_s2'],float)
            if correction_bounds.shape!=(len(values),) or finest_bound.shape!=(1,) or not np.isfinite(correction_bounds).all() or not np.isfinite(finest_bound).all() or np.any(correction_bounds<0) or finest_bound[0]<0:
                raise ValueError('Wake ensemble lacks complete native/reference pair support uncertainty arrays')
            actual_reference_error+=float(finest_bound[0])
            actual_reference_error+=float(np.max(correction_bounds))
            support_ok=bool(np.all(correction_bounds<=report['request']['maximum_native_pair_support_error_cm_s2']))
            declared=report['native_pair_support_budget']['status']=='WITHIN_DECLARED_PAIR_SUPPORT_LIMIT'
            if support_ok!=declared:raise ValueError('Native pair support status changes its original force bounds')
            pair_support_decisions.append(support_ok)
        if actual_reference_error!=report['reference_residual_shell_norm_sum_cm_s2']:raise ValueError('Wake reference status changes its actual finest shell residuals')
        vectors.append(selected.ravel());reference_residuals.append(actual_reference_error)
    if operators and len(operators)!=len(phases):raise ValueError('Wake ensemble mixes collisionless and active SIDM seed definitions')
    state_decisions={};state_output={};state_definition=None
    if varying_state:
        velocities=np.asarray(velocities);densities=np.asarray(densities);bath_covariances=np.asarray(bath_covariances)
        mean_velocity=velocities.mean(axis=0);mean_density=float(densities.mean());mean_bath_covariance=bath_covariances.mean(axis=0)
        identical_states=bool(np.all(velocities==velocities[0]) and np.all(densities==densities[0]) and np.all(bath_covariances==bath_covariances[0]))
        if identical_states:
            mean_velocity=velocities[0].copy();mean_density=float(densities[0]);mean_bath_covariance=bath_covariances[0].copy()
        velocity_deviation=np.linalg.norm(velocities-mean_velocity,axis=1)
        density_deviation=np.abs(densities-mean_density)/max(mean_density,request['density_floor_msun_kpc3'])
        covariance_deviation=np.linalg.norm((bath_covariances-mean_bath_covariance).reshape(len(velocities),-1),axis=1)/max(
            float(np.linalg.norm(mean_bath_covariance)),request['covariance_floor_km2_s2'])
        mean_speed=float(np.linalg.norm(mean_velocity))
        if not math.isfinite(mean_speed) or mean_speed<=0:raise ValueError('Measured ensemble mean relative velocity has no representable physical direction')
        state_decisions={'velocity_state_variation':bool(np.all(velocity_deviation<=request['maximum_velocity_deviation_km_s'])),
            'density_state_variation':bool(np.all(density_deviation<=request['maximum_density_relative_deviation'])),
            'covariance_state_variation':bool(np.all(covariance_deviation<=request['maximum_covariance_relative_deviation']))}
        prototype=dict(prototype);prototype['relative_velocity']=mean_velocity.tolist()
        prototype['state_variation_contract']=signature_definition['state_variation_contract']
        definition=canonical_digest(prototype)
        state_output={'realization_relative_velocity_km_s':velocities,'realization_bath_density_msun_kpc3':densities,
            'realization_bath_velocity_covariance_km2_s2':bath_covariances,'relative_velocity_deviation_km_s':velocity_deviation,
            'bath_density_relative_deviation':density_deviation,'bath_covariance_relative_deviation':covariance_deviation}
        state_definition={'reference':'equal-realization mean of actual finest background states; not an unmeasured homogeneous bath',
            'mean_relative_velocity_km_s':mean_velocity.tolist(),'mean_bath_density_msun_kpc3':mean_density,
            'mean_bath_velocity_covariance_km2_s2':mean_bath_covariance.tolist(),
            'speed_only_interpolant_compatibility':'IDENTICAL_MEASURED_STATES' if identical_states else 'REQUIRES_STATE_CLOUD_CALIBRATION'}
    values=np.asarray(vectors);count=len(values);mean=values.mean(axis=0);centered=values-mean
    covariance=centered.T@centered/(count-1);mean_covariance=covariance/count
    standard_error=np.sqrt(np.maximum(np.diag(mean_covariance),0.))
    shell_mean=mean.reshape(layout);aperture_values=values.reshape((count,*layout)).sum(axis=1)
    aperture_mean=aperture_values.mean(axis=0);aperture_centered=aperture_values-aperture_mean
    aperture_covariance=aperture_centered.T@aperture_centered/(count-1);aperture_mean_covariance=aperture_covariance/count
    velocity=np.asarray(prototype['relative_velocity'],float);axis=velocity/np.linalg.norm(velocity);projection=np.eye(3)-np.outer(axis,axis)
    parallel=float(aperture_mean@axis);parallel_error=math.sqrt(max(float(axis@aperture_mean_covariance@axis),0.))
    transverse=projection@aperture_mean;transverse_error=math.sqrt(max(float(np.trace(projection@aperture_mean_covariance@projection)),0.))
    reference_error=max(reference_residuals);multiplier=request['mean_error_multiplier']
    deceleration_lower=-parallel-multiplier*parallel_error-reference_error
    transverse_upper=float(np.linalg.norm(transverse))+multiplier*transverse_error+reference_error
    decisions={'reference_resolution':reference_error<=request['maximum_reference_residual_cm_s2'],
        'positive_resolved_deceleration':deceleration_lower>=request['minimum_parallel_deceleration_cm_s2'],
        'scalar_drag_direction':deceleration_lower>=0 and transverse_upper<=request['maximum_transverse_fraction']*deceleration_lower}
    if pair_support_decisions:decisions['native_pair_support']=bool(all(pair_support_decisions))
    decisions.update(state_decisions)
    output={'realization_shell_force_contrasts_cm_s2':values.reshape((count,*layout)),'mean_shell_force_contrast_cm_s2':shell_mean,
        'shell_force_sample_covariance':covariance,'shell_force_mean_standard_error_cm_s2':standard_error.reshape(layout),
        'realization_aperture_force_contrasts_cm_s2':aperture_values,'mean_aperture_force_contrast_cm_s2':aperture_mean,
        'aperture_force_sample_covariance':aperture_covariance,'reference_resolution_residual_cm_s2':np.asarray(reference_residuals),
        'parallel_mean_force_contrast_cm_s2':np.asarray([parallel]),'parallel_mean_standard_error_cm_s2':np.asarray([parallel_error]),
        'transverse_mean_force_contrast_cm_s2':transverse,'transverse_mean_standard_error_norm_cm_s2':np.asarray([transverse_error]),
        'operational_deceleration_lower_cm_s2':np.asarray([deceleration_lower]),'operational_transverse_upper_cm_s2':np.asarray([transverse_upper])}
    output.update(state_output)
    if varying_state:
        joint=np.concatenate((aperture_values,velocities),axis=1);joint_centered=joint-joint.mean(axis=0)
        output['force_velocity_joint_sample_covariance']=joint_centered.T@joint_centered/(count-1)
        output['force_velocity_joint_sample_mean']=joint.mean(axis=0)
    if any(not np.isfinite(value).all() for value in output.values()):raise ArithmeticError('Wake ensemble moment accumulation overflows')
    source=Path(__file__).resolve();sources[str(source)]=file_sha256(source)
    report={'schema':'measured-native-dm-wake-ensemble-v1','execution_status':'COMPLETE','request':request,'request_base':str(Path(parameters['request_base']).resolve()),
        'fixed_wake_definition':prototype,'fixed_wake_signature':definition,'ordered_realization_seeds':seed_records,'realization_count':count,
        'ordered_native_state_measurements':state_measurements,
        'measured_state_cloud':state_definition,
        'decisions':decisions,'status':'SCALAR_DRAG_MEAN_BUDGETS_SUPPORTED' if all(decisions.values()) else 'UNRESOLVED_DRAG_MEAN',
        'statistical_definition':'unbiased paired-realization covariance; mean standard error assumes independent phase/operator realizations; multipliers are operational budgets, not coverage probabilities',
        'scope':'measured single probe/operator/epoch/resolution ensemble; declared background roles and relative velocity remain assumptions; no actual native-force support equivalence or calibrated executable drag law',
        'science_acceptance':'NOT_ESTABLISHED','source_hashes':sources}
    state_units={'realization_relative_velocity_km_s':'km / s','relative_velocity_deviation_km_s':'km / s',
        'realization_bath_density_msun_kpc3':'Msun / kpc^3','realization_bath_velocity_covariance_km2_s2':'(km / s)^2',
        'bath_density_relative_deviation':'dimensionless','bath_covariance_relative_deviation':'dimensionless',
        'force_velocity_joint_sample_covariance':'axis-dependent products','force_velocity_joint_sample_mean':'axis-dependent'}
    metadata={name:{'units':state_units.get(name,'(cm / s^2)^2' if 'covariance' in name else 'cm / s^2'),
        'quantity':'measured independent wake '+name} for name in output}
    for name in ('force_velocity_joint_sample_covariance','force_velocity_joint_sample_mean'):
        if name in metadata:
            metadata[name]['axis_quantities']=['aperture force x','aperture force y','aperture force z','relative velocity x','relative velocity y','relative velocity z']
            metadata[name]['axis_units']=['cm / s^2']*3+['km / s']*3
    return export_numeric_audit(destination,report,output,metadata,
        algorithm=source,source_hashes=sources,context={'epoch':prototype['epoch'],'coordinates':{'frame':'common physical probe, aligned complete radial shell vector coordinates'}})


def validate_dm_wake_response_request(parameters):
    request=parameters['request']
    keys={'schema','training','heldout','maximum_points','memory_gib','maximum_absolute_prediction_error_cm_s2',
        'maximum_relative_prediction_error','prediction_error_floor_cm_s2','mean_error_multiplier'}
    if not isinstance(request,dict) or set(request)!=keys or request['schema']!='native-dm-wake-speed-response-request-v1':
        raise ValueError('Complete measured speed-response training and held-out calibration request required')
    if type(request['maximum_points'])is not int or request['maximum_points']<3:raise ValueError('At least three bounded measured speed knots required')
    for name,minimum in (('training',3),('heldout',1)):
        if not isinstance(request[name],list) or not minimum<=len(request[name])<=request['maximum_points']:raise ValueError('Bounded measured '+name+' ensembles required')
    for name in ('memory_gib','prediction_error_floor_cm_s2'):finite_number(request[name],name,strict=True)
    for name in ('maximum_absolute_prediction_error_cm_s2','maximum_relative_prediction_error','mean_error_multiplier'):finite_number(request[name],name)
    return request


def dm_wake_response_input_bindings(parameters):
    from core.sidm_analysis import _load_product
    request=validate_dm_wake_response_request(parameters);sources={}
    for reference in [*request['training'],*request['heldout']]:
        _,_,binding=_load_product(reference,parameters['request_base'],request['memory_gib']*1024**3/4);sources.update(binding)
    return sources


def publish_dm_wake_response(parameters,destination):
    """Calibrate a local, fixed-physics reference-operator speed interpolant.

    Every training interval needs a disjoint held-out interior speed. Positive
    knot amplitudes are interpolated linearly; no asymptote or extrapolation is
    invented. This product alone cannot enable a general native BH drag law.
    """
    import copy
    import numpy as np
    from pathlib import Path
    from core.sidm_analysis import _load_product
    from core.data import export_numeric_audit
    from core.foundation import canonical_digest,file_sha256
    request=validate_dm_wake_response_request(parameters);sources={};samples={'training':[],'heldout':[]};definition=None;prototype=None
    state_measurements={'training':[],'heldout':[]}
    seeds={'training':set(),'heldout':set()};phases={'training':set(),'heldout':set()};operators={'training':set(),'heldout':set()};axis=None
    for role in ('training','heldout'):
        for reference in request[role]:
            report,arrays,binding=_load_product(reference,parameters['request_base'],request['memory_gib']*1024**3/4);sources.update(binding)
            if report.get('schema')!='measured-native-dm-wake-ensemble-v1' or report.get('status')!='SCALAR_DRAG_MEAN_BUDGETS_SUPPORTED':
                raise ValueError('Speed response requires actually supported measured wake ensembles')
            cloud=report.get('measured_state_cloud')
            if cloud is not None and cloud['speed_only_interpolant_compatibility']!='IDENTICAL_MEASURED_STATES':
                raise ValueError('A measured varying-state cloud requires a state-cloud response producer; it cannot be substituted for a fixed-state speed knot')
            frozen=copy.deepcopy(report['fixed_wake_definition']);velocity=np.asarray(frozen.pop('relative_velocity'),float);speed=float(np.linalg.norm(velocity))
            if cloud is not None:
                frozen['measured_bath_definition']={key:cloud[key] for key in ('mean_bath_density_msun_kpc3','mean_bath_velocity_covariance_km2_s2')}
            direction=velocity/speed
            if axis is not None and not np.allclose(direction,axis,rtol=0,atol=1e-12):raise ValueError('Fixed-geometry speed response changes its physical velocity direction')
            axis=direction;frozen['operator'].pop('probe_velocity_peculiar_km_s')
            signature=canonical_digest(frozen)
            if definition is not None and signature!=definition:raise ValueError('Speed response changes unrelated physical probe, epoch, model or native resolution')
            definition=signature;prototype=frozen
            amplitude=-float(np.asarray(arrays['parallel_mean_force_contrast_cm_s2'])[0]);error=float(np.asarray(arrays['parallel_mean_standard_error_cm_s2'])[0])
            reference_error=float(np.max(arrays['reference_resolution_residual_cm_s2']))
            if not np.isfinite([speed,amplitude,error,reference_error]).all() or speed<=0 or min(amplitude,error,reference_error)<0:
                raise ValueError('Finite supported positive-speed deceleration measurements required')
            for item in report['ordered_realization_seeds']:
                seed=(item['phase_seed'],item['operator_seed']);seeds[role].add(seed);phases[role].add(seed[0])
                if seed[1]is not None:operators[role].add(seed[1])
            samples[role].append((speed,amplitude,error,reference_error))
            state_measurements[role].append({'ensemble_product_sha256':reference['sha256'],'measured_speed_km_s':speed,
                'relative_velocity_definition':report['fixed_wake_definition'].get('relative_velocity_definition','declared peculiar probe/background relative velocity'),
                'ordered_native_state_measurements':report.get('ordered_native_state_measurements',[])})
    if phases['training']&phases['heldout'] or operators['training']&operators['heldout']:
        raise ValueError('Held-out response calibration reuses actual training phase/operator seed axes')
    training=np.asarray(sorted(samples['training']));heldout=np.asarray(sorted(samples['heldout']));speeds=training[:,0]
    if np.any(np.diff(speeds)<=0) or np.any(np.diff(heldout[:,0])<=0):raise ValueError('Response speeds must be distinct actual measured points')
    covered=np.zeros(len(speeds)-1,bool);residuals=[];allowances=[];intervals=[];passed=[]
    for speed,amplitude,error,reference_error in heldout:
        interval=int(np.searchsorted(speeds,speed,side='right')-1)
        if not 0<=interval<len(speeds)-1 or speed in speeds:raise ValueError('Held-out calibration points must lie strictly inside measured training intervals')
        left,right=training[interval:interval+2];weight=(speed-left[0])/(right[0]-left[0])
        prediction=(1-weight)*left[1]+weight*right[1]
        # Training knots may share realization seeds. A linear sum of their
        # standard errors is conservative without assuming knot independence.
        prediction_error=(1-weight)*left[2]+weight*right[2]
        systematic=reference_error+(1-weight)*left[3]+weight*right[3]
        allowed=request['maximum_absolute_prediction_error_cm_s2']+request['maximum_relative_prediction_error']*max(abs(amplitude),abs(prediction),request['prediction_error_floor_cm_s2'])
        allowed+=request['mean_error_multiplier']*(error+prediction_error)+systematic
        residuals.append(amplitude-prediction);allowances.append(allowed);intervals.append(interval);passed.append(abs(amplitude-prediction)<=allowed);covered[interval]=True
    supported=covered.copy()
    for interval,okay in zip(intervals,passed):supported[interval]&=okay
    source=Path(__file__).resolve();sources[str(source)]=file_sha256(source)
    output={'speed_knots_km_s':speeds,'deceleration_knots_cm_s2':training[:,1],'knot_mean_standard_error_cm_s2':training[:,2],
        'knot_reference_error_cm_s2':training[:,3],'heldout_speeds_km_s':heldout[:,0],'heldout_prediction_residual_cm_s2':np.asarray(residuals),
        'heldout_allowed_prediction_error_cm_s2':np.asarray(allowances),'heldout_interval_index':np.asarray(intervals,np.int64),
        'heldout_within_budget':np.asarray(passed,bool),'interval_calibration_supported':supported}
    if any(not np.isfinite(value).all() for value in output.values()):raise ArithmeticError('Wake speed calibration overflows')
    report={'schema':'calibrated-native-wake-speed-response-v1','execution_status':'COMPLETE','request':request,'request_base':str(Path(parameters['request_base']).resolve()),
        'fixed_response_definition':prototype,'fixed_response_signature':definition,'velocity_direction':axis.tolist(),
        'training_native_seeds':[list(value) for value in sorted(seeds['training'],key=repr)],'heldout_native_seeds':[list(value) for value in sorted(seeds['heldout'],key=repr)],
        'native_state_measurements':state_measurements,
        'status':'ALL_MEASURED_INTERVALS_SUPPORTED' if supported.all() else 'PARTIALLY_SUPPORTED_OR_UNRESOLVED',
        'interpolation':'piecewise linear physical speed/deceleration; unsupported intervals and extrapolation rejected',
        'scope':'fixed physical reference-operator wake response on measured/held-out speed intervals; declared bath/perturbation assumptions remain; no general native friction, arbitrary BH mass/density/collisionality scaling or global coverage claim',
        'source_hashes':sources,'science_acceptance':'NOT_ESTABLISHED'}
    return export_numeric_audit(destination,report,output,{name:{'units':'km / s' if 'speed' in name else 'dimensionless' if name.endswith(('index','supported','budget')) else 'cm / s^2','quantity':'calibrated measured wake '+name} for name in output},
        algorithm=source,source_hashes=sources,context={'epoch':prototype['epoch'],'coordinates':{'frame':'fixed physical probe and direction; measured peculiar speed'}})


def evaluate_dm_wake_response(reference,relative_velocity_km_s,*,request_base,fixed_response_signature,memory_gib):
    import numpy as np
    from core.sidm_analysis import _load_product
    report,arrays,sources=_load_product(reference,request_base,finite_number(memory_gib,'response evaluation memory',strict=True)*1024**3)
    if report.get('schema')!='calibrated-native-wake-speed-response-v1' or report['fixed_response_signature']!=fixed_response_signature:
        raise ValueError('Measured wake response changes the explicitly frozen physical state/operator')
    velocity=np.asarray(relative_velocity_km_s,float);speed=float(np.linalg.norm(velocity))
    if velocity.shape!=(3,) or not np.isfinite(velocity).all() or speed<=0 or not np.allclose(velocity/speed,report['velocity_direction'],rtol=0,atol=1e-12):
        raise ValueError('Wake response requires its measured physical speed direction')
    knots=arrays['speed_knots_km_s']
    if speed<knots[0] or speed>knots[-1]:raise ValueError('Wake response cannot extrapolate beyond measured speeds')
    interval=min(int(np.searchsorted(knots,speed,side='right')-1),len(knots)-2)
    if not bool(arrays['interval_calibration_supported'][interval]):raise ValueError('Wake response interval lacks held-out calibration support')
    amplitude=float(np.interp(speed,knots,arrays['deceleration_knots_cm_s2']))
    return -amplitude*velocity/speed,{'interval':interval,'sources':sources,'scope':report['scope']}


WAKE_STATE_COORDINATES=('relative_velocity_x_km_s','relative_velocity_y_km_s','relative_velocity_z_km_s',
    'bath_density_msun_kpc3','bath_covariance_xx_km2_s2','bath_covariance_yy_km2_s2','bath_covariance_zz_km2_s2',
    'bath_covariance_xy_km2_s2','bath_covariance_xz_km2_s2','bath_covariance_yz_km2_s2')


def _wake_state_coordinates(velocity,density,covariance):
    import numpy as np
    velocity=np.asarray(velocity,float);covariance=np.asarray(covariance,float)
    density=finite_number(density,'measured bath density',strict=True)
    if velocity.shape!=(3,) or covariance.shape!=(3,3) or not np.isfinite(velocity).all() or not np.isfinite(covariance).all():
        raise ValueError('Complete physical velocity and bath covariance required')
    if not 0<float(np.linalg.norm(velocity))<math.inf:raise ValueError('Representable nonzero bath-relative speed required')
    if not np.allclose(covariance,covariance.T,rtol=1e-12,atol=0.) or np.linalg.eigvalsh(covariance).min()<-1e-12*max(1.,float(np.max(np.abs(covariance)))):
        raise ValueError('Symmetric positive semidefinite physical bath covariance required')
    return np.r_[velocity,density,np.diag(covariance),covariance[0,1],covariance[0,2],covariance[1,2]]


def validate_dm_wake_state_response_request(parameters):
    import numpy as np
    request=parameters['request']
    keys={'schema','training','heldout','maximum_products','maximum_realizations','memory_gib','active_coordinates',
        'coordinate_scales','minimum_design_singular_ratio','convex_weight_residual_tolerance',
        'maximum_absolute_prediction_error_cm_s2','maximum_relative_prediction_error','prediction_error_floor_cm_s2'}
    if not isinstance(request,dict) or set(request)!=keys or request['schema']!='native-wake-state-response-request-v1':
        raise ValueError('Complete native state-cloud response request required')
    for name in ('maximum_products','maximum_realizations'):
        if type(request[name])is not int or not 1<=request[name]<=4096:raise ValueError('Exact bounded state response work required')
    for role in ('training','heldout'):
        if not isinstance(request[role],list) or not 1<=len(request[role])<=request['maximum_products']:raise ValueError('Explicit bounded training and held-out products required')
    active=request['active_coordinates']
    if not isinstance(active,list) or not active or len(set(active))!=len(active) or any(name not in WAKE_STATE_COORDINATES for name in active):
        raise ValueError('Distinct declared physical response coordinates required')
    scales=np.asarray(request['coordinate_scales'],float)
    if scales.shape!=(10,) or not np.isfinite(scales).all() or np.any(scales<=0):raise ValueError('Ten positive dimensional coordinate scales required')
    for name in ('memory_gib','minimum_design_singular_ratio','convex_weight_residual_tolerance','prediction_error_floor_cm_s2'):finite_number(request[name],name,strict=True)
    if request['minimum_design_singular_ratio']>=1 or request['convex_weight_residual_tolerance']>1e-6:raise ValueError('Conditioning and convex-weight residual budgets are outside supported range')
    for name in ('maximum_absolute_prediction_error_cm_s2','maximum_relative_prediction_error'):finite_number(request[name],name)
    if request['maximum_realizations']**2*256+request['maximum_realizations']*4096>request['memory_gib']*1024**3/2:
        raise MemoryError('Bounded fit and convex-weight solver workspace exceeds allocation')
    return request


def dm_wake_state_response_input_bindings(parameters):
    from core.sidm_analysis import _load_product
    request=validate_dm_wake_state_response_request(parameters);sources={}
    for role in ('training','heldout'):
        for reference in request[role]:sources.update(_load_product(reference,parameters['request_base'],request['memory_gib']*1024**3/4)[2])
    return sources


def _wake_convex_weights(training,query,tolerance):
    """A query must be a measured-state convex combination in every coordinate.

    Use all coordinates, including those not fitted, to prevent unconstrained
    variation along omitted physics axes. No triangulation, jitter or box-only
    extrapolation fallback is allowed.
    """
    import numpy as np
    from scipy.optimize import linprog
    target=np.r_[1.,query];matrix=np.vstack((np.ones(len(training)),training.T))
    result=linprog(np.zeros(len(training)),A_eq=matrix,b_eq=target,bounds=(0.,1.),method='highs',
        options={'primal_feasibility_tolerance':max(1e-10,min(tolerance,1e-7))})
    if not result.success or result.x is None:raise ValueError('State lies outside supported measured convex domain')
    weights=result.x
    if not np.isfinite(weights).all() or np.any(weights<0) or np.max(np.abs(matrix@weights-target))>tolerance:
        raise ValueError('Convex-domain solve exceeds its declared physical-coordinate residual budget')
    return weights


def publish_dm_wake_state_response(parameters,destination):
    import copy
    import numpy as np
    from pathlib import Path
    from core.sidm_analysis import _load_product
    from core.foundation import canonical_digest,file_sha256
    from core.data import export_numeric_audit
    request=validate_dm_wake_state_response_request(parameters);sources={};signature=None;fixed=None
    records={'training':[],'heldout':[]};phases={'training':set(),'heldout':set()};operators={'training':set(),'heldout':set()}
    seen_products=set();provenance={'training':[],'heldout':[]}
    for role in ('training','heldout'):
        for reference in request[role]:
            if reference['sha256'] in seen_products:raise ValueError('State response reuses an ensemble product')
            seen_products.add(reference['sha256'])
            report,arrays,binding=_load_product(reference,parameters['request_base'],request['memory_gib']*1024**3/4);sources.update(binding)
            if report.get('schema')!='measured-native-dm-wake-ensemble-v1' or report.get('status')!='SCALAR_DRAG_MEAN_BUDGETS_SUPPORTED' or report['request']['schema']!='native-dm-wake-ensemble-request-v2':
                raise ValueError('State response requires supported measured native warm-bath ensembles')
            definition=copy.deepcopy(report['fixed_wake_definition']);definition.pop('relative_velocity');definition.pop('state_variation_contract',None)
            definition['operator'].pop('probe_velocity_peculiar_km_s')
            current=canonical_digest(definition)
            if signature is not None and signature!=current:raise ValueError('State response changes unmodeled BH/DM physics, probe, epoch or native resolution')
            signature=current;fixed=definition
            velocities=np.asarray(arrays['realization_relative_velocity_km_s'],float);densities=np.asarray(arrays['realization_bath_density_msun_kpc3'],float)
            covariances=np.asarray(arrays['realization_bath_velocity_covariance_km2_s2'],float);forces=np.asarray(arrays['realization_aperture_force_contrasts_cm_s2'],float)
            errors=np.asarray(arrays['reference_resolution_residual_cm_s2'],float);count=report['realization_count']
            if velocities.shape!=(count,3) or densities.shape!=(count,) or covariances.shape!=(count,3,3) or forces.shape!=(count,3) or errors.shape!=(count,):
                raise ValueError('Complete aligned native realization state/force vectors required')
            if not np.isfinite(forces).all() or not np.isfinite(errors).all() or np.any(errors<0):raise ValueError('Finite signed measured forces and reference errors required')
            seeds=report['ordered_realization_seeds']
            if len(seeds)!=count:raise ValueError('Incomplete original native realization seed census')
            if len(records[role])+count>request['maximum_realizations']:raise MemoryError('State fit exceeds actual realization ceiling')
            for index,seed in enumerate(seeds):
                phase=seed['phase_seed'];operator=seed['operator_seed']
                if phase in phases[role] or operator is not None and operator in operators[role]:raise ValueError('State fit requires independent native phase/operator axes within each role')
                phases[role].add(phase)
                if operator is not None:operators[role].add(operator)
                coordinates=_wake_state_coordinates(velocities[index],float(densities[index]),covariances[index])
                records[role].append((coordinates,forces[index],float(errors[index])))
                provenance[role].append({'ensemble_sha256':reference['sha256'],'realization_index':index,'native_seed':seed})
    if phases['training']&phases['heldout'] or operators['training']&operators['heldout']:raise ValueError('Held-out state response reuses training native seed axes')
    scales=np.asarray(request['coordinate_scales'],float);active=np.asarray([WAKE_STATE_COORDINATES.index(name) for name in request['active_coordinates']],int)
    training=np.asarray([row[0] for row in records['training']]);heldout=np.asarray([row[0] for row in records['heldout']]);center=training.mean(axis=0)
    normalized=(training-center)/scales;queries=(heldout-center)/scales
    design=np.c_[np.ones(len(training)),normalized[:,active]];target=np.asarray([row[1] for row in records['training']])
    if len(training)<=design.shape[1]:raise ValueError('Overdetermined native state design required')
    singular=np.linalg.svd(design,compute_uv=False)
    if not np.isfinite(singular).all() or singular[-1]/singular[0]<request['minimum_design_singular_ratio']:raise ValueError('Native state response design is rank deficient or unsupported')
    coefficients=np.linalg.lstsq(design,target,rcond=request['minimum_design_singular_ratio'])[0]
    residuals=[];allowed=[];supported=[];weights=[];inside_flags=[];observed=np.asarray([row[1] for row in records['heldout']])
    training_errors=np.asarray([row[2] for row in records['training']])
    for query,(_,force,error) in zip(queries,records['heldout']):
        try:weight=_wake_convex_weights(normalized,query,request['convex_weight_residual_tolerance'])
        except ValueError:weight=np.zeros(len(training));inside=False
        else:inside=True
        prediction=np.r_[1.,query[active]]@coefficients;residual=float(np.linalg.norm(force-prediction))
        allowance=request['maximum_absolute_prediction_error_cm_s2']+request['maximum_relative_prediction_error']*max(float(np.linalg.norm(force)),float(np.linalg.norm(prediction)),request['prediction_error_floor_cm_s2'])
        allowance+=error+float(weight@training_errors)
        residuals.append(residual);allowed.append(allowance);supported.append(inside and residual<=allowance);weights.append(weight);inside_flags.append(inside)
    source=Path(__file__).resolve();sources[str(source)]=file_sha256(source)
    output={'training_coordinates':training,'heldout_coordinates':heldout,'coordinate_center':center,'coordinate_scales':scales,
        'active_coordinate_indices':active,'affine_force_coefficients_cm_s2':coefficients,'design_singular_values':singular,
        'training_force_cm_s2':target,'training_fit_residual_cm_s2':target-design@coefficients,'training_reference_errors_cm_s2':training_errors,
        'heldout_force_cm_s2':observed,'heldout_residual_norm_cm_s2':np.asarray(residuals),'heldout_error_allowance_cm_s2':np.asarray(allowed),
        'heldout_supported':np.asarray(supported,bool),'heldout_inside_measured_domain':np.asarray(inside_flags,bool),'heldout_convex_weights':np.asarray(weights)}
    if any(not np.isfinite(value).all() for value in output.values()):raise ArithmeticError('Native state response fit overflows')
    report={'schema':'calibrated-native-wake-state-response-v1','execution_status':'COMPLETE','request':request,
        'fixed_response_definition':fixed,'fixed_response_signature':signature,'coordinate_names':list(WAKE_STATE_COORDINATES),
        'native_realization_provenance':provenance,'status':'HELDOUT_STATE_BUDGETS_SUPPORTED' if all(supported) else 'UNRESOLVED_STATE_RESPONSE',
        'scope':'local affine vector force of measured native state clouds inside actual convex domain; empirical held-out budgets only; no global coverage, arbitrary BH mass scaling or native tree-force equivalence',
        'science_acceptance':'NOT_ESTABLISHED','source_hashes':sources}
    metadata={name:{'units':'cm / s^2' if 'cm_s2' in name else 'dimensionless' if name in ('active_coordinate_indices','design_singular_values','heldout_supported','heldout_convex_weights') else 'coordinate-dependent',
        'quantity':'native state response '+name} for name in output}
    metadata['heldout_inside_measured_domain']['units']='dimensionless'
    for name in ('training_coordinates','heldout_coordinates','coordinate_center','coordinate_scales'):
        metadata[name]['coordinate_names']=list(WAKE_STATE_COORDINATES);metadata[name]['coordinate_units']=['km / s']*3+['Msun / kpc^3']+['(km / s)^2']*6
    return export_numeric_audit(destination,report,output,metadata,algorithm=source,source_hashes=sources,
        context={'epoch':fixed['epoch'],'coordinates':{'frame':'fixed proper probe; original vector acceleration and measured peculiar bath-relative state'}})


def evaluate_dm_wake_state_response(reference,velocity,density,covariance,*,request_base,fixed_response_signature,memory_gib):
    import numpy as np
    from core.sidm_analysis import _load_product
    report,arrays,sources=_load_product(reference,request_base,finite_number(memory_gib,'state query memory',strict=True)*1024**3/4)
    if report.get('schema')!='calibrated-native-wake-state-response-v1' or report.get('status')!='HELDOUT_STATE_BUDGETS_SUPPORTED' or report['fixed_response_signature']!=fixed_response_signature:
        raise ValueError('Supported actual state response with unchanged fixed native physics required')
    request=validate_dm_wake_state_response_request({'request':report['request']})
    if request['memory_gib']>memory_gib:raise MemoryError('State domain solver exceeds query allocation')
    coordinates=_wake_state_coordinates(velocity,density,covariance);query=(coordinates-arrays['coordinate_center'])/arrays['coordinate_scales']
    training=(arrays['training_coordinates']-arrays['coordinate_center'])/arrays['coordinate_scales']
    weights=_wake_convex_weights(training,query,request['convex_weight_residual_tolerance'])
    prediction=np.r_[1.,query[np.asarray(arrays['active_coordinate_indices'],int)]]@arrays['affine_force_coefficients_cm_s2']
    if not np.isfinite(prediction).all():raise ArithmeticError('Native state response evaluation overflows')
    return {'acceleration_cm_s2':prediction,'convex_weights':weights,'reference_error_cm_s2':float(weights@arrays['training_reference_errors_cm_s2']),
        'empirical_heldout_residual_max_cm_s2':float(np.max(arrays['heldout_residual_norm_cm_s2'])),'source_hashes':sources,
        'scope':'measured convex-domain local vector response; conservative source caller must independently require dissipative work'}


def dm_wake_state_response_runner(parameters,context):
    from pathlib import Path
    if parameters['request']['memory_gib']>context.resources['memory_gib']:raise MemoryError('State response exceeds allocated memory')
    destination=Path(parameters['destination']).resolve()
    if not destination.is_relative_to(context.generation):raise ValueError('State response escapes its owned generation')
    return publish_dm_wake_state_response(parameters,destination)


def dm_wake_state_response_main():
    import argparse,json
    from pathlib import Path
    parser=argparse.ArgumentParser(description='Fit held-out native wake state-cloud vector responses')
    parser.add_argument('parameters',type=Path);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();path=args.parameters.resolve();parameters=json.loads(path.read_text());parameters.setdefault('request_base',str(path.parent))
    publish_dm_wake_state_response(parameters,args.output);return 0


def stage_calibrated_bh_drag_step(response_reference,background_reference,carrier,bath,*,request_base,
    fixed_response_signature,proper_start_seconds,proper_end_seconds,maximum_proper_step_seconds,
    maximum_reference_error_cm_s2,maximum_heldout_error_cm_s2,seed,counter,conservative_limits):
    """Source-bound measured background -> calibrated force -> staged kick.

    The background and live recipient bath have separate physical roles.
    Returns new velocities and a proposed counter without mutating either
    state. The production caller still owns source scheduling/commit/restart.
    """
    import numpy as np
    from core.sidm_analysis import _load_product
    if type(counter)is not int or not 0<=counter<2**64-1:raise ValueError('Unexhausted exact source counter required')
    start=finite_number(proper_start_seconds,'proper source start',minimum=-math.inf)
    end=finite_number(proper_end_seconds,'proper source end',minimum=-math.inf)
    maximum_dt=finite_number(maximum_proper_step_seconds,'maximum proper source step',strict=True)
    dt=end-start
    if not math.isfinite(dt) or not 0<dt<=maximum_dt:raise ValueError('Explicit increasing proper source interval within its step bound required')
    reference_limit=finite_number(maximum_reference_error_cm_s2,'reference source error limit')
    heldout_limit=finite_number(maximum_heldout_error_cm_s2,'held-out source error limit')
    keys={'maximum_particles','memory_gib','maximum_fractional_relative_velocity_change','minimum_inertia_eigenvalue_ratio',
        'conservation_relative_tolerance','momentum_absolute_floor_msun_km_s','energy_absolute_floor_msun_km2_s2','angular_absolute_floor_msun_kpc_km_s'}
    if not isinstance(conservative_limits,dict) or set(conservative_limits)!=keys:raise ValueError('Complete conservative recipient/source limits required')
    memory=finite_number(conservative_limits['memory_gib'],'coupled source memory',strict=True)
    response,response_arrays,response_sources=_load_product(response_reference,request_base,memory*1024**3/4)
    del response_arrays
    background,arrays,background_sources=_load_product(background_reference,request_base,memory*1024**3/4)
    if response.get('schema')!='calibrated-native-wake-state-response-v1' or response['fixed_response_signature']!=fixed_response_signature:
        raise ValueError('Original calibrated response with unchanged fixed native signature required')
    if background.get('schema')!='measured-native-dm-wake-force-v1' or background.get('execution_status')!='COMPLETE':
        raise ValueError('Original source-bound measured background force/state product required')
    fixed=response['fixed_response_definition'];perturber=fixed['perturber']
    if perturber.get('kind')!='original_native_bh_particle' or carrier['particle_id']!=perturber['particle_id'] or carrier['particle_namespace']!=perturber['particle_namespace']:
        raise ValueError('Coupled source changes its calibrated actual native BH identity')
    if carrier['physical_bh_mass_msun']!=perturber['physical_bh_mass_msun'] or carrier['gravity_mass_msun']!=perturber['gravity_carrier_mass_msun']:
        raise ValueError('Coupled source changes calibrated physical/carrier BH mass')
    if list(carrier['position_proper_kpc'])!=fixed['operator']['probe_position_proper_kpc']:
        raise ValueError('Coupled source changes its calibrated physical probe position')
    operator=dict(background['operator_definition']);operator.pop('probe_velocity_peculiar_km_s')
    if operator!=fixed['operator'] or background['native_dm_physics_signature']!=fixed['physics']:
        raise ValueError('Coupled source background changes its force operator or native DM physics')
    if {key:background['epoch'].get(key) for key in fixed['epoch']}!=fixed['epoch']:
        raise ValueError('Coupled source background changes the calibrated physical epoch')
    available=np.asarray(arrays['aperture_velocity_moments_available'])
    mean=np.asarray(arrays['aperture_mean_velocity_km_s'],float);covariance=np.asarray(arrays['aperture_velocity_covariance_km2_s2'],float)
    mass=np.asarray(arrays['aperture_mass_msun'],float)
    if available.shape!=(1,) or available.dtype.kind!='b' or not available[0] or mass.shape!=(1,) or not np.isfinite(mass).all() or mass[0]<=0:
        raise ValueError('Available positive actual background aperture state required')
    edges=np.asarray(background['operator_definition']['radial_edges_proper_kpc'],float)
    density=float(mass[0]/(4*math.pi/3*(edges[-1]**3-edges[0]**3)))
    velocity=np.asarray(carrier['velocity_peculiar_km_s'],float)-mean
    del arrays
    evaluated=evaluate_dm_wake_state_response(response_reference,velocity,density,covariance,request_base=request_base,
        fixed_response_signature=fixed_response_signature,memory_gib=memory)
    if evaluated['reference_error_cm_s2']>reference_limit or evaluated['empirical_heldout_residual_max_cm_s2']>heldout_limit:
        raise ValueError('Coupled calibrated force exceeds declared source error budgets')
    source_allocation=dict(conservative_limits);source_allocation['memory_gib']=memory/2
    result=conservative_bh_drag_step(carrier,bath,evaluated['acceleration_cm_s2'],dt,seed=seed,counter=counter,**source_allocation)
    sources={**response_sources,**background_sources,**evaluated['source_hashes']}
    result.update({'response_evaluation':evaluated,'background_product_sha256':background_reference['sha256'],
        'response_product_sha256':response_reference['sha256'],'proper_start_seconds':start,'proper_end_seconds':end,
        'source_counter':counter,'proposed_next_source_counter':counter+1,'source_hashes':sources,
        'commit_status':'STAGED_ONLY','scope':'fixed-epoch measured-background calibrated conservative source; production clock/commit/restart and native force equivalence required'})
    return result


DRAG_CONTEXT_LIMITS=('maximum_proper_step_seconds','maximum_reference_error_cm_s2','maximum_heldout_error_cm_s2',
    'maximum_fractional_relative_velocity_change','minimum_inertia_eigenvalue_ratio','conservation_relative_tolerance',
    'momentum_absolute_floor_msun_km_s','energy_absolute_floor_msun_km2_s2','angular_absolute_floor_msun_kpc_km_s',
    'proper_clock_absolute_tolerance_seconds','proper_clock_relative_tolerance')


def validate_dm_wake_context_request(parameters):
    request=parameters['request']
    keys={'schema','response','response_table','background_force','limits','seed','maximum_particles','maximum_workspace_bytes',
        'maximum_subcycles','maximum_context_bytes','memory_gib'}
    if isinstance(request,dict) and 'maximum_schedule_workspace_bytes' in request:keys.add('maximum_schedule_workspace_bytes')
    if not isinstance(request,dict) or set(request)!=keys or request['schema']!='native-wake-context-request-v1':raise ValueError('Complete source-bound native drag context request required')
    if type(request['seed'])is not int or not 0<=request['seed']<2**64:raise ValueError('Exact native drag seed required')
    for name,minimum in (('maximum_particles',3),('maximum_workspace_bytes',1024),('maximum_subcycles',1),('maximum_context_bytes',128)):
        if type(request[name])is not int or not minimum<=request[name]<2**64:raise ValueError('Exact bounded native drag context work required')
    if request['maximum_subcycles']>4096:raise ValueError('Native drag subcycle ceiling exceeds supported domain')
    schedule=request.get('maximum_schedule_workspace_bytes',request['maximum_subcycles']*1024)
    if type(schedule)is not int or not 1024<=schedule<2**64:raise ValueError('Exact positive bounded native source schedule workspace required')
    finite_number(request['memory_gib'],'context memory',strict=True)
    if request['maximum_context_bytes']*8>request['memory_gib']*1024**3/4 or request['maximum_workspace_bytes']+schedule>request['memory_gib']*1024**3:
        raise MemoryError('Native context/source workspace exceeds allocation')
    limits=request['limits']
    if not isinstance(limits,dict) or set(limits)!=set(DRAG_CONTEXT_LIMITS):raise ValueError('Complete explicit native source numerical limits required')
    for name in DRAG_CONTEXT_LIMITS:
        finite_number(limits[name],name,strict=name not in ('maximum_reference_error_cm_s2','maximum_heldout_error_cm_s2','proper_clock_absolute_tolerance_seconds','proper_clock_relative_tolerance'))
    for name in ('maximum_fractional_relative_velocity_change','minimum_inertia_eigenvalue_ratio','conservation_relative_tolerance'):
        if limits[name]>=1:raise ValueError('Native source fraction/conditioning/conservation bounds must be below one')
    if limits['proper_clock_relative_tolerance']>1e-6:raise ValueError('Unsupported proper source clock tolerance')
    return request


def dm_wake_context_input_bindings(parameters):
    from core.sidm_analysis import _load_product
    request=validate_dm_wake_context_request(parameters);sources={}
    for name in ('response','response_table','background_force'):
        sources.update(_load_product(request[name],parameters['request_base'],request['memory_gib']*1024**3/4)[2])
    return sources


def publish_dm_wake_context(parameters,destination):
    import numpy as np,zlib
    from pathlib import Path
    from core.sidm_analysis import _load_product
    from core.foundation import atomic_copy,atomic_text,atomic_json,file_sha256
    from core.data import export_numeric_audit
    request=validate_dm_wake_context_request(parameters);sources={};products={}
    for name in ('response','response_table','background_force'):
        report,arrays,binding=_load_product(request[name],parameters['request_base'],request['memory_gib']*1024**3/4);sources.update(binding)
        products[name]=(report,arrays)
    response,_=products['response'];table,_=products['response_table'];background,bath=products['background_force']
    if response.get('schema')!='calibrated-native-wake-state-response-v1' or response.get('status')!='HELDOUT_STATE_BUDGETS_SUPPORTED' or table.get('schema')!='native-wake-state-table-v1':raise ValueError('Actual supported native response/table pair required')
    if table['response_product_sha256']!=request['response']['sha256'] or table['fixed_response_signature']!=response['fixed_response_signature']:raise ValueError('Native context changes its response/table binding')
    if background.get('schema')!='measured-native-dm-wake-force-v1' or background.get('execution_status')!='COMPLETE':raise ValueError('Complete measured background required')
    fixed=response['fixed_response_definition'];operator=dict(background['operator_definition']);operator.pop('probe_velocity_peculiar_km_s')
    if operator!=fixed['operator'] or background['native_dm_physics_signature']!=fixed['physics'] or {key:background['epoch'].get(key) for key in fixed['epoch']}!=fixed['epoch']:
        raise ValueError('Native context background changes actual operator, physics or epoch')
    bh=fixed['perturber']
    if bh.get('kind')!='original_native_bh_particle':raise ValueError('Actual native BH identity and masses required')
    available=np.asarray(bath['aperture_velocity_moments_available']);mass=np.asarray(bath['aperture_mass_msun'],float)
    if available.shape!=(1,) or available.dtype.kind!='b' or not available[0] or mass.shape!=(1,) or not np.isfinite(mass).all() or mass[0]<=0:raise ValueError('Available original bath aperture state required')
    edges=np.asarray(operator['radial_edges_proper_kpc'],float);density=float(mass[0]/(4*math.pi/3*(edges[-1]**3-edges[0]**3)))
    mean=np.asarray(bath['aperture_mean_velocity_km_s'],float);covariance=np.asarray(bath['aperture_velocity_covariance_km2_s2'],float)
    # Validate physical bath moments without manufacturing a query velocity.
    _wake_state_coordinates([1.,0.,0.],density,covariance)
    if mean.shape!=(3,) or not np.isfinite(mean).all():raise ValueError('Complete original bath mean velocity required')
    epoch=fixed['epoch'];native_epoch=epoch.get('actual_scale_factor') if epoch['canonical_time']=='scale_factor' else epoch.get('native_time_code')
    finite_number(native_epoch,'actual calibrated native epoch')
    macro=fixed['macro_masses'][fixed['coarse_resolution_index']];finite_number(macro,'actual calibrated recipient macro mass',strict=True)
    box=background['native_proper_box_kpc']
    if not operator['periodic_nearest_image'] or not edges[-1]<box/2:raise ValueError('Native context requires actual periodic nearest-image support')
    state=[native_epoch,*operator['probe_position_proper_kpc'],bh['physical_bh_mass_msun'],bh['gravity_carrier_mass_msun'],
        *mean,density,*np.diag(covariance),covariance[0,1],covariance[0,2],covariance[1,2],edges[0],edges[-1],macro,box]
    if not np.isfinite(state).all():raise ValueError('Native context physical fields overflow')
    source_table=Path(table['table_path'])
    if source_table.is_symlink():raise ValueError('Native context table must be an original regular source asset')
    source_table=source_table.resolve()
    if sources.get(str(source_table))!=table['table_sha256'] or file_sha256(source_table)!=table['table_sha256']:
        raise ValueError('Native context table is outside its original sealed source binding')
    schedule_workspace=request.get('maximum_schedule_workspace_bytes',request['maximum_subcycles']*1024)
    table_bounds=table['request']
    if (request['maximum_workspace_bytes']+schedule_workspace+request['maximum_context_bytes']*8+
            table_bounds['maximum_table_bytes']*8+table_bounds['maximum_states']*4096)>request['memory_gib']*1024**3:
        raise MemoryError('Combined native asset, source and schedule workspace exceeds the context allocation')
    destination=Path(destination).resolve();directory=destination.with_name(destination.stem+'-assets')
    if destination.exists() or directory.exists():raise FileExistsError('Retain existing source-bound native context')
    lines=['RESEARCH2_WAKE_CONTEXT_V1',request['response']['sha256'],response['fixed_response_signature'],fixed['physics'],table['table_sha256'],
        str(table['table_crc32']),str(bh['particle_id']),f"{request['seed']} {request['maximum_particles']} {request['maximum_workspace_bytes']} {request['maximum_subcycles']}",
        ' '.join(format(float(x),'.17g') for x in state),' '.join(format(request['limits'][name],'.17g') for name in DRAG_CONTEXT_LIMITS)]
    payload='\n'.join(lines)+'\n';encoded=payload.encode('ascii')
    if len(encoded)>request['maximum_context_bytes']:raise MemoryError('Native context bytes exceed declared limit')
    directory.mkdir(parents=True);target=directory/'response-table.dat';atomic_copy(source_table,target)
    if file_sha256(target)!=table['table_sha256']:raise ValueError('Native response table changed during context materialization')
    context_path=directory/'context.dat';atomic_text(context_path,payload)
    launch={'schema':'native-wake-launch-assets-v1','context_path':str(context_path),
        'context_sha256':file_sha256(context_path),'context_crc32':int(zlib.crc32(encoded)),
        'table_path':str(target),'maximum_context_bytes':request['maximum_context_bytes'],
        'maximum_table_bytes':table_bounds['maximum_table_bytes'],'maximum_states':table_bounds['maximum_states'],
        'maximum_domain_iterations':table_bounds['maximum_domain_iterations'],
        'maximum_schedule_workspace_bytes':schedule_workspace,'expected_native_physics_signature':fixed['physics'],
        'response_product_sha256':request['response']['sha256'],'fixed_response_signature':response['fixed_response_signature'],
        'native_epoch':native_epoch,'maximum_subcycles':request['maximum_subcycles'],
        'binding_policy':'caller supplies actual native physics signature and current epoch; checked asset wrappers reject stale contexts',
        'production_enablement':'NOT_ENABLED; native force equivalence and production epoch/model selection required'}
    launch_path=directory/'launch-assets.json';atomic_json(launch_path,launch)
    sources[str(launch_path)]=file_sha256(launch_path)
    sources[str(target)]=file_sha256(target);sources[str(context_path)]=file_sha256(context_path);source=Path(__file__).resolve();sources[str(source)]=file_sha256(source)
    report={'schema':'native-wake-context-v1','execution_status':'COMPLETE','request':request,'context_path':str(context_path),
        'context_sha256':file_sha256(context_path),'context_crc32':int(zlib.crc32(encoded)),'context_bytes':len(encoded),
        'response_table_path':str(target),'response_table_sha256':file_sha256(target),'response_table_crc32':table['table_crc32'],
        'native_epoch':native_epoch,'particle_namespace':bh['particle_namespace'],'source_hashes':sources,'science_acceptance':'NOT_ESTABLISHED',
        'launch_assets':{'path':str(launch_path),'sha256':file_sha256(launch_path)},'launch_settings':launch,
        'scope':'original measured BH/background/response native intake; caller must bind actual launch physics and force equivalence; no production enablement'}
    return export_numeric_audit(destination,report,{'context_byte_count':np.asarray([len(encoded)],np.uint64)},
        {'context_byte_count':{'units':'bytes','quantity':'actual native context size'}},algorithm=source,source_hashes=sources,context={'epoch':epoch})


def read_dm_wake_launch_assets(reference,*,request_base,memory_gib,actual_native_physics_signature,actual_native_epoch):
    """Resolve launch settings from a sealed context, never from unbound paths.

    The caller derives the actual physics identity/epoch from its native
    configuration. This intake does not certify native force equivalence or
    select a response outside its measured epoch/model domain.
    """
    import json,zlib
    from pathlib import Path
    from core.foundation import file_sha256
    from core.sidm_analysis import _load_product
    finite_number(memory_gib,'launch intake memory',strict=True);finite_number(actual_native_epoch,'actual native launch epoch')
    if not isinstance(actual_native_physics_signature,str) or len(actual_native_physics_signature)!=64 or any(c not in '0123456789abcdef' for c in actual_native_physics_signature):
        raise ValueError('Exact actual native physics SHA256 identity required')
    product,_,sources=_load_product(reference,request_base,memory_gib*1024**3/3)
    if product.get('schema')!='native-wake-context-v1' or product.get('execution_status')!='COMPLETE':
        raise ValueError('Actual complete sealed native drag context required')
    validate_dm_wake_context_request({'request':product['request']})
    pointer=product['launch_assets'];path=Path(pointer['path'])
    if path.is_symlink() or not path.is_absolute():raise ValueError('Native launch settings must be an original absolute regular asset')
    path=path.resolve()
    if sources.get(str(path))!=pointer['sha256'] or file_sha256(path)!=pointer['sha256'] or path.stat().st_size*8>memory_gib*1024**3/3:
        raise ValueError('Native launch settings change their sealed source or exceed intake allocation')
    launch=json.loads(path.read_text())
    if launch!=product['launch_settings'] or launch.get('schema')!='native-wake-launch-assets-v1':
        raise ValueError('Native launch settings differ from the published context contract')
    if launch['expected_native_physics_signature']!=actual_native_physics_signature or launch['native_epoch']!=actual_native_epoch:
        raise ValueError('Actual launch physics/epoch differ from the measured calibrated context')
    for field,prefix in (('context_path','context'),('table_path','response_table')):
        asset=Path(launch[field])
        if asset.is_symlink() or not asset.is_absolute():raise ValueError('Native launch payload must be an original absolute regular asset')
        asset=asset.resolve();expected=product[prefix+'_sha256']
        if str(asset)!=product[prefix+'_path'] or sources.get(str(asset))!=expected or file_sha256(asset)!=expected:
            raise ValueError('Native launch payload changes its original source binding')
        ceiling=launch['maximum_context_bytes'] if field=='context_path' else launch['maximum_table_bytes']
        if type(ceiling)is not int or not 128<=ceiling or asset.stat().st_size>ceiling or asset.stat().st_size*8>memory_gib*1024**3/3:
            raise MemoryError('Native launch payload exceeds its complete byte allocation')
        data=asset.read_bytes()
        expected_crc=product[prefix+'_crc32']
        if int(zlib.crc32(data))!=expected_crc:raise ValueError('Native launch payload CRC differs from the source contract')
    if launch['context_crc32']!=product['context_crc32'] or launch['context_sha256']!=product['context_sha256']:
        raise ValueError('Native launch context SHA/CRC arguments changed')
    total=(8*launch['maximum_context_bytes']+8*launch['maximum_table_bytes']+4096*launch['maximum_states']+
        launch['maximum_schedule_workspace_bytes']+product['request']['maximum_workspace_bytes'])
    if total>memory_gib*1024**3:raise MemoryError('Combined calibrated source/asset/schedule intake exceeds allocation')
    source=Path(__file__).resolve();sources[str(source)]=file_sha256(source)
    return {'launch_settings':launch,'source_hashes':sources,'context_product_sha256':reference['sha256'],
        'scope':'checked native asset arguments at one actual measured epoch; no force equivalence or production enablement'}


def read_dm_wake_transition_assets(previous_reference,next_reference,*,request_base,memory_gib,
        actual_native_physics_signature,actual_native_epoch):
    """Bind a response replacement without changing the physical experiment.

    Both contexts must describe the same current background/carrier/operator
    and random stream. Different epochs or physical definitions require a
    separately implemented transition, not a response-table replacement.
    """
    import numpy as np
    from core.sidm_analysis import _load_product
    finite_number(memory_gib,'transition asset memory',strict=True)
    if previous_reference['sha256']==next_reference['sha256']:raise ValueError('Response replacement requires distinct context products')
    records=[];contexts=[];sources={}
    for reference in (previous_reference,next_reference):
        bound=read_dm_wake_launch_assets(reference,request_base=request_base,memory_gib=memory_gib/2,
            actual_native_physics_signature=actual_native_physics_signature,actual_native_epoch=actual_native_epoch)
        product,_,binding=_load_product(reference,request_base,memory_gib*1024**3/6)
        sources.update(bound['source_hashes']);sources.update(binding);contexts.append(product);records.append(bound['launch_settings'])
    left,right=records;old,new=contexts
    if left['response_product_sha256']==right['response_product_sha256'] or left['fixed_response_signature']!=right['fixed_response_signature']:
        raise ValueError('Response replacement must change the response product while retaining its fixed physical definition')
    if old['request']['seed']!=new['request']['seed'] or old['request']['background_force']['sha256']!=new['request']['background_force']['sha256']:
        raise ValueError('Response replacement changes original background or random stream')
    if old['particle_namespace']!=new['particle_namespace']:raise ValueError('Response replacement changes native identity namespace')
    # The same background product and fixed signature bind the carrier,
    # aperture and physical epoch. Native wrappers additionally compare the
    # complete parsed physical fields before committing a metadata transition.
    for name in ('maximum_particles','maximum_workspace_bytes','maximum_subcycles'):
        if old['request'][name]!=new['request'][name]:raise ValueError('Response replacement changes source allocation independently')
    if not np.isfinite([left['native_epoch'],right['native_epoch']]).all():raise ValueError('Response replacement epoch is nonfinite')
    return {'schema':'native-wake-response-transition-assets-v1','previous_launch_settings':left,'next_launch_settings':right,
        'previous_context_sha256':previous_reference['sha256'],'next_context_sha256':next_reference['sha256'],
        'source_hashes':sources,'state_policy':'retain source counter, clocks, cumulative heat and last applied acceleration/errors; journal explicit old/new identities',
        'scope':'same-definition response replacement only; current-state native domain/error checks required before commit'}


def dm_wake_context_runner(parameters,context):
    from pathlib import Path
    if parameters['request']['memory_gib']>context.resources['memory_gib']:raise MemoryError('Native context exceeds allocated memory')
    destination=Path(parameters['destination']).resolve()
    if not destination.is_relative_to(context.generation):raise ValueError('Native context escapes its owned generation')
    return publish_dm_wake_context(parameters,destination)


def dm_wake_context_main():
    import argparse,json
    from pathlib import Path
    parser=argparse.ArgumentParser(description='Export source-bound measured native BH drag context')
    parser.add_argument('parameters',type=Path);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();path=args.parameters.resolve();parameters=json.loads(path.read_text());parameters.setdefault('request_base',str(path.parent))
    publish_dm_wake_context(parameters,args.output);return 0


def validate_dm_wake_state_table_request(parameters):
    request=parameters['request']
    keys={'schema','response','memory_gib','maximum_states','maximum_table_bytes','maximum_domain_iterations'}
    if not isinstance(request,dict) or set(request)!=keys or request['schema']!='native-wake-state-table-request-v1':
        raise ValueError('Complete native state-response export request required')
    for name,minimum in (('maximum_states',3),('maximum_table_bytes',128),('maximum_domain_iterations',1)):
        if type(request[name])is not int or request[name]<minimum:raise ValueError('Exact bounded native state table work required')
    if request['maximum_states']>4096 or request['maximum_domain_iterations']>1000000:raise ValueError('Native state table work exceeds supported bounds')
    finite_number(request['memory_gib'],'native state export memory',strict=True)
    if request['maximum_table_bytes']*8+request['maximum_states']*4096>request['memory_gib']*1024**3:
        raise MemoryError('Native state serialization workspace exceeds allocation')
    return request


def dm_wake_state_table_input_bindings(parameters):
    from core.sidm_analysis import _load_product
    request=validate_dm_wake_state_table_request(parameters)
    return _load_product(request['response'],parameters['request_base'],request['memory_gib']*1024**3/4)[2]


def publish_dm_wake_state_table(parameters,destination):
    import numpy as np,zlib
    from pathlib import Path
    from core.sidm_analysis import _load_product
    from core.foundation import atomic_text,file_sha256
    from core.data import export_numeric_audit
    request=validate_dm_wake_state_table_request(parameters)
    response,arrays,sources=_load_product(request['response'],parameters['request_base'],request['memory_gib']*1024**3/4)
    if response.get('schema')!='calibrated-native-wake-state-response-v1' or response.get('status')!='HELDOUT_STATE_BUDGETS_SUPPORTED':
        raise ValueError('Native state table requires the supported actual held-out response')
    center=np.asarray(arrays['coordinate_center'],float);scales=np.asarray(arrays['coordinate_scales'],float)
    coordinates=np.asarray(arrays['training_coordinates'],float);active=np.asarray(arrays['active_coordinate_indices'])
    coefficients=np.asarray(arrays['affine_force_coefficients_cm_s2'],float);errors=np.asarray(arrays['training_reference_errors_cm_s2'],float)
    supported=np.asarray(arrays['heldout_supported']);heldout=np.asarray(arrays['heldout_residual_norm_cm_s2'],float)
    n=len(coordinates);k=len(active)
    if center.shape!=(10,) or scales.shape!=(10,) or coordinates.shape!=(n,10) or not 3<=n<=request['maximum_states'] or not 1<=k<=10 or n<=k+1:
        raise ValueError('Complete bounded native state coordinate design required')
    if active.dtype.kind not in 'iu' or len(np.unique(active))!=k or np.any(active<0) or np.any(active>=10) or coefficients.shape!=(k+1,3) or errors.shape!=(n,):
        raise ValueError('Native state coefficients/coordinate identities are incomplete')
    if any(not np.isfinite(value).all() for value in (center,scales,coordinates,coefficients,errors,heldout)) or np.any(scales<=0) or np.any(errors<0):
        raise ValueError('Native state table has nonfinite or nonphysical scales/errors')
    if supported.dtype.kind!='b' or supported.ndim!=1 or not len(supported) or not supported.all() or heldout.shape!=supported.shape or np.any(heldout<0):
        raise ValueError('Native state table lacks complete held-out support')
    normalized=(coordinates-center)/scales
    if not np.isfinite(normalized).all():raise ValueError('Scaled native state table overflows')
    tolerance=response['request']['convex_weight_residual_tolerance'];heldout_error=float(heldout.max())
    lines=['RESEARCH2_WAKE_STATE_RESPONSE_V1',request['response']['sha256'],response['fixed_response_signature'],
        f"{n} {k} {request['maximum_domain_iterations']}",f'{tolerance:.17g} {heldout_error:.17g}',
        ' '.join(format(x,'.17g') for x in center),' '.join(format(x,'.17g') for x in scales),
        ' '.join(str(int(x)) for x in active)]
    lines.extend(' '.join(format(x,'.17g') for x in row) for row in coefficients)
    lines.extend(' '.join(format(x,'.17g') for x in [*row,error]) for row,error in zip(normalized,errors))
    payload='\n'.join(lines)+'\n';encoded=payload.encode('ascii')
    if len(encoded)>request['maximum_table_bytes']:raise MemoryError('Actual native state table exceeds byte ceiling')
    destination=Path(destination).resolve();table=destination.with_name(destination.stem+'-table.dat')
    if destination.exists() or table.exists():raise FileExistsError('Retain previous native state response exports')
    atomic_text(table,payload);sources[str(table)]=file_sha256(table);source=Path(__file__).resolve();sources[str(source)]=file_sha256(source)
    report={'schema':'native-wake-state-table-v1','execution_status':'COMPLETE','request':request,'table_path':str(table),
        'table_sha256':file_sha256(table),'table_crc32':int(zlib.crc32(encoded)),'table_bytes':len(encoded),
        'response_product_sha256':request['response']['sha256'],'fixed_response_signature':response['fixed_response_signature'],
        'coordinate_names':response['coordinate_names'],'training_state_count':n,'active_coordinate_count':k,
        'scope':'source-bound measured affine vector table; native projected-gradient domain solve is independently residual-gated and may reject on iteration exhaustion; no production force equivalence or enablement',
        'source_hashes':sources,'science_acceptance':'NOT_ESTABLISHED'}
    return export_numeric_audit(destination,report,{'table_byte_count':np.asarray([len(encoded)],np.uint64)},
        {'table_byte_count':{'units':'bytes','quantity':'actual retained native state table size'}},algorithm=source,source_hashes=sources,
        context={'epoch':response['fixed_response_definition']['epoch'],'coordinates':{'frame':'fixed measured native probe and physical bath state'}})


def dm_wake_state_table_runner(parameters,context):
    from pathlib import Path
    if parameters['request']['memory_gib']>context.resources['memory_gib']:raise MemoryError('State table export exceeds allocation')
    destination=Path(parameters['destination']).resolve()
    if not destination.is_relative_to(context.generation):raise ValueError('State table escapes its owned generation')
    return publish_dm_wake_state_table(parameters,destination)


def dm_wake_state_table_main():
    import argparse,json
    from pathlib import Path
    parser=argparse.ArgumentParser(description='Export held-out native wake state-response tables')
    parser.add_argument('parameters',type=Path);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();path=args.parameters.resolve();parameters=json.loads(path.read_text());parameters.setdefault('request_base',str(path.parent))
    publish_dm_wake_state_table(parameters,args.output);return 0


def validate_dm_wake_native_table_request(parameters):
    request=parameters['request']
    keys={'schema','response','memory_gib','maximum_knots','maximum_table_bytes'}
    if not isinstance(request,dict) or set(request)!=keys or request['schema']!='native-wake-response-table-request-v1':
        raise ValueError('Complete measured wake-response native table export request required')
    for key in ('maximum_knots','maximum_table_bytes'):
        if type(request[key])is not int or request[key]<(3 if key=='maximum_knots' else 128):raise ValueError('Exact bounded native response export limits required')
    finite_number(request['memory_gib'],'native response export memory',strict=True)
    if request['maximum_table_bytes']*8+request['maximum_knots']*512>request['memory_gib']*1024**3:
        raise MemoryError('Native response serialization workspace exceeds allocation')
    return request


def dm_wake_native_table_input_bindings(parameters):
    from core.sidm_analysis import _load_product
    request=validate_dm_wake_native_table_request(parameters)
    return _load_product(request['response'],parameters['request_base'],request['memory_gib']*1024**3/4)[2]


def publish_dm_wake_native_table(parameters,destination):
    import numpy as np
    import zlib
    from pathlib import Path
    from core.sidm_analysis import _load_product
    from core.foundation import atomic_text,file_sha256
    from core.data import export_numeric_audit
    request=validate_dm_wake_native_table_request(parameters)
    response,arrays,sources=_load_product(request['response'],parameters['request_base'],request['memory_gib']*1024**3/4)
    if response.get('schema')!='calibrated-native-wake-speed-response-v1':raise ValueError('Native table requires the actual held-out calibrated response producer')
    names=('speed_knots_km_s','deceleration_knots_cm_s2','knot_mean_standard_error_cm_s2','knot_reference_error_cm_s2')
    count=len(arrays[names[0]]);supported=np.asarray(arrays['interval_calibration_supported'])
    if not 3<=count<=request['maximum_knots'] or supported.dtype!=np.dtype(bool) or supported.shape!=(count-1,):
        raise ValueError('Native table lacks complete measured knots/interval support')
    for name in names:
        value=np.asarray(arrays[name])
        if value.shape!=(count,) or not np.isfinite(value).all() or np.any(value<0):raise ValueError('Native response knots are incomplete or nonphysical')
    if np.any(arrays[names[0]]<=0) or np.any(np.diff(arrays[names[0]])<=0):raise ValueError('Native response requires strictly increasing positive physical speeds')
    axis=np.asarray(response['velocity_direction'],float)
    if axis.shape!=(3,) or not np.isfinite(axis).all() or not np.isclose(np.linalg.norm(axis),1.,rtol=0,atol=1e-12):raise ValueError('Native response direction is unsupported')
    lines=['RESEARCH2_WAKE_SPEED_RESPONSE_V1',request['response']['sha256'],response['fixed_response_signature'],str(count),
        ' '.join(format(value,'.17g') for value in axis)]
    lines.extend(' '.join(format(arrays[name][row],'.17g') for name in names) for row in range(count))
    lines.append(' '.join(str(int(value)) for value in supported));payload='\n'.join(lines)+'\n';encoded=payload.encode('ascii')
    if len(encoded)>request['maximum_table_bytes']:raise MemoryError('Actual native response table exceeds its byte ceiling')
    destination=Path(destination).resolve();table=destination.with_name(destination.stem+'-table.dat')
    if destination.exists() or table.exists():raise FileExistsError('Keep previous native calibrated response exports')
    atomic_text(table,payload);sources[str(table)]=file_sha256(table);source=Path(__file__).resolve();sources[str(source)]=file_sha256(source)
    report={'schema':'native-calibrated-wake-speed-table-v1','execution_status':'COMPLETE','request':request,
        'table':{'path':str(table),'sha256':sources[str(table)],'crc32':zlib.crc32(encoded)&0xffffffff,'bytes':len(encoded)},
        'response_sha256':request['response']['sha256'],'fixed_response_signature':response['fixed_response_signature'],
        'fixed_response_definition':response['fixed_response_definition'],'velocity_direction':axis.tolist(),
        'native_reader':'black_holes/research_sidm_drag_response.h','scope':response['scope'],
        'science_acceptance':'NOT_ESTABLISHED','source_hashes':sources}
    return export_numeric_audit(destination,report,{name:arrays[name] for name in (*names,'interval_calibration_supported')},
        {name:{'units':'km / s' if name=='speed_knots_km_s' else 'dimensionless' if name=='interval_calibration_supported' else 'cm / s^2',
            'quantity':'retained native measured wake '+name} for name in (*names,'interval_calibration_supported')},
        algorithm=source,source_hashes=sources,context={'epoch':response['fixed_response_definition']['epoch'],'coordinates':{'frame':'recorded physical probe direction/speed domain'}})


def dm_wake_native_table_runner(parameters,context):
    from pathlib import Path
    if parameters['request']['memory_gib']>context.resources['memory_gib']:raise MemoryError('Native response export exceeds allocation')
    destination=Path(parameters['destination']).resolve()
    if not destination.is_relative_to(context.generation):raise ValueError('Native response table escapes its owned generation')
    return publish_dm_wake_native_table(parameters,destination)


def dm_wake_native_table_main():
    import argparse,json
    from pathlib import Path
    parser=argparse.ArgumentParser(description='Export a held-out calibrated wake response to a bounded native table')
    parser.add_argument('parameters',type=Path);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();source=args.parameters.resolve();parameters=json.loads(source.read_text());parameters.setdefault('request_base',str(source.parent))
    publish_dm_wake_native_table(parameters,args.output);return 0


def dm_wake_response_runner(parameters,context):
    from pathlib import Path
    if parameters['request']['memory_gib']>context.resources['memory_gib']:raise MemoryError('Wake response exceeds workflow allocation')
    destination=Path(parameters['destination']).resolve()
    if not destination.is_relative_to(context.generation):raise ValueError('Wake response escapes its owned generation')
    return publish_dm_wake_response(parameters,destination)


def dm_wake_response_main():
    import argparse,json
    from pathlib import Path
    parser=argparse.ArgumentParser(description='Produce held-out calibrated local native wake speed responses')
    parser.add_argument('parameters',type=Path);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();source=args.parameters.resolve();parameters=json.loads(source.read_text());parameters.setdefault('request_base',str(source.parent))
    publish_dm_wake_response(parameters,args.output);return 0


def dm_wake_ensemble_runner(parameters,context):
    from pathlib import Path
    if parameters['request']['memory_gib']>context.resources['memory_gib']:raise MemoryError('Wake ensemble exceeds allocated memory')
    destination=Path(parameters['destination']).resolve()
    if not destination.is_relative_to(context.generation):raise ValueError('Wake ensemble escapes its owned generation')
    return publish_dm_wake_ensemble(parameters,destination)


def dm_wake_ensemble_main():
    import argparse,json
    from pathlib import Path
    parser=argparse.ArgumentParser(description='Measure independent native wake-resolution ensemble means and uncertainty')
    parser.add_argument('parameters',type=Path);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();source=args.parameters.resolve();parameters=json.loads(source.read_text());parameters.setdefault('request_base',str(source.parent))
    publish_dm_wake_ensemble(parameters,args.output);return 0


def dm_wake_contrast_runner(parameters,context):
    from pathlib import Path
    if parameters['request']['memory_gib']>context.resources['memory_gib']:raise MemoryError('Wake contrast exceeds workflow allocation')
    destination=Path(parameters['destination']).resolve()
    if not destination.is_relative_to(context.generation):raise ValueError('Wake contrast escapes its owned generation')
    return publish_dm_wake_contrast(parameters,destination)


def dm_wake_contrast_main():
    import argparse,json
    from pathlib import Path
    parser=argparse.ArgumentParser(description='Measure matched native DM wake/background contrasts at three or more actual resolutions')
    parser.add_argument('parameters',type=Path);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();source=args.parameters.resolve();parameters=json.loads(source.read_text());parameters.setdefault('request_base',str(source.parent))
    publish_dm_wake_contrast(parameters,args.output);return 0


def dm_wake_force_runner(parameters,context):
    from pathlib import Path
    if parameters['request']['memory_gib']>context.resources['memory_gib']:raise MemoryError('Wake force exceeds workflow allocation')
    destination=Path(parameters['destination']).resolve()
    if not destination.is_relative_to(context.generation):raise ValueError('Wake force escapes the owned generation')
    return publish_dm_wake_force(parameters,destination)


def dm_wake_force_main():
    import argparse,json
    from pathlib import Path
    parser=argparse.ArgumentParser(description='Measure original native DM wake forces and bath moments under a declared probe operator')
    parser.add_argument('parameters',type=Path);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();source=args.parameters.resolve();publish_dm_wake_force(json.loads(source.read_text()),args.output);return 0


def black_hole_remnant_fit(mass1_msun,mass2_msun,spin1,spin2,orbital_axis,*,recoil_phase,orbital_radial_axis,
    merger_separation_gm_c2=10.):
    """Near-merger NR fits adapted from pinned PRECESSION (MIT, Gerosa).

    Mass: Barausse/Morozova/Rezzolla 2012; spin: HBR16_34corr; recoil:
    Kesden/Lousto/Zlochower coefficients. Inputs must already be transported
    through a supported PN inspiral to the declared near-merger separation.
    No physical merger is inferred by this fit evaluator.
    """
    import numpy as np
    m1=finite_number(mass1_msun,'mass1',strict=True);m2=finite_number(mass2_msun,'mass2',strict=True)
    a1=np.asarray(spin1,dtype=float);a2=np.asarray(spin2,dtype=float);axis=np.asarray(orbital_axis,dtype=float)
    if any(x.shape!=(3,) or not np.isfinite(x).all() for x in (a1,a2,axis)):raise ValueError('Finite spin/orbit vectors required')
    n=float(np.linalg.norm(axis))
    if n==0:raise ValueError('A remnant fit needs a measured orbital axis')
    axis=axis/n;phase=finite_number(recoil_phase,'recoil phase');r=finite_number(merger_separation_gm_c2,'fit separation',strict=True)
    if not 8<=r<=12:raise ValueError('Declare spins transported to a supported near-merger fit reference')
    swapped=m2>m1
    if swapped:m1,m2=m2,m1;a1,a2=a2,a1
    q=m2/m1;chi1=float(np.linalg.norm(a1));chi2=float(np.linalg.norm(a2));eta=q/(1+q)**2
    if q<.05 or max(chi1,chi2)>.99:raise ValueError('Binary outside declared analytic-fit implementation domain')
    c1=float(np.dot(a1,axis))/chi1 if chi1 else 1.;c2=float(np.dot(a2,axis))/chi2 if chi2 else 1.
    theta1=math.acos(float(np.clip(c1,-1,1)));theta2=math.acos(float(np.clip(c2,-1,1)))
    angle12=math.acos(float(np.clip(np.dot(a1,a2)/(chi1*chi2),-1,1))) if chi1*chi2 else 0.
    def isco(spin):
        if abs(spin)>1:raise ValueError('Fit effective spin exceeds Kerr domain')
        z1=1+np.cbrt(1-spin*spin)*(np.cbrt(1+spin)+np.cbrt(1-spin));z2=math.sqrt(3*spin*spin+z1*z1)
        radius=3+z2-np.sign(spin)*math.sqrt(max(0.,(3-z1)*(3+z1+2*z2)))
        energy=math.sqrt(1-2/(3*radius));angular=2/(3*math.sqrt(3))*(1+2*math.sqrt(3*radius-2))
        return energy,angular
    atot=(np.dot(a1,axis)+q*q*np.dot(a2,axis))/(1+q)**2;eisco,_=isco(float(atot))
    radiated=eta*(1-eisco)+4*eta*eta*(4*.04827+16*.01707*atot*(atot+1)+eisco-1)
    corrected1=theta1+.024*math.sin(theta1);corrected2=theta2+.024*math.sin(theta2)
    atot=(chi1*math.cos(corrected1)+q*q*chi2*math.cos(corrected2))/(1+q)**2
    aeff=atot+.474046*eta*(chi1*math.cos(corrected1)+chi2*math.cos(corrected2))
    eisco,lisco=isco(aeff)
    k=np.array([[0.,3.39221,4.48865,-5.77101,-13.0459],[35.1278,-72.9336,-86.0036,93.7371,200.975],
        [-146.822,387.184,447.009,-467.383,-884.339],[223.911,-648.502,-697.177,753.738,1166.89]])
    k[0,0]=16*(.68646-sum(k[i,0]/4**(i+2) for i in range(1,4))-math.sqrt(3)/2)
    ell=abs(lisco-2*atot*(eisco-1)+sum(k[i,j]*eta**(i+1)*aeff**j for i in range(4) for j in range(5)))
    spin_squared=(chi1**2+chi2**2*q**4+2*chi1*chi2*q*q*math.cos(angle12)+
        2*(chi1*math.cos(corrected1)+chi2*q*q*math.cos(corrected2))*ell*q+(ell*q)**2)/(1+q)**4
    spin_magnitude=math.sqrt(max(0.,spin_squared))
    jtotal=eta*math.sqrt(r)*axis+(a1+q*q*a2)/(1+q)**2;jnorm=float(np.linalg.norm(jtotal))
    if jnorm==0 or not 0<radiated<1 or not 0<=spin_magnitude<=1:raise ValueError('NR fit produced an unsupported physical remnant')
    delta=(a1-q*a2)/(1+q);chit=(a1+q*q*a2)/(1+q)**2
    dpar=float(np.dot(delta,axis));dper=float(np.linalg.norm(np.cross(delta,axis)));spar=float(np.dot(chit,axis));sper=float(np.linalg.norm(np.cross(chit,axis)))
    vm=1.2e4*eta*eta*(1-.93*eta)*(1-q)/(1+q);vper=6.9e3*eta*eta*dpar
    vpar=16*eta*eta*(dper*(3677.76+2*2481.21*spar+4*1792.45*spar*spar+8*1506.52*spar**3)+sper*dpar*(2*1140+4*2481*spar))*math.cos(phase)
    reference=np.asarray(orbital_radial_axis,dtype=float)
    if reference.shape!=(3,) or not np.isfinite(reference).all():raise ValueError("Measured coorbital radial direction required")
    if swapped:reference=-reference
    ex=reference-axis*np.dot(reference,axis)
    if np.linalg.norm(ex)<=0:raise ValueError("Coorbital radial direction is parallel to orbital axis")
    ex/=np.linalg.norm(ex);ey=np.cross(axis,ex)
    kick=(vm+vper*math.cos(math.radians(145)))*ex+vper*math.sin(math.radians(145))*ey+vpar*axis
    mass=(m1+m2)*(1-radiated)
    return {'remnant_mass_msun':mass,'radiated_mass_msun':(m1+m2)-mass,'radiated_energy_erg':((m1+m2)-mass)*MSUN_G*C_CGS**2,
        'remnant_spin':(spin_magnitude*jtotal/jnorm).tolist(),'recoil_km_s':kick.tolist(),
        'fit_reference_separation_gm_c2':r,'physical_acceptance':'NOT_ACCEPTED'}

import json
import os
from pathlib import Path
import tempfile

import h5py
import numpy as np
import yaml
from core.data import hdf5_scalar_attr

FIELDS = {
    "BH_Mass": ("subgrid_mass_msun", "mass"),
    "Masses": ("gravity_mass_msun", "mass"),
    "BH_InflowRate": ("inflow_msun_year", "rate"),
    "BH_RadiatedMass": ("radiated_mass_msun", "mass"),
    "BH_FeedbackEnergy": ("coupled_energy_erg", "energy"),
}


class NoBlackHoleData(ValueError):
    """A validated enabled run formed no BHs in its saved outputs."""


def measured_epoch(paths, snapshot, settings):
    from analysis import validate_gadget_snapshot_set
    from halos import _unit_conversion
    from core.foundation import file_sha256 as digest
    from core.data import exact_uint64_ids, native_unit_factor

    sources = [{'path':str(Path(p).resolve()),'sha256':digest(p)} for p in paths]
    validate_gadget_snapshot_set(paths, snapshot)
    _unit_conversion(paths)
    data = {key: [] for key in ("ParticleIDs", "Coordinates", "Velocities", *FIELDS)}
    epoch = None
    for path in paths:
        with h5py.File(path, "r") as f:
            header, params = f["Header"].attrs, f["Parameters"].attrs
            a, h = (
                hdf5_scalar_attr(header["Time"], "Time"),
                float(params["HubbleParam"]),
            )
            current = (a, h)
            if epoch is not None and current != epoch:
                raise ValueError("BH snapshot epochs/h disagree across shards")
            epoch = current
            for native, requested in native_parameters(settings).items():
                if isinstance(requested,str):
                    actual=params.get(native);actual=actual.decode("ascii") if isinstance(actual,bytes) else actual
                    if actual!=requested:raise ValueError("BH exact string runtime parameter differs: "+native)
                    continue
                if native not in params or not math.isclose(
                    float(params[native]), float(requested), rel_tol=1e-12, abs_tol=0
                ):
                    raise ValueError(
                        "BH snapshot runtime choices disagree with config: " + native
                    )
            stars = f.get("PartType5")
            if stars is None:
                continue
            ids = exact_uint64_ids(stars["ParticleIDs"][:],unique=False)
            n = len(ids)
            for key in data:
                if key not in stars:
                    raise ValueError(
                        "BH physics diagnostics require native field " + key
                    )
                values = np.asarray(stars[key][:])
                if values.shape != (
                    (n, 3) if key in {"Coordinates", "Velocities"} else (n,)
                ):
                    raise ValueError(
                        "BH field shape does not align with ParticleIDs: " + key
                    )
                if key != "ParticleIDs" and (
                    values.dtype.kind not in "fiu" or np.any(~np.isfinite(values))
                ):
                    raise ValueError("BH fields must be numeric and finite")
                if key in FIELDS:
                    attrs = stars[key].attrs
                    native_kind = FIELDS[key][1]
                    expected_h = 0.0 if native_kind == "rate" else -1.0
                    if not all(
                        k in attrs for k in ("to_cgs", "a_scaling", "h_scaling")
                    ):
                        raise ValueError(
                            "BH field lacks explicit physical-unit metadata: " + key
                        )
                    if (
                        float(attrs["a_scaling"]) != 0
                        or float(attrs["h_scaling"]) != expected_h
                    ):
                        raise ValueError("Unsupported BH unit convention: " + key)
                    cgs = float(attrs["to_cgs"])
                    if not math.isfinite(cgs) or cgs <= 0:
                        raise ValueError(
                            "BH unit conversion must be positive and finite"
                        )
                    values = values * cgs * (h**expected_h)
                    if native_kind == "mass":
                        values = values / MSUN_G
                    if native_kind == "rate":
                        values = values * SECONDS_PER_YEAR / MSUN_G
                    if np.any(values < 0):
                        raise ValueError(
                            "BH masses, inflows and cumulative energy must be nonnegative"
                        )
                if key == "Coordinates":
                    values = values * native_unit_factor('position_proper_kpc',scale_factor=a,hubble_param=h)
                if key == "Velocities":
                    values = values * native_unit_factor('particle_velocity_peculiar_kms',scale_factor=a,hubble_param=h)
                if key == "ParticleIDs":
                    values = ids
                data[key].append(values)
    if epoch is None:
        raise ValueError("No BH snapshot files")
    result = {
        key: np.concatenate(chunks)
        if chunks
        else np.empty(
            (0, 3) if key in {"Coordinates", "Velocities"} else (0,),
            dtype=np.uint64 if key == "ParticleIDs" else float,
        )
        for key, chunks in data.items()
    }
    live=~((result["ParticleIDs"]==0)&(result["Masses"]==0))
    result={key:values[live] for key,values in result.items()}
    ids = result["ParticleIDs"]
    if len(np.unique(ids)) != len(ids):
        raise ValueError("Duplicated BH IDs across snapshot shards")
    if np.any(result["BH_Mass"] <= 0) or np.any(result["Masses"] < result["BH_Mass"]):
        raise ValueError(
            "BH subgrid mass must be positive and fit within its dynamical mass"
        )
    for entry in sources:
        if digest(entry['path']) != entry['sha256']:
            raise ValueError('BH snapshot changed during measurement: '+entry['path'])
    return result, {
        "snapshot": snapshot,
        "scale_factor": epoch[0],
        "redshift": 1 / epoch[0] - 1,
        "h": epoch[1],
        "source_files": [str(p) for p in paths],
        "source_sha256": sources,
    }


from core.native import native_run_owner, directory_owner


@native_run_owner
@directory_owner('output_directory',default_relative='results/black_holes')
def analyze_black_holes(run_dir, *, output_directory=None, memory_gib=None):
    """Publish measured type-5 histories from complete saved native epochs.

    Convert dataset units before comparing subgrid/gravity masses or source
    increments. Luminosity is the selected efficiency times saved mean inflow;
    it is a model quantity, not an observed disk spectrum. Existing destinations
    are preserved, and row/workspace limits are checked before bulk intake.
    """
    from analysis import snapshot_ids, find_hdf5_files
    from black_holes import NATIVE_PARAMETERS
    from core.data import run_epoch_contract, coordinate_contract, write_dataset_contract, exact_uint64_ids
    from core.foundation import product_provenance, publish_directory, file_sha256 as digest, verify_product_provenance

    run_dir = Path(run_dir).resolve()
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    from core.resources import memory_limit_gb
    from core.data import snapshot_total_counts

    budget = memory_limit_gb(cfg) if memory_gib is None else memory_gib
    if not math.isfinite(budget) or budget <= 0:
        raise ValueError("BH history memory budget must be finite and positive")
    settings = cfg.get("black_holes", {"enabled": False})
    parameters = native_parameters(settings)
    if not parameters:
        raise ValueError(
            "BH growth analysis requires an enabled native model and matching snapshot fields"
        )
    snapshots = snapshot_ids(run_dir)
    if not snapshots:
        raise ValueError("BH analysis needs completed snapshots")
    total_rows = 0
    for snapshot in snapshots:
        paths = find_hdf5_files(run_dir / "data", f"snapshot_{snapshot:03d}")
        if not paths:
            raise ValueError("BH history requires complete saved snapshots")
        with h5py.File(paths[0]) as source:
            total_rows += int(snapshot_total_counts(source["Header"].attrs)[5])
    # Dict histories, plotting, JSON serialization and per-epoch arrays all
    # consume memory; reject large requests before allocating any BH payload.
    if 4096 * total_rows + 64 * 1024**2 > budget * 1024**3:
        raise MemoryError("BH history exceeds declared analysis memory budget")
    destination = (
        Path(output_directory).resolve()
        if output_directory
        else run_dir / "results" / "black_holes"
    )
    if destination.exists():
        raise FileExistsError("BH result already exists: " + str(destination))
    destination.parent.mkdir(parents=True, exist_ok=True)
    model = ThermalModel(
        *(
            settings[key]
            for key in (
                "radiative_efficiency",
                "feedback_coupling",
                "bondi_boost",
                "eddington_factor",
            )
        )
    )
    record = {
        "schema": "darkmattersim-native-bh-history-v1",
        'science_acceptance':'NOT_ACCEPTED',
        "parameters": settings,
        "epochs": [],
        "history": [],
        "source_files": [],
        "interpretation": "Measured source-step mean inflow; subgrid luminosity L=epsilon_r*inflow*c^2. Gravity mass includes the unresolved seed reservoir. No resolved disk spectrum, horizon, or intrinsic emitted-light image.",
        "units": {
            "Coordinates": "wrapped proper kpc",
            "Velocities": "peculiar km/s",
            "mass": "Msun",
            "inflow": "Msun/year",
            "luminosity": "erg/s",
            "energy": "erg",
        },
        "references": ["https://arxiv.org/abs/astro-ph/0411108"],
    }
    # Runtime choices are all checked; import helps make the schema's supported
    # native parameter set auditable against the solver registration.
    record["native_parameter_names"] = list(NATIVE_PARAMETERS.values()) + [
        "BHSeedingMode"
    ]
    previous, previous_a = {}, -1
    stage=Path(tempfile.mkdtemp(prefix='.bh-analysis-',dir=destination.parent))
    from core.foundation import atomic_json
    try:
        with h5py.File(stage / "history.hdf5", "w") as output:
            for snapshot in snapshots:
                paths = find_hdf5_files(run_dir / "data", f"snapshot_{snapshot:03d}")
                data, meta = measured_epoch(paths, snapshot, settings)
                data['ParticleIDs'] = exact_uint64_ids(data['ParticleIDs'])
                meta['epoch_contract'] = run_epoch_contract(run_dir,int(snapshot),meta['scale_factor'])
                if meta["scale_factor"] <= previous_a:
                    raise ValueError("BH saved epochs do not increase in time")
                previous_a = meta["scale_factor"]
                record["epochs"].append(meta)
                record["source_files"].extend(meta["source_files"])
                epoch = output.create_group(f"Epochs/{snapshot:03d}")
                epoch.attrs['metadata_json'] = json.dumps(meta,allow_nan=False)
                for key, values in data.items():
                    epoch.create_dataset(key,data=values,chunks=True,fletcher32=True)
                    units = ('identity' if key=='ParticleIDs' else 'proper kpc' if key=='Coordinates'
                             else 'km/s' if key=='Velocities' else 'Msun/year' if key=='BH_InflowRate'
                             else 'erg' if key=='BH_FeedbackEnergy' else 'Msun')
                    write_dataset_contract(epoch[key],units=units,epoch=meta['epoch_contract'],
                        provenance={'source_sha256':meta['source_sha256']},
                        quantity='saved native black-hole '+key+'; conversion declared in measured epoch')
                for i, particle_id in enumerate(data["ParticleIDs"]):
                    item = {
                        "snapshot": snapshot,
                        "scale_factor": meta["scale_factor"],
                        "particle_id": int(particle_id),
                    }
                    item.update(
                        {
                            label: float(data[key][i])
                            for key, (label, _) in FIELDS.items()
                        }
                    )
                    item["coordinates_proper_kpc"] = data["Coordinates"][i].tolist()
                    item["peculiar_velocity_km_s"] = data["Velocities"][i].tolist()
                    luminosity = (
                        settings["radiative_efficiency"]
                        * item["inflow_msun_year"]
                        * MSUN_G
                        / SECONDS_PER_YEAR
                        * (C_SI * 100) ** 2
                    )
                    item["bolometric_luminosity_erg_s"] = luminosity
                    item["thermal_power_erg_s"] = (
                        settings["feedback_coupling"] * luminosity
                    )
                    # Eddington ratio uses the mass at this saved output;
                    # inflow is the preceding source-step average.
                    cap = accretion_rates(
                        item["subgrid_mass_msun"], 0.0, 1.0, 0.0, model
                    )["eddington_inflow_msun_year"]
                    item["mean_inflow_over_saved_mass_eddington_cap"] = (
                        item["inflow_msun_year"] / cap
                    )
                    old = previous.get(int(particle_id))
                    if old is not None:
                        for field in (
                            "subgrid_mass_msun",
                            "radiated_mass_msun",
                            "coupled_energy_erg",
                        ):
                            if item[field] < old[field] * (1 - 1e-12):
                                raise ValueError(
                                    "BH monotonic field decreases without a declared event: "
                                    + field
                                )
                    previous[int(particle_id)] = item
                    record["history"].append(item)
            output.create_dataset(
                "ProvenanceJSON",
                data=json.dumps(
                    {key: value for key, value in record.items() if key != "history"},
                    allow_nan=False,
                ),
                dtype=h5py.string_dtype("utf-8"),
            )
            output.attrs["provenance_dataset"] = "ProvenanceJSON"
        if not record["history"]:
            raise NoBlackHoleData(
                "No BHs formed in the saved outputs; no growth figure can be produced"
            )
        from evolution import history_cadence_support
        record['event_cadence'] = history_cadence_support(record['epochs'])
        (stage / "provenance.json").write_text(
            json.dumps(record, indent=2, allow_nan=False) + "\n"
        )
        record['data_contract'] = product_provenance(run_dir,algorithm=__file__,
            selection='All actual saved native black-hole particles; no merger ancestry invented',
            source_files=[entry for epoch in record['epochs'] for entry in epoch['source_sha256']],coordinates=coordinate_contract(position_units='proper kpc',
                origin='global simulation Cartesian box origin',periodic_wrapping='wrapped periodic coordinates',
                velocity_convention='peculiar km/s'),helper_sources=[Path(__file__).with_name('evolution.py')])
        with h5py.File(stage/'history.hdf5','r+') as output:
            output.attrs['data_contract_json'] = json.dumps(record['data_contract'],allow_nan=False)
            namespace = record['data_contract']['identity_namespaces']['black_hole']
            for epoch in output['Epochs'].values():
                epoch['ParticleIDs'].attrs['identity_namespace'] = namespace
                for key in ('Coordinates','Velocities'):
                    epoch[key].attrs['coordinate_conventions_json'] = json.dumps(record['data_contract']['coordinates'])
                    epoch[key].attrs['coordinates_status'] = 'PRODUCER_DECLARED'
        (stage/'data_contract.json').write_text(json.dumps(record['data_contract'],indent=2)+'\n')
        record['history_sha256'] = digest(stage/'history.hdf5')
        (stage/'provenance.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
        (stage/'provenance.sha256').write_text(digest(stage/'provenance.json')+'\n')
        bh_figure(stage,record)
        verify_product_provenance(run_dir,record['data_contract'])
        publish_directory(stage, destination)
    except BaseException as exc:
        if stage.exists():atomic_json(stage/'export_failure.json',{'status':'FAIL','science_acceptance':'NOT_ACCEPTED',
            'failure':{'type':type(exc).__name__,'message':str(exc)}})
        raise
    return {
        "directory": str(destination),
        "catalogue": str(destination / "history.hdf5"),
        "figure": str(destination / "black_hole_histories.png"),
        "black_holes": len(previous),
        "snapshots": len(snapshots),
    }


def bh_figure(directory, record):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from core.data import verify_hdf5_dataset_payloads
    verify_hdf5_dataset_payloads(Path(directory)/'history.hdf5')

    ids = sorted({row["particle_id"] for row in record["history"]})
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    fields = (
        "subgrid_mass_msun",
        "inflow_msun_year",
        "bolometric_luminosity_erg_s",
        "gravity_mass_msun",
        "mean_inflow_over_saved_mass_eddington_cap",
        "coupled_energy_erg",
    )
    labels = (
        "BH subgrid mass [Msun]",
        "Mean inflow [Msun/year]",
        "Model bolometric luminosity [erg/s]",
        "Gravity mass incl. seed reservoir [Msun]",
        "Mean inflow / saved-mass Eddington cap",
        "Cumulative coupled energy [erg]",
    )
    # Display up to 12 most massive final systems; the full catalogue survives.
    final = {
        particle_id: max(
            row["subgrid_mass_msun"]
            for row in record["history"]
            if row["particle_id"] == particle_id
        )
        for particle_id in ids
    }
    shown = sorted(ids, key=lambda particle_id: final[particle_id], reverse=True)[:12]
    for index, particle_id in enumerate(shown):
        rows = [row for row in record["history"] if row["particle_id"] == particle_id]
        for axis, field, label in zip(axes.flat, fields, labels):
            axis.plot(
                [r["scale_factor"] for r in rows],
                [r[field] for r in rows],
                "o-",
                ms=3,
                label=f"BH {index + 1}",
            )
            axis.set(xlabel="Scale factor a", ylabel=label)
            axis.ticklabel_format(axis="x", style="plain", useOffset=False)
            axis.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(
        "Native black-hole histories • live gas accretion and thermal feedback"
    )
    from figures import export_figure
    export_figure(fig,Path(directory)/'black_hole_histories',formats=('png','pdf'),dpi=200,close=False,
        metadata={'science_acceptance':'NOT_ACCEPTED','source_sha256':record['history_sha256'],
            'epochs':[epoch['epoch_contract'] for epoch in record['epochs']],
            'data_contract':record['data_contract'],'x_units':'scale factor',
            'y_units':['Msun','Msun/year','erg/s','Msun','dimensionless','erg']})
    plt.close(fig)

# ========================================================================
# schwarzschild_lensing

# Workflow and units
# Maps a user-supplied equirectangular panorama through an isolated Schwarzschild vacuum geometry.
# Observer and photon impact radii are in r_s=2GM/c^2; black-hole mass is solar masses and the exported r_s is km.
# Null-geodesic quadrature handles capture/turning points explicitly. This separate rendering calculation supplies neither Kerr spin nor an accretion/emission model.
# Bundled source patches: core/patches. Runtime choices are saved with each run.

import argparse
import hashlib
import sys
from typing import Callable
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))

G_SI = G_CGS*1e-3  # cm^3/g/s^2 -> m^3/kg/s^2.
M_SUN_KG = MSUN_G*1e-3  # grams -> kilograms; share the simulation mass convention.
SCHWARZSCHILD_RS_PER_SOLAR_MASS_KM = 2.0 * G_SI * M_SUN_KG / (C_SI * C_SI) / 1000.0
CRITICAL_IMPACT_RS = 3.0 * math.sqrt(3.0) / 2.0
MODEL_NAME = "Schwarzschild vacuum null geodesics; non-rotating isolated black hole"


def schwarzschild_radius_km(mass_solar: float) -> float:
    """Return r_s=2GM/c^2 in km for a positive mass in solar masses."""
    if not math.isfinite(mass_solar) or mass_solar <= 0.0:
        raise ValueError("Black-hole mass must be finite and positive")
    return SCHWARZSCHILD_RS_PER_SOLAR_MASS_KM * mass_solar


def apparent_shadow_radius_rad(observer_rs: float) -> float:
    """Angular shadow radius for a static observer at finite Schwarzschild r."""
    if not math.isfinite(observer_rs) or observer_rs <= 1.5:
        raise ValueError("Static-observer radius must be outside the photon sphere (r > 1.5 r_s)")
    lapse = 1.0 - 1.0 / observer_rs
    return math.asin(min(1.0, CRITICAL_IMPACT_RS * math.sqrt(lapse) / observer_rs))


def _adaptive_simpson(function: Callable[[float], float], left: float, right: float,
                      tolerance: float = 2.0e-9, max_depth: int = 18) -> float:
    """Adaptive Simpson quadrature on a finite, smooth integration interval."""
    if right <= left:
        return 0.0
    middle = 0.5 * (left + right)
    f_left, f_middle, f_right = function(left), function(middle), function(right)
    whole = (right - left) * (f_left + 4.0 * f_middle + f_right) / 6.0

    def refine(a, b, fa, fm, fb, estimate, tol, depth):
        m = 0.5 * (a + b)
        lm, rm = 0.5 * (a + m), 0.5 * (m + b)
        flm, frm = function(lm), function(rm)
        left_estimate = (m - a) * (fa + 4.0 * flm + fm) / 6.0
        right_estimate = (b - m) * (fm + 4.0 * frm + fb) / 6.0
        delta = left_estimate + right_estimate - estimate
        if not math.isfinite(delta):
            raise ArithmeticError("Nonfinite geodesic quadrature")
        if abs(delta) <= 15.0 * tol:
            return left_estimate + right_estimate + delta / 15.0
        if depth <= 0:
            raise ArithmeticError("Geodesic quadrature did not converge at the requested tolerance")
        return (refine(a, m, fa, flm, fm, left_estimate, tol / 2.0, depth - 1)
                + refine(m, b, fm, frm, fb, right_estimate, tol / 2.0, depth - 1))

    return refine(left, right, f_left, f_middle, f_right, whole, tolerance, max_depth)


def _turning_radius_rs(impact_rs: float) -> float:
    """Outer radial turning point for a scattered photon, in units of r_s."""
    if impact_rs <= CRITICAL_IMPACT_RS:
        raise ValueError("A captured ray has no exterior turning point")
    low = 1.5
    high = max(impact_rs + 2.0, 3.0)

    def impact_at(radius):
        return radius / math.sqrt(1.0 - 1.0 / radius)

    while impact_at(high) < impact_rs:
        high *= 2.0
    for _ in range(80):
        middle = 0.5 * (low + high)
        if impact_at(middle) < impact_rs:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


def _turning_sweep_integral(impact_rs: float, turning_rs: float, t_max: float) -> float:
    """Azimuth swept from r_min to r_min/(1-t_max^2), with the root desingularized."""
    if t_max <= 0.0:
        return 0.0
    ratio = 1.0 / turning_rs
    amplitude = (impact_rs / turning_rs) ** 2

    def integrand(t):
        if t <= 1.0e-14:
            return 2.0 / math.sqrt(max(amplitude * (2.0 - 3.0 * ratio), 1.0e-300))
        t2 = t * t
        reduced = amplitude * ((2.0 - 3.0 * ratio) + (-1.0 + 3.0 * ratio) * t2 - ratio * t2 * t2)
        if reduced <= 0.0:
            raise ArithmeticError("Scattered-ray radial potential is not positive")
        return 2.0 / math.sqrt(reduced)

    # Near the critical impact parameter, most of the logarithmic winding is
    # concentrated at t ~ sqrt(2 - 3 r_s/r_min). Split there so quadrature does
    # not smooth away higher-order photon-ring images.
    scale = math.sqrt(max(2.0 - 3.0 * ratio, 1.0e-16))
    cuts = sorted({0.0, t_max, min(t_max, scale), min(t_max, 8.0 * scale),
                   min(t_max, 64.0 * scale)})
    integral = sum(
        _adaptive_simpson(integrand, left, right, tolerance=5.0e-9 / max(1, len(cuts) - 1))
        for left, right in zip(cuts[:-1], cuts[1:]) if right > left
    )
    return (impact_rs / turning_rs) * integral


def _outgoing_sweep_integral(impact_rs: float, observer_rs: float) -> float:
    """Azimuth swept by an initially outgoing ray from the observer to infinity."""
    amplitude = (impact_rs / observer_rs) ** 2
    ratio = 1.0 / observer_rs

    def integrand(t):
        if t <= 1.0e-14:
            if abs(1.0 - amplitude * (1.0 - ratio)) < 1.0e-10:
                return 2.0 / math.sqrt(max(amplitude * (2.0 - 3.0 * ratio), 1.0e-300))
            return 0.0
        x = 1.0 - t * t
        radicand = 1.0 - amplitude * x * x * (1.0 - ratio * x)
        if radicand <= 0.0:
            return 0.0
        return 2.0 * t / math.sqrt(radicand)

    return (impact_rs / observer_rs) * _adaptive_simpson(integrand, 0.0, 1.0)


def _source_angle(theta_rad: float, observer_rs: float) -> float | None:
    """Map one local viewing angle to its asymptotic sky angle.

    ``theta_rad`` is measured from the observer's inward radial direction.
    The return value is an *unwrapped* signed angle; sine/cosine map multiple
    winding images continuously onto the celestial sphere. ``None`` means
    that the ray crosses the capture boundary.
    """
    lapse = 1.0 - 1.0 / observer_rs
    impact_rs = observer_rs * math.sin(theta_rad) / math.sqrt(lapse)
    if theta_rad < math.pi / 2.0:
        if impact_rs <= CRITICAL_IMPACT_RS:
            return None
        turning = _turning_radius_rs(impact_rs)
        to_infinity = _turning_sweep_integral(impact_rs, turning, 1.0)
        to_observer = _turning_sweep_integral(
            impact_rs, turning, math.sqrt(max(0.0, 1.0 - turning / observer_rs)),
        )
        total_sweep = to_infinity + to_observer
    else:
        total_sweep = _outgoing_sweep_integral(impact_rs, observer_rs)
    return math.pi - total_sweep




def _sha256(path: Path) -> str:
    """Hash a source/input asset in streaming blocks for reproducible provenance."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _build_radial_map(observer_rs: float, theta_max: float, count: int):
    """Sample the azimuthal map densely near the logarithmic photon-ring limit."""
    theta_critical = math.asin(min(
        1.0, CRITICAL_IMPACT_RS * math.sqrt(1.0 - 1.0 / observer_rs) / observer_rs,
    ))
    if theta_max <= theta_critical:
        return np.asarray([theta_critical]), np.asarray([np.nan]), theta_critical
    span = theta_max - theta_critical
    minimum_offset = max(1.0e-13, span * 1.0e-10)
    offsets = np.geomspace(minimum_offset, span, count)
    angles = theta_critical + offsets
    mapped = np.empty(len(angles), dtype=np.float64)
    for index, angle in enumerate(angles):
        value = _source_angle(float(angle), observer_rs)
        mapped[index] = np.nan if value is None else value
    return angles, mapped, theta_critical


def _sample_panorama(panorama: np.ndarray, longitude: np.ndarray, latitude: np.ndarray) -> np.ndarray:
    """Bilinearly sample an equirectangular panorama with wrapped longitude."""
    height, width, _ = panorama.shape
    x = np.mod((longitude / (2.0 * math.pi) + 0.5) * width - 0.5, width)
    y = np.clip((0.5 - latitude / math.pi) * height - 0.5, 0.0, height - 1.0)
    x0f, y0f = np.floor(x), np.floor(y)
    x0, y0 = x0f.astype(np.int64), y0f.astype(np.int64)
    x1, y1 = (x0 + 1) % width, np.minimum(y0 + 1, height - 1)
    dx, dy = (x - x0f)[..., None], (y - y0f)[..., None]
    a = panorama[y0, x0] * (1.0 - dx) + panorama[y0, x1] * dx
    b = panorama[y1, x0] * (1.0 - dx) + panorama[y1, x1] * dx
    return a * (1.0 - dy) + b * dy


def raytrace_panorama(panorama: Image.Image, *, observer_rs: float, fov_deg: float,
                      width: int, height: int, radial_samples: int | None = None,
                      roll_deg: float = 0.0, return_arrays: bool = False):
    """Render a perspective image by tracing camera rays through Schwarzschild space."""
    if not math.isfinite(observer_rs) or observer_rs <= 1.5:
        raise ValueError("Observer radius must be outside the photon sphere (r > 1.5 r_s)")
    if not math.isfinite(fov_deg) or not 1.0 <= fov_deg <= 120.0:
        raise ValueError("Horizontal field of view must be between 1 and 120 degrees")
    if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) for value in (width, height)):
        raise ValueError("Image dimensions must be integer pixel counts")
    if width < 128 or height < 128 or width > 8192 or height > 8192:
        raise ValueError("Image dimensions must each be between 128 and 8192 pixels")
    if width * height > 8_000_000:
        raise ValueError("Image area is limited to 8 million pixels to bound ray-map memory use")
    if not math.isfinite(roll_deg):
        raise ValueError("Panorama roll must be finite")
    if panorama.width < 2 or panorama.height < 1 or not math.isclose(
            panorama.width / panorama.height, 2.0, rel_tol=0.0, abs_tol=0.02):
        raise ValueError("Source must be a full 2:1 equirectangular panorama centered on the black-hole view axis")

    half_width = math.tan(math.radians(fov_deg) / 2.0)
    half_height = half_width * height / width
    max_theta = math.atan(math.hypot(half_width, half_height))
    count = radial_samples if radial_samples is not None else max(4097, min(32769, max(width, height) * 4 + 1))
    if isinstance(count, (bool, np.bool_)) or not isinstance(count, (int, np.integer)):
        raise ValueError("Radial sample count must be an integer")
    if count < 1024 or count > 65537:
        raise ValueError("Radial sample count must be between 1024 and 65537")
    angle_grid, source_grid, theta_critical = _build_radial_map(observer_rs, max_theta, count)

    x = ((np.arange(width, dtype=np.float64) + 0.5) / width * 2.0 - 1.0) * half_width
    y = (1.0 - (np.arange(height, dtype=np.float64) + 0.5) / height * 2.0) * half_height
    xx, yy = np.meshgrid(x, y)
    theta = np.arctan(np.hypot(xx, yy))
    azimuth = np.arctan2(yy, xx)
    captured = theta <= theta_critical
    escaped = ~captured
    # Captured rays have no sky direction. Do not send NaN angles into image
    # indexing, especially when the entire frame lies inside the shadow.
    # Conversely, a non-finite escaped ray is a numerical failure, not a
    # physically justified black pixel or an enlarged shadow.
    source_pixels = np.asarray(panorama.convert("RGB"), dtype=np.float32)
    sampled = np.zeros((height, width, 3), dtype=np.float32)
    if np.any(escaped):
        beta = np.interp(theta[escaped], angle_grid, source_grid)
        if np.any(~np.isfinite(beta)):
            raise ArithmeticError("An escaped ray has no finite Schwarzschild sky mapping")
        transverse = np.sin(beta)
        direction_x = np.cos(beta)
        direction_y = transverse*np.cos(azimuth[escaped])
        direction_z = transverse*np.sin(azimuth[escaped])
        longitude = np.arctan2(direction_y, direction_x)+math.radians(roll_deg)
        latitude = np.arcsin(np.clip(direction_z, -1., 1.))
        sampled[escaped] = _sample_panorama(source_pixels, longitude, latitude)
    result = Image.fromarray(np.clip(np.rint(sampled), 0, 255).astype(np.uint8), mode="RGB")
    metadata = {
        "model": MODEL_NAME,
        "method": "Schwarzschild null-geodesic first integrals; adaptive Simpson quadrature; static finite-radius observer; asymptotic celestial-sphere source",
        "assumptions": [
            "Spherical, vacuum Schwarzschild exterior with zero spin and no external matter in the metric.",
            "Observer is static and outside the photon sphere; the panorama is treated as a distant celestial sphere.",
            "This is gravitational ray mapping only: no emission, radiative transfer, Doppler shift, plasma, or accretion-flow model.",
            "The input panorama is visual source data; lensing does not make it a GADGET simulation product.",
        ],
        "observer_radius_r_s": float(observer_rs),
        "panorama_projection": "2:1 equirectangular; longitude 0 and latitude 0 define the unlensed optical axis",
        "horizontal_fov_deg": float(fov_deg),
        "image_size_px": [int(width), int(height)],
        "radial_map_samples": int(len(angle_grid)),
        "critical_impact_parameter_r_s": float(CRITICAL_IMPACT_RS),
        "photon_sphere_radius_r_s": 1.5,
        "shadow_radius_at_observer_rad": float(theta_critical),
        "shadow_diameter_at_observer_deg": float(2.0 * math.degrees(theta_critical)),
        "captured_rays_rendered": "black",
        "captured_pixel_count": int(np.count_nonzero(captured)),
        "escaped_pixel_count": int(np.count_nonzero(escaped)),
        "numerical_failure_policy": "non-finite escaped rays raise an error; they never masquerade as shadow pixels",
        "shadow_reference": "Perlick & Tsupko, Physics Reports 947 (2022), arXiv:2105.07101",
        "background_roll_deg": float(roll_deg),
    }
    if return_arrays:
        return result, metadata, {"observer_angle_rad": angle_grid, "source_angle_rad": source_grid,
            "screen_angle_rad": theta, "screen_azimuth_rad": azimuth, "captured": captured,
            "source_rgb": np.asarray(panorama.convert("RGB")), "output_rgb": np.asarray(result)}
    return result, metadata


def main() -> None:
    """Parse command arguments and dispatch this module's documented workflow."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panorama", type=Path, required=True, help="equirectangular all-sky source image")
    parser.add_argument("--output", type=Path, default=Path('black_holes.png'))
    parser.add_argument("--mass-solar", type=float, required=True, help="black-hole mass in solar masses")
    parser.add_argument("--observer-rs", type=float, default=50.0, help="static observer radius in Schwarzschild radii")
    parser.add_argument("--fov-deg", type=float, default=60.0, help="horizontal perspective field of view")
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument("--radial-samples", type=int, help="more samples resolve photon-ring winding more finely")
    parser.add_argument("--roll-deg", type=float, default=0.0, help="rotate the panorama horizontally")
    args = parser.parse_args()

    rs_km = schwarzschild_radius_km(args.mass_solar)
    apparent_shadow_radius_rad(args.observer_rs)
    panorama_path = args.panorama.expanduser().resolve()
    if not panorama_path.is_file():
        raise FileNotFoundError("Panorama image does not exist: " + str(panorama_path))
    panorama_before = _sha256(panorama_path)
    with Image.open(panorama_path) as opened:
        panorama = opened.convert("RGB")
    source_record = {
        "kind": "user-supplied equirectangular panorama",
        "path": str(panorama_path),
        "sha256": panorama_before,
        "size_px": [int(panorama.width), int(panorama.height)],
    }

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    image, record, arrays = raytrace_panorama(
        panorama, observer_rs=float(args.observer_rs), fov_deg=float(args.fov_deg),
        width=int(args.width), height=int(args.height), radial_samples=args.radial_samples,
        roll_deg=float(args.roll_deg), return_arrays=True,
    )
    record.update({
        "black_hole_mass_solar": float(args.mass_solar),
        "schwarzschild_radius_km": float(rs_km),
        "photon_sphere_radius_km": float(1.5 * rs_km),
        "critical_impact_parameter_km": float(CRITICAL_IMPACT_RS * rs_km),
        "event_horizon_surface_drawn": False,
        "source_panorama": source_record,
        "output_image": str(output),
        "software": Path(__file__).name,
        "software_sha256": _sha256(Path(__file__)),
        "constants_si": {"G_m3_kg_s2": G_SI, "c_m_s": C_SI, "solar_mass_kg": M_SUN_KG},
        "formula_reference": "For Schwarzschild r_s=2GM/c^2, r_ph=3GM/c^2, and b_crit=3*sqrt(3)*GM/c^2. The apparent shadow edge is the capture boundary, not the horizon surface.",
    })
    metadata_path = output.with_suffix(".json")
    from core.data import export_image_product
    from core.foundation import physical_constants_contract
    specifications = {name: {'units': 'rad', 'quantity': name.replace('_', ' '),
        'missing_policy': 'Sky direction is undefined for captured rays; native radial NaN is retained'}
        for name in ('observer_angle_rad', 'source_angle_rad', 'screen_angle_rad', 'screen_azimuth_rad')}
    specifications['captured'] = {'units': 'dimensionless', 'quantity': 'Null ray capture indicator'}
    for name in ('source_rgb', 'output_rgb'):
        specifications[name] = {'units': 'uint8 display channel code', 'quantity': name.replace('_', ' '),
            'annotations': {'interpretation': 'User panorama/display RGB codes; not calibrated physical luminosity'}}
    export_image_product(output, image, record, arrays, specifications, algorithm=__file__,
        context={'epoch': {'canonical_time': 'static stationary model', 'actual_scale_factor': None,
            'redshift': None, 'cosmic_time_gyr': None, 'cosmic_time_status': 'NOT_APPLICABLE'},
            'coordinates': {'frame': 'Static Schwarzschild observer local angular screen',
                'origin': 'Optical axis', 'angle_units': 'rad', 'source_longitude_periodicity': '2 pi',
                'velocity_convention': 'Static observer; no particle velocities'},
            'adopted_constants': physical_constants_contract(), 'model': record['model']},
        source_hashes={str(panorama_path): panorama_before})
    print("Schwarzschild ray trace: " + str(output))
    print("Run metadata: " + str(metadata_path))
    print("Horizon radius: %.6g km | apparent shadow diameter: %.6g deg" % (
        rs_km, float(record["shadow_diameter_at_observer_deg"]),
    ))

SCENE_COMMANDS = {
    'schwarzschild-lensing': main,
}

# Native operator journals are authoritative source events, independently of
# plotting/scene histories. Integer identities never pass through a float array.
_BH_EVENT_FIELDS={
    'seed':{'previous_species','gravity_mass_native','physical_mass_native','rule'},
    'gas_accretion':{'source_id','mass_before_native','mass_after_native','velocity_before_native','velocity_after_native','internal_energy_before_native','internal_energy_after_native'},
    'accretion':{'mass_before_native','mass_after_native','physical_mass_before_native','physical_mass_after_native','inflow_mass_native','radiated_mass_native','radiated_momentum_native','velocity_before_native','velocity_after_native','thermal_energy_native','kinetic_energy_native','advected_internal_energy_native','mixing_energy_native','density_g_cm3','sound_speed_cm_s','relative_speed_cm_s','specific_angular_momentum_cm2_s','rotational_suppression','eddington_ratio'},
    'friction':{'impulse_g_cm_s','bath_heat_erg','velocity_before_cm_s','velocity_after_cm_s'},
    'binary_capture':{'companion_id','semimajor_axis_cm','eccentricity','private_binding_energy_erg','constraint_reservoir_energy_erg','angular_reservoir_g_cm2_s','capture_model'},
    'binary_hardening':{'companion_id','semimajor_axis_cm','eccentricity','gw_energy_erg','stellar_energy_erg','constraint_reservoir_energy_erg','angular_reservoir_g_cm2_s','gw_angular_momentum_g_cm2_s'},
    'merger':{'parent_ids','gravity_mass_before_g','physical_mass_before_g','gravity_mass_after_g','physical_mass_after_g','radiated_mass_g','radiated_energy_erg','radiated_momentum_g_cm_s','velocity_before_cm_s','velocity_after_cm_s','physical_recoil_cm_s','composite_recoil_cm_s','spin','fit_reference_gm_c2','inspiral_gw_energy_erg','stellar_scattering_energy_erg','rest_mass_deficit_energy_erg','private_binding_energy_erg','composite_kinetic_before_erg','composite_kinetic_after_erg','post_gate_radiated_angular_momentum_g_cm2_s'},
}


from core.black_hole_ledger import SEED_EVENT_FIELDS
_BH_EVENT_FIELDS.update(SEED_EVENT_FIELDS)
from core.black_hole_ledger import ACCRETION_INTEGRAL_FIELDS, ACCRETION_MERGE_FIELDS
_BH_EVENT_FIELDS['gas_feedback']=_BH_EVENT_FIELDS['gas_accretion']
_BH_EVENT_FIELDS['accretion_integral']=ACCRETION_INTEGRAL_FIELDS
_BH_EVENT_FIELDS['accretion_merge']=ACCRETION_MERGE_FIELDS

_BH_EVENT_FIELDS['sidm_drag']={'previous_counter','recipient_count','source_native_begin','source_native_end','proper_begin_seconds','proper_end_seconds',
    'response_product_sha256','fixed_response_signature','table_crc32','heat_msun_km2_s2','acceleration_cm_s2',
    'rounded_momentum_residual_msun_km_s','rounded_angular_residual_msun_kpc_km_s','rounded_energy_residual_msun_km2_s2',
    'momentum_budget_msun_km_s','angular_budget_msun_kpc_km_s','energy_budget_msun_km2_s2','reference_error_cm_s2','heldout_error_cm_s2'}
_BH_EVENT_FIELDS['sidm_drag_initialize']={'initialized_source_counter','proper_origin_seconds','response_product_sha256','fixed_response_signature','table_crc32'}
_BH_EVENT_FIELDS['sidm_drag_response_transition']={'retained_source_counter','proper_origin_seconds','retained_heat_erg',
    'previous_response_product_sha256','response_product_sha256','fixed_response_signature','previous_table_crc32','table_crc32'}


def validate_native_black_hole_event(row,parameters):
    """Validate the native wire contract and add unit-explicit analysis columns."""
    if not isinstance(row,dict) or row.get('kind') not in _BH_EVENT_FIELDS or set(row)!={'kind','native_time','particle_id','counter',*_BH_EVENT_FIELDS[row['kind']]}:
        raise ValueError('Unknown or partial native BH event schema')
    if row['kind'] in ('sidm_drag','sidm_drag_initialize','sidm_drag_response_transition'):
        import re
        for name in ('response_product_sha256','fixed_response_signature'):
            if not isinstance(row[name],str) or re.fullmatch('[0-9a-f]{64}',row[name]) is None:raise ValueError('Exact native drag model/signature identity required')
        if type(row['table_crc32'])is not int or not 0<=row['table_crc32']<2**32:raise ValueError('Exact native drag table CRC required')
    if row['kind']=='sidm_drag_response_transition':
        import re
        if not isinstance(row['previous_response_product_sha256'],str) or re.fullmatch('[0-9a-f]{64}',row['previous_response_product_sha256']) is None or row['previous_response_product_sha256']==row['response_product_sha256']:
            raise ValueError('Native response transition needs distinct exact old/new product identities')
        if type(row['previous_table_crc32'])is not int or not 0<=row['previous_table_crc32']<2**32:
            raise ValueError('Exact previous native response table CRC required')
        if type(row['retained_source_counter'])is not int or not 0<=row['retained_source_counter']<2**64-1 or row['counter']!=row['retained_source_counter']+1:
            raise ValueError('Response metadata transition must retain its source counter')
    if row['kind']=='sidm_drag_initialize':
        if type(row['initialized_source_counter'])is not int or row['initialized_source_counter']!=0 or row['counter']!=1:
            raise ValueError('Native drag initialization must start a zero source counter')
    if row['kind']=='sidm_drag':
        if type(row['previous_counter'])is not int or not 0<=row['previous_counter']<2**64-1 or row['counter']!=row['previous_counter']+1:
            raise ValueError('Native drag event does not advance exactly one source counter')
        if type(row['recipient_count'])is not int or not 3<=row['recipient_count']<2**64:raise ValueError('Complete nonempty native drag recipient census required')
        if type(row['table_crc32'])is not int or not 0<=row['table_crc32']<2**32:raise ValueError('Exact native drag table CRC required')
    def identity(value,allow_zero=False):
        if type(value) is not int or not (0<=value<2**64 if allow_zero else 0<value<2**64):raise ValueError('BH IDs/counters require exact unsigned 64-bit integers')
    identity(row['particle_id'],allow_zero=row['kind'] in ('gas_accretion','gas_feedback','seed_donor'))
    identity(row['counter'])
    for name in ('source_id','companion_id'):
        if name in row:identity(row[name])
    if 'parent_ids' in row:
        if not isinstance(row['parent_ids'],list) or len(row['parent_ids'])!=2 or len(set(row['parent_ids']))!=2:raise ValueError('A native BH merger requires two distinct parents')
        for pid in row['parent_ids']:identity(pid)
        if row['parent_ids'][0]!=row['particle_id']:raise ValueError('Native merger survivor must be the declared first parent')
    vectors={'velocity_native','radiated_momentum_native','velocity_before_native','velocity_after_native','impulse_g_cm_s',
        'velocity_before_cm_s','velocity_after_cm_s','radiated_momentum_g_cm_s','physical_recoil_cm_s','composite_recoil_cm_s','spin','angular_reservoir_g_cm2_s','gw_angular_momentum_g_cm2_s','post_gate_radiated_angular_momentum_g_cm2_s'}
    vectors|={'acceleration_cm_s2','rounded_momentum_residual_msun_km_s','rounded_angular_residual_msun_kpc_km_s'}
    vectors|={'offset_proper_cm','captured_orbital_angular_momentum_g_cm2_s'}
    element_vectors={'element_mass_before_g','element_mass_after_g','captured_element_mass_g','carrier_element_mass_before_g'}
    pairs={'gravity_mass_before_g','physical_mass_before_g'}
    finite_pairs=set()
    if row['kind']=='accretion_merge':
        from core.black_hole_ledger import ACCRETION_STATE_KEYS
        finite_pairs={key+'_before_native' for key in ACCRETION_STATE_KEYS}
    signed={'private_binding_energy_erg','constraint_reservoir_energy_erg','mixing_energy_native'}
    signed|={'proper_begin_seconds','proper_end_seconds','proper_origin_seconds','rounded_energy_residual_msun_km2_s2'}
    signed|={key for key in row if 'mass_correction_' in key}
    for key,value in row.items():
        if key in {'kind','particle_id','counter','source_id','companion_id','parent_ids','rule','capture_model','previous_species','response_product_sha256','previous_response_product_sha256','fixed_response_signature'}:continue
        if key in vectors or key in pairs or key in element_vectors or key in finite_pairs:
            length=11 if key in element_vectors else 3 if key in vectors else 2
            if not isinstance(value,list) or len(value)!=length or any(type(x) not in (int,float) or not math.isfinite(x) for x in value):raise ValueError('Invalid BH event vector: '+key)
            if key in pairs and min(value)<=0:raise ValueError('BH parent masses must be positive')
            if key in element_vectors and min(value)<0:raise ValueError('Negative seed element ledger')
        elif type(value) not in (int,float) or not math.isfinite(value) or key not in signed and value<0:
            raise ValueError('Invalid BH source measurement: '+key)
    if row['kind'] in ('accretion_integral','accretion_merge'):
        from core.black_hole_ledger import validate_roundoff_event
        validate_roundoff_event(row)
    if row['kind']=='gas_feedback' and row['mass_before_native']!=row['mass_after_native']:
        raise ValueError('Feedback-only native recipient changes its mass')
    if row['kind'] in SEED_EVENT_FIELDS:
        if type(row['transaction_index']) is not int or not 1<=row['transaction_index']<2**64 or row['counter']!=1:
            raise ValueError('Seed assembly requires exact positive transaction index and initial source counter')
        if not 0<row['mass_before_native'] or not 0<row['mass_after_native']:
            raise ValueError('Seed assembly must preserve positive live masses')
        if row['kind']=='seed_assembly':
            if type(row['previous_species']) is not int or row['previous_species']!=0 or type(row['donor_count']) is not int or row['donor_count']<0 or not 0<row['physical_mass_native']<=row['mass_after_native'] or not 0<row['maximum_donor_fraction']<=.1 or row['aperture_proper_kpc']<=0:
                raise ValueError('Invalid explicit seed assembly prescription')
    if row['kind']=='seed' and (type(row['previous_species']) is not int or row['previous_species'] not in (0,5) or row['rule'] not in {'ic_particles','density_exclusion'} or (row['previous_species']==0)!=(row['rule']=='density_exclusion')):
        raise ValueError('Inconsistent native BH seed rule')
    if row['kind']=='binary_capture' and row['capture_model']!='quasi_circular_COM_carrier':raise ValueError('Unsupported native BH capture model')
    if row['kind']=='sidm_drag':
        if not row['proper_end_seconds']>row['proper_begin_seconds'] or not row['source_native_begin']<row['source_native_end']<=row['native_time']:
            raise ValueError('Native drag event has a regressed/empty source interval')
        for residual,budget in (('rounded_momentum_residual_msun_km_s','momentum_budget_msun_km_s'),('rounded_angular_residual_msun_kpc_km_s','angular_budget_msun_kpc_km_s')):
            if row[budget]<=0 or float(np.linalg.norm(row[residual]))>row[budget]:raise ValueError('Native rounded drag conservation exceeds its recorded budget')
        if row['energy_budget_msun_km2_s2']<=0 or abs(row['rounded_energy_residual_msun_km2_s2'])>row['energy_budget_msun_km2_s2']:
            raise ValueError('Native rounded drag energy does not close within its recorded budget')
    if 'eccentricity' in row and row['eccentricity']>.05:raise ValueError('Native BH event outside quasi-circular profile')
    if 'semimajor_axis_cm' in row and row['semimajor_axis_cm']<=0:raise ValueError('Native binary axis must be positive')
    if row['kind']=='merger' and (row['fit_reference_gm_c2']!=10 or np.linalg.norm(row['spin'])>1):raise ValueError('Invalid remnant fit reference/spin')
    if row['kind']=='merger':
        residual=row['rest_mass_deficit_energy_erg']+row['private_binding_energy_erg']+row['composite_kinetic_before_erg']-row['composite_kinetic_after_erg']-row['radiated_energy_erg']
        scale=max(row['rest_mass_deficit_energy_erg'],row['composite_kinetic_before_erg'],row['composite_kinetic_after_erg'])
        if abs(residual)>1.e-10*scale or row['radiated_energy_erg']<np.linalg.norm(row['radiated_momentum_g_cm_s'])*2.99792458e10:raise ValueError('Native merger post-gate energy/radiation momentum does not close')
    mass=parameters['unit_mass_g']/MSUN_G/(parameters['hubble_param'] if parameters['comoving'] else 1.)
    velocity=parameters['unit_velocity_cm_s']/1.e5;energy=mass*MSUN_G*parameters['unit_velocity_cm_s']**2
    out=dict(row)
    # Keep the linear mass conversion beside raw endpoints so source replay
    # can subtract in native units before converting a small recorded transfer.
    if row['kind'] in ('gas_accretion','gas_feedback','accretion','accretion_integral','accretion_merge','seed_donor','seed_assembly'):
        out['native_mass_unit_msun']=mass
    for key,value in row.items():
        if key.endswith('_native'):
            prefix=key[:-7]
            if 'momentum' in key:out[prefix+'_msun_km_s']=(np.asarray(value)*mass*velocity).tolist()
            elif 'velocity' in key:out[prefix+'_km_s']=(np.asarray(value)*velocity).tolist()
            elif key in {'internal_energy_before_native','internal_energy_after_native'}:
                out[prefix+'_cm2_s2']=float(value)*parameters['unit_velocity_cm_s']**2
            elif 'energy' in key:out[prefix+'_erg']=float(value)*energy
            elif 'mass' in key:
                converted=np.asarray(value)*mass;out[prefix+'_msun']=converted.tolist() if converted.shape else float(converted)
        elif key.endswith('_g') and 'mass' in key:
            converted=np.asarray(value)/MSUN_G;out[key[:-2]+'_msun']=converted.tolist() if converted.shape else float(converted)
        elif key.endswith('_cm_s') and 'velocity' in key:out[key[:-5]+'_km_s']=(np.asarray(value)/1.e5).tolist()
        elif key=='radiated_momentum_g_cm_s':out['radiated_momentum_msun_km_s']=(np.asarray(value)/(MSUN_G*1.e5)).tolist()
    return out


def read_native_black_hole_journal(directory,*,before,after,memory_bytes):
    """Read all rank payloads between exact snapshot counters; retain source hashes."""
    import zlib,re
    from core.foundation import file_sha256 as digest
    directory=Path(directory).resolve();rows=[];commits=[];sources={};seen=set();drag_history={}
    start=before.get('bh_commit_count',0);stop=after.get('bh_commit_count',0)
    count=before.get('bh_event_count',0);target=after.get('bh_event_count',0)
    if stop<start or target<count or stop==start and count!=target:raise ValueError('Incomplete/regressed BH journal counter chain')
    if type(memory_bytes) is not int or (target-count)*16384+16*1024**2>memory_bytes:raise MemoryError('BH event parser exceeds declared workspace')
    def unique(items):
        out={}
        for key,value in items:
            if key in out:raise ValueError('Duplicate native BH JSON key')
            out[key]=value
        return out
    def record(path):
        if not path.is_file() or path.is_symlink():raise ValueError('Missing/linked native BH journal payload')
        sources[str(path)]=digest(path)
    last=before['native_time'];fingerprint=None;parameters=after['parameters']
    for sequence in range(start+1,stop+1):
        path=directory/f'black_holes_{sequence:020d}.commit.json';record(path)
        if path.stat().st_size>min(16*1024**2,memory_bytes//16):raise MemoryError('BH commit size exceeds declared workspace')
        commit=json.loads(path.read_text(),object_pairs_hook=unique,parse_constant=lambda x:(_ for _ in ()).throw(ValueError('Nonfinite BH JSON')))
        required={'schema','sequence','previous_count','cumulative_count','event_count','native_time','native_step_begin','source_fingerprint','comoving','unit_mass_g','unit_velocity_cm_s','unit_time_s','hubble_param','mpi_ranks','parts'}
        if set(commit)!=required or commit['schema']!='native-black-hole-events-v1':raise ValueError('Unknown/partial native BH commit schema')
        for key in ('sequence','previous_count','cumulative_count','event_count','mpi_ranks'):
            if type(commit[key]) is not int or not 0<=commit[key]<2**64:raise ValueError('BH commit counters must be exact uint64')
        if commit['sequence']!=sequence or commit['previous_count']!=count or commit['event_count']==0 or commit['cumulative_count']!=count+commit['event_count'] or commit['cumulative_count']>target:raise ValueError('BH commit chain does not match saved snapshot census')
        time=commit['native_time'];begin=commit['native_step_begin']
        if type(begin) not in (int,float) or not math.isfinite(begin) or begin<0 or begin>time:raise ValueError('BH native step clock bracket is invalid')
        if type(time) not in (int,float) or not math.isfinite(time) or not last<=time<=after['native_time']:raise ValueError('BH commit outside actual epoch bracket')
        for name in ('comoving','unit_mass_g','unit_velocity_cm_s','unit_time_s','hubble_param'):
            if type(commit[name]) not in (int,float) or commit[name]!=parameters[name]:raise ValueError('BH journal units/model differ from snapshots')
        fp=commit['source_fingerprint']
        if not isinstance(fp,str) or re.fullmatch('[0-9a-f]{64}',fp) is None or fingerprint is not None and fp!=fingerprint:raise ValueError('BH native source changed within interval')
        fingerprint=fp;ranks=commit['mpi_ranks'];parts=commit['parts']
        if not 1<=ranks<=1000000 or not isinstance(parts,list) or len(parts)!=ranks:raise ValueError('Incomplete BH MPI rank census')
        observed=0
        for rank,part in enumerate(parts):
            if not isinstance(part,dict) or set(part)!={'rank','rows','bytes','crc32','filename'} or part['rank']!=rank or any(type(part[k]) is not int or part[k]<0 for k in ('rank','rows','bytes','crc32')) or part['crc32']>=2**32:raise ValueError('Invalid BH rank payload descriptor')
            if part['rows']==0:
                if part['filename'] is not None or part['bytes'] or part['crc32']:raise ValueError('Empty BH rank claims an event payload')
                continue
            name=f'black_holes_{sequence:020d}.rank_{rank:06d}.of_{ranks:06d}.jsonl'
            if part['filename']!=name or not part['rows']<=part['bytes']<=part['rows']*8192 or observed+part['rows']>commit['event_count']:raise ValueError('BH payload path/size/census mismatch')
            path=directory/name;record(path);crc=0;nbytes=0;nrows=0
            with path.open('rb') as stream:
                while True:
                    line=stream.readline(8193)
                    if not line:break
                    if len(line)>8192 or not line.endswith(b'\n'):raise ValueError('Unbounded or truncated BH event row')
                    nrows+=1;nbytes+=len(line);crc=zlib.crc32(line,crc)
                    if nrows>part['rows']:raise ValueError('Undeclared BH event rows')
                    raw=json.loads(line.decode('ascii'),object_pairs_hook=unique,parse_constant=lambda x:(_ for _ in ()).throw(ValueError('Nonfinite BH JSON')))
                    row=validate_native_black_hole_event(raw,parameters)
                    row['native_step_begin']=begin
                    row['applied_time_convention']='operator applied at step end; continuous merger occurrence remains inside the native source-step bracket'
                    if row['native_time']!=time:raise ValueError('BH payload and commit actual epochs differ')
                    if row['kind']=='sidm_drag_initialize':
                        if row['particle_id'] in drag_history:raise ValueError('Native drag initialization overwrites an observed source history')
                        drag_history[row['particle_id']]={'counter':0,'proper_end_seconds':row['proper_origin_seconds'],'source_native_end':row['native_time'],
                            'definition':(row['response_product_sha256'],row['fixed_response_signature'],row['table_crc32'])}
                    if row['kind']=='sidm_drag':
                        previous=drag_history.get(row['particle_id'])
                        definition=(row['response_product_sha256'],row['fixed_response_signature'],row['table_crc32'])
                        if row['source_native_begin']<begin:raise ValueError('Native drag interval begins outside its committed native step')
                        if previous is not None and (row['previous_counter']!=previous['counter'] or row['proper_begin_seconds']!=previous['proper_end_seconds'] or row['source_native_begin']!=previous['source_native_end'] or definition!=previous['definition']):
                            raise ValueError('Native drag counter/clock/model chain is incomplete or changes without an explicit migration')
                        drag_history[row['particle_id']]={'counter':row['counter'],'proper_end_seconds':row['proper_end_seconds'],'source_native_end':row['source_native_end'],'definition':definition}
                    if row['kind']=='sidm_drag_response_transition':
                        previous=drag_history.get(row['particle_id'])
                        old_definition=(row['previous_response_product_sha256'],row['fixed_response_signature'],row['previous_table_crc32'])
                        if previous is not None and (previous['definition']!=old_definition or previous['counter']!=row['retained_source_counter'] or
                                previous['proper_end_seconds']!=row['proper_origin_seconds'] or previous['source_native_end']!=row['native_time']):
                            raise ValueError('Native response transition breaks the retained source clock/counter/model history')
                        drag_history[row['particle_id']]={'counter':row['retained_source_counter'],'proper_end_seconds':row['proper_origin_seconds'],
                            'source_native_end':row['native_time'],'definition':(row['response_product_sha256'],row['fixed_response_signature'],row['table_crc32'])}
                    key=(sequence,row['kind'],row['particle_id'],row.get('source_id',row['particle_id']),row['counter'])
                    if key in seen:raise ValueError('Duplicated native BH operation row')
                    seen.add(key);rows.append(row)
            if (nrows,nbytes,crc&0xffffffff)!=(part['rows'],part['bytes'],part['crc32']):raise ValueError('BH event payload CRC/byte/row census mismatch')
            observed+=nrows
        if observed!=commit['event_count']:raise ValueError('BH event commit lacks complete rank payloads')
        count=commit['cumulative_count'];last=time;commits.append(commit)
    if count!=target:raise ValueError('BH journal does not reach requested actual snapshot counter')
    for path,checksum in sources.items():
        if digest(path)!=checksum:raise ValueError('BH journal changed while being read')
    return rows,commits,sources


def black_hole_merger_lineage(events):
    """Versioned graph: surviving IDs produce new nodes, preventing self-cycles.

    Initial versions denote the unknown earlier ancestry at the input boundary.
    Complete ancestry requires reading the inherited journal prefix as well.
    """
    versions={};active=set();retired=set();nodes={};edges=[]
    for ordinal,row in enumerate(sorted(events,key=lambda r:(r['native_time'],0 if r['kind'] in ('seed','seed_assembly') else 1,r['particle_id'],r['counter']))):
        if row['kind'] in ('seed','seed_assembly'):
            pid=row['particle_id']
            if pid in versions or pid in retired:raise ValueError('BH seed reuses an existing lineage identity')
            versions[pid]=0;active.add(pid);nodes[(pid,0)]={'particle_id':pid,'version':0,'origin':'native_seed','native_time':row['native_time']}
        elif row['kind']=='merger':
            parents=row['parent_ids']
            for pid in parents:
                if pid in retired:raise ValueError('BH merger reuses a retired parent')
                if pid not in versions:
                    versions[pid]=0;active.add(pid);nodes[(pid,0)]={'particle_id':pid,'version':0,'origin':'input_boundary_ancestry_unavailable','native_time':None}
            survivor=row['particle_id'];new=versions[survivor]+1;key=(survivor,new)
            nodes[key]={'particle_id':survivor,'version':new,'origin':'native_merger','native_time':row['native_time'],'event_counter':row['counter']}
            for pid in parents:edges.append({'parent_particle_id':pid,'parent_version':versions[pid],'child_particle_id':survivor,'child_version':new,'event_ordinal':ordinal})
            versions[survivor]=new
            for pid in parents:
                if pid!=survivor:active.discard(pid);retired.add(pid)
    return {'schema':'versioned-native-bh-lineage-v1','nodes':list(nodes.values()),'edges':edges,
        'live_ids':sorted(active),'retired_ids':sorted(retired),
        'ancestry_scope':'Inherited prefix is required for ancestry earlier than the selected input boundary'}


def publish_black_hole_environment_history(request,destination):
    """Measured BH orbital states and time-resolved DM response in finite hosts.

    Escape refers to the declared isolated reconstructed host with Phi(infinity)
    zero. A periodic cosmological potential has no escape-to-infinity energy.
    Host omissions, field-error bounds and unresolved radii remain explicit.
    """
    from core.foundation import file_sha256 as digest
    from core.data import verify_numeric_audit,export_numeric_audit
    from halos import verify_halo_structure,ReconstructedHostPotential
    keys={'hydro_history','host_epochs','selected_particle_ids','central_radius_proper_kpc','energy_error_km2_s2','dm_radial_edges_proper_kpc','memory_gib'}
    if not isinstance(request,dict) or set(request)!=keys:raise ValueError('Declare full BH environment history request')
    from core.data import exact_uint64_ids
    selected=request['selected_particle_ids']
    if not isinstance(selected,list) or any(type(i) is not int or not 0<=i<2**64 for i in selected):raise ValueError('Select exact uint64 BH identities')
    ids=exact_uint64_ids(np.asarray(selected,dtype=np.uint64),allow_empty=False)
    if len(np.unique(ids))!=len(ids):raise ValueError('Duplicate selected BH identity')
    central=finite_number(request['central_radius_proper_kpc'],'central aperture',strict=True)
    error=finite_number(request['energy_error_km2_s2'],'energy error');memory=finite_number(request['memory_gib'],'workspace',strict=True)*1024**3
    edges=np.asarray(request['dm_radial_edges_proper_kpc'],dtype=float)
    if edges.ndim!=1 or len(edges)<3 or not np.isfinite(edges).all() or edges[0]<=0 or np.any(np.diff(edges)<=0):raise ValueError('Declare increasing positive DM radial edges')
    path=Path(request['hydro_history']).resolve();report=verify_numeric_audit(path)
    if report.get('schema')!='native-hydro-history-v1':raise ValueError('Use a source-certified native six-species hydro history')
    sources={str(path):digest(path),**report['source_hashes'],str(Path(__file__).resolve()):digest(__file__)}
    for source,checksum in report['source_hashes'].items():
        if digest(source)!=checksum:raise ValueError('Hydro history source changed before BH environment intake')
    archive=path.parent/report['numeric_data']['directory']/'arrays.npz';sources[str(archive)]=digest(archive)
    import zipfile
    with zipfile.ZipFile(archive) as archive_index:allocation=sum(item.file_size for item in archive_index.infolist())
    if allocation+len(ids)*len(report['epochs'])*512+64*1024**2>memory:raise MemoryError('BH environment numeric intake exceeds declared workspace')
    declared=request['host_epochs']
    if not isinstance(declared,list) or len(declared)!=len(report['epochs']):raise ValueError('Declare a measured host at every selected epoch')
    payload={'selected_particle_ids':ids};specs={'selected_particle_ids':{'units':'identity','quantity':'selected native BH identities','identity_namespace':report['provenance']['identity_namespaces']['native_particle']}}
    metadata=[];previous_mass=None
    with np.load(archive,allow_pickle=False) as data:
        for ordinal,(epoch,host) in enumerate(zip(report['epochs'],declared)):
            if set(host)!={'snapshot','structure_product','host_bulk_velocity_peculiar_km_s','maximum_field_radius_proper_kpc','periodic_box_proper_kpc'} or host['snapshot']!=epoch['snapshot']:raise ValueError('Declare matching host field, velocity frame and domain')
            structure_path=Path(host['structure_product']).resolve();structure=verify_halo_structure(structure_path)
            if structure['request']['snapshot']!=epoch['snapshot'] or Path(structure['request']['run_directory']).resolve()!=Path(report['request']['run_directory']).resolve():raise ValueError('Host field belongs to another native snapshot/run')
            sources[str(structure_path)]=digest(structure_path);sources.update(structure['source_hashes'])
            if any(5 in component['species'] for component in structure['request']['components'].values()):raise ValueError('BH host binding requires a host field excluding native type-5 sinks; self potential cannot bind its own BH')
            model=ReconstructedHostPotential.from_coefficients(structure_path.parent,structure['potential_coefficients'])
            context=structure['source_context'];center=np.asarray(context['center_comoving_kpc_h'],dtype=float)*context['scale_factor']/context['h']
            bulk=np.asarray(host['host_bulk_velocity_peculiar_km_s'],dtype=float);box=finite_number(host['periodic_box_proper_kpc'],'proper box',strict=True);domain=finite_number(host['maximum_field_radius_proper_kpc'],'field domain',strict=True)
            if center.shape!=(3,) or bulk.shape!=(3,) or not np.isfinite(center).all() or not np.isfinite(bulk).all() or domain>=box/2 or domain>context['support_radius_proper_kpc'] or edges[-1]>=box/2:raise ValueError('Host environment frame/domain is invalid')
            prefix=f'epoch_{ordinal:03d}_';native_ids=data[prefix+'particle_ids'];species=data[prefix+'species'];mass=data[prefix+'mass_msun'];xyz=data[prefix+'positions_proper_kpc'];velocity=data[prefix+'velocities_peculiar_km_s']
            offsets=(xyz-center+box/2)%box-box/2;radii=np.linalg.norm(offsets,axis=1);index={int(pid):i for i,pid in enumerate(native_ids)}
            out={'present':np.zeros(len(ids),dtype=bool),'state':np.zeros(len(ids),dtype=np.int8),'radius_proper_kpc':np.zeros(len(ids)),
                'specific_host_energy_km2_s2':np.zeros(len(ids)),'physical_mass_msun':np.zeros(len(ids)),'gravity_mass_msun':np.zeros(len(ids)),
                'spin':np.zeros((len(ids),3)),'spin_available':np.zeros(len(ids),dtype=bool)}
            for j,pid in enumerate(ids):
                i=index.get(int(pid))
                if i is None or species[i]!=5:continue
                out['present'][j]=True;out['radius_proper_kpc'][j]=radii[i];out['gravity_mass_msun'][j]=mass[i];out['physical_mass_msun'][j]=data[prefix+'BH_Mass'][i]
                if data[prefix+'bh_native_state_available'][i]:out['spin'][j]=data[prefix+'BHSpin'][i];out['spin_available'][j]=True
                if radii[i]>domain or structure['field_validation_status']!='PASS':out['state'][j]=4;continue
                phi=model.evaluate(offsets[i:i+1])[0][0];energy=.5*float(np.dot(velocity[i]-bulk,velocity[i]-bulk))+float(phi);out['specific_host_energy_km2_s2'][j]=energy
                out['state'][j]=3 if energy>error else 5 if abs(energy)<=error else 1 if radii[i]<=central else 2
            dm=(species==1)&(mass>0);shell=np.histogram(radii[dm],bins=edges,weights=mass[dm])[0];out['dm_shell_mass_msun']=shell
            out['dm_shell_density_msun_kpc3']=shell/(4*np.pi/3*np.diff(edges**3))
            out['dm_mass_change_since_first_msun']=shell.copy()*0 if previous_mass is None else shell-previous_mass
            if previous_mass is None:previous_mass=shell.copy()
            for name,values in out.items():
                payload[prefix+name]=values;specs[prefix+name]={'units':'proper kpc' if 'radius' in name else '(km/s)^2' if 'energy' in name else 'Msun/kpc^3' if 'density' in name else 'Msun' if 'mass' in name else 'dimensionless','quantity':name,'epoch':epoch['epoch']}
            metadata.append({'snapshot':epoch['snapshot'],'epoch':epoch['epoch'],'structure_product':str(structure_path),'states':{'0':'not observed as a live BH','1':'retained central','2':'bound wandering','3':'unbound from declared finite host','4':'host field domain unavailable','5':'energy sign unresolved'},'maximum_field_radius_proper_kpc':domain})
    for source,checksum in sources.items():
        if digest(source)!=checksum:raise ValueError('BH environment source changed before publication')
    output={'schema':'native-bh-environment-history-v1','request':request,'epochs':metadata,'source_hashes':sources,
        'dm_response_scope':'Measured fixed proper-aperture DM evolution about selected host; causal feedback attribution requires a matched BH-disabled run. No unresolved spike claim.',
        'escape_scope':'Finite host reconstruction, not cosmological escape; unresolved field/domain and energy sign stay masked',
        'physics_acceptance':'NOT_ACCEPTED'}
    return export_numeric_audit(destination,output,payload,specs,algorithm=__file__,context={'provenance':report['provenance'],'selection':'selected native BHs and host DM radial states'},source_hashes=sources)


def integrate_peters_reference(mass1_msun,mass2_msun,semimajor_axis_cm,eccentricity,times_seconds,*,relative_tolerance,absolute_axis_tolerance_cm):
    """Independent adaptive orbit-averaged reference with a declared PN gate.

    Includes the exact circular a^4 law for an independent limiting check.
    This function does not invoke the production binary timestep operator.
    """
    from scipy.integrate import solve_ivp
    times=np.asarray(times_seconds,dtype=float)
    m1=finite_number(mass1_msun,'mass1',strict=True);m2=finite_number(mass2_msun,'mass2',strict=True)
    a0=finite_number(semimajor_axis_cm,'axis',strict=True);e0=finite_number(eccentricity,'eccentricity')
    rtol=finite_number(relative_tolerance,'relative tolerance',strict=True);atol=finite_number(absolute_axis_tolerance_cm,'absolute axis tolerance',strict=True)
    if e0>=1 or rtol>=.01 or times.ndim!=1 or len(times)<2 or times[0]!=0 or not np.isfinite(times).all() or np.any(np.diff(times)<=0):raise ValueError('Declare a bounded physical orbit and increasing reference times starting at zero')
    gate=10*G_CGS*(m1+m2)*MSUN_G/C_CGS**2
    if a0<=gate:raise ValueError('Reference inspiral starts inside the declared near-merger gate')
    def rhs(t,y):
        rates=peters_binary_rates(m1,m2,float(y[0]),max(0.,float(y[1])))
        return [rates['da_cm_s'],rates['de_per_s']]
    def event(t,y):return y[0]-gate
    event.terminal=True;event.direction=-1
    solution=solve_ivp(rhs,(0,float(times[-1])),[a0,e0],method='DOP853',rtol=rtol,atol=[atol,rtol*1.e-3],events=event,dense_output=True)
    if not solution.success:raise ValueError('Independent Peters integration failed: '+solution.message)
    stop=float(solution.t[-1]);supported=times<=stop;axis=np.zeros(len(times));e=np.zeros(len(times));axis[supported],e[supported]=solution.sol(times[supported])
    rates=peters_binary_rates(m1,m2,a0,e0);coefficient=-rates['da_cm_s']*a0**3
    circular=np.maximum(0.,a0**4-4*coefficient*times)**.25 if e0==0 else None
    return {'times_seconds':times,'semimajor_axis_cm':axis,'eccentricity':e,'supported':supported,
        'gate_time_seconds':float(solution.t_events[0][0]) if len(solution.t_events[0]) else None,
        'circular_exact_axis_cm':circular,'relative_tolerance':rtol,'absolute_axis_tolerance_cm':atol,'gate_gm_c2':10.,'physics_acceptance':'NOT_ACCEPTED'}
