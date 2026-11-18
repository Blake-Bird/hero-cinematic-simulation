"""Native units, exact identities, data contracts and scientific publication."""
from __future__ import annotations

# ========================================================================
# gadget_io: Shared GADGET HDF5 bookkeeping; no solver or plotting imports.

# Workflow and units
# Provides shared HDF5 particle-count and exact-ID membership bookkeeping.
# IDs and counts stay uint64; group offsets/lengths address concatenated per-species shards with half-open spans.
# GADGET-4 full totals and classic low/high-word totals are decoded explicitly. Floating-point IDs and nearest-position membership are forbidden.
# Bundled source patches: core/patches. Runtime choices are saved with each run.

import numpy as np

CANONICAL_SCHEMA = 'darkmattersim-data-contract-v1'
IDENTITY_KINDS = ('dark_matter','gas','star_population','black_hole','stream_tracer',
                  'collisional_star','halo','subhalo','galaxy','progenitor','tree_node')

def native_physical_unit_scales(*, unit_mass_g, unit_length_cm, unit_velocity_cm_s, comoving, hubble_param):
    """Base physical scales for native extensive quantities and proper lengths.

    GADGET cosmological base mass/length carry 1/h; isolated units do not.
    Scale-factor conventions of individual saved datasets (including encoded
    velocities) are applied separately through dataset_to_cgs_factor.
    """
    if type(comoving) not in (bool,int) or comoving not in (0,1):
        raise ValueError('Explicit native integration mode 0 or 1 required')
    for value in (unit_mass_g,unit_length_cm,unit_velocity_cm_s):
        if isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,float,np.integer,np.floating)) or not np.isfinite(value) or value<=0:
            raise ValueError('Finite positive native physical base units required')
    h=1.
    if comoving:
        native_unit_factor('catalogue_velocity_peculiar_kms',scale_factor=1.,hubble_param=hubble_param)
        h=float(hubble_param)
    with np.errstate(over='ignore',under='ignore',invalid='ignore'):
        mass=np.float64(unit_mass_g)/h;length=np.float64(unit_length_cm)/h;velocity=np.float64(unit_velocity_cm_s)
        result={'mass_g':mass,'comoving_length_cm' if comoving else 'length_cm':length,
            'velocity_cm_s':velocity,'time_s':np.float64(unit_length_cm)/velocity,
            'energy_erg':mass*velocity**2,'specific_energy_cm2_s2':velocity**2,
            'angular_momentum_g_cm2_s':mass*length*velocity,
            'specific_angular_momentum_cm2_s':length*velocity}
    if any(not np.isfinite(value) or value<=0 for value in result.values()):
        raise ValueError('Native physical unit scale overflowed or underflowed')
    return {key:float(value) for key,value in result.items()}


def native_time_unit_seconds(parameters):
    """GADGET begrun.cc defines UnitTime = UnitLength / UnitVelocity.

    Parameters attributes contain registered input parameters; this derived
    field is not necessarily serialized. An explicit field must agree with
    that native definition. No cosmological h/a factor enters the base unit.
    """
    values=[]
    for name in ('UnitLength_in_cm','UnitVelocity_in_cm_per_s'):
        value=np.asarray(parameters[name])
        if value.shape!=() or value.dtype.kind not in 'fiu' or not np.isfinite(value) or value<=0:
            raise ValueError('Positive scalar native unit required: '+name)
        values.append(float(value))
    derived=values[0]/values[1]
    if not np.isfinite(derived) or derived<=0:raise ValueError('Native time unit overflows')
    if 'UnitTime_in_s' in parameters:
        explicit=np.asarray(parameters['UnitTime_in_s'])
        if explicit.shape!=() or explicit.dtype.kind not in 'fiu' or not np.isfinite(explicit) or not np.isclose(float(explicit),derived,rtol=16*np.finfo(float).eps,atol=0):
            raise ValueError('Explicit native time unit differs from UnitLength/UnitVelocity')
    return derived

# Exact quantities in the existing analysis producers, not filename heuristics.
ANALYSIS_DATASET_UNITS = {
    'hmf': {'mass_msun_h':'Msun/h','mass_bin_edges_msun_h':'Msun/h',
            'cumulative_mass_msun_h':'Msun/h','dn_dlogM':'(Mpc/h)^-3',
            'dn_dM':'(Msun/h)^-1 (Mpc/h)^-3','counts':'halo count',
            'cumulative_n_gt':'(Mpc/h)^-3'},
    'pk': {'k_h_Mpc':'h/Mpc','P_Mpc_h3':'(Mpc/h)^3','delta2':'dimensionless',
           'mode_count':'Fourier mode count','fold_segment':'fold-band index',
           'delta2_raw_shot_included':'dimensionless','delta2_shot_noise':'dimensionless',
           'P_raw_shot_included_Mpc_h3':'(Mpc/h)^3','P_shot_noise_Mpc_h3':'(Mpc/h)^3',
           'delta2_gaussian_1sigma':'dimensionless','P_gaussian_1sigma_Mpc_h3':'(Mpc/h)^3',
           'fold_band_lower_h_Mpc':'h/Mpc','fold_band_upper_h_Mpc':'h/Mpc',
           'P_native_column_Mpc_h3':'(Mpc/h)^3','native_column_normalization_relative_residual':'dimensionless',
           'time':'scale factor','box_mpc_h':'comoving Mpc/h','pm_grid':'grid cells per axis',
           'growth_factor':'dimensionless','fold_factor':'dimensionless',
           'k_fundamental_h_Mpc':'h/Mpc','k_nyquist_h_Mpc':'h/Mpc',
           'k_selection_upper_base_h_Mpc':'h/Mpc','selected_k_min_h_Mpc':'h/Mpc',
           'selected_k_max_h_Mpc':'h/Mpc','nonpositive_shot_subtracted_bin_count':'bin count',
           'native_column_max_relative_normalization_residual':'dimensionless'},
}


# These names correspond to fields written by analysis.py. Nested paths
# are resolved per producer; the same leaf name can mean different quantities.
ANALYSIS_DATASET_UNITS.update({
    'hydro': {
        'gas_density_log10_nH_cm3_bin_edges':'dimensionless log10(n_H / cm^-3)',
        'gas_density_probability_density':'probability per dex in n_H',
        'temperature_log10_bin_edges':'dimensionless log10(T / K)',
        'temperature_probability_density':'probability per dex in temperature',
        'phase_log10_density_bin_edges':'dimensionless log10(n_H / cm^-3)',
        'phase_log10_temperature_bin_edges':'dimensionless log10(T / K)',
        'phase_particle_count_2d':'sampled gas particle count',
        'phase_particle_count_estimate_2d':'gas particle count (inverse-inclusion estimate)',
        'temperature_excluded_star_forming_count':'sampled gas particle count',
        'temperature_excluded_invalid_sfr_count':'sampled gas particle count',
        'temperature_valid_sample_count':'sampled gas particle count',
        'sfr_total_snapshot_units':'native StarFormationRate code units',
        'sfr_total_raw_code_value':'native StarFormationRate code units',
        'sfr_particle_count':'gas particle count','sfr_finite_count':'gas particle count',
        'sfr_invalid_value_count':'gas particle count','sfr_negative_value_count':'gas particle count',
        'sfr_summed_value_count':'gas particle count','sfr_total_msun_per_year':'Msun/year',
        'sfr_dataset_to_cgs_factor':'g/s per native StarFormationRate unit',
        'baryon_fraction_halo_mass_msun_h':'Msun/h','gas_mass_fraction':'dimensionless',
        'stellar_mass_fraction':'dimensionless'},
    'concentration': {'mass_msun_h':'Msun/h','concentration_cvir':'dimensionless'},
    'spatial': {'position_mpc_h':'comoving Mpc/h','mass_msun_h':'Msun/h'},
    'subhmf': {'mass_msun_h':'Msun/h','dn_dlogM':'(Mpc/h)^-3'},
    'spin': {'specific_angular_momentum':'proper kpc km/s'},
    'correlation': {'r':'comoving Mpc/h','bin_edges':'comoving Mpc/h',
        'xi':'dimensionless','xi_poisson_68_lower':'dimensionless','xi_poisson_68_upper':'dimensionless',
        'pair_count':'unordered pair count','random_pair_expectation':'expected unordered pair count',
        'excluded_close_pair_count':'unordered pair count','input_point_count':'halo count',
        'used_point_count':'halo count','excluded_nonfinite_point_count':'halo count',
        'halo_count':'halo count','minimum_separation_mpc_h':'comoving Mpc/h',
        'maximum_separation_mpc_h':'comoving Mpc/h'},
    'bias': {'k_h_Mpc':'h/Mpc','bin_edges_h_Mpc':'h/Mpc',
        'bias':'dimensionless','bias_gaussian_68_sigma':'dimensionless',
        'bias_auto_hh_mm':'dimensionless','bias_auto_hh_mm_gaussian_68_sigma':'dimensionless',
        'cross_correlation_coefficient_hm':'dimensionless','mode_count':'Fourier mode count',
        'cross_power_Mpc_h3':'(Mpc/h)^3','halo_power_Mpc_h3':'(Mpc/h)^3',
        'matter_power_Mpc_h3':'(Mpc/h)^3','matter_shot_noise_Mpc_h3':'(Mpc/h)^3',
        'halo_shot_noise_Mpc_h3':'(Mpc/h)^3','nyquist_fraction':'dimensionless',
        'grid_size':'grid cells per axis','min_modes_per_bin':'Fourier mode count',
        'halo_count':'halo count','matter_particle_count':'particle count',
        'minimum_halo_particles':'particle count','selected_k_min_h_Mpc':'h/Mpc',
        'selected_k_max_h_Mpc':'h/Mpc'},
})

ANALYSIS_DATASET_UNITS['hmf'].update({
    'particle_filter/minimum_particles':'particle count',
    'particle_filter/input_halo_count':'halo count','particle_filter/included_halo_count':'halo count',
    'particle_filter/excluded_halo_count':'halo count',
    'resolution_selection/minimum_particle_count_requested':'particle count',
    'resolution_selection/minimum_particle_count_retained':'particle count',
    'resolution_selection/minimum_selected_mass_msun_h':'Msun/h',
    'resolution_selection/maximum_selected_mass_msun_h':'Msun/h',
    'resolution_selection/selected_halo_count':'halo count',
    'resolution_selection/convergence_completeness_mass_msun_h':'Msun/h',
    'spatial_jackknife/regions':'region count','spatial_jackknife/regions_with_halos':'region count',
    'spatial_jackknife/valid_halo_count':'halo count',
    'spatial_jackknife/sigma_dn_dlog10M_per_mpc_h3':'(Mpc/h)^-3',
    'spatial_jackknife/sigma_dn_dM_h4_per_msun_mpc3':'(Msun/h)^-1 (Mpc/h)^-3',
    'rare_halos/maximum_mass_msun_h':'Msun/h','rare_halos/halo_count':'halo count',
})
for _field in ('input_power_redshift','output_redshift','growth_rescaling','growth_factor_at_input_redshift',
    'growth_factor_at_output_redshift','omega_m0','delta_c','sigma_linear_top_hat',
    'dln_sigma_inverse_dlnM','measurement_over_fit_ratio','input_k_coverage_mask'):
    ANALYSIS_DATASET_UNITS['hmf']['analytic_sheth_tormen/'+_field]='dimensionless'
for _field in ('k_min_h_mpc','k_max_h_mpc'):
    ANALYSIS_DATASET_UNITS['hmf']['analytic_sheth_tormen/'+_field]='h/Mpc'
for _field in ('supported_mass_bin_count','compared_bin_count'):
    ANALYSIS_DATASET_UNITS['hmf']['analytic_sheth_tormen/'+_field]='mass bin count'
ANALYSIS_DATASET_UNITS['hmf'].update({'analytic_sheth_tormen/mass_msun_h':'Msun/h',
    'analytic_sheth_tormen/dn_dlog10M_mpc_h3':'(Mpc/h)^-3'})

FINDER_DATASET_UNITS={
    'mass':'Msun/h','gadget_dn_dlogM':'(Mpc/h)^-3','rockstar_dn_dlogM':'(Mpc/h)^-3',
    'ratio':'dimensionless','host_count_gadget':'halo count','host_count_rockstar':'halo count',
    'per_halo_match_max_separation_mpc_h':'comoving Mpc/h','per_halo_match_radius_particle_spacings':'dimensionless',
    'per_halo_match_gadget_count':'halo count','per_halo_match_rockstar_count':'halo count','per_halo_match_count':'halo count',
    'per_halo_match_gadget_fraction':'dimensionless','per_halo_match_rockstar_fraction':'dimensionless',
    'per_halo_center_separation_mpc_h':'comoving Mpc/h',
    'per_halo_gadget_mass_msun_h':'Msun/h','per_halo_rockstar_mass_msun_h':'Msun/h',
    'per_halo_rockstar_to_gadget_mass_ratio':'dimensionless','per_halo_log10_rockstar_to_gadget_mass_ratio':'dimensionless',
    'gadget_mass_msun_h':'Msun/h','rockstar_mass_msun_h':'Msun/h','rockstar_to_gadget_ratio':'dimensionless',
    'valid_halo_count':'halo count','median_ratio':'dimensionless','ratio_percentile_16_84':'dimensionless',
    'mean_log10_ratio':'dimensionless','rockstar_to_gadget_mass_ratio':'dimensionless','fof_mass_msun_h':'Msun/h','subfind_central_bound_mass_msun_h':'Msun/h',
    'subfind_to_fof_ratio':'dimensionless','rockstar_to_fof_ratio':'dimensionless','subfind_to_rockstar_ratio':'dimensionless',
}
ANALYSIS_IDENTITY_FIELDS={
    'halo_id':'rockstar_halo','per_halo_rockstar_id':'rockstar_halo','rockstar_halo_id':'rockstar_halo',
    'per_halo_gadget_index':'gadget_fof_row','gadget_catalogue_index':'gadget_fof_row','gadget_group_index':'gadget_fof_row',
    'per_halo_rockstar_index':'rockstar_row','rockstar_catalogue_index':'rockstar_row','subfind_central_index':'gadget_subhalo_row',
    'concentration_halo_id':'rockstar_halo','finder_match_gadget_index':'gadget_fof_row',
    'finder_match_rockstar_index':'rockstar_row','finder_match_rockstar_id':'rockstar_halo',
    'halo_catalogue_row':'gadget_fof_row','selected_halo_catalogue_rows':'gadget_fof_row',
}


PROFILE_DATASET_UNITS = {
    'halo_catalogue_row':'epoch-local FOF row','mass_msun_h':'Msun/h',
    'center_group_pos_comoving_mpc_h':'comoving Mpc/h',
    'r200c_comoving_mpc_h':'comoving Mpc/h','radial_bin_edges_comoving_mpc_h':'comoving Mpc/h',
    'radius_comoving_mpc_h':'comoving Mpc/h','profile_inner_radius_comoving_mpc_h':'comoving Mpc/h',
    'profile_outer_radius_comoving_mpc_h':'comoving Mpc/h',
    'dm_mass_shell_msun_h':'Msun/h','gas_mass_shell_msun_h':'Msun/h',
    'estimated_mass_inside_inner_cut_msun_h':'Msun/h','estimated_gas_mass_inside_inner_cut_msun_h':'Msun/h',
    'estimated_gas_mass_in_aperture_msun_h':'Msun/h','estimated_mass_msun_h':'Msun/h',
    'dm_mass_density_comoving_msun_h_per_mpc_h3':'(Msun/h)/(comoving Mpc/h)^3',
    'dm_mass_density_proper_msun_h_per_mpc_h3':'(Msun/h)/(proper Mpc/h)^3',
    'gas_mass_density_comoving_msun_h_per_mpc_h3':'(Msun/h)/(comoving Mpc/h)^3',
    'gas_mass_density_proper_msun_h_per_mpc_h3':'(Msun/h)/(proper Mpc/h)^3',
    'gas_temperature_mass_weighted_K':'K','velocity_dispersion_3d_peculiar_km_s':'km/s',
    'gas_velocity_dispersion_3d_peculiar_km_s':'km/s',
    'tensor_eigenvalues_mpc_h2':'(comoving Mpc/h)^2',
    'principal_axis_eigenvectors_columns':'dimensionless','b_over_a':'dimensionless',
    'c_over_a':'dimensionless','triaxiality':'dimensionless',
}
for _name in ('sampled_particle_count_per_shell','particle_count_estimate_per_shell',
              'sampled_particles_in_aperture','estimated_particles_in_aperture',
              'sampled_particles_inside_inner_cut','estimated_particles_inside_inner_cut',
              'sampled_velocity_particle_count_per_shell','velocity_particle_count_estimate_per_shell',
              'gas_particle_count','gas_particle_count_estimate','gas_temperature_particle_count',
              'gas_temperature_particle_count_estimate','gas_velocity_particle_count',
              'gas_velocity_particle_count_estimate','sampled_particle_count','estimated_particle_count',
              'snapshot_particle_count','sampled_particle_weight_sum','invalid_particle_position_count'):
    PROFILE_DATASET_UNITS[_name]='particle count (inverse-inclusion estimate where indicated)'
PROFILE_DATASET_UNITS['scale_factor']='dimensionless scale factor'
PROFILE_DATASET_UNITS['selected_halo_catalogue_rows']='epoch-local FOF row'


def _analysis_dataset_unit(product, name):
    """Resolve only audited producer paths, never physical units by suffix."""
    key=name.rsplit('/',1)[-1]
    if product=='hmf' and name.startswith('rare_halos/') and key not in ('maximum_mass_msun_h','halo_count'):
        try:
            if float(key)>0:return 'halo count'
        except ValueError:pass
    if product in ('finder','concentration') and key in ANALYSIS_IDENTITY_FIELDS:return 'identity'
    if product in ('finder','concentration') and key.endswith('_available') and key[:-10] in ANALYSIS_IDENTITY_FIELDS:return 'dimensionless'
    if product=='finder':
        if name.startswith('per_halo_center_separation_summary_mpc_h/'):return 'comoving Mpc/h'
        if name.startswith('per_halo_mass_ratio_summary/'):return 'dimensionless'
        return FINDER_DATASET_UNITS.get(key)
    if product=='legacy' and name.startswith('halo_mass_definition_comparison/'):
        if '/mass_msun_h/' in name:return 'Msun/h'
        if '/pairwise_ratios/' in name and key in ('mass_ratio','log10_mass_ratio'):return 'dimensionless'
    if product=='hydro':
        if name.startswith('diagnostics/'):name=name[len('diagnostics/'):]
        return ANALYSIS_DATASET_UNITS['hydro'].get(name)
    if product=='pk' and name.startswith('particle_type_spectra/'):
        return ANALYSIS_DATASET_UNITS['pk'].get(key)
    if product in ('profiles','shape'):
        # Scatter values inherit the measured field, not the percentile label.
        if key in ('lower_16','median','upper_84'):
            if name.startswith('density_profile_scatter/'):
                return PROFILE_DATASET_UNITS['dm_mass_density_proper_msun_h_per_mpc_h3']
            if name.startswith('velocity_profile_scatter/'):
                return 'km/s'
            parts=name.split('/')
            if len(parts)==4 and parts[:2]==['gas_profiles','profile_halo_to_halo_scatter']:
                return PROFILE_DATASET_UNITS.get(parts[2])
            return None
        if key=='radius_over_r200c':return 'dimensionless'
        if key in ('valid_halo_count','selected_halo_count','halo_count','selected_valid_halo_count',
                   'skipped_halo_periodic_geometry_count','skipped_halo_low_particle_count',
                   'skipped_periodic_geometry_count','skipped_low_particle_count'):
            return 'halo count'
        return PROFILE_DATASET_UNITS.get(key)
    if product=='kinematics' and name.startswith(('particle_velocity/','halo_velocity/')):
        if '/native_unit_metadata/' in name:
            return {'scale_factor':'dimensionless scale factor','hubble_param':'dimensionless h',
                    'unit_length_cm':'cm per native code length','unit_velocity_cm_s':'cm/s per native code velocity'}.get(key)
        return {'bin_edges':'km/s','probability_density':'(km/s)^-1',
                'sample_count':'particle count' if name.startswith('particle_velocity/') else 'catalogue object count'}.get(key)
    return ANALYSIS_DATASET_UNITS.get(product,{}).get(name)


def attach_analysis_dataset_units(group, product, *, epoch=None, provenance=None):
    """Attach audited units and source context; explicitly retain unresolved semantics."""
    import h5py
    def annotate(name,dataset):
        if not isinstance(dataset,h5py.Dataset) or dataset.dtype.kind not in 'biufc':
            return
        unit=_analysis_dataset_unit(product,name)
        if product=='hydro_map' or product=='hydro' and ('/hydro_field_maps/fields/' in '/'+name or name.startswith('fields/')):
            key=name.rsplit('/',1)[-1]
            parent=dataset.parent
            declared=parent['units'].asstr()[()] if 'units' in parent else parent.attrs.get('units')
            if key in ('pixel_values','color_range_symmetric','color_range') and declared:
                unit='dimensionless log10(value / ('+declared+'))' if key=='color_range' and 'pixel_values_log10' in parent else declared
            elif key=='pixel_values_log10' and declared:
                unit='dimensionless log10(value / ('+declared+'))'
            elif key=='valid_particle_count':unit='sampled gas particle count'
            elif key=='grid_size':unit='grid cells per axis'
            elif key=='native_to_output_factor' and declared:unit=declared+' per native source-field unit'
            elif '/source_unit_attributes/' in name:
                unit='CGS conversion per native source-field unit' if key=='to_cgs' else 'dimensionless scaling exponent' if key in ('a_scaling','h_scaling') else None
        existing_unit=dataset.attrs.get('units')
        coordinates=None
        if (product=='spatial' and name=='position_mpc_h') or name.rsplit('/',1)[-1]=='center_group_pos_comoving_mpc_h':
            coordinates=coordinate_contract(position_units='comoving Mpc/h',
                origin='global simulation Cartesian box origin',periodic_wrapping='wrapped periodic simulation box',
                velocity_convention='not applicable to position-only dataset')
        # Do not erase specialized contracts (e.g. exact subhalo IDs) when
        # adding snapshot provenance to previously migrated quantities.
        old={key:dataset.attrs[key] for key in ('coordinate_conventions_json','coordinates_status',
             'identity_namespace','quantity_definition','quantity_definition_status','missing_policy',
             'ancestry_json','ancestry_status') if key in dataset.attrs}
        write_dataset_contract(dataset,units=unit or existing_unit or 'UNDECLARED',
            quantity='analysis.py '+product+' producer: '+name if unit is not None else None,
            coordinates=coordinates,epoch=epoch,provenance=provenance,
            missing_policy='NaN is unavailable or estimator has no support; finite nonpositive values are retained')
        if unit is None or 'identity_namespace' in old:
            for key,value in old.items():dataset.attrs[key]=value
        dataset.attrs['units_status']='PRODUCER_DECLARED' if unit is not None or (existing_unit and existing_unit!='UNDECLARED') else 'UNDECLARED'
    if isinstance(group,h5py.Dataset):
        annotate('',group)
    else:
        group.visititems(annotate)
    if not isinstance(group,h5py.Dataset):seal_hdf5_numeric_datasets(group)
    group.attrs['dataset_contract_scope']='Migrated producer quantities only; UNDECLARED datasets require further audit'



def attach_analysis_identity_contracts(group,product,*,snapshot,identity_namespaces,epoch,provenance):
    """Canonicalize only audited finder identity fields, preserving missing links."""
    import h5py,uuid
    if product not in ('finder','concentration','profiles','shape','legacy'):return
    paths=[]
    group.visititems(lambda name,item:paths.append(name) if isinstance(item,h5py.Dataset)
        and name.rsplit('/',1)[-1] in ANALYSIS_IDENTITY_FIELDS else None)
    for name in paths:
        original=group[name];key=name.rsplit('/',1)[-1]
        values,available=canonical_native_links(original[()])
        parent=original.parent;temporary='.canonical-id-'+uuid.uuid4().hex
        target=parent.create_dataset(temporary,data=values,chunks=True,fletcher32=True) if values.ndim and values.size else parent.create_dataset(temporary,data=values)
        for attribute,value in original.attrs.items():target.attrs[attribute]=value
        del parent[key];parent.move(temporary,key);target=parent[key]
        kind=ANALYSIS_IDENTITY_FIELDS[key]
        namespace=identity_namespaces['halo']+':snapshot:'+str(snapshot)+':'+kind
        if kind=='gadget_subhalo_row':namespace=identity_namespaces['subhalo']+':snapshot:'+str(snapshot)+':gadget_subhalo_row'
        mask=parent.create_dataset(key+'_available',data=available,chunks=True,fletcher32=True) if available.ndim and available.size else parent.create_dataset(key+'_available',data=available)
        write_dataset_contract(mask,units='dimensionless',quantity='Availability of '+name,epoch=epoch,provenance=provenance)
        write_dataset_contract(target,units='identity',quantity='Epoch-local '+kind,identity_namespace=namespace,
            epoch=epoch,provenance=provenance,missing_policy='uint64 zero placeholder only when '+key+'_available is false',
            ancestry={'semantics':'Finder match is a positional association, not cross-finder identity equivalence or temporal persistence',
                'finder_identity_domain':kind,'snapshot_storage_index':int(snapshot)})


def attach_legacy_analysis_units(group, *, epoch=None, provenance=None):
    """Explicit field aliases written by the existing full-analysis HMF/P(k) route."""
    import h5py
    # Every numeric legacy field receives context, including unresolved fields.
    attach_analysis_dataset_units(group,'legacy',epoch=epoch,provenance=provenance)
    aliases={'hmf_mass':'mass_msun_h','hmf_mass_bin_edges_msun_h':'mass_bin_edges_msun_h',
             'hmf_counts':'counts','hmf_dn_dlogM':'dn_dlogM','hmf_dn_dM':'dn_dM'}
    units={key:ANALYSIS_DATASET_UNITS['hmf'][value] for key,value in aliases.items()}
    for kind in ['lower','upper']:
        units['hmf_poisson_68_'+kind+'_dn_dM']=ANALYSIS_DATASET_UNITS['hmf']['dn_dM']
        units['hmf_poisson_68_'+kind+'_dn_dlogM']=ANALYSIS_DATASET_UNITS['hmf']['dn_dlogM']
    units['hmf_spatial_jackknife_sigma_dn_dM']=ANALYSIS_DATASET_UNITS['hmf']['dn_dM']
    units['hmf_spatial_jackknife_sigma_dn_dlog10M']=ANALYSIS_DATASET_UNITS['hmf']['dn_dlogM']
    pk_aliases={'pk_k':'k_h_Mpc','pk_P':'P_Mpc_h3','pk_delta2':'delta2','pk_mode_count':'mode_count',
                'pk_fold_segment':'fold_segment','pk_delta2_raw_shot_included':'delta2_raw_shot_included',
                'pk_delta2_shot_noise':'delta2_shot_noise','pk_P_raw_shot_included':'P_raw_shot_included_Mpc_h3',
                'pk_P_shot_noise':'P_shot_noise_Mpc_h3','pk_delta2_gaussian_1sigma':'delta2_gaussian_1sigma',
                'pk_P_gaussian_1sigma':'P_gaussian_1sigma_Mpc_h3','pk_P_native_column':'P_native_column_Mpc_h3',
                'pk_native_normalization_relative_residual':'native_column_normalization_relative_residual',
                'pk_fold_band_lower_h_Mpc':'fold_band_lower_h_Mpc','pk_fold_band_upper_h_Mpc':'fold_band_upper_h_Mpc'}
    units.update({key:ANALYSIS_DATASET_UNITS['pk'][value] for key,value in pk_aliases.items()})
    explicit_aliases={
        'concentration_mass_msun_h':('concentration','mass_msun_h'),
        'concentration_cvir':('concentration','concentration_cvir'),
        'host_halo_positions_mpc_h':('spatial','position_mpc_h'),
        'host_halo_mass_spatial_msun_h':('spatial','mass_msun_h'),
        'subhmf_mass':('subhmf','mass_msun_h'),'subhmf_dn':('subhmf','dn_dlogM'),
        'xi_r':('correlation','r'),'xi':('correlation','xi'),
        'xi_bin_edges':('correlation','bin_edges'),'xi_pair_count':('correlation','pair_count'),
        'xi_random_pair_expectation':('correlation','random_pair_expectation'),
        'xi_poisson_68_lower':('correlation','xi_poisson_68_lower'),
        'xi_poisson_68_upper':('correlation','xi_poisson_68_upper'),
        'bias_k':('bias','k_h_Mpc'),'bias':('bias','bias'),
        'bias_gaussian_68_sigma':('bias','bias_gaussian_68_sigma'),
        'bias_auto_hh_mm':('bias','bias_auto_hh_mm'),
        'bias_auto_hh_mm_gaussian_68_sigma':('bias','bias_auto_hh_mm_gaussian_68_sigma'),
        'bias_cross_correlation_coefficient_hm':('bias','cross_correlation_coefficient_hm'),
        'bias_cross_power_Mpc_h3':('bias','cross_power_Mpc_h3'),
        'bias_matter_power_Mpc_h3':('bias','matter_power_Mpc_h3'),
        'bias_halo_power_Mpc_h3':('bias','halo_power_Mpc_h3'),
        'bias_mode_count':('bias','mode_count'),'bias_bin_edges_h_Mpc':('bias','bin_edges_h_Mpc'),
    }
    units.update({key:ANALYSIS_DATASET_UNITS[product][field] for key,(product,field) in explicit_aliases.items()})
    units.update({'finder_comparison_mass':'Msun/h','finder_comparison_gadget_dn_dlogM':'(Mpc/h)^-3',
        'finder_comparison_rockstar_dn_dlogM':'(Mpc/h)^-3','finder_comparison_ratio':'dimensionless',
        'finder_match_center_separation_mpc_h':'comoving Mpc/h','finder_match_gadget_mass_msun_h':'Msun/h',
        'finder_match_rockstar_mass_msun_h':'Msun/h','finder_match_rockstar_to_gadget_mass_ratio':'dimensionless',
        'finder_match_log10_rockstar_to_gadget_mass_ratio':'dimensionless','cumulative_mass':'Msun/h',
        'cumulative_n_gt':'(Mpc/h)^-3','shape_mass_m200c_msun_h':'Msun/h','shape_b_over_a':'dimensionless',
        'shape_c_over_a':'dimensionless','shape_triaxiality':'dimensionless','shape_dm_particle_count':'particle count'})
    units.update({'dm_speed_bin_edges':'km/s','dm_speed_probability_density':'(km/s)^-1'})
    units.update({'halo_speed_bin_edges':'km/s','halo_speed_probability_density':'(km/s)^-1',
                  'specific_angular_momentum':'proper kpc km/s'})
    units.update(ANALYSIS_DATASET_UNITS['hydro'])
    for name,unit in units.items():
        if name in group and isinstance(group[name],h5py.Dataset):
            coordinates=coordinate_contract(position_units='comoving Mpc/h',origin='global simulation Cartesian box origin',
                periodic_wrapping='wrapped periodic simulation box',velocity_convention='not applicable to position-only dataset') if name=='host_halo_positions_mpc_h' else None
            write_dataset_contract(group[name],units=unit,epoch=epoch,provenance=provenance,coordinates=coordinates,
                quantity='existing legacy analysis: '+name,
                missing_policy='NaN is unavailable; nonpositive shot-subtracted signal is retained')
            group[name].attrs['units_status']='PRODUCER_DECLARED'
            group[name].attrs['quantity_definition']='existing legacy analysis: '+name
    for name in ('finder_match_mass_definitions','finder_match_subfind_central'):
        if name in group:attach_analysis_dataset_units(group[name],'finder',epoch=epoch,provenance=provenance)
    if 'pk_particle_type_spectra' in group:
        for type_group in group['pk_particle_type_spectra'].values():
            attach_analysis_dataset_units(type_group,'pk',epoch=epoch,provenance=provenance)
    for name,product in [('halo_profiles','profiles'),('velocity_dispersion_profiles','profiles'),
                         ('gas_profiles','profiles'),('halo_shape_profiles','shape')]:
        if name in group:
            attach_analysis_dataset_units(group[name],product,epoch=epoch,provenance=provenance)
    if 'hydro_field_maps' in group:
        for target in group['hydro_field_maps'].values():
            if isinstance(target,h5py.Group):attach_analysis_dataset_units(target,'hydro_map',epoch=epoch,provenance=provenance)


TIME_SERIES_DATASET_UNITS = {
    'snapshot':'storage index (not physical time)','final_snapshot':'storage index (not physical time)',
    'redshift':'dimensionless redshift','scale_factor':'dimensionless scale factor',
    'total_sfr_msun_per_year':'Msun/year','star_particle_count':'star population particle count',
    'age_distribution_bin_edges_gyr':'Gyr','age_distribution_estimated_particle_count':'estimated star population particle count',
    'age_distribution_surviving_mass_msun_h':'Msun/h','birth_cohort_formation_time_bin_edges_gyr':'Gyr',
    'birth_cohort_particle_count':'star population particle count','birth_cohort_stellar_mass_msun_h':'Msun/h',
    'mean_age_gyr_mass_weighted':'Gyr','median_age_gyr_mass_weighted':'Gyr',
    'young_mass_fraction_lt_100_myr':'dimensionless','estimated_particle_count':'estimated star population particle count',
    'estimated_stellar_mass_msun_h':'Msun/h','mass_msun_h':'Msun/h','final_mass_msun_h':'Msun/h',
    'mvir_msun_h':'Msun/h',
    'main_mass_msun_h':'Msun/h','secondary_mass_msun_h':'Msun/h','mass_ratio':'dimensionless',
    'formation_redshift':'dimensionless redshift','particle_floor':'particle count',
    'halo_id':'identity','final_halo_id':'identity','original_halo_id':'identity',
    'descendant_id':'identity','main_progenitor_id':'identity','secondary_progenitor_id':'identity',
    'original_halo_id_available':'boolean availability mask',
    'original_halo_id_native':'native signed identity/link; -1 is unavailable',
}


def canonical_halo_history_payload(value):
    """Preserve exact halo identities, including explicit availability for native -1 links."""
    if isinstance(value,dict):
        result={key:canonical_halo_history_payload(item) for key,item in value.items()}
        for key in ('halo_id','final_halo_id','original_halo_id','descendant_id',
                    'main_progenitor_id','secondary_progenitor_id'):
            if key not in value:continue
            item=value[key]
            if key=='original_halo_id' and isinstance(item,(int,np.integer)) and int(item)==-1:
                result[key]=np.uint64(0)
                result['original_halo_id_available']=False
                result['original_halo_id_native']=np.int64(-1)
            else:
                result[key]=exact_uint64_ids([item])[0]
                if key=='original_halo_id':result['original_halo_id_available']=True
        return result
    if isinstance(value,(list,tuple)):return [canonical_halo_history_payload(item) for item in value]
    return value


def attach_time_series_contract(group, *, clock=None, snapshot_epochs=None, provenance=None,
                                identity_namespaces=None):
    """Attach actual row epochs to histories; never convert storage labels into time."""
    import h5py
    seal_hdf5_numeric_datasets(group)
    epochs=snapshot_epochs or {}
    namespaces=identity_namespaces or {}
    def row_epoch(parent):
        if parent.name.rsplit('/',1)[-1]=='omitted_snapshots':
            return {'representation':'omitted measurements; storage labels only',
                    'epochs_by_snapshot':epochs,'storage_indices_are_not_time':True}
        node=parent
        while node.name.startswith(group.name):
            if 'redshift' in node and isinstance(node['redshift'],h5py.Dataset):
                redshift=np.asarray(node['redshift'][()])
                if redshift.ndim>1:raise ValueError('History epoch coordinate must be scalar or a vector')
                rows=np.atleast_1d(redshift)
                if not np.isfinite(rows).all() or np.any(rows<0):
                    raise ValueError('History redshift must be finite and nonnegative')
                contracts=[epoch_contract(1/(1+float(z)),clock=clock) for z in rows]
                return contracts[0] if redshift.ndim==0 else {'representation':'row-aligned actual epochs','rows':contracts}
            if 'redshift' in node.attrs:
                z=float(node.attrs['redshift'])
                if not np.isfinite(z) or z<0:raise ValueError('History redshift must be finite and nonnegative')
                return epoch_contract(1/(1+z),clock=clock)
            if node==group:break
            node=node.parent
        return {'representation':'multi-epoch history','epochs_by_snapshot':epochs,
                'storage_indices_are_not_time':True}
    def annotate(name,dataset):
        if not isinstance(dataset,h5py.Dataset):return
        key=name.rsplit('/',1)[-1]
        unit=TIME_SERIES_DATASET_UNITS.get(key)
        namespace=None
        if unit=='identity':
            if dataset.dtype!=np.dtype('uint64'):raise ValueError('Canonical history identity must be uint64: '+dataset.name)
            namespace=namespaces.get('halo') if key=='original_halo_id' else namespaces.get('tree_node')
        epoch=row_epoch(dataset.parent)
        if key=='final_halo_id' and 'rows' in epoch:
            epoch=epoch['rows'][-1]
        if key.startswith('age_distribution_') and 'age_distribution_snapshot' in group.attrs:
            epoch=epochs.get(str(int(group.attrs['age_distribution_snapshot'])),
                {'status':'UNAVAILABLE','reason':'No actual epoch supplied for the age-distribution storage index'})
        if key.startswith('birth_cohort_') and 'birth_cohort_snapshot' in group.attrs:
            epoch=epochs.get(str(int(group.attrs['birth_cohort_snapshot'])),
                {'status':'UNAVAILABLE','reason':'No actual epoch supplied for the birth-cohort storage index'})
        write_dataset_contract(dataset,units=unit or ('text metadata' if dataset.dtype.kind in 'OSU' else 'UNDECLARED'),
            epoch=epoch,provenance=provenance,identity_namespace=namespace,
            quantity='saved history '+name if unit else None,
            ancestry={'relation':'native descendant links select maximum-mass main progenitor; no positional links',
                'target_namespace':namespaces.get('tree_node'),'original_halo_id_requires_availability_mask':True} if unit=='identity' else None,
            missing_policy='NaN is unavailable; original_halo_id zero is meaningful only when its availability mask is true')
        dataset.attrs['units_status']='PRODUCER_DECLARED' if unit or dataset.dtype.kind in 'OSU' else 'UNDECLARED'
    group.visititems(annotate)


def exact_uint64_ids(values, *, unique=True, allow_empty=True):
    """Reject lossy coercions before conversion, including mixed Python int lists."""
    if isinstance(values, (list, tuple)):
        if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer))
               or not 0 <= int(v) < 2**64 for v in values):
            raise ValueError('IDs must be exact uint64 integers without truncation')
        array = np.asarray(values, dtype=np.uint64)
    else:
        array = np.asarray(values)
        if array.dtype.kind not in 'iu' or np.any(array < 0):
            raise ValueError('IDs must be exact uint64 integers without truncation')
        array = array.astype(np.uint64, copy=False)
    if array.ndim != 1 or (not allow_empty and not len(array)):
        raise ValueError('IDs must be a nonempty vector' if not allow_empty else 'IDs must be a vector')
    if unique and len(np.unique(array)) != len(array):
        raise ValueError('duplicate IDs in identity namespace')
    return array


def canonical_native_links(values, *, missing_sentinel=-1):
    """Preserve exact native sentinel links as uint64 values plus availability."""
    if isinstance(missing_sentinel,bool) or not isinstance(missing_sentinel,int) or missing_sentinel>=0:
        raise ValueError('Native missing-link sentinel must be a declared negative integer')
    array=np.asarray(values)
    if array.dtype.kind not in 'iu':raise ValueError('Native identity links must be exact integers')
    available=array!=missing_sentinel if array.dtype.kind=='i' else np.ones(array.shape,dtype=bool)
    if np.any(array[available]<0):raise ValueError('Undeclared negative native identity sentinel')
    canonical=np.zeros(array.shape,dtype=np.uint64)
    canonical[available]=array[available].astype(np.uint64)
    return canonical,available


def validate_signed_native_particle_ids(paths, species=1, *, chunk_rows=250000):
    """Reject unsigned IDs outside the pinned signed native reader's domain."""
    import h5py
    if type(species) is not int or not 0 <= species <= 5:
        raise ValueError('Native species must be an integer from zero through five')
    if isinstance(chunk_rows,bool) or not isinstance(chunk_rows,int) or chunk_rows<1:
        raise ValueError('Native ID workspace must have a positive integer row bound')
    checked=0
    for path in paths:
        with h5py.File(path,'r') as source:
            group=source.get('PartType'+str(species))
            if group is None:continue
            dataset=group['ParticleIDs']
            if dataset.ndim!=1 or dataset.dtype.kind not in 'iu':
                raise ValueError('Native particle identity must be an exact integer vector')
            for first in range(0,len(dataset),chunk_rows):
                ids=dataset[first:first+chunk_rows]
                if np.any(ids<0) or np.any(ids>np.iinfo(np.int64).max):
                    raise ValueError('Pinned native signed-int64 particle reader cannot represent these uint64 IDs')
                checked+=len(ids)
    return {'checked_particle_ids':checked,'native_domain':'nonnegative signed int64',
        'science_acceptance':'NOT_ACCEPTED'}


def exact_id_join(requested, available, *, requested_namespace, available_namespace, require_all=True):
    """Exact namespace-aware join; missing IDs are -1 only when explicitly allowed."""
    if (not isinstance(requested_namespace, str) or not requested_namespace.strip()
            or not isinstance(available_namespace, str) or requested_namespace != available_namespace):
        raise ValueError('ID joins require one identical nonempty identity namespace')
    if type(require_all) is not bool:
        raise ValueError('ID joins require an explicit boolean missing-identity policy')
    left, right = exact_uint64_ids(requested), exact_uint64_ids(available)
    order = np.argsort(right)
    slots = np.searchsorted(right[order], left)
    present = slots < len(right)
    present[present] &= right[order[slots[present]]] == left[present]
    if require_all and not np.all(present):
        raise ValueError('ID join has missing requested identities')
    result = np.full(len(left), -1, dtype=np.int64)
    result[present] = order[slots[present]]
    return result


def retired_black_hole_carriers(species, ids, masses):
    """Mark only the native species-5/ID-zero/mass-zero retirement tuple.

    ID zero with positive mass remains a live identity. Other zero-mass rows
    are not retired BHs and must still fail the owning reader's mass checks.
    """
    ids, masses = np.asarray(ids), np.asarray(masses)
    if (type(species) is not int or not 0 <= species <= 5
            or ids.ndim != 1 or ids.dtype.kind not in 'iu' or np.any(ids < 0)
            or masses.shape != ids.shape or masses.dtype.kind not in 'fiu'):
        raise ValueError('Aligned exact native identities, real masses and species 0..5 required')
    return (species == 5) & (ids == 0) & (masses == 0)


def read_species_phase_space(group, *, epoch, memory_bytes, start=0, stop=None):
    """Read one bounded native species slice in declared physical conventions.

    The owning consumer must validate and hash-bind the complete snapshot set
    before and after all slices. This adapter verifies the slice's native unit,
    epoch, identity and mass contracts without sampling or inferring physics.
    Cosmological and isolated modes retain distinct clocks and coordinates.
    Isolated velocities have no sqrt(a) factor or Hubble-parameter division.
    Retired type-5 carriers remain explicitly marked, never counted as live BHs.
    """
    import re
    from core.foundation import KPC_CM, MSUN_G
    match = re.fullmatch(r'/PartType([0-5])',group.name)
    if match is None:
        raise ValueError('A native PartType0..5 group is required')
    species = int(match.group(1)); source = group.file
    header,parameters = source['Header'].attrs,source['Parameters'].attrs
    from core.foundation import SECONDS_PER_YEAR
    mode = hdf5_scalar_attr(parameters.get('ComovingIntegrationOn'),'ComovingIntegrationOn')
    if mode not in (0,1):
        raise ValueError('Explicit native comoving integration mode 0 or 1 required')
    comoving = mode==1
    time = hdf5_scalar_attr(header['Time'],'Header/Time')
    if not isinstance(epoch,dict):
        raise ValueError('Explicit native epoch contract required')
    units = {name:hdf5_scalar_attr(parameters[name],name) for name in
             ['UnitLength_in_cm','UnitMass_in_g','UnitVelocity_in_cm_per_s']}
    if any(not np.isfinite(v) or v<=0 for v in units.values()):
        raise ValueError('Finite positive native length/mass/velocity units required')
    if comoving:
        if epoch.get('canonical_time')!='scale_factor' or epoch.get('actual_scale_factor')!=time:
            raise ValueError('Phase-space epoch differs from actual saved scale factor')
        a = time;h = hdf5_scalar_attr(parameters['HubbleParam'],'Parameters/HubbleParam')
        for name,expected in [('UnitLength_in_cm',KPC_CM),('UnitMass_in_g',1e10*MSUN_G),
                              ('UnitVelocity_in_cm_per_s',1e5)]:
            if not np.isclose(units[name],expected,rtol=1e-8,atol=0):
                raise ValueError('Cosmological phase-space route requires pinned native units: '+name)
        output_epoch = dict(epoch)
    else:
        if epoch.get('canonical_time')!='isolated_native_elapsed_time' or epoch.get('native_time_code')!=time:
            raise ValueError('Isolated phase-space requires the actual native elapsed-time epoch')
        a,h = 1.,1.  # Unit exponents are zero in pinned isolated native output.
        time_unit = units['UnitLength_in_cm']/units['UnitVelocity_in_cm_per_s']/(1e9*SECONDS_PER_YEAR)
        declared = epoch.get('time_unit_gyr')
        if declared is not None and not np.isclose(declared,time_unit,rtol=1e-10,atol=0):
            raise ValueError('Isolated clock conversion contradicts native length/velocity units')
        output_epoch = {**epoch,**isolated_epoch_contract(time,time_unit_gyr=time_unit,
            snapshot=epoch.get('snapshot_storage_index'))}
    count = len(group['ParticleIDs'])
    if stop is None:stop=count
    if (type(start) is not int or type(stop) is not int or not 0<=start<=stop<=count
            or type(memory_bytes) is not int or memory_bytes<=0):
        raise ValueError('Bounded integer row slice and memory budget required')
    if (stop-start)*256+4096>memory_bytes:
        raise MemoryError('Native phase-space slice exceeds declared workspace')
    if int(header['NumPart_ThisFile'][species])!=count:
        raise ValueError('Native species count differs from the saved header')
    factors = ({'Coordinates':native_unit_factor('position_proper_kpc',scale_factor=a,hubble_param=h),
                'Velocities':native_unit_factor('particle_velocity_peculiar_kms',scale_factor=a,hubble_param=h),
                'Masses':native_unit_factor('mass_msun',scale_factor=a,hubble_param=h)} if comoving else
               {'Coordinates':units['UnitLength_in_cm']/KPC_CM,
                'Velocities':units['UnitVelocity_in_cm_per_s']/1e5,
                'Masses':units['UnitMass_in_g']/MSUN_G})
    cgs = {'Coordinates':KPC_CM,'Velocities':1e5,'Masses':MSUN_G}
    for field in ['Coordinates','Velocities','ParticleIDs']:
        expected_shape = (count,) if field=='ParticleIDs' else (count,3)
        if field not in group or group[field].shape!=expected_shape:
            raise ValueError('Complete aligned native phase-space field required: '+field)
    table = np.asarray(header['MassTable'],dtype=float)
    if table.shape!=(6,) or not np.isfinite(table).all() or np.any(table<0):
        raise ValueError('Six finite nonnegative native mass-table entries required')
    if 'Masses' in group:
        if group['Masses'].shape!=(count,) or table[species]!=0:
            raise ValueError('Variable masses conflict with header mass table or row count')
    elif table[species]<=0:
        raise ValueError('Native species has neither variable masses nor a positive fixed mass')
    for field,factor in factors.items():
        if field not in group:continue
        if group[field].dtype.kind not in 'fiu':
            raise ValueError('Physical native field must be real numeric: '+field)
        if not comoving and any(group[field].attrs.get(key)!=0 for key in ('a_scaling','h_scaling')):
            raise ValueError('Isolated native field cannot use cosmological a/h scaling: '+field)
        actual = dataset_to_cgs_factor(group[field].attrs,a,h)/cgs[field]
        if not np.isclose(actual,factor,rtol=3e-5,atol=0):
            raise ValueError('Native phase-space field contradicts pinned units: '+field)
    rows = slice(start,stop)
    ids = exact_uint64_ids(group['ParticleIDs'][rows],unique=False)
    position = np.asarray(group['Coordinates'][rows],dtype=float)
    velocity = np.asarray(group['Velocities'][rows],dtype=float)*factors['Velocities']
    mass = (np.asarray(group['Masses'][rows],dtype=float) if 'Masses' in group
            else np.full(stop-start,table[species]))*factors['Masses']
    with np.errstate(over='ignore',invalid='ignore'):
        proper_position=position*factors['Coordinates']
    retired = retired_black_hole_carriers(species, ids, mass)
    if (not np.isfinite(position).all() or not np.isfinite(velocity).all()
            or not np.isfinite(proper_position).all() or not np.isfinite(mass).all() or np.any((mass<=0)&~retired)):
        raise ValueError('Invalid live native phase-space mass, identity or coordinates')
    # Native lattice identities may legitimately start at zero. Retirement is
    # the complete species5/id0/mass0 tuple, never identity zero alone.
    exact_uint64_ids(ids[~retired])
    return {'particle_ids':ids,**({'position_comoving_kpc_h':position} if comoving else {}),
        'position_proper_kpc':proper_position,
        'velocity_peculiar_kms':velocity,'mass_msun':mass,'live_supported':~retired,
        'species':species,'epoch':output_epoch,
        'coordinates':coordinate_contract(position_units='comoving kpc/h' if comoving else 'proper kpc',
            velocity_convention='physical peculiar km/s; saved velocity times sqrt(a)' if comoving else 'physical inertial km/s; no expansion',
            origin='global simulation Cartesian origin',periodic_wrapping='native saved image; no implicit unwrapping')}


def native_box_proper_kpc(header, parameters):
    """Use the saved box, native length unit and explicit integration mode.

    Isolated lengths have no cosmological a/h factor, including when an
    unused HubbleParam is present in the input parameter table.
    """
    from core.foundation import KPC_CM
    box=hdf5_scalar_attr(header['BoxSize'],'Header/BoxSize')
    length=hdf5_scalar_attr(parameters['UnitLength_in_cm'],'UnitLength_in_cm')
    mode=hdf5_scalar_attr(parameters['ComovingIntegrationOn'],'ComovingIntegrationOn')
    if mode not in (0,1) or not np.isfinite(box) or box<=0 or not np.isfinite(length) or length<=0:
        raise ValueError('Positive saved box/length unit and explicit integration mode required')
    factor=1.
    if mode==1:
        factor=native_unit_factor('position_proper_kpc',
            scale_factor=hdf5_scalar_attr(header['Time'],'Header/Time'),
            hubble_param=hdf5_scalar_attr(parameters['HubbleParam'],'HubbleParam'))
    with np.errstate(over='ignore',invalid='ignore',under='ignore'):
        result=float(np.float64(box)*(length/KPC_CM)*factor)
    if not np.isfinite(result) or result<=0:
        raise ValueError('Native proper box conversion overflowed or underflowed')
    return result


def apply_recorded_periodic_images(position_proper_kpc, image_offsets, *, box_proper_kpc_by_epoch):
    """Apply explicit integer images once, using each saved epoch's proper box.

    Supports T by 3 and T by N by 3 histories. The caller verifies the image
    record's namespace, identities, saved times and history binding. This
    operation never estimates winding from adjacent wrapped positions.
    """
    position = np.asarray(position_proper_kpc)
    images = np.asarray(image_offsets)
    boxes = np.asarray(box_proper_kpc_by_epoch)
    if (position.dtype.kind not in 'fiu' or position.ndim not in (2,3) or position.shape[-1] != 3
            or images.dtype.kind not in 'iu' or images.shape != position.shape
            or boxes.dtype.kind not in 'fiu' or boxes.shape != (position.shape[0],)):
        raise ValueError('Aligned physical positions, exact integer images and one proper box per epoch required')
    position = np.asarray(position,dtype=float); boxes = np.asarray(boxes,dtype=float)
    if not np.isfinite(position).all() or not np.isfinite(boxes).all() or np.any(boxes <= 0):
        raise ValueError('Finite positions and positive proper periodic boxes required')
    # A float64 position cannot distinguish adjacent whole-box image integers
    # outside this exact-integer range. Preserve the original integers rather
    # than quietly rounding an unsupported recorded spatial image.
    if np.any(images > 2**53) or (images.dtype.kind == 'i' and np.any(images < -(2**53))):
        raise ValueError('Recorded image integer exceeds exact float64 coordinate conversion support')
    shape = (len(boxes),)+(1,)*(position.ndim-1)
    with np.errstate(over='ignore',invalid='ignore'):
        result = position + images.astype(float)*boxes.reshape(shape)
    if not np.isfinite(result).all():
        raise ValueError('Recorded periodic image coordinates overflow')
    return result


def proper_position_derivative_kms(position_proper_kpc, velocity_peculiar_kms, *,
                                   hubble_kms_mpc, origin_proper_kpc,
                                   origin_velocity_peculiar_kms):
    """Derivative of proper Cartesian separation from the declared origin.

    For r=a(x-x_origin), dr/dt=v_pec-v_origin+H*r. The caller supplies
    recorded spatial images before this conversion. This function does not
    infer winding, change identities, rotate frames, or convert native v.
    H uses km/s/Mpc; r uses proper kpc; the output uses km/s.
    """
    position = np.asarray(position_proper_kpc, dtype=float)
    velocity = np.asarray(velocity_peculiar_kms, dtype=float)
    origin = np.asarray(origin_proper_kpc, dtype=float)
    origin_velocity = np.asarray(origin_velocity_peculiar_kms, dtype=float)
    if position.ndim < 1 or position.shape[-1] != 3 or velocity.shape != position.shape:
        raise ValueError('Aligned Cartesian positions and peculiar velocities required')
    if origin.shape != (3,) or origin_velocity.shape != (3,):
        raise ValueError('One declared proper origin and peculiar origin velocity required')
    if any(not np.isfinite(value).all() for value in (position, velocity, origin, origin_velocity)):
        raise ValueError('Finite proper position/frame data required')
    if (isinstance(hubble_kms_mpc, (bool, np.bool_)) or not isinstance(hubble_kms_mpc, (int, float, np.integer, np.floating))
            or not np.isfinite(hubble_kms_mpc) or hubble_kms_mpc < 0):
        raise ValueError('Explicit finite nonnegative expansion rate required; isolated uses zero')
    with np.errstate(over='ignore',invalid='ignore'):
        derivative = velocity - origin_velocity + hubble_kms_mpc * (position - origin) / 1000.
    if not np.isfinite(derivative).all():
        raise ValueError('Proper-position derivative overflows its declared physical frame')
    return derivative


def relative_proper_state(offsets_proper_kpc, particle_velocity_peculiar_kms,
                          host_velocity_peculiar_kms, *, hubble_kms_mpc):
    """Proper separation derivative, including H*r; origin particles are unresolved.

    Inputs have already passed their native dataset/unit conversion contracts.
    The host velocity must refer to the same declared center/frame as offsets.
    """
    from core.foundation import KPC_CM, SECONDS_PER_YEAR
    position = np.asarray(offsets_proper_kpc, dtype=float)
    velocity = np.asarray(particle_velocity_peculiar_kms, dtype=float)
    host = np.asarray(host_velocity_peculiar_kms, dtype=float)
    if position.ndim != 2 or position.shape[1] != 3 or velocity.shape != position.shape or host.shape != (3,):
        raise ValueError('Aligned proper positions and physical peculiar velocities required')
    if not np.isfinite(position).all() or not np.isfinite(velocity).all() or not np.isfinite(host).all():
        raise ValueError('Finite relative phase space required')
    if isinstance(hubble_kms_mpc, (bool, np.bool_)) or not np.isfinite(hubble_kms_mpc) or hubble_kms_mpc < 0:
        raise ValueError('Declared nonnegative H in km/s/Mpc required; zero denotes isolated expansion')
    radius = np.linalg.norm(position, axis=1)
    supported = radius > 0
    derivative = np.full(len(radius), np.nan)
    peculiar = np.sum(position[supported]*(velocity[supported]-host), axis=1)/radius[supported]
    derivative[supported] = (peculiar+hubble_kms_mpc*radius[supported]/1000.)*(1e5*1e6*SECONDS_PER_YEAR/KPC_CM)
    return {'radius_proper_kpc': radius, 'radial_derivative_kpc_myr': derivative,
            'supported': supported, 'velocity_definition': 'proper-time radial separation derivative; H*r plus relative peculiar velocity'}


PINNED_NATIVE_FACTORS = {'position_comoving_mpc_h':1/1000.,'mass_msun_h':1.e10}


def convert_native_catalogue_kinematics(values, field, *, scale_factor, hubble_param,
                                      unit_length_cm, unit_velocity_cm_s):
    """Convert pinned GADGET FOF/SUBFIND fields using actual native unit metadata.

    fof_finish_group_properties stores mass-averaged canonical a*v_pec in
    GroupVel; subfind_properties divides internal velocity by a before storing
    SubhaloVel. SubhaloSpin uses proper native-length offsets crossed with
    peculiar velocity (the radial Hubble term has zero cross product).
    """
    from core.foundation import KPC_CM
    native_unit_factor('catalogue_velocity_peculiar_kms',scale_factor=scale_factor,hubble_param=hubble_param)
    for value in (unit_length_cm,unit_velocity_cm_s):
        if isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,float,np.integer,np.floating)) or not np.isfinite(value) or value<=0:
            raise ValueError('Catalogue kinematics requires finite positive native unit metadata')
    array=np.asarray(values)
    if array.dtype.kind not in 'fiu' or array.ndim!=2 or array.shape[1]!=3:
        raise ValueError('Catalogue vector must be a numeric N by 3 array')
    factors={'GroupVel':unit_velocity_cm_s/1.e5/scale_factor,
        'SubhaloVel':unit_velocity_cm_s/1.e5,
        'SubhaloSpin':unit_length_cm/KPC_CM/hubble_param*unit_velocity_cm_s/1.e5}
    if field not in factors:raise ValueError('Unsupported native catalogue convention: '+str(field))
    with np.errstate(over='raise',invalid='ignore'):
        return np.asarray(array,dtype=float)*factors[field]


def convert_pinned_native(values, quantity, *, inverse=False):
    """Convert declared pinned units with unchanged h scaling and no epoch guess.

    These two conversions need no a or h: kpc/h -> Mpc/h and
    1e10 Msun/h -> Msun/h. NaNs remain NaNs for explicit missing-data handling.
    Metadata validation belongs to the saved-input reader; this does not infer
    a unit convention from values, box size, or filenames.
    """
    if quantity not in PINNED_NATIVE_FACTORS:
        raise ValueError('Conversion needs epoch/h or an unsupported pinned quantity: '+str(quantity))
    array=np.asarray(values)
    if array.dtype.kind not in 'fiu':
        raise ValueError('Native physical quantities must be numeric, not boolean or text')
    array=np.asarray(array,dtype=float)
    factor=PINNED_NATIVE_FACTORS[quantity]
    if quantity=='position_comoving_mpc_h':
        return array*(1/factor) if inverse else array/(1/factor)
    return array/factor if inverse else array*factor


def native_unit_factor(quantity, *, scale_factor, hubble_param):
    """Pinned native kpc/h, 1e10 Msun/h, km/s conventions in one conversion layer."""
    a, h = scale_factor, hubble_param
    if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, float, np.integer, np.floating))
           or not np.isfinite(v) or v <= 0 for v in (a,h)):
        raise ValueError('Unit conversion needs finite positive a and h')
    # Evaluate only the requested conversion. A finite clock must not overflow
    # an unrelated quantity while converting, for example, catalogue velocity.
    a, h = np.float64(a), np.float64(h)
    factors = {'position_proper_kpc':lambda:a/h, 'position_comoving_kpc':lambda:1/h,
               'position_comoving_mpc_h':lambda:PINNED_NATIVE_FACTORS['position_comoving_mpc_h'],
               'mass_msun':lambda:PINNED_NATIVE_FACTORS['mass_msun_h']/h,
               'mass_msun_h':lambda:PINNED_NATIVE_FACTORS['mass_msun_h'],
               'particle_velocity_peculiar_kms':lambda:np.sqrt(a),
               'catalogue_velocity_peculiar_kms':lambda:1.}
    if quantity not in factors:
        raise ValueError('Unknown native quantity/unit convention: '+str(quantity))
    with np.errstate(over='ignore',under='ignore',invalid='ignore',divide='ignore'):
        factor = float(factors[quantity]())
    if not np.isfinite(factor) or factor <= 0:
        raise ValueError('Native unit conversion overflowed or underflowed: '+quantity)
    return factor


def dataset_to_cgs_factor(unit_attributes, scale_factor, hubble_param):
    """Evaluate the saved to_cgs * a^a_scaling * h^h_scaling convention."""
    if unit_attributes is None:
        return None
    native_unit_factor('catalogue_velocity_peculiar_kms',scale_factor=scale_factor,hubble_param=hubble_param)
    try:
        values=[unit_attributes[name] for name in ['to_cgs','a_scaling','h_scaling']]
    except (KeyError,TypeError) as exc:
        raise ValueError('Dataset unit metadata requires to_cgs, a_scaling and h_scaling') from exc
    if any(isinstance(v,(bool,np.bool_)) or not isinstance(v,(int,float,np.integer,np.floating)) or not np.isfinite(v) for v in values) or values[0]<=0:
        raise ValueError('Dataset conversion and exponents must be finite; to_cgs must be positive')
    with np.errstate(over='ignore',under='ignore',invalid='ignore'):
        factor=float(values[0]*np.float64(scale_factor)**values[1]*np.float64(hubble_param)**values[2])
    if not np.isfinite(factor) or factor<=0:
        raise ValueError('Dataset unit conversion overflowed or underflowed')
    return factor


def coordinate_contract(*, position_units, velocity_convention, origin, periodic_wrapping, observer_frame=None):
    """Explicit coordinate descriptors travel with each dataset, including offsets."""
    positions = {'proper kpc':('proper',False),'comoving kpc':('comoving',False),
                 'comoving kpc/h':('comoving',True),'comoving Mpc/h':('comoving',True)}
    if position_units not in positions or not all(isinstance(v,str) and v.strip() for v in (velocity_convention,origin,periodic_wrapping)):
        raise ValueError('Declare supported position units and nonempty coordinate conventions')
    representation, h_scaled = positions[position_units]
    return {'schema':CANONICAL_SCHEMA,'position_units':position_units,'position_representation':representation,
            'position_h_scaled':h_scaled,'origin':origin,'periodic_wrapping':periodic_wrapping,
            'velocity_convention':velocity_convention,'observer_frame':observer_frame}


def epoch_contract(scale_factor, *, requested_scale_factor=None, clock=None):
    """a is physics time; snapshot indices are storage labels only."""
    from core.cosmology import interpolate_proper_time
    native_unit_factor('catalogue_velocity_peculiar_kms',scale_factor=scale_factor,hubble_param=1.)
    a = float(scale_factor)
    redshift=1/a-1
    if not np.isfinite(redshift):raise ValueError('Scale factor has no finite representable redshift')
    if requested_scale_factor is not None:
        native_unit_factor('catalogue_velocity_peculiar_kms',scale_factor=requested_scale_factor,hubble_param=1.)
    requested = float(requested_scale_factor) if requested_scale_factor is not None else None
    scale, times, source = clock if clock is not None else (None,None,{'status':'UNAVAILABLE','reason':'No source clock supplied'})
    age = interpolate_proper_time(scale,times,a) if scale is not None else None
    requested_age = interpolate_proper_time(scale,times,requested) if scale is not None and requested is not None else None
    epoch_source = dict(source)
    if source.get('numerical_error_envelope') is not None:
        envelope = source['numerical_error_envelope']
        # A per-epoch product stores its local bound and a compact source
        # summary; it must not duplicate an entire CLASS grid for every row.
        epoch_source['numerical_error_envelope'] = {key:envelope[key] for key in (
            'schema','status','reference_model','maximum_absolute_bound_gyr',
            'solver_and_background_sampling_error','quadrature')}
    return {'schema':CANONICAL_SCHEMA,'canonical_time':'scale_factor','scale_factor':a,'redshift':redshift,
            'cosmic_time_gyr':age,'requested_scale_factor':requested,'actual_scale_factor':a,
            'requested_cosmic_time_gyr':requested_age,'cosmic_time_source':epoch_source,
            'cosmic_time_numerical_error_gyr':proper_time_error_at_epoch(source,a),
            'snapshot_number_role':'storage label; never used as physical time'}


def proper_time_error_at_epoch(source, scale_factor):
    """Return a declared numerical error envelope, never a cosmology error claim."""
    envelope = source.get('numerical_error_envelope')
    if envelope is None:
        return {'status':'UNBOUNDED','reason':'No expansion-based clock envelope supplied'}
    grid = np.asarray(envelope['scale_factors'],dtype=float)
    bounds = np.asarray(envelope['interval_absolute_bound_gyr'],dtype=float)
    if (grid.ndim != 1 or len(grid) < 2 or not np.isfinite(grid).all()
            or np.any(grid <= 0) or np.any(np.diff(grid) <= 0)
            or bounds.shape != (len(grid)-1,) or not np.isfinite(bounds).all() or np.any(bounds<0)):
        raise ValueError('Malformed clock error envelope')
    if (isinstance(scale_factor, (bool, np.bool_))
            or not isinstance(scale_factor, (int, float, np.integer, np.floating))
            or not np.isfinite(scale_factor) or not grid[0] <= scale_factor <= grid[-1]):
        raise ValueError('Epoch is outside the clock error-envelope domain')
    slot = min(len(bounds)-1,max(0,int(np.searchsorted(grid,scale_factor,side='right')-1)))
    return {'status':'MODEL_CONDITIONAL_BOUND','absolute_bound_gyr':float(bounds[slot]),
            'reference_model':envelope['reference_model'],
            'scope':'Numerical interpolation/integration only; CLASS and cosmological model errors excluded'}


def run_epoch_contract(run_dir, snapshot, scale_factor, *, clock=None):
    from pathlib import Path
    import json
    from core.cosmology import run_proper_time_clock
    if type(snapshot) is not int or snapshot<0:raise ValueError('Exact nonnegative saved snapshot storage label required')
    requested = None
    path = Path(run_dir)/'manifest.json'
    if path.is_file():
        manifest = json.loads(path.read_text())
        redshifts = manifest.get('snapshot_redshifts_requested')
        if redshifts is not None:
            if type(snapshot) is not int or not 0 <= snapshot < len(redshifts):
                raise ValueError('Snapshot storage label is absent from requested output schedule')
            z = redshifts[snapshot]
            if isinstance(z,bool) or not isinstance(z,(float,int)) or not np.isfinite(z) or z<0:
                raise ValueError('Requested redshift must be finite and nonnegative')
            requested = 1/(1+z)
    return {**epoch_contract(scale_factor,requested_scale_factor=requested,
                          clock=run_proper_time_clock(run_dir) if clock is None else clock),
            'snapshot_storage_index':snapshot}


def hdf5_numeric_payload_sha256(dataset, *, memory_bytes=16*1024**2, require_finite=False):
    """Hash numeric payload bytes and their actual dtype/shape with bounded reads."""
    import hashlib,json
    from core.numeric import c_order_slices
    if type(require_finite) is not bool:
        raise ValueError('Finite-value validation requires an explicit boolean')
    if type(memory_bytes) is not int or memory_bytes <= 0:
        raise ValueError('Numeric checksum workspace must be positive integer bytes')
    memory_bytes=min(memory_bytes,getattr(dataset,'memory_bytes',memory_bytes))
    if dataset.dtype.kind not in 'biufc':return None
    checksum=hashlib.sha256()
    checksum.update(json.dumps({'dtype':dataset.dtype.str,'shape':list(dataset.shape)},sort_keys=True).encode())
    for selection in c_order_slices(dataset.shape, dataset.dtype.itemsize, memory_bytes):
        values = np.asarray(dataset[selection], dtype=dataset.dtype)
        if require_finite and values.dtype.kind in 'fc' and not np.isfinite(values).all():
            raise ValueError('Nonfinite native payload: ' + getattr(dataset, 'name', 'numeric array'))
        checksum.update(values.tobytes(order='C'))
    return checksum.hexdigest()


def seal_hdf5_numeric_datasets(group, *, memory_bytes=16*1024**2):
    """Migrate owned output arrays to Fletcher32 chunks, preserving all existing attributes.

    Use only on caller-owned staged products, never on native input snapshots.
    A failed copy leaves the old dataset and temporary bytes for inspection.
    Scalar/empty HDF5 datasets cannot have chunk filters; their payload digest
    is attached by write_dataset_contract instead.
    """
    import h5py,shutil,uuid
    from pathlib import Path
    from core.numeric import c_order_slices
    if type(memory_bytes) is not int or memory_bytes <= 0:
        raise ValueError('HDF5 sealing workspace must be positive integer bytes')
    datasets=[]
    group.visititems(lambda name,item:datasets.append(item.name) if isinstance(item,h5py.Dataset) and
        item.dtype.kind in 'biufc' and item.ndim and item.size and not item.fletcher32 else None)
    for name in datasets:
        original=group.file[name]
        if original.dtype.itemsize > memory_bytes:
            raise MemoryError('HDF5 numeric scalar exceeds sealing workspace')
        if shutil.disk_usage(Path(group.file.filename).parent).free<original.size*original.dtype.itemsize+memory_bytes:
            raise OSError('Insufficient disk margin for checksummed HDF5 dataset migration')
        parent=original.parent;leaf=original.name.rsplit('/',1)[-1]
        temporary='.checksum-copy-'+uuid.uuid4().hex
        copied=parent.create_dataset(temporary,shape=original.shape,dtype=original.dtype,chunks=True,fletcher32=True)
        for selection in c_order_slices(original.shape, original.dtype.itemsize, memory_bytes):
            copied[selection] = original[selection]
        for key,value in original.attrs.items():copied.attrs[key]=value
        if 'chunk_integrity' in copied.attrs:
            copied.attrs['chunk_integrity'] = 'HDF5 Fletcher32'
        group.file.flush()
        del parent[leaf]
        parent.move(temporary,leaf)


def verify_hdf5_dataset_payloads(path, *, require_contract=False, memory_bytes=16*1024**2):
    """Read checksum-filtered numeric chunks and reject changed canonical payloads."""
    import h5py
    checked=[];unsealed=[]
    with h5py.File(path,'r') as source:
        def verify(name,item):
            if not isinstance(item,h5py.Dataset) or item.dtype.kind not in 'biufc':return
            expected=item.attrs.get('payload_sha256')
            if expected is None:
                if require_contract:raise ValueError('Dataset lacks canonical payload digest: '+name)
                unsealed.append(name)
                return
            if hdf5_numeric_payload_sha256(item,memory_bytes=memory_bytes)!=expected:raise ValueError('HDF5 dataset payload checksum mismatch: '+name)
            verify_dataset_semantics(item)
            checked.append(name)
        source.visititems(verify)
    return {'checked':checked,'legacy_unsealed':unsealed,'scope':'numeric payload integrity; no scientific acceptance'}



def numpy_archive_contract(path, specifications, *, epoch=None, coordinates=None, provenance=None,
                           memory_bytes=1024**3):
    """Contract every pickle-free NPZ member using bounded numeric reads."""
    import zipfile
    from pathlib import Path
    from core.foundation import file_sha256 as digest
    from core.numeric import numpy_member_datasets
    path=Path(path)
    if isinstance(memory_bytes,bool) or not isinstance(memory_bytes,int) or memory_bytes<=0:
        raise ValueError('NPZ workspace must be a positive integer byte count')
    before=digest(path);records={}
    with zipfile.ZipFile(path) as zipped:
        if len(zipped.namelist())!=len(set(zipped.namelist())):
            raise ValueError('NPZ duplicate member names are ambiguous')
        if {info.filename for info in zipped.infolist()}!={name+'.npy' for name in specifications}:
            raise ValueError('NPZ contract must explicitly describe every array')
        for name,dataset in numpy_member_datasets(zipped,memory_bytes=memory_bytes):
            spec=dict(specifications[name]);annotations=spec.pop('annotations',{})
            if spec.get('identity_namespace') is not None and dataset.dtype!=np.dtype('uint64'):
                raise ValueError('Canonical NPZ identity must be uint64: '+name)
            spec.setdefault('epoch',epoch);spec.setdefault('coordinates',coordinates);spec.setdefault('provenance',provenance)
            write_dataset_contract(dataset,**spec)
            dataset.attrs['container_integrity']='NPZ ZIP member CRC and archive SHA256'
            if dataset.dtype.kind in 'biufc':dataset.attrs['chunk_integrity']='numeric payload SHA256; ZIP member CRC'
            records[name]={**dataset.attrs,'annotations':annotations}
    if digest(path)!=before:raise ValueError('NPZ archive changed during contract creation')
    return {'schema':CANONICAL_SCHEMA,'container':'pickle-free NPZ','path':path.name,'sha256':before,'datasets':records}


def verify_numpy_archive_contract(path, contract, *, memory_bytes=1024**3):
    """Verify sealed numeric payloads without loading whole archive members."""
    import json,zipfile
    from core.foundation import file_sha256 as digest
    from core.numeric import numpy_member_datasets
    if isinstance(memory_bytes,bool) or not isinstance(memory_bytes,int) or memory_bytes<=0:
        raise ValueError('NPZ workspace must be a positive integer byte count')
    if (not isinstance(contract, dict) or contract.get('schema') != CANONICAL_SCHEMA
            or contract.get('container') != 'pickle-free NPZ'):
        raise ValueError('Unsupported central NPZ container contract')
    if digest(path)!=contract.get('sha256'):
        raise ValueError('NPZ archive contract checksum mismatch')
    with zipfile.ZipFile(path) as zipped:
        if len(zipped.namelist())!=len(set(zipped.namelist())):raise ValueError('NPZ duplicate member names are ambiguous')
        if {info.filename for info in zipped.infolist()}!={name+'.npy' for name in contract['datasets']}:
            raise ValueError('NPZ contract member inventory mismatch')
        for name,dataset in numpy_member_datasets(zipped,memory_bytes=memory_bytes):
            record=contract['datasets'][name]
            if list(dataset.shape)!=json.loads(record['shape_json']) or str(dataset.dtype)!=record['dtype']:
                raise ValueError('NPZ contract shape/dtype mismatch: '+name)
            if record.get('identity_namespace') and dataset.dtype!=np.dtype('uint64'):
                raise ValueError('NPZ contract identity is not uint64: '+name)
            if dataset.dtype.kind in 'biufc' and hdf5_numeric_payload_sha256(dataset)!=record.get('payload_sha256'):
                raise ValueError('NPZ contract payload checksum mismatch: '+name)
    if digest(path)!=contract['sha256']:raise ValueError('NPZ archive changed during verification')
    return contract


def write_dataset_contract(dataset, *, units, coordinates=None, identity_namespace=None, missing_policy='none',
                           quantity=None, epoch=None, provenance=None, ancestry=None, quality='NOT_ACCEPTED',
                           require_finite=False):
    """Attach units and semantic metadata to the actual HDF5 dataset."""
    import json
    if type(require_finite) is not bool:
        raise ValueError('Finite-value validation requires an explicit boolean')
    if not isinstance(units,str) or not units:
        raise ValueError('Every dataset requires explicit units')
    if identity_namespace is not None:
        if not isinstance(identity_namespace,str) or not identity_namespace:
            raise ValueError('Identity namespace must be a nonempty string')
        if dataset.dtype!=np.dtype('uint64'):
            raise ValueError('Canonical identity dataset must use exact uint64 storage')
    dataset.attrs['data_contract_schema'] = CANONICAL_SCHEMA
    dataset.attrs['units'] = units
    dataset.attrs['missing_policy'] = missing_policy
    dataset.attrs['shape_json'] = json.dumps(list(dataset.shape))
    dataset.attrs['dtype'] = str(dataset.dtype)
    checksum=hdf5_numeric_payload_sha256(dataset, require_finite=require_finite)
    if checksum is not None:
        dataset.attrs['payload_sha256']=checksum
        dataset.attrs['payload_checksum_schema']='dtype-shape-c-order-bytes-sha256-v1'
        dataset.attrs['chunk_integrity']='HDF5 Fletcher32' if dataset.fletcher32 else 'scalar/empty payload SHA256' if not dataset.ndim or not dataset.size else 'payload SHA256; chunk filter unavailable'
    dataset.attrs['quantity_definition_status'] = 'PRODUCER_DECLARED' if quantity else 'UNDECLARED'
    if quantity is not None:dataset.attrs['quantity_definition'] = quantity
    dataset.attrs['quality_status'] = quality
    for key,value in [('epoch',epoch),('provenance',provenance),('ancestry',ancestry)]:
        dataset.attrs[key+'_status'] = 'PRODUCER_DECLARED' if value is not None else 'UNDECLARED'
        if value is not None:dataset.attrs[key+'_json'] = json.dumps(value,sort_keys=True,allow_nan=False)
    dataset.attrs['coordinates_status'] = 'PRODUCER_DECLARED' if coordinates is not None else 'UNDECLARED'
    if coordinates is not None:
        dataset.attrs['coordinate_conventions_json'] = json.dumps(coordinates,sort_keys=True,allow_nan=False)
    if identity_namespace is not None:
        dataset.attrs['identity_namespace'] = identity_namespace


def audit_native_snapshot_contract(run_dir, snapshot, *, memory_gib=1., require_chunk_checksums=False):
    """Public integrity audit of full native payloads, not just plausible headers."""
    import h5py
    from pathlib import Path
    from analysis import validate_gadget_snapshot_set
    from halos import _snapshot_files, _unit_conversion
    from core.foundation import product_provenance, file_sha256 as digest
    if type(require_chunk_checksums) is not bool:
        raise ValueError('require_chunk_checksums must be boolean')
    if type(snapshot) is not int or snapshot < 0:
        raise ValueError('snapshot must be a nonnegative storage index')
    if isinstance(memory_gib,bool) or not isinstance(memory_gib,(int,float)) or not np.isfinite(memory_gib) or memory_gib <= 0:
        raise ValueError('memory_gib must be finite and positive')
    root = Path(run_dir).resolve()
    paths = _snapshot_files(root,snapshot)
    publication = verify_native_hdf5_set(paths)
    metadata = validate_gadget_snapshot_set(paths,snapshot)
    _unit_conversion(paths)
    total = sum(metadata['particle_counts_by_type'])
    if total*32 + 64*1024**2 > memory_gib*1024**3:
        raise MemoryError('Full exact-ID integrity audit exceeds declared workspace')
    before = [{'path':str(p.resolve()),'sha256':digest(p)} for p in paths]
    ids, fields, fields_by_file = [], [], {}
    retired_count = 0
    for path in paths:
        with h5py.File(path) as source:
            for species in range(6):
                group = source.get(f'PartType{species}')
                if group is None:
                    continue
                if group['ParticleIDs'].dtype != np.dtype('uint64'):
                    raise ValueError('Native particle IDs must be uint64; narrower/signed IDs cannot prove no truncation')
                live_ids = exact_uint64_ids(group['ParticleIDs'][:], unique=False)
                if species == 5:
                    masses = (group['Masses'][:] if 'Masses' in group else
                              np.full(len(live_ids), source['Header'].attrs['MassTable'][species]))
                    retired = retired_black_hole_carriers(species, live_ids, masses)
                    retired_count += int(np.count_nonzero(retired))
                    if np.any(retired):
                        live_ids = live_ids[~retired]
                    del masses, retired
                ids.append(live_ids)
                del live_ids
                for name,dataset in group.items():
                    if not isinstance(dataset,h5py.Dataset):
                        raise ValueError('Unexpected nested native particle payload')
                    attributes = {k:float(dataset.attrs[k]) for k in ('to_cgs','a_scaling','h_scaling') if k in dataset.attrs}
                    if name not in ('ParticleIDs',) and len(attributes) != 3:
                        raise ValueError('Native field lacks declared physical unit scaling: '+dataset.name)
                    if attributes and (not np.isfinite(list(attributes.values())).all() or attributes['to_cgs'] <= 0):
                        raise ValueError('Invalid native field unit scaling: '+dataset.name)
                    protected = bool(dataset.fletcher32) if dataset.size else None
                    if require_chunk_checksums and dataset.size and not protected:
                        raise ValueError('Native numeric field lacks Fletcher32 chunk protection: '+dataset.name)
                    record = {'file':str(path.resolve()),'field':dataset.name,'shape':list(dataset.shape),
                                   'chunk_integrity':{'algorithm':'Fletcher32' if protected else None,
                                       'status':'PROTECTED' if protected else ('EMPTY' if protected is None else 'LEGACY_UNPROTECTED'),
                                       'chunks':list(dataset.chunks) if dataset.chunks else None},
                                   'dtype':str(dataset.dtype),'physical_unit_scaling':attributes,
                                   'namespace_species':species}
                    fields.append(record)
                    fields_by_file.setdefault(record['file'], []).append(record)
    all_ids = np.concatenate(ids) if ids else np.empty(0,dtype=np.uint64)
    exact_uint64_ids(all_ids)
    a = metadata['scale_factor']
    coordinates = coordinate_contract(position_units='comoving kpc/h',origin='global simulation Cartesian box origin',
        periodic_wrapping='native coordinates wrapped to [0, box)',velocity_convention='snapshot sqrt(a) km/s; peculiar velocity is native v*sqrt(a)')
    provenance = product_provenance(root,algorithm=__file__,selection='Every particle and native dataset in complete saved snapshot',
                                    source_files=before,coordinates=coordinates,snapshot=snapshot)
    epoch=run_epoch_contract(root,snapshot,a)
    native_contracts=[]
    physical_dimensions={'Coordinates':'proper cm','Velocities':'peculiar cm/s',
        'Masses':'g','Density':'g/cm^3','InternalEnergy':'cm^2/s^2',
        'Potential':'cm^2/s^2','Acceleration':'cm/s^2','HydroAcceleration':'cm/s^2','AccelerationGradient':'s^-2',
        'SmoothingLength':'proper cm','SSPInitialMass':'g','SSPInitialLockedMass':'g',
        'SSPPromptReturnedMass':'g','SSPCurrentMass':'g','SSPBirthScaleFactor':'dimensionless scale factor',
        'SSPBirthNativeTime':'dimensionless scale factor or native proper time according to ComovingIntegrationOn',
        'StellarFormationTime':'dimensionless scale factor or native proper time according to ComovingIntegrationOn',
        'SSPBirthMetallicity':'metal mass fraction','Metallicity':'metal mass fraction',
        'SSPParentGasID':'exact particle identity','SSPBirthStateVersion':'schema version',
        'SSPBirthMode':'conversion/spawn enum','StellarRandomCounter':'stochastic opportunity count',
        'GrackleAtomicSpeciesFractions':'dimensionless atomic mass-density fractions; electron lane in hydrogen-mass-equivalent units',
        'GrackleMolecularSpeciesFractions':'dimensionless HM/H2I/H2II/DI/DII/HDI mass-density fractions',
        'GrackleGammaPressure':'dimensionless caloric EOS coefficient',
        'GrackleGammaSound':'dimensionless Grackle sound-speed EOS coefficient',
        'GrackleTemperature':'K','GrackleMeanMolecularWeight':'dimensionless proton-mass units',
        'GrackleInverseCoolingTime':'s^-1; positive net cooling, negative net heating, zero equilibrium',
        'SIDMInteractionCount':'exact uint64 macro scattering event count',
        'SIDMOperatorSubstep':'exact uint64 native operator substep',
        'SIDMExposure':'dimensionless accumulated angular-domain optical depth',
        'SIDMScatteringSupportLength':'proper cm',
        'SIDMKernelCoarseSupportFraction':'dimensionless normalized kernel fraction in coarse material',
        'SIDMOmittedScatteringUpperBound':'dimensionless cumulative omitted angular-domain optical depth bound',
        'SIDMCoarseGuardVersion':'native support guard schema version',
        'PeriodicImageVersion':'native periodic image schema version',
        'PeriodicImageStatus':'native periodic image validity enum',
        'PeriodicImageAnchorNativeTime':'dimensionless scale factor or native proper time according to ComovingIntegrationOn',
        'PeriodicImageOffsets':'exact signed integer whole-box spatial image offsets'}
    species_kinds={0:'gas',1:'dark_matter',4:'star_population',5:'black_hole'}
    for path in paths:
        specifications={}
        with h5py.File(path) as source:
            def describe(name,dataset):
                if not isinstance(dataset,h5py.Dataset):return
                leaf=name.rsplit('/',1)[-1]
                is_particle=name.startswith('PartType')
                record={'units':'UNDECLARED native metadata','quantity':None}
                if is_particle:
                    species=int(name.split('/')[0][8:])
                    record={'units':'native stored field unit','quantity':'native '+leaf}
                    unit_attributes={k:float(dataset.attrs[k]) for k in ('to_cgs','a_scaling','h_scaling') if k in dataset.attrs}
                    if unit_attributes:
                        record['provenance']={**provenance,'field_unit_conversion':{
                            'saved_scaling':unit_attributes,'factor_at_actual_epoch':dataset_to_cgs_factor(unit_attributes,a,metadata['hubble_param']),
                            'target_units':physical_dimensions.get(leaf,'UNDECLARED CGS dimension'),
                            'operation':'metadata only; original native values unchanged'}}
                    if leaf=='ParticleIDs':
                        kind=species_kinds.get(species,'native_particle')
                        record.update(units='identity',identity_namespace=provenance['identity_namespaces'][kind],
                            ancestry={'relation':'native particle identity in declared IC lineage',
                                'species':species,'identity_scope':provenance['identity_scope']})
                    elif leaf=='SSPParentGasID':
                        record.update(units='identity',identity_namespace=provenance['identity_namespaces']['native_particle'],
                            quantity='Exact gas identity at native SSP birth; version zero means unavailable legacy ancestry',
                            ancestry={'relation':'native gas-to-SSP birth parent; stored ID is not a snapshot row',
                                      'identity_scope':provenance['identity_scope']})
                specifications[name]=record
            source.visititems(describe)
        # One bounded payload pass validates finite values and builds the
        # actual external contracts. Reuse its digests in the field summary.
        contract = external_hdf5_contract(
            path, specifications, epoch=epoch, coordinates=coordinates,
            provenance=provenance, require_finite=True,
        )
        native_contracts.append({'file':str(path.resolve()),'contract':contract})
        for field in fields_by_file.get(str(path.resolve()), []):
            field['payload_sha256'] = contract['datasets'][field['field'].lstrip('/')].get('payload_sha256')
    if any(digest(r['path']) != r['sha256'] for r in before):
        raise RuntimeError('Native snapshot changed during payload audit')
    if verify_native_hdf5_set(paths) != publication:
        raise RuntimeError('Native snapshot generation changed during audit')
    return {'schema':CANONICAL_SCHEMA,'status':'TECHNICAL_PASS','science_acceptance':'NOT_ACCEPTED',
            'snapshot':snapshot,'epoch':epoch,'particle_count':total,
            'live_particle_count':len(all_ids),'retired_black_hole_carriers':retired_count,
            'native_set_publication':publication,
            'native_dataset_contracts':native_contracts,
            'chunk_checksum_requirement':require_chunk_checksums,
            'fields':fields,'provenance':provenance,'scope':'Full native payload readability, finite numeric fields, unique live uint64 identities, declared units and checksum-bound metadata; retired species-5/id0/mass0 slots retained; not force/science acceptance'}


def snapshot_total_counts(header):
    """Return six uint64 totals from modern or classic split-word headers.

    Classic GADGET stores N = low + (high << 32). GADGET-4 also supports full
    uint64 totals, omitting high words or writing zero high words. Combining
    an already full count with a nonzero high word is ambiguous, not another
    encoding. Check high-word range before shifting so overflow cannot wrap.
    """
    raw_low = np.asarray(header.get("NumPart_Total", []))
    raw_high = np.asarray(header.get("NumPart_Total_HighWord", np.zeros(6, dtype=np.uint64)))
    if raw_low.shape != (6,) or raw_high.shape != (6,):
        raise ValueError("GADGET Header/NumPart_Total must contain six particle-type counts")
    for name, values in (("NumPart_Total", raw_low), ("NumPart_Total_HighWord", raw_high)):
        if not np.issubdtype(values.dtype, np.integer) or np.any(values < 0):
            raise ValueError("GADGET Header/%s must contain non-negative integer counts" % name)
    low = raw_low.astype(np.uint64, copy=False)
    high = raw_high.astype(np.uint64, copy=False)
    maximum_word = np.iinfo(np.uint32).max
    if np.any(high > maximum_word) or np.any((high != 0) & (low > maximum_word)):
        raise ValueError("GADGET particle totals contain ambiguous full counts plus high words or overflowing high words")
    return low + (high << np.uint64(32))


def validate_native_shard_counts(paths, total_name):
    """Require exact complete native count contracts, including empty shards."""
    import h5py
    from pathlib import Path
    contracts = {'Nsubhalos_Total':('NumFiles','Nsubhalos_ThisFile'),
                 'Ngroups_Total':('NumFiles','Ngroups_ThisFile'),
                 'NumPart_Total':('NumFilesPerSnapshot','NumPart_ThisFile')}
    if total_name not in contracts:
        raise ValueError('Unsupported native series count contract')
    paths = [Path(path).resolve() for path in paths]
    if not paths or len(set(paths))!=len(paths):
        raise ValueError('Native HDF5 series requires distinct nonempty shard paths')
    file_key,count_key = contracts[total_name]
    expected_files,expected_total = None,None
    accumulated = [0]*6 if total_name=='NumPart_Total' else 0
    def scalar(value,name):
        array = np.asarray(value)
        if array.ndim!=0 or array.dtype.kind not in 'ui' or int(array)<0:
            raise ValueError('Native shard '+name+' must be an exact nonnegative scalar integer')
        return int(array)
    for path in paths:
        with h5py.File(path,'r') as source:
            if 'Header' not in source:raise ValueError('Native shard lacks Header: '+str(path))
            header = source['Header'].attrs
            for name in (file_key,total_name,count_key):
                if name not in header:raise ValueError('Native shard lacks required '+name)
            files = scalar(header[file_key],file_key)
            if files<1:raise ValueError('Native shard count must be positive')
            if expected_files is not None and files!=expected_files:
                raise ValueError('file shards disagree about '+file_key)
            expected_files = files
            if total_name=='NumPart_Total':
                total = [int(value) for value in snapshot_total_counts(header)]
                raw = np.asarray(header[count_key])
                if raw.shape!=(6,) or raw.dtype.kind not in 'ui' or np.any(raw<0):
                    raise ValueError('Native shard particle counts require six exact nonnegative integers')
                accumulated = [old+int(value) for old,value in zip(accumulated,raw)]
            else:
                total = scalar(header[total_name],total_name)
                accumulated += scalar(header[count_key],count_key)
            if expected_total is not None and total!=expected_total:
                raise ValueError('file shards disagree about '+total_name)
            expected_total = total
    if expected_files!=len(paths):
        raise ValueError('incomplete HDF5 series: expected %d files, found %d'%(expected_files,len(paths)))
    if accumulated!=expected_total:
        raise ValueError('catalogue shard counts do not sum to '+total_name)


def snapshot_id_shards(files, particle_type=1):
    """Index one particle type in numeric shard order using exact integer IDs.

    Returned [start,stop) offsets refer to the concatenated group-ordered
    particle type, exactly as GADGET's GroupOffsetType/SubhaloOffsetType do.
    Returns (records, total_count). No position-based membership or float
    conversion of 64-bit IDs is used.
    """
    from pathlib import Path
    import h5py
    if isinstance(particle_type, bool) or not isinstance(particle_type, (int, np.integer)) or not 0 <= particle_type <= 5:
        raise ValueError("Particle type must be an integer from 0 through 5")
    files = list(files)
    validate_native_shard_counts(files,'NumPart_Total')
    records, cursor, declared_total = [], 0, None
    group = "PartType%d" % particle_type
    for filename in files:
        path = Path(filename)
        with h5py.File(path, "r") as source:
            header = source["Header"].attrs
            total = int(snapshot_total_counts(header)[particle_type])
            if declared_total is not None and total != declared_total:
                raise ValueError("Snapshot shards disagree about total particle count")
            declared_total = total
            count = int(np.asarray(header["NumPart_ThisFile"])[particle_type])
            if count:
                if group not in source or "ParticleIDs" not in source[group]:
                    raise ValueError("Exact particle tracing needs %s/ParticleIDs" % group)
                ids = source[group]["ParticleIDs"]
                if ids.dtype.kind not in "ui" or ids.shape != (count,):
                    raise ValueError("Particle IDs must be an integer vector matching the header count")
                if source[group]["Coordinates"].shape != (count, 3):
                    raise ValueError("Coordinates must align with the ID vector")
            records.append({"path": path, "start": cursor, "stop": cursor+count, "count": count,
                            "particle_type": particle_type})
            cursor += count
    if declared_total is not None and cursor != declared_total:
        raise ValueError("Particle ID shards do not cover the declared total")
    return records, cursor


def read_selected_group_ids(offsets, lengths, group_indices, records, *, particle_type=1, max_ids=30_000_000):
    """Read exact type-specific memberships from non-overlapping catalogue slices.

    ``offsets`` and ``lengths`` are catalogue arrays (rows,6), ``group_indices``
    selects host/subhalo rows, and records come from snapshot_id_shards.
    The returned vectors are uint64 ParticleIDs and int64 catalogue row numbers.
    The memory budget is checked before reading or allocating any ID payload.
    """
    import h5py
    if isinstance(particle_type, bool) or not isinstance(particle_type, (int, np.integer)) or not 0 <= particle_type <= 5:
        raise ValueError("Particle type must be an integer from 0 through 5")
    if isinstance(max_ids, bool) or not isinstance(max_ids, (int, np.integer)) or max_ids < 0:
        raise ValueError("ID budget must be a nonnegative integer")
    offsets, lengths, rows = map(np.asarray, (offsets, lengths, group_indices))
    if (offsets.dtype.kind not in "ui" or lengths.dtype.kind not in "ui"
            or offsets.ndim != 2 or offsets.shape[1] != 6 or offsets.shape != lengths.shape
            or np.any(offsets < 0) or np.any(lengths < 0)
            or rows.ndim != 1 or rows.dtype.kind not in "ui" or np.any(rows < 0)
            or np.any(rows >= len(offsets)) or len(np.unique(rows)) != len(rows)):
        raise ValueError("Catalogue memberships need aligned six-type integer offsets/counts and unique row indices")
    total_particles = records[-1]["stop"] if records else 0
    spans = []
    for row in rows:
        start, count = int(offsets[row, particle_type]), int(lengths[row, particle_type])
        if count == 0:
            continue
        if start < 0 or start+count > total_particles:
            raise ValueError("Catalogue particle membership extends outside the group-ordered snapshot")
        spans.append((start, start+count, int(row)))
    spans.sort()
    if any(current[0] < previous[1] for previous, current in zip(spans, spans[1:])):
        raise ValueError("Selected catalogue memberships overlap")
    requested = sum(stop-start for start, stop, _ in spans)
    if requested > max_ids:
        raise OverflowError("Selected particle membership exceeds the explicit ID memory budget")
    ids, labels = np.empty(requested, dtype=np.uint64), np.empty(requested, dtype=np.int64)
    cursor = 0
    for record in records:
        if not record["count"]:
            continue
        with h5py.File(record["path"], "r") as source:
            dataset = source["PartType%d/ParticleIDs" % particle_type]
            for start, stop, row in spans:
                left, right = max(start, record["start"]), min(stop, record["stop"])
                if left >= right:
                    continue
                values = np.asarray(dataset[left-record["start"]:right-record["start"]])
                if values.dtype.kind == "i" and np.any(values < 0):
                    raise ValueError("Negative particle IDs are invalid")
                n = right-left
                ids[cursor:cursor+n], labels[cursor:cursor+n] = values, row
                cursor += n
    if cursor != requested or len(np.unique(ids)) != len(ids):
        raise ValueError("Catalogue particle IDs are incomplete or duplicated")
    return ids, labels


def hdf5_scalar_attr(value, name):
    """Read a scalar GADGET header field from either scalar or length-one form."""
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError("GADGET Header/%s must contain exactly one value" % name)
    try:
        return float(array.reshape(-1)[0])
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("GADGET Header/%s must be numeric" % name) from exc


def particle_id_hash64(ids):
    """Return a stable, well-mixed 64-bit hash for deterministic ID sampling."""
    values = np.asarray(ids)
    if values.ndim != 1 or values.dtype.kind not in "iu":
        raise ValueError("ID sampling requires a one-dimensional integer ParticleIDs array")
    if values.dtype.kind == "i" and np.any(values < 0):
        raise ValueError("ParticleIDs must be non-negative")
    with np.errstate(over="ignore"):
        hashed = values.astype(np.uint64, copy=False) + np.uint64(0x9E3779B97F4A7C15)
        hashed = (hashed ^ (hashed >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        hashed = (hashed ^ (hashed >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        hashed ^= hashed >> np.uint64(31)
    return hashed


# IC transformations use the same durable publisher as scientific products.
# A shard set cannot be renamed atomically as individual files: publication
# therefore uses a replayable journal, and every public discovery rejects an
# unfinished journal. Original/staged bytes remain in the immutable archive.
def assert_ic_publication_complete(paths):
    import json
    from pathlib import Path
    for parent in {Path(p).resolve().parent for p in paths}:
        for journal in parent.glob('.ic-transform-*.json'):
            record = json.loads(journal.read_text())
            if record.get('schema') != 'ic-transform-v1' or record.get('status') != 'COMPLETE':
                raise ValueError('Incomplete IC transformation; replay the recorded transaction: '+str(journal))


def replay_ic_transform(journal):
    """Finish a reviewed staged commit without repeating the physical transform."""
    import json
    from pathlib import Path
    from core.foundation import atomic_copy, atomic_json, file_sha256 as digest
    journal = Path(journal).resolve()
    record = json.loads(journal.read_text())
    if record.get('schema') != 'ic-transform-v1' or record.get('status') not in ('PENDING', 'COMPLETE'):
        raise ValueError('Invalid IC publication transaction')
    archive = (journal.parent / record['archive']).resolve()
    if archive.parent != journal.parent or not archive.name.startswith('.ic-transform-'):
        raise ValueError('IC transaction archive escapes its source directory')
    plan_path = archive/'plan.json'
    if digest(plan_path) != record.get('plan_sha256'):
        raise ValueError('IC transaction plan changed')
    plan = json.loads(plan_path.read_text())
    if any(record.get(k) != v for k,v in plan.items() if k != 'status'):
        raise ValueError('IC transaction differs from its immutable plan')
    # Verify the entire archive and every destination before replacing any shard.
    for row in record['files']:
        name = row['name']
        if Path(name).name != name or name in ('', '.', '..'):
            raise ValueError('Invalid IC shard name in transaction')
        for branch,key in [('original','before_sha256'),('transformed','after_sha256')]:
            if digest(archive/branch/name) != row[key]:
                raise ValueError('IC archive is corrupt: '+name)
        current = digest(journal.parent/name)
        if current not in (row['before_sha256'], row['after_sha256']):
            raise ValueError('IC destination changed outside transaction: '+name)
        if record['status'] == 'COMPLETE' and current != row['after_sha256']:
            raise ValueError('Completed IC transaction no longer matches its published bytes')
    if record['status'] == 'COMPLETE':
        return record
    for row in record['files']:
        if digest(journal.parent/row['name']) != row['after_sha256']:
            atomic_copy(archive/'transformed'/row['name'], journal.parent/row['name'])
    record['status'] = 'COMPLETE'
    atomic_json(journal, record)
    return record


def transactional_ic_transform(paths, transform, *, operation, configuration):
    """Run a real chunked transform on copies; failed validation never edits ICs.

    The callback receives same-name staged shards and returns finite JSON model
    provenance. Disk demand is two retained copies plus one active copy. Kernel
    ownership excludes overlapping single/multiple-shard writers in a directory.
    Publication replay is explicit; a PENDING journal closes public discovery.
    """
    import json
    import shutil
    import inspect
    import tempfile
    import uuid
    from pathlib import Path
    import h5py
    from core.numeric import c_order_slices
    from core.native import native_run_lock
    from core.foundation import atomic_copy, atomic_json, canonical_digest, file_sha256 as digest, publish_directory
    paths = [Path(p).expanduser().resolve() for p in paths]
    if not paths or len(set(paths)) != len(paths) or len({p.parent for p in paths}) != 1:
        raise ValueError('IC transform requires a unique nonempty shard set in one directory')
    parent = paths[0].parent
    config_hash = canonical_digest(configuration)
    with native_run_lock(parent, operation='IC transform '+operation):
        assert_ic_publication_complete(paths)
        required = 3*sum(p.stat().st_size for p in paths)
        if shutil.disk_usage(parent).free < required:
            raise OSError('IC transformation needs at least '+str(required)+' free staging bytes')
        name = '.ic-transform-'+uuid.uuid4().hex
        stage = Path(tempfile.mkdtemp(prefix=name+'-stage-', dir=parent))
        for branch in ('original','transformed'):
            (stage/branch).mkdir()
        rows = []
        try:
            for path in paths:
                before = atomic_copy(path, stage/'original'/path.name)
                atomic_copy(stage/'original'/path.name, stage/'transformed'/path.name)
                rows.append({'name':path.name,'before_sha256':before})
            result = transform([stage/'transformed'/p.name for p in paths])
            # Read all staged payloads in bounded slices. HDF5 verifies each
            # Fletcher32 chunk touched by a read, including partial chunk reads.
            for row,path in zip(rows, paths):
                staged = stage/'transformed'/path.name
                with h5py.File(staged, 'r+') as saved:
                    saved.attrs['ic_transform_schema'] = 'ic-transform-v1'
                    saved.attrs['ic_transform_operation'] = operation
                    saved.attrs['ic_transform_configuration_sha256'] = config_hash
                    saved.attrs['ic_transform_source_sha256'] = row['before_sha256']
                    def read_dataset(_name, ds):
                        if not isinstance(ds,h5py.Dataset):
                            return
                        for selection in c_order_slices(ds.shape, ds.dtype.itemsize, 8*1024**2):
                            ds[selection]
                    saved.visititems(read_dataset)
                    saved.flush()
                row['after_sha256'] = digest(staged)
                if digest(path) != row['before_sha256']:
                    raise ValueError('Original IC changed during transformation: '+str(path))
            plan = {'schema':'ic-transform-v1','status':'PENDING','archive':name,
                    'operation':operation,'configuration_sha256':config_hash,
                    'algorithm_sha256':digest(__file__),'transform_source_sha256':digest(inspect.getsourcefile(transform)),
                    'files':rows,'model_report':result,
                    'science_status':'NOT_ACCEPTED'}
            atomic_json(stage/'plan.json',plan)
            plan_hash = digest(stage/'plan.json')
            publish_directory(stage,parent/name)
            journal = parent/(name+'.json')
            atomic_json(journal,dict(plan,plan_sha256=plan_hash))
            replay_ic_transform(journal)
            return result
        except BaseException as exc:
            if stage.exists():
                atomic_json(stage/'failure.json',{'operation':operation,'status':'FAILED',
                            'type':type(exc).__name__,'message':str(exc)})
            raise


class _ReadOnlyDatasetContract:
    """Reuse the central writer for external native bytes without modifying them."""
    def __init__(self, dataset):
        self.dataset=dataset;self.attrs={}
    def __getattr__(self, key):return getattr(self.dataset,key)
    def __len__(self):return len(self.dataset)
    def __getitem__(self, key):return self.dataset[key]


def external_hdf5_contract(path, specifications, *, epoch=None, coordinates=None, provenance=None,
                           require_finite=False):
    """Contract native HDF5 fields with bounded reads and whole-file identity."""
    import h5py
    from core.foundation import file_sha256 as digest
    if type(require_finite) is not bool:
        raise ValueError('Finite-value validation requires an explicit boolean')
    before=digest(path);records={}
    with h5py.File(path) as source:
        datasets={}
        def collect(name,item):
            if isinstance(item,h5py.Dataset):datasets[name]=item
        source.visititems(collect)
        if set(datasets)!=set(specifications):raise ValueError('Declare every external HDF5 dataset')
        for name,dataset in datasets.items():
            view=_ReadOnlyDatasetContract(dataset)
            spec=dict(specifications[name]);spec.setdefault('epoch',epoch)
            spec.setdefault('coordinates',coordinates);spec.setdefault('provenance',provenance)
            if require_finite:
                spec['require_finite'] = True
            write_dataset_contract(view,**spec)
            records[name]=view.attrs
    if digest(path)!=before:raise RuntimeError('External HDF5 source changed during contract creation')
    return {'schema':CANONICAL_SCHEMA,'container':'external native HDF5','sha256':before,'datasets':records}


def verify_external_hdf5_contract(path, contract):
    """Reject native byte changes and unreadable/changed contracted datasets."""
    import h5py,json
    from core.foundation import file_sha256 as digest
    before=digest(path)
    if contract.get('schema')!=CANONICAL_SCHEMA or contract.get('container')!='external native HDF5':
        raise ValueError('Unsupported external HDF5 contract')
    if before!=contract['sha256']:raise ValueError('External HDF5 file checksum mismatch')
    with h5py.File(path) as source:
        datasets={}
        def collect(name,item):
            if isinstance(item,h5py.Dataset):datasets[name]=item
        source.visititems(collect)
        if set(datasets)!=set(contract['datasets']):raise ValueError('External HDF5 dataset inventory mismatch')
        for name,dataset in datasets.items():
            record=contract['datasets'][name]
            if str(dataset.dtype)!=record['dtype'] or list(dataset.shape)!=json.loads(record['shape_json']):
                raise ValueError('External HDF5 dtype/shape mismatch: '+name)
            if record.get('identity_namespace') and dataset.dtype!=np.dtype('uint64'):
                raise ValueError('External HDF5 identity must be uint64: '+name)
            if hdf5_numeric_payload_sha256(dataset)!=record.get('payload_sha256'):
                raise ValueError('External HDF5 numeric payload mismatch: '+name)
    if digest(path)!=before:raise RuntimeError('External HDF5 changed while verifying')
    return contract


def isolated_epoch_contract(time_code, *, time_unit_gyr=None, snapshot=None):
    """Declare isolated noncomoving time without fabricating a cosmological a."""
    if isinstance(time_code, (bool, np.bool_)) or not np.isfinite(time_code):
        raise ValueError('Isolated native time must be finite')
    if time_unit_gyr is not None:
        if isinstance(time_unit_gyr, (bool, np.bool_)) or not np.isfinite(time_unit_gyr) or time_unit_gyr <= 0:
            raise ValueError('Isolated time conversion must be finite and positive')
    with np.errstate(over='ignore',invalid='ignore'):
        elapsed=None if time_unit_gyr is None else float(np.float64(time_code)*time_unit_gyr)
    if elapsed is not None and not np.isfinite(elapsed):raise ValueError('Isolated elapsed-time conversion overflows')
    return {'schema': CANONICAL_SCHEMA, 'canonical_time': 'isolated_native_elapsed_time',
            'native_time_code': float(time_code), 'time_unit_gyr': time_unit_gyr,
            'elapsed_time_gyr': elapsed,
            'actual_scale_factor': None, 'requested_scale_factor': None, 'redshift': None,
            'cosmic_time_gyr': None, 'cosmic_time_source': {'status': 'NOT_APPLICABLE',
                'reason': 'Noncomoving isolated run; Header/Time is not a scale factor'},
            'snapshot_storage_index': snapshot, 'requested_native_time_code': None}


# The semantic seal covers contract attributes only. Container publication hashes
# additionally bind all producer attributes, group metadata and the shard inventory.
_SEMANTIC_ATTRIBUTES = (
    'data_contract_schema', 'units', 'missing_policy', 'shape_json', 'dtype',
    'quantity_definition_status', 'quantity_definition', 'quality_status',
    'epoch_status', 'epoch_json', 'provenance_status', 'provenance_json',
    'ancestry_status', 'ancestry_json', 'coordinates_status',
    'coordinate_conventions_json', 'identity_namespace', 'strict_semantics',
)


def _semantic_digest(dataset):
    import hashlib, json
    values = {key: dataset.attrs[key] for key in _SEMANTIC_ATTRIBUTES if key in dataset.attrs}
    return hashlib.sha256(json.dumps(values, sort_keys=True, allow_nan=False,
                                    separators=(',', ':')).encode()).hexdigest()


def seal_dataset_semantics(dataset):
    dataset.attrs['semantics_sha256'] = _semantic_digest(dataset)


def _validate_scientific_spec(spec):
    """Require a declared frame even for scalars (explicitly 'not applicable')."""
    import json
    if not isinstance(spec, dict):
        raise ValueError('Scientific semantic specification must be an object')
    for key in ('units', 'quantity', 'missing_policy'):
        if not isinstance(spec.get(key), str) or not spec[key].strip() or spec[key] == 'UNDECLARED':
            raise ValueError('Scientific dataset requires explicit ' + key)
    if not isinstance(spec.get('provenance'), dict) or not spec['provenance']:
        raise ValueError('Scientific dataset requires explicit provenance')
    frame = spec.get('coordinates')
    if not isinstance(frame, dict) or not frame:
        raise ValueError('Scientific dataset requires explicit coordinate frame')
    if frame.get('position_representation') != 'not applicable':
        rebuilt = coordinate_contract(position_units=frame.get('position_units'),
            origin=frame.get('origin'), periodic_wrapping=frame.get('periodic_wrapping'),
            velocity_convention=frame.get('velocity_convention'), observer_frame=frame.get('observer_frame'))
        if any(frame.get(k) != rebuilt[k] for k in ('position_representation', 'position_h_scaled')):
            raise ValueError('Coordinate frame disagrees with declared units')
    elif not frame.get('reason'):
        raise ValueError('Not-applicable coordinate frame requires a reason')
    epoch = spec.get('epoch')
    if not isinstance(epoch, dict):
        raise ValueError('Scientific dataset requires explicit epoch')
    if epoch.get('canonical_time') == 'scale_factor':
        a = epoch.get('scale_factor')
        if isinstance(a, bool) or not isinstance(a, (int, float)) or not np.isfinite(a) or a <= 0:
            raise ValueError('Invalid scientific scale factor')
        if epoch.get('actual_scale_factor') != a or epoch.get('redshift') != 1 / a - 1:
            raise ValueError('Scientific epoch fields disagree')
    elif epoch.get('canonical_time') == 'isolated_native_elapsed_time':
        t = epoch.get('native_time_code')
        if isinstance(t, bool) or not isinstance(t, (int, float)) or not np.isfinite(t):
            raise ValueError('Invalid isolated native epoch')
        if epoch.get('scale_factor') is not None:
            raise ValueError('Isolated epoch cannot declare cosmological scale factor')
    else:
        raise ValueError('Scientific epoch requires a canonical time definition')
    json.dumps(spec, allow_nan=False)


def publish_scientific_dataset(dataset, semantic_spec):
    """Contract and seal an already-created, caller-owned staged numeric dataset.

    Publication of its containing file uses publish_hdf5_product: this function
    never changes public aliases or rewrites immutable native inputs.
    """
    import h5py
    spec = dict(semantic_spec)
    _validate_scientific_spec(spec)
    if dataset.dtype.kind not in 'biufc':
        raise ValueError('Scientific dataset requires numeric storage')
    declared_dtype = spec.pop('dtype', None)
    if declared_dtype is not None and np.dtype(declared_dtype) != dataset.dtype:
        raise ValueError('Scientific dataset dtype disagrees with specification')
    if spec.get('identity_namespace') is not None:
        if dataset.dtype != np.dtype('uint64'):
            raise ValueError('Scientific identities require exact uint64 storage; never cast IDs')
    write_dataset_contract(dataset, **spec)
    dataset.attrs['strict_semantics'] = 'scientific-dataset-v1'
    seal_dataset_semantics(dataset)
    if isinstance(dataset, h5py.Dataset):
        name = dataset.name
        seal_hdf5_numeric_datasets(dataset.parent)
        dataset = dataset.file[name]
        # Chunk-filter annotation reflects the final on-disk dataset.
        dataset.attrs['chunk_integrity'] = ('HDF5 Fletcher32' if dataset.fletcher32
                                           else 'scalar/empty payload SHA256')
        dataset.file.flush()
    return dataset


def verify_dataset_semantics(dataset, *, require_explicit=False):
    """Check actual dtype/shape/identity and the sealed semantic declaration."""
    import json
    attrs = dataset.attrs
    if attrs.get('data_contract_schema') != CANONICAL_SCHEMA:
        raise ValueError('Dataset lacks canonical semantic schema')
    if json.loads(attrs.get('shape_json', 'null')) != list(dataset.shape) or attrs.get('dtype') != str(dataset.dtype):
        raise ValueError('Dataset semantic dtype/shape mismatch')
    if attrs.get('identity_namespace') is not None and dataset.dtype != np.dtype('uint64'):
        raise ValueError('Dataset identity storage is not exact uint64')
    if attrs.get('semantics_sha256') is not None and attrs['semantics_sha256'] != _semantic_digest(dataset):
        raise ValueError('Dataset semantics checksum mismatch')
    if require_explicit or attrs.get('strict_semantics') == 'scientific-dataset-v1':
        if attrs.get('semantics_sha256') is None:
            raise ValueError('Scientific dataset lacks semantic seal')
        _validate_scientific_spec({'units': attrs.get('units'), 'quantity': attrs.get('quantity_definition'),
            'missing_policy': attrs.get('missing_policy'),
            'coordinates': json.loads(attrs.get('coordinate_conventions_json', 'null')),
            'epoch': json.loads(attrs.get('epoch_json', 'null')),
            'provenance': json.loads(attrs.get('provenance_json', 'null'))})


class ScientificProductReader:
    """Closed-by-default, bounded reads from a verified immutable generation.

    Each iterator verifies before and after use so a changed alias cannot silently
    finish consumption. Callers must not publish derived results until iteration
    finishes successfully. No HDF5 handle survives an iterator's lifetime.
    """
    def __init__(self, paths, records, memory_bytes, verify):
        self.paths = tuple(paths)
        self.datasets = records
        self.memory_bytes = memory_bytes
        self._verify = verify

    def iter_dataset(self, name):
        import h5py
        self._verify()
        if name not in self.datasets:
            raise KeyError(name)
        for path in self.paths:
            with h5py.File(path, 'r') as source:
                data = source[name]
                row_bytes = data.dtype.itemsize * int(np.prod(data.shape[1:], dtype=object))
                if row_bytes > self.memory_bytes:
                    raise MemoryError('Scientific dataset row exceeds reader workspace')
                if data.ndim == 0:
                    yield data[()]
                else:
                    step = max(1, self.memory_bytes // max(1, row_bytes))
                    for start in range(0, len(data), step):
                        yield data[start:start + step]
        self._verify()


def read_scientific_product(product, *, memory_bytes=16*1024**2):
    """Verify existing HDF5 generations and return a bounded numeric reader.

    A path denotes one durable published HDF5 product. A descriptor containing
    'shards' denotes an explicitly complete native set with its native COMPLETE
    marker; every numeric dataset must also carry strict scientific semantics.
    """
    import h5py, json
    from pathlib import Path
    from core.foundation import file_sha256 as digest
    if type(memory_bytes) is not int or memory_bytes <= 0:
        raise ValueError('Scientific reader workspace must be positive integer bytes')
    native = isinstance(product, dict)
    if native:
        if set(product) != {'shards'} or not isinstance(product['shards'], (list, tuple)):
            raise ValueError('Scientific shard descriptor requires an explicit shard inventory')
        paths = [Path(p).resolve() for p in product['shards']]
        native_set = verify_native_hdf5_set(paths, require_publication=True)
        if native_set['native_time'] is None:
            raise ValueError('Native scientific shard time is missing')
        paths.sort(key=lambda p: int(p.stem.rsplit('.', 1)[-1]) if p.stem.rsplit('.', 1)[-1].isdigit() else -1)
    else:
        paths = [Path(product).resolve()]
        if verify_hdf5_product(paths[0], memory_bytes=memory_bytes).get('status') == 'LEGACY_UNSEALED':
            raise ValueError('Scientific product requires a complete generation manifest')
    hashes = {p: digest(p) for p in paths}
    records = {}; epochs = set()
    for path in paths:
        current = {}
        with h5py.File(path, 'r') as source:
            def inspect(name, data):
                if not isinstance(data, h5py.Dataset) or data.dtype.kind not in 'biufc':
                    return
                verify_dataset_semantics(data, require_explicit=True)
                if hdf5_numeric_payload_sha256(data, memory_bytes=memory_bytes) != data.attrs.get('payload_sha256'):
                    raise ValueError('Scientific dataset payload checksum mismatch: ' + name)
                current[name] = {'dtype': data.dtype.str, 'shape': list(data.shape),
                    'units': data.attrs['units'], 'epoch': json.loads(data.attrs['epoch_json']),
                    'frame': json.loads(data.attrs['coordinate_conventions_json']),
                    'identity_namespace': data.attrs.get('identity_namespace')}
                epochs.add(data.attrs['epoch_json'])
                if native and native_set['native_time'] is not None:
                    epoch = current[name]['epoch']
                    actual = epoch.get('actual_scale_factor') if epoch.get('canonical_time') == 'scale_factor' else epoch.get('native_time_code')
                    if actual != native_set['native_time']:
                        raise ValueError('Scientific dataset epoch disagrees with native shard header')
            source.visititems(inspect)
        if not current:
            raise ValueError('Scientific product has no numeric datasets')
        if records and set(current) != set(records):
            raise ValueError('Scientific shard dataset inventory mismatch')
        for name, record in current.items():
            if name in records:
                old = records[name]
                if any(record[k] != old[k] for k in ('dtype', 'units', 'epoch', 'frame', 'identity_namespace')) or record['shape'][1:] != old['shape'][1:]:
                    raise ValueError('Scientific shard semantics mismatch: ' + name)
            else:
                records[name] = {**record, 'shard_shapes': [], 'total_rows': 0 if record['shape'] else None}
            records[name]['shard_shapes'].append(record['shape'])
            if record['shape']:
                records[name]['total_rows'] += record['shape'][0]
    if len(epochs) != 1:
        raise ValueError('Scientific product contains mismatched epochs')
    def verify():
        if native:
            verify_native_hdf5_set(paths, require_publication=True)
        else:
            verify_hdf5_product(paths[0], memory_bytes=memory_bytes)
        if any(digest(path) != sha for path, sha in hashes.items()):
            raise ValueError('Scientific shard changed during consumption')
    verify()
    return ScientificProductReader(paths, records, memory_bytes, verify)


def validate_gadget_native_particle_ids(paths, *, chunk_rows=250000):
    """IDS_64BIT stores 63 identity bits plus the native most-bound marker.

    Canonical scientific storage retains the full uint64 domain. The pinned
    solver must reject unsupported input before any native ID masking occurs.
    """
    import h5py
    if type(chunk_rows) is not int or chunk_rows <= 0:
        raise ValueError('Native ID workspace requires positive integer rows')
    checked = 0
    for path in paths:
        with h5py.File(path, 'r') as source:
            for species in range(6):
                group = source.get('PartType' + str(species))
                if group is None:
                    continue
                data = group['ParticleIDs']
                if data.ndim != 1 or data.dtype.kind not in 'iu':
                    raise ValueError('GADGET particle identity must be an exact integer vector')
                for first in range(0, len(data), chunk_rows):
                    values = data[first:first + chunk_rows]
                    if np.any(values < 0) or np.any(values > np.iinfo(np.int64).max):
                        raise ValueError('GADGET identity exceeds 63-bit native domain; most-significant bit is reserved')
                    checked += len(values)
    return {'checked_particle_ids': checked, 'native_domain': '0..2^63-1',
            'canonical_storage': 'uint64; original input bytes are unchanged',
            'science_acceptance': 'NOT_ACCEPTED'}

# ========================================================================
# native_hdf5_sets: Native multi-shard publication guards; legacy copies remain explicitly unsealed.
from pathlib import Path
import json
import re
from numbers import Integral
import h5py


def verify_native_hdf5_set(files, *, require_publication=False):
    paths = [Path(p).resolve() for p in files]
    if not paths or len(set(paths)) != len(paths):
        raise ValueError('Native shard set is empty or contains duplicates')
    bases = set()
    generations = []
    counts = []
    shard_indices = []
    epochs = []
    identities = []
    for path in paths:
        match = re.fullmatch(r'(.+?)(?:\.(\d+))?\.hdf5', path.name)
        if not match:
            raise ValueError('Invalid native shard filename')
        bases.add(path.parent / match.group(1))
        shard_indices.append(None if match.group(2) is None else int(match.group(2)))
        with h5py.File(path, 'r') as source:
            header = source['Header'].attrs
            epoch = header.get('Time')
            if epoch is not None:
                import math
                from numbers import Real
                if isinstance(epoch, bool) or not isinstance(epoch, Real) or not math.isfinite(epoch):
                    raise ValueError('Native shard epoch must be finite numeric time')
            epochs.append(None if epoch is None else float(epoch))
            identities.append((path, path.stat().st_ino, path.stat().st_size, path.stat().st_mtime_ns))
            generation = header.get('Research2Generation')
            if isinstance(generation, bytes):
                generation = generation.decode('ascii')
            if generation is not None:
                if not isinstance(generation, str) or not generation:
                    raise ValueError('Invalid native generation attribute')
                count = header.get('Research2ShardCount')
                if isinstance(count, bool) or not isinstance(count, Integral) or count <= 0:
                    raise ValueError('Native generation requires an exact positive shard count')
            else:
                count = len(paths)
            generations.append(generation)
            counts.append(int(count))
    if len(set(epochs)) != 1:
        raise ValueError('Mixed native shard epochs')
    if len(bases) != 1:
        raise ValueError('Native shards have distinct publication bases')
    marker = Path(str(next(iter(bases))) + '.publication.json')
    marker_bytes = marker.read_bytes() if marker.exists() else None
    if len(set(generations)) != 1 or any(c != len(paths) for c in counts):
        raise ValueError('Mixed or incomplete native shard generations')
    generation = generations[0]
    if generation is not None:
        expected_indices = [None] if len(paths) == 1 else list(range(len(paths)))
        if set(shard_indices) != set(expected_indices):
            raise ValueError("Native shard indices do not match the declared layout")
    if marker_bytes is not None:
        record = json.loads(marker_bytes)
        if (record.get('schema') != 'native-hdf5-set-publication-v1' or
                record.get('state') != 'COMPLETE' or
                not isinstance(record.get('generation'), str) or not record['generation'] or
                record.get('generation') != generation or
                type(record.get('shard_count')) is not int or
                record['shard_count'] != len(paths)):
            raise ValueError('Native shard publication is incomplete or mismatched')
        if marker.read_bytes() != marker_bytes:
            raise ValueError('Native publication changed during read')
        status = 'COMPLETE'
    else:
        if require_publication:
            raise ValueError('Native shard publication marker is missing')
        status = 'COPIED_GENERATION_UNSEALED' if generation else 'LEGACY_UNSEALED'
    if any((path.stat().st_ino, path.stat().st_size, path.stat().st_mtime_ns) != (inode, size, modified)
           for path, inode, size, modified in identities):
        raise ValueError('Native shards changed during verification')
    return {'schema': 'native-hdf5-set-verification-v1', 'status': status,
            'generation': generation, 'shard_count': len(paths), 'native_time': epochs[0],
            'science_acceptance': 'NOT_ACCEPTED'}


def recover_native_hdf5_set(base, generation):
    """Restore an explicitly chosen complete retained generation, never infer latest."""
    from core.native import native_run_lock
    from core.foundation import atomic_copy, atomic_json, file_sha256 as digest, publish_directory
    from core.numeric import c_order_slices
    import tempfile
    import uuid
    base = Path(base).resolve()
    if not isinstance(generation, str) or not generation:
        raise ValueError('Recovery requires an explicit native generation')
    with native_run_lock(base.parent, operation='native shard generation recovery'):
        marker = Path(str(base) + '.publication.json')
        retained = [marker, *base.parent.glob(marker.name + '.previous.*')]
        committed = []
        for path in retained:
            try:
                record = json.loads(path.read_bytes())
            except (ValueError, OSError):
                continue
            if (record.get('schema') == 'native-hdf5-set-publication-v1' and
                    record.get('state') == 'COMPLETE' and record.get('generation') == generation):
                committed.append(record)
        if not committed:
            raise ValueError('No retained COMPLETE marker proves the requested generation')
        count = committed[0]['shard_count']
        if type(count) is not int or count <= 0 or any(r['shard_count'] != count for r in committed):
            raise ValueError('Invalid or conflicting retained native shard counts')
        selected = []
        for index in range(count):
            target = base.parent / (base.name + ('.%d' % index if count > 1 else '') + '.hdf5')
            candidates = [target, *base.parent.glob(target.name + '.previous.*')]
            matches = []
            for path in candidates:
                if not path.is_file(): continue
                try:
                    with h5py.File(path, 'r') as source:
                        actual = source['Header'].attrs.get('Research2Generation')
                        if isinstance(actual, bytes): actual = actual.decode('ascii')
                        if actual != generation or int(source['Header'].attrs.get('Research2ShardCount', 0)) != count:
                            continue
                        def read_payload(name, dataset):
                            if not isinstance(dataset, h5py.Dataset): return
                            for selection in c_order_slices(dataset.shape, dataset.dtype.itemsize, 16*1024**2):
                                dataset[selection]
                        source.visititems(read_payload)
                    matches.append((path, digest(path)))
                except (OSError, ValueError, KeyError):
                    continue
            if not matches or len({sha for _, sha in matches}) != 1:
                raise ValueError('Missing, corrupt or conflicting retained shard: ' + target.name)
            selected.append((target, *matches[0]))
        stage = Path(tempfile.mkdtemp(prefix='.native-set-recovery-', dir=base.parent))
        inventory = []
        for target, source, sha in selected:
            atomic_copy(source, stage / target.name)
            if digest(source) != sha or digest(stage / target.name) != sha:
                raise ValueError('Native recovery source changed during copy')
            inventory.append({'name': target.name, 'sha256': sha})
        atomic_json(stage / marker.name, committed[0])
        atomic_json(stage / 'recovery_manifest.json', {'schema': 'native-set-recovery-v1',
                    'generation': generation, 'artifacts': inventory, 'science_acceptance': 'NOT_ACCEPTED'})
        verify_native_hdf5_set([stage / r['name'] for r in inventory], require_publication=True)
        displaced = stage / 'displaced'
        displaced.mkdir()
        if marker.is_file(): atomic_copy(marker, displaced / marker.name)
        for target, _, _ in selected:
            if target.is_file(): atomic_copy(target, displaced / target.name)
        archive = base.parent / '.native_set_history' / uuid.uuid4().hex
        archive.parent.mkdir(exist_ok=True)
        publish_directory(stage, archive)
        atomic_json(marker, {**committed[0], 'state': 'PENDING', 'recovery_archive': str(archive)})
        for target, _, _ in selected: atomic_copy(archive / target.name, target)
        atomic_json(marker, committed[0])
        result = verify_native_hdf5_set([target for target, _, _ in selected], require_publication=True)
        return {**result, 'recovery_archive': str(archive)}


def native_publication_copy_members(outputs):
    """Carry native COMPLETE markers through exact shard copies/base renames.

    Returns additional (source, relative destination) members. It never creates
    a completion claim for legacy files lacking a native marker.
    """
    groups = {}
    for source, destination in outputs:
        source, destination = Path(source), Path(destination)
        source_match = re.fullmatch(r'(.+?)(?:\.(\d+))?\.hdf5', source.name)
        destination_match = re.fullmatch(r'(.+?)(?:\.(\d+))?\.hdf5', destination.name)
        if not source_match or not destination_match: continue
        source_base = source.parent / source_match.group(1)
        marker = Path(str(source_base) + '.publication.json')
        if not marker.is_file(): continue
        if source_match.group(2) != destination_match.group(2):
            raise ValueError('Native copy changes the shard index layout')
        destination_base = destination.parent / destination_match.group(1)
        groups.setdefault(marker, []).append((source, destination_base))
    result = []
    for marker, rows in groups.items():
        verify_native_hdf5_set([source for source, _ in rows], require_publication=True)
        bases = {base for _, base in rows}
        if len(bases) != 1:
            raise ValueError('Native source set is copied into distinct destination bases')
        result.append((marker, Path(str(next(iter(bases))) + '.publication.json')))
    return result


def native_publication_markers(files):
    """Existing native marker paths accompanying an archived shard inventory."""
    markers = set()
    for file in files:
        file = Path(file)
        match = re.fullmatch(r'(.+?)(?:\.(\d+))?\.hdf5', file.name)
        if match:
            marker = file.parent / (match.group(1) + '.publication.json')
            if marker.is_file(): markers.add(marker)
    return sorted(markers)


def verify_discovered_native_sets(files):
    """Guard marked native sets at shared discovery without interpreting legacy data."""
    groups = {}
    for file in files:
        file = Path(file)
        match = re.fullmatch(r'(.+?)(?:\.(\d+))?\.hdf5', file.name)
        if match:
            base = file.parent / match.group(1)
            groups.setdefault(base, []).append(file)
    for base, members in groups.items():
        if Path(str(base) + '.publication.json').exists():
            verify_native_hdf5_set(members, require_publication=True)
    return files

# ========================================================================
# native_checkpoints: Pre-allocation integrity checks for native rank checkpoint receipts.
import math
import zlib
from core.foundation import file_sha256 as digest
from core.native import directory_owner


def verify_native_checkpoints(directory, ranks, *, require_receipts=False):
    directory=Path(directory)
    if type(ranks) is not int or ranks<=0:raise ValueError('Checkpoint rank count must be positive')
    files={}
    for path in directory.glob('restart.*'):
        match=re.fullmatch(r'restart\.(\d+)',path.name)
        if match:
            rank=int(match.group(1))
            if path.name!=f'restart.{rank}' or rank in files:
                raise ValueError('Noncanonical or duplicate checkpoint rank filename')
            files[rank]=path
    if set(files)!=set(range(ranks)) or any(not p.is_file() or p.stat().st_size==0 for p in files.values()):
        raise ValueError('Checkpoint requires one nonempty file per original rank')
    receipts=[Path(str(files[i])+'.checksum.json') for i in range(ranks)]
    if not any(p.exists() for p in receipts):
        if require_receipts:raise ValueError('Native checkpoint checksum receipts are missing')
        return {'status':'LEGACY_UNVERIFIED','ranks':ranks,'limitations':['No native checkpoint checksum/epoch receipt; legacy payload integrity unavailable']}
    rows=[]
    for rank,path in sorted(files.items()):
        rows.append(_verify_rank(path,rank,ranks))
    if any(row['state_layout'] != rows[0]['state_layout'] for row in rows):
        raise ValueError('Mixed checkpoint native state layouts across ranks')
    if any((row['clock_representation'],row['clock_value'])!=(rows[0]['clock_representation'],rows[0]['clock_value']) for row in rows):raise ValueError('Mixed checkpoint epochs across MPI ranks')
    if any(row['generation']!=rows[0]['generation'] for row in rows):raise ValueError('Mixed checkpoint generations across MPI ranks')
    return {'status':'VERIFIED_NATIVE_CHECKSUM','ranks':ranks,'a':rows[0]['a'],'z':rows[0]['z'],'clock_representation':rows[0]['clock_representation'],'clock_value':rows[0]['clock_value'],'files':rows,'checksum_scope':'CRC32 detects accidental corruption; no authenticity claim','cosmic_time_status':'NOT_APPLICABLE' if rows[0]['clock_representation']=='native_time_L_over_V' else 'Native receipt unavailable; derive from checksum-bound run clock separately'}


def _verify_rank(path,rank,ranks):
    receipt=Path(str(path)+'.checksum.json')
    if not receipt.is_file():raise ValueError('Incomplete checkpoint receipt set')
    receipt_sha=digest(receipt)
    entry=json.loads(receipt.read_text())
    if entry.get('schema') not in ('gadget-rank-checkpoint-v1','gadget-rank-checkpoint-v2') or type(entry.get('rank')) is not int or entry['rank']!=rank or type(entry.get('ranks')) is not int or entry['ranks']!=ranks:
        raise ValueError('Checkpoint receipt rank/schema mismatch')
    if type(entry.get('generation')) is not int or entry['generation']<=0:raise ValueError('Invalid checkpoint generation')
    comoving=entry['schema']=='gadget-rank-checkpoint-v1'
    a=entry.get('a' if comoving else 'native_time')
    if isinstance(a,bool) or not isinstance(a,(float,int)) or not math.isfinite(a) or a<0 or (comoving and a==0):raise ValueError('Invalid checkpoint native clock')
    if ('native_time' in entry if comoving else 'a' in entry):raise ValueError('Ambiguous native checkpoint clock representation')
    if type(entry.get('size_bytes')) is not int or entry['size_bytes']<=0 or entry['size_bytes']!=path.stat().st_size:raise ValueError('Checkpoint size mismatch')
    payload_sha=digest(path)
    crc=0
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):crc=zlib.crc32(block,crc)
    if entry.get('crc32')!=f'{crc:08x}':raise ValueError('Checkpoint checksum mismatch')
    if digest(path)!=payload_sha:raise ValueError('Checkpoint payload changed during validation')
    if digest(receipt)!=receipt_sha:raise ValueError('Checkpoint receipt changed during validation')
    epoch_path=Path(str(path)+'.epoch.json')
    epoch_metadata=None
    if epoch_path.exists():
        epoch_sha=digest(epoch_path)
        epoch=json.loads(epoch_path.read_text())
        if (epoch.get('schema')!=('native-checkpoint-epoch-v1' if comoving else 'native-checkpoint-epoch-v2') or epoch.get('generation')!=entry['generation']
                or epoch.get('rank')!=rank or epoch.get('a' if comoving else 'native_time')!=float(a)):
            raise ValueError('Checkpoint epoch sidecar belongs to another generation')
        if not comoving and (epoch.get('clock_representation')!='native_time_L_over_V' or epoch.get('cosmic_time_status')!='NOT_APPLICABLE' or epoch.get('cosmic_time_gyr') is not None or 'a' in epoch or 'z' in epoch):
            raise ValueError('Isolated checkpoint cannot declare cosmological epochs')
        if digest(epoch_path)!=epoch_sha:raise ValueError('Checkpoint epoch metadata changed during validation')
        epoch_metadata={'path':str(epoch_path.resolve()),'sha256':epoch_sha}
    state_layout = read_native_checkpoint_layout(path)
    if state_layout.get('ranks', ranks) != ranks:
        raise ValueError('Native state layout rank count disagrees with receipt')
    return {'state_layout':state_layout,'epoch_metadata':epoch_metadata,'rank':rank,'path':str(path.resolve()),'sha256':payload_sha,'receipt_sha256':receipt_sha,'generation':entry['generation'],'a':float(a) if comoving else None,'z':1/float(a)-1 if comoving else None,'clock_representation':'scale_factor' if comoving else 'native_time_L_over_V','clock_value':float(a),'size_bytes':entry['size_bytes']}


@directory_owner("directory")
def recover_native_checkpoints(directory, ranks):
    """Recover the newest coherent verified current/backup set with replayable writes.

    The run owner must be stopped and serialized. Original bytes, including
    malformed candidates and interrupted temp files, remain in an immutable archive.
    """
    import tempfile
    import uuid
    from core.foundation import atomic_copy, atomic_json, publish_directory
    directory=Path(directory)
    if type(ranks) is not int or ranks<=0:raise ValueError('Checkpoint rank count must be positive')
    transaction=directory/'checkpoint_recovery.json'
    existing=json.loads(transaction.read_text()) if transaction.is_file() else None
    if existing and existing.get('status')=='PENDING':
        record=existing
    else:
        try:
            current=verify_native_checkpoints(directory,ranks,require_receipts=True)
            return {'status':'ALREADY_VERIFIED','checkpoint':current}
        except (ValueError,OSError):
            pass
        candidates=[]; failures=[]
        for rank in range(ranks):
            options=[]
            for prefix in ['restart','bak-restart']:
                path=directory/f'{prefix}.{rank}'
                try:options.append(_verify_rank(path,rank,ranks))
                except (ValueError,OSError) as exc:
                    failures.append({'path':path.name,'error':str(exc)})
            candidates.append(options)
        common=set((r['clock_representation'],r['clock_value'],r['generation']) for r in candidates[0]) if candidates else set()
        for options in candidates[1:]:common &= {(r['clock_representation'],r['clock_value'],r['generation']) for r in options}
        if not common:raise ValueError('No complete coherent verified checkpoint generation is recoverable')
        if len({r['clock_representation'] for options in candidates for r in options})>1:
            raise ValueError('Cannot recover across different native clock representations')
        chosen=max(common)
        selected=[next(r for r in options if (r['clock_representation'],r['clock_value'],r['generation'])==chosen) for options in candidates]
        stage=Path(tempfile.mkdtemp(prefix='.checkpoint-recovery-',dir=directory))
        (stage/'original').mkdir();(stage/'selected').mkdir()
        artifacts=[]
        def copying(source,destination,expected=None):
            checksum=atomic_copy(source,destination)
            if expected is not None and checksum!=expected:raise ValueError('Checkpoint recovery candidate changed')
            artifacts.append({'path':str(destination.relative_to(stage)),'sha256':checksum})
        try:
            for path in sorted(directory.iterdir()):
                if path.is_file():copying(path,stage/'original'/path.name)
            for row in selected:
                path=Path(row['path']);target=stage/'selected'/f"restart.{row['rank']}"
                copying(path,target,row['sha256'])
                copying(Path(str(path)+'.checksum.json'),Path(str(target)+'.checksum.json'),row['receipt_sha256'])
                if row.get('epoch_metadata'):
                    copying(Path(row['epoch_metadata']['path']),Path(str(target)+'.epoch.json'),row['epoch_metadata']['sha256'])
            verify_native_checkpoints(stage/'selected',ranks,require_receipts=True)
            archive_name='checkpoint_recovery_'+uuid.uuid4().hex
            record={'schema':'native-checkpoint-recovery-v1','status':'PENDING','archive':archive_name,
                    'ranks':ranks,'a':chosen[1] if chosen[0]=='scale_factor' else None,'clock_representation':chosen[0],'clock_value':chosen[1],'generation':chosen[2],'artifacts':artifacts,'candidate_failures':failures}
            atomic_json(stage/'recovery.json',record)
            publish_directory(stage,directory/archive_name)
            record['archive_manifest_sha256']=digest(directory/archive_name/'recovery.json')
            atomic_json(transaction,record)
        except BaseException as exc:
            if stage.exists():atomic_json(stage/'recovery_failure.json',{'type':type(exc).__name__,'error':str(exc)})
            raise
    name=record.get('archive','')
    if not re.fullmatch(r'checkpoint_recovery_[0-9a-f]{32}',name) or record.get('ranks')!=ranks:
        raise ValueError('Malformed checkpoint recovery transaction')
    archive=directory/name
    if digest(archive/'recovery.json')!=record.get('archive_manifest_sha256'):
        raise ValueError('Checkpoint recovery archive manifest changed')
    archived_record=json.loads((archive/'recovery.json').read_text())
    if any(record.get(key)!=value for key,value in archived_record.items()):
        raise ValueError('Checkpoint recovery transaction differs from archived plan')
    for item in record['artifacts']:
        path=(archive/item['path']).resolve()
        if not path.is_relative_to(archive.resolve()) or digest(path)!=item['sha256']:
            raise ValueError('Checkpoint recovery archived bytes changed')
    verify_native_checkpoints(archive/'selected',ranks,require_receipts=True)
    for rank in range(ranks):
        source=archive/'selected'/f'restart.{rank}'
        atomic_copy(source,directory/source.name)
        atomic_copy(Path(str(source)+'.checksum.json'),directory/(source.name+'.checksum.json'))
        epoch_source=Path(str(source)+'.epoch.json')
        epoch_target=directory/(source.name+'.epoch.json')
        if epoch_source.exists():atomic_copy(epoch_source,epoch_target)
        elif epoch_target.exists():
            # Original sidecars remain in the archive; stale current-generation
            # time must not be paired with an older recovered payload.
            epoch_target.unlink()
            from core.foundation import fsync_directory
            fsync_directory(directory)
    verified=verify_native_checkpoints(directory,ranks,require_receipts=True)
    record.update(status='COMPLETE',checkpoint=verified)
    atomic_json(transaction,record)
    return record


def stage_native_checkpoint_clock(run_dir):
    """Supply a portable, checksum-bound clock to every periodic native writer.

    This is data/metadata transport. No GADGET background or force is changed.
    Legacy runs without a CLASS clock remain explicitly time-unavailable.
    """
    from core.cosmology import run_proper_time_clock
    from core.foundation import atomic_text, atomic_json
    root=Path(run_dir).resolve()
    a,t,source=run_proper_time_clock(root)
    path=root/'data'/'canonical-clock.tsv'
    if a is None:
        if path.exists():
            raise ValueError('An unbound native checkpoint clock already exists')
        return {'status':'UNAVAILABLE','reason':source['reason']}
    payload=''.join(f'{x:.17g} {y:.17g}\n' for x,y in zip(a,t))
    crc=f'{zlib.crc32(payload.encode()):08x}'
    text=f"# native-canonical-clock-v1 {source['sha256']} {crc}\n"+payload
    if path.exists() and path.read_text()!=text:
        raise ValueError('Prepared native checkpoint clock differs from its bound source')
    if not path.exists():atomic_text(path,text)
    record={'schema':'native-canonical-clock-v1','status':'VERIFIED','path':'data/canonical-clock.tsv',
            'sha256':digest(path),'source_sha256':source['sha256'],'payload_crc32':crc,
            'interpolation':'piecewise linear in a; no extrapolation','units':'Gyr',
            'numerical_error_envelope':source.get('numerical_error_envelope'),
            'science_status':'NOT_ACCEPTED'}
    sidecar=path.with_suffix('.json')
    atomic_json(sidecar,record)
    return record


def verify_checkpoint_epochs(directory, checkpoint, run_dir):
    """Validate periodic native epoch sidecars against payload generation and clock."""
    from core.cosmology import run_proper_time_clock, interpolate_proper_time
    directory=Path(directory)
    if checkpoint.get('clock_representation')=='native_time_L_over_V':
        current=verify_native_checkpoints(directory,checkpoint['ranks'],require_receipts=True)
        if current!=checkpoint:raise ValueError('Isolated checkpoint changed before epoch verification')
        return [json.loads((directory/f'restart.{rank}.epoch.json').read_text()) for rank in range(checkpoint['ranks'])]
    a,t,source=run_proper_time_clock(run_dir)
    rows=[]
    for rank in range(checkpoint['ranks']):
        path=directory/f'restart.{rank}.epoch.json'
        if not path.exists():
            rows.append({'rank':rank,'status':'LEGACY_UNAVAILABLE'})
            continue
        record=json.loads(path.read_text())
        if (record.get('schema')!='native-checkpoint-epoch-v1' or record.get('rank')!=rank
                or record.get('generation')!=checkpoint['files'][rank]['generation']
                or record.get('a')!=checkpoint['a'] or record.get('z')!=1/checkpoint['a']-1):
            raise ValueError('Periodic checkpoint epoch/generation sidecar mismatch')
        if record.get('cosmic_time_status')=='AVAILABLE':
            if a is None or record.get('clock_source_sha256')!=source['sha256']:
                raise ValueError('Periodic checkpoint time has no matching bound CLASS source')
            expected=interpolate_proper_time(a,t,checkpoint['a'])
            if not math.isclose(record['cosmic_time_gyr'],expected,rel_tol=2.e-14,abs_tol=1.e-14):
                raise ValueError('Periodic checkpoint time differs from canonical source interpolation')
        elif record.get('cosmic_time_status')!='UNAVAILABLE' or record.get('cosmic_time_gyr') is not None:
            raise ValueError('Malformed periodic checkpoint cosmic time')
        rows.append(dict(record,sha256=digest(path)))
    return rows


def read_native_checkpoint_layout(path, *, expected=None, require_layout=False):
    """Decode the native ABI prefix without restoring All or allocating particles.

    Legacy raw restarts remain explicitly unguarded. The native loader requires
    exact equality with its own generated header and refuses legacy prefixes.
    """
    import struct
    with Path(path).open('rb') as stream:
        header = stream.read(128)
    if header[:8] != b'R2RST001':
        if require_layout:
            raise ValueError('Native checkpoint lacks versioned state layout')
        return {'status': 'LEGACY_ABI_UNGUARDED'}
    if len(header) != 128:
        raise ValueError('Truncated native checkpoint layout')
    endian = 'little' if header[48:56] == bytes.fromhex('0807060504030201') else 'big'
    values = struct.unpack(('<' if endian == 'little' else '>') + '8s7Q64s', header)
    _, version, global_bytes, particle_bytes, sph_bytes, ranks, marker, compiler, fingerprint = values
    if version != 1 or marker != 0x0102030405060708 or min(global_bytes, particle_bytes, sph_bytes, ranks) <= 0:
        raise ValueError('Invalid native checkpoint layout')
    fingerprint = fingerprint.decode('ascii')
    if not re.fullmatch('[0-9a-f]{64}', fingerprint):
        raise ValueError('Invalid native source/configuration fingerprint')
    result = {'status': 'VERSIONED_STATE_LAYOUT', 'version': version, 'global_bytes': global_bytes,
        'particle_bytes': particle_bytes, 'sph_bytes': sph_bytes, 'ranks': ranks,
        'byte_order': endian, 'compiler_fingerprint': compiler,
        'source_config_sha256': fingerprint,
        # The ABI prefix identifies byte sizes and a source fingerprint, not
        # the fields inside a historical particle struct. Only an exact build
        # comparison can establish a particular physical state schema.
        'source_state_schema': 'BUILD_FINGERPRINT_REQUIRED'}
    if expected is not None and result != expected:
        raise ValueError('Native checkpoint state layout differs from expected build')
    return result

# ========================================================================
# hdf5_publication: Durable HDF5 product generations using the shared dataset contracts.
import tempfile,uuid
from core.foundation import atomic_copy, atomic_json, publish_directory


@directory_owner('destination')
def publish_hdf5_product(stage,destination, *, semantic_specs=None):
    stage,target=Path(stage),Path(destination).resolve()
    if semantic_specs is not None:
        import h5py
        with h5py.File(stage, 'r+') as source:
            numeric = []
            source.visititems(lambda name, data: numeric.append(name) if isinstance(data, h5py.Dataset) and data.dtype.kind in 'biufc' else None)
            if set(numeric) != set(semantic_specs):
                raise ValueError('Scientific publication requires semantics for every numeric dataset')
            for name in numeric:
                publish_scientific_dataset(source[name], semantic_specs[name])
    verify_hdf5_dataset_payloads(stage,require_contract=True)
    history=target.parent/'.hdf5_product_history';history.mkdir(parents=True,exist_ok=True)
    temporary=Path(tempfile.mkdtemp(prefix='.generation-',dir=history));final=history/(target.name+'_'+uuid.uuid4().hex)
    try:
        before=None
        if target.is_file():before=atomic_copy(target,temporary/'previous.hdf5')
        checksum=atomic_copy(stage,temporary/target.name)
        record={'schema':'hdf5-product-publication-v1','file':target.name,'sha256':checksum,
            'previous_public_sha256':before,'generation':final.relative_to(target.parent).as_posix(),
            'science_acceptance':'NOT_ACCEPTED'}
        atomic_json(temporary/'publication.json',record)
        (temporary/'publication.sha256').write_text(digest(temporary/'publication.json')+'\n')
        publish_directory(temporary,final)
        atomic_json(str(target)+'.publication.json',{'state':'PENDING','generation':record['generation'],
            'manifest_sha256':digest(final/'publication.json')})
        atomic_copy(final/target.name,target)
        atomic_json(str(target)+'.publication.json',{'state':'COMPLETE','generation':record['generation'],
            'manifest_sha256':digest(final/'publication.json')})
        return record
    except BaseException as exc:
        if temporary.exists():atomic_json(temporary/'failure.json',{'status':'FAILED','type':type(exc).__name__,'message':str(exc)})
        raise


def verify_hdf5_product(path, *, memory_bytes=16*1024**2):
    path=Path(path).resolve();pointer=Path(str(path)+'.publication.json')
    if not pointer.exists():
        return {'status':'LEGACY_UNSEALED','numeric_payloads':verify_hdf5_dataset_payloads(path, memory_bytes=memory_bytes),
            'science_acceptance':'NOT_ACCEPTED'}
    index=json.loads(pointer.read_text())
    if index.get('state')!='COMPLETE':raise ValueError('HDF5 publication is pending; explicitly recover its sealed generation')
    bundle=(path.parent/index['generation']).resolve()
    if not bundle.is_relative_to(path.parent/'.hdf5_product_history'):raise ValueError('HDF5 generation escapes history')
    if digest(bundle/'publication.json')!=index['manifest_sha256']:raise ValueError('HDF5 publication manifest changed')
    record=json.loads((bundle/'publication.json').read_text())
    if record.get('schema')!='hdf5-product-publication-v1' or record['file']!=path.name:
        raise ValueError('HDF5 generation does not belong to requested product')
    if digest(path)!=record['sha256'] or digest(bundle/path.name)!=record['sha256']:
        raise ValueError('HDF5 product payload changed')
    verify_hdf5_dataset_payloads(path,require_contract=True, memory_bytes=memory_bytes)
    return record


@directory_owner('destination')
def recover_hdf5_product(destination,generation):
    """Replay archived bytes and their pointer after an interrupted publication."""
    target=Path(destination).resolve();bundle=Path(generation)
    if not bundle.is_absolute():bundle=target.parent/bundle
    bundle=bundle.resolve()
    if not bundle.is_relative_to(target.parent/'.hdf5_product_history'):
        raise ValueError('HDF5 recovery generation escapes history')
    manifest=bundle/'publication.json'
    if digest(manifest)!=(bundle/'publication.sha256').read_text().strip():
        raise ValueError('HDF5 recovery manifest changed')
    record=json.loads(manifest.read_text())
    if record.get('schema')!='hdf5-product-publication-v1' or record['file']!=target.name:
        raise ValueError('HDF5 recovery generation belongs to another product')
    if digest(bundle/target.name)!=record['sha256']:raise ValueError('HDF5 recovery bytes changed')
    verify_hdf5_dataset_payloads(bundle/target.name,require_contract=True)
    atomic_json(str(target)+'.publication.json',{'state':'PENDING','generation':bundle.relative_to(target.parent).as_posix(),
        'manifest_sha256':digest(manifest)})
    atomic_copy(bundle/target.name,target)
    atomic_json(str(target)+'.publication.json',{'state':'COMPLETE','generation':bundle.relative_to(target.parent).as_posix(),
        'manifest_sha256':digest(manifest)})
    return verify_hdf5_product(target)


@directory_owner('destination')
def publish_numeric_table(destination,table,specifications,*,algorithm,context=None,
                          source_hashes=None,fmt='%.17e',header=''):
    """Contract actual serialized columns, preserving native text format and units."""
    target=Path(destination).resolve();target.parent.mkdir(parents=True,exist_ok=True)
    table=np.asarray(table)
    if table.ndim!=2 or not len(table) or table.shape[1]!=len(specifications):
        raise ValueError('Declare every nonempty numeric table column in order')
    before={str(Path(path).resolve()):checksum for path,checksum in (source_hashes or {}).items()}
    for path,checksum in before.items():
        if digest(path)!=checksum:raise ValueError('Numeric table input changed before publication')
    stage=Path(tempfile.mkdtemp(prefix='.numeric-table-',dir=target.parent))
    history=target.parent/'.numeric_table_history';history.mkdir(exist_ok=True)
    final=history/(target.name+'_'+uuid.uuid4().hex)
    context=context or {}
    try:
        previous=None
        if target.is_file():
            previous=atomic_copy(target,stage/'previous_public_table.dat')
            for suffix in ('.contract.json','.contract.sha256'):
                old=Path(str(target)+suffix)
                if old.is_file():atomic_copy(old,stage/('previous_public_table'+suffix))
        np.savetxt(stage/target.name,table,fmt=fmt,header=header)
        actual=np.loadtxt(stage/target.name,ndmin=2)
        arrays={name:actual[:,column] for column,name in enumerate(specifications)}
        np.savez_compressed(stage/'columns.npz',**arrays)
        contract=numpy_archive_contract(stage/'columns.npz',specifications,epoch=context.get('epoch'),
            coordinates=context.get('coordinates'),provenance={**context,'algorithm_sha256':digest(algorithm),
                'input_hashes':before,'decimal_serialization':fmt})
        record={'schema':'native-numeric-table-v1','table_name':target.name,'table_sha256':digest(stage/target.name),
            'columns':list(specifications),'numeric_contract':contract,'context':context,
            'previous_public_table_sha256':previous,
            'generation':final.relative_to(target.parent).as_posix(),'science_acceptance':'NOT_ACCEPTED'}
        atomic_json(stage/'table.contract.json',record)
        (stage/'table.contract.sha256').write_text(digest(stage/'table.contract.json')+'\n')
        for path,checksum in before.items():
            if digest(path)!=checksum:raise ValueError('Numeric table input changed during publication')
        publish_directory(stage,final)
        # Close discovery before changing the public text/contract pair.
        atomic_json(str(target)+'.publication.json',{'state':'PENDING','generation':record['generation'],
            'manifest_sha256':digest(final/'table.contract.json')})
        atomic_copy(final/target.name,target)
        atomic_copy(final/'table.contract.json',Path(str(target)+'.contract.json'))
        atomic_copy(final/'table.contract.sha256',Path(str(target)+'.contract.sha256'))
        atomic_json(str(target)+'.publication.json',{'state':'COMPLETE','generation':record['generation'],
            'manifest_sha256':digest(final/'table.contract.json')})
        return record
    except BaseException as exc:
        if stage.exists():atomic_json(stage/'failure.json',{'status':'FAILED','type':type(exc).__name__,'message':str(exc)})
        raise


def verify_numeric_table(path):
    """Reject changed published tables and generation arrays; legacy tables stay explicit."""
    path=Path(path).resolve();manifest=Path(str(path)+'.contract.json')
    pointer=Path(str(path)+'.publication.json');index=None
    if pointer.exists():
        index=json.loads(pointer.read_text())
        if index.get('state')!='COMPLETE':raise ValueError('Numeric table publication is pending; explicitly recover its sealed generation')
    if not manifest.exists():return {'status':'LEGACY_UNSEALED','science_acceptance':'NOT_ACCEPTED'}
    if digest(manifest)!=Path(str(path)+'.contract.sha256').read_text().strip():
        raise ValueError('Numeric table manifest checksum mismatch')
    record=json.loads(manifest.read_text())
    if index is not None and (index['manifest_sha256']!=digest(manifest) or index['generation']!=record['generation']):
        raise ValueError('Numeric table publication pointer mismatch')
    if record.get('schema')!='native-numeric-table-v1' or digest(path)!=record['table_sha256']:
        raise ValueError('Numeric table payload checksum mismatch')
    generation=(path.parent/record['generation']).resolve()
    if not generation.is_relative_to(path.parent/'.numeric_table_history'):
        raise ValueError('Numeric table generation escapes history')
    if digest(generation/'table.contract.json')!=digest(manifest) or digest(generation/path.name)!=record['table_sha256']:
        raise ValueError('Numeric table generation checksum mismatch')
    verify_numpy_archive_contract(generation/'columns.npz',record['numeric_contract'])
    return record


@directory_owner('destination')
def recover_numeric_table(destination,generation):
    """Replay a sealed generation after interrupted native text publication."""
    target=Path(destination).resolve();bundle=Path(generation)
    if not bundle.is_absolute():bundle=target.parent/bundle
    bundle=bundle.resolve()
    if not bundle.is_relative_to(target.parent/'.numeric_table_history'):
        raise ValueError('Recovery generation escapes numeric table history')
    manifest=bundle/'table.contract.json'
    if digest(manifest)!=(bundle/'table.contract.sha256').read_text().strip():
        raise ValueError('Recovery generation manifest changed')
    record=json.loads(manifest.read_text())
    if record.get('schema')!='native-numeric-table-v1' or record['table_name']!=target.name:
        raise ValueError('Recovery generation belongs to another native table')
    if digest(bundle/target.name)!=record['table_sha256']:raise ValueError('Recovery table changed')
    verify_numpy_archive_contract(bundle/'columns.npz',record['numeric_contract'])
    atomic_json(str(target)+'.publication.json',{'state':'PENDING','generation':record['generation'],
        'manifest_sha256':digest(manifest)})
    atomic_copy(bundle/target.name,target)
    atomic_copy(manifest,Path(str(target)+'.contract.json'))
    atomic_copy(bundle/'table.contract.sha256',Path(str(target)+'.contract.sha256'))
    atomic_json(str(target)+'.publication.json',{'state':'COMPLETE','generation':record['generation'],
        'manifest_sha256':digest(manifest)})
    return verify_numeric_table(target)


@directory_owner('destination')
def copy_numeric_table(source, destination):
    """Copy a verified native table with its current immutable contract history."""
    source, target = Path(source).resolve(), Path(destination).resolve()
    if source == target:raise ValueError('Numeric table copy needs distinct source and destination')
    if target.exists():raise FileExistsError('Keep previous staged numeric table')
    before=digest(source)
    record=verify_numeric_table(source)
    target.parent.mkdir(parents=True,exist_ok=True)
    if record.get('status')!='LEGACY_UNSEALED':
        # Preserve native contract filenames; renaming requires republishing.
        if target.name!=source.name:raise ValueError('Contracted table staging preserves its native filename')
        relative=Path(record['generation'])
        generation=source.parent/relative
        copied=target.parent/relative
        copied.parent.mkdir(parents=True,exist_ok=True)
        if copied.exists():raise FileExistsError('Keep previous numeric table history')
        stage=Path(tempfile.mkdtemp(prefix='.numeric-table-copy-',dir=target.parent))
        try:
            for path in generation.rglob('*'):
                if path.is_file():atomic_copy(path,stage/path.relative_to(generation))
            publish_directory(stage,copied)
        except BaseException as exc:
            if stage.exists():atomic_json(stage/'failure.json',{'status':'FAIL','message':str(exc)})
            raise
        atomic_json(str(target)+'.publication.json',{'state':'PENDING','generation':record['generation'],
            'manifest_sha256':digest(source.parent/relative/'table.contract.json')})
        atomic_copy(source,target)
        for suffix in ('.contract.json','.contract.sha256'):
            atomic_copy(str(source)+suffix,str(target)+suffix)
        atomic_json(str(target)+'.publication.json',{'state':'COMPLETE','generation':record['generation'],
            'manifest_sha256':digest(str(target)+'.contract.json')})
    else:
        atomic_copy(source,target)
    sidecar=Path(str(source)+'.json')
    if sidecar.is_file():atomic_copy(sidecar,str(target)+'.json')
    if digest(source)!=before:raise ValueError('Numeric table source changed while staging')
    verify_numeric_table(source)
    verify_numeric_table(target)
    return record

# ========================================================================
# catalogue_publication: Replayable publication of one native catalogue plus matching particle order.

from core.native import native_run_owner
from core.foundation import fsync_directory

SCHEMA='native-catalogue-publication-v1'


def _inside(root,relative):
    relative=Path(relative)
    if relative.is_absolute() or '..' in relative.parts or not relative.parts:
        raise ValueError('Catalogue transaction path escapes its declared root')
    path=root/relative
    if not path.resolve().is_relative_to(root.resolve()) or path.is_symlink():
        raise ValueError('Catalogue transaction path is an external alias')
    return path


def _journal(root,snapshot):
    if isinstance(snapshot,bool) or not isinstance(snapshot,int) or snapshot<0:
        raise ValueError('Catalogue snapshot must be a nonnegative storage index')
    return root/'data/.catalogue_publications'/f'{snapshot:03d}.json'


def _load(root,journal):
    record=json.loads(journal.read_text())
    if record.get('schema')!=SCHEMA or record.get('status') not in ('PENDING','COMPLETE'):
        raise ValueError('Invalid native catalogue publication journal')
    if journal!=_journal(root,record['snapshot']):raise ValueError('Catalogue journal storage index mismatch')
    archive=_inside(root,record['archive']);plan_path=archive/'plan.json'
    if digest(plan_path)!=record['plan_sha256']:raise ValueError('Catalogue immutable publication plan checksum mismatch')
    plan=json.loads(plan_path.read_text())
    if plan['schema']!=SCHEMA or plan['snapshot']!=record['snapshot']:
        raise ValueError('Catalogue journal/plan epoch mismatch')
    for section in ('previous','next'):
        names=[entry['path'] for entry in plan[section]]
        if len(names)!=len(set(names)):raise ValueError('Duplicate catalogue transaction members')
        for entry in plan[section]:
            _inside(root/'data',entry['path'])
            saved=_inside(archive/section,entry['path'])
            if not saved.is_file() or digest(saved)!=entry['sha256']:
                raise ValueError('Catalogue transaction archived bytes missing or changed: '+entry['path'])
    return record,plan,archive


def assert_catalogue_publication_ready(run_dir,snapshot):
    """Reject pending generations or changes to a published checksummed set."""
    root=Path(run_dir).resolve()
    # Selected analysis owns an isolated workspace with a read-only link to
    # the actual source data. Resolve journal archives against that source run.
    if (root/'data').is_symlink():
        actual=(root/'data').resolve()
        if actual.name!='data':raise ValueError('Catalogue data alias lacks a declared source run layout')
        root=actual.parent
    journal=_journal(root,int(snapshot))
    if not journal.exists():return
    record,plan,_=_load(root,journal)
    if record['status']!='COMPLETE':
        raise ValueError('Incomplete catalogue publication; explicit replay required: '+str(journal))
    for entry in plan['next']:
        path=_inside(root/'data',entry['path'])
        if not path.is_file() or digest(path)!=entry['sha256']:
            raise ValueError('Published native catalogue checksum mismatch: '+entry['path'])
    new_names={entry['path'] for entry in plan['next']}
    for entry in plan['previous']:
        if entry['path'] not in new_names and _inside(root/'data',entry['path']).exists():
            raise ValueError('Stale native catalogue member reappeared after publication')


@native_run_owner
def replay_catalogue_publication(run_dir,snapshot):
    root=Path(run_dir).resolve();journal=_journal(root,snapshot)
    if (root/'data').is_symlink():raise ValueError('Catalogue recovery requires an owned data directory')
    record,plan,archive=_load(root,journal)
    previous={entry['path']:entry['sha256'] for entry in plan['previous']}
    following={entry['path']:entry['sha256'] for entry in plan['next']}
    # Validate the whole current set before the first replay mutation.
    for name in set(previous)|set(following):
        target=_inside(root/'data',name)
        current=digest(target) if target.is_file() else None
        allowed={previous.get(name),following.get(name)}
        if current not in allowed:raise ValueError('Catalogue target changed outside its publication transaction: '+name)
        if record['status']=='COMPLETE' and current!=following.get(name):
            raise ValueError('Completed native catalogue publication no longer matches its bytes')
    if record['status']=='COMPLETE':return record
    for name,checksum in following.items():
        target=_inside(root/'data',name)
        if target.is_file() and digest(target)==checksum:continue
        atomic_copy(_inside(archive/'next',name),target)
    for name in set(previous)-set(following):
        target=_inside(root/'data',name)
        if target.exists():target.unlink();fsync_directory(target.parent)
    for name,checksum in following.items():
        if digest(_inside(root/'data',name))!=checksum:raise ValueError('Catalogue commit checksum mismatch')
    completed={**record,'status':'COMPLETE','science_acceptance':'NOT_ACCEPTED'}
    atomic_json(journal,completed)
    atomic_json(archive/'completion.json',completed)
    return completed


@native_run_owner
def publish_catalogue_set(run_dir,snapshot,outputs,previous,*,epoch,provenance):
    """Save both generations, close discovery, then publish exact staged bytes.

    outputs is a list of (closed native source, relative data destination).
    previous is the complete discovered old catalogue/ordered-particle set.
    Caller validated native set completeness and actual source epoch beforehand.
    """
    root=Path(run_dir).resolve();journal=_journal(root,snapshot)
    if (root/'data').is_symlink():raise ValueError('Catalogue publication requires an owned data directory')
    outputs = list(outputs)
    previous = list(previous)
    outputs += native_publication_copy_members(outputs)
    previous = list(dict.fromkeys(previous + native_publication_markers(previous)))
    assert_catalogue_publication_ready(root,snapshot)
    archive_parent=root/'results/catalogue_publications';archive_parent.mkdir(parents=True,exist_ok=True)
    stage=Path(tempfile.mkdtemp(prefix='.catalogue-',dir=archive_parent))
    archive=archive_parent/(f'snapshot_{snapshot:03d}_'+uuid.uuid4().hex)
    try:
        plan={'schema':SCHEMA,'snapshot':snapshot,'epoch_contract':epoch,'data_contract':provenance,
            'science_acceptance':'NOT_ACCEPTED','previous':[],'next':[]}
        for source in previous:
            source=Path(source).resolve();relative=source.relative_to(root/'data').as_posix()
            _inside(root/'data',relative)
            checksum = digest(source)
            saved = _inside(stage/'previous',relative)
            atomic_copy(source,saved)
            if digest(source) != checksum or digest(saved) != checksum:
                raise ValueError('Previous native catalogue changed during archive copy')
            plan['previous'].append({'path':relative,'sha256':checksum})
        for source,relative in outputs:
            relative=Path(relative).as_posix();_inside(root/'data',relative)
            checksum = digest(source)
            saved = _inside(stage/'next',relative)
            atomic_copy(source,saved)
            if digest(source) != checksum or digest(saved) != checksum:
                raise ValueError('Next native catalogue changed during archive copy')
            plan['next'].append({'path':relative,'sha256':checksum})
        if not plan['next']:raise ValueError('Cannot publish an empty native catalogue set')
        for section in ('previous','next'):
            names=[entry['path'] for entry in plan[section]]
            if len(names)!=len(set(names)):raise ValueError('Duplicate catalogue transaction members')
        atomic_json(stage/'plan.json',plan)
        publish_directory(stage,archive)
        pending={'schema':SCHEMA,'status':'PENDING','snapshot':snapshot,
            'archive':archive.relative_to(root).as_posix(),'plan_sha256':digest(archive/'plan.json'),
            'science_acceptance':'NOT_ACCEPTED'}
        # Only after both immutable generations are durable may discovery close.
        atomic_json(journal,pending)
        return replay_catalogue_publication(root,snapshot)
    except BaseException as exc:
        retained=stage if stage.exists() else archive
        if retained.exists():atomic_json(retained/'failure.json',{'status':'FAIL','science_acceptance':'NOT_ACCEPTED',
            'failure':{'type':type(exc).__name__,'message':str(exc)}})
        raise

# ========================================================================
# native_catalogue_bundles: Immutable Rockstar/tree generations using the central scientific contract.
import functools
import inspect
import shutil
from core.foundation import atomic_text, product_provenance, verify_product_provenance


def rockstar_directory(run_dir):
    """Resolve the current sealed generation, retaining explicit legacy fallback."""
    root=Path(run_dir).resolve()
    if (root/'data').is_symlink() and (root/'data').resolve().name=='data':root=(root/'data').resolve().parent
    index=root/'input/native_catalogue_index.json'
    if not index.is_file():return root/'data/rockstar'
    record=json.loads(index.read_text())
    generation=Path(record['rockstar']['directory'])
    if not generation.is_absolute():generation=root/generation
    generation=generation.resolve()
    if not generation.is_relative_to(root/'results/native_catalogues'):
        raise ValueError('Native catalogue generation escapes its archive root')
    manifest=generation/'bundle.json'
    if digest(manifest)!=record['rockstar']['manifest_sha256']:raise ValueError('Native catalogue manifest checksum mismatch')
    saved=json.loads(manifest.read_text())
    if saved.get('schema')!='native-catalogue-bundle-v1' or saved.get('status')!='COMPLETE':
        raise ValueError('Native catalogue index references an incomplete generation')
    for item in saved['assets']:
        relative=Path(item['path']);asset=generation/'data/rockstar'/relative
        if relative.is_absolute() or '..' in relative.parts or not asset.resolve().is_relative_to(generation/'data/rockstar'):
            raise ValueError('Invalid native catalogue asset path')
        if not asset.is_file() or digest(asset)!=item['sha256']:raise ValueError('Native catalogue asset checksum mismatch: '+str(relative))
    for item in saved.get('input_assets',[]):
        relative=Path(item['path']);asset=generation/'input'/relative
        if relative.is_absolute() or '..' in relative.parts or not asset.resolve().is_relative_to(generation/'input'):
            raise ValueError('Invalid native recipe/log asset path')
        if not asset.is_file() or digest(asset)!=item['sha256']:raise ValueError('Native recipe/log checksum mismatch')
    return generation/'data/rockstar'



def rebase_recipe_paths(text, replacements):
    """Replace original path prefixes once, longest first, without rebasing inserts."""
    prefixes = sorted(replacements, key=len, reverse=True)
    if not prefixes:
        return text
    pattern = re.compile("(?:" + "|".join(re.escape(p) for p in prefixes) + r')(?=/|[\s"\']|$)')
    return pattern.sub(lambda match: replacements[match.group(0)], text)


def staged_native_catalogues(function):
    """Run existing native code in an owned copy and publish its completed bytes."""
    signature=inspect.signature(function)
    @functools.wraps(function)
    @native_run_owner
    def wrapped(run_dir,*args,**kwargs):
        bound=signature.bind(run_dir,*args,**kwargs);bound.apply_defaults()
        if bound.arguments.get('dry_run'):return function(run_dir,*args,**kwargs)
        if function.__name__=='run_rockstar':
            from halos import rockstar_worker_count
            if rockstar_worker_count(bound.arguments['cfg'])<8:return function(run_dir,*args,**kwargs)
        root=Path(run_dir).resolve();parent=root/'results/native_catalogues';parent.mkdir(parents=True,exist_ok=True)
        stage=Path(tempfile.mkdtemp(prefix='.'+function.__name__+'-',dir=parent))
        final=parent/(function.__name__+'_'+uuid.uuid4().hex)
        (stage/'input').mkdir();(stage/'data').mkdir()
        sources=[]
        try:
            for name in ('config.yaml','manifest.json'):
                if (root/name).is_file():atomic_copy(root/name,stage/name)
            # Scientific particle data remain read-only source references. Native
            # catalogue/log/config writes all stay inside this isolated workspace.
            for source in (root/'data').iterdir() if (root/'data').is_dir() else []:
                if source.name=='rockstar' or source.name.startswith('.'):continue
                (stage/'data'/source.name).symlink_to(source.resolve(),target_is_directory=source.is_dir())
            for source in (root/'input').iterdir() if (root/'input').is_dir() else []:
                if source.name.startswith(('halo_stage','rockstar_snapshots','native_catalogue_index')):continue
                if source.is_file():atomic_copy(source,stage/'input'/source.name)
                elif source.is_dir():
                    # Recipes and central generation histories may be written by
                    # staged callers. A directory alias would mutate the source run.
                    shutil.copytree(source,stage/'input'/source.name,symlinks=False)
            previous=rockstar_directory(root)
            if function.__name__!='run_rockstar' and previous.is_dir():
                shutil.copytree(previous,stage/'data/rockstar',symlinks=False)
                sources.extend(path for path in previous.rglob('*') if path.is_file())
                previous_inputs=previous.parent.parent/'input'
                if previous_inputs.is_dir():
                    for source in previous_inputs.glob('*.cfg'):
                        atomic_copy(source,stage/'input'/source.name);sources.append(source)
            from halos import snapshot_sets
            from analysis import validate_gadget_snapshot_set
            sets=snapshot_sets(root);sources.extend(path for paths in sets.values() for path in paths)
            epochs={str(number):run_epoch_contract(root,number,validate_gadget_snapshot_set(paths,number)['scale_factor'])
                    for number,paths in sets.items()}
            contract=product_provenance(root,algorithm=inspect.getfile(inspect.unwrap(function)),source_files=sources,
                helper_sources=[Path(__file__),Path(__file__).with_name('foundation.py')],selection='Native '+function.__name__+' on exact saved source catalogues/particles',
                coordinates=coordinate_contract(position_units='comoving Mpc/h',origin='global simulation box origin',
                    periodic_wrapping='native periodic box',velocity_convention='physical peculiar km/s in Rockstar catalogues'))
            # Native Perl generators reuse absolute paths from finder recipes.
            # Rebase only these native workspace recipes, never scientific data.
            for recipe in [* (stage/'input').glob('*.cfg'),* (stage/'data/rockstar').rglob('*.cfg')]:
                atomic_text(recipe,rebase_recipe_paths(recipe.read_text(), {str(previous.parent.parent):str(stage), str(root):str(stage)}))
            from core.cosmology import native_source_manifest,verify_native_source_manifest
            software=Path(__file__).resolve().parent/'software'
            engines={'run_rockstar':['rockstar'],'run_consistent_trees':['rockstar','consistent-trees'],
                'prepare_rockstar_host_catalogues':['rockstar']}.get(function.__name__,[])
            native_sources={engine:native_source_manifest(engine,software/engine) for engine in engines}
            atomic_json(stage/'native_sources.json',native_sources)
            result=function(stage,*args,**kwargs)
            for source_manifest in native_sources.values():verify_native_source_manifest(source_manifest)
            native=stage/'data/rockstar'
            if not native.is_dir():raise ValueError('Native catalogue route produced no output directory')
            for recipe in [*(stage/'input').glob('*.cfg'),*native.rglob('*.cfg')]:
                atomic_text(recipe,recipe.read_text().replace(str(stage),str(final)))
            def relocate(value):
                if isinstance(value,str):return value.replace(str(stage),str(final))
                if isinstance(value,dict):return {key:relocate(item) for key,item in value.items()}
                if isinstance(value,list):return [relocate(item) for item in value]
                return value
            result=relocate(result)
            assets=[]
            for path in sorted(native.rglob('*')):
                if path.is_symlink():raise ValueError('Native catalogue output contains an external alias')
                if path.is_file():assets.append({'path':path.relative_to(native).as_posix(),'sha256':digest(path),'size_bytes':path.stat().st_size})
            manifest={'schema':'native-catalogue-bundle-v1','operation':function.__name__,'status':result.get('status'),
                'science_acceptance':'NOT_ACCEPTED','data_contract':contract,'epochs':epochs,'assets':assets,
                'input_assets':[{'path':path.relative_to(stage/'input').as_posix(),'sha256':digest(path),'size_bytes':path.stat().st_size}
                    for path in sorted((stage/'input').rglob('*')) if path.is_file() and not path.is_symlink()],
                'identity_policy':'Native signed finder/tree links retain explicit -1 sentinels; public scientific identity consumers convert available IDs exactly to uint64',
                'native_status':result,'native_sources':native_sources}
            atomic_json(stage/'bundle.json',manifest);verify_product_provenance(root,contract)
            publish_directory(stage,final)
            if result.get('status')=='COMPLETE':
                atomic_json(root/'input/native_catalogue_index.json',{'schema':'native-catalogue-index-v1',
                    'rockstar':{'directory':final.relative_to(root).as_posix(),'manifest_sha256':digest(final/'bundle.json')},
                    'science_acceptance':'NOT_ACCEPTED'})
            return {**result,'bundle_directory':str(final),'science_acceptance':'NOT_ACCEPTED'}
        except BaseException as exc:
            if stage.exists():atomic_json(stage/'failure.json',{'status':'FAIL','science_acceptance':'NOT_ACCEPTED',
                'failure':{'type':type(exc).__name__,'message':str(exc)}})
            raise
    return wrapped

# ========================================================================
# comparison_contracts: Audited paired-analysis semantics using the shared dataset contract writer.
from core.foundation import canonical_digest

UNITS={}
for names,unit in [
    ('mass_edges_msun_h mass_centers_msun_h mass_bin_edges_msun_h mass_bin_centers_msun_h hmf_mass_bin_centers_msun_h concentration_mass_bin_centers_msun_h mass_a_msun_h mass_b_msun_h progenitor_mass_msun_h_a progenitor_mass_msun_h_b matched_mass_msun_h_a_b','Msun/h'),
    ('position_a_mpc_h position_b_mpc_h position_delta_b_minus_a_mpc_h position_displacement_mpc_h r200_critical_a_mpc_h r200_critical_b_mpc_h r200_mean_a_mpc_h r200_mean_b_mpc_h xi_r','comoving Mpc/h'),
    ('pk_k_h_mpc pk_ratio_x bias_x','h/Mpc'),
    ('subhalo_hmf_x','Msun/h'),
    ('dm_velocity_pdf_x halo_velocity_pdf_x central_subhalo_vmax_a central_subhalo_vmax_b','km/s'),
    ('dn_dlogM_a dn_dlogM_b','(Mpc/h)^-3'),
    ('sfr_a_msun_per_year sfr_b_msun_per_year','Msun/year'),
    ('counts_a counts_b hmf_counts_a hmf_counts_b concentration_count_a concentration_count_b gas_density_valid_halo_count gas_temperature_valid_halo_count','halo count'),
    ('shared_particle_count_a shared_particle_count_b shared_dark_matter_particle_count dark_matter_particle_count_a dark_matter_particle_count_b','particle count'),
    ('redshift snapshot_redshift_a snapshot_redshift_b formation_redshift_half_mass_a formation_redshift_half_mass_b formation_redshift_shift_b_minus_a hmf_ratio_redshift pk_ratio_redshift','dimensionless redshift'),
    ('log10_density_bin_edges','log10(n_H / cm^-3)'),
    ('log10_temperature_bin_edges','log10(T / K)'),
    ('log10_ratio_b_over_a hmf_ratio_log10_ratio pk_ratio_log10_ratio','dimensionless log10(B/A)'),
    ('log_ratio_paired_jackknife_covariance log_hmf_jackknife_covariance_a log_hmf_jackknife_covariance_b log_hmf_jackknife_cross_covariance_a_b','dimensionless covariance of natural logarithms'),
    ('fraction_a fraction_b ratio_b_over_a ratio_poisson_68_lower ratio_poisson_68_upper ratio_paired_jackknife_68_lower ratio_paired_jackknife_68_upper hmf_ratio_b_over_a hmf_ratio_poisson_68_lower hmf_ratio_poisson_68_upper hmf_ratio_paired_jackknife_68_lower hmf_ratio_paired_jackknife_68_upper pk_ratio_b_over_a xi_ratio_b_over_a bias_ratio_b_over_a subhalo_hmf_ratio_b_over_a dm_velocity_pdf_ratio_b_over_a halo_velocity_pdf_ratio_b_over_a concentration_median_ratio_b_over_a mass_ratio_b_over_a overlap_fraction_a overlap_fraction_b overlap_merit central_subhalo_vmax_ratio_b_over_a radius_over_r200c gas_density_ratio_b_over_a gas_temperature_ratio_b_over_a gas_density_median_ratio gas_temperature_median_ratio gas_a_median_fraction gas_b_median_fraction stars_a_median_fraction stars_b_median_fraction gas_fraction_difference_b_minus_a stellar_fraction_difference_b_minus_a','dimensionless'),
    ('hmf_ratio_mass_edges_msun_h hmf_ratio_mass_centers_msun_h','Msun/h'),
    ('formation_status_a formation_status_b','UTF8 categorical status'),
]:
    UNITS.update({name:unit for name in names.split()})

IDENTITIES={'group_index_a':'a','group_index_b':'b','unmatched_group_indices_a':'a',
    'unmatched_group_indices_b':'b','progenitor_group_index_a':'a','progenitor_group_index_b':'b',
    'matched_group_index_a_b':'both'}


def attach_comparison_contracts(handle, context, pair_epochs):
    """Declare known physical arrays and preserve unknown semantics explicitly."""
    coordinates=coordinate_contract(position_units='comoving Mpc/h',origin='global periodic simulation Cartesian box',
        periodic_wrapping='positions wrapped; differences use minimum periodic image',velocity_convention='peculiar km/s')
    provenance={'input_contract':'comparison_inputs.json','algorithm':'compare_runs',
        'source_analyses':context['analyses'],'science_acceptance':'NOT_ACCEPTED'}
    handle.attrs['data_contract_json']=json.dumps(context,sort_keys=True,allow_nan=False)
    datasets=[]
    handle.visititems(lambda name,item:datasets.append(name) if isinstance(item,h5py.Dataset) else None)
    identity_records={}
    for name in datasets:
        dataset=handle[name];leaf=name.rsplit('/',1)[-1]
        pair=name.split('/')[1] if name.startswith('pairs/') else None
        epochs=pair_epochs.get(pair)
        if leaf not in IDENTITIES:continue
        side=IDENTITIES[leaf];values,available=canonical_native_links(dataset[()])
        parent=dataset.parent;temporary='.identity-'+uuid.uuid4().hex
        replaced=parent.create_dataset(temporary,data=values,chunks=True,fletcher32=True) if values.ndim and values.size else parent.create_dataset(temporary,data=values)
        for key,value in dataset.attrs.items():replaced.attrs[key]=value
        del parent[leaf];parent.move(temporary,leaf)
        mask_name=leaf+'_available'
        mask=parent.create_dataset(mask_name,data=available,chunks=True,fletcher32=True) if available.ndim and available.size else parent.create_dataset(mask_name,data=available)
        namespaces={s:context['run_'+s]['identity_namespaces']['halo']+':snapshot:'+
            str(handle['pairs'][pair].attrs.get('snapshot_'+s,pair))+':gadget_fof_row' for s in ('a','b')}
        if leaf.startswith('progenitor_group_index_'):
            namespaces={s:context['run_'+s]['identity_namespaces']['halo']+':fof_rows_by_saved_epoch' for s in ('a','b')}
        ancestry={'relation':'reciprocal exact particle-overlap association; catalogue rows are not shared IDs',
            'source_domains':namespaces,'side':side,'temporal_persistence':'NOT_DECLARED'}
        if leaf.startswith('progenitor_group_index_'):
            redshifts=parent['snapshot_redshift_'+side][:]
            ancestry['column_epochs']=[epoch_contract(1/(1+float(z))) for z in redshifts]
            ancestry['column_domain']='epoch-local FOF row; each column has its own saved epoch'
        namespace=namespaces[side] if side!='both' else 'paired-fof-row-union:'+canonical_digest(namespaces)
        identity_records[name]={'namespace':namespace,'ancestry':ancestry,'epochs':epochs}
        write_dataset_contract(mask,units='dimensionless',quantity='Availability of '+name,epoch=epochs,provenance=provenance)
    seal_hdf5_numeric_datasets(handle)
    datasets=[]
    handle.visititems(lambda name,item:datasets.append(name) if isinstance(item,h5py.Dataset) else None)
    for name in datasets:
        dataset=handle[name];leaf=name.rsplit('/',1)[-1]
        pair=name.split('/')[1] if name.startswith('pairs/') else None
        epochs=pair_epochs.get(pair)
        if name in identity_records:
            identity=identity_records[name]
            write_dataset_contract(dataset,units='identity',quantity='Exact epoch-local native FOF row association',
                identity_namespace=identity['namespace'],ancestry=identity['ancestry'],epoch=epochs,provenance=provenance,
                missing_policy='zero placeholder only where '+leaf+'_available is false')
            continue
        if leaf.endswith('_available') and 'quantity_definition' in dataset.attrs:continue
        unit=UNITS.get(leaf)
        write_dataset_contract(dataset,units=unit or 'UNDECLARED comparison quantity',
            quantity='compare_runs producer: '+name if unit else None,epoch=epochs,provenance=provenance,
            coordinates=coordinates if unit=='comoving Mpc/h' else None,
            missing_policy='NaN denotes undefined/insufficient measurements; categorical statuses retain their declared meaning')
        if epochs is None and 'redshift' in dataset.parent:
            redshift=dataset.parent['redshift']
            dataset.attrs['epoch_axis_reference']=redshift.name
            dataset.attrs['series_epoch_contract_json']=json.dumps([epoch_contract(1/(1+float(z))) for z in redshift[()]],allow_nan=False)
        if epochs is None and leaf in ('hmf_ratio_log10_ratio','pk_ratio_log10_ratio','redshift'):
            reference='hmf_ratio_redshift' if leaf=='hmf_ratio_log10_ratio' else 'pk_ratio_redshift' if leaf=='pk_ratio_log10_ratio' else dataset.name
            dataset.attrs['epoch_axis_reference']=reference


@directory_owner('destination')
def export_model_arrays(destination,arrays,specifications,*,algorithm,context=None,hdf5_arrays=()):
    """Save exact supplied graph inputs; never infer a physical epoch or source."""
    destination=Path(destination).resolve();destination.parent.mkdir(parents=True,exist_ok=True)
    final=destination.with_name(destination.name+'.data')
    if final.exists():raise FileExistsError('Model graph inputs already exist: '+str(final))
    stage=Path(tempfile.mkdtemp(prefix='.'+destination.name+'-data-',dir=destination.parent))
    context=context or {}
    try:
        from core.numeric import NumericArraySpool
        if isinstance(arrays,NumericArraySpool):arrays.write_npz(stage/'arrays.npz')
        else:np.savez_compressed(stage/'arrays.npz',**arrays)
        contract=numpy_archive_contract(stage/'arrays.npz',specifications,epoch=context.get('epoch'),
            coordinates=context.get('coordinates'),provenance={**context,'algorithm_sha256':digest(algorithm)})
        if hdf5_arrays:
            names = list(hdf5_arrays)
            if len(set(names)) != len(names) or not set(names) <= set(arrays):
                raise ValueError('HDF5 mirror must select distinct published numeric arrays')
            with h5py.File(stage/'arrays.hdf5', 'w') as target:
                for name in names:
                    array = np.asarray(arrays[name])
                    if array.dtype.kind not in 'biufc':
                        raise ValueError('HDF5 scientific mirror requires numeric arrays')
                    dataset = target.create_dataset(name, data=array,
                        **({'chunks':True, 'fletcher32':True} if array.ndim and array.size else {}))
                    spec = dict(specifications[name]); spec.pop('annotations', None)
                    spec.setdefault('epoch', context.get('epoch'))
                    spec.setdefault('coordinates', context.get('coordinates'))
                    spec.setdefault('provenance', {**context, 'algorithm_sha256':digest(algorithm)})
                    write_dataset_contract(dataset, **spec)
            contract['hdf5_mirror'] = {'path':'arrays.hdf5', 'sha256':digest(stage/'arrays.hdf5'),
                                      'datasets':names}
        atomic_json(stage/'contract.json',contract)
        (stage/'contract.sha256').write_text(digest(stage/'contract.json')+'\n')
        publish_directory(stage,final)
        return {'directory':str(final),'array_contract':contract,'science_acceptance':'NOT_ACCEPTED'}
    except BaseException as exc:
        if stage.exists():atomic_json(stage/'failure.json',{'status':'FAIL','science_acceptance':'NOT_ACCEPTED',
            'failure':{'type':type(exc).__name__,'message':str(exc)}})
        raise


def verify_model_arrays(directory):
    """Read a sealed model bundle and verify its exact numeric payload."""
    import json
    root=Path(directory).resolve()
    if digest(root/'contract.json')!=(root/'contract.sha256').read_text().strip():
        raise ValueError('Model array contract checksum mismatch')
    contract=json.loads((root/'contract.json').read_text())
    verify_numpy_archive_contract(root/'arrays.npz',contract)
    if 'hdf5_mirror' in contract:
        mirror = contract['hdf5_mirror']
        if mirror.get('path') != 'arrays.hdf5' or digest(root/'arrays.hdf5') != mirror.get('sha256'):
            raise ValueError('Model HDF5 mirror checksum mismatch')
    return contract


@directory_owner('destination')
def export_numeric_audit(destination, report, arrays, specifications, *, algorithm, context, source_hashes,hdf5_arrays=()):
    """Publish immutable measured audit arrays and a sealed human-readable report."""
    from core.foundation import atomic_text
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError('Keep previous numeric audit: ' + str(destination))
    for path, expected in source_hashes.items():
        if digest(path) != expected:
            raise RuntimeError('Numeric audit source changed: ' + str(path))
    inputs = export_model_arrays(destination, arrays, specifications, algorithm=algorithm,
                                context={**context, 'sources': source_hashes},hdf5_arrays=hdf5_arrays)
    inputs = {**inputs, 'directory': Path(inputs['directory']).name}
    result = {**report, 'numeric_data': inputs, 'science_acceptance': 'NOT_ACCEPTED'}
    for path, expected in source_hashes.items():
        if digest(path) != expected:
            raise RuntimeError('Numeric audit source changed before report publication: ' + str(path))
    atomic_json(destination, result)
    atomic_text(destination.with_name(destination.name + '.sha256'), digest(destination) + '\n')
    return result


def verify_numeric_audit(path):
    """Reject incomplete report seals or changed numeric measurements."""
    import json
    path = Path(path).resolve()
    before = digest(path)
    if before != path.with_name(path.name + '.sha256').read_text().strip():
        raise ValueError('Numeric audit report checksum mismatch')
    report = json.loads(path.read_text())
    expected = path.with_name(path.name + '.data')
    declared = Path(report['numeric_data']['directory'])
    if not declared.is_absolute(): declared = path.parent / declared
    if declared.resolve() != expected or expected.is_symlink():
        raise ValueError('Numeric audit data must be its immutable sibling')
    contract = verify_model_arrays(expected)
    if contract != report['numeric_data']['array_contract']:
        raise ValueError('Numeric audit report/data contract mismatch')
    if digest(path) != before:
        raise RuntimeError('Numeric audit report changed while reading')
    return report


@directory_owner('output')
def export_image_product(output, image, report, arrays, specifications, *, algorithm, context, source_hashes):
    """Publish actual numeric image inputs and pixels as one immutable generation."""
    from core.foundation import atomic_text, atomic_copy
    from blender import verify_image_publication, recover_image_publication
    output = Path(output).resolve()
    pointer = output.with_name(output.name + '.publication.json')
    archive = output.with_name(output.name + '.product')
    metadata_name = output.with_suffix('.json').name
    if any(p.exists() for p in (output, output.with_suffix('.json'), pointer, archive)):
        raise FileExistsError('Keep image generation evidence; choose a fresh output')
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='.image-product-', dir=output.parent))
    try:
        for source, expected in source_hashes.items():
            if digest(source) != expected: raise ValueError('Image source changed before publication')
        image.save(stage / output.name)
        bundle = export_model_arrays(stage / output.name, arrays, specifications,
            algorithm=algorithm, context={**context, 'sources': source_hashes})
        verify_model_arrays(bundle['directory'])
        record = {**report, 'schema': 'scientific-image-product-v1',
            'publication_contract': 'scientific-image-publication-v1', 'science_acceptance': 'NOT_ACCEPTED',
            'output_image': str(output), 'output_sha256': digest(stage / output.name),
            'numeric_data': {'directory': Path(bundle['directory']).name, 'array_contract': bundle['array_contract']}}
        record['artifacts'] = [{'path': p.relative_to(stage).as_posix(), 'sha256': digest(p)}
            for p in sorted(stage.rglob('*')) if p.is_file() and not any(v.startswith('.') for v in p.relative_to(stage).parts)]
        atomic_json(stage / metadata_name, record)
        atomic_text(stage / (metadata_name + '.sha256'), digest(stage / metadata_name) + '\n')
        for source, expected in source_hashes.items():
            if digest(source) != expected: raise ValueError('Image source changed during publication')
        publish_directory(stage, archive)
        pending = {'schema': 'scientific-image-publication-v1', 'state': 'PENDING',
            'archive': archive.name, 'metadata': metadata_name,
            'metadata_sha256': digest(archive / metadata_name),
            'image': output.name, 'science_acceptance': 'NOT_ACCEPTED'}
        atomic_json(pointer, pending)
        recover_image_publication(output, archive)
        return record
    except BaseException as exc:
        failure = stage / 'failure.json' if stage.exists() else archive.with_name(archive.name + '.failure.json')
        atomic_json(failure, {'status': 'FAIL', 'science_acceptance': 'NOT_ACCEPTED',
            'failure': {'type': type(exc).__name__, 'message': str(exc)}})
        raise


def numeric_audit_arrays_path(report_path, report):
    """Resolve the archive declared by a sealed numeric audit, including relocation."""
    root=Path(report_path).expanduser().resolve()
    entry=report.get('numeric_data',{}).get('directory')
    if not isinstance(entry,str) or not entry:raise ValueError('Numeric audit has no archive directory contract')
    directory=Path(entry)
    if not directory.is_absolute():
        if directory.name!=entry or entry in {'.','..'}:raise ValueError('Numeric audit directory must be a sibling bundle name')
        directory=root.parent/directory
    archive=directory/'arrays.npz'
    if not archive.is_file():raise FileNotFoundError('Numeric audit archive is unavailable: '+str(archive))
    return archive.resolve()


def numeric_audit_source_bindings(report):
    """Read sources from actual per-dataset provenance_json contracts."""
    sources = dict(report.get('source_hashes',{}))
    contract = report['numeric_data']['array_contract']
    for dataset in contract['datasets'].values():
        provenance = json.loads(dataset['provenance_json']) if 'provenance_json' in dataset else {}
        for path,checksum in provenance.get('sources',{}).items():
            if path in sources and sources[path] != checksum:
                raise ValueError('Contradictory numeric dataset source bindings')
            sources[path] = checksum
    if not sources:
        raise ValueError('Retained scientific numeric product has no exact source provenance')
    for path,expected in sources.items():
        if digest(path) != expected:
            raise ValueError('Numeric scientific product source changed')
    return sources
