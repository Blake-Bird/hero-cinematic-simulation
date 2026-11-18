/* Bounded synchronized reference operator for numerical SIDM controls.
 * This synchronized single-rank all-pair implementation is a reference backend,
 * not a scalable cosmological neighbor-exchange implementation. Its supported
 * domain is explicit and enforced before any source update. */
#include "gadgetconfig.h"
#ifdef RESEARCH_SIDM_REFERENCE
#if !defined(FORCE_EQUAL_TIMESTEPS) || defined(HIERARCHICAL_GRAVITY) || defined(LEAN)
#error "RESEARCH_SIDM_REFERENCE requires synchronized full particle state"
#endif
#include "../main/simulation.h"
#include "research_rare_pair.h"
#include "research_kernel_overlap.h"
#include "research_rare_substeps.h"
#include "research_frequent_pair.h"
#include "research_hybrid_pair.h"
#include "research_physical_clock.h"
#ifdef RESEARCH_SIDM_CANONICAL_CONTROL
#include "research_canonical_control.h"
#endif
#include "../time_integration/driftfac.h"
#include <memory>
#include <algorithm>
#include <cerrno>
#include <climits>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {
std::uint64_t exact_integer(const char *text)
{
  if(!text || !*text)Terminate("SIDM exact integer is missing");
  for(const char *p=text;*p;p++)if(*p<'0' || *p>'9')Terminate("SIDM exact integer is not unsigned decimal");
  errno=0;char *end=nullptr;unsigned long long value=std::strtoull(text,&end,10);
  if(errno || !end || *end)Terminate("SIDM exact integer is outside uint64 range");
  return std::uint64_t(value);
}
}
void sim::sidm_reference_source_step(void)
{
  try {research_sidm::prepare_native_differential_model(All.SIDMReferenceAngularKind,All.SIDMNativeTablePath,All.SIDMAngularModelHash,All.SIDMNativeTableSHA256);}
  catch(const std::exception &error){Terminate("SIDM model preparation failure: %s",error.what());}
  if(All.SIDMReferenceAngularKind==2 && All.SIDMReferenceParticleConvention!=1)
    Terminate("Born Moller requires folded identical-particle convention");
  if(All.SIDMEventAuditModulus<0 || All.SIDMEventAuditResidue<0 || (All.SIDMEventAuditModulus==0 && All.SIDMEventAuditResidue!=0) || (All.SIDMEventAuditModulus>0 && All.SIDMEventAuditResidue>=All.SIDMEventAuditModulus))Terminate("SIDM audit population parameters invalid");
  if(NTask!=1 || Sp.NumGas!=0)
    Terminate("SIDM reference backend requires one declared rank and DM-only particles");
  if(!std::isfinite(All.HubbleParam) || All.HubbleParam<=0)
    Terminate("SIDM physical unit conversions require the actual positive native Hubble parameter");
  if(All.SIDMReferenceMaximumParticles<=0 || Sp.NumPart>All.SIDMReferenceMaximumParticles ||
      All.SIDMReferenceMaximumClockPanels<2 ||
      double(Sp.NumPart)*160+double(All.SIDMReferenceMaximumClockPanels)*128>All.MaxMemSize*1024.*1024.*.25)
    Terminate("SIDM reference particle/workspace allocation exceeded");
  if(All.SIDMReferenceAngularKind!=0 && All.SIDMReferenceAngularKind!=1 && All.SIDMReferenceAngularKind!=2 && All.SIDMReferenceAngularKind!=3)
    Terminate("SIDM reference angular model must be isotropic, screened Rutherford or Born Moller");
  if(All.SIDMReferenceParticleConvention!=0 && All.SIDMReferenceParticleConvention!=1)
    Terminate("SIDM native particle convention must be directed (0) or folded identical (1)");
  if(All.SIDMReferenceParticleConvention==1 && All.SIDMReferenceSplitAngleRadians>=1.5707963267948966192313216916398)
    Terminate("SIDM folded hybrid partition must be strictly below pi/2");
  if((All.SIDMReferenceKernelKind!=0 && All.SIDMReferenceKernelKind!=1) ||
      !std::isfinite(All.SIDMReferenceKernelOriginRelativeBound) ||
      All.SIDMReferenceKernelOriginRelativeBound<=0 || All.SIDMReferenceKernelOriginRelativeBound>=1)
    Terminate("SIDM reference requires an explicit kernel kind and finite origin approximation bound");
  const std::uint64_t maximum_pairs=exact_integer(All.SIDMReferenceMaximumPairVisits);
  const std::uint64_t possible=std::uint64_t(Sp.NumPart)*std::uint64_t(std::max(0,Sp.NumPart-1))/2;
  if(double(possible)*32>All.MaxMemSize*1024.*1024.*.125)Terminate("SIDM reference randomized pair ordering exceeds declared native workspace");
  if(possible>maximum_pairs)Terminate("SIDM reference all-pair work exceeds the explicit allocation");
  if(All.SIDMReferenceMaximumSubsteps<=0 || !std::isfinite(All.SIDMReferenceMaximumParticleProbability) ||
      All.SIDMReferenceMaximumParticleProbability<=0 || All.SIDMReferenceMaximumParticleProbability>=1)
    Terminate("SIDM reference needs explicit positive source-substep and particle probability limits");
  if((All.SIDMReferenceSchemeKind<0 || All.SIDMReferenceSchemeKind>2) ||
      !std::isfinite(All.SIDMReferenceMaximumTransferDepth) || All.SIDMReferenceMaximumTransferDepth<=0 ||
      All.SIDMReferenceMaximumTransferDepth>=1 || !std::isfinite(All.SIDMReferenceMaximumNeglectedMomentRatio) ||
      All.SIDMReferenceMaximumNeglectedMomentRatio<=0 || All.SIDMReferenceMaximumNeglectedMomentRatio>=2)
    Terminate("SIDM reference requires rare (0), directed frequent (1) or directed hybrid (2) scheme and explicit controls");
  if(!std::isfinite(All.SIDMReferenceSplitAngleRadians) || All.SIDMReferenceSplitAngleRadians<=0 ||
      All.SIDMReferenceSplitAngleRadians>=3.1415926535897932384626433832795)
    Terminate("SIDM reference requires an explicit split angle strictly between zero and pi");
  if(!std::isfinite(All.TimeStep) || All.TimeStep<=0 || !std::isfinite(All.UnitTime_in_s) || All.UnitTime_in_s<=0)
    Terminate("SIDM reference needs an actual positive physical elapsed interval");
  double radius=All.SIDMReferenceKernelRadiusProperKpc*3.0856775814913673e21;
  if(!std::isfinite(radius) || radius<=0)Terminate("SIDM reference kernel support invalid");
  double scale_factor=All.ComovingIntegrationOn?All.Time:1.;
  const double coordinate_scale=All.ComovingIntegrationOn?All.Time:1.;
#ifdef RESEARCH_SIDM_CANONICAL_CONTROL
  research_sidm::canonical_source_interval canonical_interval;
  std::unique_ptr<research_sidm::canonical_control_clock> canonical_clock;
  if(All.ComovingIntegrationOn || All.TimeBegin!=0 || All.SIDMReferenceBackendKind!=0 || All.ICFormat!=3 || All.SnapFormat!=3
      || All.SIDMCanonicalMaximumAssetBytes<128 || All.SIDMCanonicalMaximumRows<2
      || All.SIDMCanonicalMaximumClockWork<1)
    Terminate("Canonical SIDM requires static tau starting at zero, the one-rank all-pair backend and explicit asset/work bounds");
  if(double(All.SIDMCanonicalMaximumAssetBytes)+double(All.SIDMCanonicalMaximumRows)*16>All.MaxMemSize*1024.*1024.*.125)
    Terminate("Canonical SIDM clock asset/workspace exceeds native memory allocation");
  try {
    canonical_clock.reset(new research_sidm::canonical_control_clock(research_sidm::canonical_control_clock::read(
        All.SIDMCanonicalClockPath,All.SIDMCanonicalClockSHA256,All.SIDMAngularModelHash,
        std::size_t(All.SIDMCanonicalMaximumAssetBytes),std::size_t(All.SIDMCanonicalMaximumRows))));
    if(canonical_clock->initial_scale_factor()!=All.SIDMCanonicalInitialScaleFactor)
      Terminate("Canonical SIDM clock initial epoch differs from its native IC binding");
    canonical_interval=research_sidm::canonical_native_source_interval(*canonical_clock,All.TimeOld,All.Time,
        All.UnitTime_in_s,All.HubbleParam,All.SIDMReferenceMaximumScaleFactorFractionalChange,
        std::size_t(All.SIDMCanonicalMaximumClockWork),false);
    scale_factor=canonical_interval.frozen_scale_factor;
    radius=research_sidm::canonical_kernel_support_cm(radius,scale_factor);
  } catch(const std::exception &error){Terminate("Canonical SIDM source clock failure: %s",error.what());}
#endif
  if(!std::isfinite(scale_factor) || scale_factor<=0)
    Terminate("SIDM source scale factor invalid");
  if(All.ComovingIntegrationOn && (!std::isfinite(All.TimeOld) || All.TimeOld<=0 || All.Time<=All.TimeOld ||
      !std::isfinite(All.SIDMReferenceMaximumScaleFactorFractionalChange) || All.SIDMReferenceMaximumScaleFactorFractionalChange<=0 ||
      (All.Time-All.TimeOld)/All.TimeOld>All.SIDMReferenceMaximumScaleFactorFractionalChange))
    Terminate("SIDM frozen source scale-factor change exceeds the declared integration domain; refine the native gravity interval");
#ifdef PERIODIC
  const double proper_box=All.BoxSize*All.UnitLength_in_cm*coordinate_scale/All.HubbleParam;
  if(!std::isfinite(proper_box) || proper_box<=0 || 4*radius>=proper_box)
    Terminate("SIDM periodic convolution support must fit inside the unique nearest-image domain");
#endif
  if(All.SIDMReferenceMaximumEventsPerStep<=0 || std::strlen(All.SIDMAngularModelHash)!=64 || !*All.SIDMEventIdentityNamespace)
    Terminate("SIDM reference event allocation or physical model provenance missing");
  struct state {int native_index;research_sidm::pair_particle value;};
  std::vector<state> particles;
  particles.reserve(Sp.NumPart);
  const double velocity_unit=All.UnitVelocity_in_cm_per_s/(1e5*coordinate_scale);
  const double mass_unit=All.UnitMass_in_g/(All.HubbleParam*1.98847e33);
  if(!std::isfinite(velocity_unit) || velocity_unit<=0 || !std::isfinite(mass_unit) || mass_unit<=0 ||
      !std::isfinite(All.UnitLength_in_cm) || All.UnitLength_in_cm<=0)
    Terminate("SIDM native physical mass, velocity or length conversion invalid");
  for(int i=0;i<Sp.NumPart;i++) {
    if(Sp.P[i].getType()!=1 || Sp.P[i].getMass()<=0)
      Terminate("SIDM isolated reference requires a complete fine-DM-only native census");
    state row;row.native_index=i;row.value.id=Sp.P[i].ID.get();row.value.mass_msun=Sp.P[i].getMass()*mass_unit;
    for(int k=0;k<3;k++)row.value.velocity_kms[k]=Sp.P[i].Vel[k]*velocity_unit;
    particles.push_back(row);
  }
  std::sort(particles.begin(),particles.end(),[](const state &left,const state &right){return left.value.id<right.value.id;});
  for(std::size_t i=1;i<particles.size();i++)if(particles[i].value.id==particles[i-1].value.id)
    Terminate("SIDM reference exact native identities repeat");
  char path[1024];int written=std::snprintf(path,sizeof(path),"%s/sidm_events_%llu.hdf5",All.OutputDir,
      static_cast<unsigned long long>(All.Ti_Current));
  if(written<0 || std::size_t(written)>=sizeof(path))Terminate("SIDM reference event output path too long");
  try {
    research_sidm::physical_clock_result clock;
#ifdef RESEARCH_SIDM_CANONICAL_CONTROL
    clock.elapsed_seconds=canonical_interval.canonical_dt_seconds;
    clock.estimated_absolute_error_seconds=canonical_interval.maximum_clock_absolute_error_estimate_seconds;
#else
    if(All.ComovingIntegrationOn) {
      clock=research_sidm::cosmological_elapsed_seconds(All.TimeOld,All.Time,All.UnitTime_in_s,All.HubbleParam,
          [](double a){return driftfac::hubble_function(a);},All.SIDMReferenceClockAbsoluteToleranceSeconds,
          All.SIDMReferenceClockRelativeTolerance,std::size_t(All.SIDMReferenceMaximumClockPanels));
    } else clock.elapsed_seconds=All.TimeStep*All.UnitTime_in_s/All.HubbleParam;
#endif
    research_sidm_event_log log(path,All.SIDMAngularModelHash,All.SIDMEventIdentityNamespace,All.HubbleParam,
        std::uint64_t(All.SIDMReferenceMaximumEventsPerStep),
        All.SIDMReferenceSchemeKind==1?"frequent_drag_diffusion_pair":"rare_elastic_pair");
    log.configure_audit(std::uint64_t(All.SIDMEventAuditModulus),std::uint64_t(All.SIDMEventAuditResidue));
    log.declare_angular_model(All.SIDMReferenceAngularKind==1 || All.SIDMReferenceAngularKind==2,All.SIDMReferenceScreeningVelocityKmPerS,All.SIDMReferenceAngularKind==2,All.SIDMReferenceAngularKind==3,All.SIDMNativeTableSHA256);
    log.declare_particle_convention(All.SIDMReferenceParticleConvention==1);
    const auto declare_background=[&](research_sidm_event_log &output) {
#ifdef RESEARCH_SIDM_CANONICAL_CONTROL
      output.declare_canonical_background(All.TimeOld,All.Time,canonical_interval.begin_tau_seconds,canonical_interval.end_tau_seconds,
          canonical_interval.begin_scale_factor,canonical_interval.end_scale_factor,scale_factor,
          canonical_interval.maximum_clock_absolute_error_estimate_seconds,canonical_interval.maximum_inverse_bracket_relative_width,
          All.UnitMass_in_g,All.UnitLength_in_cm,All.UnitTime_in_s,All.HubbleParam,
          canonical_clock->asset_sha256(),canonical_clock->background_sha256(),radius);
      output.declare_canonical_substep_epochs();
#else
      output.declare_native_background(All.ComovingIntegrationOn,All.TimeOld,All.Time,clock.elapsed_seconds,
          clock.estimated_absolute_error_seconds,All.UnitMass_in_g,All.UnitLength_in_cm,All.UnitTime_in_s,All.HubbleParam);
#endif
    };
    declare_background(log);
    if(All.SIDMReferenceSchemeKind==1)log.declare_frequent_controls(All.SIDMReferenceMaximumTransferDepth,
        All.SIDMReferenceMaximumNeglectedMomentRatio);
    std::unique_ptr<research_sidm_event_log> diffusion_log;
    if(All.SIDMReferenceSchemeKind==2) {
      char diffusion_path[1024];
      const int size=std::snprintf(diffusion_path,sizeof(diffusion_path),"%s/sidm_diffusion_%llu.hdf5",All.OutputDir,
          static_cast<unsigned long long>(All.Ti_Current));
      if(size<0 || std::size_t(size)>=sizeof(diffusion_path))Terminate("SIDM hybrid diffusion output path too long");
      diffusion_log.reset(new research_sidm_event_log(diffusion_path,All.SIDMAngularModelHash,
          All.SIDMEventIdentityNamespace,All.HubbleParam,std::uint64_t(All.SIDMReferenceMaximumEventsPerStep),
          "frequent_drag_diffusion_pair"));
      diffusion_log->configure_audit(std::uint64_t(All.SIDMEventAuditModulus),std::uint64_t(All.SIDMEventAuditResidue));
      diffusion_log->declare_angular_model(All.SIDMReferenceAngularKind==1 || All.SIDMReferenceAngularKind==2,All.SIDMReferenceScreeningVelocityKmPerS,All.SIDMReferenceAngularKind==2,All.SIDMReferenceAngularKind==3,All.SIDMNativeTableSHA256);
      diffusion_log->declare_particle_convention(All.SIDMReferenceParticleConvention==1);
      declare_background(*diffusion_log);
      diffusion_log->declare_frequent_controls(All.SIDMReferenceMaximumTransferDepth,All.SIDMReferenceMaximumNeglectedMomentRatio);
      log.declare_hybrid_split(All.SIDMReferenceSplitAngleRadians);
      diffusion_log->declare_hybrid_split(All.SIDMReferenceSplitAngleRadians);
    }
    const auto overlap_for_pair=[&](std::size_t left,std::size_t right) {
      double delta[3];
      Sp.nearest_image_intpos_to_pos(Sp.P[particles[left].native_index].IntPos,
          Sp.P[particles[right].native_index].IntPos,delta);
      const double distance=std::hypot(delta[0],std::hypot(delta[1],delta[2]))*All.UnitLength_in_cm*coordinate_scale/All.HubbleParam;
      return research_sidm::measured_kernel_overlap(All.SIDMReferenceKernelKind,distance,radius,radius,
          All.SIDMReferenceKernelOriginRelativeBound);
    };
    const double interval_seconds=clock.elapsed_seconds;
    if(!std::isfinite(interval_seconds) || interval_seconds<=0)
      Terminate("SIDM reference physical source interval overflow");
    double elapsed=0;
    double smallest_step=interval_seconds;
    double largest_step=0;
    std::uint64_t visits=0;
    int substeps=0;
    research_sidm::source_resource_measurements resource_measurements;
    std::uint64_t completed_pair_operations=0;
#ifdef RESEARCH_SIDM_CANONICAL_CONTROL
    double source_begin_scale=canonical_interval.begin_scale_factor,source_end_scale=source_begin_scale;
    double largest_scale_span=0,substep_clock_error=canonical_interval.maximum_clock_absolute_error_estimate_seconds;
    double substep_inverse_bracket=canonical_interval.maximum_inverse_bracket_relative_width;
    std::size_t source_clock_work=canonical_interval.function_evaluations;
    double source_begin_bracket=canonical_interval.begin_inverse_bracket_relative_width;
#endif
    while(elapsed<interval_seconds) {
      if(substeps>=All.SIDMReferenceMaximumSubsteps)
        Terminate("SIDM reference source-substep allocation exhausted before the full interval");
      // Any directed transfer moment is <=2. Bounding full optical depth by
      // half the requested transfer depth therefore bounds the drag increment
      // even before the current angular moments have been evaluated.
      const double planning_pair_probability=All.SIDMReferenceSchemeKind!=0?
          std::min(All.SIDMReferenceMaximumEventProbability,-std::expm1(-All.SIDMReferenceMaximumTransferDepth/2)):
          All.SIDMReferenceMaximumEventProbability;
#ifdef RESEARCH_SIDM_CANONICAL_CONTROL
      scale_factor=source_begin_scale;
      radius=research_sidm::canonical_kernel_support_cm(All.SIDMReferenceKernelRadiusProperKpc*3.0856775814913673e21,scale_factor);
      double expansion_seconds=interval_seconds-elapsed;
      research_sidm::canonical_expansion_cap expansion_cap;
      if(source_clock_work>=std::size_t(All.SIDMCanonicalMaximumClockWork))Terminate("Canonical expansion source clock allocation exhausted");
      try {
        expansion_cap=research_sidm::limit_canonical_expansion_step(*canonical_clock,canonical_interval.begin_tau_seconds+elapsed,
            canonical_interval.end_tau_seconds,source_begin_scale,source_begin_bracket,All.SIDMReferenceMaximumScaleFactorFractionalChange,
            std::size_t(All.SIDMCanonicalMaximumClockWork)-source_clock_work);
        source_clock_work+=expansion_cap.function_evaluations;expansion_seconds=std::min(expansion_seconds,expansion_cap.seconds);
        substep_clock_error=std::max(substep_clock_error,expansion_cap.maximum_clock_error_seconds);
        substep_inverse_bracket=std::max(substep_inverse_bracket,expansion_cap.maximum_inverse_bracket_relative_width);
      } catch(const std::exception&error){Terminate("Canonical adaptive expansion-step failure: %s",error.what());}
      const auto plan=research_sidm::plan_canonical_substep(particles,overlap_for_pair,expansion_seconds,
          All.SIDMReferenceCrossSectionNormalizationCm2PerG,All.SIDMReferenceAngularKind,
          All.SIDMReferenceScreeningVelocityKmPerS,scale_factor,planning_pair_probability,
          All.SIDMReferenceMaximumParticleProbability,All.SIDMReferenceConservationTolerance,
          All.SIDMReferenceEnergyFloor,maximum_pairs-visits,All.SIDMReferenceSchemeKind==2?3:1);
#else
      const auto plan=research_sidm::plan_rare_substep(particles,overlap_for_pair,interval_seconds-elapsed,
          All.SIDMReferenceCrossSectionNormalizationCm2PerG,All.SIDMReferenceAngularKind,
          All.SIDMReferenceScreeningVelocityKmPerS,planning_pair_probability,
          All.SIDMReferenceMaximumParticleProbability,All.SIDMReferenceConservationTolerance,
          All.SIDMReferenceEnergyFloor,maximum_pairs-visits,All.SIDMReferenceSchemeKind==2?3:1);
#endif
      visits+=plan.inspected_pairs;
      const double step_seconds=plan.physical_dt_seconds;
      smallest_step=std::min(smallest_step,step_seconds);
      largest_step=std::max(largest_step,step_seconds);
      const double next_elapsed=step_seconds>=interval_seconds-elapsed?interval_seconds:elapsed+step_seconds;
      if(next_elapsed<=elapsed || next_elapsed>interval_seconds)
        Terminate("SIDM source substep cannot advance its physical clock at declared precision");
      double event_native_time=All.Time;
#ifdef RESEARCH_SIDM_CANONICAL_CONTROL
      const double endpoint_tau=next_elapsed==interval_seconds?canonical_interval.end_tau_seconds:canonical_interval.begin_tau_seconds+next_elapsed;
      const double start_tau=canonical_interval.begin_tau_seconds+elapsed;
      double source_end_bracket=canonical_interval.end_inverse_bracket_relative_width;
      if(!std::isfinite(endpoint_tau) || endpoint_tau<=start_tau || endpoint_tau>canonical_interval.end_tau_seconds)
        Terminate("Canonical source substep loses its elapsed-clock endpoint precision");
      if(next_elapsed==interval_seconds)source_end_scale=canonical_interval.end_scale_factor;
      else if(endpoint_tau==expansion_cap.end_tau_seconds) {
        source_end_scale=expansion_cap.end_scale_factor;source_end_bracket=expansion_cap.end_inverse_bracket_relative_width;
      }
      else {
        if(source_clock_work>=std::size_t(All.SIDMCanonicalMaximumClockWork))Terminate("Canonical source aggregate substep clock work exhausted");
        try {
          const auto endpoint=canonical_clock->epoch(endpoint_tau,std::size_t(All.SIDMCanonicalMaximumClockWork)-source_clock_work);
          if(endpoint.function_evaluations>std::size_t(All.SIDMCanonicalMaximumClockWork)-source_clock_work)
            Terminate("Canonical substep inverse-clock accounting exceeds its aggregate bound");
          source_clock_work+=endpoint.function_evaluations;source_end_scale=endpoint.scale_factor;
          source_end_bracket=endpoint.bracket_relative_width;
          substep_clock_error=std::max(substep_clock_error,endpoint.clock_absolute_error_estimate_seconds);
          substep_inverse_bracket=std::max(substep_inverse_bracket,endpoint.bracket_relative_width);
        } catch(const std::exception &error){Terminate("Canonical source substep inverse-clock failure: %s",error.what());}
      }
      if(source_end_scale<source_begin_scale || source_end_scale>canonical_interval.end_scale_factor)
        Terminate("Canonical source substep epochs regress or leave the actual source interval");
      const double source_span=std::log(source_end_scale)-std::log(source_begin_scale)+std::log1p(source_begin_bracket)+std::log1p(source_end_bracket);
      if(!std::isfinite(source_span)||source_span<0||source_span>std::log1p(All.SIDMReferenceMaximumScaleFactorFractionalChange))
        Terminate("Canonical selected source substep exceeds its expansion/inverse-bracket allocation");
      largest_scale_span=std::max(largest_scale_span,std::expm1(source_span));
      event_native_time=next_elapsed==interval_seconds?All.Time:endpoint_tau*All.HubbleParam/All.UnitTime_in_s;
      if(!std::isfinite(event_native_time) || event_native_time<=All.TimeOld || event_native_time>All.Time)
        Terminate("Canonical substep native event endpoint leaves the original source interval");
#endif
      std::vector<long double> actual_particle_depth(particles.size(),0);
      const long double particle_depth_limit=-std::log1p(-static_cast<long double>(All.SIDMReferenceMaximumParticleProbability));
      std::vector<std::pair<std::size_t,std::size_t>> ordered_pairs;
      ordered_pairs.reserve(particles.size()*(particles.size()?particles.size()-1:0)/2);
      for(std::size_t left=0;left<particles.size();left++)for(std::size_t right=left+1;right<particles.size();right++)ordered_pairs.push_back({left,right});
      const auto pair_priority=[&](const std::pair<std::size_t,std::size_t> &pair){return research_sidm::pair_random_bits(exact_integer(All.SIDMOperatorSeed),particles[pair.first].value.id,particles[pair.second].value.id,std::uint64_t(All.Ti_Current),All.SIDMReferenceSubstepCount,std::uint64_t(research_sidm::draw_purpose::pair_priority));};
      std::sort(ordered_pairs.begin(),ordered_pairs.end(),[&](const std::pair<std::size_t,std::size_t> &left,const std::pair<std::size_t,std::size_t> &right){
        const auto lp=pair_priority(left),rp=pair_priority(right);return lp<rp || (lp==rp && left<right);});
      for(const auto &pair:ordered_pairs) {
      const auto left=pair.first,right=pair.second;
      if(visits>=maximum_pairs)Terminate("SIDM reference source application work allocation exhausted");
      ++visits;
      const double overlap=overlap_for_pair(left,right);
      if(overlap==0)continue;
      research_sidm::vector3 relative;
      for(int k=0;k<3;k++)relative[k]=particles[left].value.velocity_kms[k]-particles[right].value.velocity_kms[k];
      const double speed=research_sidm::norm(relative);
      double amplitude=0;
#ifndef RESEARCH_SIDM_CANONICAL_CONTROL
      amplitude=speed==0?0:research_sidm::native_total_cross_section(All.SIDMReferenceAngularKind,All.SIDMReferenceCrossSectionNormalizationCm2PerG,speed,All.SIDMReferenceScreeningVelocityKmPerS);
#endif
      // Explicit fields retain the original physical values under C++11,
      // where default member initializers make this type non-aggregate.
      research_sidm::rare_parameters parameters{};
      parameters.operator_seed=exact_integer(All.SIDMOperatorSeed);
      parameters.cross_section_per_mass_cm2_g=amplitude;
      parameters.kernel_overlap_inverse_cm3=overlap;
      parameters.physical_dt_seconds=step_seconds;
      parameters.maximum_event_probability=All.SIDMReferenceMaximumEventProbability;
      parameters.macro_mass_relative_tolerance=All.SIDMReferenceMacroMassTolerance;
      parameters.conservation_relative_tolerance=All.SIDMReferenceConservationTolerance;
      parameters.energy_absolute_floor=All.SIDMReferenceEnergyFloor;
      parameters.momentum_absolute_floor=All.SIDMReferenceMomentumFloor;
      parameters.screened_rutherford=All.SIDMReferenceAngularKind==1 || All.SIDMReferenceAngularKind==2;
      parameters.screening_velocity_kms=All.SIDMReferenceScreeningVelocityKmPerS;
      parameters.cdf_absolute_tolerance=All.SIDMReferenceCDFTolerance;
      parameters.native_velocity_unit_kms=velocity_unit;
      parameters.native_velocity_precision_bits=8*sizeof(MyFloat);
      parameters.folded_identical_particle=All.SIDMReferenceParticleConvention==1;
      parameters.born_moller=All.SIDMReferenceAngularKind==2;
      parameters.resources=&resource_measurements;
      parameters.differential_model=research_sidm::native_differential_model();
      parameters.differential_normalization=All.SIDMReferenceCrossSectionNormalizationCm2PerG;
      parameters.random_substep=All.SIDMReferenceSubstepCount;
#ifdef RESEARCH_SIDM_CANONICAL_CONTROL
      // The adapter expects a physical-template rounding unit. Canonical
      // storage rounds u with velocity_unit, so v would round with unit/a.
      parameters.native_velocity_unit_kms=velocity_unit/scale_factor;
      parameters=research_sidm::canonical_native_pair_parameters(parameters,All.SIDMReferenceAngularKind,
          scale_factor,speed,step_seconds,overlap);
      amplitude=parameters.cross_section_per_mass_cm2_g;
#endif
      const long double actual_depth=static_cast<long double>(amplitude)*particles[left].value.mass_msun*1.98847e33L*
          speed*1.e5L*overlap*step_seconds;
      if(!std::isfinite(actual_depth) || actual_depth<0)
        Terminate("SIDM actual source pair depth outside finite physical domain");
      actual_particle_depth[left]+=actual_depth;actual_particle_depth[right]+=actual_depth;
      if(actual_particle_depth[left]>particle_depth_limit || actual_particle_depth[right]>particle_depth_limit)
        Terminate("SIDM actual particle hazard exceeds its pre-draw substep bound");
      if(completed_pair_operations==UINT64_MAX)Terminate("SIDM completed pair operation count exhausted");
      ++completed_pair_operations;
      research_sidm::source_cpu_scope pair_resource_scope(&resource_measurements,0);
      research_sidm::pair_result result;
      bool rare_scattered=false;
      unsigned frequent_updates=0;
      if(All.SIDMReferenceSchemeKind==0) {
        result=research_sidm::apply_rare_pair(particles[left].value,particles[right].value,parameters,
            std::uint64_t(All.Ti_Current),All.SIDMReferenceAttemptCount,event_native_time,scale_factor,0,All.SIDMReferenceEventCount,&log);
        rare_scattered=result.scattered;
      } else if(All.SIDMReferenceSchemeKind==1) {
        result=research_sidm::apply_frequent_pair(particles[left].value,particles[right].value,parameters,
              All.SIDMReferenceParticleConvention==1?1.5707963267948966192313216916398:3.1415926535897932384626433832795,
              All.SIDMReferenceMaximumTransferDepth,
              All.SIDMReferenceMaximumNeglectedMomentRatio,std::uint64_t(All.Ti_Current),
              All.SIDMReferenceAttemptCount,event_native_time,scale_factor,0,All.SIDMReferenceEventCount,&log);
        frequent_updates=unsigned(result.scattered);
      } else {
        const auto hybrid=research_sidm::apply_hybrid_pair(particles[left].value,particles[right].value,parameters,
            All.SIDMReferenceSplitAngleRadians,All.SIDMReferenceMaximumTransferDepth,All.SIDMReferenceMaximumNeglectedMomentRatio,
            std::uint64_t(All.Ti_Current),All.SIDMReferenceAttemptCount,event_native_time,scale_factor,0,All.SIDMReferenceEventCount,
            &log,diffusion_log.get());
        result=hybrid.aggregate;rare_scattered=hybrid.rare_scattered;frequent_updates=hybrid.frequent_updates;
      }
      All.SIDMReferenceOpticalDepthSum+=result.optical_depth;
      if(!std::isfinite(All.SIDMReferenceOpticalDepthSum))Terminate("SIDM reference exposure accumulator overflow");
      for(std::size_t index:{left,right}) {
        auto &source=Sp.P[particles[index].native_index].ResearchSourceState;
        source.sidm_optical_depth+=result.optical_depth;
        source.sidm_random_counter=All.SIDMReferenceSubstepCount;
        if(!std::isfinite(source.sidm_optical_depth))Terminate("SIDM native particle exposure accumulator overflow");
      }
      if(result.scattered) {
        if(frequent_updates) {
          if(frequent_updates>std::numeric_limits<std::uint64_t>::max()-All.SIDMReferenceFrequentUpdateCount)
            Terminate("SIDM frequent source update counter exhausted");
          All.SIDMReferenceFrequentUpdateCount+=frequent_updates;
        }
        for(std::size_t index:{left,right}) {
          auto &native=Sp.P[particles[index].native_index];
          for(int k=0;k<3;k++)native.Vel[k]=particles[index].value.velocity_kms[k]/velocity_unit;
          native.ResearchSourceState.sidm_last_update_time=All.Time;
          if(rare_scattered) {
            if(native.ResearchSourceState.sidm_collision_count==std::numeric_limits<std::uint64_t>::max())Terminate("SIDM particle collision counter exhausted");
            native.ResearchSourceState.sidm_collision_count++;
          }
          native.ResearchSourceState.sidm_random_counter=All.SIDMReferenceSubstepCount;
        }
      }
    }
      elapsed=next_elapsed;
#ifdef RESEARCH_SIDM_CANONICAL_CONTROL
      source_begin_scale=source_end_scale;
      source_begin_bracket=source_end_bracket;
#endif
      ++substeps;
      if(All.SIDMReferenceSubstepCount==std::numeric_limits<std::uint64_t>::max())
        Terminate("SIDM persistent source-substep counter exhausted");
      ++All.SIDMReferenceSubstepCount;
      if(All.SIDMReferenceSmallestSourceStepSeconds==0 || step_seconds<All.SIDMReferenceSmallestSourceStepSeconds)
        All.SIDMReferenceSmallestSourceStepSeconds=step_seconds;
    }
    if(visits>std::numeric_limits<std::uint64_t>::max()-All.SIDMReferencePairVisitCount)
      Terminate("SIDM persistent source-work counter exhausted");
    All.SIDMReferencePairVisitCount+=visits;
    log.record_source_interval(interval_seconds,std::uint64_t(substeps),visits,smallest_step,
        All.SIDMReferenceMaximumEventProbability,All.SIDMReferenceMaximumParticleProbability,
        All.SIDMReferenceKernelKind,All.SIDMReferenceKernelRadiusProperKpc);
    if(diffusion_log)diffusion_log->record_source_interval(interval_seconds,std::uint64_t(substeps),visits,smallest_step,
        All.SIDMReferenceMaximumEventProbability,All.SIDMReferenceMaximumParticleProbability,
        All.SIDMReferenceKernelKind,All.SIDMReferenceKernelRadiusProperKpc);
    log.record_largest_source_step(largest_step);
    if(diffusion_log)diffusion_log->record_largest_source_step(largest_step);
#ifdef RESEARCH_SIDM_CANONICAL_CONTROL
    log.record_canonical_substep_clock(source_clock_work,largest_scale_span,substep_clock_error,substep_inverse_bracket);
    if(diffusion_log)diffusion_log->record_canonical_substep_clock(source_clock_work,largest_scale_span,substep_clock_error,substep_inverse_bracket);
#endif
    std::uint64_t measured_peak_rss=0;
    const bool rss_available=research_sidm::process_peak_rss_bytes(measured_peak_rss);
    const auto record_resources=[&](research_sidm_event_log &output) {
      output.record_distributed_exchange(0,1,particles.size()>1?std::uint64_t(particles.size())*(particles.size()-1)/2:0,0,0,0,0,0,
          completed_pair_operations,true);
      output.record_resource_measurements(resource_measurements.cpu_available,resource_measurements.pair_cpu_seconds,
          resource_measurements.rare_cpu_seconds,resource_measurements.frequent_cpu_seconds,
          resource_measurements.rare_calls,resource_measurements.frequent_calls,rss_available,measured_peak_rss,0,research_sidm::native_differential_model()?research_sidm::native_differential_model()->memory_bytes():0);
    };
    record_resources(log);if(diffusion_log)record_resources(*diffusion_log);
    log.finish();
    if(diffusion_log)diffusion_log->finish();
  } catch(const std::exception &error) {Terminate("SIDM reference source failure: %s",error.what());}
}
#endif
