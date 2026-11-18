/* Native synchronized distributed SIDM source using spatial ghosts and
 * conflict-free owner commits. Gravity/source and angular approximation errors
 * remain subject to the independent controls in the physical run request. */
#include "gadgetconfig.h"
#ifdef RESEARCH_SIDM_REFERENCE
#if !defined(FORCE_EQUAL_TIMESTEPS) || defined(HIERARCHICAL_GRAVITY) || defined(LEAN)
#error "Distributed SIDM source requires synchronized full particle state"
#endif
#if defined(LONG_X) || defined(LONG_Y) || defined(LONG_Z)
#error "Distributed SIDM spatial source currently requires a cubic periodic geometry"
#endif
#include "../main/simulation.h"
#include "../time_integration/driftfac.h"
#include "research_spatial_pairs.h"
#include "research_distributed_rates.h"
#include "research_hybrid_pair.h"
#include "research_physical_clock.h"
#include "research_coarse_guard.h"
#include <cerrno>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <memory>

namespace {
std::uint64_t distributed_exact_integer(const char *text)
{
  if(!text || !*text)throw std::runtime_error("Distributed SIDM exact integer is missing");
  for(const char *p=text;*p;p++)if(*p<'0' || *p>'9')throw std::runtime_error("Distributed SIDM integer must be unsigned decimal");
  errno=0;char *end=nullptr;const auto value=std::strtoull(text,&end,10);
  if(errno || !end || *end)throw std::runtime_error("Distributed SIDM exact integer exceeds uint64 domain");
  return std::uint64_t(value);
}
void accumulate(std::uint64_t &destination,std::uint64_t value)
{
  if(value>UINT64_MAX-destination)throw std::runtime_error("Distributed SIDM persistent telemetry counter exhausted");
  destination+=value;
}
}
void sim::sidm_distributed_source_step(void)
{
  try {research_sidm::prepare_native_differential_model(All.SIDMReferenceAngularKind,All.SIDMNativeTablePath,All.SIDMAngularModelHash,All.SIDMNativeTableSHA256);}
  catch(const std::exception &error){Terminate("SIDM model preparation failure: %s",error.what());}
  if(All.SIDMReferenceAngularKind==2 && All.SIDMReferenceParticleConvention!=1)
    Terminate("Born Moller requires folded identical-particle convention");
  try {
    using namespace research_sidm;
    bool valid=All.SIDMEventAuditModulus>=0 && All.SIDMEventAuditResidue>=0 && (All.SIDMEventAuditModulus==0?All.SIDMEventAuditResidue==0:All.SIDMEventAuditResidue<All.SIDMEventAuditModulus) && NTask>0 && NTask<=65536 && All.SIDMReferenceMaximumParticles>0 &&
        Sp.NumPart<=All.SIDMReferenceMaximumParticles && All.SIDMReferenceMaximumLocalPairs>0 &&
        All.SIDMReferenceMaximumMatchingRounds>0 && All.SIDMReferenceMaximumSubsteps>0 &&
        All.SIDMReferenceMaximumEventsPerStep>0 && All.SIDMReferenceMaximumClockPanels>=2 &&
        All.SIDMReferenceSchemeKind>=0 && All.SIDMReferenceSchemeKind<=2 &&
        (All.SIDMReferenceAngularKind==0 || All.SIDMReferenceAngularKind==1 || All.SIDMReferenceAngularKind==2 || All.SIDMReferenceAngularKind==3) &&
        (All.SIDMReferenceParticleConvention==0 || All.SIDMReferenceParticleConvention==1) &&
        (All.SIDMReferenceParticleConvention==0 || All.SIDMReferenceSplitAngleRadians<1.5707963267948966192313216916398) &&
        std::strlen(All.SIDMAngularModelHash)==64 && *All.SIDMEventIdentityNamespace;
    const double a=All.ComovingIntegrationOn?All.Time:1.;
    valid=valid && std::isfinite(a) && a>0 && std::isfinite(All.HubbleParam) && All.HubbleParam>0 &&
        std::isfinite(All.TimeStep) && All.TimeStep>0 && std::isfinite(All.SIDMReferenceExchangeMemoryMiB) &&
        All.SIDMReferenceExchangeMemoryMiB>0 && All.SIDMReferenceExchangeMemoryMiB<=All.MaxMemSize*.25;
    require_collective(Communicator,valid,"Native distributed SIDM state, angular scheme or allocation is invalid");
    const std::uint64_t workspace=std::uint64_t(All.SIDMReferenceExchangeMemoryMiB*1024.*1024.);
    require_collective(Communicator,double(All.SIDMReferenceMaximumClockPanels)*128<=workspace/16,
        "Native distributed SIDM physical-clock workspace exceeds its source allocation");
    pair_exchange_limits limits{std::uint64_t(All.SIDMReferenceMaximumLocalPairs),
        std::uint64_t(All.SIDMReferenceMaximumMatchingRounds),distributed_exact_integer(All.SIDMReferenceMaximumExchangeRecords),workspace};
    const std::uint64_t maximum_visits=distributed_exact_integer(All.SIDMReferenceMaximumPairVisits);
    const std::uint64_t operator_seed=distributed_exact_integer(All.SIDMOperatorSeed);
    require_collective(Communicator,workspace/512>=std::uint64_t(Sp.NumPart) &&
        workspace/512>=limits.maximum_local_pairs && maximum_visits>0,
        "Native distributed SIDM owner/pair storage exceeds explicit workspace");
    const double radius=All.SIDMReferenceKernelRadiusProperKpc*3.0856775814913673e21;
    const double coarse_radius=All.SIDMReferenceCoarseKernelRadiusProperKpc*3.0856775814913673e21;
    const double mass_unit=All.UnitMass_in_g/(All.HubbleParam*1.98847e33);
    const double velocity_unit=All.UnitVelocity_in_cm_per_s/(a*1.e5);
    const double length_unit=All.UnitLength_in_cm*a/All.HubbleParam;
    require_positive(radius,"Native distributed SIDM proper support invalid");
    require_positive(coarse_radius,"Native distributed SIDM coarse proper support invalid");
    require_collective(Communicator,std::isfinite(All.SIDMReferenceMaximumKernelCoarseSupportFraction) &&
        All.SIDMReferenceMaximumKernelCoarseSupportFraction>=0 && All.SIDMReferenceMaximumKernelCoarseSupportFraction<=1 &&
        std::isfinite(All.SIDMReferenceMaximumOmittedScatteringUpperBound) && All.SIDMReferenceMaximumOmittedScatteringUpperBound>=0,
        "Native SIDM coarse-support and omitted-hazard limits must be explicitly finite nonnegative bounds");
    require_positive(mass_unit,"Native distributed SIDM physical mass conversion invalid");
    require_positive(velocity_unit,"Native distributed SIDM physical velocity conversion invalid");
    require_positive(length_unit,"Native distributed SIDM proper length conversion invalid");
    double proper_box=0;
#ifdef PERIODIC
    proper_box=All.BoxSize*length_unit;
#endif
    std::map<std::uint64_t,int> native_indices;
    std::set<std::uint64_t> fine_owned;
    std::vector<spatial_pair_particle> spatial;
    spatial.reserve(Sp.NumPart);
    valid=true;
    for(int index=0;index<Sp.NumPart;index++) {
      auto &particle=Sp.P[index];
      // Hydro, stellar and BH particles participate in native gravity and
      // their own source operators; only fine/coarse DM enters SIDM support.
      if(particle.getType()!=1 && particle.getType()!=2)continue;
      valid=valid && particle.getMass()>0;
      spatial_pair_particle row{};row.id=particle.ID.get();row.owner=ThisTask;
      row.mass_msun=particle.getMass()*mass_unit;row.native_type=particle.getType();
      row.proper_support_radius_cm=row.native_type==1?radius:coarse_radius;
      if(row.native_type==1)fine_owned.insert(row.id);
      double position[3];Sp.intpos_to_pos(particle.IntPos,position);
      for(int axis=0;axis<3;axis++)row.proper_position_cm[axis]=position[axis]*length_unit;
      valid=native_indices.emplace(row.id,index).second && valid;
      spatial.push_back(row);
    }
    require_collective(Communicator,valid,"Native distributed SIDM requires complete fine/coarse-DM census and unique local identities");
    const auto products=build_spatial_pairs(Communicator,spatial,std::max(radius,coarse_radius),proper_box,All.SIDMReferenceKernelKind,
        All.SIDMReferenceKernelOriginRelativeBound,maximum_visits,limits);
    std::vector<distributed_pair> scattering_pairs;
    for(const auto &pair:products.pairs)if(pair.left_type==1 && pair.right_type==1)scattering_pairs.push_back(pair);
    std::uint64_t visits=products.examined_neighbors;
    physical_clock_result clock;
    if(All.ComovingIntegrationOn) {
      require_collective(Communicator,std::isfinite(All.TimeOld) && All.TimeOld>0 && All.Time>All.TimeOld &&
          std::isfinite(All.SIDMReferenceMaximumScaleFactorFractionalChange) && All.SIDMReferenceMaximumScaleFactorFractionalChange>0 &&
          (All.Time-All.TimeOld)/All.TimeOld<=All.SIDMReferenceMaximumScaleFactorFractionalChange,
          "Native distributed SIDM cosmological source interval exceeds its frozen scale-factor domain");
      clock=cosmological_elapsed_seconds(All.TimeOld,All.Time,All.UnitTime_in_s,All.HubbleParam,
          [](double value){return driftfac::hubble_function(value);},All.SIDMReferenceClockAbsoluteToleranceSeconds,
          All.SIDMReferenceClockRelativeTolerance,std::size_t(All.SIDMReferenceMaximumClockPanels));
    } else clock.elapsed_seconds=All.TimeStep*All.UnitTime_in_s/All.HubbleParam;
    require_positive(clock.elapsed_seconds,"Native distributed SIDM elapsed physical interval invalid");
    char path[1024],diffusion_path[1024];
    const int size=std::snprintf(path,sizeof(path),"%s/sidm_events_%llu_rank%05d.hdf5",All.OutputDir,
        static_cast<unsigned long long>(All.Ti_Current),ThisTask);
    const int diffusion_size=std::snprintf(diffusion_path,sizeof(diffusion_path),"%s/sidm_diffusion_%llu_rank%05d.hdf5",All.OutputDir,
        static_cast<unsigned long long>(All.Ti_Current),ThisTask);
    if(size<0 || diffusion_size<0 || std::size_t(size)>=sizeof(path) || std::size_t(diffusion_size)>=sizeof(diffusion_path))
      throw std::runtime_error("Native distributed SIDM output path exceeds allocation");
    research_sidm_event_log log(path,All.SIDMAngularModelHash,All.SIDMEventIdentityNamespace,All.HubbleParam,
        std::uint64_t(All.SIDMReferenceMaximumEventsPerStep),All.SIDMReferenceSchemeKind==1?
        "frequent_drag_diffusion_pair":"rare_elastic_pair");
    std::unique_ptr<research_sidm_event_log> diffusion;
    if(All.SIDMReferenceSchemeKind==2)diffusion.reset(new research_sidm_event_log(diffusion_path,All.SIDMAngularModelHash,
        All.SIDMEventIdentityNamespace,All.HubbleParam,std::uint64_t(All.SIDMReferenceMaximumEventsPerStep),"frequent_drag_diffusion_pair"));
    const auto declare=[&](research_sidm_event_log &output,bool frequent) {
      output.configure_audit(std::uint64_t(All.SIDMEventAuditModulus),std::uint64_t(All.SIDMEventAuditResidue));
      output.declare_angular_model(All.SIDMReferenceAngularKind==1 || All.SIDMReferenceAngularKind==2,All.SIDMReferenceScreeningVelocityKmPerS,All.SIDMReferenceAngularKind==2,All.SIDMReferenceAngularKind==3,All.SIDMNativeTableSHA256);
      output.declare_particle_convention(All.SIDMReferenceParticleConvention==1);
      output.declare_native_background(All.ComovingIntegrationOn,All.TimeOld,All.Time,clock.elapsed_seconds,
          clock.estimated_absolute_error_seconds,All.UnitMass_in_g,All.UnitLength_in_cm,All.UnitTime_in_s,All.HubbleParam);
      if(frequent)output.declare_frequent_controls(All.SIDMReferenceMaximumTransferDepth,All.SIDMReferenceMaximumNeglectedMomentRatio);
      if(All.SIDMReferenceSchemeKind==2)output.declare_hybrid_split(All.SIDMReferenceSplitAngleRadians);
    };
    declare(log,All.SIDMReferenceSchemeKind==1);if(diffusion)declare(*diffusion,true);
    const auto read_owner=[&](std::uint64_t id) {
      const auto found=native_indices.find(id);
      if(found==native_indices.end())throw std::runtime_error("Native SIDM endpoint migrated or disappeared during synchronized source");
      auto &native=Sp.P[found->second];pair_particle state{};state.id=id;state.mass_msun=native.getMass()*mass_unit;
      for(int axis=0;axis<3;axis++)state.velocity_kms[axis]=native.Vel[axis]*velocity_unit;
      return state;
    };
    double elapsed=0,smallest_step=clock.elapsed_seconds;
    double maximum_coarse_fraction=0,maximum_omitted_depth=0;
    std::uint64_t substeps=0,matching_rounds=0,sent=products.exchange.sent_bytes,received=products.exchange.received_bytes,remote=0;
    std::uint64_t completed_pair_operations=0;
    source_resource_measurements resource_measurements;
    resource_measurements.exchange_buffer_peak_bytes=products.exchange.buffer_payload_peak_bytes;
    while(elapsed<clock.elapsed_seconds) {
      require_collective(Communicator,substeps<std::uint64_t(All.SIDMReferenceMaximumSubsteps) && visits<=maximum_visits &&
          products.pairs.size()<=(maximum_visits-visits)/2,"Native distributed SIDM source work/substep allocation exhausted");
      std::vector<pair_particle> current;current.reserve(spatial.size());
      for(const auto &particle:spatial)current.push_back(read_owner(particle.id));
      const double probability=All.SIDMReferenceSchemeKind==0?All.SIDMReferenceMaximumEventProbability:
          std::min(All.SIDMReferenceMaximumEventProbability,-std::expm1(-All.SIDMReferenceMaximumTransferDepth/2));
      const auto rate=plan_distributed_rates(Communicator,current,scattering_pairs,clock.elapsed_seconds-elapsed,
          All.SIDMReferenceCrossSectionNormalizationCm2PerG,All.SIDMReferenceAngularKind,
          All.SIDMReferenceScreeningVelocityKmPerS,probability,All.SIDMReferenceMaximumParticleProbability,
          All.SIDMReferenceConservationTolerance,All.SIDMReferenceEnergyFloor,All.SIDMReferenceSchemeKind==2?3:1,limits);
      accumulate(sent,rate.exchange.sent_bytes);accumulate(received,rate.exchange.received_bytes);
      resource_measurements.exchange_buffer_peak_bytes=std::max(resource_measurements.exchange_buffer_peak_bytes,rate.exchange.buffer_payload_peak_bytes);
      const double dt=rate.physical_dt_seconds;
      const double next=dt>=clock.elapsed_seconds-elapsed?clock.elapsed_seconds:elapsed+dt;
      require_collective(Communicator,next>elapsed && next<=clock.elapsed_seconds,"Native distributed SIDM physical source clock cannot advance");
      const auto coarse=measure_distributed_coarse_support(Communicator,current,fine_owned,products.pairs,scattering_pairs.size(),
          All.SIDMReferenceSchemeKind==2?3:1,dt,All.SIDMReferenceCrossSectionNormalizationCm2PerG,
          All.SIDMReferenceAngularKind,All.SIDMReferenceScreeningVelocityKmPerS,
          All.SIDMReferenceConservationTolerance,All.SIDMReferenceEnergyFloor,limits);
      accumulate(sent,coarse.exchange.sent_bytes);accumulate(received,coarse.exchange.received_bytes);
      resource_measurements.exchange_buffer_peak_bytes=std::max(resource_measurements.exchange_buffer_peak_bytes,coarse.exchange.buffer_payload_peak_bytes);
      bool guard_ok=true;
      for(const auto &row:coarse.particles) {
        auto &source=Sp.P[native_indices.at(row.first)].ResearchSourceState;
        source.sidm_kernel_coarse_support_fraction=row.second.kernel_coarse_support_fraction;
        const double cumulative=source.sidm_omitted_scattering_upper_bound+row.second.omitted_scattering_upper_bound;
        maximum_coarse_fraction=std::max(maximum_coarse_fraction,row.second.kernel_coarse_support_fraction);
        maximum_omitted_depth=std::max(maximum_omitted_depth,cumulative);
        guard_ok=guard_ok && std::isfinite(cumulative) &&
            row.second.kernel_coarse_support_fraction<=All.SIDMReferenceMaximumKernelCoarseSupportFraction &&
            cumulative<=All.SIDMReferenceMaximumOmittedScatteringUpperBound;
      }
      require_collective(Communicator,guard_ok,"Native SIDM fine-particle kernel coarse support or cumulative omitted scattering exceeds declared zoom limits");
      for(const auto &row:coarse.particles) {
        auto &source=Sp.P[native_indices.at(row.first)].ResearchSourceState;
        source.sidm_omitted_scattering_upper_bound+=row.second.omitted_scattering_upper_bound;
        source.sidm_coarse_guard_version=1;
      }
      std::map<std::uint64_t,long double> particle_depth;
      const long double depth_limit=-std::log1p(-static_cast<long double>(All.SIDMReferenceMaximumParticleProbability));
      const auto apply=[&](const distributed_pair &pair,pair_particle left,pair_particle right) {
        source_cpu_scope pair_resource_scope(&resource_measurements,0);
        if(left.mass_msun!=pair.left_mass_msun || right.mass_msun!=pair.right_mass_msun)
          throw std::runtime_error("Native distributed SIDM particle mass changed after frozen neighbor discovery");
        vector3 relative;for(int axis=0;axis<3;axis++)relative[axis]=left.velocity_kms[axis]-right.velocity_kms[axis];
        const double speed=norm(relative);
        const double amplitude=speed==0?0:native_total_cross_section(All.SIDMReferenceAngularKind,All.SIDMReferenceCrossSectionNormalizationCm2PerG,speed,All.SIDMReferenceScreeningVelocityKmPerS);
        // Explicit physical assignments are valid with the pinned C++11
        // compiler as well as the standalone controls' newer language modes.
        rare_parameters parameters{};
        parameters.operator_seed=operator_seed;
        parameters.cross_section_per_mass_cm2_g=amplitude;
        parameters.kernel_overlap_inverse_cm3=pair.kernel_overlap_inverse_cm3;
        parameters.physical_dt_seconds=dt;
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
        std::uint64_t attempt=distributed_pair_attempt_key(pair.left_id,pair.right_id,All.SIDMReferenceSubstepCount)&UINT64_C(0x7fffffffffffffff);
        const std::uint64_t first_attempt=attempt;
        pair_result result;distributed_pair_effect effect;
        if(All.SIDMReferenceSchemeKind==0) {
          result=apply_rare_pair(left,right,parameters,std::uint64_t(All.Ti_Current),attempt,All.Time,a,ThisTask,All.SIDMReferenceEventCount,&log);
          effect.rare_events=result.scattered;
        } else if(All.SIDMReferenceSchemeKind==1) {
          result=apply_frequent_pair(left,right,parameters,
              All.SIDMReferenceParticleConvention==1?1.5707963267948966192313216916398:3.1415926535897932384626433832795,
              All.SIDMReferenceMaximumTransferDepth,
              All.SIDMReferenceMaximumNeglectedMomentRatio,std::uint64_t(All.Ti_Current),attempt,All.Time,a,ThisTask,All.SIDMReferenceEventCount,&log);
          effect.frequent_updates=result.scattered;
        } else {
          const auto hybrid=apply_hybrid_pair(left,right,parameters,All.SIDMReferenceSplitAngleRadians,
              All.SIDMReferenceMaximumTransferDepth,All.SIDMReferenceMaximumNeglectedMomentRatio,std::uint64_t(All.Ti_Current),
              attempt,All.Time,a,ThisTask,All.SIDMReferenceEventCount,&log,diffusion.get());
          result=hybrid.aggregate;effect.rare_events=hybrid.rare_scattered;effect.frequent_updates=hybrid.frequent_updates;
        }
        effect.left=left;effect.right=right;effect.optical_depth=result.optical_depth;
        accumulate(All.SIDMReferenceAttemptCount,attempt-first_attempt);
        accumulate(All.SIDMReferenceFrequentUpdateCount,effect.frequent_updates);
        All.SIDMReferenceOpticalDepthSum+=effect.optical_depth;
        if(!std::isfinite(All.SIDMReferenceOpticalDepthSum))throw std::runtime_error("Native distributed SIDM exposure overflow");
        return effect;
      };
      const auto commit=[&](const pair_particle &state,double depth,std::uint64_t rare_events,std::uint64_t frequent_updates) {
        particle_depth[state.id]+=depth;
        if(!std::isfinite(particle_depth[state.id]) || particle_depth[state.id]>depth_limit)
          throw std::runtime_error("Native distributed SIDM actual particle hazard exceeds the declared source bound");
        auto &native=Sp.P[native_indices.at(state.id)];
        for(int axis=0;axis<3;axis++)native.Vel[axis]=state.velocity_kms[axis]/velocity_unit;
        auto &source=native.ResearchSourceState;source.sidm_optical_depth+=depth;
        accumulate(source.sidm_collision_count,rare_events);source.sidm_random_counter=All.SIDMReferenceSubstepCount;
        if(rare_events || frequent_updates)source.sidm_last_update_time=All.Time;
        if(!std::isfinite(source.sidm_optical_depth))throw std::runtime_error("Native distributed SIDM particle exposure overflow");
      };
      const auto exchange=execute_distributed_pairs(Communicator,scattering_pairs,limits,read_owner,apply,commit,operator_seed,std::uint64_t(All.Ti_Current),All.SIDMReferenceSubstepCount);
      accumulate(sent,exchange.sent_bytes);accumulate(received,exchange.received_bytes);
      accumulate(matching_rounds,exchange.matching_rounds);accumulate(remote,exchange.remote_endpoint_updates);
      accumulate(completed_pair_operations,exchange.completed_local_pairs);
      resource_measurements.exchange_buffer_peak_bytes=std::max(resource_measurements.exchange_buffer_peak_bytes,exchange.buffer_payload_peak_bytes);
      visits+=2*products.pairs.size();elapsed=next;smallest_step=std::min(smallest_step,dt);++substeps;
      accumulate(All.SIDMReferenceSubstepCount,1);
      if(All.SIDMReferenceSmallestSourceStepSeconds==0 || dt<All.SIDMReferenceSmallestSourceStepSeconds)All.SIDMReferenceSmallestSourceStepSeconds=dt;
    }
    accumulate(All.SIDMReferencePairVisitCount,visits);
    accumulate(All.SIDMReferenceMatchingRoundCount,matching_rounds);accumulate(All.SIDMReferenceRemoteEndpointUpdateCount,remote);
    accumulate(All.SIDMReferenceExchangeSentBytes,sent);accumulate(All.SIDMReferenceExchangeReceivedBytes,received);
    std::uint64_t measured_peak_rss=0;
    const bool rss_available=process_peak_rss_bytes(measured_peak_rss);
    const auto finish=[&](research_sidm_event_log &output) {
      output.record_resource_measurements(resource_measurements.cpu_available,resource_measurements.pair_cpu_seconds,
          resource_measurements.rare_cpu_seconds,resource_measurements.frequent_cpu_seconds,
          resource_measurements.rare_calls,resource_measurements.frequent_calls,rss_available,measured_peak_rss,
          resource_measurements.exchange_buffer_peak_bytes,native_differential_model()?native_differential_model()->memory_bytes():0);
      output.record_source_interval(clock.elapsed_seconds,substeps,visits,smallest_step,All.SIDMReferenceMaximumEventProbability,
          All.SIDMReferenceMaximumParticleProbability,All.SIDMReferenceKernelKind,All.SIDMReferenceKernelRadiusProperKpc);
      output.record_distributed_exchange(ThisTask,NTask,products.pairs.size(),products.received_ghost_records,matching_rounds,remote,sent,received,
          completed_pair_operations);
      output.record_coarse_support(maximum_coarse_fraction,maximum_omitted_depth,
          All.SIDMReferenceMaximumKernelCoarseSupportFraction,All.SIDMReferenceMaximumOmittedScatteringUpperBound);
      output.finish();
    };
    finish(log);if(diffusion)finish(*diffusion);
  } catch(const std::exception &error) {Terminate("Native distributed SIDM source failure: %s",error.what());}
}
#endif
