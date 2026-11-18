/* First-party bounded native SIDM diagnostic output.
 * Used only when a physical operator explicitly records an applied update.
 * Incomplete logs remain .partial files. The caller supplies a persistent
 * per-rank event sequence and must checkpoint it along with native physics. */
#ifndef RESEARCH_SIDM_EVENT_LOG_H
#define RESEARCH_SIDM_EVENT_LOG_H
#include "research_sidm_event_record.h"
#include <hdf5.h>
#include <algorithm>
#include <cmath>
#include <array>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <string>
#include <unistd.h>

class research_sidm_event_log
{
  std::string final_name,partial_name;
  hid_t file=-1,group=-1;
  std::array<hid_t,7> datasets;
  hsize_t count=0,maximum;
  bool published=false,poisoned=false,model_declared=false,replay=false;
  hsize_t replay_rows=0;
  std::uint64_t last_event_id=0;
  std::uint64_t audit_modulus=1,audit_residue=0,observed_updates=0;
  double maximum_pair_energy_relative_residual=0;
  static std::uint64_t audit_hash(std::uint64_t value)
  {
    value+=UINT64_C(0x9e3779b97f4a7c15);value=(value^(value>>30))*UINT64_C(0xbf58476d1ce4e5b9);
    value=(value^(value>>27))*UINT64_C(0x94d049bb133111eb);return value^(value>>31);
  }
  static hid_t checked(hid_t value,const char *message)
  { if(value<0) throw std::runtime_error(message); return value; }
  static void status(herr_t value,const char *message)
  { if(value<0) throw std::runtime_error(message); }
  static void text(hid_t object,const char *name,const std::string &value)
  {
    hid_t type=checked(H5Tcopy(H5T_C_S1),"SIDM log text type failed");
    status(H5Tset_size(type,value.size()+1),"SIDM log text size failed");
    hid_t space=checked(H5Screate(H5S_SCALAR),"SIDM log attribute space failed");
    hid_t attribute=checked(H5Acreate2(object,name,type,space,H5P_DEFAULT,H5P_DEFAULT),"SIDM log attribute create failed");
    status(H5Awrite(attribute,type,value.c_str()),"SIDM log text write failed");
    H5Aclose(attribute);H5Sclose(space);H5Tclose(type);
  }
  static void scalar(hid_t object,const char *name,double value)
  {
    hid_t space=checked(H5Screate(H5S_SCALAR),"SIDM log scalar space failed");
    hid_t attribute=checked(H5Acreate2(object,name,H5T_IEEE_F64LE,space,H5P_DEFAULT,H5P_DEFAULT),"SIDM log scalar create failed");
    status(H5Awrite(attribute,H5T_NATIVE_DOUBLE,&value),"SIDM log scalar write failed");
    H5Aclose(attribute);H5Sclose(space);
  }
  hid_t column(const char *name,hid_t type,int rank,hsize_t second=1,hsize_t third=1)
  {
    hsize_t shape[3]={0,second,third},limit[3]={H5S_UNLIMITED,second,third},chunk[3]={256,second,third};
    hid_t space=checked(H5Screate_simple(rank,shape,limit),"SIDM log dataset space failed");
    hid_t properties=checked(H5Pcreate(H5P_DATASET_CREATE),"SIDM log properties failed");
    status(H5Pset_chunk(properties,rank,chunk),"SIDM log chunk allocation failed");
    hid_t dataset=checked(H5Dcreate2(group,name,type,space,H5P_DEFAULT,properties,H5P_DEFAULT),"SIDM log dataset create failed");
    H5Pclose(properties);H5Sclose(space);return dataset;
  }
  void append_column(hid_t dataset,hid_t type,const void *data,int rank,hsize_t second=1,hsize_t third=1)
  {
    hsize_t dimensions[3]={count+1,second,third},begin[3]={count,0,0},extent[3]={1,second,third};
    status(H5Dset_extent(dataset,dimensions),"SIDM log dataset extension failed");
    hid_t space=checked(H5Dget_space(dataset),"SIDM log append space failed");
    status(H5Sselect_hyperslab(space,H5S_SELECT_SET,begin,nullptr,extent,nullptr),"SIDM log row selection failed");
    hid_t memory=checked(H5Screate_simple(rank,extent,nullptr),"SIDM log append memory failed");
    status(H5Dwrite(dataset,type,memory,space,H5P_DEFAULT,data),"SIDM log row write failed");
    H5Sclose(memory);H5Sclose(space);
  }
  static std::string read_text(hid_t object,const char *name)
  {
    hid_t attribute=checked(H5Aopen(object,name,H5P_DEFAULT),"SIDM replay attribute missing");
    hid_t type=checked(H5Aget_type(attribute),"SIDM replay attribute type failed");
    const std::size_t bytes=H5Tget_size(type);
    if(!bytes || bytes>4096)throw std::runtime_error("SIDM replay attribute bound exceeded");
    std::string value(bytes,'\0');status(H5Aread(attribute,type,&value[0]),"SIDM replay attribute read failed");
    H5Tclose(type);H5Aclose(attribute);value.resize(std::strlen(value.c_str()));return value;
  }
  static double read_scalar(hid_t object,const char *name)
  {
    hid_t attribute=checked(H5Aopen(object,name,H5P_DEFAULT),"SIDM replay scalar missing");
    double value;status(H5Aread(attribute,H5T_NATIVE_DOUBLE,&value),"SIDM replay scalar read failed");H5Aclose(attribute);return value;
  }
  template<typename T> void replay_column(hid_t dataset,hid_t type,const T *expected,int rank,hsize_t second=1,hsize_t third=1)
  {
    hsize_t begin[3]={count,0,0},extent[3]={1,second,third},shape[3]={0,0,0};
    hid_t space=checked(H5Dget_space(dataset),"SIDM replay space failed");
    if(H5Sget_simple_extent_ndims(space)!=rank)throw std::runtime_error("SIDM replay row rank differs");
    H5Sget_simple_extent_dims(space,shape,nullptr);
    if(shape[0]!=replay_rows || (rank>1 && shape[1]!=second) || (rank>2 && shape[2]!=third))
      throw std::runtime_error("SIDM replay column census differs");
    status(H5Sselect_hyperslab(space,H5S_SELECT_SET,begin,nullptr,extent,nullptr),"SIDM replay row selection failed");
    hid_t memory=checked(H5Screate_simple(rank,extent,nullptr),"SIDM replay memory failed");
    std::array<T,6> actual{};
    status(H5Dread(dataset,type,memory,space,H5P_DEFAULT,actual.data()),"SIDM replay row read failed");
    H5Sclose(memory);H5Sclose(space);
    for(hsize_t index=0;index<second*third;index++)if(actual[index]!=expected[index])
      throw std::runtime_error("SIDM checkpoint replay differs from retained native event states");
  }
  void close_handles() noexcept
  {
    for(hid_t &dataset:datasets) if(dataset>=0) {H5Dclose(dataset);dataset=-1;}
    if(group>=0) {H5Gclose(group);group=-1;}
    if(file>=0) {H5Fclose(file);file=-1;}
  }
public:
  research_sidm_event_log(const std::string &path,const std::string &model_hash,
                          const std::string &identity_namespace,double hubble_parameter,
                          std::uint64_t maximum_records,const std::string &operator_kind="rare_elastic_pair"):
                          final_name(path),partial_name(path+".partial"),maximum(maximum_records)
  {
    datasets.fill(-1);
    if((operator_kind!="rare_elastic_pair" && operator_kind!="frequent_drag_diffusion_pair") ||
       model_hash.size()!=64 || identity_namespace.empty() || maximum_records==0 ||
       !std::isfinite(hubble_parameter) || hubble_parameter<=0)
      throw std::runtime_error("SIDM log needs physical model, identity namespace and allocation");
    if(access(final_name.c_str(),F_OK)==0) {
      replay=true;
      try {
        file=checked(H5Fopen(final_name.c_str(),H5F_ACC_RDONLY,H5P_DEFAULT),"SIDM replay log open failed");
        if(read_text(file,"publication_state")!="COMPLETE")throw std::runtime_error("SIDM replay log is incomplete");
        group=checked(H5Gopen2(file,"SIDMScatteringEvents",H5P_DEFAULT),"SIDM replay group open failed");
        if(read_text(group,"schema")!="native-sidm-event-log-v1" || read_text(group,"angular_model_hash")!=model_hash ||
           read_text(group,"event_identity_namespace")!=identity_namespace ||
           read_text(group,"operator_kind")!=operator_kind)throw std::runtime_error("SIDM replay source model differs");
        hid_t params=checked(H5Gopen2(file,"Parameters",H5P_DEFAULT),"SIDM replay parameters missing");
        const double old_hubble=read_scalar(params,"HubbleParam");H5Gclose(params);
        if(old_hubble!=hubble_parameter)throw std::runtime_error("SIDM replay unit convention differs");
        const char *names[7]={"EventID","ParticleIDs","VelocityBefore","VelocityAfter","Masses","Time","ScaleFactor"};
        for(int index=0;index<7;index++)datasets[index]=checked(H5Dopen2(group,names[index],H5P_DEFAULT),"SIDM replay dataset missing");
        hid_t space=checked(H5Dget_space(datasets[0]),"SIDM replay census missing");
        if(H5Sget_simple_extent_ndims(space)!=1)throw std::runtime_error("SIDM replay identity census invalid");
        H5Sget_simple_extent_dims(space,&replay_rows,nullptr);H5Sclose(space);
        if(replay_rows>maximum)throw std::runtime_error("SIDM replay allocation changed");
      } catch(...) {close_handles();throw;}
      return;
    }
    if(access(partial_name.c_str(),F_OK)==0) {
      const std::string retained=partial_name+".interrupted."+std::to_string(static_cast<long long>(getpid()));
      if(link(partial_name.c_str(),retained.c_str())!=0 || unlink(partial_name.c_str())!=0)
        throw std::runtime_error("SIDM incomplete log preservation failed");
    }
    try {
      file=checked(H5Fcreate(partial_name.c_str(),H5F_ACC_EXCL,H5P_DEFAULT,H5P_DEFAULT),"SIDM log create failed");
      text(file,"publication_state","OPEN");
      group=checked(H5Gcreate2(file,"SIDMScatteringEvents",H5P_DEFAULT,H5P_DEFAULT,H5P_DEFAULT),"SIDM event group create failed");
      text(group,"schema","native-sidm-event-log-v1");text(group,"angular_model_hash",model_hash);
      text(group,"event_identity_namespace",identity_namespace);text(group,"operator_kind",operator_kind);
      hid_t params=checked(H5Gcreate2(file,"Parameters",H5P_DEFAULT,H5P_DEFAULT,H5P_DEFAULT),"SIDM log parameter group failed");
      scalar(params,"HubbleParam",hubble_parameter);H5Gclose(params);
      datasets[0]=column("EventID",H5T_STD_U64LE,1);
      datasets[1]=column("ParticleIDs",H5T_STD_U64LE,2,2);
      datasets[2]=column("VelocityBefore",H5T_IEEE_F64LE,3,2,3);
      datasets[3]=column("VelocityAfter",H5T_IEEE_F64LE,3,2,3);
      datasets[4]=column("Masses",H5T_IEEE_F64LE,2,2);
      datasets[5]=column("Time",H5T_IEEE_F64LE,1);
      datasets[6]=column("ScaleFactor",H5T_IEEE_F64LE,1);
      for(int index=2;index<=4;index++) {
        scalar(datasets[index],"to_cgs",index==4?1.98847e33:1e5);
        scalar(datasets[index],"a_scaling",0.);scalar(datasets[index],"h_scaling",0.);
      }
      text(datasets[2],"units","physical peculiar km/s");text(datasets[3],"units","physical peculiar km/s");
      text(datasets[4],"units","Msun");
    } catch(...) {close_handles();throw;}
  }
  research_sidm_event_log(const research_sidm_event_log &)=delete;
  research_sidm_event_log &operator=(const research_sidm_event_log &)=delete;
  ~research_sidm_event_log() {close_handles();}
  void configure_audit(std::uint64_t modulus,std::uint64_t residue)
  {
    if(published || poisoned || file<0 || count || (modulus && residue>=modulus) || (!modulus && residue))
      throw std::runtime_error("SIDM selected event-audit population invalid");
    audit_modulus=modulus;audit_residue=residue;
    const std::string selection=modulus==0?"aggregate_only":modulus==1?"all_particles":"hashed_uint64_particle_ids_either_endpoint";
    if(replay) {
      if(read_text(group,"audit_population")!=selection || read_text(group,"audit_modulus")!=std::to_string(modulus) ||
          read_text(group,"audit_residue")!=std::to_string(residue))throw std::runtime_error("SIDM replay changes selected audit population");
      return;
    }
    text(group,"audit_population",selection);text(group,"audit_modulus",std::to_string(modulus));text(group,"audit_residue",std::to_string(residue));
  }
  void declare_angular_model(bool screened,double screening_velocity_kms,bool born_moller=false,bool tabulated=false,const char *table_hash="NONE")
  {
    if(published || poisoned || file<0 || model_declared || (screened && (!std::isfinite(screening_velocity_kms) || screening_velocity_kms<=0)))
      throw std::runtime_error("SIDM event log angular model declaration invalid");
    if(tabulated && (!table_hash || std::strlen(table_hash)!=64))throw std::runtime_error("SIDM event differential table hash missing");
    const char *kind=tabulated?"differential_table":born_moller?"born_moller":screened?"screened_rutherford":"isotropic";
    if(born_moller && !screened)throw std::runtime_error("Born Moller requires a finite screening scale");
    if(replay) {
      if(read_text(group,"angular_kind")!=kind ||
         read_scalar(group,"screening_velocity_kms")!=(screened?screening_velocity_kms:0.))
        throw std::runtime_error("SIDM checkpoint replay changes actual angular operator");
      if(tabulated && read_text(group,"differential_table_sha256")!=table_hash)throw std::runtime_error("SIDM replay changes differential table bytes");
      model_declared=true;return;
    }
    text(group,"angular_kind",kind);
    text(group,"random_generator","Random123_Philox4x64_10");
    text(group,"probability_law","small_step_Bernoulli_Gamma_dt");
    if(tabulated)text(group,"differential_table_sha256",table_hash);
    scalar(group,"screening_velocity_kms",screened?screening_velocity_kms:0.);
    model_declared=true;
  }
  void record_applied_update(const research_sidm_event_record &event)
  {
    if(published || poisoned || !model_declared || file<0) throw std::runtime_error("SIDM event log allocation exhausted or closed");
    event.require_physical();
    if(observed_updates==UINT64_MAX)throw std::runtime_error("SIDM aggregate applied-update counter exhausted");
    ++observed_updates;
    long double before=0,after=0;
    for(int endpoint=0;endpoint<2;endpoint++)for(int axis=0;axis<3;axis++) {
      const double vb=event.velocity_before_kms[3*endpoint+axis],va=event.velocity_after_kms[3*endpoint+axis];
      before+=.5L*event.mass_msun[endpoint]*vb*vb;after+=.5L*event.mass_msun[endpoint]*va*va;
    }
    if(before>0)maximum_pair_energy_relative_residual=std::max(maximum_pair_energy_relative_residual,double(std::abs(after-before)/before));
    const bool selected=audit_modulus && ((audit_hash(event.particle_ids[0])%audit_modulus==audit_residue) ||
        (audit_hash(event.particle_ids[1])%audit_modulus==audit_residue));
    if(!selected)return;
    if(count>=maximum)throw std::runtime_error("SIDM selected immutable event chunk exceeds declared record allocation");
    if(count && event.event_id<=last_event_id) throw std::runtime_error("SIDM log requires increasing persistent event IDs");
    if(replay) {
      if(count>=replay_rows)throw std::runtime_error("SIDM checkpoint replay adds an unrecorded event");
      replay_column(datasets[0],H5T_NATIVE_UINT64,&event.event_id,1);
      replay_column(datasets[1],H5T_NATIVE_UINT64,event.particle_ids.data(),2,2);
      replay_column(datasets[2],H5T_NATIVE_DOUBLE,event.velocity_before_kms.data(),3,2,3);
      replay_column(datasets[3],H5T_NATIVE_DOUBLE,event.velocity_after_kms.data(),3,2,3);
      replay_column(datasets[4],H5T_NATIVE_DOUBLE,event.mass_msun.data(),2,2);
      replay_column(datasets[5],H5T_NATIVE_DOUBLE,&event.native_time,1);
      replay_column(datasets[6],H5T_NATIVE_DOUBLE,&event.scale_factor,1);
      ++count;last_event_id=event.event_id;return;
    }
    try {
    append_column(datasets[0],H5T_NATIVE_UINT64,&event.event_id,1);
    append_column(datasets[1],H5T_NATIVE_UINT64,event.particle_ids.data(),2,2);
    append_column(datasets[2],H5T_NATIVE_DOUBLE,event.velocity_before_kms.data(),3,2,3);
    append_column(datasets[3],H5T_NATIVE_DOUBLE,event.velocity_after_kms.data(),3,2,3);
    append_column(datasets[4],H5T_NATIVE_DOUBLE,event.mass_msun.data(),2,2);
    append_column(datasets[5],H5T_NATIVE_DOUBLE,&event.native_time,1);
    append_column(datasets[6],H5T_NATIVE_DOUBLE,&event.scale_factor,1);
    ++count;last_event_id=event.event_id;
    } catch(...) {poisoned=true;throw;}
  }
  void record_source_interval(double interval_seconds,std::uint64_t substeps,std::uint64_t pair_visits,
                              double smallest_step_seconds,double pair_probability_limit,double particle_probability_limit,
                              int kernel_kind,double kernel_radius_proper_kpc)
  {
    if(published || poisoned || !model_declared || file<0 || !std::isfinite(interval_seconds) || interval_seconds<=0 ||
        !substeps || !std::isfinite(smallest_step_seconds) || smallest_step_seconds<=0)
      throw std::runtime_error("SIDM measured source interval invalid");
    const std::string step_text=std::to_string(substeps),visit_text=std::to_string(pair_visits);
    if(replay) {
      if(read_text(group,"source_scheduler")!="pre_draw_bounded_rare_substeps" ||
          read_text(group,"source_substep_count")!=step_text || read_text(group,"source_pair_visits")!=visit_text ||
          read_scalar(group,"source_interval_seconds")!=interval_seconds ||
          read_scalar(group,"smallest_source_step_seconds")!=smallest_step_seconds ||
          read_scalar(group,"maximum_pair_probability")!=pair_probability_limit ||
          read_scalar(group,"maximum_particle_probability")!=particle_probability_limit ||
          read_scalar(group,"kernel_kind")!=kernel_kind || read_scalar(group,"kernel_radius_proper_kpc")!=kernel_radius_proper_kpc)
        throw std::runtime_error("SIDM checkpoint replay changes measured source scheduling or kernel");
      return;
    }
    text(group,"source_scheduler","pre_draw_bounded_rare_substeps");
    text(group,"event_clock_scope","native endpoint source kick; spatial kernels frozen over source interval");
    text(group,"source_substep_count",step_text);text(group,"source_pair_visits",visit_text);
    scalar(group,"source_interval_seconds",interval_seconds);scalar(group,"smallest_source_step_seconds",smallest_step_seconds);
    scalar(group,"maximum_pair_probability",pair_probability_limit);scalar(group,"maximum_particle_probability",particle_probability_limit);
    scalar(group,"kernel_kind",kernel_kind);scalar(group,"kernel_radius_proper_kpc",kernel_radius_proper_kpc);
  }
  void record_largest_source_step(double seconds)
  {
    if(published || poisoned || file<0 || !std::isfinite(seconds) || seconds<=0)
      throw std::runtime_error("SIDM largest actual source step invalid");
    if(replay) {
      if(read_scalar(group,"largest_source_step_seconds")!=seconds)
        throw std::runtime_error("SIDM replay changes largest actual source step");
      return;
    }
    scalar(group,"largest_source_step_seconds",seconds);
  }
  void declare_frequent_controls(double maximum_transfer_depth,double maximum_moment_ratio)
  {
    if(published || poisoned || file<0 || count || !std::isfinite(maximum_transfer_depth) ||
        maximum_transfer_depth<=0 || maximum_transfer_depth>=1 || !std::isfinite(maximum_moment_ratio) ||
        maximum_moment_ratio<=0 || maximum_moment_ratio>=2)
      throw std::runtime_error("SIDM frequent approximation declaration invalid");
    if(replay) {
      if(read_scalar(group,"maximum_transfer_depth")!=maximum_transfer_depth ||
          read_scalar(group,"maximum_neglected_moment_ratio")!=maximum_moment_ratio)
        throw std::runtime_error("SIDM replay changes frequent approximation controls");
      return;
    }
    scalar(group,"maximum_transfer_depth",maximum_transfer_depth);
    scalar(group,"maximum_neglected_moment_ratio",maximum_moment_ratio);
    if(H5Aexists(group,"angular_particle_convention")==0)text(group,"angular_particle_convention","directed_distinguishable");
    text(group,"record_scope","effective macro drag/diffusion updates; not microscopic scattering events");
  }
  void declare_canonical_substep_epochs()
  {
    if(published || poisoned || file<0 || count)throw std::runtime_error("Canonical epoch policy must precede event rows");
    const char *policy="source_substep_left_endpoint";
    const char *radius="whole_interval_midpoint_reference_only; actual support is proper_radius/row_source_scale";
    if(replay) {
      if(read_text(group,"canonical_source_epoch_policy")!=policy || read_text(group,"canonical_kernel_radius_role")!=radius ||
          read_text(group,"canonical_source_expansion_policy")!="adaptive_substep_span_with_inverse_brackets")
        throw std::runtime_error("Canonical replay changes its source epoch/support policy");
      return;
    }
    text(group,"canonical_source_epoch_policy",policy);text(group,"canonical_kernel_radius_role",radius);
    text(group,"canonical_source_expansion_policy","adaptive_substep_span_with_inverse_brackets");
  }
  void record_canonical_substep_clock(std::size_t evaluations,double largest_span,double maximum_error,double maximum_bracket)
  {
    if(published || poisoned || file<0 || !std::isfinite(largest_span) || largest_span<0 ||
        !std::isfinite(maximum_error) || maximum_error<0 || !std::isfinite(maximum_bracket) || maximum_bracket<0)
      throw std::runtime_error("Canonical substep clock measurements invalid");
    const char *names[]={"canonical_source_clock_function_evaluations","canonical_largest_substep_scale_fraction",
      "canonical_substep_maximum_clock_error_seconds","canonical_substep_maximum_inverse_bracket_relative_width"};
    const double values[]={double(evaluations),largest_span,maximum_error,maximum_bracket};
    if(replay) {for(int i=0;i<4;i++)if(read_scalar(group,names[i])!=values[i])throw std::runtime_error("Canonical replay changes realized substep clock work/domain");return;}
    for(int i=0;i<4;i++)scalar(group,names[i],values[i]);
  }
  void declare_hybrid_split(double split_angle)
  {
    if(published || poisoned || file<0 || count || !std::isfinite(split_angle) || split_angle<=0 ||
        split_angle>=3.1415926535897932384626433832795)
      throw std::runtime_error("SIDM hybrid angular partition invalid");
    if(replay) {
      if(read_scalar(group,"split_angle_radians")!=split_angle ||
          read_text(group,"source_composition")!="small_half_large_full_small_half")
        throw std::runtime_error("SIDM replay changes hybrid angular partition or source composition");
      return;
    }
    scalar(group,"split_angle_radians",split_angle);
    text(group,"source_composition","small_half_large_full_small_half");
  }
  void declare_particle_convention(bool folded)
  {
    if(published || poisoned || file<0 || count)throw std::runtime_error("SIDM particle convention declaration invalid");
    const std::string value=folded?"folded_identical_particle":"directed_distinguishable";
    if(replay) {
      if(read_text(group,"angular_particle_convention")!=value)
        throw std::runtime_error("SIDM replay changes actual angular particle convention");
      return;
    }
    text(group,"angular_particle_convention",value);
  }
  void declare_native_background(bool comoving,double begin,double end,double elapsed,double estimated_time_error,
                                 double unit_mass,double unit_length,double unit_time,double hubble)
  {
    if(published || poisoned || file<0 || count || !std::isfinite(begin) || !std::isfinite(end) || end<=begin ||
        !std::isfinite(elapsed) || elapsed<=0 || !std::isfinite(estimated_time_error) || estimated_time_error<0)
      throw std::runtime_error("SIDM actual background interval invalid");
    const char *names[8]={"native_interval_begin","native_interval_end","physical_elapsed_seconds","estimated_clock_error_seconds",
      "native_unit_mass_g","native_unit_length_cm","native_unit_time_s","native_hubble_parameter"};
    const double values[8]={begin,end,elapsed,estimated_time_error,unit_mass,unit_length,unit_time,hubble};
    if(replay) {
      if(read_text(group,"native_clock_coordinate")!=(comoving?"scale_factor":"native_time_code"))
        throw std::runtime_error("SIDM replay changes cosmological clock convention");
      for(int i=0;i<8;i++)if(read_scalar(group,names[i])!=values[i])
        throw std::runtime_error("SIDM replay changes actual native physical background interval or units");
      return;
    }
    text(group,"native_clock_coordinate",comoving?"scale_factor":"native_time_code");
    text(group,"velocity_conversion","native canonical velocity / a to physical peculiar velocity");
    for(int i=0;i<8;i++)scalar(group,names[i],values[i]);
  }
  void declare_canonical_background(double native_begin,double native_end,double tau_begin,double tau_end,
      double begin_a,double end_a,double frozen_a,double clock_error,double inverse_bracket,
      double unit_mass,double unit_length,double unit_time,double hubble,
      const std::string &clock_sha,const std::string &background_sha,double canonical_radius_cm)
  {
    if(published || poisoned || file<0 || count || clock_sha.size()!=64 || background_sha.size()!=64
        || native_begin<0 || native_end<=native_begin || tau_begin<0 || tau_end<=tau_begin
        || begin_a<=0 || end_a<begin_a || frozen_a<begin_a || frozen_a>end_a
        || clock_error<0 || inverse_bracket<0 || canonical_radius_cm<=0)
      throw std::runtime_error("SIDM canonical background declaration invalid");
    const char *names[]={"native_interval_begin","native_interval_end","canonical_tau_begin_seconds","canonical_tau_end_seconds",
      "physical_scale_factor_begin","physical_scale_factor_end","frozen_physical_scale_factor","estimated_clock_error_seconds",
      "inverse_epoch_bracket_relative_width","native_unit_mass_g","native_unit_length_cm","native_unit_time_s",
      "native_hubble_parameter","canonical_kernel_radius_cm"};
    const double values[]={native_begin,native_end,tau_begin,tau_end,begin_a,end_a,frozen_a,clock_error,inverse_bracket,
      unit_mass,unit_length,unit_time,hubble,canonical_radius_cm};
    for(double value:values)if(!std::isfinite(value))throw std::runtime_error("SIDM canonical background has nonfinite values");
    if(unit_mass<=0 || unit_length<=0 || unit_time<=0 || hubble<=0)
      throw std::runtime_error("SIDM canonical native units invalid");
    const char *text_names[]={"native_clock_coordinate","velocity_frame","position_frame","source_clock_frame",
      "effective_differential_law","canonical_clock_sha256","physical_background_sha256"};
    const std::string text_values[]={"canonical_native_time_code","canonical_u=a*v_pec","canonical_x=r/a","tau_integral_dt_over_a_squared",
      "a^-2*f_physical(u/a,mu)",clock_sha,background_sha};
    const std::string velocity_units="canonical a*v_pec km/s";
    if(replay) {
      for(int index:{2,3})if(read_text(datasets[index],"units")!=velocity_units)
        throw std::runtime_error("SIDM replay changes canonical event velocity units");
      for(std::size_t i=0;i<sizeof(values)/sizeof(values[0]);i++)
        if(read_scalar(group,names[i])!=values[i])throw std::runtime_error("SIDM replay changes canonical source interval");
      for(std::size_t i=0;i<sizeof(text_names)/sizeof(text_names[0]);i++)
        if(read_text(group,text_names[i])!=text_values[i])throw std::runtime_error("SIDM replay changes canonical frame or assets");
      return;
    }
    for(int index:{2,3}) {
      status(H5Adelete(datasets[index],"units"),"SIDM canonical velocity unit replacement failed");
      text(datasets[index],"units",velocity_units);
    }
    for(std::size_t i=0;i<sizeof(values)/sizeof(values[0]);i++)scalar(group,names[i],values[i]);
    for(std::size_t i=0;i<sizeof(text_names)/sizeof(text_names[0]);i++)text(group,text_names[i],text_values[i]);
  }
  void record_distributed_exchange(int rank,int ranks,std::uint64_t pairs,std::uint64_t ghosts,std::uint64_t rounds,
                                   std::uint64_t remote,std::uint64_t sent,std::uint64_t received,
                                   std::uint64_t completed_pair_operations=0,bool reference_schedule=false)
  {
    if(published || poisoned || file<0 || rank<0 || ranks<1 || rank>=ranks)
      throw std::runtime_error("SIDM native MPI source telemetry has invalid rank ownership");
    const char *names[9]={"mpi_rank","mpi_ranks","local_candidate_pairs","received_ghost_records",
        "matching_rounds","remote_endpoint_updates","exchange_sent_bytes","exchange_received_bytes","completed_pair_operations"};
    const std::uint64_t values[9]={std::uint64_t(rank),std::uint64_t(ranks),pairs,ghosts,rounds,remote,sent,received,completed_pair_operations};
    const char *schedule=reference_schedule?"single_rank_random_priority_all_pairs":"mutual_random_priority_endpoint_choices";
    if(reference_schedule && (rank!=0 || ranks!=1 || ghosts || rounds || remote || sent || received))
      throw std::runtime_error("SIDM reference schedule cannot claim distributed exchange");
    if(replay) {
      if(read_text(group,"pair_schedule")!=schedule)
        throw std::runtime_error("SIDM replay changes native MPI pair scheduling");
      for(int index=0;index<9;index++)if(read_text(group,names[index])!=std::to_string(values[index]))
        throw std::runtime_error("SIDM replay changes measured native MPI exchange telemetry");
      return;
    }
    text(group,"pair_schedule",schedule);
    for(int index=0;index<9;index++)text(group,names[index],std::to_string(values[index]));
  }
  void record_resource_measurements(bool cpu_available,double pair_cpu,double rare_cpu,double frequent_cpu,
                                    std::uint64_t rare_calls,std::uint64_t frequent_calls,bool rss_available,
                                    std::uint64_t peak_rss,std::uint64_t exchange_payload_peak,std::uint64_t angular_table_memory_bytes=0)
  {
    if(published || poisoned || file<0 || !std::isfinite(pair_cpu) || pair_cpu<0 ||
        !std::isfinite(rare_cpu) || rare_cpu<0 || !std::isfinite(frequent_cpu) || frequent_cpu<0)
      throw std::runtime_error("SIDM resource record has invalid measured CPU values");
    // Timings are not deterministic physics. Replaying a completed log keeps
    // its original producer's measurements, rather than asserting equality
    // or substituting timings from the replaying process.
    if(replay)return;
    text(group,"resource_measurement_schema","native-sidm-source-resources-v1");
    text(group,"resource_measurement_origin","original_completed_log_producer");
    text(group,"cpu_measurement_status",cpu_available?"MEASURED":"UNAVAILABLE");
    text(group,"rss_measurement_status",rss_available?"MEASURED":"UNAVAILABLE");
    text(group,"cpu_scope","calling_thread; pair dispatch and rare/frequent primitives include event-log writes");
    text(group,"rss_scope","process lifetime high-water through source-interval completion; not a simultaneous MPI aggregate");
    text(group,"exchange_memory_scope","largest single exchange vector capacity payload; allocator and MPI-private memory excluded");
    scalar(group,"native_pair_cpu_seconds",pair_cpu);
    scalar(group,"native_rare_cpu_seconds",rare_cpu);
    scalar(group,"native_frequent_cpu_seconds",frequent_cpu);
    text(group,"native_rare_operator_calls",std::to_string(rare_calls));
    text(group,"native_frequent_operator_calls",std::to_string(frequent_calls));
    text(group,"process_peak_rss_bytes",std::to_string(peak_rss));
    text(group,"exchange_buffer_payload_peak_bytes",std::to_string(exchange_payload_peak));
    text(group,"kernel_lookup_table_memory_bytes","0");
    text(group,"angular_lookup_table_memory_bytes",std::to_string(angular_table_memory_bytes));
    text(group,"table_memory_scope","retained differential velocity/angular/density vector capacity payload; excludes allocator and transient row interpolation storage");
  }
  void record_coarse_support(double fraction,double cumulative_omitted,double fraction_limit,double omitted_limit)
  {
    if(published || poisoned || file<0 || !std::isfinite(fraction) || fraction<0 || fraction>1 ||
        !std::isfinite(cumulative_omitted) || cumulative_omitted<0)
      throw std::runtime_error("SIDM native coarse-support record outside finite physical domain");
    const char *names[4]={"maximum_kernel_coarse_support_fraction","maximum_cumulative_omitted_scattering_upper_bound",
        "kernel_coarse_support_fraction_limit","cumulative_omitted_scattering_upper_bound_limit"};
    const double values[4]={fraction,cumulative_omitted,fraction_limit,omitted_limit};
    if(replay) {
      for(int index=0;index<4;index++)if(read_scalar(group,names[index])!=values[index])
        throw std::runtime_error("SIDM checkpoint replay changes actual coarse-support measurements");
      return;
    }
    text(group,"coarse_support_policy","coarse gravity retained; coarse scattering omitted only within explicit measured bounds");
    for(int index=0;index<4;index++)scalar(group,names[index],values[index]);
  }
  void finish()
  {
    if(published || poisoned || !model_declared || file<0) throw std::runtime_error("SIDM event log already closed");
    if(replay) {
      if(count!=replay_rows)throw std::runtime_error("SIDM checkpoint replay omits retained native events");
      if(read_text(group,"aggregate_applied_updates")!=std::to_string(observed_updates) ||
          read_scalar(group,"maximum_pair_energy_relative_residual")!=maximum_pair_energy_relative_residual)
        throw std::runtime_error("SIDM replay changes aggregate source-update measurements");
      close_handles();published=true;return;
    }
    text(group,"aggregate_applied_updates",std::to_string(observed_updates));
    text(group,"retained_audit_records",std::to_string(count));
    scalar(group,"maximum_pair_energy_relative_residual",maximum_pair_energy_relative_residual);
    status(H5Adelete(file,"publication_state"),"SIDM log state replacement failed");
    text(file,"publication_state","COMPLETE");
    status(H5Fflush(file,H5F_SCOPE_GLOBAL),"SIDM log flush failed");close_handles();
    int descriptor=open(partial_name.c_str(),O_RDONLY);
    if(descriptor<0) throw std::runtime_error("SIDM log durability open failed");
    int synced=fsync(descriptor);close(descriptor);
    if(synced!=0) throw std::runtime_error("SIDM log durability synchronization failed");
    // Exclusive hard-link publication cannot replace earlier evidence.
    if(link(partial_name.c_str(),final_name.c_str())!=0) throw std::runtime_error("SIDM log exclusive publication failed");
    if(unlink(partial_name.c_str())!=0) throw std::runtime_error("SIDM log temporary unlink failed");
    std::size_t slash=final_name.find_last_of('/');
    std::string parent=slash==std::string::npos?".":slash==0?"/":final_name.substr(0,slash);
    descriptor=open(parent.c_str(),O_RDONLY);
    if(descriptor<0) throw std::runtime_error("SIDM log publication directory open failed");
    synced=fsync(descriptor);close(descriptor);
    if(synced!=0) throw std::runtime_error("SIDM log publication directory synchronization failed");
    published=true;
  }
  std::uint64_t saved_records() const {return count;}
};
#endif
