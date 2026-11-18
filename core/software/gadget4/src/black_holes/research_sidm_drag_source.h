/* Research2 conservative BH/bath source reference, original implementation.
 * Units: gravitating Msun, proper kpc, peculiar km/s, cm/s^2 and proper s.
 * MPI reductions retain the actual distributed bath; no replicated census.
 * This kernel does not select a drag law or enable a production SIDM profile.
 */
#ifndef RESEARCH_SIDM_DRAG_SOURCE_H
#define RESEARCH_SIDM_DRAG_SOURCE_H
#include <mpi.h>
#include <array>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <climits>
#include <stdexcept>
#include "../sidm/sidm_random.h"
namespace research_sidm_drag_source {
using vec=std::array<double,3>;
using matrix=std::array<double,9>;
struct bath_particle {std::uint64_t id; double mass; vec position,velocity;};
struct limits {double fraction,inertia_ratio,relative_tolerance,momentum_floor,energy_floor,angular_floor;std::uint64_t maximum_particles,maximum_workspace_bytes;};
struct result {vec carrier_velocity,impulse,momentum_residual,angular_residual;double heat,energy_residual;std::vector<vec> local_velocities;};
inline vec add(vec a,const vec&b){for(int k=0;k<3;k++)a[k]+=b[k];return a;}
inline vec scale(vec a,double s){for(double &v:a)v*=s;return a;}
inline double dot(const vec&a,const vec&b){return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];}
inline double norm(const vec&a){return std::hypot(std::hypot(a[0],a[1]),a[2]);}
inline vec cross(const vec&a,const vec&b){return {{a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]}};}
inline bool finite(const vec&v){return std::isfinite(v[0])&&std::isfinite(v[1])&&std::isfinite(v[2]);}
inline void require_collective(bool okay,MPI_Comm comm,const char*message){
  int bad=okay?0:1,global=0;MPI_Allreduce(&bad,&global,1,MPI_INT,MPI_MAX,comm);if(global)throw std::runtime_error(message);
}
template<std::size_t N> inline std::array<double,N> sum(const std::array<double,N>&local,MPI_Comm comm){
  std::array<double,N> out{};MPI_Allreduce(local.data(),out.data(),N,MPI_DOUBLE,MPI_SUM,comm);return out;
}
inline matrix conditioned_inverse(matrix input,double minimum_ratio){
  // Symmetric Jacobi rotations resolve eigenvalues without an unchecked
  // singular inverse or a hidden pseudo-inverse geometry fallback.
  matrix diagonal=input,vectors{{1,0,0,0,1,0,0,0,1}};
  for(int iteration=0;iteration<64;iteration++){
    int p=0,q=1;double off=std::abs(diagonal[1]);
    for(auto pair: {std::array<int,2>{{0,2}},std::array<int,2>{{1,2}}})if(std::abs(diagonal[3*pair[0]+pair[1]])>off){p=pair[0];q=pair[1];off=std::abs(diagonal[3*p+q]);}
    double magnitude=std::max({std::abs(diagonal[0]),std::abs(diagonal[4]),std::abs(diagonal[8])});
    if(off<=32*2.2204460492503131e-16*magnitude)break;
    double angle=.5*std::atan2(2*diagonal[3*p+q],diagonal[3*q+q]-diagonal[3*p+p]),c=std::cos(angle),s=std::sin(angle);
    matrix rotation{{1,0,0,0,1,0,0,0,1}};rotation[3*p+p]=rotation[3*q+q]=c;rotation[3*p+q]=s;rotation[3*q+p]=-s;
    matrix temporary{},next{},new_vectors{};
    for(int i=0;i<3;i++)for(int j=0;j<3;j++)for(int k=0;k<3;k++)temporary[3*i+j]+=diagonal[3*i+k]*rotation[3*k+j];
    for(int i=0;i<3;i++)for(int j=0;j<3;j++)for(int k=0;k<3;k++)next[3*i+j]+=rotation[3*k+i]*temporary[3*k+j];
    for(int i=0;i<3;i++)for(int j=0;j<3;j++)for(int k=0;k<3;k++)new_vectors[3*i+j]+=vectors[3*i+k]*rotation[3*k+j];
    diagonal=next;vectors=new_vectors;
  }
  double low=std::min({diagonal[0],diagonal[4],diagonal[8]}),high=std::max({diagonal[0],diagonal[4],diagonal[8]});
  if(!(std::isfinite(low)&&std::isfinite(high)&&high>0&&low>minimum_ratio*high))throw std::runtime_error("Unsupported distributed BH bath inertia");
  double remaining=std::max({std::abs(diagonal[1]),std::abs(diagonal[2]),std::abs(diagonal[5])});
  if(remaining>128*2.2204460492503131e-16*high)throw std::runtime_error("BH inertia eigensystem did not converge");
  matrix inverse{};for(int i=0;i<3;i++)for(int j=0;j<3;j++)for(int k=0;k<3;k++)inverse[3*i+j]+=vectors[3*i+k]*vectors[3*j+k]/diagonal[3*k+k];
  return inverse;
}
inline vec apply(const matrix&m,const vec&v){vec out{};for(int i=0;i<3;i++)for(int j=0;j<3;j++)out[i]+=m[3*i+j]*v[j];return out;}
inline result kick(const std::vector<bath_particle>&bath,std::uint64_t carrier_id,double carrier_mass,
    const vec&carrier_position,const vec&carrier_velocity,const vec&acceleration,double dt,
    std::uint64_t seed,std::uint64_t counter,const limits&bound,MPI_Comm comm){
  bool okay=carrier_id!=0&&carrier_mass>0&&std::isfinite(carrier_mass)&&finite(carrier_position)&&finite(carrier_velocity)&&finite(acceleration)&&dt>0&&std::isfinite(dt);
  okay=okay&&bound.fraction>0&&bound.fraction<1&&bound.inertia_ratio>0&&bound.inertia_ratio<1&&bound.relative_tolerance>0&&bound.relative_tolerance<1;
  okay=okay&&bound.momentum_floor>0&&std::isfinite(bound.momentum_floor)&&bound.energy_floor>0&&std::isfinite(bound.energy_floor)&&bound.angular_floor>0&&std::isfinite(bound.angular_floor);
  for(const auto&p:bath)okay=okay&&p.id!=0&&p.id!=carrier_id&&p.mass>0&&std::isfinite(p.mass)&&finite(p.position)&&finite(p.velocity);
  require_collective(okay,comm,"Invalid distributed BH/bath source state or limits");
  std::array<double,17> replicated{{carrier_mass,carrier_position[0],carrier_position[1],carrier_position[2],carrier_velocity[0],carrier_velocity[1],carrier_velocity[2],acceleration[0],acceleration[1],acceleration[2],dt,bound.fraction,bound.inertia_ratio,bound.relative_tolerance,bound.momentum_floor,bound.energy_floor,bound.angular_floor}};
  std::array<double,17> minimum{},maximum{};MPI_Allreduce(replicated.data(),minimum.data(),17,MPI_DOUBLE,MPI_MIN,comm);MPI_Allreduce(replicated.data(),maximum.data(),17,MPI_DOUBLE,MPI_MAX,comm);
  require_collective(minimum==maximum,comm,"BH source ranks disagree on the carrier or source settings");
  std::array<unsigned long long,5> keys{{carrier_id,seed,counter,bound.maximum_particles,bound.maximum_workspace_bytes}},key_min{},key_max{};
  MPI_Allreduce(keys.data(),key_min.data(),5,MPI_UNSIGNED_LONG_LONG,MPI_MIN,comm);MPI_Allreduce(keys.data(),key_max.data(),5,MPI_UNSIGNED_LONG_LONG,MPI_MAX,comm);
  require_collective(key_min==key_max,comm,"BH source ranks disagree on exact identities/counters or resource limits");
  unsigned long long count=bath.size(),global_count=0;MPI_Allreduce(&count,&global_count,1,MPI_UNSIGNED_LONG_LONG,MPI_SUM,comm);
  int ranks=0;MPI_Comm_size(comm,&ranks);
  require_collective(global_count>=3&&global_count<=bound.maximum_particles&&global_count<=INT_MAX&&32*global_count+128*count+32*static_cast<unsigned long long>(ranks)<=bound.maximum_workspace_bytes,comm,"Distributed BH bath exceeds its census/workspace bound");
  // Replicate only bounded exact identities to reject cross-rank aliases;
  // masses, positions, velocities and source updates stay distributed.
  std::vector<int> sizes,offsets;std::vector<unsigned long long> local_ids,all_ids;bool identity_allocated=true;
  try{sizes.resize(ranks);offsets.resize(ranks);local_ids.resize(bath.size());all_ids.resize(global_count);}catch(const std::exception&){identity_allocated=false;}
  require_collective(identity_allocated,comm,"BH source identity workspace allocation failed");
  for(std::size_t j=0;j<bath.size();j++)local_ids[j]=bath[j].id;
  int local_count=static_cast<int>(count);MPI_Allgather(&local_count,1,MPI_INT,sizes.data(),1,MPI_INT,comm);
  for(int rank=1;rank<ranks;rank++)offsets[rank]=offsets[rank-1]+sizes[rank-1];
  MPI_Allgatherv(local_ids.data(),local_count,MPI_UNSIGNED_LONG_LONG,all_ids.data(),sizes.data(),offsets.data(),MPI_UNSIGNED_LONG_LONG,comm);
  std::sort(all_ids.begin(),all_ids.end());require_collective(std::adjacent_find(all_ids.begin(),all_ids.end())==all_ids.end(),comm,"BH bath identities alias across native MPI owners");
  std::array<double,7> local{};
  for(const auto&p:bath){local[0]+=p.mass;vec r=add(p.position,scale(carrier_position,-1));for(int k=0;k<3;k++){local[1+k]+=p.mass*p.velocity[k];local[4+k]+=p.mass*r[k];}}
  auto global=sum(local,comm);double mass=global[0];vec mean{},center{};for(int k=0;k<3;k++){mean[k]=global[1+k]/mass;center[k]=global[4+k]/mass;}
  require_collective(mass>0&&std::isfinite(mass)&&finite(mean)&&finite(center),comm,"Distributed bath moment overflow");
  vec relative=add(carrier_velocity,scale(mean,-1)),delta=scale(acceleration,dt/1e5),impulse=scale(delta,carrier_mass);
  double speed=norm(relative);require_collective(dot(delta,relative)<=0&&(speed>0?norm(delta)*(1+carrier_mass/mass)<=bound.fraction*speed:norm(delta)==0),comm,"BH drag source requires a smaller timestep or supported relative state");
  matrix local_inertia{};for(const auto&p:bath){vec R=add(add(p.position,scale(carrier_position,-1)),scale(center,-1));double rr=dot(R,R);for(int i=0;i<3;i++)for(int j=0;j<3;j++)local_inertia[3*i+j]+=p.mass*((i==j?rr:0)-R[i]*R[j]);}
  matrix inertia=sum(local_inertia,comm),inverse=conditioned_inverse(inertia,bound.inertia_ratio);
  vec rotation=apply(inverse,cross(center,impulse)),com=scale(add(scale(carrier_velocity,carrier_mass),scale(mean,mass)),1/(carrier_mass+mass));
  result out{};out.carrier_velocity=add(carrier_velocity,delta);out.impulse=impulse;
  std::vector<vec> kicks;bool allocated=true;
  try{out.local_velocities.resize(bath.size());kicks.resize(bath.size());}catch(const std::exception&){allocated=false;}
  require_collective(allocated,comm,"Distributed bath source workspace allocation failed");
  std::array<double,2> work{};
  for(std::size_t j=0;j<bath.size();j++){const auto&p=bath[j];vec R=add(add(p.position,scale(carrier_position,-1)),scale(center,-1)),dv=add(scale(impulse,-1/mass),cross(rotation,R));out.local_velocities[j]=add(p.velocity,dv);work[0]+=p.mass*dot(add(p.velocity,scale(com,-1)),dv);work[1]+=.5*p.mass*dot(dv,dv);}
  auto energy=sum(work,comm);energy[0]+=carrier_mass*dot(add(carrier_velocity,scale(com,-1)),delta);energy[1]+=.5*carrier_mass*dot(delta,delta);out.heat=-energy[0]-energy[1];
  require_collective(std::isfinite(out.heat)&&out.heat>=0,comm,"Torque-balanced BH source is not dissipative");
  if(out.heat>0){
    vec local_mean{};for(std::size_t j=0;j<bath.size();j++){for(int k=0;k<3;k++)kicks[j][k]=2*research_sidm::pair_uniform(seed,carrier_id,bath[j].id,counter,0,static_cast<std::uint64_t>(research_sidm::draw_purpose::bh_drag_heat_x)+k)-1;local_mean=add(local_mean,scale(kicks[j],bath[j].mass));}
    vec kick_mean=scale(sum(local_mean,comm),1/mass),local_torque{};
    for(std::size_t j=0;j<bath.size();j++){vec R=add(add(bath[j].position,scale(carrier_position,-1)),scale(center,-1));kicks[j]=add(kicks[j],scale(kick_mean,-1));local_torque=add(local_torque,scale(cross(R,kicks[j]),bath[j].mass));}
    vec spin=apply(inverse,sum(local_torque,comm));std::array<double,2> quadratic{};
    for(std::size_t j=0;j<bath.size();j++){vec R=add(add(bath[j].position,scale(carrier_position,-1)),scale(center,-1));kicks[j]=add(kicks[j],scale(cross(spin,R),-1));quadratic[0]+=.5*bath[j].mass*dot(kicks[j],kicks[j]);quadratic[1]+=bath[j].mass*dot(add(out.local_velocities[j],scale(com,-1)),kicks[j]);}
    auto q=sum(quadratic,comm);require_collective(q[0]>0&&std::isfinite(q[0])&&std::isfinite(q[1]),comm,"Bath lacks constrained heat degrees of freedom");
    double radical=std::hypot(q[1],2*std::sqrt(q[0])*std::sqrt(out.heat)),amplitude=q[1]>=0?2*out.heat/(radical+q[1]):(radical-q[1])/(2*q[0]);
    for(std::size_t j=0;j<bath.size();j++)out.local_velocities[j]=add(out.local_velocities[j],scale(kicks[j],amplitude));
  }
  std::array<double,10> residual{};okay=finite(out.carrier_velocity);
  for(std::size_t j=0;j<bath.size();j++){const auto&p=bath[j];vec dv=add(out.local_velocities[j],scale(p.velocity,-1)),r=add(p.position,scale(carrier_position,-1)),angular=scale(cross(r,dv),p.mass);okay=okay&&finite(out.local_velocities[j]);for(int k=0;k<3;k++){residual[k]+=p.mass*dv[k];residual[3+k]+=angular[k];}residual[6]+=p.mass*dot(add(p.velocity,scale(com,-1)),dv)+.5*p.mass*dot(dv,dv);residual[7]+=p.mass*norm(dv);residual[8]+=norm(angular);}
  auto totals=sum(residual,comm);vec actual=add(out.carrier_velocity,scale(carrier_velocity,-1));for(int k=0;k<3;k++){out.momentum_residual[k]=totals[k]+carrier_mass*actual[k];out.angular_residual[k]=totals[3+k];}
  out.energy_residual=totals[6]+carrier_mass*dot(add(carrier_velocity,scale(com,-1)),actual)+.5*carrier_mass*dot(actual,actual);
  double ps=std::max({bound.momentum_floor,norm(impulse),totals[7]}),ls=std::max(bound.angular_floor,totals[8]),es=std::max({bound.energy_floor,out.heat,std::abs(energy[0]),energy[1]});
  require_collective(okay&&finite(out.momentum_residual)&&finite(out.angular_residual)&&std::isfinite(out.energy_residual)&&norm(out.momentum_residual)<=bound.relative_tolerance*ps&&norm(out.angular_residual)<=bound.relative_tolerance*ls&&std::abs(out.energy_residual)<=bound.relative_tolerance*es,comm,"Rounded native BH/bath source exceeds conservation budgets");
  return out;
}
}
#endif
