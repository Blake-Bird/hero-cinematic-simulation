/* Bounded collective preparation for explicit seed reservoir transactions.
 * The caller supplies a proper-aperture selection and consistent physical
 * units. No rank may mutate particles before this collective returns a
 * supported plan and all ranks have validated their local commit state. */
#ifndef RESEARCH_BH_SEED_COLLECTIVE_H
#define RESEARCH_BH_SEED_COLLECTIVE_H
#include "research_bh_seed_reservoir.h"
#include <mpi.h>
#include <climits>
#include <type_traits>

namespace research_bh_seed {
struct collective_plan {
  gas carrier{};
  std::vector<gas> records;
  plan assembly;
};
inline void collective_require(bool local_ok,MPI_Comm communicator,const char *message) {
  int ok=local_ok ? 1 : 0,global_ok=0;
  if(MPI_Allreduce(&ok,&global_ok,1,MPI_INT,MPI_MIN,communicator)!=MPI_SUCCESS)
    throw std::runtime_error("Seed reservoir collective validation failed");
  if(!global_ok)throw std::invalid_argument(message);
}
inline collective_plan prepare_collective(const std::vector<gas> &local,
    std::uint64_t carrier_id,double seed_mass,double maximum_donor_fraction,
    std::size_t maximum_records,MPI_Comm communicator) {
  static_assert(std::is_trivially_copyable<gas>::value,"Seed MPI records require plain value storage");
  int ranks=0;MPI_Comm_size(communicator,&ranks);
  // Bounds include the carrier. Reserve at least 1024 bytes per record in the
  // caller's memory preflight for gathered values, sorted copies and ledgers.
  bool valid=carrier_id>0 && maximum_records>0 && maximum_records<=std::size_t(INT_MAX)/sizeof(gas) &&
    local.size()<=maximum_records && ranks>0 && std::size_t(ranks)<=maximum_records &&
    std::isfinite(seed_mass) && seed_mass>0 && std::isfinite(maximum_donor_fraction) &&
    maximum_donor_fraction>0 && maximum_donor_fraction<=.1;
  try {for(const auto &p:local){validate(p);if(p.id==0)valid=false;}}
  catch(const std::exception &){valid=false;}
  collective_require(valid,communicator,"Invalid or oversized local seed reservoir input");
  // The recipe is replicated, but check rather than assume that every rank
  // received the same identity, limits and physical parameters.
  unsigned long long recipe[2]={static_cast<unsigned long long>(carrier_id),static_cast<unsigned long long>(maximum_records)},lo[2],hi[2];
  MPI_Allreduce(recipe,lo,2,MPI_UNSIGNED_LONG_LONG,MPI_MIN,communicator);
  MPI_Allreduce(recipe,hi,2,MPI_UNSIGNED_LONG_LONG,MPI_MAX,communicator);
  double parameters[2]={seed_mass,maximum_donor_fraction},pmin[2],pmax[2];
  MPI_Allreduce(parameters,pmin,2,MPI_DOUBLE,MPI_MIN,communicator);
  MPI_Allreduce(parameters,pmax,2,MPI_DOUBLE,MPI_MAX,communicator);
  collective_require(lo[0]==hi[0] && lo[1]==hi[1] && pmin[0]==pmax[0] && pmin[1]==pmax[1],
                     communicator,"MPI ranks disagree about seed reservoir recipe");
  int bytes=int(local.size()*sizeof(gas));
  std::vector<int> sizes(ranks),offsets(ranks);
  if(MPI_Allgather(&bytes,1,MPI_INT,sizes.data(),1,MPI_INT,communicator)!=MPI_SUCCESS)
    throw std::runtime_error("Seed reservoir count exchange failed");
  std::size_t total=0;
  for(int rank=0;rank<ranks;rank++) {
    if(sizes[rank]<0 || sizes[rank]%sizeof(gas)!=0 || total+std::size_t(sizes[rank])>maximum_records*sizeof(gas))
      throw std::invalid_argument("Global seed reservoir exceeds declared allocation");
    offsets[rank]=int(total);total+=sizes[rank];
  }
  collective_plan result;result.records.resize(total/sizeof(gas));
  if(MPI_Allgatherv(local.data(),bytes,MPI_BYTE,result.records.data(),sizes.data(),offsets.data(),MPI_BYTE,communicator)!=MPI_SUCCESS)
    throw std::runtime_error("Seed reservoir value exchange failed");
  std::vector<gas> donors;donors.reserve(result.records.size());int carrier_count=0;
  for(const auto &p:result.records) {
    if(p.id==carrier_id){result.carrier=p;++carrier_count;}else donors.push_back(p);
  }
  if(carrier_count!=1)throw std::invalid_argument("Seed reservoir requires exactly one live carrier owner");
  result.assembly=prepare(result.carrier,std::move(donors),seed_mass,maximum_donor_fraction,maximum_records);
  return result;
}
inline bool same_gas(const gas &a,const gas &b) {
  return a.id==b.id && a.mass==b.mass && a.specific_internal_energy==b.specific_internal_energy &&
    a.offset==b.offset && a.velocity==b.velocity && a.mass_fractions==b.mass_fractions;
}
inline void validate_collective_commit(const collective_plan &prepared,
    const std::vector<gas> &current_local,MPI_Comm communicator) {
  collective_require(prepared.assembly.supported && !prepared.records.empty() &&
      prepared.records.size()<=std::size_t(INT_MAX),communicator,
      "Unsupported seed reservoir transaction cannot commit");
  // All ranks validate the same complete aperture census, including unused
  // donors. A changed composition, thermal state, velocity or owner count must
  // be discovered before the first destructive particle update.
  std::vector<const gas*> ordered;ordered.reserve(prepared.records.size());
  for(const auto &record:prepared.records)ordered.push_back(&record);
  std::sort(ordered.begin(),ordered.end(),[](const gas *a,const gas *b){return a->id<b->id;});
  std::vector<int> seen(ordered.size(),0),global_seen(ordered.size(),0);bool valid=true;
  for(const auto &current:current_local) {
    auto found=std::lower_bound(ordered.begin(),ordered.end(),current.id,
      [](const gas *a,std::uint64_t id){return a->id<id;});
    if(found==ordered.end() || (*found)->id!=current.id || !same_gas(**found,current)) {valid=false;continue;}
    const auto index=std::size_t(found-ordered.begin());
    if(seen[index]!=0)valid=false;else seen[index]=1;
  }
  collective_require(valid,communicator,"Live gas changed after seed reservoir preparation");
  if(MPI_Allreduce(seen.data(),global_seen.data(),int(seen.size()),MPI_INT,MPI_SUM,communicator)!=MPI_SUCCESS)
    throw std::runtime_error("Seed reservoir commit ownership exchange failed");
  collective_require(std::all_of(global_seen.begin(),global_seen.end(),[](int n){return n==1;}),
                     communicator,"Seed reservoir commit requires exactly one live owner per gas ID");
}
}
#endif
