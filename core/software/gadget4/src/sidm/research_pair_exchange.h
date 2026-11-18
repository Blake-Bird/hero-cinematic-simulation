/* First-party bounded distributed SIDM pair scheduling.
 * Endpoint owners select their smallest pending canonical pair. Only mutual
 * choices execute, so no particle receives concurrent pair updates. Current
 * physical endpoint states are supplied by their actual owners each round.
 * Neighbor discovery belongs to the spatial producer, not this scheduler.
 * MPI wire records require the same native binary/ABI across communicator.
 */
#ifndef RESEARCH_SIDM_PAIR_EXCHANGE_H
#define RESEARCH_SIDM_PAIR_EXCHANGE_H
#include "research_rare_pair.h"
#include <mpi.h>
#include <climits>
#include <map>
#include <set>
#include <tuple>
#include <type_traits>
#include <vector>

namespace research_sidm {
struct pair_exchange_limits
{
  std::uint64_t maximum_local_pairs,maximum_rounds,maximum_exchange_records,memory_bytes;
};
struct distributed_pair
{
  std::uint64_t left_id,right_id;
  int left_owner,right_owner;
  double kernel_overlap_inverse_cm3;
  double left_mass_msun,right_mass_msun;
  int left_type,right_type;
};
struct distributed_pair_effect
{
  pair_particle left,right;
  double optical_depth=0;
  std::uint64_t rare_events=0,frequent_updates=0;
};
struct pair_exchange_measurements
{
  std::uint64_t matching_rounds=0,completed_local_pairs=0,remote_endpoint_updates=0;
  std::uint64_t sent_bytes=0,received_bytes=0,buffer_payload_peak_bytes=0;
};
inline void require_collective(MPI_Comm communicator,bool local_condition,const char *message)
{
  int local=local_condition?1:0,global=0;
  if(MPI_Allreduce(&local,&global,1,MPI_INT,MPI_MIN,communicator)!=MPI_SUCCESS)
    throw std::runtime_error("SIDM collective domain agreement failed");
  if(!global)throw std::runtime_error(message);
}
template<class Record> struct received_pair_records
{
  std::vector<Record> values;
  std::vector<int> origins;
};
template<class Record>
received_pair_records<Record> exchange_pair_records(MPI_Comm communicator,const std::vector<std::pair<int,Record>> &outgoing,
                                                    const pair_exchange_limits &limits,pair_exchange_measurements &measurements)
{
  static_assert(std::is_trivially_copyable<Record>::value,"SIDM MPI records must contain no owning pointers");
  int ranks=0;MPI_Comm_size(communicator,&ranks);
  bool valid=ranks>0 && outgoing.size()<=limits.maximum_exchange_records &&
      outgoing.size()<=std::size_t(INT_MAX)/sizeof(Record) &&
      limits.memory_bytes/8>=outgoing.size()*sizeof(Record)+std::uint64_t(ranks)*sizeof(int)*4;
  for(const auto &row:outgoing)valid=valid && row.first>=0 && row.first<ranks;
  require_collective(communicator,valid,"SIDM outgoing exchange exceeds declared rank, record or workspace bounds");
  std::vector<int> send_counts(ranks,0),receive_counts(ranks,0),send_offsets(ranks,0),receive_offsets(ranks,0);
  for(const auto &row:outgoing)send_counts[row.first]+=sizeof(Record);
  if(MPI_Alltoall(send_counts.data(),1,MPI_INT,receive_counts.data(),1,MPI_INT,communicator)!=MPI_SUCCESS)
    throw std::runtime_error("SIDM pair exchange census failed");
  std::uint64_t sent=0,received=0;
  valid=true;
  for(int rank=0;rank<ranks;rank++) {
    valid=valid && receive_counts[rank]>=0 && receive_counts[rank]%sizeof(Record)==0;
    if(sent<=INT_MAX)send_offsets[rank]=int(sent);else valid=false;
    if(received<=INT_MAX)receive_offsets[rank]=int(received);else valid=false;
    sent+=send_counts[rank];received+=std::uint64_t(std::max(0,receive_counts[rank]));
  }
  valid=valid && sent<=INT_MAX && received<=INT_MAX && received/sizeof(Record)<=limits.maximum_exchange_records &&
      limits.memory_bytes/8>=sent+received+std::uint64_t(ranks)*sizeof(int)*4;
  require_collective(communicator,valid,"SIDM incoming exchange exceeds its frozen record or memory allocation");
  std::vector<Record> packed(sent/sizeof(Record));
  std::vector<int> cursors=send_offsets;
  for(const auto &row:outgoing) {packed[cursors[row.first]/sizeof(Record)]=row.second;cursors[row.first]+=sizeof(Record);}
  received_pair_records<Record> result;
  result.values.resize(received/sizeof(Record));result.origins.resize(result.values.size());
  // Actual vector capacity payload retained together at the exchange boundary;
  // excludes allocator headers, maps, MPI-private buffers and caller workspaces.
  const std::uint64_t buffer_payload=std::uint64_t(outgoing.capacity())*sizeof(std::pair<int,Record>)+
      std::uint64_t(packed.capacity()+result.values.capacity())*sizeof(Record)+
      std::uint64_t(result.origins.capacity()+send_counts.capacity()+receive_counts.capacity()+
          send_offsets.capacity()+receive_offsets.capacity()+cursors.capacity())*sizeof(int);
  measurements.buffer_payload_peak_bytes=std::max(measurements.buffer_payload_peak_bytes,buffer_payload);
  if(MPI_Alltoallv(packed.data(),send_counts.data(),send_offsets.data(),MPI_BYTE,result.values.data(),
      receive_counts.data(),receive_offsets.data(),MPI_BYTE,communicator)!=MPI_SUCCESS)
    throw std::runtime_error("SIDM bounded pair record exchange failed");
  for(int rank=0;rank<ranks;rank++)for(std::size_t index=receive_offsets[rank]/sizeof(Record);
      index<std::size_t(receive_offsets[rank]+receive_counts[rank])/sizeof(Record);index++)result.origins[index]=rank;
  require_collective(communicator,sent<=UINT64_MAX-measurements.sent_bytes && received<=UINT64_MAX-measurements.received_bytes,
      "SIDM pair exchange telemetry counter exhausted");
  measurements.sent_bytes+=sent;measurements.received_bytes+=received;
  return result;
}
inline std::uint64_t distributed_pair_attempt_key(std::uint64_t left,std::uint64_t right,std::uint64_t source_substep)
{
  return mix(left)^mix(right+UINT64_C(0x2718281828459045))^mix(source_substep);
}

template<class ReadOwner,class ApplyPair,class CommitOwner>
pair_exchange_measurements execute_distributed_pairs(MPI_Comm communicator,const std::vector<distributed_pair> &pairs,
                                                     const pair_exchange_limits &limits,ReadOwner read_owner,
                                                     ApplyPair apply_pair,CommitOwner commit_owner,
                                                     std::uint64_t seed=0,std::uint64_t interval=0,std::uint64_t substep=0)
{
  int ranks=0,rank=0;MPI_Comm_size(communicator,&ranks);MPI_Comm_rank(communicator,&rank);
  pair_exchange_measurements measurements;
  bool valid=limits.maximum_rounds>0 && limits.maximum_exchange_records>0 &&
      pairs.size()<=limits.maximum_local_pairs && limits.memory_bytes/256>=pairs.size();
  for(const auto &pair:pairs)valid=valid && pair.left_id<pair.right_id && pair.left_owner>=0 && pair.left_owner<ranks &&
      pair.right_owner>=0 && pair.right_owner<ranks && std::isfinite(pair.kernel_overlap_inverse_cm3) && pair.kernel_overlap_inverse_cm3>0;
  require_collective(communicator,valid,"SIDM distributed pair matrix changes canonical identity, owner or allocation");
  // Establish exactly one worker per unordered pair before applying updates.
  struct identity_record {std::uint64_t left,right;};
  {
  std::vector<std::pair<int,identity_record>> identities;
  identities.reserve(pairs.size());
  for(const auto &pair:pairs)identities.push_back({int((mix(pair.left_id)^mix(pair.right_id))%std::uint64_t(ranks)),
      {pair.left_id,pair.right_id}});
  const auto directory=exchange_pair_records(communicator,identities,limits,measurements);
  std::set<std::pair<std::uint64_t,std::uint64_t>> unique;
  valid=true;for(const auto &identity:directory.values)valid=unique.insert({identity.left,identity.right}).second && valid;
  require_collective(communicator,valid,"SIDM unordered pair is duplicated across native work ranks");
  }
  std::vector<unsigned char> completed(pairs.size(),0);
  struct proposal {std::uint64_t particle,left,right,index,priority;int worker;};
  struct response {std::uint64_t index,particle;int selected;pair_particle state;};
  struct mutation {std::uint64_t particle,left,right,index,rare_events,frequent_updates;int worker;
                   pair_particle state;double optical_depth;};
  for(std::uint64_t round=0;;round++) {
    int local_pending=std::find(completed.begin(),completed.end(),0)!=completed.end(),global_pending=0;
    if(MPI_Allreduce(&local_pending,&global_pending,1,MPI_INT,MPI_MAX,communicator)!=MPI_SUCCESS)
      throw std::runtime_error("SIDM distributed matching progress agreement failed");
    if(!global_pending)break;
    require_collective(communicator,round<limits.maximum_rounds,"SIDM distributed matching round allocation exhausted");
    std::vector<std::pair<int,proposal>> outgoing;
    require_collective(communicator,pairs.size()<=limits.maximum_exchange_records/2,"SIDM endpoint proposals exceed exchange allocation");
    outgoing.reserve(2*pairs.size());
    for(std::size_t index=0;index<pairs.size();index++)if(!completed[index]) {
      const auto &pair=pairs[index];
      const auto priority=pair_random_bits(seed,pair.left_id,pair.right_id,interval,substep,std::uint64_t(draw_purpose::pair_priority));
      outgoing.push_back({pair.left_owner,{pair.left_id,pair.left_id,pair.right_id,index,priority,rank}});
      outgoing.push_back({pair.right_owner,{pair.right_id,pair.left_id,pair.right_id,index,priority,rank}});
    }
    const auto incoming=exchange_pair_records(communicator,outgoing,limits,measurements);
    std::map<std::uint64_t,proposal> choices;
    valid=true;
    for(std::size_t index=0;index<incoming.values.size();index++) {
      const auto &row=incoming.values[index];
      valid=valid && row.worker==incoming.origins[index] && row.left<row.right && (row.particle==row.left || row.particle==row.right);
      const auto old=choices.find(row.particle);
      if(old==choices.end() || std::tie(row.priority,row.left,row.right)<std::tie(old->second.priority,old->second.left,old->second.right))choices[row.particle]=row;
    }
    require_collective(communicator,valid,"SIDM endpoint proposal ownership or canonical pair differs");
    std::vector<std::pair<int,response>> replies;
    replies.reserve(incoming.values.size());
    for(const auto &row:incoming.values) {
      const auto &chosen=choices.at(row.particle);
      const bool selected=chosen.left==row.left && chosen.right==row.right;
      pair_particle state{};
      if(selected) {state=read_owner(row.particle);if(state.id!=row.particle)throw std::runtime_error("SIDM current endpoint is absent from its declared native owner");}
      replies.push_back({row.worker,{row.index,row.particle,selected?1:0,state}});
    }
    const auto returned=exchange_pair_records(communicator,replies,limits,measurements);
    struct selection {int left=0,right=0;pair_particle l{},r{};};
    std::vector<selection> selections(pairs.size());
    valid=true;
    for(std::size_t index=0;index<returned.values.size();index++) {
      const auto &row=returned.values[index];
      if(row.index>=pairs.size() || completed[row.index] || (row.selected!=0 && row.selected!=1)) {valid=false;continue;}
      const auto &pair=pairs[row.index];auto &choice=selections[row.index];
      if(row.particle==pair.left_id && returned.origins[index]==pair.left_owner && !choice.left) {
        choice.left=row.selected?2:1;choice.l=row.state;
      } else if(row.particle==pair.right_id && returned.origins[index]==pair.right_owner && !choice.right) {
        choice.right=row.selected?2:1;choice.r=row.state;
      } else valid=false;
    }
    for(std::size_t index=0;index<pairs.size();index++)if(!completed[index])valid=valid && selections[index].left && selections[index].right;
    require_collective(communicator,valid,"SIDM current endpoint response is duplicated, missing or supplied by another owner");
    std::vector<std::pair<int,mutation>> updates;
    std::vector<std::size_t> accepted;
    for(std::size_t index=0;index<pairs.size();index++)if(!completed[index] && selections[index].left==2 && selections[index].right==2) {
      const auto &pair=pairs[index];const auto &selection=selections[index];
      auto effect=apply_pair(pair,selection.l,selection.r);
      if(effect.left.id!=pair.left_id || effect.right.id!=pair.right_id || effect.left.mass_msun!=selection.l.mass_msun ||
          effect.right.mass_msun!=selection.r.mass_msun || !std::isfinite(effect.optical_depth) || effect.optical_depth<0 ||
          effect.rare_events>1 || effect.frequent_updates>2)
        throw std::runtime_error("SIDM applied pair changes native identity/mass or supplies invalid exposure");
      updates.push_back({pair.left_owner,{pair.left_id,pair.left_id,pair.right_id,index,effect.rare_events,effect.frequent_updates,
          rank,effect.left,effect.optical_depth}});
      updates.push_back({pair.right_owner,{pair.right_id,pair.left_id,pair.right_id,index,effect.rare_events,effect.frequent_updates,
          rank,effect.right,effect.optical_depth}});
      accepted.push_back(index);
    }
    const auto commits=exchange_pair_records(communicator,updates,limits,measurements);
    std::set<std::uint64_t> touched;
    valid=true;
    for(std::size_t index=0;index<commits.values.size();index++) {
      const auto &row=commits.values[index];const auto choice=choices.find(row.particle);
      valid=valid && touched.insert(row.particle).second && row.worker==commits.origins[index] && choice!=choices.end();
      if(choice!=choices.end())valid=valid && row.worker==choice->second.worker && row.index==choice->second.index &&
          row.left==choice->second.left && row.right==choice->second.right && row.state.id==row.particle;
      for(double velocity:row.state.velocity_kms)valid=valid && std::isfinite(velocity);
    }
    require_collective(communicator,valid,"SIDM simultaneous native commits conflict or differ from mutually chosen pairs");
    for(const auto &row:commits.values) {
      commit_owner(row.state,row.optical_depth,row.rare_events,row.frequent_updates);
      if(row.worker!=rank)++measurements.remote_endpoint_updates;
    }
    for(std::size_t index:accepted)completed[index]=1;
    measurements.completed_local_pairs+=accepted.size();++measurements.matching_rounds;
    int local_progress=!accepted.empty(),global_progress=0;
    if(MPI_Allreduce(&local_progress,&global_progress,1,MPI_INT,MPI_MAX,communicator)!=MPI_SUCCESS)
      throw std::runtime_error("SIDM pair commit progress reduction failed");
    if(!global_progress)throw std::runtime_error("SIDM distributed canonical matching made no progress");
  }
  return measurements;
}
}
#endif
