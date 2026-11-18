/* Distributed spatial cell ghosting for compact native SIDM pair support.
 * A particle is sent only to its cell and adjacent cells. The worker owning
 * the lower-ID endpoint's primary cell creates the canonical pair once.
 * Velocity updates remain with native owners through research_pair_exchange.
 */
#ifndef RESEARCH_SIDM_SPATIAL_PAIRS_H
#define RESEARCH_SIDM_SPATIAL_PAIRS_H
#include "research_pair_exchange.h"
#include "research_kernel_overlap.h"
#include <array>

namespace research_sidm {
struct spatial_pair_particle
{
  std::uint64_t id;
  int owner;
  std::array<double,3> proper_position_cm;
  double proper_support_radius_cm;
  double mass_msun;
  int native_type;
};
struct spatial_pair_products
{
  std::vector<distributed_pair> pairs;
  pair_exchange_measurements exchange;
  std::uint64_t received_ghost_records=0,examined_neighbors=0;
};
inline spatial_pair_products build_spatial_pairs(MPI_Comm communicator,const std::vector<spatial_pair_particle> &local,
                                               double maximum_radius_cm,double periodic_box_cm,int kernel_kind,
                                               double kernel_origin_bound,std::uint64_t maximum_neighbor_visits,
                                               const pair_exchange_limits &limits)
{
  int ranks=0,rank=0;MPI_Comm_size(communicator,&ranks);MPI_Comm_rank(communicator,&rank);
  bool valid=std::isfinite(maximum_radius_cm) && maximum_radius_cm>0 && std::isfinite(periodic_box_cm) &&
      periodic_box_cm>=0 && maximum_neighbor_visits>0 && limits.memory_bytes/256>=local.size();
  for(const auto &particle:local) {
    valid=valid && particle.owner==rank && std::isfinite(particle.proper_support_radius_cm) &&
        particle.proper_support_radius_cm>0 && particle.proper_support_radius_cm<=maximum_radius_cm &&
        std::isfinite(particle.mass_msun) && particle.mass_msun>0 && (particle.native_type==1 || particle.native_type==2);
    for(double position:particle.proper_position_cm)valid=valid && std::isfinite(position);
  }
  require_collective(communicator,valid,"SIDM spatial ghost input changes native owner, physical support or workspace");
  double minimum_radius=0,maximum_radius=0,minimum_box=0,maximum_box=0;
  MPI_Allreduce(&maximum_radius_cm,&minimum_radius,1,MPI_DOUBLE,MPI_MIN,communicator);
  MPI_Allreduce(&maximum_radius_cm,&maximum_radius,1,MPI_DOUBLE,MPI_MAX,communicator);
  MPI_Allreduce(&periodic_box_cm,&minimum_box,1,MPI_DOUBLE,MPI_MIN,communicator);
  MPI_Allreduce(&periodic_box_cm,&maximum_box,1,MPI_DOUBLE,MPI_MAX,communicator);
  require_collective(communicator,minimum_radius==maximum_radius && minimum_box==maximum_box,
      "SIDM spatial ranks disagree about physical kernel support or periodic geometry");
  int minimum_kind=0,maximum_kind=0;
  double minimum_origin_bound=0,maximum_origin_bound=0;
  MPI_Allreduce(&kernel_kind,&minimum_kind,1,MPI_INT,MPI_MIN,communicator);
  MPI_Allreduce(&kernel_kind,&maximum_kind,1,MPI_INT,MPI_MAX,communicator);
  MPI_Allreduce(&kernel_origin_bound,&minimum_origin_bound,1,MPI_DOUBLE,MPI_MIN,communicator);
  MPI_Allreduce(&kernel_origin_bound,&maximum_origin_bound,1,MPI_DOUBLE,MPI_MAX,communicator);
  require_collective(communicator,minimum_kind==maximum_kind && (kernel_kind==0 || kernel_kind==1) &&
      std::isfinite(kernel_origin_bound) && kernel_origin_bound>0 && kernel_origin_bound<1 &&
      minimum_origin_bound==maximum_origin_bound,"SIDM ranks disagree about the actual scattering kernel definition");
  double width=2*maximum_radius_cm;
  std::int64_t periodic_cells=0;
  if(periodic_box_cm>0) {
    const double cells=std::floor(periodic_box_cm/width);
    require_collective(communicator,cells>=4 && cells<double(INT64_MAX/2),
        "SIDM periodic spatial support does not fit the unique neighbor-cell domain");
    periodic_cells=std::int64_t(cells);width=periodic_box_cm/periodic_cells;
  }
  spatial_pair_products result;
  struct ghost {spatial_pair_particle particle;std::array<std::int64_t,3> primary,target;};
  const auto worker=[ranks](const std::array<std::int64_t,3> &cell) {
    return int((mix(std::uint64_t(cell[0]))^mix(std::uint64_t(cell[1])+UINT64_C(0x1618033988749894))^
        mix(std::uint64_t(cell[2])+UINT64_C(0x1414213562373095)))%std::uint64_t(ranks));
  };
  // Establish global ID uniqueness before ghost copies can obscure ownership.
  {
    struct identity {std::uint64_t id;int owner;};
    std::vector<std::pair<int,identity>> outgoing;
    outgoing.reserve(local.size());
    for(const auto &particle:local)outgoing.push_back({int(mix(particle.id)%std::uint64_t(ranks)),{particle.id,rank}});
    const auto incoming=exchange_pair_records(communicator,outgoing,limits,result.exchange);
    std::set<std::uint64_t> ids;
    valid=true;for(std::size_t index=0;index<incoming.values.size();index++)valid=
        ids.insert(incoming.values[index].id).second && incoming.values[index].owner==incoming.origins[index] && valid;
    require_collective(communicator,valid,"SIDM spatial native particle identity is duplicated or has inconsistent ownership");
  }
  require_collective(communicator,local.size()<=limits.maximum_exchange_records/27,
      "SIDM spatial ghost replication exceeds the recorded exchange allocation");
  std::vector<std::pair<int,ghost>> outgoing;
  outgoing.reserve(27*local.size());
  for(const auto &original:local) {
    auto particle=original;
    std::array<std::int64_t,3> primary;
    for(int axis=0;axis<3;axis++) {
      if(periodic_cells) {
        particle.proper_position_cm[axis]=std::fmod(particle.proper_position_cm[axis],periodic_box_cm);
        if(particle.proper_position_cm[axis]<0)particle.proper_position_cm[axis]+=periodic_box_cm;
      }
      const double coordinate=std::floor(particle.proper_position_cm[axis]/width);
      if(!std::isfinite(coordinate) || coordinate<=double(INT64_MIN/2) || coordinate>=double(INT64_MAX/2))
        throw std::runtime_error("SIDM spatial native coordinate outside finite cell-index domain");
      primary[axis]=std::int64_t(coordinate);
      if(periodic_cells && primary[axis]>=periodic_cells)primary[axis]=periodic_cells-1;
    }
    for(int dx=-1;dx<=1;dx++)for(int dy=-1;dy<=1;dy++)for(int dz=-1;dz<=1;dz++) {
      std::array<std::int64_t,3> target={{primary[0]+dx,primary[1]+dy,primary[2]+dz}};
      if(periodic_cells)for(auto &coordinate:target)coordinate=(coordinate+periodic_cells)%periodic_cells;
      outgoing.push_back({worker(target),{particle,primary,target}});
    }
  }
  auto incoming=exchange_pair_records(communicator,outgoing,limits,result.exchange);
  result.received_ghost_records=incoming.values.size();
  valid=true;for(std::size_t index=0;index<incoming.values.size();index++)valid=valid &&
      incoming.values[index].particle.owner==incoming.origins[index] && worker(incoming.values[index].target)==rank;
  require_collective(communicator,valid,"SIDM spatial ghost source or target-cell ownership differs");
  std::sort(incoming.values.begin(),incoming.values.end(),[](const ghost &left,const ghost &right) {
    return std::tie(left.target,left.particle.id)<std::tie(right.target,right.particle.id);
  });
  for(std::size_t begin=0;begin<incoming.values.size();) {
    std::size_t end=begin+1;while(end<incoming.values.size() && incoming.values[end].target==incoming.values[begin].target)++end;
    std::vector<dm_neighbor_point> tree_points;tree_points.reserve(end-begin);
    for(std::size_t index=begin;index<end;index++) {const auto &point=incoming.values[index].particle;
      tree_points.push_back({point.id,index,point.proper_position_cm,point.proper_support_radius_cm});}
    dm_neighbor_tree dm_tree(std::move(tree_points),periodic_box_cm,limits.memory_bytes/8);
    for(std::size_t left=begin;left<end;left++) {
      const auto &first=incoming.values[left];
      if(first.primary!=first.target)continue;
      const dm_neighbor_point target={first.particle.id,left,first.particle.proper_position_cm,first.particle.proper_support_radius_cm};
      const auto neighbors=dm_tree.overlapping(target,result.examined_neighbors,maximum_neighbor_visits);
      for(std::size_t right:neighbors) {
        const auto &second=incoming.values[right];
        if(first.particle.id>=second.particle.id)continue;
        if(first.particle.native_type==2 && second.particle.native_type==2)continue;
        double displacement[3];
        for(int axis=0;axis<3;axis++) {
          displacement[axis]=first.particle.proper_position_cm[axis]-second.particle.proper_position_cm[axis];
          if(periodic_cells)displacement[axis]-=periodic_box_cm*std::nearbyint(displacement[axis]/periodic_box_cm);
        }
        const double distance=std::hypot(displacement[0],std::hypot(displacement[1],displacement[2]));
        if(distance>=first.particle.proper_support_radius_cm+second.particle.proper_support_radius_cm)continue;
        const double overlap=measured_kernel_overlap(kernel_kind,distance,first.particle.proper_support_radius_cm,
            second.particle.proper_support_radius_cm,kernel_origin_bound);
        if(overlap==0)continue;
        if(result.pairs.size()>=limits.maximum_local_pairs)
          throw std::runtime_error("SIDM native candidate pair count exceeds its recorded allocation");
        result.pairs.push_back({first.particle.id,second.particle.id,first.particle.owner,second.particle.owner,overlap,
            first.particle.mass_msun,second.particle.mass_msun,first.particle.native_type,second.particle.native_type});
      }
    }
    begin=end;
  }
  return result;
}
}
#endif
