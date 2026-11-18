/* Native DM octree. GADGET4 basenode supplies the same threaded subtree skip
 * semantics as the gravity tree; spatial ghost exchange supplies MPI support.
 * No gas-only SPH neighborhood or gas smoothing length enters this search.
 */
#include "gadgetconfig.h"
#ifdef RESEARCH_SIDM_REFERENCE
#include "sidm.h"
#include "../tree/tree.h"
#include <algorithm>
#include <climits>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace research_sidm {
struct dm_neighbor_tree::implementation
{
  struct dmnode : basenode {
    std::array<double,3> lower,upper;
    double maximum_support=0;
    std::size_t begin=0,end=0;
    bool leaf=false;
  };
  std::vector<dm_neighbor_point> points;
  std::vector<std::size_t> order;
  std::unique_ptr<dmnode[]> nodes;
  std::size_t capacity=0,used=0;
  double box=0;

  int build(std::size_t begin,std::size_t end,int father,unsigned level)
  {
    if(used>=capacity || used>std::size_t(INT_MAX))throw std::runtime_error("SIDM native DM tree node allocation exhausted");
    const int index=int(used++);auto &node=nodes[index];node.father=father;node.level=static_cast<unsigned char>(level);
    node.begin=begin;node.end=end;node.nextnode=-1;node.sibling=-1;node.not_empty=begin<end;
    node.lower={{INFINITY,INFINITY,INFINITY}};node.upper={{-INFINITY,-INFINITY,-INFINITY}};
    for(std::size_t i=begin;i<end;i++) {
      const auto &p=points[order[i]];node.maximum_support=std::max(node.maximum_support,p.support_cm);
      for(int axis=0;axis<3;axis++){node.lower[axis]=std::min(node.lower[axis],p.position_cm[axis]);node.upper[axis]=std::max(node.upper[axis],p.position_cm[axis]);}
    }
    node.leaf=end-begin<=TREE_NUM_BEFORE_NODESPLIT || level>=60;
    std::array<double,3> middle;bool any_extent=false;
    for(int axis=0;axis<3;axis++){middle[axis]=node.lower[axis]/2+node.upper[axis]/2;any_extent=any_extent || node.lower[axis]<node.upper[axis];}
    if(!any_extent)node.leaf=true;
    if(node.leaf)return index;
    const auto octant=[&](std::size_t i){int cell=0;for(int axis=0;axis<3;axis++)if(points[i].position_cm[axis]>=middle[axis])cell|=1<<axis;return cell;};
    std::stable_sort(order.begin()+begin,order.begin()+end,[&](std::size_t left,std::size_t right){
      const int l=octant(left),r=octant(right);return l<r || (l==r && points[left].id<points[right].id);
    });
    // A rounded bounding-box midpoint can leave all points in one octant.
    // Keep that node as an exact leaf instead of creating unary chains which
    // violate the 2*N node allocation bound without refining the population.
    if(octant(order[begin])==octant(order[end-1])){node.leaf=true;return index;}
    int previous=-1;
    for(std::size_t first=begin;first<end;) {
      std::size_t last=first+1;while(last<end && octant(order[last])==octant(order[first]))last++;
      const int child=build(first,last,index,level+1);
      if(node.nextnode<0)node.nextnode=child;
      if(previous>=0)nodes[previous].sibling=child;
      previous=child;first=last;
    }
    return index;
  }
  void thread_siblings(int index,int sibling)
  {
    nodes[index].sibling=sibling;if(nodes[index].leaf)return;
    int child=nodes[index].nextnode;
    while(child>=0 && child!=sibling) {
      const int next=nodes[child].sibling;
      thread_siblings(child,next>=0?next:sibling);
      if(next<0)break;child=next;
    }
  }
};
dm_neighbor_tree::dm_neighbor_tree(std::vector<dm_neighbor_point> points,double box,std::uint64_t memory):data_(new implementation)
{
  if(!std::isfinite(box) || box<0)throw std::runtime_error("SIDM native DM tree periodic geometry invalid");
  data_->points=std::move(points);data_->box=box;
  const std::uint64_t n=data_->points.size();
  if(n>std::uint64_t(INT_MAX)/2 || n>memory/(2*sizeof(implementation::dmnode)+sizeof(dm_neighbor_point)+sizeof(std::size_t)))
    throw std::runtime_error("SIDM native DM octree exceeds its declared memory allocation");
  data_->capacity=std::max<std::size_t>(1,2*n);data_->nodes.reset(new implementation::dmnode[data_->capacity]);
  data_->order.resize(n);std::iota(data_->order.begin(),data_->order.end(),0);
  for(const auto &point:data_->points) {
    if(!std::isfinite(point.support_cm) || point.support_cm<=0)throw std::runtime_error("SIDM native DM tree support invalid");
    for(double value:point.position_cm)if(!std::isfinite(value))throw std::runtime_error("SIDM native DM tree position invalid");
  }
  if(n){data_->build(0,n,-1,0);data_->thread_siblings(0,-1);}
}
dm_neighbor_tree::~dm_neighbor_tree()=default;
std::vector<std::size_t> dm_neighbor_tree::overlapping(const dm_neighbor_point &target,std::uint64_t &visits,std::uint64_t limit) const
{
  std::vector<std::size_t> result;if(!data_->used)return result;
  int index=0;
  while(index>=0) {
    const auto &node=data_->nodes[index];const double reach=target.support_cm+node.maximum_support;
    bool overlap=false;
    // Wrapped nearest images are explored against the same physical tree;
    // only one result per retained identity is emitted after the union.
    const int images=data_->box>0?1:0;
    for(int dx=-images;dx<=images && !overlap;dx++)for(int dy=-images;dy<=images && !overlap;dy++)for(int dz=-images;dz<=images && !overlap;dz++) {
      const int shifts[3]={dx,dy,dz};long double distance2=0;
      for(int axis=0;axis<3;axis++) {
        const double position=target.position_cm[axis]+shifts[axis]*data_->box;
        const double delta=position<node.lower[axis]?node.lower[axis]-position:position>node.upper[axis]?position-node.upper[axis]:0;
        distance2+=static_cast<long double>(delta)*delta;
      }
      overlap=distance2<static_cast<long double>(reach)*reach;
    }
    if(!overlap){index=node.sibling;continue;}
    if(!node.leaf){index=node.nextnode;continue;}
    for(std::size_t i=node.begin;i<node.end;i++) {
      const auto &point=data_->points[data_->order[i]];if(point.id<=target.id)continue;
      if(visits>=limit)throw std::runtime_error("SIDM actual DM tree neighbor work exceeds declared allocation");++visits;
      std::array<double,3> delta;
      for(int axis=0;axis<3;axis++){delta[axis]=point.position_cm[axis]-target.position_cm[axis];if(data_->box>0)delta[axis]-=data_->box*std::nearbyint(delta[axis]/data_->box);}
      if(std::hypot(delta[0],std::hypot(delta[1],delta[2]))<point.support_cm+target.support_cm)result.push_back(point.source_index);
    }
    index=node.sibling;
  }
  std::sort(result.begin(),result.end());result.erase(std::unique(result.begin(),result.end()),result.end());return result;
}
}
#endif
