// Experimental bounded conservative source model for numerical controls.
// Cell-average interpretation requires separate physical validation.
#pragma once
#include "mesh.hh"
#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <vector>

namespace hierarchy_source {
enum class SamplingConvention { CellAverage, CellCenteredPoint };
using Coord = std::array<int64_t,3>;
using Shape = std::array<size_t,3>;
inline int64_t wrap(int64_t x,int64_t n){return (x%n+n)%n;}
inline size_t cells(const Shape &s){
 size_t product=1;
 for(size_t n:s){if(!n||n>size_t(1)<<30||product>std::numeric_limits<size_t>::max()/n)throw std::runtime_error("Invalid source rectangle size");product*=n;}
 return product;
}
struct Box {
 Shape shape; std::vector<double> values;
 explicit Box(Shape s):shape(s),values(cells(s)){}
 double& at(size_t i,size_t j,size_t k){return values[(i*shape[1]+j)*shape[2]+k];}
 double at(size_t i,size_t j,size_t k)const{return values[(i*shape[1]+j)*shape[2]+k];}
};
struct Grid { unsigned level; Coord origin; Box data; };

class Sampler {
 std::vector<Grid> grids_;
 int degree_; size_t source_cells_,budget_bytes_; SamplingConvention sampling_;
 static void budget(long double bytes,size_t limit){if(bytes>limit)throw std::runtime_error("Hierarchy source exceeds declared workspace budget");}
 void validate(const std::vector<Grid>& grids)const{
  if(grids.empty())throw std::runtime_error("Source hierarchy is empty");
  for(size_t i=0;i<grids.size();++i){
   const auto &g=grids[i];if(g.level<2||g.level>30||(i&&g.level!=grids[i-1].level+1))throw std::runtime_error("Source levels must be contiguous");
   const size_t side=size_t(1)<<g.level;
   if(g.data.values.size()!=cells(g.data.shape))throw std::runtime_error("Source payload size mismatch");
   for(int a=0;a<3;++a){
    if(g.data.shape[a]%2||g.data.shape[a]>side)throw std::runtime_error("Invalid source grid geometry");
    if(i&&wrap(g.origin[a],side)%2)throw std::runtime_error("Unaligned child source cells");
    if(!i&&(g.origin[a]!=0||g.data.shape[a]!=side||side<size_t(degree_+1)))throw std::runtime_error("Full supported periodic source base required");
   }
   for(double x:g.data.values)if(!std::isfinite(x))throw std::runtime_error("Nonfinite hierarchy source");
   if(i){const auto &p=grids[i-1];const int64_t n=int64_t(1)<<p.level;
    for(int a=0;a<3;++a){const auto start=wrap(wrap(g.origin[a],side)/2-wrap(p.origin[a],n),n);
     if(p.data.shape!=Shape{{size_t(n),size_t(n),size_t(n)}}&&size_t(start)+g.data.shape[a]/2>p.data.shape[a])throw std::runtime_error("Child source outside finite parent");}
   }
  }
 }
 static std::vector<Grid> capture(const grid_hierarchy &source,int degree,size_t budget_bytes){
  if((degree!=0&&degree!=3&&degree!=5)||!budget_bytes||source.levelmin()<2||source.levelmax()>30)
   throw std::runtime_error("Invalid immutable source capture request");
  size_t total=0;
  for(unsigned level=source.levelmin();level<=source.levelmax();++level){
   const size_t count=cells(Shape{{source.size(level,0),source.size(level,1),source.size(level,2)}});
   if(total>std::numeric_limits<size_t>::max()-count)throw std::runtime_error("Source capture size overflow");total+=count;
  }
  budget(24.L*total,budget_bytes);
  std::vector<Grid> result;
  for(unsigned level=source.levelmin();level<=source.levelmax();++level){
   Shape shape={{source.size(level,0),source.size(level,1),source.size(level,2)}};Box data(shape);
   for(size_t i=0;i<shape[0];++i)for(size_t j=0;j<shape[1];++j)for(size_t k=0;k<shape[2];++k)data.at(i,j,k)=(*source.get_grid(level))(i,j,k);
   result.push_back({level,{{source.offset_abs(level,0),source.offset_abs(level,1),source.offset_abs(level,2)}},std::move(data)});
  }
  return result;
 }
 using Axes=std::array<std::vector<int64_t>,3>;
 Box evaluate(size_t index,const Axes& coords)const{
  const auto &g=grids_[index];const int64_t side=int64_t(1)<<g.level;
  Shape target={{coords[0].size(),coords[1].size(),coords[2].size()}};
  if(index==0){Box result(target);for(size_t i=0;i<target[0];++i)for(size_t j=0;j<target[1];++j)for(size_t k=0;k<target[2];++k)result.at(i,j,k)=g.data.at(coords[0][i],coords[1][j],coords[2][k]);return result;}
  const int radius=degree_?(degree_+1)/2:0;const int count=2*radius+1;const int64_t parent_side=side/2;
  Axes parents;std::array<std::vector<size_t>,3> lookup;
  for(int axis=0;axis<3;++axis){
   for(auto coordinate:coords[axis])for(int o=-radius;o<=radius;++o)parents[axis].push_back(wrap(coordinate/2+o,parent_side));
   auto &p=parents[axis];std::sort(p.begin(),p.end());p.erase(std::unique(p.begin(),p.end()),p.end());
   lookup[axis].reserve(coords[axis].size()*count);
   for(auto coordinate:coords[axis])for(int o=-radius;o<=radius;++o)lookup[axis].push_back(std::lower_bound(p.begin(),p.end(),wrap(coordinate/2+o,parent_side))-p.begin());
  }
  Box result=evaluate(index-1,parents);
  const double cubic[]={-3./128,22./128,1.,-22./128,3./128};
  const double quintic[]={5./1024,-44./1024,201./1024,1.,-201./1024,44./1024,-5./1024};
  const double point_cubic[]={-5./256,25./128,15./16,-17./128,5./256};
  const double point_quintic[]={63./16384,-153./4096,3675./16384,945./1024,-2331./16384,133./4096,-63./16384};
  for(int axis=0;axis<3;++axis){
   Shape shape=result.shape;shape[axis]=target[axis];Box refined(shape);
   for(int column=0;column<count;++column)
    for(size_t i=0;i<shape[0];++i)for(size_t j=0;j<shape[1];++j)for(size_t k=0;k<shape[2];++k){
     Shape at={{i,j,k}};const size_t position=at[axis];at[axis]=lookup[axis][position*count+column];
     const int coefficient=coords[axis][position]%2?count-1-column:column;
     const double *polynomial=degree_==3?
        (sampling_==SamplingConvention::CellAverage?cubic:point_cubic):
        (sampling_==SamplingConvention::CellAverage?quintic:point_quintic);
     const double weight=degree_==0?1.:polynomial[coefficient];
     refined.at(i,j,k)+=result.at(at[0],at[1],at[2])*weight;
    }
   result=std::move(refined);
  }
  for(size_t i=0;i<target[0];++i)for(size_t j=0;j<target[1];++j)for(size_t k=0;k<target[2];++k){
   Coord local={{wrap(coords[0][i]-g.origin[0],side),wrap(coords[1][j]-g.origin[1],side),wrap(coords[2][k]-g.origin[2],side)}};
   if(size_t(local[0])<g.data.shape[0]&&size_t(local[1])<g.data.shape[1]&&size_t(local[2])<g.data.shape[2])result.at(i,j,k)=g.data.at(local[0],local[1],local[2]);
  }
  return result;
 }
public:
 Sampler(const grid_hierarchy& input,int degree,size_t budget_bytes,SamplingConvention sampling=SamplingConvention::CellAverage):Sampler(capture(input,degree,budget_bytes),degree,budget_bytes,sampling){}
 Sampler(const std::vector<Grid>& input,int degree,size_t budget_bytes,SamplingConvention sampling=SamplingConvention::CellAverage):degree_(degree),source_cells_(0),budget_bytes_(budget_bytes),sampling_(sampling){
  if((degree!=0&&degree!=3&&degree!=5)||!budget_bytes)throw std::runtime_error("Invalid bounded source degree/budget");
  if(sampling_!=SamplingConvention::CellAverage&&sampling_!=SamplingConvention::CellCenteredPoint)throw std::runtime_error("Invalid hierarchy source sampling convention");
  validate(input);for(const auto &g:input){if(source_cells_>std::numeric_limits<size_t>::max()-g.data.values.size())throw std::runtime_error("Source size overflow");source_cells_+=g.data.values.size();}
  budget(24.L*source_cells_,budget_bytes_);grids_=input;
  for(auto &g:grids_){const int64_t side=int64_t(1)<<g.level;for(auto &x:g.origin)x=wrap(x,side);}
  for(size_t index=grids_.size()-1;sampling_==SamplingConvention::CellAverage&&index>0;--index){const auto &child=grids_[index];auto &parent=grids_[index-1];const int64_t side=int64_t(1)<<parent.level;Coord start;
   for(int a=0;a<3;++a)start[a]=wrap(child.origin[a]/2-parent.origin[a],side);
   for(size_t i=0;i<child.data.shape[0]/2;++i)for(size_t j=0;j<child.data.shape[1]/2;++j)for(size_t k=0;k<child.data.shape[2]/2;++k){
    double mean=0.;for(int x=0;x<2;++x)for(int y=0;y<2;++y)for(int z=0;z<2;++z)mean+=child.data.at(2*i+x,2*j+y,2*k+z)/8.;
    parent.data.at((start[0]+i)%parent.data.shape[0],(start[1]+j)%parent.data.shape[1],(start[2]+k)%parent.data.shape[2])=mean;
   }
  }
 }
 Box sample(Coord origin,Shape shape)const{
  cells(shape);Shape estimated=shape;long double workspace=0;
  for(size_t i=0;i<grids_.size();++i){workspace+=48.L*estimated[0]*estimated[1]*estimated[2];for(auto &n:estimated)n=(n+1)/2+degree_+3;}
  budget(workspace+24.L*source_cells_,budget_bytes_);
  Axes coords;const int64_t side=int64_t(1)<<grids_.back().level;
  for(int a=0;a<3;++a){coords[a].resize(shape[a]);const int64_t start=wrap(origin[a],side);for(size_t i=0;i<shape[a];++i)coords[a][i]=wrap(start+i,side);}
  return evaluate(grids_.size()-1,coords);
 }
};
} // namespace hierarchy_source
