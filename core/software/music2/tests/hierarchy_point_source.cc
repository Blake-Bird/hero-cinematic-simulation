// Cell-centred point reconstruction: analytic polynomials and periodic waves.
#include "hierarchy_source.hh"
#include <iostream>
#include <iomanip>
using namespace hierarchy_source;

static double polynomial(double x,double y,double z,int degree){
 return std::pow(x,degree)+2*y*y-.7*z+x*y;
}
static std::vector<Grid> make_polynomial(int degree){
 std::vector<Grid> grids;
 for(unsigned level=4;level<=6;++level){
  const size_t side=level==6?8:16;const int64_t origin=level==4?0:level==5?8:24;
  Box data({{side,side,side}});
  for(size_t i=0;i<side;++i)for(size_t j=0;j<side;++j)for(size_t k=0;k<side;++k)
   data.at(i,j,k)=polynomial((origin+i+.5)/double(1u<<level),(origin+j+.5)/double(1u<<level),(origin+k+.5)/double(1u<<level),degree);
  grids.push_back({level,{{origin,origin,origin}},std::move(data)});
 }
 return grids;
}
int main(){try{
 std::cout<<std::setprecision(17);
 for(int degree:{3,5}){
  auto grids=make_polynomial(degree);
  Sampler point(grids,degree,16*1024*1024,SamplingConvention::CellCenteredPoint);
  auto output=point.sample({{33,33,33}},{{4,4,4}});double maximum=0;
  for(size_t i=0;i<4;++i)for(size_t j=0;j<4;++j)for(size_t k=0;k<4;++k)
   maximum=std::max(maximum,std::abs(output.at(i,j,k)-polynomial((33+i+.5)/64.,(33+j+.5)/64.,(33+k+.5)/64.,degree)));
  if(maximum>5.e-15)throw std::runtime_error("Point polynomial reconstruction failed");
  std::cout<<"POINT polynomial_degree="<<degree<<" maximum_error="<<maximum<<'\n';
  double previous=0;
  for(unsigned low:{4,5,6}){
   const size_t side=size_t(1)<<low;std::vector<Grid> wave;
   for(unsigned level=low;level<=low+1;++level){
    size_t n=level==low?side:8;int64_t origin=level==low?0:side/2;Box data({{n,n,n}});
    for(size_t i=0;i<n;++i)for(size_t j=0;j<n;++j)for(size_t k=0;k<n;++k)
     data.at(i,j,k)=std::cos(2*M_PI*(origin+i+.5)/double(size_t(1)<<level)+.37);
    wave.push_back({level,{{origin,origin,origin}},std::move(data)});
   }
   Sampler periodic(wave,degree,32*1024*1024,SamplingConvention::CellCenteredPoint);
   auto values=periodic.sample({{-3,0,0}},{{6,2,2}});double error=0;
   for(int i=0;i<6;++i)error=std::max(error,std::abs(values.at(i,0,0)-std::cos(2*M_PI*(-3+i+.5)/double(2*side)+.37)));
   if(previous&&error>=previous/10)throw std::runtime_error("Periodic point convergence failed");
   previous=error;std::cout<<"POINT wave_degree="<<degree<<" base_side="<<side<<" maximum_error="<<error<<'\n';
  }
  bool rejected=false;
  try{Sampler invalid(grids,degree,16*1024*1024,static_cast<SamplingConvention>(2));}
  catch(const std::runtime_error&){rejected=true;}
  if(!rejected)throw std::runtime_error("Invalid source convention accepted");
 }
 std::cout<<"PASS point_source_polynomial_periodic_and_contract_checks\n";
 return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
