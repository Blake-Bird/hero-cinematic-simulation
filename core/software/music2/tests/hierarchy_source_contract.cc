#include "hierarchy_source.hh"
#include <iostream>
using namespace hierarchy_source;
int main(){
 std::vector<Grid> input;for(unsigned level=3;level<=18;++level){Box box(Shape{{8,8,8}});std::fill(box.values.begin(),box.values.end(),1.);input.push_back({level,{{0,0,0}},std::move(box)});}
 size_t rejected=0;auto reject=[&](auto operation){try{operation();}catch(const std::runtime_error&){++rejected;return;}throw std::runtime_error("Expected source contract rejection");};
 Sampler bounded(input,5,32*1024*1024);auto output=bounded.sample({{-4,2,7}},{{24,24,24}});
 for(double x:output.values)if(std::abs(x-1.)>1.e-13)return 1;
 reject([&]{Sampler invalid(input,4,32*1024*1024);});
 reject([&]{Sampler invalid(input,5,1);});
 reject([&]{bounded.sample({{0,0,0}},{{100000,100000,100000}});});
 reject([&]{bounded.sample({{0,0,0}},{{0,8,8}});});
 auto bad=input;bad[1].origin[0]=1;reject([&]{Sampler invalid(bad,5,32*1024*1024);});
 bad=input;bad[1].data.values[0]=std::numeric_limits<double>::quiet_NaN();reject([&]{Sampler invalid(bad,5,32*1024*1024);});
 bad=input;bad[2].origin[0]=24;reject([&]{Sampler invalid(bad,5,32*1024*1024);});
 bad=input;bad[1].level=5;reject([&]{Sampler invalid(bad,5,32*1024*1024);});
 // Native capture must be immutable even when the actual solver changes its RHS.
 grid_hierarchy source(4);source.create_base_hierarchy(3);for(int i=0;i<8;++i)for(int j=0;j<8;++j)for(int k=0;k<8;++k)(*source.get_grid(3))(i,j,k)=1.;
 Sampler captured(source,3,1024*1024);source.zero();auto immutable=captured.sample({{-2,7,0}},{{12,8,8}});
 for(double x:immutable.values)if(x!=1.)return 1;
 std::cout<<"PASS source_contract rejections="<<rejected<<" deep_level=18 cells="<<output.values.size()<<" immutable_capture=yes\n";
 return rejected==8?0:1;
}
