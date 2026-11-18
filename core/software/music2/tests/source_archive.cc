// Native writer/independent Python reader integration fixture, not IC science.
#include "source_archive.hh"
#include <iostream>
int main(int argc,char **argv){try{
 if(argc!=2&&argc!=3)return 2;
 const bool wide=argc==3&&std::string(argv[2])=="--wide";
 if(argc==3&&!wide)return 2;
 grid_hierarchy field(4);field.create_base_hierarchy(3);
 const uint64_t expected=music_source_archive::estimated_bytes(field,4,wide?16:8);
 music_source_archive::Writer writer(argv[1],expected,wide);
#ifdef MUSIC_WIDE_MIXED_POTENTIAL
 GridHierarchy<__float128> wide_field(4);wide_field.copy_geometry_from(field);
#else
 if(wide)throw std::runtime_error("Wide archive fixture requires wide native build");
#endif
 for(unsigned t:{1,0})for(unsigned quantity=1;quantity<=7;++quantity){
  auto &grid=*field.get_grid(3);
  for(int i=-4;i<12;++i)for(int j=-4;j<12;++j)for(int k=-4;k<12;++k)
   grid(i,j,k)=quantity+10*t;
  const bool potential=quantity==2||quantity==4||quantity==7;
#ifdef MUSIC_WIDE_MIXED_POTENTIAL
  if(wide&&potential){
   auto &target=*wide_field.get_grid(3);
   for(int i=-4;i<12;++i)for(int j=-4;j<12;++j)for(int k=-4;k<12;++k)
    target(i,j,k)=__float128(1)+__float128(quantity+10*t)*std::ldexp(1.,-80)+__float128(i+4)*std::ldexp(1.,-105);
   writer.field_wide(quantity,t,wide_field,4);
  }else
#endif
   writer.field(quantity,t,field,potential?4:0);
 }
 writer.finish();std::cout<<"PASS native_source_archive expected_bytes="<<expected<<'\n';return 0;
}catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}}
