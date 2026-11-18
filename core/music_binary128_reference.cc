// Independent finite-difference reference on retained IEEE754 binary128 bits.
// No MUSIC stencil/header is included. Outputs round only after derivatives.
#include <cstdint>
#include <cstring>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <cstdio>
#include <initializer_list>
static_assert(sizeof(__float128)==16 && __FLT128_MANT_DIG__==113 && __FLT128_MAX_EXP__==16384);
using Q=__float128;
namespace {
struct Grid {
 const uint64_t* words;uint64_t ny,nz;bool little;
 Q at(uint64_t i,uint64_t j,uint64_t k)const{
  const uint64_t p=2*((i*ny+j)*nz+k);
  const unsigned __int128 bits=(static_cast<unsigned __int128>(words[p+(little?1:0)])<<64)|words[p+(little?0:1)];
  Q value;std::memcpy(&value,&bits,16);return value;
 }
};
Q weight(int order,int offset){
 if(order==2)return Q(1)/2;
 if(order==4)return offset==1?Q(2)/3:-Q(1)/12;
 return offset==1?Q(3)/4:offset==2?-Q(3)/20:Q(1)/60;
}
Q derivative2(const Grid& grid,const int64_t* cell,int a,int b,int order){
 const int radius=order==2?1:2;Q result=0;const Q centre=grid.at(cell[0],cell[1],cell[2]);
 for(int u=1;u<=radius;++u)for(int v=1;v<=radius;++v)
  for(int su:{-1,1})for(int sv:{-1,1}){
   int64_t q[3]={cell[0],cell[1],cell[2]};q[a]+=su*u;q[b]+=sv*v;
   result+=Q(su*sv)*weight(order==2?2:4,u)*weight(order==2?2:4,v)*
    (grid.at(q[0],q[1],q[2])-centre);
  }
 return result;
}
void publish(Q value,double* output){
 const double rounded=double(value);
 if(!__builtin_isfinite(value)||!std::isfinite(rounded))throw std::runtime_error("Wide reference output is nonfinite or outside binary64 observable range");
 if(value!=0 && rounded==0)throw std::runtime_error("Wide reference observable underflows binary64");
 *output=rounded;
}
}
extern "C" uint64_t music_binary128_reference_abi(){return 1;}
extern "C" int music_binary128_reference(const uint64_t* words,uint64_t word_count,int little,
 const uint64_t* shape,int halo,int level,int order,int operation,double* output,uint64_t outputs,char* error,uint64_t error_size){
 try{
  if(!words||!shape||!output||halo<0||halo>16||level<1||level>20||(little!=0&&little!=1)||
    (order!=2&&order!=4&&order!=6)||operation<0||operation>2)throw std::runtime_error("Invalid wide reference request");
  uint64_t total=1,cells=1;uint64_t inner[3];
  for(int a=0;a<3;++a){
   if(shape[a]<=uint64_t(2*halo)||shape[a]>uint64_t(1<<20)||total>UINT64_MAX/shape[a])throw std::runtime_error("Invalid wide reference geometry");
   total*=shape[a];inner[a]=shape[a]-2*halo;cells*=inner[a];
  }
  if(total>UINT64_MAX/2||word_count!=2*total||outputs!=(operation==1?3*cells:cells))throw std::runtime_error("Wide reference buffer count mismatch");
  const int radius=operation==2?(order==2?2:4):order/2;
  if(halo<radius)throw std::runtime_error("Wide reference needs original potential ghosts");
  for(uint64_t p=0;p<total;++p)if(((words[2*p+(little?1:0)]>>48)&0x7fff)==0x7fff)throw std::runtime_error("Nonfinite wide reference input");
  const Grid grid{words,shape[1],shape[2],bool(little)};const Q scale=uint64_t(1)<<level;uint64_t p=0;
  for(uint64_t i=0;i<inner[0];++i)for(uint64_t j=0;j<inner[1];++j)for(uint64_t k=0;k<inner[2];++k,++p){
   const int64_t cell[3]={int64_t(i)+halo,int64_t(j)+halo,int64_t(k)+halo};
   if(operation==0){
    const Q centre=grid.at(cell[0],cell[1],cell[2]);Q laplacian=0;
    for(int a=0;a<3;++a)for(int step=1;step<=order/2;++step){
     int64_t lo[3]={cell[0],cell[1],cell[2]},hi[3]={cell[0],cell[1],cell[2]};lo[a]-=step;hi[a]+=step;
     const Q coefficient=order==2?Q(1):order==4?(step==1?Q(4)/3:-Q(1)/12):(step==1?Q(3)/2:step==2?-Q(3)/20:Q(1)/90);
     laplacian+=coefficient*((grid.at(lo[0],lo[1],lo[2])-centre)+(grid.at(hi[0],hi[1],hi[2])-centre));
    }
    publish(-laplacian*scale*scale,output+p);
   }else if(operation==1){
    for(int a=0;a<3;++a){Q gradient=0;
     for(int step=1;step<=order/2;++step){
      int64_t lo[3]={cell[0],cell[1],cell[2]},hi[3]={cell[0],cell[1],cell[2]};lo[a]-=step;hi[a]+=step;
      gradient+=weight(order,step)*(grid.at(hi[0],hi[1],hi[2])-grid.at(lo[0],lo[1],lo[2]));
     }publish(gradient*scale,output+3*p+a);
    }
   }else{
    Q diagonal[3];for(int a=0;a<3;++a)diagonal[a]=derivative2(grid,cell,a,a,order)*scale*scale;
    Q source=0;for(int a=0;a<3;++a)for(int b=a+1;b<3;++b){
     const Q mixed=derivative2(grid,cell,a,b,order)*scale*scale;
     source+=diagonal[a]*diagonal[b]-mixed*mixed;
    }publish(source,output+p);
   }
  }return 0;
 }catch(const std::exception& exception){if(error&&error_size)std::snprintf(error,error_size,"%s",exception.what());return 1;}
}
extern "C" int music_binary128_dot(const uint64_t* words,uint64_t word_count,int little,
 const double* weights,uint64_t count,double* output,char* error,uint64_t error_size){
 try{
  if(!words||!weights||!output||count>UINT64_MAX/2||word_count!=2*count||(little!=0&&little!=1))throw std::runtime_error("Invalid wide dot buffer request");
  const Grid grid{words,1,count,bool(little)};Q sum=0,correction=0;
  for(uint64_t p=0;p<count;++p){
   const Q value=grid.at(0,0,p);
   if(!__builtin_isfinite(value)||!std::isfinite(weights[p]))throw std::runtime_error("Nonfinite wide dot input");
   const Q term=value*Q(weights[p]),adjusted=term-correction,next=sum+adjusted;
   correction=(next-sum)-adjusted;sum=next;
  }publish(sum,output);return 0;
 }catch(const std::exception& exception){if(error&&error_size)std::snprintf(error,error_size,"%s",exception.what());return 1;}
}
