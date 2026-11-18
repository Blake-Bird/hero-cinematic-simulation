// Floating-point control of the actual native rational Laplacians. This is
// an operator check, not a certificate for a solved cosmological hierarchy.
#include "fd_schemes.hh"
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

struct Samples {
    std::array<double,343> data{};
    double operator()(int i,int j,int k)const{return data.at((i+3)*49+(j+3)*7+k+3);}
    double& operator()(int i,int j,int k){return data.at((i+3)*49+(j+3)*7+k+3);}
};
template<class Stencil> void check(int order,Stencil stencil){
    Samples input;for(auto& x:input.data)x=.1;
    if(stencil.apply(input,0,0,0)!=0.)throw std::runtime_error("Constant potential must have exactly zero difference-stencil Laplacian");
    const std::array<int,3> weights=order==2?std::array<int,3>{{1,0,0}}:
        order==4?std::array<int,3>{{16,-1,0}}:std::array<int,3>{{270,-27,2}};
    const int denominator=order==2?1:order==4?12:180;
    for(int i=-3;i<=3;++i)for(int j=-3;j<=3;++j)for(int k=-3;k<=3;++k)
        input(i,j,k)=1000000.1+1.e-5*(std::sin(.31*i+.17*j-.23*k)+.17*i*i+.09*j*j-.13*k*k);
    long double expected=0.;const long double center=input(0,0,0);
    double grouped=0.;
    for(int d=1;d<=order/2;++d){
        long double difference=0.;double sum=0.;
        for(int a=0;a<3;++a)for(int sign:{-1,1}){
            int cell[3]={0,0,0};cell[a]=sign*d;const double v=input(cell[0],cell[1],cell[2]);
            difference+=(long double)v-center;sum+=v;
        }
        expected+=weights[d-1]*difference;grouped+=weights[d-1]*sum;
    }
    expected/=denominator;
    int total=0;for(int weight:weights)total+=weight;
    const double old=(grouped-6.*total*input(0,0,0))/denominator;
    const double actual=stencil.apply(input,0,0,0);
    const long double error=std::abs((long double)actual-expected),previous=std::abs((long double)old-expected);
    if(error>64*std::numeric_limits<double>::epsilon()*std::abs(expected))
        throw std::runtime_error("Difference stencil disagrees with extended-precision rational reference");
    if(previous>0.&&error>previous)throw std::runtime_error("Difference stencil increases cancellation error");
    std::cout<<"PASS order="<<order<<" rational_reference_error="<<(double)error<<" previous_grouped_error="<<(double)previous<<"\n";
}
int main(){check(2,stencil_7P());check(4,stencil_13P());check(6,stencil_19P());}
