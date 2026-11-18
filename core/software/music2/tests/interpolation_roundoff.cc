// Check the actual sixth-order native boundary polynomials against independent
// Lagrange/rational references, including a large common potential constant.
#include "general.hh"
#include "mesh.hh"
#include "mg_interp.hh"
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

int main(){
    for(double base : {.1,-.1,1000000.125,-1000000.125}){
        real_t f[7];for(auto& value:f)value=base;
        for(double x : {-.5,-.25,0.,.25,.5}){
            if(interp6(f,x)!=base || interp6(f[0],f[1],f[2],f[3],f[4],f[5],f[6],x)!=base)
                throw std::runtime_error("Sixth-order interpolation must preserve a constant exactly");
        }
        if(interp6left(f[0],f[1],f[2],f[3],f[4],f[5],f[6])!=base ||
           interp6lleft(f[0],f[1],f[2],f[3],f[4],f[5],f[6])!=base ||
           interp6llleft(f[0],f[1],f[2],f[3],f[4],f[5],f[6])!=base)
            throw std::runtime_error("Boundary extrapolation must preserve a constant exactly");
    }
    const double base=1000000.125,step=std::ldexp(1.,-32);
    const std::array<int,7> q{{248,-296,416,0,-392,232,-88}};
    real_t f[7];for(int n=0;n<7;++n)f[n]=base+q[n]*step;
    for(double x : {-.5,-.25,0.,.25,.5}){
        long double correction=0.;
        for(int n=0;n<7;++n){
            long double weight=1.;
            for(int m=0;m<7;++m)if(m!=n)weight*=((long double)x-(m-3))/(n-m);
            correction+=weight*q[n]*step;
        }
        const double expected=base+(double)correction;
        const double spacing=std::nextafter(expected,std::numeric_limits<double>::infinity())-expected;
        const double actual=interp6(f,x),overload=interp6(f[0],f[1],f[2],f[3],f[4],f[5],f[6],x);
        if(std::abs(actual-expected)>spacing || actual!=overload)
            throw std::runtime_error("Native interpolation disagrees with independent Lagrange reference");
        std::cout<<"PASS interpolation x="<<x<<" error_ulps="<<std::abs(actual-expected)/spacing<<'\n';
    }
    // Exact rational coefficients over a common integer denominator; no
    // weighted large DC terms appear in the independent reference.
    const std::array<std::array<int,7>,3> numerators{{
        {{84,-1224,29172,39780,-23562,7956,-1155}},
        {{-252,4760,68068,-46410,36652,-13923,2156}},
        {{-756,30600,48620,-66300,58905,-23868,3850}}
    }};
    const std::array<double,3> actual{{interp6left(f[0],f[1],f[2],f[3],f[4],f[5],f[6]),
        interp6lleft(f[0],f[1],f[2],f[3],f[4],f[5],f[6]),interp6llleft(f[0],f[1],f[2],f[3],f[4],f[5],f[6])}};
    for(int row=0;row<3;++row){
        int64_t numerator=0;int sum=0;
        for(int n=0;n<7;++n){numerator+=int64_t(numerators[row][n])*q[n];sum+=numerators[row][n];}
        if(sum!=51051)throw std::runtime_error("Invalid rational reference");
        const double expected=base+(double(numerator)/51051.)*step;
        const double spacing=std::nextafter(expected,std::numeric_limits<double>::infinity())-expected;
        if(std::abs(actual[row]-expected)>spacing)throw std::runtime_error("Boundary extrapolation disagrees with rational reference");
        std::cout<<"PASS extrapolation row="<<row<<" error_ulps="<<std::abs(actual[row]-expected)/spacing<<'\n';
    }
}
