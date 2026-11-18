// The native finite-difference solve plus its hybrid correction must recover
// the analytic periodic displacement, including modes near the grid cutoff.
#include "poisson.hh"
#include <omp.h>
#include <iostream>
#include <cmath>
#include <limits>
#include <algorithm>

int main(int argc,char **argv)
{
    if(argc!=4)return 2;
    config_file cf(argv[1]);
    const int correction_order=std::stoi(argv[2]);
    const int laplace_order=std::stoi(argv[3]);
    cf.insert_value("poisson","grad_order",argv[2]);
    cf.insert_value("poisson","laplace_order",argv[3]);
    omp_set_num_threads(2);
    multigrid_poisson_plugin solver(cf);
    for(int mode : {1,4,10,13}) {
        grid_hierarchy source(4);source.create_base_hierarchy(5);
        auto &grid=*source.get_grid(5);
        for(int i=0;i<32;i++)for(int j=0;j<32;j++)for(int k=0;k<32;k++)
            grid(i,j,k)=0.001*std::cos(2*M_PI*mode*(i+.5)/32.+.37);
        grid_hierarchy potential(source);potential.zero();
        solver.solve(source,potential);
        grid_hierarchy response(source);
        poisson_hybrid(*response.get_grid(5),0,correction_order,true,false,laplace_order);
        *response.get_grid(5)/=32.;
        solver.gradient_add(0,potential,response);
        long double error=0,reference=0;
        for(int i=0;i<32;i++)for(int j=0;j<32;j++)for(int k=0;k<32;k++) {
            const double expected=-0.001*std::sin(2*M_PI*mode*(i+.5)/32.+.37)/(2*M_PI*mode);
            const double difference=(*response.get_grid(5))(i,j,k)-expected;
            error+=difference*difference;reference+=expected*expected;
        }
        const double relative=std::sqrt(error/reference);
        std::cout<<"HYBRID mode="<<mode<<" correction_order="<<correction_order<<" laplace_order="<<laplace_order<<" relative_l2="<<relative<<'\n';
        if(relative>std::max(1.e-9,128.*double(std::numeric_limits<real_t>::epsilon())))return 1;
    }
}
