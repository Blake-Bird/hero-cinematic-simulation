// Shared native derivatives must consume the wider potential before rounding
// their observable output. A small polynomial on a large constant detects an
// accidental binary64 potential conversion; it is not a cosmological control.
#include "wide_scalar_operators.hh"
#include "potential_derivatives.hh"
#include "second_order_source.hh"
#include <iostream>
#include <cmath>

int main(){
    {
        grid_hierarchy original(4);original.create_base_hierarchy(3);
        original.add_patch(2,1,0,8,10,12);original.bhave_refmask=true;
        (*original.m_ref_masks.back())(0,0,0)=-1;
        (*original.m_ref_masks.front())(0,0,0)=2;
        GridHierarchy<__float128> copied(1);copied.copy_geometry_from(original);
        if(copied.levelmin()!=3 || copied.levelmax()!=4 || !copied.bhave_refmask ||
                copied.offset_abs(4,0)!=4 || copied.offset_abs(4,1)!=2 || copied.offset_abs(4,2)!=0 ||
                copied.m_ref_masks.size()!=original.m_ref_masks.size() || copied.is_in_mask(4,0,0,0))
            throw std::runtime_error("Wider hierarchy lost rectangular geometry, mask or coordinates");
        (*original.m_ref_masks.back())(0,0,0)=1;
        if(copied.is_in_mask(4,0,0,0))throw std::runtime_error("Wider hierarchy shares a mutable source mask");
        for(unsigned level=0;level<=copied.levelmax();++level){
            const auto* grid=copied.get_grid(level);const int halo=grid->m_nbnd;
            for(int i=-halo;i<int(grid->size(0))+halo;++i)for(int j=-halo;j<int(grid->size(1))+halo;++j)for(int k=-halo;k<int(grid->size(2))+halo;++k)
                if((*grid)(i,j,k)!=0.)throw std::runtime_error("Converted geometry has uninitialized field/ghost storage");
        }
        std::cout<<"PASS cross-scalar rectangular geometry, absolute offsets, deep masks and zeroed ghosts\n";
    }
    constexpr int side=16,level=4,halo=4;
    const __float128 epsilon=__float128(1.)/(uint64_t(1)<<35)/(uint64_t(1)<<35);
    GridHierarchy<__float128> potential(halo);potential.create_base_hierarchy(level);potential.zero();
    auto& grid=*potential.get_grid(level);
    for(int i=-halo;i<side+halo;++i)for(int j=-halo;j<side+halo;++j)for(int k=-halo;k<side+halo;++k){
        const __float128 x=(__float128(i)+.5)/side,y=(__float128(j)+.5)/side,z=(__float128(k)+.5)/side;
        grid(i,j,k)=1.+epsilon*(x*x*x+2*y*y*y+3*z*z*z);
        if(double(grid(i,j,k))!=1.)throw std::runtime_error("Precision fixture must round to one constant in binary64");
    }
    for(unsigned order:{2,4,6}){
        grid_hierarchy source(halo);music_finite_difference::second_order_source(potential,source,order);
        double numerator=0.,denominator=0.;
        for(int i=0;i<side;++i)for(int j=0;j<side;++j)for(int k=0;k<side;++k){
            const double x=(i+.5)/side,y=(j+.5)/side,z=(k+.5)/side;
            const double expected=double(epsilon*epsilon)*(72*x*y+108*x*z+216*y*z-99.);
            const double actual=(*source.get_grid(level))(i,j,k);
            numerator+=(actual-expected)*(actual-expected);denominator+=expected*expected;
        }
        const double relative=std::sqrt(numerator/denominator);
        if(!std::isfinite(relative)||relative>1e-12)throw std::runtime_error("Second-order source lost wider potential curvature");
        std::cout<<"PASS wide quadratic source order="<<order<<" relative_l2="<<relative<<'\n';
        for(int axis=0;axis<3;++axis){
            grid_hierarchy result(halo);result.copy_geometry_from(potential);
            if(order==2)music_finite_difference::gradient_O2(axis,potential,result);
            if(order==4)music_finite_difference::gradient_O4(axis,potential,result);
            if(order==6)music_finite_difference::gradient_O6(axis,potential,result);
            double error=0.,reference=0.;
            for(int i=0;i<side;++i)for(int j=0;j<side;++j)for(int k=0;k<side;++k){
                const int cell[3]={i,j,k};const double x=(cell[axis]+.5)/side;
                const double expected=double(epsilon)*(axis+1)*(3*x*x+(order==2?1./(side*side):0.));
                const double value=(*result.get_grid(level))(i,j,k);
                error+=(value-expected)*(value-expected);reference+=expected*expected;
            }
            const double relative_gradient=std::sqrt(error/reference);
            if(!std::isfinite(relative_gradient)||relative_gradient>1e-12)
                throw std::runtime_error("Gradient lost wider potential curvature");
            grid_hierarchy added(result);
            if(order==2)music_finite_difference::gradient_add_O2(axis,potential,added);
            if(order==4)music_finite_difference::gradient_add_O4(axis,potential,added);
            if(order==6)music_finite_difference::gradient_add_O6(axis,potential,added);
            for(int i=0;i<side;++i)for(int j=0;j<side;++j)for(int k=0;k<side;++k)
                if((*added.get_grid(level))(i,j,k)!=2*(*result.get_grid(level))(i,j,k))
                    throw std::runtime_error("Additive derivative differs from native gradient");
            std::cout<<"PASS wide gradient/add order="<<order<<" axis="<<axis<<" relative_l2="<<relative_gradient<<'\n';
        }
    }
}
