// A potential owns its computational precision through all dependent native
// derivatives. Random fields, FFTs and observable output retain native types.
#pragma once
#include "poisson.hh"
#include "perturbation_theory.hh"
#include "potential_derivatives.hh"
#include "second_order_source.hh"
#include "source_archive.hh"
#include <memory>
#include <cmath>
#ifdef MUSIC_WIDE_MIXED_POTENTIAL
#include "wide_scalar_operators.hh"
#endif

namespace mixed_flow {
inline bool wide_enabled(bool kspace){
#ifdef MUSIC_WIDE_MIXED_POTENTIAL
    return !kspace;
#else
    (void)kspace;return false;
#endif
}

class Potential {
    config_file& configuration_;
    poisson_plugin& ordinary_solver_;
    bool kspace_;
    grid_hierarchy ordinary_;
#ifdef MUSIC_WIDE_MIXED_POTENTIAL
    std::unique_ptr<GridHierarchy<__float128>> wide_;
    template<class Stencil,class Boundary>
    void solve_wide(const grid_hierarchy& original,GridHierarchy<__float128>& source,double tolerance,
                    multigrid::opt::smtype smoother,unsigned pre,unsigned post,unsigned maximum){
        multigrid::solver<Stencil,Boundary,wide_native::mg_straight,__float128> solver(source,smoother,pre,post);
        solver.maximum_iterations=maximum;
        const double error=solver.solve(*wide_,tolerance,true);
        if(!std::isfinite(error)||error>tolerance)throw std::runtime_error("Wide native Poisson solve misses declared tolerance");
        // Independently retain the ORIGINAL physical-leaf equations, even
        // though FAS temporarily modifies covered coarse source cells.
        Stencil stencil;
        for(unsigned level=original.levelmin();level<=original.levelmax();++level){
            const auto* field=original.get_grid(level);const auto* potential=wide_->get_grid(level);
            __float128 sum=0.,norm=0.;
            for(int i=0;i<int(field->size(0));++i)for(int j=0;j<int(field->size(1));++j)for(int k=0;k<int(field->size(2));++k){
                if(level<original.levelmax()){
                    const auto* child=original.get_grid(level+1);
                    if(i>=child->offset(0)&&i<child->offset(0)+int(child->size(0))/2&&
                       j>=child->offset(1)&&j<child->offset(1)+int(child->size(1))/2&&
                       k>=child->offset(2)&&k<child->offset(2)+int(child->size(2))/2)continue;
                }
                const __float128 f=(*field)(i,j,k);
                const __float128 residual=stencil.apply(*potential,i,j,k)*std::ldexp(1.,2*level)+f;
                sum+=residual*residual;norm+=f*f;
            }
            const double relative=norm>0?std::sqrt(double(sum/norm)):(sum==0?0.:std::numeric_limits<double>::infinity());
            if(!std::isfinite(relative)||relative>tolerance)
                throw std::runtime_error("Wide returned potential misses original physical-source tolerance");
            music::ilog.Print("Wide potential original-source level %u residual %g tolerance %g",level,relative,tolerance);
        }
    }
#endif
public:
    Potential(config_file& configuration,poisson_plugin& solver,bool kspace,unsigned halo)
        :configuration_(configuration),ordinary_solver_(solver),kspace_(kspace),ordinary_(halo){}
    void solve(grid_hierarchy& source){
#ifdef MUSIC_WIDE_MIXED_POTENTIAL
        if(wide_enabled(kspace_)){
            if(!wide_)wide_=std::make_unique<GridHierarchy<__float128>>(source.get_grid(source.levelmin())->m_nbnd);
            wide_->copy_geometry_from(source);
            GridHierarchy<__float128> work(source.get_grid(source.levelmin())->m_nbnd);work.copy_geometry_from(source);
            for(unsigned level=0;level<=source.levelmax();++level){
                const auto* from=source.get_grid(level);auto* to=work.get_grid(level);
                for(int i=0;i<int(from->size(0));++i)for(int j=0;j<int(from->size(1));++j)for(int k=0;k<int(from->size(2));++k)
                    (*to)(i,j,k)=(*from)(i,j,k);
            }
            const double tolerance=configuration_.get_value_safe<double>("poisson","accuracy",1e-5);
            const unsigned maximum=configuration_.get_value_safe<unsigned>("poisson","maximum_iterations",500);
            if(!std::isfinite(tolerance)||tolerance<=0||maximum<2||maximum>100000)
                throw std::runtime_error("Invalid declared Poisson tolerance/iteration budget");
            const auto name=configuration_.get_value_safe<std::string>("poisson","smoother","gs");
            auto smoother=multigrid::opt::sm_gauss_seidel;
            if(name=="jacobi")smoother=multigrid::opt::sm_jacobi;
            else if(name=="sor")smoother=multigrid::opt::sm_sor;
            const unsigned pre=configuration_.get_value_safe<unsigned>("poisson","pre_smooth",3);
            const unsigned post=configuration_.get_value_safe<unsigned>("poisson","post_smooth",3);
            const unsigned order=configuration_.get_value_safe<unsigned>("poisson","laplace_order",6);
            if(order==2)solve_wide<wide_native::stencil_7P,wide_native::interp_O3_fluxcorr>(source,work,tolerance,smoother,pre,post,maximum);
            else if(order==4)solve_wide<wide_native::stencil_13P,wide_native::interp_O5_fluxcorr>(source,work,tolerance,smoother,pre,post,maximum);
            else if(order==6)solve_wide<wide_native::stencil_19P,wide_native::interp_O7_fluxcorr>(source,work,tolerance,smoother,pre,post,maximum);
            else throw std::runtime_error("Invalid native Laplace operator order");
            return;
        }
#endif
        ordinary_=source;ordinary_.zero();ordinary_solver_.solve(source,ordinary_);
    }
    void shape(grid_hierarchy& output)const{
#ifdef MUSIC_WIDE_MIXED_POTENTIAL
        if(wide_){output.copy_geometry_from(*wide_);return;}
#endif
        output.copy_geometry_from(ordinary_);
    }
    void gradient(int axis,grid_hierarchy& output,bool add){
        if(axis<0||axis>2)throw std::runtime_error("Invalid potential gradient axis");
#ifdef MUSIC_WIDE_MIXED_POTENTIAL
        if(wide_){
            if(!add)shape(output);
            const unsigned order=configuration_.get_value_safe<unsigned>("poisson","grad_order",6);
            if(order==2){if(add)music_finite_difference::gradient_add_O2(axis,*wide_,output);else music_finite_difference::gradient_O2(axis,*wide_,output);}
            else if(order==4){if(add)music_finite_difference::gradient_add_O4(axis,*wide_,output);else music_finite_difference::gradient_O4(axis,*wide_,output);}
            else if(order==6){if(add)music_finite_difference::gradient_add_O6(axis,*wide_,output);else music_finite_difference::gradient_O6(axis,*wide_,output);}
            else throw std::runtime_error("Invalid native gradient operator order");
            return;
        }
#endif
        if(add)ordinary_solver_.gradient_add(axis,ordinary_,output);else ordinary_solver_.gradient(axis,ordinary_,output);
    }
    void second_order(grid_hierarchy& output,unsigned order){
#ifdef MUSIC_WIDE_MIXED_POTENTIAL
        if(wide_){music_finite_difference::second_order_source(*wide_,output,order);return;}
#endif
        if(kspace_)compute_2LPT_source_FFT(configuration_,ordinary_,output);else compute_2LPT_source(ordinary_,output,order);
    }
    void archive(music_source_archive::Writer& writer,unsigned quantity,unsigned species,unsigned halo)const{
#ifdef MUSIC_WIDE_MIXED_POTENTIAL
        if(wide_){writer.field_wide(quantity,species,*wide_,halo);return;}
#endif
        writer.field(quantity,species,ordinary_,halo);
    }
};
}
