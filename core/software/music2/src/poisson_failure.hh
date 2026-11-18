// Bounded, opt-in failure evidence. Never changes a source, potential or
// convergence criterion. Files are created exclusively and retained on error.
#pragma once
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

class PoissonFailureSnapshot {
    std::string path_;
    std::unique_ptr<grid_hierarchy> original_;
    static void bytes(std::FILE* file,uint64_t value,unsigned count){
        for(unsigned n=0;n<count;++n)
            if(std::fputc(int((value>>(8*n))&255),file)==EOF)
                throw std::runtime_error("Cannot write bounded Poisson failure snapshot");
    }
    static void scalar(std::FILE* file,double value){
        static_assert(sizeof(double)==8 && std::numeric_limits<double>::is_iec559);
        uint64_t bits;std::memcpy(&bits,&value,8);bytes(file,bits,8);
    }
public:
    PoissonFailureSnapshot(config_file& cf,const grid_hierarchy& source){
        path_=cf.get_value_safe<std::string>("poisson","failure_snapshot","");
        if(path_.empty())return;
        const unsigned mib=cf.get_value_safe<unsigned>("poisson","failure_snapshot_max_mib",4);
        if(mib==0 || mib>64 || std::numeric_limits<real_t>::digits>53)
            throw std::runtime_error("Poisson failure snapshot requires 1..64 MiB and at most binary64 fields");
        const std::filesystem::path path(path_);
        if(std::filesystem::exists(path) || !std::filesystem::is_directory(path.has_parent_path()?path.parent_path():std::filesystem::path(".")))
            throw std::runtime_error("Poisson failure snapshot needs a new file in an existing directory");
        // Bound both the original-source copy (including auxiliary levels)
        // and the complete three-field publication before copying anything.
        const uint64_t limit=uint64_t(mib)*1024*1024;uint64_t required=64*1024;
        for(unsigned level=0;level<=source.levelmax();++level){
            const auto* grid=source.get_grid(level);uint64_t cells=1;
            for(int axis=0;axis<3;++axis){
                const uint64_t side=uint64_t(grid->size(axis))+2*grid->m_nbnd;
                if(side==0 || cells>limit/side)throw std::runtime_error("Poisson failure snapshot exceeds explicit budget");
                cells*=side;
            }
            const uint64_t per_cell=sizeof(real_t)+sizeof(short)+(level>=source.levelmin()?24:0);
            if(cells>limit/per_cell || required>limit-cells*per_cell)
                throw std::runtime_error("Poisson failure snapshot exceeds explicit budget");
            required+=cells*per_cell;
        }
        original_=std::make_unique<grid_hierarchy>(source);
    }
    void publish(const grid_hierarchy& source,const grid_hierarchy& potential,double error,double tolerance)const{
        if(!original_)return;
        std::FILE* raw=std::fopen(path_.c_str(),"wbx");
        if(!raw)throw std::runtime_error("Cannot exclusively create Poisson failure snapshot");
        std::unique_ptr<std::FILE,decltype(&std::fclose)> file(raw,std::fclose);
        if(std::fwrite("MUSICPS1",1,8,raw)!=8)throw std::runtime_error("Cannot write Poisson failure snapshot header");
        bytes(raw,1,4);bytes(raw,source.levelmin(),4);bytes(raw,source.levelmax(),4);
        scalar(raw,error);scalar(raw,tolerance);
        for(unsigned level=source.levelmin();level<=source.levelmax();++level){
            const auto* before=original_->get_grid(level);const auto* after=source.get_grid(level);const auto* u=potential.get_grid(level);
            bytes(raw,level,4);bytes(raw,before->m_nbnd,4);
            for(int axis=0;axis<3;++axis){
                if(before->size(axis)!=after->size(axis) || before->size(axis)!=u->size(axis)
                        || before->m_nbnd!=u->m_nbnd || before->m_nbnd!=after->m_nbnd)
                    throw std::runtime_error("Poisson failure snapshot geometry changed");
                bytes(raw,before->size(axis),4);
            }
            for(int axis=0;axis<3;++axis)bytes(raw,before->offset(axis),4);
            const int halo=before->m_nbnd;
            for(const auto* grid : {before,after,u})
                for(int i=-halo;i<int(grid->size(0))+halo;++i)
                    for(int j=-halo;j<int(grid->size(1))+halo;++j)
                        for(int k=-halo;k<int(grid->size(2))+halo;++k){
                            // Source ghosts never enter a physical equation.
                            const bool ghost=i<0||j<0||k<0||i>=int(grid->size(0))||j>=int(grid->size(1))||k>=int(grid->size(2));
                            scalar(raw,grid!=u&&ghost?0.:(*grid)(i,j,k));
                        }
        }
        if(std::fflush(raw)!=0)throw std::runtime_error("Cannot flush bounded Poisson failure snapshot");
    }
};
