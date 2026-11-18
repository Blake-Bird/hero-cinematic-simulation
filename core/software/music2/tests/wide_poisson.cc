// Bounded replay of actual failed physical sources through the same native
// FAS solver with a wider scalar. This is not a wider-precision MUSIC engine,
// an FFT control, or permission to accept its rounded binary64 potential.
#include "general.hh"
#include "mesh.hh"
#include <cmath>
#include <fstream>
#include <limits>
#include <memory>
#include <cstdint>
#include <cstring>

// CMake probes the type directly: some compilers provide it without the
// historical __SIZEOF_FLOAT128__ predefined macro.
static_assert(sizeof(__float128)>sizeof(double));

#include "wide_scalar_operators.hh"

struct Record {
    unsigned level,halo;
    std::array<unsigned,3> shape,offset;
    std::vector<double> original,potential;
};
struct RoundedGrid {
    const MeshvarBnd<__float128>& grid;
    double operator()(int i,int j,int k)const{return double(grid(i,j,k));}
};
uint64_t integer(std::istream& input,unsigned count){
    uint64_t value=0;
    for(unsigned n=0;n<count;++n){int byte=input.get();if(byte==EOF)throw std::runtime_error("Truncated failed field");value|=uint64_t(byte)<<(8*n);}
    return value;
}
double scalar(std::istream& input){uint64_t bits=integer(input,8);double value;std::memcpy(&value,&bits,8);return value;}
std::vector<Record> read(const char* path,double& tolerance){
    std::ifstream input(path,std::ios::binary|std::ios::ate);
    if(!input)throw std::runtime_error("Failed-field source missing");
    const auto length=input.tellg();if(length<36 || length>64*1024*1024)throw std::runtime_error("Bounded failed-field input required");
    input.seekg(0);char magic[8];input.read(magic,8);
    if(std::memcmp(magic,"MUSICPS1",8)!=0 || integer(input,4)!=1)throw std::runtime_error("Actual native failure snapshot required");
    const unsigned first=integer(input,4),last=integer(input,4);scalar(input);tolerance=scalar(input);
    if(first<3 || first>6 || last<first || last>first+2 || !std::isfinite(tolerance) || tolerance<=0)
        throw std::runtime_error("Wide control requires a small bounded hierarchy and original positive tolerance");
    std::vector<Record> records;
    for(unsigned level=first;level<=last;++level){
        Record r;r.level=integer(input,4);r.halo=integer(input,4);
        for(auto& n:r.shape)n=integer(input,4);
        for(auto& n:r.offset)n=integer(input,4);
        if(r.level!=level || r.halo!=4)throw std::runtime_error("Unsupported native replay geometry");
        uint64_t cells=1;
        for(int axis=0;axis<3;++axis){
            const unsigned n=r.shape[axis];if(n<8 || n>64 || n%2)throw std::runtime_error("Bounded even native replay side required");
            if(level==first && (n!=(1u<<first)||r.offset[axis]!=0))throw std::runtime_error("Actual periodic base geometry required");
            if(level>first && r.offset[axis]+n/2>records.back().shape[axis])throw std::runtime_error("Native child leaves its parent");
            cells*=n+2*r.halo;
        }
        if(cells>300000 || input.tellg()+std::streamoff(cells*24)>length)throw std::runtime_error("Wide replay workspace/truncation bound exceeded");
        r.original.resize(cells);r.potential.resize(cells);
        for(auto& x:r.original){x=scalar(input);if(!std::isfinite(x))throw std::runtime_error("Nonfinite original source");}
        input.seekg(cells*8,std::ios::cur); // Returned FAS source is not the replay input.
        for(auto& x:r.potential){x=scalar(input);if(!std::isfinite(x))throw std::runtime_error("Nonfinite failed potential");}
        records.push_back(std::move(r));
    }
    if(input.tellg()!=length)throw std::runtime_error("Trailing failed-field bytes");
    return records;
}

int main(int argc,char** argv){
    try{
        if(argc!=2 && argc!=3)return 2;
        const bool zero_start=argc==3;
        if(zero_start && std::string(argv[2])!="--zero-start")throw std::runtime_error("Unknown bounded replay option");
        const __float128 increment=__float128(1.)/(uint64_t(1)<<40)/(uint64_t(1)<<40);
        if(__float128(1.)+increment==__float128(1.))throw std::runtime_error("Wide arithmetic did not retain a bit below binary64");
        double tolerance;const auto records=read(argv[1],tolerance);
        GridHierarchy<__float128> source(4);source.create_base_hierarchy(records.front().level);
        for(size_t index=1;index<records.size();++index){const auto& r=records[index];source.add_patch(r.offset[0],r.offset[1],r.offset[2],r.shape[0],r.shape[1],r.shape[2]);}
        GridHierarchy<__float128> potential(source);potential.zero();source.zero();
        for(const auto& r:records){auto* f=source.get_grid(r.level);auto* u=potential.get_grid(r.level);size_t slot=0;
            for(int i=-4;i<int(r.shape[0])+4;++i)for(int j=-4;j<int(r.shape[1])+4;++j)for(int k=-4;k<int(r.shape[2])+4;++k,++slot){
                (*f)(i,j,k)=r.original[slot];
                if(!zero_start)(*u)(i,j,k)=r.potential[slot];
            }
        }
        std::cout<<"WIDE_SCALAR bytes="<<sizeof(__float128)<<" below_binary64_increment_retained=yes start="<<(zero_start?"zero":"captured_failed_field")<<'\n';
        using Solver=multigrid::solver<wide_native::stencil_19P,wide_native::interp_O7_fluxcorr,wide_native::mg_straight,__float128>;
        Solver solver(source,multigrid::opt::sm_gauss_seidel,3,3);
        const double error=solver.solve(potential,tolerance,true);
        if(!std::isfinite(error)||error>tolerance)throw std::runtime_error("Wide native solve misses unchanged tolerance");
        wide_native::stencil_19P laplace;
        double maximum=0.;
        for(size_t index=0;index<records.size();++index){
            const auto& r=records[index];const auto* u=potential.get_grid(r.level);const unsigned ny=r.shape[1]+8,nz=r.shape[2]+8;
            __float128 sum=0.,rounded_sum=0.,source_sum=0.;size_t count=0;
            const RoundedGrid rounded{*u};
            for(unsigned i=0;i<r.shape[0];++i)for(unsigned j=0;j<r.shape[1];++j)for(unsigned k=0;k<r.shape[2];++k){
                if(index+1<records.size()){
                    const auto& c=records[index+1];
                    if(i>=c.offset[0]&&i<c.offset[0]+c.shape[0]/2&&j>=c.offset[1]&&j<c.offset[1]+c.shape[1]/2&&k>=c.offset[2]&&k<c.offset[2]+c.shape[2]/2)continue;
                }
                const __float128 f=r.original[((i+4)*ny+j+4)*nz+k+4];
                const __float128 residual=laplace.apply(*u,i,j,k)*__float128(uint64_t(1)<<(2*r.level))+f;
                const __float128 rounded_residual=laplace.apply(rounded,i,j,k)*__float128(uint64_t(1)<<(2*r.level))+f;
                sum+=residual*residual;rounded_sum+=rounded_residual*rounded_residual;source_sum+=f*f;++count;
            }
            const auto relative_norm=[&](__float128 value){
                return source_sum>0?std::sqrt(double(value/source_sum)):
                    (value==0?0.:std::numeric_limits<double>::infinity());
            };
            const double relative=relative_norm(sum);
            if(!std::isfinite(relative))throw std::runtime_error("Invalid wider physical residual");
            maximum=std::max(maximum,relative);
            std::cout<<"WIDE_PHYSICAL_ORIGINAL level="<<r.level<<" leaves="<<count<<" relative_rms="<<relative<<" tolerance="<<tolerance<<'\n';
            std::cout<<"ROUNDED_FIELD_ORIGINAL level="<<r.level<<" relative_rms="<<relative_norm(rounded_sum)<<" tolerance="<<tolerance<<'\n';
        }
        if(maximum>tolerance)throw std::runtime_error("Wide returned field misses original physical-leaf tolerance");
        std::cout<<"PASS wider native FAS physical-source replay; no binary64 downcast acceptance or full generator certificate\n";
        return 0;
    }catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
}
