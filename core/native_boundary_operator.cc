// Native MUSIC boundary adapter. Reuses the existing GPL MUSIC headers;
// no interpolation coefficients or face/corner traversal are replaced.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include <general.hh>
static_assert(sizeof(::real_t)==sizeof(double),"Boundary adapter requires the actual native binary64 precision configuration");

namespace boundary_tape {
struct Node { int64_t a,b; double wa,wb; };
struct Tape {
    uint64_t inputs,maximum_nodes;
    bool count_only=false;uint64_t counted_nodes=0;
    std::vector<Node> nodes;
    int64_t add(int64_t a,double wa,int64_t b=-1,double wb=0) {
        if(count_only){
            if(counted_nodes>=maximum_nodes)throw std::runtime_error("Native boundary operation count overflow");
            return int64_t(inputs+counted_nodes++);
        }
        if(nodes.size()>=maximum_nodes)throw std::runtime_error("Native boundary tape exceeds declared memory");
        if(nodes.size()==nodes.capacity())nodes.reserve(std::min<uint64_t>(maximum_nodes,std::max<uint64_t>(1024,2*nodes.capacity())));
        nodes.push_back({a,b,wa,wb});return int64_t(inputs+nodes.size()-1);
    }
};
thread_local Tape* active=nullptr;
struct Scalar {
    double value=0.;int64_t node=-1;
    Scalar()=default;Scalar(double x):value(x){}
    Scalar(double x,int64_t n):value(x),node(n){}
    explicit operator int()const {if(node>=0)throw std::runtime_error("Field used as boundary coordinate");return int(value);}
    Scalar operator-()const {return node<0?Scalar(-value):Scalar(-value,active->add(node,-1.));}
    Scalar& operator+=(const Scalar& x);Scalar& operator-=(const Scalar& x);
    Scalar& operator*=(const Scalar& x);Scalar& operator/=(const Scalar& x);
};
inline Scalar operator+(const Scalar& a,const Scalar& b){
    if(a.node<0)return Scalar(a.value+b.value,b.node);
    if(b.node<0)return Scalar(a.value+b.value,a.node);
    return Scalar(a.value+b.value,active->add(a.node,1.,b.node,1.));
}
inline Scalar operator-(const Scalar& a,const Scalar& b){return a+(-b);}
inline Scalar operator*(const Scalar& a,const Scalar& b){
    if(a.node>=0&&b.node>=0)throw std::runtime_error("Nonlinear native boundary field product");
    if(a.node>=0)return Scalar(a.value*b.value,active->add(a.node,b.value));
    if(b.node>=0)return Scalar(a.value*b.value,active->add(b.node,a.value));
    return Scalar(a.value*b.value);
}
inline Scalar operator/(const Scalar& a,const Scalar& b){
    if(b.node>=0||b.value==0.)throw std::runtime_error("Nonconstant native boundary denominator");
    return a*Scalar(1./b.value);
}
inline bool operator>(const Scalar& a,const Scalar& b){if(a.node>=0||b.node>=0)throw std::runtime_error("Field-dependent boundary branch");return a.value>b.value;}
inline bool operator<(const Scalar& a,const Scalar& b){return b>a;}
inline Scalar& Scalar::operator+=(const Scalar& x){return *this=*this+x;}
inline Scalar& Scalar::operator-=(const Scalar& x){return *this=*this-x;}
inline Scalar& Scalar::operator*=(const Scalar& x){return *this=*this*x;}
inline Scalar& Scalar::operator/=(const Scalar& x){return *this=*this/x;}
}

// First instantiate the actual operators with their normal binary64 types.
namespace native_double {
using real_t=double;
#include <fd_schemes.hh>
#include <mg_operators.hh>
#include <mg_interp.hh>
}

// Then instantiate the same headers with a linear-operation scalar. The
// flux methods explicitly return double in upstream MUSIC; only that type
// is adapted here. General/standard headers have already been included.
#undef __FD_SCHEMES_HH
#undef __MG_OPERATORS_HH
#undef __MG_INTERP_HH
namespace native_linear {
using real_t=boundary_tape::Scalar;
#define double boundary_tape::Scalar
#include <mg_operators.hh>
#include <fd_schemes.hh>
#undef double
#include <mg_interp.hh>
}

namespace boundary_adapter {
constexpr int halo=4;
uint64_t product(const std::array<int,3>& shape){
    uint64_t n=1;
    for(int size:shape){if(size<1||n>uint64_t(std::numeric_limits<int64_t>::max())/uint64_t(size))throw std::runtime_error("Native boundary geometry overflow");n*=size;}
    return n;
}
struct Geometry {
    std::array<int,3> shape,offset;
    uint64_t cells;
    Geometry(const int* n,const int* o):shape{{n[0],n[1],n[2]}},offset{{o[0],o[1],o[2]}} {
        for(int size:shape)if(size<4||size>(1<<20)||size%2)throw std::runtime_error("Even bounded native boundary dimensions required");
        auto stored=shape;for(auto& size:stored)size+=2*halo;cells=product(stored);
    }
    int size(int axis)const{return shape.at(axis);}
    int off(int axis)const{return offset.at(axis);}
    uint64_t index(int i,int j,int k)const{
        const int xyz[3]={i,j,k};for(int a=0;a<3;++a)if(xyz[a]<-halo||xyz[a]>=shape[a]+halo)throw std::runtime_error("Native boundary stencil escapes actual ghosts");
        return (uint64_t(i+halo)*(shape[1]+2*halo)+uint64_t(j+halo))*(shape[2]+2*halo)+uint64_t(k+halo);
    }
};
struct DenseGrid {
    int m_nbnd=halo;Geometry geometry;std::vector<double> values;
    DenseGrid(const Geometry& g,const double* data):geometry(g),values(data,data+g.cells){}
    int size(int a)const{return geometry.size(a);}int offset(int a)const{return geometry.off(a);}
    double& operator()(int i,int j,int k){return values[geometry.index(i,j,k)];}
    double operator()(int i,int j,int k)const{return values[geometry.index(i,j,k)];}
};
struct LinearGrid {
    int m_nbnd=halo;Geometry geometry;uint64_t base;std::unordered_map<uint64_t,boundary_tape::Scalar> changed;
    LinearGrid(const Geometry& g,uint64_t b):geometry(g),base(b){}
    int size(int a)const{return geometry.size(a);}int offset(int a)const{return geometry.off(a);}
    boundary_tape::Scalar& operator()(int i,int j,int k){
        auto index=geometry.index(i,j,k);auto it=changed.find(index);
        if(it==changed.end())it=changed.emplace(index,boundary_tape::Scalar(0.,int64_t(base+index))).first;
        return it->second;
    }
    boundary_tape::Scalar operator()(int i,int j,int k)const{
        auto index=geometry.index(i,j,k);auto it=changed.find(index);
        return it==changed.end()?boundary_tape::Scalar(0.,int64_t(base+index)):it->second;
    }
};

// Every zero-field grid sample is a linear variable, before and after the
// interpolation. Scalar arithmetic records operations by variable/constant
// status, never by sample identity. This projection therefore counts the same
// expressions without retaining node identities, grids or a tape. A constant
// or affine field write invalidates that invariant and rejects the counter.
struct CountGrid {
    struct Slot:boundary_tape::Scalar {
        Slot& operator=(const boundary_tape::Scalar& x){
            if(x.node<0||x.value!=0.)throw std::runtime_error("Native boundary counting requires homogeneous variable field writes");
            value=x.value;node=x.node;return *this;
        }
    };
    int m_nbnd=halo;Geometry geometry;Slot slot;
    explicit CountGrid(const Geometry& g):geometry(g){}
    int size(int a)const{return geometry.size(a);}int offset(int a)const{return geometry.off(a);}
    Slot& operator()(int i,int j,int k){geometry.index(i,j,k);slot.value=0.;slot.node=0;return slot;}
    boundary_tape::Scalar operator()(int i,int j,int k)const{geometry.index(i,j,k);return boundary_tape::Scalar(0.,0);}
};
struct CountScope {
    boundary_tape::Tape tape;boundary_tape::Tape* previous;
    CountScope():previous(boundary_tape::active){
        tape.inputs=1;tape.maximum_nodes=uint64_t(std::numeric_limits<int64_t>::max()/2);tape.count_only=true;boundary_tape::active=&tape;
    }
    ~CountScope(){boundary_tape::active=previous;}
};
template<int X,int Y,int Z> uint64_t cubic_count(){
    const int n[3]={32,32,32},zero[3]={0,0,0};CountGrid coarse(Geometry(n,zero));CountScope scope;
    const auto result=native_linear::mg_cubic().interp_cubic<X,Y,Z>(4,4,4,coarse);
    if(result.node<0||result.value!=0.)throw std::runtime_error("Unsupported native cubic counting invariant");
    return scope.tape.counted_nodes;
}
uint64_t count_kernel(int order,const std::array<int,3>& shape){
    const int cn[3]={32,32,32},zero[3]={0,0,0},offset[3]={8,8,8};
    CountGrid coarse(Geometry(cn,zero)),fine(Geometry(shape.data(),offset));CountScope scope;
    if(order==2)native_linear::interp_O3_fluxcorr().interp_coarse_fine(1,coarse,fine);
    else if(order==4)native_linear::interp_O5_fluxcorr().interp_coarse_fine(1,coarse,fine);
    else if(order==6)native_linear::interp_O7_fluxcorr().interp_coarse_fine(1,coarse,fine);
    else throw std::runtime_error("Unsupported native boundary counting order");
    return scope.tape.counted_nodes;
}
uint64_t ghost_cells(const std::array<int,3>& shape){auto padded=shape;for(auto& n:padded)n+=2*halo;return product(padded)-product(shape);}
uint64_t profile_nodes(const std::array<uint64_t,4>& profile,int order,const std::array<int,3>& shape,bool correction=true){
    uint64_t nodes=profile[0]*ghost_cells(shape);const int endpoint=order==2?0:1;
    if(correction)for(int axis=0;axis<3;++axis)nodes+=profile[axis+1]*uint64_t(shape[(axis+1)%3]/2+endpoint)*uint64_t(shape[(axis+2)%3]/2+endpoint);
    return nodes;
}
std::array<uint64_t,4> count_profile(int order){
    const uint64_t c=cubic_count<0,0,0>();
    const uint64_t parities[]={cubic_count<0,0,1>(),cubic_count<0,1,0>(),cubic_count<0,1,1>(),
        cubic_count<1,0,0>(),cubic_count<1,0,1>(),cubic_count<1,1,0>(),cubic_count<1,1,1>()};
    for(auto n:parities)if(n!=c)throw std::runtime_error("Native cubic operation count depends on parity");
    const std::array<int,3> base{{8,8,8}};const uint64_t h=order==2?4:5;
    const uint64_t kernel=count_kernel(order,base),cubic=c*ghost_cells(base);
    if(kernel<cubic||(kernel-cubic)%(h*h))throw std::runtime_error("Unsupported native face counting structure");
    const uint64_t remainder=kernel-cubic,total=remainder/(h*h);
    std::array<uint64_t,4> profile{{c,0,0,0}};
    for(int axis=0;axis<3;++axis){
        auto shape=base;shape[axis]+=2;const uint64_t n=count_kernel(order,shape),g=c*ghost_cells(shape);
        if(n<g+remainder||(n-g-remainder)%h)throw std::runtime_error("Unsupported native face counting increments");
        const uint64_t other=(n-g-remainder)/h;
        if(other>total)throw std::runtime_error("Invalid native face operation count");profile[axis+1]=total-other;
    }
    if(profile[1]+profile[2]+profile[3]!=total)throw std::runtime_error("Native face count partition mismatch");
    for(auto shape:{std::array<int,3>{{6,12,14}},std::array<int,3>{{14,10,8}}})
        if(profile_nodes(profile,order,shape)!=count_kernel(order,shape))throw std::runtime_error("Native rectangular face count certificate failed");
    return profile;
}
struct Handle {
    Geometry coarse,fine;int order;uint64_t budget;
    boundary_tape::Tape tape;
    std::vector<uint64_t> output_cells;std::vector<int64_t> output_nodes;
    Handle(int order_,const int* cn,const int* fn,const int* offset,uint64_t bytes):
        coarse(cn,std::array<int,3>{{0,0,0}}.data()),fine(fn,offset),order(order_),budget(bytes) {
        if(order!=2&&order!=4&&order!=6)throw std::runtime_error("Native boundary order must be2,4 or6");
        for(int a=0;a<3;++a)if(offset[a]<0||uint64_t(offset[a])+fine.shape[a]/2>uint64_t(coarse.shape[a]))throw std::runtime_error("Fine boundary footprint escapes parent interior");
        uint64_t inner=1;for(int n:fine.shape)inner*=uint64_t(std::max(0,n-8));
        uint64_t band=fine.cells-inner;
        const long double reserve=64.L*1024*1024+128.L*band+32.L*(coarse.cells+fine.cells);
        if(bytes<=reserve+1024*96)throw std::runtime_error("Native boundary preflight exceeds declared memory");
        tape.inputs=coarse.cells+fine.cells;
        if(tape.inputs>uint64_t(std::numeric_limits<int64_t>::max()/2))throw std::runtime_error("Native boundary tape identity overflow");
        tape.maximum_nodes=uint64_t((bytes-reserve)/96);
        LinearGrid c(coarse,0),f(fine,coarse.cells);boundary_tape::active=&tape;
        try {
            if(order==2)native_linear::interp_O3_fluxcorr().interp_coarse_fine(1,c,f);
            if(order==4)native_linear::interp_O5_fluxcorr().interp_coarse_fine(1,c,f);
            if(order==6)native_linear::interp_O7_fluxcorr().interp_coarse_fine(1,c,f);
            const int radius=order/2;
            for(int axis=0;axis<3;++axis)for(int side:{-1,1})for(int distance=1;distance<=radius;++distance)
                for(int u=0;u<fine.shape[(axis+1)%3];++u)for(int v=0;v<fine.shape[(axis+2)%3];++v){
                    int xyz[3];xyz[axis]=side<0?-distance:fine.shape[axis]+distance-1;xyz[(axis+1)%3]=u;xyz[(axis+2)%3]=v;
                    auto value=f(xyz[0],xyz[1],xyz[2]);if(value.value!=0.)throw std::runtime_error("Affine native boundary intercept is unsupported");
                    output_cells.push_back(fine.index(xyz[0],xyz[1],xyz[2]));output_nodes.push_back(value.node);
                }
        }catch(...){boundary_tape::active=nullptr;throw;}
        boundary_tape::active=nullptr;
    }
};
void error(char* target,uint64_t bytes,const std::exception& e){if(target&&bytes){std::strncpy(target,e.what(),size_t(bytes-1));target[bytes-1]=0;}}
double value(int64_t index,const double* inputs,const std::vector<double>& nodes,uint64_t count){return index<0?0.:(uint64_t(index)<count?inputs[index]:nodes.at(uint64_t(index)-count));}
}

extern "C" {
uint64_t music_boundary_abi_version(){return 1;}
int music_boundary_count_profile(int order,uint64_t* output,char* err,uint64_t errbytes){
    try{const auto profile=boundary_adapter::count_profile(order);std::copy(profile.begin(),profile.end(),output);return 0;}
    catch(const std::exception& e){boundary_adapter::error(err,errbytes,e);return 1;}
}
void* music_boundary_create(int order,const int* coarse,const int* fine,const int* offset,uint64_t budget,char* err,uint64_t errbytes){
    try{return new boundary_adapter::Handle(order,coarse,fine,offset,budget);}catch(const std::exception& e){boundary_adapter::error(err,errbytes,e);return nullptr;}
}
void music_boundary_destroy(void* p){delete static_cast<boundary_adapter::Handle*>(p);}
uint64_t music_boundary_inputs(void* p){return static_cast<boundary_adapter::Handle*>(p)->tape.inputs;}
uint64_t music_boundary_outputs(void* p){return static_cast<boundary_adapter::Handle*>(p)->output_nodes.size();}
uint64_t music_boundary_nodes(void* p){return static_cast<boundary_adapter::Handle*>(p)->tape.nodes.size();}
void music_boundary_output_indices(void* p,uint64_t* indices){auto& h=*static_cast<boundary_adapter::Handle*>(p);std::copy(h.output_cells.begin(),h.output_cells.end(),indices);}
int music_boundary_apply(void* p,const double* input,double* output,char* err,uint64_t errbytes){
    try{auto& h=*static_cast<boundary_adapter::Handle*>(p);std::vector<double> values(h.tape.nodes.size());
        for(uint64_t i=0;i<values.size();++i){const auto& n=h.tape.nodes[i];values[i]=n.wa*boundary_adapter::value(n.a,input,values,h.tape.inputs)+n.wb*boundary_adapter::value(n.b,input,values,h.tape.inputs);}
        for(uint64_t i=0;i<h.output_nodes.size();++i)output[i]=boundary_adapter::value(h.output_nodes[i],input,values,h.tape.inputs);return 0;
    }catch(const std::exception& e){boundary_adapter::error(err,errbytes,e);return 1;}
}
int music_boundary_transpose(void* p,const double* weight,double* output,char* err,uint64_t errbytes){
    try{auto& h=*static_cast<boundary_adapter::Handle*>(p);std::vector<double> gradient(h.tape.nodes.size(),0.);std::fill(output,output+h.tape.inputs,0.);
        auto add=[&](int64_t index,double v){if(index<0)return;if(uint64_t(index)<h.tape.inputs)output[index]+=v;else gradient.at(uint64_t(index)-h.tape.inputs)+=v;};
        for(uint64_t i=0;i<h.output_nodes.size();++i)add(h.output_nodes[i],weight[i]);
        for(uint64_t i=gradient.size();i-->0;){const auto& n=h.tape.nodes[i];add(n.a,n.wa*gradient[i]);add(n.b,n.wb*gradient[i]);}return 0;
    }catch(const std::exception& e){boundary_adapter::error(err,errbytes,e);return 1;}
}
int music_boundary_direct(void* p,const double* input,double* output,char* err,uint64_t errbytes){
    try{auto& h=*static_cast<boundary_adapter::Handle*>(p);boundary_adapter::DenseGrid c(h.coarse,input),f(h.fine,input+h.coarse.cells);
        if(h.order==2)native_double::interp_O3_fluxcorr().interp_coarse_fine(1,c,f);
        if(h.order==4)native_double::interp_O5_fluxcorr().interp_coarse_fine(1,c,f);
        if(h.order==6)native_double::interp_O7_fluxcorr().interp_coarse_fine(1,c,f);
        for(uint64_t i=0;i<h.output_cells.size();++i)output[i]=f.values.at(h.output_cells[i]);return 0;
    }catch(const std::exception& e){boundary_adapter::error(err,errbytes,e);return 1;}
}
}
