// Header-format control only: no particle population or IC is synthesized.
#include <iostream>
#include <string>
#include "particle_count_header.hh"
int main(int argc,char **argv) {
    if(argc!=2)return 2;
    const std::string base=argv[1];
    const std::vector<long long> total={14179436936LL,14179436936LL,69723767LL,0,0,0};
    const std::vector<unsigned long long> full(total.begin(),total.end());
    for(bool modern:{false,true}) {
        const std::string file=base+(modern?"-modern.hdf5":"-legacy-shard.hdf5");
        HDFCreateFile(file);HDFCreateGroup(file,"Header");
        auto local=full;
        if(!modern)for(auto &count:local)count/=4;
        music_particle_counts::write(file,local,total,modern);
        HDFWriteGroupAttribute(file,"Header","CountContractOnly",1);
        hid_t handle=H5Fopen(file.c_str(),H5F_ACC_RDONLY,H5P_DEFAULT);
        hid_t group=H5Gopen(handle,"Header");
        std::vector<unsigned long long> observed(6),counts(6);std::vector<unsigned int> high(6);
        hid_t attribute=H5Aopen_name(group,"NumPart_ThisFile"),type=H5Aget_type(attribute);
        if(H5Tget_class(type)!=H5T_INTEGER || H5Tget_sign(type)!=H5T_SGN_NONE || H5Tget_size(type)!=(modern?8:4))return 5;
        H5Aread(attribute,H5T_NATIVE_ULLONG,observed.data());H5Tclose(type);H5Aclose(attribute);
        attribute=H5Aopen_name(group,"NumPart_Total");H5Aread(attribute,H5T_NATIVE_ULLONG,counts.data());H5Aclose(attribute);
        attribute=H5Aopen_name(group,"NumPart_Total_HighWord");H5Aread(attribute,H5T_NATIVE_UINT,high.data());H5Aclose(attribute);
        H5Gclose(group);H5Fclose(handle);
        for(size_t i=0;i<6;++i)if(observed[i]!=local[i] || counts[i]+(static_cast<unsigned long long>(high[i])<<32)!=full[i])return 6;
    }
    const std::string invalid=base+"-invalid-legacy.hdf5";
    HDFCreateFile(invalid);HDFCreateGroup(invalid,"Header");
    bool rejected=false;
    try{music_particle_counts::write(invalid,full,total,false);}catch(const std::runtime_error &){rejected=true;}
    if(!rejected)return 3;
    auto bad=full;bad[0]++;
    try{music_particle_counts::write(invalid,bad,total,true);return 4;}catch(const std::runtime_error &){}
    std::cout<<"PASS: full uint64 and legacy shard count attributes; invalid/truncated counts rejected\n";
    return 0;
}
