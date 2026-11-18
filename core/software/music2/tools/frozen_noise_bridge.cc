// Reuse the established MUSIC generator to bridge a frozen periodic field.
#include "plugins/random_music_wnoise_generator.hh"
#include "plugins/frozen_noise_validation.hh"
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <cstdint>
#include <cmath>
#include <memory>
#include <vector>

using noise = music_wnoise_generator<real_t>;
int main(int argc,char**argv) {
  try {
    if(argc!=7) throw std::runtime_error("usage: bridge parent_level input.grafic new_seed output.grafic memory_mib metrics.json");
    int level=std::stoi(argv[1]);long seed=std::stol(argv[3]);double memory=std::stod(argv[5]);
    if(level<5 || level>9 || seed<=0 || seed>2147483647L || !std::isfinite(memory) || memory<=0)
      throw std::runtime_error("unsupported bridge level, seed or memory allocation");
    unsigned n=1u<<level, fine_n=2*n;
    double bound=12.0*sizeof(double)*(double(n)*n*n+double(fine_n)*fine_n*fine_n);
    if(bound>memory*1024*1024) throw std::runtime_error("bridge working-set estimate exceeds allocation");
    if(std::filesystem::exists(argv[4]) || std::filesystem::exists(argv[6]))
      throw std::runtime_error("keep previous bridge artifacts");
    noise parent(n,std::string(argv[2]),true);
    noise fine(parent,32,seed);
    validate_frozen_low_modes(parent,fine);
    noise restricted(fine);
    long double sum=0,den=0;double maximum=0;
    for(unsigned i=0;i<n;i++)for(unsigned j=0;j<n;j++)for(unsigned k=0;k<n;k++) {
      double before=parent(i,j,k),difference=restricted(i,j,k)-before;
      if(!std::isfinite(difference))throw std::runtime_error("nonfinite bridge restriction diagnostic");
      sum+=difference*difference;den+=before*before;maximum=std::max(maximum,std::abs(difference));
    }
    if(den<=0)throw std::runtime_error("empty parent field");
    std::filesystem::path temporary=std::string(argv[4])+".partial";
    if(std::filesystem::exists(temporary))throw std::runtime_error("keep partial bridge evidence");
    std::ofstream out(temporary,std::ios::binary);
    std::int32_t header[6]={16,static_cast<std::int32_t>(fine_n),static_cast<std::int32_t>(fine_n),static_cast<std::int32_t>(fine_n),0,16};
    out.write(reinterpret_cast<char*>(header),sizeof(header));
    std::vector<double> plane(size_t(fine_n)*fine_n);
    std::int32_t marker=static_cast<std::int32_t>(plane.size()*sizeof(double));
    for(unsigned k=0;k<fine_n;k++) {
      for(unsigned j=0;j<fine_n;j++)for(unsigned i=0;i<fine_n;i++)plane[size_t(j)*fine_n+i]=fine(i,j,k);
      out.write(reinterpret_cast<char*>(&marker),4);out.write(reinterpret_cast<char*>(plane.data()),marker);out.write(reinterpret_cast<char*>(&marker),4);
    }
    out.close();if(!out)throw std::runtime_error("bridge file write failed");
    std::filesystem::rename(temporary,argv[4]);
    std::ofstream metrics(argv[6]);metrics<<std::setprecision(17)
      <<"{\"parent_level\":"<<level<<",\"bridge_level\":"<<level+1
      <<",\"seed\":"<<seed<<",\"restriction_relative_l2\":"<<double(std::sqrt(sum/den))
      <<",\"restriction_maximum_absolute\":"<<maximum
      <<",\"working_set_estimate_bytes\":"<<bound<<",\"numerical_acceptance\":\"NOT_ACCEPTED\"}\n";
    if(!metrics)throw std::runtime_error("bridge metrics write failed");
    return 0;
  } catch(const std::exception&error) {std::cerr<<error.what()<<std::endl;return 1;}
}
