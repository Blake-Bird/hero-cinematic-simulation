#include "hierarchy_source.hh"
#include <fstream>
#include <iostream>
using namespace hierarchy_source;
int main(int argc,char **argv){
 try{
  if(argc!=14)return 2;
  std::vector<Grid> input;
  for(int argument=1;argument<=3;++argument){
   std::ifstream f(argv[argument],std::ios::binary);int32_t meta[8];f.read(reinterpret_cast<char*>(meta),sizeof(meta));
   if(!f||meta[0]<2||meta[0]>30||meta[7]<0||meta[7]>16)throw std::runtime_error("Bad hierarchy dump metadata");
   Shape shape,stored;for(int axis=0;axis<3;++axis){if(meta[axis+1]<2||meta[axis+1]>256)throw std::runtime_error("Fixture input budget");shape[axis]=meta[axis+1];stored[axis]=meta[axis+1]+2*meta[7];}
   Box values(shape);
   for(size_t i=0;i<stored[0];++i)for(size_t j=0;j<stored[1];++j)for(size_t k=0;k<stored[2];++k){double value;f.read(reinterpret_cast<char*>(&value),sizeof(value));if(!f)throw std::runtime_error("Truncated hierarchy dump");
    if(i>=size_t(meta[7])&&i<size_t(meta[7])+shape[0]&&j>=size_t(meta[7])&&j<size_t(meta[7])+shape[1]&&k>=size_t(meta[7])&&k<size_t(meta[7])+shape[2])values.at(i-meta[7],j-meta[7],k-meta[7])=value;
   }
   if(f.peek()!=std::char_traits<char>::eof())throw std::runtime_error("Trailing hierarchy dump");
   input.push_back({unsigned(meta[0]),{{meta[4],meta[5],meta[6]}},std::move(values)});
  }
  const int degree=std::stoi(argv[4]);Coord origin={{std::stoll(argv[5]),std::stoll(argv[6]),std::stoll(argv[7])}};
  Shape shape={{std::stoull(argv[8]),std::stoull(argv[9]),std::stoull(argv[10])}};
  Sampler sampler(input,degree,std::stoull(argv[11]));auto output=sampler.sample(origin,shape);
  std::ifstream reference(argv[12],std::ios::binary);unsigned side;reference.read(reinterpret_cast<char*>(&side),sizeof(side));if(!reference||shape[0]!=side||shape[1]!=side||shape[2]!=side)throw std::runtime_error("Reference fixture geometry mismatch");
  long double numerator=0,denominator=0;double maximum=0;
  for(double value:output.values){double expected;reference.read(reinterpret_cast<char*>(&expected),sizeof(expected));if(!reference||!std::isfinite(expected))throw std::runtime_error("Reference payload mismatch");const double difference=value-expected;maximum=std::max(maximum,std::abs(difference));numerator+=difference*difference;denominator+=expected*expected;}
  if(reference.peek()!=std::char_traits<char>::eof())throw std::runtime_error("Trailing reference payload");
  std::ofstream saved(argv[13],std::ios::binary);saved.write(reinterpret_cast<char*>(&side),sizeof(side));saved.write(reinterpret_cast<const char*>(output.values.data()),output.values.size()*sizeof(double));if(!saved)throw std::runtime_error("Output fixture failed");
  double relative=std::sqrt(numerator/std::max(denominator,1.e-100L));std::cout<<"BOUNDED_SOURCE cells="<<output.values.size()<<" degree="<<degree<<" relative_l2="<<relative<<" max_abs="<<maximum<<'\n';
  return relative<1.e-12?0:1;
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
