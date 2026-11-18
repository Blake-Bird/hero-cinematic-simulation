/* Original Research2 network bookkeeping. Reactions/rates are solved by Grackle.
 * Atomic lanes: HI,HII,HeI,HeII,HeIII,e(m_H equivalent).
 * Molecular lanes: HM,H2I,H2II,DI,DII,HDI (mass densities).
 * Element H includes protium+deuterium; electrons are not baryonic mass.
 */
#ifndef RESEARCH_PRIMORDIAL_NETWORK_H
#define RESEARCH_PRIMORDIAL_NETWORK_H
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>
#include "../data/research_source_state.h"
namespace research_primordial {
inline int version(int network){return network==0?2:network==1?1:network+1;}
inline double hydrogen(const double *x){return x[0]+x[1]+x[6]+x[7]+x[8]+x[11]/3;}
inline double helium(const double *x){return x[2]+x[3]+x[4];}
inline double deuterium(const double *x){return x[9]+x[10]+2*x[11]/3;}
inline double electrons(const double *x){return x[1]+x[3]/4+x[4]/2+x[8]/2+x[10]/2-x[6];}
inline void seed(research_source_state &s,const double *number,double ratio,int network) {
  for(int k=0;k<12;k++)if(!std::isfinite(number[k]) || number[k]<0)throw std::runtime_error("Invalid primordial seed abundance");
  const double h=number[0]+number[1]+number[6]+2*(number[7]+number[8])+number[11];
  const double he=number[2]+number[3]+number[4],d=number[9]+number[10]+number[11];
  if(std::abs(h-1)>1e-12 || std::abs(he-1)>1e-12 || std::abs(d-ratio)>1e-12 || number[5]!=0)
    throw std::runtime_error("Primordial seed nuclear fractions fail their declared H/He/D normalization");
  if(network<2 && (number[6] || number[7] || number[8]) || network<3 && d!=0)
    throw std::runtime_error("Primordial seed species lie outside the chosen network");
  const double hydrogen_mass=s.element_mass_fraction[0]/(1+2*ratio),helium_mass=s.element_mass_fraction[1];
  double x[12]={};
  for(int k:{0,1,6})x[k]=hydrogen_mass*number[k];
  for(int k:{2,3,4})x[k]=helium_mass*number[k];
  for(int k:{7,8,9,10})x[k]=2*hydrogen_mass*number[k];
  x[11]=3*hydrogen_mass*number[11];x[5]=electrons(x);
  if(x[5]<0)throw std::runtime_error("Primordial seed charge is negative");
  for(int k=0;k<6;k++){s.grackle_species_mass_fraction[k]=x[k];s.grackle_molecular_mass_fraction[k]=x[k+6];}
}
inline void restore(const research_source_state &s,double density,double ratio,int network,double *x) {
  for(int k=0;k<6;k++){x[k]=s.grackle_species_mass_fraction[k]*density;x[k+6]=s.grackle_molecular_mass_fraction[k]*density;}
  if(hydrogen(x)+helium(x)+deuterium(x)==0){x[0]=density*s.element_mass_fraction[0]/(1+2*ratio);x[2]=density*s.element_mass_fraction[1];x[9]=2*ratio*x[0];}
  if(network<2){x[6]=x[7]=x[8]=0;}
  if(network<3){x[9]=x[10]=x[11]=0;}
  const double target_d=deuterium(x),target_h=density*s.element_mass_fraction[0]-target_d,target_he=density*s.element_mass_fraction[1];
  if(target_h<0)throw std::runtime_error("Transported deuterium exceeds the measured hydrogen-isotope reservoir");
  const double current_h=hydrogen(x),current_he=helium(x),current_d=deuterium(x);
  // Preserve HD's shared nuclei and the transported deuterium reservoir.
  // Stellar yields add neutral H/He through conservative deposition, not D.
  const double hd_h=x[11]/3,pure_h=current_h-hd_h;
  if(target_h<hd_h)throw std::runtime_error("HD exceeds the enriched measured protium reservoir");
  const double scale_h=pure_h>0?(target_h-hd_h)/pure_h:0,scale_d=current_d>0?target_d/current_d:0;
  for(int k:{0,1,6,7,8})x[k]*=scale_h;
  if(pure_h==0)x[0]=target_h-hd_h;
  for(int k:{2,3,4})x[k]*=current_he>0?target_he/current_he:0;
  if(current_he==0)x[2]=target_he;
  for(int k:{9,10})x[k]*=scale_d;
  if(current_d==0)x[9]=target_d;
  x[5]=electrons(x);
}
// In Grackle 3.4.1 make_consistent_g, the H and He totals are forced
// to sum to solver_baryons before deuterium is accounted for. H in HD
// is also excluded from that H normalization. Validate the expected
// upstream convention before converting to the transported reservoirs.
inline void undo_grackle_hd_normalization(double *x,double h,double he,double d,double solver_baryons) {
  const double hd_h=x[11]/3;
  const double scale=(h+he)/solver_baryons;
  if(!std::isfinite(scale) || scale<=0 || h<=0 || hd_h<0 || hd_h>=h)
    throw std::runtime_error("Invalid HD normalization reservoirs");
  for(auto pair:{std::pair<double,double>{hydrogen(x)-hd_h,h/scale},{helium(x),he/scale},{deuterium(x),d}})
    if(!std::isfinite(pair.first) || std::abs(pair.first-pair.second)>1e-6*std::max(pair.second,1e-30))
      throw std::runtime_error("Grackle returned unexpected H/He/D normalization");
  const double hydrogen_scale=scale*(1-hd_h/h);
  for(int k:{0,1,6,7,8})x[k]*=hydrogen_scale;
  for(int k:{2,3,4})x[k]*=scale;
}
inline void conserve(double *x,double target_h,double target_he,double target_d,int network) {
  for(int k=0;k<12;k++)if(!std::isfinite(x[k]) || x[k]<0)throw std::runtime_error("Grackle returned invalid primordial species");
  for(auto pair:{std::pair<double,double>{hydrogen(x),target_h},{helium(x),target_he},{deuterium(x),target_d}})
    if(std::abs(pair.first-pair.second)>1e-6*std::max(pair.second,1e-30))
      throw std::runtime_error("Grackle violated H/He/D nucleus conservation");
  const double electron=electrons(x);
  if(electron<0 || !std::isfinite(electron))throw std::runtime_error("Grackle species violate nonnegative charge balance");
  x[5]=electron; // charge balance, not artificial molecule/atom renormalization
  if(network<2 && (x[6] || x[7] || x[8]) || network<3 && (x[9] || x[10] || x[11]))
    throw std::runtime_error("Species lie outside their recorded primordial network");
}
inline double inverse_mu(const double *x,double density){
  return (x[0]+x[1]+x[6]+x[5]+(x[2]+x[3]+x[4])/4+(x[7]+x[8]+x[9]+x[10])/2+x[11]/3)/density;
}
}
#endif
