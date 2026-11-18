/* Research2 BH dynamics; original integration, fitted formulae adapted from
 * Davide Gerosa PRECESSION cf850140d1a6af1594d63ec996d4fd2c752fafef (MIT).
 * Copyright (c) 2020 Davide Gerosa. Permission is hereby granted, free of charge,
 * to any person obtaining a copy of this software and associated documentation
 * files (the "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish, distribute,
 * sublicense, and/or sell copies of the Software, and to permit persons to whom
 * the Software is furnished to do so, subject to the following conditions:
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED
 * "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT
 * LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
 * PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
 * HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
 * ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 * WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 * All private binary lengths, clocks, angular momenta and energies use CGS.
 * The supported inspiral is quasi-circular (capture e <= declared <= .05).
 * Larger eccentricity and uncalibrated SIDM backgrounds fail explicitly.
 */
#ifndef RESEARCH_BH_DYNAMICS_H
#define RESEARCH_BH_DYNAMICS_H
#include <array>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <sstream>
#include <vector>
#include "../data/research_stellar_birth.h"
#include "../time_integration/driftfac.h"
#include "research_restart_fingerprint.h"
namespace research_bh {
constexpr double G=6.67430e-8,c=2.99792458e10,solar=1.98847e33,kpc=3.0856775814913673e21;
using vec=std::array<double,3>;
inline vec add(vec a,vec b){for(int k=0;k<3;k++)a[k]+=b[k];return a;}
inline vec mul(vec a,double s){for(double &x:a)x*=s;return a;}
inline double dot(vec a,vec b){double s=0;for(int k=0;k<3;k++)s+=a[k]*b[k];return s;}
inline vec cross(vec a,vec b){return {{a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]}};}
inline double norm(vec a){return std::sqrt(dot(a,a));}
inline vec unit(vec a){double n=norm(a);if(!(n>0) || !std::isfinite(n))Terminate("BH invalid vector normalization");return mul(a,1/n);}
inline double proper_seconds(double begin,double end) {
  if(!(end>=begin) || !std::isfinite(begin+end))Terminate("BH clock regressed");
  if(!All.ComovingIntegrationOn)return (end-begin)*All.UnitTime_in_s;
  if(begin<=0)return 0;
  double l=std::log(begin),r=std::log(end),previous=-1;
  for(int n=16;n<=16384;n*=2){double sum=0,h=(r-l)/n;
    for(int j=0;j<=n;j++){double inverse=1/driftfac::hubble_function(std::exp(l+j*h));if(!(inverse>0))Terminate("BH invalid native expansion rate");sum+=(j==0 || j==n ? 1 : j%2 ? 4 : 2)*inverse;}
    double now=sum*h/3*All.UnitTime_in_s/All.HubbleParam;
    if(previous>=0 && fabs(now-previous)<=1.e-10*std::max(1.,now))return now;previous=now;
  }Terminate("BH proper clock integration failed to converge");return 0;
}
inline double softened_potential(double r,double support){
  if(!(support>0))Terminate("BH capture requires finite force softening");
  double u=r/support;if(u>=1)return -1/r;
  if(u<.5)return (-2.8+u*u*(16./3+u*u*(6.4*u-9.6)))/support;
  return (-3.2+1/(15*u)+u*u*(32./3+u*(-16+u*(9.6-64./30*u))))/support;
}
inline uint64_t keyed_phase(uint64_t a,uint64_t b,uint64_t counter,uint64_t seed){
  uint64_t x=a^(b+UINT64_C(0x9e3779b97f4a7c15))^counter^seed;
  x=(x^(x>>30))*UINT64_C(0xbf58476d1ce4e5b9);x=(x^(x>>27))*UINT64_C(0x94d049bb133111eb);return x^(x>>31);
}
struct remnant {double mass,radiated;vec spin,kick;};
inline std::array<double,2> isco(double a){
  if(!std::isfinite(a) || fabs(a)>1)Terminate("BH fit effective spin outside Kerr domain");
  double z1=1+std::cbrt(1-a*a)*(std::cbrt(1+a)+std::cbrt(1-a)),z2=sqrt(3*a*a+z1*z1);
  double r=3+z2-(a>0 ? 1 : a<0 ? -1 : 0)*sqrt(std::max(0.,(3-z1)*(3+z1+2*z2)));
  return {{sqrt(1-2/(3*r)),2/(3*sqrt(3.))*(1+2*sqrt(3*r-2))}};
}
inline remnant fit(double m1,double m2,vec s1,vec s2,vec L,vec radial,double phase){
  bool swapped=m2>m1;if(swapped){std::swap(m1,m2);std::swap(s1,s2);radial=mul(radial,-1);}
  L=unit(L);vec ex=unit(add(radial,mul(L,-dot(radial,L)))),ey=cross(L,ex);
  double q=m2/m1,eta=q/((1+q)*(1+q)),chi1=norm(s1),chi2=norm(s2);
  if(q<.05 || chi1>.99 || chi2>.99)Terminate("BH merger outside pinned analytic-fit domain");
  double ct1=chi1 ? std::max(-1.,std::min(1.,dot(s1,L)/chi1)) : 1;
  double ct2=chi2 ? std::max(-1.,std::min(1.,dot(s2,L)/chi2)) : 1;
  double atot=(dot(s1,L)+q*q*dot(s2,L))/((1+q)*(1+q)),ei=isco(atot)[0];
  double lost=eta*(1-ei)+4*eta*eta*(4*.04827+16*.01707*atot*(atot+1)+ei-1);
  double theta1=acos(ct1),theta2=acos(ct2);ct1=cos(theta1+.024*sin(theta1));ct2=cos(theta2+.024*sin(theta2));
  atot=(chi1*ct1+q*q*chi2*ct2)/((1+q)*(1+q));double ae=atot+.474046*eta*(chi1*ct1+chi2*ct2);auto ii=isco(ae);
  double coeff[4][5]={{0,3.39221,4.48865,-5.77101,-13.0459},{35.1278,-72.9336,-86.0036,93.7371,200.975},{-146.822,387.184,447.009,-467.383,-884.339},{223.911,-648.502,-697.177,753.738,1166.89}};
  coeff[0][0]=16*(.68646-coeff[1][0]/64-coeff[2][0]/256-coeff[3][0]/1024-sqrt(3.)/2);
  double ell=ii[1]-2*atot*(ii[0]-1);for(int i=0;i<4;i++)for(int j=0;j<5;j++)ell+=coeff[i][j]*pow(eta,i+1)*pow(ae,j);ell=fabs(ell);
  double mag2=(chi1*chi1+pow(q,4)*chi2*chi2+2*q*q*dot(s1,s2)+2*(chi1*ct1+q*q*chi2*ct2)*ell*q+ell*ell*q*q)/pow(1+q,4);
  vec direction=add(mul(L,eta*sqrt(10.)),mul(add(s1,mul(s2,q*q)),1/((1+q)*(1+q))));
  if(!(lost>0 && lost<1 && mag2>=0 && mag2<=1))Terminate("BH fit returned an unphysical remnant");
  vec delta=mul(add(s1,mul(s2,-q)),1/(1+q)),chit=mul(add(s1,mul(s2,q*q)),1/((1+q)*(1+q)));
  double dp=dot(delta,L),dt=norm(cross(delta,L)),sp=dot(chit,L),st=norm(cross(chit,L));
  double vm=1.2e4*eta*eta*(1-.93*eta)*(1-q)/(1+q),vp=6.9e3*eta*eta*dp;
  double vz=16*eta*eta*(dt*(3677.76+2*2481.21*sp+4*1792.45*sp*sp+8*1506.52*pow(sp,3))+st*dp*(2*1140+4*2481*sp))*cos(phase);
  double zeta=145*M_PI/180;vec kick=mul(add(add(mul(ex,vm+vp*cos(zeta)),mul(ey,vp*sin(zeta))),mul(L,vz)),1.e5);
  return {(m1+m2)*(1-lost),(m1+m2)*lost,mul(unit(direction),sqrt(mag2)),kick};
}
/* Orbit-averaged 1.5PN+2PN spin transport. a evolves using Peters plus
 * explicit stellar scattering; fractional changes and spin rotation are bounded.
 * Near-circular approximation is declared and e>.05 never silently accepted. */
inline void spin_rhs(double m1,double m2,double a,vec s1,vec s2,vec L,vec &d1,vec &d2,vec &dL){
  double q=m2/m1,eta=q/pow(1+q,2),v=sqrt(G*(m1+m2)/(a*c*c)),clock=c*c*c/(G*(m1+m2));
  vec S1=mul(s1,1/pow(1+q,2)),S2=mul(s2,q*q/pow(1+q,2));
  vec o1=add(mul(L,eta*pow(v,5)*(2+1.5*q)),mul(add(S2,mul(L,-3*dot(S2,L)-3*q*dot(S1,L))),.5*pow(v,6)));
  vec o2=add(mul(L,eta*pow(v,5)*(2+1.5/q)),mul(add(S1,mul(L,-3*dot(S1,L)-3*dot(S2,L)/q)),.5*pow(v,6)));
  d1=mul(cross(o1,s1),clock);d2=mul(cross(o2,s2),clock);
  dL=mul(add(mul(d1,1/pow(1+q,2)),mul(d2,q*q/pow(1+q,2))),-v/eta);
}
inline void advance_spins(double m1,double m2,double a,double dt,vec &s1,vec &s2,vec &L,vec &radial,double &phase){
  vec k1,k2,kL,j1,j2,jL;spin_rhs(m1,m2,a,s1,s2,L,k1,k2,kL);
  spin_rhs(m1,m2,a,add(s1,mul(k1,dt/2)),add(s2,mul(k2,dt/2)),unit(add(L,mul(kL,dt/2))),j1,j2,jL);
  double c1=norm(s1),c2=norm(s2);s1=add(s1,mul(j1,dt));s2=add(s2,mul(j2,dt));
  if(c1)s1=mul(unit(s1),c1);if(c2)s2=mul(unit(s2),c2);L=unit(add(L,mul(jL,dt)));
  radial=unit(add(radial,mul(L,-dot(radial,L))));
  double angle=fmod(sqrt(G*(m1+m2)/(a*a*a))*dt,2*M_PI);radial=add(mul(radial,cos(angle)),mul(cross(L,radial),sin(angle)));phase=fmod(phase+angle,2*M_PI);
}
inline std::string scalar_json(double value){if(!std::isfinite(value))Terminate("Nonfinite BH journal scalar");std::ostringstream out;out<<std::setprecision(17)<<value;return out.str();}
inline std::string vector_json(vec a){std::ostringstream s;s<<std::setprecision(17)<<'['<<a[0]<<','<<a[1]<<','<<a[2]<<']';return s.str();}
inline void append(std::vector<std::string>&rows,const std::string &row){
  if((rows.size()+1)*8192.>All.MaxMemSize*1024.*1024.*.125 || row.size()>8192)Terminate("BH journal memory allowance exhausted");rows.push_back(row+"\n");
}
inline void publish(const std::vector<std::string>&rows,MPI_Comm comm){
  int rank,ranks;MPI_Comm_rank(comm,&rank);MPI_Comm_size(comm,&ranks);uint64_t n=rows.size(),total=0;MPI_Allreduce(&n,&total,1,MPI_UINT64_T,MPI_SUM,comm);if(!total)return;
  if(All.BHEventCommitCount==UINT64_MAX || All.BHEventCount>UINT64_MAX-total)Terminate("BH journal counter overflow");
  uint64_t seq=All.BHEventCommitCount+1,bytes=0,crc=0;char name[160];snprintf(name,sizeof(name),"black_holes_%020llu.rank_%06d.of_%06d.jsonl",(unsigned long long)seq,rank,ranks);
  int good=1;if(n)good=research_stellar_publish_file(std::string(All.OutputDir)+"/"+name,[&](FILE*f){for(auto &row:rows)if(fwrite(row.data(),1,row.size(),f)!=row.size())return false;return true;},bytes,crc);
  int ok;MPI_Allreduce(&good,&ok,1,MPI_INT,MPI_MIN,comm);if(!ok)Terminate("BH event payload publication failed");
  uint64_t tuple[3]={n,bytes,crc};std::vector<uint64_t> all(size_t(ranks)*3);MPI_Allgather(tuple,3,MPI_UINT64_T,all.data(),3,MPI_UINT64_T,comm);
  if(rank==0){snprintf(name,sizeof(name),"black_holes_%020llu.commit.json",(unsigned long long)seq);good=research_stellar_publish_file(std::string(All.OutputDir)+"/"+name,[&](FILE*f){
    fprintf(f,"{\"schema\":\"native-black-hole-events-v1\",\"sequence\":%llu,\"previous_count\":%llu,\"cumulative_count\":%llu,\"event_count\":%llu,\"native_time\":%.17g,\"native_step_begin\":%.17g,\"source_fingerprint\":\"%s\",\"comoving\":%d,\"unit_mass_g\":%.17g,\"unit_velocity_cm_s\":%.17g,\"unit_time_s\":%.17g,\"hubble_param\":%.17g,\"mpi_ranks\":%d,\"parts\":[",(unsigned long long)seq,(unsigned long long)All.BHEventCount,(unsigned long long)(All.BHEventCount+total),(unsigned long long)total,All.Time,All.TimeOld,research_restart_fingerprint,All.ComovingIntegrationOn,All.UnitMass_in_g,All.UnitVelocity_in_cm_per_s,All.UnitTime_in_s,All.HubbleParam,ranks);
    for(int r=0;r<ranks;r++){snprintf(name,sizeof(name),"black_holes_%020llu.rank_%06d.of_%06d.jsonl",(unsigned long long)seq,r,ranks);fprintf(f,"%s{\"rank\":%d,\"rows\":%llu,\"bytes\":%llu,\"crc32\":%llu,\"filename\":",r ? "," : "",r,(unsigned long long)all[3*r],(unsigned long long)all[3*r+1],(unsigned long long)all[3*r+2]);if(all[3*r])fprintf(f,"\"%s\"}",name);else fprintf(f,"null}");}fprintf(f,"]}\n");return !ferror(f);},bytes,crc);}
  MPI_Bcast(&good,1,MPI_INT,0,comm);if(!good)Terminate("BH event commit publication failed");All.BHEventCommitCount=seq;All.BHEventCount+=total;
}
}
#endif
