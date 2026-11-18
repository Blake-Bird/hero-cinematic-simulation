"""Compile and exercise the actual retained GADGET cooling methods in isolation.

Only MPI/particle I/O and global unit plumbing are replaced. Rate tables,
abundance iteration, temperature closure and implicit energy integration are
extracted verbatim from the shipped source, not reimplemented in Python.
"""
from pathlib import Path
import argparse
import hashlib
import json
import re
import shutil
import subprocess

ROOT = Path(__file__).resolve().parent / 'software/gadget4/src'
METHODS = ('DoCooling', 'convert_u_to_temp', 'find_abundances_and_rates',
           'CoolingRateFromU', 'CoolingRate', 'MakeRateTable')


def method_source(source, name):
    match = re.search(r'(?:double|void) coolsfr::' + name + r'\(', source)
    if match is None:
        raise ValueError('Native cooling method missing: ' + name)
    begin = source.index('{', match.start())
    depth = 1
    end = begin + 1
    while depth and end < len(source):
        depth += (source[end] == '{') - (source[end] == '}')
        end += 1
    if depth:
        raise ValueError('Unbalanced native method: ' + name)
    return source[match.start():end]


def control_source(source_path=None):
    source = Path(source_path or ROOT/'cooling_sfr/cooling.cc').read_text()
    header = (ROOT/'cooling_sfr/cooling.h').read_text()
    methods = [method_source(source, name) for name in METHODS]
    structs = []
    for name in ('gas_state', 'rate_table', 'photo_current', 'do_cool_data'):
        found = re.search(r'struct ' + name + r'\s*\{.*?\};', header, re.S)
        if found is None:
            raise ValueError('Native structure missing: ' + name)
        structs.append(found.group())
    declarations = [body[:body.index('{')].replace('coolsfr::', '')+';' for body in methods]
    return r'''
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdarg>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>
#include <limits>
#include "constants.h"
#define NCOOLTAB 2000
#define gsl_finite std::isfinite
[[noreturn]] void control_terminate(const char *format, ...) {
  char buffer[2048]; va_list args; va_start(args,format);
  vsnprintf(buffer,sizeof(buffer),format,args);va_end(args);
  throw std::runtime_error(buffer);
}
#define Terminate(...) control_terminate(__VA_ARGS__)
struct {double UnitDensity_in_cgs=1.,UnitPressure_in_cgs=1.,HubbleParam=1.,
  UnitTime_in_s=1.,MinEgySpec=0.,Time=1.;int ComovingIntegrationOn=0;} All;
class coolsfr { public:
''' + '\n'.join(structs + declarations) + r'''
gas_state GasState{};
do_cool_data DoCoolData{};
photo_current pc{};
rate_table rates[NCOOLTAB+1]{};
rate_table *RateT=rates;
double Tmin=1.,Tmax=9.,deltaT=0.;
coolsfr(){GasState.XH=HYDROGEN_MASSFRAC;MakeRateTable();}
};
''' + '\n\n'.join(methods) + r'''
int main(int argc,char **argv) {
  try {
    coolsfr c;
    const double y=c.GasState.yhelium;
    const double neutral_mu=(1+4*y)/(1+y);
    const double to_temperature=GAMMA_MINUS1*PROTONMASS/BOLTZMANN;
    const bool reproduce=argc>1 && std::string(argv[1])=="reproduce";
    int cases=0,implicit_cases=0;double max_energy_error=0.,max_charge_error=0.,max_implicit_error=0.;
    auto check=[&](double neutral_temp,double nH,double gamma,double guess) {
      c.pc.J_UV=gamma>0;c.pc.gJH0=gamma;c.pc.gJHe0=.7*gamma;c.pc.gJHep=.1*gamma;
      c.pc.epsH0=gamma*3.e-12;c.pc.epsHe0=.7*gamma*5.e-12;c.pc.epsHep=.1*gamma*1.e-11;
      auto gs=c.GasState;auto data=c.DoCoolData;
      const double rho=nH*PROTONMASS/HYDROGEN_MASSFRAC;
      const double energy=neutral_temp/(to_temperature*neutral_mu);
      double ne=guess;
      data.u_old_input=energy;data.rho_input=rho;data.ne_guess_input=guess;
      const double temp=c.convert_u_to_temp(energy,rho,&ne,&gs,&data);
      const double recovered=temp*(1+y+ne)/(to_temperature*(1+4*y));
      double error=std::abs(recovered/energy-1);
      const double charge=gs.nHp+gs.nHep+2*gs.nHepp;
      max_energy_error=std::max(max_energy_error,error);
      max_charge_error=std::max(max_charge_error,std::abs(charge-ne));
      if(!std::isfinite(temp) || temp<=0 || !std::isfinite(ne) || ne<0 || ne>1+2*y || error>2.e-8)
        control_terminate("Thermodynamic closure failed: Tneutral=%g nH=%g gamma=%g guess=%g T=%g ne=%g relative_energy_error=%g",neutral_temp,nH,gamma,guess,temp,ne,error);
      if(std::abs(gs.nH0+gs.nHp-1)>1.e-10 || std::abs(gs.nHe0+gs.nHep+gs.nHepp-y)>1.e-10)
        control_terminate("Species conservation failed");
      ++cases;
    };
    if(reproduce) {
      for(double temp=10.01;temp<15.;temp+=.1)check(temp,.01,1.e-14,0.);
    } else {
      for(double nH: {1.e-10,1.e-7,1.e-4,.01,1.,1.e3,1.e6})
        for(double gamma: {0.,1.e-18,1.e-14,1.e-10})
          for(double guess: {0.,.3,1.+2*y})
            for(double temp: {.01,1.,9.9,10.,10.01,10.2,11.,15.,100.,1.e3,1.e4,1.e5,1.e6,1.e8,1.e9,1.e10})
              check(temp,nH,gamma,guess);
    }
    if(!reproduce) {
      // Independent analytic H-only photoionization balance at the low-table
      // endpoint: alpha*nH*x^2 = Gamma*(1-x). Helium stays neutral.
      {
        const double nH=.01,gamma=1.e-14;
        c.pc={};c.pc.J_UV=1;c.pc.gJH0=gamma;
        const double alpha=c.RateT[0].AlphaHp;
        const double expected_ne=2*gamma/(gamma+std::sqrt(gamma*gamma+4*alpha*nH*gamma));
        const double energy=1./(to_temperature*neutral_mu);
        auto gs=c.GasState;auto data=c.DoCoolData;double ne=0;
        const double temp=c.convert_u_to_temp(energy,nH*PROTONMASS/HYDROGEN_MASSFRAC,&ne,&gs,&data);
        const double expected_temp=(1+y)/(1+y+expected_ne);
        if(std::abs(ne-expected_ne)>1.e-10 || std::abs(temp/expected_temp-1)>1.e-8 || temp>=10)
          control_terminate("Analytic cold photoionization control failed");
      }
      // Unit changes must commute with the physical energy solve.
      {
        c.pc={};c.pc.J_UV=1;c.pc.gJH0=1.e-14;c.pc.epsH0=3.e-26;
        const double rho=.01*PROTONMASS/HYDROGEN_MASSFRAC;
        const double u=1.e4/(to_temperature*neutral_mu),dt=1.e13;
        auto first=c.GasState;auto second=c.GasState;auto data=c.DoCoolData;
        double ne1=.3,ne2=.3;
        const double cgs=c.DoCooling(u,rho,dt,&ne1,&first,&data);
        All.UnitDensity_in_cgs=1.e-22;All.UnitPressure_in_cgs=1.e-12;
        All.UnitTime_in_s=3.e13;All.HubbleParam=.7;
        const double energy_unit=All.UnitPressure_in_cgs/All.UnitDensity_in_cgs;
        const double native=c.DoCooling(u/energy_unit,rho/(All.UnitDensity_in_cgs*.7*.7),dt*.7/All.UnitTime_in_s,&ne2,&second,&data);
        if(std::abs(native*energy_unit/cgs-1)>2.e-8 || std::abs(ne1-ne2)>1.e-9)
          control_terminate("Cooling physical-unit invariance failed");
        All.UnitDensity_in_cgs=1.;All.UnitPressure_in_cgs=1.;All.UnitTime_in_s=1.;All.HubbleParam=1.;
      }
      for(double scale_factor: {0.,.01,.1,.5,1.}) {
        All.ComovingIntegrationOn=scale_factor>0;
        All.Time=scale_factor>0 ? scale_factor : 1.;
      for(double nH: {1.e-7,1.e-3,1.,1.e3})
        for(double gamma: {0.,1.e-14,1.e-10})
          for(double neutral_temp: {0.,.01,10.,11.,100.,1.e4,1.e6,1.e8})
            for(double dt: {0.,1.e3,1.e9,1.e13,1.e16}) {
              c.pc.J_UV=gamma>0;c.pc.gJH0=gamma;c.pc.gJHe0=.7*gamma;c.pc.gJHep=.1*gamma;
              c.pc.epsH0=gamma*3.e-12;c.pc.epsHe0=.7*gamma*5.e-12;c.pc.epsHep=.1*gamma*1.e-11;
              auto gs=c.GasState;auto data=c.DoCoolData;double ne=.3;
              const double rho=nH*PROTONMASS/HYDROGEN_MASSFRAC;
              const double before=neutral_temp/(to_temperature*neutral_mu);
              const double after=c.DoCooling(before,rho,dt,&ne,&gs,&data);
              const double end_ne=ne;
              const double rate=gs.nHcgs*gs.nHcgs/rho*c.CoolingRateFromU(after,rho,&ne,&gs,&data);
              const double scale=std::max(before,after);
              const double error=scale>0 ? std::abs((after-before)-dt*rate)/scale : std::abs(dt*rate);
              max_implicit_error=std::max(max_implicit_error,error);
              if(!std::isfinite(after) || after<0 || error>2.e-8 || std::abs(end_ne-ne)>1.e-9 || (dt==0 && after!=before))
                control_terminate("Implicit energy control failed: nH=%g UV=%g Tneutral=%g dt=%g error=%g ne_change=%g",nH,gamma,neutral_temp,dt,error,end_ne-ne);
              ++implicit_cases;
            }
      }
      for(double dt: {-1.,std::numeric_limits<double>::quiet_NaN()}) {
        auto gs=c.GasState;auto data=c.DoCoolData;double ne=0.;bool rejected=false;
        try{c.DoCooling(1.,1.e-25,dt,&ne,&gs,&data);}catch(const std::exception&){rejected=true;}
        if(!rejected)control_terminate("Invalid cooling timestep accepted");
      }
      for(int kind=0;kind<3;kind++) {
        auto gs=c.GasState;auto data=c.DoCoolData;double ne=kind==2 ? -1. : 0.;bool rejected=false;
        try{c.convert_u_to_temp(kind==0 ? -1.:1.,kind==1 ? 0.:1.e-25,&ne,&gs,&data);}
        catch(const std::exception&){rejected=true;}
        if(!rejected)control_terminate("Invalid thermodynamic state accepted");
      }
    }
    printf("{\"cases\":%d,\"implicit_cases\":%d,\"max_energy_relative_error\":%.17g,\"max_charge_absolute_error\":%.17g,\"max_implicit_residual_relative_error\":%.17g}\n",cases,implicit_cases,max_energy_error,max_charge_error,max_implicit_error);
    return 0;
  } catch(const std::exception &e) { fprintf(stderr,"%s\n",e.what());return 1; }
}
'''


def run_control(output, mode='sweep', source_path=None):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    compiler = shutil.which('c++')
    if not compiler:
        raise RuntimeError('A C++ compiler is required for native cooling controls')
    source = output/'cooling_control.cc'
    native_source = Path(source_path or ROOT/'cooling_sfr/cooling.cc')
    source.write_text(control_source(native_source))
    binary = output/'cooling_control'
    command = [compiler, '-std=c++11', '-O2', '-I', str(ROOT/'data'),
               '-I', str(ROOT/'cooling_sfr'), str(source), '-o', str(binary)]
    build = subprocess.run(command, capture_output=True, text=True, timeout=60)
    (output/'build.log').write_text(build.stdout+build.stderr)
    if build.returncode:
        raise RuntimeError('Cooling control compilation failed: ' + build.stderr[-4000:])
    result = subprocess.run([str(binary), mode], capture_output=True, text=True, timeout=60)
    (output/'stdout.log').write_text(result.stdout)
    (output/'stderr.log').write_text(result.stderr)
    inputs = [native_source,ROOT/'cooling_sfr/cooling.h',ROOT/'data/constants.h',Path(__file__)]
    record = {'schema':'native-cooling-control-v1','mode':mode,
              'status':'PASS' if result.returncode==0 else 'FAIL',
              'returncode':result.returncode,'compile_command':command,
              'source_sha256':{str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs},
              'scope':'isolated shipped native rate/thermodynamic methods; not a cosmological run or restart compatibility proof'}
    if result.returncode==0:
        record['measurements']=json.loads(result.stdout.splitlines()[-1])
    else:
        record['failure']=result.stderr[-4000:]
    (output/'report.json').write_text(json.dumps(record,indent=2)+'\n')
    return record


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    parser.add_argument('--mode',choices=['sweep','reproduce'],default='sweep')
    args=parser.parse_args()
    report=run_control(args.output,args.mode)
    print(json.dumps(report,indent=2))
    raise SystemExit(0 if report['status']=='PASS' else 1)
