/*******************************************************************************
 * \copyright   This file is part of the GADGET4 N-body/SPH code developed
 * \copyright   by Volker Springel. Copyright (C) 2014-2020 by Volker Springel
 * \copyright   (vspringel@mpa-garching.mpg.de) and all contributing authors.
 *******************************************************************************/

/*! \file cooling.cc
 *
 *  \brief Module for gas radiative cooling
 */

#include "gadgetconfig.h"

#ifdef COOLING

#include <gsl/gsl_math.h>
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <cmath>

#include "../cooling_sfr/cooling.h"
#include "research_stellar_returns.h"
#include "../data/allvars.h"
#include "../data/dtypes.h"
#include "../data/mymalloc.h"
#include "../logs/logs.h"
#include "../logs/timer.h"
#include "../system/system.h"
#include "../time_integration/timestep.h"

/** \brief Compute the new internal energy per unit mass.
 *
 *   The function solves for the new internal energy per unit mass of the gas by integrating the equation
 *   for the internal energy with an implicit Euler scheme. The root of resulting non linear equation,
 *   which gives tnew internal energy, is found with the bisection method.
 *   Arguments are passed in code units.
 *
 *   \param u_old the initial (before cooling is applied) internal energy per unit mass of the gas particle
 *   \param rho   the proper density of the gas particle
 *   \param dt    the duration of the time step
 *   \param ne_guess electron number density relative to hydrogen number density (for molecular weight computation)
 *   \return the new internal energy per unit mass of the gas particle
 */
double coolsfr::DoCooling(double u_old, double rho, double dt, double *ne_guess, gas_state *gs, do_cool_data *DoCool)
{
  DoCool->u_old_input    = u_old;
  DoCool->rho_input      = rho;
  DoCool->dt_input       = dt;
  DoCool->ne_guess_input = *ne_guess;

  if(!gsl_finite(u_old) || u_old < 0 || !gsl_finite(rho) || rho <= 0 ||
     !gsl_finite(dt) || dt < 0)
    Terminate("Invalid cooling inputs: u_old=%g rho=%g dt=%g\n",u_old,rho,dt);

  const double energy_unit = All.UnitPressure_in_cgs / All.UnitDensity_in_cgs;
  rho *= All.UnitDensity_in_cgs * All.HubbleParam * All.HubbleParam;
  u_old *= energy_unit;
  dt *= All.UnitTime_in_s / All.HubbleParam;
  if(!gsl_finite(u_old) || !gsl_finite(rho) || rho <= 0 || !gsl_finite(dt) || dt < 0 ||
     !gsl_finite(energy_unit) || energy_unit <= 0)
    Terminate("Invalid physical cooling units or transformed state\n");
  gs->nHcgs = gs->XH * rho / PROTONMASS;
  const double ratefact = gs->nHcgs * gs->nHcgs / rho;
  if(!gsl_finite(ratefact) || ratefact <= 0)
    Terminate("Invalid physical hydrogen density in cooling solve\n");

  // A zero step still refreshes the ionization state at the unchanged energy.
  if(dt == 0)
    {
      CoolingRateFromU(u_old, rho, ne_guess, gs, DoCool);
      return DoCool->u_old_input;
    }
  auto residual = [&](double energy) {
    const double value = energy - u_old - ratefact * CoolingRateFromU(energy, rho, ne_guess, gs, DoCool) * dt;
    if(!gsl_finite(value))
      Terminate("Nonfinite implicit cooling residual: u=%g rho=%g dt=%g\n",energy,rho,dt);
    return value;
  };
  // Retain the original local bracketing policy: search outward from u_old
  // in the direction of its net source. A global [0,large] bracket can select
  // a different thermal branch when the cooling curve has multiple roots.
  const double initial = residual(u_old);
  if(initial == 0) return DoCool->u_old_input;
  double lower = u_old, upper = u_old;
  bool bracketed = false;
  for(int expansion = 0; expansion < 1024; expansion++)
    {
      if(initial > 0) // cooling
        {
          lower = upper / 1.1;
          if(lower == upper) lower = 0;
          if(residual(lower) <= 0) { bracketed = true; break; }
          upper = lower;
        }
      else // heating; the table's neutral energy is a trial scale, not a floor
        {
          upper = lower > 0 ? lower * 1.1 : gs->ethmin;
          if(!gsl_finite(upper) || upper <= lower) break;
          if(residual(upper) >= 0) { bracketed = true; break; }
          lower = upper;
        }
    }
  if(!bracketed)
    Terminate("Could not bracket implicit cooling within 1024 local expansions: u_old=%g rho=%g dt=%g\n",u_old,rho,dt);
  for(int iter = 0; iter < 256; iter++)
    {
      const double energy = lower + 0.5 * (upper - lower);
      const double value = residual(energy);
      const double scale = std::max(u_old, energy);
      if(value == 0 || (scale > 0 && fabs(value) / scale <= 1.e-8))
        return energy / energy_unit;
      if(value > 0) upper = energy;
      else lower = energy;
      if(upper == lower || std::nextafter(lower, upper) == upper) break;
    }
  Terminate("Implicit cooling residual did not converge: u_old=%g rho=%g dt=%g bracket=[%g,%g]\n",u_old,rho,dt,lower,upper);
}

/** \brief Return the cooling time.
 *
 *  If we actually have heating, a cooling time of 0 is returned.
 *
 *  \param u_old the initial (before cooling is applied) internal energy per unit mass of the gas particle
 *  \param rho   the proper density of the gas particle
 *  \param ne_guess electron number density relative to hydrogen number density (for molecular weight computation)
 */
double coolsfr::GetCoolingTime(double u_old, double rho, double *ne_guess, gas_state *gs, do_cool_data *DoCool)
{
  DoCool->u_old_input    = u_old;
  DoCool->rho_input      = rho;
  DoCool->ne_guess_input = *ne_guess;

  rho *= All.UnitDensity_in_cgs * All.HubbleParam * All.HubbleParam; /* convert to physical cgs units */
  u_old *= All.UnitPressure_in_cgs / All.UnitDensity_in_cgs;

  gs->nHcgs       = gs->XH * rho / PROTONMASS; /* hydrogen number dens in cgs units */
  double ratefact = gs->nHcgs * gs->nHcgs / rho;

  double u = u_old;

  double LambdaNet = CoolingRateFromU(u, rho, ne_guess, gs, DoCool);

  /* bracketing */

  if(LambdaNet >= 0) /* ups, we have actually heating due to UV background */
    return 0;

  double coolingtime = u_old / (-ratefact * LambdaNet);

  coolingtime *= All.HubbleParam / All.UnitTime_in_s;

  return coolingtime;
}

/** \brief Compute gas temperature from internal energy per unit mass.
 *
 *   This function determines the electron fraction, and hence the mean
 *   molecular weight. With it arrives at a self-consistent temperature.
 *   Element abundances and the rates for the emission are also computed
 *
 *  \param u   internal energy per unit mass
 *  \param rho gas density
 *  \param ne_guess electron number density relative to hydrogen number density
 *  \return the gas temperature
 */
double coolsfr::convert_u_to_temp(double u, double rho, double *ne_guess, gas_state *gs, const do_cool_data *DoCool)
{
  const double ne_input = *ne_guess;
  const double max_ne = 1.0 + 2.0 * gs->yhelium;
  if(!gsl_finite(u) || u < 0 || !gsl_finite(rho) || rho <= 0 ||
     !gsl_finite(ne_input) || ne_input < 0 || ne_input > max_ne ||
     !gsl_finite(gs->yhelium) || gs->yhelium < 0)
    Terminate("Invalid temperature-closure inputs: u=%g rho=%g ne=%g\n", u, rho, ne_input);

  // Ideal-gas closure bounds: fully ionized and neutral primordial gas.
  // Solve T*(1+y+ne(T)) = (gamma-1)*u*mp/kB*(1+4*y), not a
  // damped fixed point whose small step can falsely look converged.
  const double neutral = (GAMMA_MINUS1 * PROTONMASS / BOLTZMANN) * u *
                         (1.0 + 4.0 * gs->yhelium) / (1.0 + gs->yhelium);
  if(!gsl_finite(neutral))
    Terminate("Nonfinite neutral temperature for u=%g rho=%g\n", u, rho);
  if(neutral == 0)
    {
      find_abundances_and_rates(Tmin, rho, ne_guess, gs, DoCool);
      return 0;
    }
  double lower = neutral * (1.0 + gs->yhelium) / (1.0 + gs->yhelium + max_ne);
  double upper = neutral;
  auto evaluate = [&](double temp) {
    *ne_guess = 0.5 * max_ne;
    find_abundances_and_rates(log10(temp), rho, ne_guess, gs, DoCool);
    const double value = (temp / neutral) *
                         (1.0 + gs->yhelium + *ne_guess) / (1.0 + gs->yhelium) - 1.0;
    if(!gsl_finite(value))
      Terminate("Nonfinite temperature residual: u=%g rho=%g T=%g ne=%g\n",u,rho,temp,*ne_guess);
    return value;
  };
  double f_upper = evaluate(upper);
  if(fabs(f_upper) <= 1.e-8) return upper;
  double f_lower = evaluate(lower);
  if(fabs(f_lower) <= 1.e-8) return lower;
  if(f_lower > 0 || f_upper < 0)
    Terminate("Ionization state lies outside the physical temperature bracket\n");
  for(int iter = 0; iter < 256; iter++)
    {
      double fraction = -f_lower / (f_upper - f_lower);
      // Safeguarded interpolation is fast for nearly constant molecular
      // weight; regular bisection keeps difficult transitions bounded.
      if(iter % 4 == 3 || !(fraction > 0.01 && fraction < 0.99)) fraction = 0.5;
      const double temp = lower + fraction * (upper - lower);
      // Every residual evaluates the same deterministic ionization closure.
      // A stale caller guess must not select a different thermodynamic branch.
      const double residual = evaluate(temp);
      if(fabs(residual) <= 1.e-8)
        return temp;
      if(residual > 0)
        { upper = temp; f_upper = residual; }
      else
        { lower = temp; f_lower = residual; }
      if(upper == lower || std::nextafter(lower, upper) == upper)
        break;
    }
  Terminate("Temperature closure is unresolved: u=%g rho=%g initial_ne=%g bracket=[%g,%g]\n",
            u, rho, ne_input, lower, upper);
}

/** \brief Computes the actual abundance ratios.
 *
 *  The chemical composition of the gas is primordial (no metals are present)
 *
 *  \param logT     log10 of gas temperature
 *  \param rho      gas density
 *  \param ne_guess electron number density relative to hydrogen number density
 */
void coolsfr::find_abundances_and_rates(double logT, double rho, double *ne_guess, gas_state *gs, const do_cool_data *DoCool)
{
  double logT_input = logT;
  double rho_input  = rho;
  double ne_input   = *ne_guess;

  if(!gsl_finite(logT))
    Terminate("logT=%g\n", logT);

  // Below the rate table, use its endpoint coefficients, just as the
  // cooling-rate evaluator does. T is NOT floored and u is NOT modified.
  // Forcing neutrality here while CoolingRate keeps UV photoionization
  // creates an artificial jump in molecular weight at 10 K.
  if(logT < Tmin)
    logT = Tmin;

  if(logT >= Tmax) /* everything is ionized */
    {
      gs->nH0   = 0;
      gs->nHe0  = 0;
      gs->nHp   = 1.0;
      gs->nHep  = 0;
      gs->nHepp = gs->yhelium;
      gs->ne    = gs->nHp + 2.0 * gs->nHepp;
      *ne_guess = gs->ne; /* note: in units of the hydrogen number density */
      return;
    }

  double t    = (logT - Tmin) / deltaT;
  int j       = (int)t;
  double fhi  = t - j;
  double flow = 1 - fhi;

  if(*ne_guess == 0)
    *ne_guess = 1.0;

  gs->nHcgs = gs->XH * rho / PROTONMASS; /* hydrogen number dens in cgs units */

  gs->ne       = *ne_guess;
  double neold = gs->ne;
  int niter    = 0;
  gs->necgs    = gs->ne * gs->nHcgs;

  /* evaluate number densities iteratively (cf KWH eqns 33-38) in units of nH */
  do
    {
      niter++;

      gs->aHp   = flow * RateT[j].AlphaHp + fhi * RateT[j + 1].AlphaHp;
      gs->aHep  = flow * RateT[j].AlphaHep + fhi * RateT[j + 1].AlphaHep;
      gs->aHepp = flow * RateT[j].AlphaHepp + fhi * RateT[j + 1].AlphaHepp;
      gs->ad    = flow * RateT[j].Alphad + fhi * RateT[j + 1].Alphad;
      gs->geH0  = flow * RateT[j].GammaeH0 + fhi * RateT[j + 1].GammaeH0;
      gs->geHe0 = flow * RateT[j].GammaeHe0 + fhi * RateT[j + 1].GammaeHe0;
      gs->geHep = flow * RateT[j].GammaeHep + fhi * RateT[j + 1].GammaeHep;

      if(gs->necgs <= 1.e-25 || pc.J_UV == 0)
        {
          gs->gJH0ne = gs->gJHe0ne = gs->gJHepne = 0;
        }
      else
        {
          gs->gJH0ne  = pc.gJH0 / gs->necgs;
          gs->gJHe0ne = pc.gJHe0 / gs->necgs;
          gs->gJHepne = pc.gJHep / gs->necgs;
        }

      gs->nH0 = gs->aHp / (gs->aHp + gs->geH0 + gs->gJH0ne); /* eqn (33) */
      gs->nHp = 1.0 - gs->nH0;                               /* eqn (34) */

      if((gs->gJHe0ne + gs->geHe0) <= SMALLNUM) /* no ionization at all */
        {
          gs->nHep  = 0.0;
          gs->nHepp = 0.0;
          gs->nHe0  = gs->yhelium;
        }
      else
        {
          gs->nHep = gs->yhelium /
                     (1.0 + (gs->aHep + gs->ad) / (gs->geHe0 + gs->gJHe0ne) + (gs->geHep + gs->gJHepne) / gs->aHepp); /* eqn (35) */
          gs->nHe0  = gs->nHep * (gs->aHep + gs->ad) / (gs->geHe0 + gs->gJHe0ne);                                     /* eqn (36) */
          gs->nHepp = gs->nHep * (gs->geHep + gs->gJHepne) / gs->aHepp;                                               /* eqn (37) */
        }

      neold = gs->ne;

      gs->ne    = gs->nHp + gs->nHep + 2 * gs->nHepp; /* eqn (38) */
      gs->necgs = gs->ne * gs->nHcgs;

      if(pc.J_UV == 0)
        break;

      double nenew = 0.5 * (gs->ne + neold);
      gs->ne       = nenew;
      gs->necgs    = gs->ne * gs->nHcgs;

      if(fabs(gs->ne - neold) < 1.0e-12)
        break;

      if(niter > (MAXITER - 10))
        printf("ne= %g  niter=%d\n", gs->ne, niter);
    }
  while(niter < MAXITER);

  if(niter >= MAXITER)
    Terminate(
        "no convergence reached in find_abundances_and_rates(): logT_input= %g  rho_input= %g  ne_input= %g "
        "DoCool->u_old_input=%g\nDoCool->rho_input= %g\nDoCool->dt_input= %g\nDoCool->ne_guess_input= %g\n",
        logT_input, rho_input, ne_input, DoCool->u_old_input, DoCool->rho_input, DoCool->dt_input, DoCool->ne_guess_input);

  gs->bH0  = flow * RateT[j].BetaH0 + fhi * RateT[j + 1].BetaH0;
  gs->bHep = flow * RateT[j].BetaHep + fhi * RateT[j + 1].BetaHep;
  gs->bff  = flow * RateT[j].Betaff + fhi * RateT[j + 1].Betaff;

  *ne_guess = gs->ne;
}

/** \brief Get cooling rate from gas internal energy.
 *
 *  This function first computes the self-consistent temperature
 *  and abundance ratios, and then it calculates
 *  (heating rate-cooling rate)/n_h^2 in cgs units
 *
 *  \param u   gas internal energy per unit mass
 *  \param rho gas density
 *  \param ne_guess electron number density relative to hydrogen number density
 */
double coolsfr::CoolingRateFromU(double u, double rho, double *ne_guess, gas_state *gs, const do_cool_data *DoCool)
{
  double temp = convert_u_to_temp(u, rho, ne_guess, gs, DoCool);

  return CoolingRate(temp > 0 ? log10(temp) : Tmin, rho, ne_guess, gs, DoCool);
}

/** \brief  This function computes the self-consistent temperature and abundance ratios.
 *
 *  Used only in the file io.c (maybe it is not necessary)
 *
 *  \param u   internal energy per unit mass
 *  \param rho gas density
 *  \param ne_guess electron number density relative to hydrogen number density
 *  \param nH0_pointer pointer to the neutral hydrogen fraction (set to current value in the GasState struct)
 *  \param nHeII_pointer pointer to the ionised helium fraction (set to current value in the GasState struct)
 */
double coolsfr::AbundanceRatios(double u, double rho, double *ne_guess, double *nH0_pointer, double *nHeII_pointer)
{
  gas_state gs          = GasState;
  do_cool_data DoCool   = DoCoolData;
  DoCool.u_old_input    = u;
  DoCool.rho_input      = rho;
  DoCool.ne_guess_input = *ne_guess;

  rho *= All.UnitDensity_in_cgs * All.HubbleParam * All.HubbleParam; /* convert to physical cgs units */
  u *= All.UnitPressure_in_cgs / All.UnitDensity_in_cgs;

  double temp = convert_u_to_temp(u, rho, ne_guess, &gs, &DoCool);

  *nH0_pointer   = gs.nH0;
  *nHeII_pointer = gs.nHep;

  return temp;
}

/** \brief  Calculate (heating rate-cooling rate)/n_h^2 in cgs units.
 *
 *  \param logT     log10 of gas temperature
 *  \param rho      gas density
 *  \param nelec    electron number density relative to hydrogen number density
 *  \return         (heating rate-cooling rate)/n_h^2
 */
double coolsfr::CoolingRate(double logT, double rho, double *nelec, gas_state *gs, const do_cool_data *DoCool)
{
  double Lambda, Heat;

  if(logT <= Tmin)
    logT = Tmin; /* endpoint rate coefficients; no thermal-energy floor */

  gs->nHcgs = gs->XH * rho / PROTONMASS; /* hydrogen number dens in cgs units */

  if(logT < Tmax)
    {
      find_abundances_and_rates(logT, rho, nelec, gs, DoCool);

      /* Compute cooling and heating rate (cf KWH Table 1) in units of nH**2 */
      double T = pow(10.0, logT);

      double LambdaExcH0   = gs->bH0 * gs->ne * gs->nH0;
      double LambdaExcHep  = gs->bHep * gs->ne * gs->nHep;
      double LambdaExc     = LambdaExcH0 + LambdaExcHep; /* excitation */
      double LambdaIonH0   = 2.18e-11 * gs->geH0 * gs->ne * gs->nH0;
      double LambdaIonHe0  = 3.94e-11 * gs->geHe0 * gs->ne * gs->nHe0;
      double LambdaIonHep  = 8.72e-11 * gs->geHep * gs->ne * gs->nHep;
      double LambdaIon     = LambdaIonH0 + LambdaIonHe0 + LambdaIonHep; /* ionization */
      double LambdaRecHp   = 1.036e-16 * T * gs->ne * (gs->aHp * gs->nHp);
      double LambdaRecHep  = 1.036e-16 * T * gs->ne * (gs->aHep * gs->nHep);
      double LambdaRecHepp = 1.036e-16 * T * gs->ne * (gs->aHepp * gs->nHepp);
      double LambdaRecHepd = 6.526e-11 * gs->ad * gs->ne * gs->nHep;
      double LambdaRec     = LambdaRecHp + LambdaRecHep + LambdaRecHepp + LambdaRecHepd;
      double LambdaFF      = gs->bff * (gs->nHp + gs->nHep + 4 * gs->nHepp) * gs->ne;
      Lambda               = LambdaExc + LambdaIon + LambdaRec + LambdaFF;

      if(All.ComovingIntegrationOn)
        {
          double redshift    = 1 / All.Time - 1;
          double LambdaCmptn = 5.65e-36 * gs->ne * (T - 2.73 * (1. + redshift)) * pow(1. + redshift, 4.) / gs->nHcgs;

          Lambda += LambdaCmptn;
        }

      Heat = 0;
      if(pc.J_UV != 0)
        Heat += (gs->nH0 * pc.epsH0 + gs->nHe0 * pc.epsHe0 + gs->nHep * pc.epsHep) / gs->nHcgs;
    }
  else /* here we're outside of tabulated rates, T>Tmax K */
    {
      /* at high T (fully ionized); only free-free and Compton cooling are present. Assumes no heating. */

      Heat = 0;

      /* very hot: H and He both fully ionized */
      gs->nHp   = 1.0;
      gs->nHep  = 0;
      gs->nHepp = gs->yhelium;
      gs->ne    = gs->nHp + 2.0 * gs->nHepp;
      *nelec    = gs->ne; /* note: in units of the hydrogen number density */

      double T        = pow(10.0, logT);
      double LambdaFF = 1.42e-27 * sqrt(T) * (1.1 + 0.34 * exp(-(5.5 - logT) * (5.5 - logT) / 3)) * (gs->nHp + 4 * gs->nHepp) * gs->ne;
      double LambdaCmptn;
      if(All.ComovingIntegrationOn)
        {
          double redshift = 1 / All.Time - 1;
          /* add inverse Compton cooling off the microwave background */
          LambdaCmptn = 5.65e-36 * gs->ne * (T - 2.73 * (1. + redshift)) * pow(1. + redshift, 4.) / gs->nHcgs;
        }
      else
        LambdaCmptn = 0;

      Lambda = LambdaFF + LambdaCmptn;
    }

  return (Heat - Lambda);
}

/** \brief Make cooling rates interpolation table.
 *
 *  Set up interpolation tables in T for cooling rates given in KWH, ApJS, 105, 19
 */
void coolsfr::MakeRateTable(void)
{
  GasState.yhelium = (1 - GasState.XH) / (4 * GasState.XH);
  GasState.mhboltz = PROTONMASS / BOLTZMANN;

  deltaT          = (Tmax - Tmin) / NCOOLTAB;
  GasState.ethmin = pow(10.0, Tmin) * (1. + GasState.yhelium) / ((1. + 4. * GasState.yhelium) * GasState.mhboltz * GAMMA_MINUS1);
  /* minimum internal energy for neutral gas */

  for(int i = 0; i <= NCOOLTAB; i++)
    {
      RateT[i].BetaH0 = RateT[i].BetaHep = RateT[i].Betaff = RateT[i].AlphaHp = RateT[i].AlphaHep = RateT[i].AlphaHepp =
          RateT[i].Alphad = RateT[i].GammaeH0 = RateT[i].GammaeHe0 = RateT[i].GammaeHep = 0;

      double T     = pow(10.0, Tmin + deltaT * i);
      double Tfact = 1.0 / (1 + sqrt(T / 1.0e5));

      /* collisional excitation */
      /* Cen 1992 */
      if(118348 / T < 70)
        RateT[i].BetaH0 = 7.5e-19 * exp(-118348 / T) * Tfact;
      if(473638 / T < 70)
        RateT[i].BetaHep = 5.54e-17 * pow(T, -0.397) * exp(-473638 / T) * Tfact;

      /* free-free */
      RateT[i].Betaff = 1.43e-27 * sqrt(T) * (1.1 + 0.34 * exp(-(5.5 - log10(T)) * (5.5 - log10(T)) / 3));

      /* recombination */

      /* Cen 1992 */
      /* Hydrogen II */
      RateT[i].AlphaHp = 8.4e-11 * pow(T / 1000, -0.2) / (1. + pow(T / 1.0e6, 0.7)) / sqrt(T);
      /* Helium II */
      RateT[i].AlphaHep = 1.5e-10 * pow(T, -0.6353);
      /* Helium III */
      RateT[i].AlphaHepp = 4. * RateT[i].AlphaHp;
      /* Cen 1992 */
      /* dielectric recombination */
      if(470000 / T < 70)
        RateT[i].Alphad = 1.9e-3 * pow(T, -1.5) * exp(-470000 / T) * (1. + 0.3 * exp(-94000 / T));

      /* collisional ionization */
      /* Cen 1992 */
      /* Hydrogen */
      if(157809.1 / T < 70)
        RateT[i].GammaeH0 = 5.85e-11 * sqrt(T) * exp(-157809.1 / T) * Tfact;
      /* Helium */
      if(285335.4 / T < 70)
        RateT[i].GammaeHe0 = 2.38e-11 * sqrt(T) * exp(-285335.4 / T) * Tfact;
      /* Hellium II */
      if(631515.0 / T < 70)
        RateT[i].GammaeHep = 5.68e-12 * sqrt(T) * exp(-631515.0 / T) * Tfact;
    }
}

/** \brief Read table input for ionizing parameters.
 *
 *  \param file that contains the tabulated parameters
 */
void coolsfr::ReadIonizeParams(char *fname)
{
  NheattabUVB = 0;
  int i, iter;
  for(iter = 0, i = 0; iter < 2; iter++)
    {
      FILE *fdcool;
      if(!(fdcool = fopen(fname, "r")))
        Terminate(" Cannot read ionization table in file `%s'\n", fname);
      if(iter == 0)
        while(fscanf(fdcool, "%*g %*g %*g %*g %*g %*g %*g") != EOF)
          NheattabUVB++;
      if(iter == 1)
        while(fscanf(fdcool, "%g %g %g %g %g %g %g", &PhotoTUVB[i].variable, &PhotoTUVB[i].gH0, &PhotoTUVB[i].gHe, &PhotoTUVB[i].gHep,
                     &PhotoTUVB[i].eH0, &PhotoTUVB[i].eHe, &PhotoTUVB[i].eHep) != EOF)
          i++;
      fclose(fdcool);

      if(iter == 0)
        {
          PhotoTUVB = (photo_table *)Mem.mymalloc("PhotoT", NheattabUVB * sizeof(photo_table));
          mpi_printf("COOLING: read ionization table with %d entries in file `%s'.\n", NheattabUVB, fname);
        }
    }
  /* ignore zeros at end of treecool file */
  for(i = 0; i < NheattabUVB; ++i)
    if(PhotoTUVB[i].gH0 == 0.0)
      break;

  NheattabUVB = i;
  mpi_printf("COOLING: using %d ionization table entries from file `%s'.\n", NheattabUVB, fname);

  if(NheattabUVB < 1)
    Terminate("The length of the cooling table has to have at least one entry");
}

/** \brief Set the ionization parameters for the UV background.
 */
void coolsfr::IonizeParamsUVB(void)
{
  if(!All.ComovingIntegrationOn)
    {
      SetZeroIonization();
      return;
    }

  if(NheattabUVB == 1)
    {
      /* treat the one value given as constant with redshift */
      pc.J_UV   = 1;
      pc.gJH0   = PhotoTUVB[0].gH0;
      pc.gJHe0  = PhotoTUVB[0].gHe;
      pc.gJHep  = PhotoTUVB[0].gHep;
      pc.epsH0  = PhotoTUVB[0].eH0;
      pc.epsHe0 = PhotoTUVB[0].eHe;
      pc.epsHep = PhotoTUVB[0].eHep;
    }
  else
    {
      double redshift = 1 / All.Time - 1;
      double logz     = log10(redshift + 1.0);
      int ilow        = 0;
      for(int i = 0; i < NheattabUVB; i++)
        {
          if(PhotoTUVB[i].variable < logz)
            ilow = i;
          else
            break;
        }

      if(logz > PhotoTUVB[NheattabUVB - 1].variable || ilow >= NheattabUVB - 1)
        {
          SetZeroIonization();
        }
      else
        {
          double dzlow = logz - PhotoTUVB[ilow].variable;
          double dzhi  = PhotoTUVB[ilow + 1].variable - logz;

          if(PhotoTUVB[ilow].gH0 == 0 || PhotoTUVB[ilow + 1].gH0 == 0)
            {
              SetZeroIonization();
            }
          else
            {
              pc.J_UV   = 1;
              pc.gJH0   = pow(10., (dzhi * log10(PhotoTUVB[ilow].gH0) + dzlow * log10(PhotoTUVB[ilow + 1].gH0)) / (dzlow + dzhi));
              pc.gJHe0  = pow(10., (dzhi * log10(PhotoTUVB[ilow].gHe) + dzlow * log10(PhotoTUVB[ilow + 1].gHe)) / (dzlow + dzhi));
              pc.gJHep  = pow(10., (dzhi * log10(PhotoTUVB[ilow].gHep) + dzlow * log10(PhotoTUVB[ilow + 1].gHep)) / (dzlow + dzhi));
              pc.epsH0  = pow(10., (dzhi * log10(PhotoTUVB[ilow].eH0) + dzlow * log10(PhotoTUVB[ilow + 1].eH0)) / (dzlow + dzhi));
              pc.epsHe0 = pow(10., (dzhi * log10(PhotoTUVB[ilow].eHe) + dzlow * log10(PhotoTUVB[ilow + 1].eHe)) / (dzlow + dzhi));
              pc.epsHep = pow(10., (dzhi * log10(PhotoTUVB[ilow].eHep) + dzlow * log10(PhotoTUVB[ilow + 1].eHep)) / (dzlow + dzhi));
            }
        }
    }
}

/** \brief Reset the ionization parameters.
 */
void coolsfr::SetZeroIonization(void) { memset(&pc, 0, sizeof(photo_current)); }

/** \brief Wrapper function to set the ionizing background.
 */
void coolsfr::IonizeParams(void) { IonizeParamsUVB(); }

/** \brief Initialize the cooling module.
 *
 *   This function initializes the cooling module. In particular,
 *   it allocates the memory for the cooling rate and ionization tables
 *   and initializes them.
 */
void coolsfr::InitCool(void)
{
  /* set default hydrogen mass fraction */
  GasState.XH = HYDROGEN_MASSFRAC;

  /* zero photo-ionization/heating rates */
  SetZeroIonization();

  /* allocate and construct rate table */
  RateT = (rate_table *)Mem.mymalloc("RateT", (NCOOLTAB + 1) * sizeof(rate_table));
  ;
  MakeRateTable();

  /* read photo tables */
  ReadIonizeParams(All.TreecoolFile);

  All.Time = All.TimeBegin;
  All.set_cosmo_factors_for_current_time();

  IonizeParams();
}

/** \brief Apply the isochoric cooling to all the active gas particles.
 *
 */
void coolsfr::cooling_only(simparticles *Sp) /* normal cooling routine when star formation is disabled */
{
  TIMER_START(CPU_COOLING_SFR);
  All.set_cosmo_factors_for_current_time();

  gas_state gs        = GasState;
  do_cool_data DoCool = DoCoolData;

  for(int i = 0; i < Sp->TimeBinsHydro.NActiveParticles; i++)
    {
      int target = Sp->TimeBinsHydro.ActiveParticleList[i];
      if(Sp->P[target].getType() == 0)
        {
          if(Sp->P[target].getMass() == 0 && Sp->P[target].ID.get() == 0)
            continue; /* skip particles that have been swallowed or eliminated */

          cool_sph_particle(Sp, target, &gs, &DoCool);
        }
    }
  TIMER_STOP(CPU_COOLING_SFR);
}

/** \brief Apply the isochoric cooling to a given gas particle.
 *
 *  This function applies the normal isochoric cooling to a single gas particle.
 *  Once the cooling has been applied according to one of the cooling models implemented,
 *  the internal energy per unit mass, the total energy and the pressure of the particle are updated.
 *
 *  \param i index of the gas particle to which cooling is applied
 */
void coolsfr::cool_sph_particle(simparticles *Sp, int i, gas_state *gs, do_cool_data *DoCool)
{
  double dens = Sp->SphP[i].Density;

  double dt = (Sp->P[i].getTimeBinHydro() ? (((integertime)1) << Sp->P[i].getTimeBinHydro()) : 0) * All.Timebase_interval;

  double dtime = All.cf_atime * dt / All.cf_atime_hubble_a;

  double utherm = Sp->get_utherm_from_entropy(i);

  double ne      = Sp->SphP[i].Ne; /* electron abundance (gives ionization state and mean molecular weight) */
  double unew;
#ifdef RESEARCH_GRACKLE
  // This branch is reached only by single-phase cooling. SH03 effective-EOS
  // gas retains its own hot/cold closure; it is never cooled twice.
  unew = research_grackle_cool_particle(Sp,i);
  ne=Sp->SphP[i].Ne;
#else
  unew = DoCooling(std::max<double>(All.MinEgySpec, utherm), dens * All.cf_a3inv, dtime, &ne, gs, DoCool);
#endif
  Sp->SphP[i].Ne = ne;

  if(unew < 0)
    Terminate("invalid temperature: i=%d unew=%g\n", i, unew);

  double du = unew - utherm;

  if(unew < All.MinEgySpec)
    du = All.MinEgySpec - utherm;

  utherm += du;

#ifdef OUTPUT_COOLHEAT
  if(dtime > 0)
    Sp->SphP[i].CoolHeat = du * Sp->P[i].getMass() / dtime;
#endif

  Sp->set_entropy_from_utherm(utherm, i);
  Sp->SphP[i].set_thermodynamic_variables();
}

#endif
