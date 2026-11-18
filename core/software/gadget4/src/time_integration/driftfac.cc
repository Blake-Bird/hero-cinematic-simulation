/*******************************************************************************
 * \copyright   This file is part of the GADGET4 N-body/SPH code developed
 * \copyright   by Volker Springel. Copyright (C) 2014-2020 by Volker Springel
 * \copyright   (vspringel@mpa-garching.mpg.de) and all contributing authors.
 *******************************************************************************/

/*! \file  driftfac.cc
 *
 *  \brief tabulates cosmological drift/kick factors for fast look-up
 */

#include "gadgetconfig.h"

#include <gsl/gsl_integration.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_odeiv2.h>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <sstream>
#include <string.h>
#include <string>
#include <vector>

#include "../data/allvars.h"
#include "../data/dtypes.h"
#include "../mpi_utils/mpi_utils.h"
#include "../time_integration/driftfac.h"

namespace
{
std::vector<double> CosmologyTableA;
std::vector<double> CosmologyTableE;
std::vector<double> CosmologyTableD;
std::vector<double> CosmologyTableF;
bool CosmologyTableLoaded = false;

bool cosmology_table_requested()
{
  return All.CosmologyTableFile[0] && strcmp(All.CosmologyTableFile, "none") != 0;
}

double interpolate_log_positive(const std::vector<double> &values, double a)
{
  if(!CosmologyTableLoaded || a < CosmologyTableA.front() || a > CosmologyTableA.back())
    Terminate("Cosmology table interpolation requested outside its scale-factor range: a=%g, range=[%g,%g]\n", a,
             CosmologyTableA.empty() ? 0.0 : CosmologyTableA.front(), CosmologyTableA.empty() ? 0.0 : CosmologyTableA.back());

  auto upper = std::lower_bound(CosmologyTableA.begin(), CosmologyTableA.end(), a);
  if(upper == CosmologyTableA.begin())
    return values.front();
  if(upper == CosmologyTableA.end())
    return values.back();
  const size_t hi = static_cast<size_t>(upper - CosmologyTableA.begin());
  if(*upper == a)
    return values[hi];
  const size_t lo = hi - 1;
  const double fraction = (log(a) - log(CosmologyTableA[lo])) / (log(CosmologyTableA[hi]) - log(CosmologyTableA[lo]));
  return exp(log(values[lo]) + fraction * (log(values[hi]) - log(values[lo])));
}

double interpolate_linear_log_a(const std::vector<double> &values, double a)
{
  if(!CosmologyTableLoaded || a < CosmologyTableA.front() || a > CosmologyTableA.back())
    Terminate("Cosmology table interpolation requested outside its scale-factor range: a=%g, range=[%g,%g]\n", a,
             CosmologyTableA.empty() ? 0.0 : CosmologyTableA.front(), CosmologyTableA.empty() ? 0.0 : CosmologyTableA.back());

  auto upper = std::lower_bound(CosmologyTableA.begin(), CosmologyTableA.end(), a);
  if(upper == CosmologyTableA.begin())
    return values.front();
  if(upper == CosmologyTableA.end())
    return values.back();
  const size_t hi = static_cast<size_t>(upper - CosmologyTableA.begin());
  if(*upper == a)
    return values[hi];
  const size_t lo = hi - 1;
  const double fraction = (log(a) - log(CosmologyTableA[lo])) / (log(CosmologyTableA[hi]) - log(CosmologyTableA[lo]));
  return values[lo] + fraction * (values[hi] - values[lo]);
}

double integrate_tabulated_factor(double a0, double a1, double power)
{
  if(a1 < a0 || a0 < CosmologyTableA.front() || a1 > CosmologyTableA.back() || All.Hubble <= 0)
    Terminate("Invalid interval for direct cosmology-table integral: [%g,%g]\n", a0, a1);

  double integral = 0.0;
  double a = a0;
  while(a < a1)
    {
      auto upper = std::upper_bound(CosmologyTableA.begin(), CosmologyTableA.end(), a);
      if(upper == CosmologyTableA.end())
        break;
      const size_t hi = static_cast<size_t>(upper - CosmologyTableA.begin());
      const size_t lo = hi - 1;
      const double b = std::min(a1, CosmologyTableA[hi]);
      const double x0 = log(a);
      const double x1 = log(b);
      const double slope = (log(CosmologyTableE[hi]) - log(CosmologyTableE[lo])) /
                           (log(CosmologyTableA[hi]) - log(CosmologyTableA[lo]));
      const double e_at_a = exp(log(CosmologyTableE[lo]) + slope * (x0 - log(CosmologyTableA[lo])));
      const double exponent = 1.0 - power - slope;
      const double dx = x1 - x0;
      const double scaled_interval = fabs(exponent * dx) < 1.0e-8 ? dx : expm1(exponent * dx) / exponent;
      integral += exp((1.0 - power) * x0 - log(All.Hubble) - log(e_at_a)) * scaled_interval;
      a = b;
    }
  return integral;
}
}

double driftfac::E_of_a(double a)
{
  if(cosmology_table_requested())
    return interpolate_log_positive(CosmologyTableE, a);
  return sqrt(All.Omega0 / pow(a, 3) + All.OmegaCurvature / pow(a, 2) + All.OmegaRadiation / pow(a, 4) + All.OmegaLambda);
}

double driftfac::E_of_a_diff(double a)
{
  if(cosmology_table_requested())
    {
      const double da = 1.0e-5 * a;
      const double lo = std::max(CosmologyTableA.front(), a - da);
      const double hi = std::min(CosmologyTableA.back(), a + da);
      return (E_of_a(hi) - E_of_a(lo)) / (hi - lo);
    }
  return (-3 * All.Omega0 / pow(a, 4) - 2 * All.OmegaCurvature / pow(a, 3) - 4 * All.OmegaRadiation / pow(a, 5)) / (2 * E_of_a(a));
}

void driftfac::load_cosmology_table(void)
{
  if(!cosmology_table_requested() || CosmologyTableLoaded)
    return;

  std::ifstream input(All.CosmologyTableFile);
  if(!input)
    Terminate("Could not open CosmologyTableFile '%s'\n", All.CosmologyTableFile);

  std::string line;
  int line_number = 0;
  while(std::getline(input, line))
    {
      line_number++;
      const size_t first = line.find_first_not_of(" \t\r");
      if(first == std::string::npos || line[first] == '#')
        continue;
      std::istringstream row(line);
      double a, e, d, f;
      if(!(row >> a >> e >> d >> f) || !std::isfinite(a) || !std::isfinite(e) || !std::isfinite(d) || !std::isfinite(f) ||
         a <= 0 || e <= 0 || d <= 0 || f <= 0 || (!CosmologyTableA.empty() && a <= CosmologyTableA.back()))
        Terminate("Invalid or unordered row %d in cosmology table '%s'\n", line_number, All.CosmologyTableFile);
      CosmologyTableA.push_back(a);
      CosmologyTableE.push_back(e);
      CosmologyTableD.push_back(d);
      CosmologyTableF.push_back(f);
    }

  if(CosmologyTableA.size() < 2 || CosmologyTableA.front() > std::min(All.TimeBegin, 1.0e-3) || CosmologyTableA.back() < 1.0)
    Terminate("CosmologyTableFile '%s' must cover from a<=min(TimeBegin,0.001) through a=1\n", All.CosmologyTableFile);

  CosmologyTableLoaded = true;
}

void driftfac::init_drift_table(void)
{
#define WORKSIZE 100000

  load_cosmology_table();

  gsl_function F;
  gsl_integration_workspace *workspace;

  logTimeBegin = log(All.TimeBegin);
  logTimeMax   = log(All.TimeMax);

  workspace = gsl_integration_workspace_alloc(WORKSIZE);

  for(int i = 0; i < DRIFT_TABLE_LENGTH; i++)
    {
      double result, abserr;

      F.function = &driftfac::drift_integ;
      gsl_integration_qag(&F, exp(logTimeBegin), exp(logTimeBegin + ((logTimeMax - logTimeBegin) / DRIFT_TABLE_LENGTH) * (i + 1)), 0,
                          1.0e-8, WORKSIZE, GSL_INTEG_GAUSS41, workspace, &result, &abserr);
      DriftTable[i] = result;

      F.function = &gravkick_integ;
      gsl_integration_qag(&F, exp(logTimeBegin), exp(logTimeBegin + ((logTimeMax - logTimeBegin) / DRIFT_TABLE_LENGTH) * (i + 1)), 0,
                          1.0e-8, WORKSIZE, GSL_INTEG_GAUSS41, workspace, &result, &abserr);
      GravKickTable[i] = result;

      F.function = &hydrokick_integ;
      gsl_integration_qag(&F, exp(logTimeBegin), exp(logTimeBegin + ((logTimeMax - logTimeBegin) / DRIFT_TABLE_LENGTH) * (i + 1)), 0,
                          1.0e-8, WORKSIZE, GSL_INTEG_GAUSS41, workspace, &result, &abserr);
      HydroKickTable[i] = result;
    }

  if(cosmology_table_requested())
    {
      double max_drift_quadrature_error = 0.0;
      double max_gravkick_quadrature_error = 0.0;
      double max_hydrokick_quadrature_error = 0.0;
      double max_drift_lookup_error = 0.0;
      double max_gravkick_lookup_error = 0.0;
      double max_hydrokick_lookup_error = 0.0;
      const double log_step = (logTimeMax - logTimeBegin) / DRIFT_TABLE_LENGTH;
      const double hydro_power = 1.0 + 3.0 * GAMMA_MINUS1;

      for(int i = 0; i < DRIFT_TABLE_LENGTH; i += 10)
        {
          const double a_end = exp(logTimeBegin + log_step * (i + 1));
          const double drift_exact = integrate_tabulated_factor(All.TimeBegin, a_end, 3.0);
          const double gravkick_exact = integrate_tabulated_factor(All.TimeBegin, a_end, 2.0);
          const double hydrokick_exact = integrate_tabulated_factor(All.TimeBegin, a_end, hydro_power);
          max_drift_quadrature_error = std::max(max_drift_quadrature_error, fabs(DriftTable[i] / drift_exact - 1.0));
          max_gravkick_quadrature_error = std::max(max_gravkick_quadrature_error, fabs(GravKickTable[i] / gravkick_exact - 1.0));
          max_hydrokick_quadrature_error = std::max(max_hydrokick_quadrature_error, fabs(HydroKickTable[i] / hydrokick_exact - 1.0));
        }

      for(int i = 2; i < DRIFT_TABLE_LENGTH - 1; i += 10)
        {
          const double a_mid = exp(logTimeBegin + log_step * (i + 0.5));
          const double drift_exact = integrate_tabulated_factor(All.TimeBegin, a_mid, 3.0);
          const double gravkick_exact = integrate_tabulated_factor(All.TimeBegin, a_mid, 2.0);
          const double hydrokick_exact = integrate_tabulated_factor(All.TimeBegin, a_mid, hydro_power);
          const double drift_lookup = 0.5 * (DriftTable[i - 1] + DriftTable[i]);
          const double gravkick_lookup = 0.5 * (GravKickTable[i - 1] + GravKickTable[i]);
          const double hydrokick_lookup = 0.5 * (HydroKickTable[i - 1] + HydroKickTable[i]);
          max_drift_lookup_error = std::max(max_drift_lookup_error, fabs(drift_lookup / drift_exact - 1.0));
          max_gravkick_lookup_error = std::max(max_gravkick_lookup_error, fabs(gravkick_lookup / gravkick_exact - 1.0));
          max_hydrokick_lookup_error = std::max(max_hydrokick_lookup_error, fabs(hydrokick_lookup / hydrokick_exact - 1.0));
        }

      const double tolerance = All.CosmologyTableValidationTolerance;
      const double max_error = std::max(
          std::max(std::max(max_drift_quadrature_error, max_gravkick_quadrature_error), max_hydrokick_quadrature_error),
          std::max(std::max(max_drift_lookup_error, max_gravkick_lookup_error), max_hydrokick_lookup_error));
      if(max_error > tolerance)
        Terminate("CLASS/AxiCLASS drift/kick table validation failed: max relative error=%g, tolerance=%g\n", max_error, tolerance);

      int rank = 0;
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      if(rank == 0)
        {
          char fname[MAXLEN_PATH_EXTRA];
          snprintf(fname, MAXLEN_PATH_EXTRA, "%scosmology_table_validation.txt", All.OutputDir);
          FILE *fd = fopen(fname, "w");
          if(!fd)
            Terminate("Could not write cosmology-table validation report '%s'\n", fname);
          fprintf(fd, "status PASS\n");
          fprintf(fd, "interval %.17g %.17g\n", All.TimeBegin, All.TimeMax);
          fprintf(fd, "tolerance %.17g\n", tolerance);
          fprintf(fd, "max_relative_drift_quadrature_error %.17g\n", max_drift_quadrature_error);
          fprintf(fd, "max_relative_gravkick_quadrature_error %.17g\n", max_gravkick_quadrature_error);
          fprintf(fd, "max_relative_hydrokick_quadrature_error %.17g\n", max_hydrokick_quadrature_error);
          fprintf(fd, "max_relative_drift_lookup_error %.17g\n", max_drift_lookup_error);
          fprintf(fd, "max_relative_gravkick_lookup_error %.17g\n", max_gravkick_lookup_error);
          fprintf(fd, "max_relative_hydrokick_lookup_error %.17g\n", max_hydrokick_lookup_error);
          fclose(fd);
        }
    }

  gsl_integration_workspace_free(workspace);
}

/*! This function integrates the cosmological prefactor for a drift
 *   step between time0 and time1. The value returned is
 *  \f[ \int_{a_0}^{a_1} \frac{{\rm d}a}{a^3 * H(a)}
 *  \f]
 *
 *  A lookup-table is used for reasons of speed.
 */
double driftfac::get_drift_factor(integertime time0, integertime time1)
{
  static integertime last_time0 = -1, last_time1 = -1;
  static double last_value;

  if(time0 == last_time0 && time1 == last_time1)
    return last_value;

  /* note: will only be called for cosmological integration */

  double a1 = logTimeBegin + time0 * All.Timebase_interval;
  double a2 = logTimeBegin + time1 * All.Timebase_interval;

  double u1 = (a1 - logTimeBegin) / (logTimeMax - logTimeBegin) * DRIFT_TABLE_LENGTH;
  int i1    = (int)u1;
  if(i1 >= DRIFT_TABLE_LENGTH)
    i1 = DRIFT_TABLE_LENGTH - 1;

  double df1;
  if(i1 <= 1)
    df1 = u1 * DriftTable[0];
  else
    df1 = DriftTable[i1 - 1] + (DriftTable[i1] - DriftTable[i1 - 1]) * (u1 - i1);

  double u2 = (a2 - logTimeBegin) / (logTimeMax - logTimeBegin) * DRIFT_TABLE_LENGTH;
  int i2    = (int)u2;
  if(i2 >= DRIFT_TABLE_LENGTH)
    i2 = DRIFT_TABLE_LENGTH - 1;

  double df2;
  if(i2 <= 1)
    df2 = u2 * DriftTable[0];
  else
    df2 = DriftTable[i2 - 1] + (DriftTable[i2] - DriftTable[i2 - 1]) * (u2 - i2);

  last_time0 = time0;
  last_time1 = time1;

  return last_value = (df2 - df1);
}

double driftfac::get_gravkick_factor(integertime time0, integertime time1)
{
  static integertime last_time0 = -1, last_time1 = -1;
  static double last_value;

  if(time0 == last_time0 && time1 == last_time1)
    return last_value;

  /* note: will only be called for cosmological integration */

  double a1 = logTimeBegin + time0 * All.Timebase_interval;
  double a2 = logTimeBegin + time1 * All.Timebase_interval;

  double u1 = (a1 - logTimeBegin) / (logTimeMax - logTimeBegin) * DRIFT_TABLE_LENGTH;
  int i1    = (int)u1;
  if(i1 >= DRIFT_TABLE_LENGTH)
    i1 = DRIFT_TABLE_LENGTH - 1;

  double df1;
  if(i1 <= 1)
    df1 = u1 * GravKickTable[0];
  else
    df1 = GravKickTable[i1 - 1] + (GravKickTable[i1] - GravKickTable[i1 - 1]) * (u1 - i1);

  double u2 = (a2 - logTimeBegin) / (logTimeMax - logTimeBegin) * DRIFT_TABLE_LENGTH;
  int i2    = (int)u2;
  if(i2 >= DRIFT_TABLE_LENGTH)
    i2 = DRIFT_TABLE_LENGTH - 1;

  double df2;
  if(i2 <= 1)
    df2 = u2 * GravKickTable[0];
  else
    df2 = GravKickTable[i2 - 1] + (GravKickTable[i2] - GravKickTable[i2 - 1]) * (u2 - i2);

  last_time0 = time0;
  last_time1 = time1;

  return last_value = (df2 - df1);
}

double driftfac::get_hydrokick_factor(integertime time0, integertime time1)
{
  static integertime last_time0 = -1, last_time1 = -1;
  static double last_value;

  if(time0 == last_time0 && time1 == last_time1)
    return last_value;

  /* note: will only be called for cosmological integration */

  double a1 = logTimeBegin + time0 * All.Timebase_interval;
  double a2 = logTimeBegin + time1 * All.Timebase_interval;

  double u1 = (a1 - logTimeBegin) / (logTimeMax - logTimeBegin) * DRIFT_TABLE_LENGTH;
  int i1    = (int)u1;
  if(i1 >= DRIFT_TABLE_LENGTH)
    i1 = DRIFT_TABLE_LENGTH - 1;

  double df1;
  if(i1 <= 1)
    df1 = u1 * HydroKickTable[0];
  else
    df1 = HydroKickTable[i1 - 1] + (HydroKickTable[i1] - HydroKickTable[i1 - 1]) * (u1 - i1);

  double u2 = (a2 - logTimeBegin) / (logTimeMax - logTimeBegin) * DRIFT_TABLE_LENGTH;
  int i2    = (int)u2;
  if(i2 >= DRIFT_TABLE_LENGTH)
    i2 = DRIFT_TABLE_LENGTH - 1;

  double df2;
  if(i2 <= 1)
    df2 = u2 * HydroKickTable[0];
  else
    df2 = HydroKickTable[i2 - 1] + (HydroKickTable[i2] - HydroKickTable[i2 - 1]) * (u2 - i2);

  last_time0 = time0;
  last_time1 = time1;

  return last_value = (df2 - df1);
}

double driftfac::get_comoving_distance(integertime time0)
{
  /* we just need to multiply this with the speed of light to get the correct cosmological factor */
  double fac = get_gravkick_factor(time0, TIMEBASE);

  return fac * (CLIGHT / All.UnitVelocity_in_cm_per_s);
}

double driftfac::get_comoving_distance_for_scalefactor(double ascale)
{
  integertime time0 = log(ascale / All.TimeBegin) / All.Timebase_interval;

  /* we just need to multiply this with the speed of light to get the correct cosmological factor */
  double fac = get_gravkick_factor(time0, TIMEBASE);

  return fac * (CLIGHT / All.UnitVelocity_in_cm_per_s);
}

double driftfac::get_scalefactor_for_comoving_distance(double dist)
{
  double fac = dist / (CLIGHT / All.UnitVelocity_in_cm_per_s);

  integertime time0 = get_gravkick_factor_inverse(fac);

  double ascale = All.TimeBegin * exp(time0 * All.Timebase_interval);

  return ascale;
}

integertime driftfac::get_gravkick_factor_inverse(double fac)
{
  integertime time1 = TIMEBASE;

  double a2 = logTimeBegin + time1 * All.Timebase_interval;

  double u2 = (a2 - logTimeBegin) / (logTimeMax - logTimeBegin) * DRIFT_TABLE_LENGTH;
  int i2    = (int)u2;
  if(i2 >= DRIFT_TABLE_LENGTH)
    i2 = DRIFT_TABLE_LENGTH - 1;

  double df2;
  if(i2 <= 1)
    df2 = u2 * GravKickTable[0];
  else
    df2 = GravKickTable[i2 - 1] + (GravKickTable[i2] - GravKickTable[i2 - 1]) * (u2 - i2);

  double df1 = df2 - fac;

  int i0 = 0;
  int i1 = DRIFT_TABLE_LENGTH - 1;

  if(df1 < 0 || df1 > GravKickTable[i1])
    Terminate("out of range:  df1=%g  GravKickTable[i0]=%g  GravKickTable[i1]=%g\n", df1, GravKickTable[i0], GravKickTable[i1]);

  double u1;

  if(df1 <= GravKickTable[0])
    u1 = df1 / GravKickTable[0];
  else
    {
      while(i1 - i0 > 1)
        {
          int im = (i0 + i1) / 2;
          if(df1 < GravKickTable[im])
            i1 = im;
          else
            i0 = im;
        }

      u1 = (df1 - GravKickTable[i0]) / (GravKickTable[i1] - GravKickTable[i0]) + i1;
    }

  double a1 = u1 * (logTimeMax - logTimeBegin) / DRIFT_TABLE_LENGTH + logTimeBegin;

  integertime time0 = (a1 - logTimeBegin) / All.Timebase_interval;

  return time0;
}

double driftfac::linear_growth_factor(const double astart, const double aend)
{
  if(cosmology_table_requested())
    {
      load_cosmology_table();
      return interpolate_log_positive(CosmologyTableD, aend) / interpolate_log_positive(CosmologyTableD, astart);
    }
  // if we have a simple cosmology without radiation, with matter, a cosmological constant, and flat space,
  // we can apply the simple integral solution:

  ///  return linear_growth_simple(aend) / linear_growth_simple(astart);

  // but in the more general case, we need to integrate the ODE directly to find the correct
  // growth factor.

  return linear_growth_ode(aend) / linear_growth_ode(astart);
}

double driftfac::linear_growth_rate(const double a)
{
  if(cosmology_table_requested())
    {
      load_cosmology_table();
      return interpolate_linear_log_a(CosmologyTableF, a);
    }
  double rate;
  linear_growth_ode(a, &rate);
  return rate;
}

double driftfac::linear_growth_ode(const double a, double *growth_rate)
{
  gsl_odeiv2_system sys = {growth_ode_int, NULL, 2, NULL};
  gsl_odeiv2_driver *d  = gsl_odeiv2_driver_alloc_y_new(&sys, gsl_odeiv2_step_rk8pd, 1e-6, 1e-6, 0.0);

  double amin = 0.00001;  // start time

  // In a matter+radiation background, initialize with the regular growing
  // Meszaros solution while radiation perturbations are smooth.  The old
  // matter-only initial condition gives an incorrect early growth rate.
  double D[2];
  if(All.OmegaRadiation > 0)
    {
      const double a_eq = All.OmegaRadiation / All.Omega0;
      const double y    = amin / a_eq;
      D[0]              = 1.0 + 1.5 * y;
      D[1]              = 1.5 / a_eq;
    }
  else
    {
      double epsilon = 3 + 2 * amin * E_of_a_diff(amin) / E_of_a(amin);
      double w       = 1 - get_OmegaMatter_a(amin);
      double n       = 0.25 * (-1 - epsilon + sqrt(pow(1 + epsilon, 2) + 24 * (1 - w)));
      D[0]            = 1.0;
      D[1]            = n / amin;
    }

  double acurrent = amin;

  int status = gsl_odeiv2_driver_apply(d, &acurrent, a, D);

  if(status != GSL_SUCCESS)
    Terminate("error, ODE return value=%d\n", status);

  if(growth_rate)
    *growth_rate = a * D[1] / D[0];

  gsl_odeiv2_driver_free(d);

  return D[0];
}

double driftfac::linear_growth_simple(const double a)
{
  /* this simple integral solution is only correct for LCDM variants, i.e.
   * no radition, flat space, and simple cosmological constant
   */
  const int worksize = WORKSIZE;

  double result, abserr;
  gsl_function F;

  gsl_integration_workspace *workspace = gsl_integration_workspace_alloc(worksize);
  F.function                           = &growth_simple_int;

  gsl_integration_qag(&F, 0, a, 0, 1.0e-8, worksize, GSL_INTEG_GAUSS41, workspace, &result, &abserr);

  gsl_integration_workspace_free(workspace);

  const double hubble_a = hubble_function(a) / All.Hubble;

  return hubble_a * result;
}
