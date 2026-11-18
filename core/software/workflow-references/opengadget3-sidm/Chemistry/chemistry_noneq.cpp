/**
 * @file
 *
 * This file is part of the N-body/SPH code OpenGadget3.
 *
 * @brief This file contains various variables and functions related to the chemistry module.
 *
 * @author OpenGadget3-development team, in particular Umberto Maio.
 *
 * @copyright Copyright (c) 2026, OpenGadget3-development team lead by Klaus Dolag and Stefano Borgani
 * (see change logs). All rights reserved.
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License v3 or later.
 * See <https://www.gnu.org/licenses/>.
 */

/* this part must be included in the Makefile with the OPT -DCHEMISTRY off */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "../CodeBase/allvars.hpp"
#include "../CodeBase/proto.hpp"
#include "../System/tags.hpp"
#include "chemistry.hpp"

#if defined(UM_CHEMISTRY) && !(defined(CHEMISTRY))
/*  chemical reactions                                  */
/*  mostly from Abel, Annions, Zhang & Norman (1997), with recent updated rates

NB: Abel recombination fits are for "case A recombination" and are not appropriate for high density (redshift) gas;
better to use Hui&Gnedin1997 case B recombination!

   ------- 1:	HI    + e   -> HII   + 2e    hydrogen collisional ionization
   ------- 2:	HII   + e   -> H     + p     hydrogen recombination
   ------- 3:	HeI   + e   -> HeII  + 2e    He collisional ionization
   ------- 4:	HeII  + e   -> HeI   + p     He recombination
   ------- 5:	HeII  + e   -> HeIII + 2e    HeII collisional ionization
   ------- 6:	HeIII + e   -> HeII  + p     HeII recombination
   ------- 7:	HI    + e   -> HM    + p     photo-attachment
   ------- 8:	HM    + HI  -> H2I*  + e     H2 formation path (H-)
   ------- 9:   HI    + HII -> H2II  + p     H+ channel
   ------- 10:	H2II  + HI  -> H2I*  + HII   H+ channel
   ------- 11:	H2I   + H   -> 3H
   ------- 12:	H2I   + HII -> H2II  + H
   ------- 13:    omitted
   ------- 14:	H2I   + e   -> 2HI   + e     collisional dissociation
   ------- 15:    omitted
   ------- 16:	HM    + e   -> HI    + 2e
   ------- 17:	HM    + HI  -> 2H    + e
   ------- 18:	HM    + HII -> 2HI
   ------- 19:	HM    + HII -> H2II  + e
   ------- 20:	H2II  + e   -> 2HI
   ------- 21:	H2II  + HM  -> HI    + H2I

   ------- 24:	HI    + p   -> HII   + e     photo-ionization H
   ------- 25:	HeII  + p   -> HeIII + e     HeII ionization
   ------- 26:	HeI   + p   -> HeII  + e     photo-ionization He
   ------- 27:	HM    + p   -> HI    + e     photo-detachment
   ------- 28:	H2II  + p   -> HI    + HII   low energy photo-dissociation
   ------- 29:	H2I   + p   -> H2II  + e
   ------- 30:	H2II  + p   -> 2HII  + e
   ------- 31:	H2I   + p   -> 2HI           photo-dissociation      */

/*for HD :
 ------- 1:     DI   + H2   ->   HD   + HI
 ------- 2:     DII  + H2   ->   HD   + HII
 ------- 3:     HD   + HI   ->   DI   + H2
 ------- 4:     HD   + HII  ->   DII  + H2
 ------- 5:     HII  + DI   ->   H    + DII
 ------- 6:     H    + DII  ->   HII  + DI
 ------- 7:     DII   + e   ->   DI + p     D recombination  (as H in reaction 2! unknown/unsure)
 ------- 8:     DI    + p   ->   DII + e    photo-ionization D (as H in reaction 24! unknown/unsure)

cooling fct from
Lipovka, Nunez-Lopez & Avila-Reese (MNRAS, 361, 850-854, 2005)  */

/* ----- Some grid bounds. */
#define T_start 0.001
#define T_end 1.0e9
#define nu_start 0.74
#define nu_end 100.0

static struct base_info
{
    float len;
    float time;
    float vol;
    float numden;
    float J;
    float kunit;
} base;

/* -------------- VARIABLES ----------------- */

/* ----- Grid spacings. */
static double dlog_T;

#ifndef UM_NO_RADIATION
static double dlog_nu;
#endif // UM_NO_RADIATION

static double piHI, piHeI, piHeII;

#ifdef UM_SEN_H2_SOLOMON_PROCESS
float z_JLW[JLWn], JLW[JLWn];
static double solH2I;
#endif // UM_SEN_H2_SOLOMON_PROCESS

/* ----- Energy arrays  */
static double edot_ceHI, edot_ceHeI, edot_ceHeII;
static double edot_ciHI, edot_ciHeI, edot_ciHeII;
static double edot_ciHeIS, edot_reHI, edot_reHeII1;
static double edot_reHeII2, edot_reHeIII, edot_comp;
static double edot_brem, edot_H2;
static double edot_radHI, edot_radHeI, edot_radHeII, edot_rad;
static double edot_elecH2II, edot_HydH2II;
static double edot_HD;

#ifdef UM_SEN_H2_SOLOMON_PROCESS
static double edot_solH2I;
#endif // UM_SEN_H2_SOLOMON_PROCESS

static double k24, k25, k26, k27, k28, k29, k30, k31;
static double kHeHII3, kHD8;

#ifdef UM_SEN_H2_SOLOMON_PROCESS
static double k31sol;
#endif // UM_SEN_H2_SOLOMON_PROCESS

#if defined(UM_METAL_COOLING) && (defined(LT_METAL_COOLING) || defined(LT_METAL_COOLING_WAL)) &&                       \
    defined(UM_MET_IN_NONEQ_COOLING)
static double Hmass;
static double edot_met;
static double dummy_average_atomic_mass = 40.;
#endif /* UM_METAL_COOLING + LT_METAL_COOLING + UM_MET_IN_NONEQ_COOLING */
// static double um_u;

#ifdef UM_SEN_H2_SOLOMON_PROCESS
void solomon_rates_init(void);
void load_JLW(void);
double get_JLW(double);
#endif // UM_SEN_H2_SOLOMON_PROCESS

/* energy_solver (used here only) */
int energy_solver(int solve_it_flag, int ithis, double current_a, double dt, double rho_gas, double T_gas,
                  double *T_gas_out, double *t_cool, double *um_coolingfct);

// *** int energy_solver(int solve_it_flag, int ithis, double current_a, double dt, double rho_gas, double T_gas, double
// *T_gas_out, double *t_cool)
int species_solver(int solve_it, int ithis, double a, double dt, double rho_gas, double T_gas, double *elec_dot);

int compute_abundances(int mode, int ithis, double a_start, double a_end, double *um_energy);

int radiative_rates(int i, double a, double J_21);

int chem_tabulate(void);
int init_rad(double);
int heap_sort(unsigned long n, double ra[]);
double radiation_field(double a);
int compute_gamma(int ipart);

/**
 * @brief Computes the effective adiabatic index (Gamma) for a gas particle.
 *
 * This function calculates the adiabatic index \(\Gamma\) based on the particle's
 * current state, including hydrogen and helium ionization states, and electron density.
 * The computation involves converting entropy to temperature and evaluating various
 * contributions to determine the final Gamma value.
 *
 * @param ithis The index of the particle for which Gamma is being computed.
 *
 * @return An integer indicating the success of the operation. (Always returns 1)
 */
int compute_gamma(int ithis)
{
    /* variable gamma
       Equations (5)-(7) in Omukai-Nishi */
    int i = 1;
    double y_HI, y_HII, y_HM, y_HeI, y_HeII, y_HeIII, y_elec, y_H2;
    double n_H; /* number density of hydrogen nuclei */
    double x, egyspec_to_temp, gas_temp, denominator, sum_y;
    double gamma_H2_minus1_inv;
    double rho_gas, a3inv, egyspec;

    exit(3333); /* not yet clarified so not used */

    a3inv = 1 / (All.Time * All.Time * All.Time);
    rho_gas = All.UnitDensity_in_cgs * SphP[ithis].Density * All.HubbleParam * All.HubbleParam * a3inv;
    Um_Compute_n_den(rho_gas, ithis);
    double mean_nden = Um_Compute_mean_nden(ithis);

    /* unit conversion */
    egyspec_to_temp = All.UnitPressure_in_cgs / All.UnitDensity_in_cgs * (SphP[ithis].Gamma - 1.0) / BOLTZMANN *
                      PROTONMASS / (mean_nden * base.numden);

    egyspec = SphP[ithis].Entropy / GAMMA_MINUS1 * pow(SphP[ithis].Density * a3inv, GAMMA_MINUS1);
    gas_temp = egyspec * egyspec_to_temp;

    n_H = SphP[ithis].HI + SphP[ithis].HII + SphP[ithis].HM + 2 * SphP[ithis].H2I;

    y_HI = SphP[ithis].HI / n_H;
    y_HII = SphP[ithis].HII / n_H;
    y_HeI = SphP[ithis].HeI / n_H;
    y_HeII = SphP[ithis].HeII / n_H;
    y_HeIII = SphP[ithis].HeIII / n_H;
    y_HM = SphP[ithis].HM / n_H;
    y_H2 = SphP[ithis].H2I / n_H;
    y_elec = SphP[ithis].elec / n_H;

    x = 6100.0 / gas_temp;
    gamma_H2_minus1_inv = 0.5 * (5.0 + 2.0 * x * x * exp(x) / (exp(x) - 1.0) / (exp(x) - 1.0));
    sum_y = y_HI + y_HII + y_HM + y_HeI + y_HeII + y_HeIII + y_elec + y_H2;
    denominator = 1.5 * (y_HI + y_HII + y_HM + y_HeI + y_HeII + y_HeIII + y_elec) + gamma_H2_minus1_inv * y_H2;

    SphP[ithis].Gamma = 1.0 + sum_y / denominator;

    ENDRUN_IF((SphP[ithis].Gamma > 1.8 || SphP[ithis].Gamma < 1.4), "Gama is out of range: %g\n", SphP[ithis].Gamma);

    return (i);
}

/**
 * @brief Computes the new temperature, number density, and fraction of a gas particle.
 *
 * This function calculates the updated properties of a gas particle by evolving its temperature and density
 * over a specified range of scale factors. It handles different modes of operation and considers whether
 * comoving integration is active. The function also computes cooling and heating rates and updates species
 * abundances based on the current state of the particle.
 *
 * @param mode An integer representing the mode of operation:
 *             - 0: Only compute the time step and return.
 *             - 1: Perform full evolution of the particle.
 * @param ithis The index of the particle.
 * @param a_start The starting scale factor for the computation.
 * @param a_end The ending scale factor for the computation.
 * @param um_energy Pointer to a double where the energy per unit mass will be stored.
 *
 * @return The number of iterations performed during the evolution.
 */
int compute_abundances(int mode, int ithis, double a_start, double a_end, double *um_energy)
{
    /* *******************************************************************
       Variables:
       ********************************************************************  */

    if (a_start == a_end)
    {
        a_end = a_start * (1 + 0.001); /* arbitrary shift */
    }

    /*    time variable */
    double a_now;

    /*    temperature, density in g/cm^3, pressure in K/cm^3  */
    double T_gas, rho_gas, T_gas_in, T_gas_out;

    /*    time stepping, epsilon: fraction of t_cool over which we subcycle. */
    double dt = 0, dt_in = 0, t_cool = 0, elec_dot = 0;
    double dt_min = 0, time1 = 0, time2 = 0, time3 = 0;

    int advance_flag, MAX_ITERATION;
    int i = 0;

    /* hydro and geometry: [T_gas]=K, [rho_gas]=gram/cm^3  */
    double i_T_gas, i_rho_gas;

    /* Radiation field: J21 is the max amplitude of the radiation field.
       The time (in)dependent radiation field is specified in
       radiation_model.f */
    double J_21;
    double a3inv, hubble_a, da;
    double egyspec_to_temp, egyspec;
    double mean_nden, molecular_weight;
    double um_coolingfct;

    /* *******************************************************************
       Initialization
       ******************************************************************* */

#ifndef UM_DISABLE_CMB
    J_21 = 1; /* no normalization necessary for CMB */
#else // UM_DISABLE_CMB
    J_21 = 0; /* radiation field */
#endif // UM_DISABLE_CMB

    kHeHII3 = 0;
    kHD8 = 0;
#ifndef UM_DISABLE_RADIATION
    int ifunc = radiative_rates(ithis, a_start, J_21);
#else // UM_DISABLE_RADIATION
    k24 = k25 = k26 = k27 = k28 = k29 = k30 = k31 = 0;
    piHI = piHeI = piHeII = 0;
#endif // UM_DISABLE_RADIATION

    if (All.ComovingIntegrationOn)
    {
        a3inv = 1 / (a_start * a_start * a_start);
        rho_gas = All.UnitDensity_in_cgs * SphP[ithis].Density * All.HubbleParam * All.HubbleParam *
                  a3inv; /* physical gram / cm^3 */
    }
    else
    {
        a3inv = 1;
        rho_gas = All.UnitDensity_in_cgs * SphP[ithis].Density;
    }

    /* here mean molecular weight must be taken into account */
    base.numden = Um_Compute_basenumden(rho_gas, ithis);
    mean_nden = Um_Compute_mean_nden(ithis);

    /* this returns SphP[].Um_MeanMolecularWeight: */
    molecular_weight = Um_Compute_MeanMolecularWeight(ithis);

    /* unit conversion */
    egyspec_to_temp = All.UnitPressure_in_cgs / All.UnitDensity_in_cgs * (SphP[ithis].Gamma - 1.0) / BOLTZMANN *
                      PROTONMASS * molecular_weight;

    egyspec = SphP[ithis].Entropy / GAMMA_MINUS1 * pow(SphP[ithis].Density * a3inv, GAMMA_MINUS1);
    T_gas = egyspec * egyspec_to_temp;

    i_T_gas = T_gas; /* starting value */
    i_rho_gas = rho_gas;

    if (P[ithis].ID == 1)
    {
        VERBOSE_ALL(2, " **** UV photoionization rates in cgs: gJH0uvb = %g, gJHe0uvb = %g, gJHepuvb = %g\n", gJH0uvb,
                    gJHe0uvb, gJHepuvb);
        VERBOSE_ALL(2, " **** UV photoheating rates in cgs: epsH0uvb = %g, epsHe0uvb = %g, epsHepuvb = %g\n", epsH0uvb,
                    epsHe0uvb, epsHepuvb);
        VERBOSE_ALL(2, " **** updated chemistry rates (internal units of 1/time) k24 = %g, k26 = %g, k25 = %g\n", k24,
                    k26, k25);
        VERBOSE_ALL(2, " **** internal units: base.time = 1e6 sec-per-yr = %g, base.numden = %g\n", base.time,
                    base.numden);
        VERBOSE_ALL(2, "ID=1: Initial temperature=%g, entropy=%g,\n GAMMA_MINUS1=%g, density=%g, a3inv=%g\n", i_T_gas,
                    SphP[ithis].Entropy, GAMMA_MINUS1, SphP[ithis].Density, a3inv);
        VERBOSE_ALL(2, "Initial mass density in cgs = %g\n", i_rho_gas);
        VERBOSE_ALL(2, "Initial number density in cgs = %g\n", i_rho_gas / (PROTONMASS * molecular_weight));
        VERBOSE_ALL(2, "egyspec_to_temp = %g, mean_nden = %g, molecular_weight = %g\n", egyspec_to_temp, mean_nden,
                    molecular_weight);
    }

    /*here: cooling............................................ */
    advance_flag = 0;
    dt_in = 0;
    T_gas_in = T_gas;

    /* check this */
    if (SphP[ithis].elec < 1e-20)
    {
        SphP[ithis].elec =
            SphP[ithis].HII + SphP[ithis].HeII + 2. * SphP[ithis].HeIII + SphP[ithis].H2II - SphP[ithis].HM;
    }

    ifunc = energy_solver(advance_flag, ithis, a_start, dt_in, rho_gas, T_gas_in, &T_gas_out, &t_cool, &um_coolingfct);

    if (P[ithis].ID == 1)
    {
        VERBOSE_ALL(2, "Initial cooling time in Myrs=%g\n", t_cool);
        VERBOSE_ALL(2, "Initial cooling function in cgs=%g\n", um_coolingfct);
        VERBOSE_ALL(2, "Initial cooling egyspec = %g\n", egyspec);
        VERBOSE_ALL(2, "Initial SphP.elec=%g\n", SphP[ithis].elec);
        VERBOSE_ALL(2, "T_gas_in = %g, T_gas_out = %g, T_gas = %g\n", T_gas_in, T_gas_out, T_gas);
    }

    //  if (ThisTask == 0 && ithis == 0)
    //    {
    //      VERBOSE("mode = %d ", mode);
    //    }

    if (mode == 0)
    { /* just compute the timestep and then return it */
        T_gas_in = T_gas;
        ifunc = species_solver(advance_flag = 0, ithis, a_start, dt_in, rho_gas, T_gas_in, &elec_dot);

        SphP[ithis].t_cool = t_cool * 1.0e6; /* yrs */
        SphP[ithis].t_elec = SphP[ithis].elec * base.numden / elec_dot * 1.0e6; /* yrs */

        /* energy per unit mass (u=U/m) - internal units: */

        if (ThisTask == 0 && ithis == 0)
        {
            VERBOSE(2, " - in mode=0, assign *um_energy=egyspec=%g\n", egyspec);
        }
        *um_energy = egyspec;
        return (ithis);
    }

    /*    intialize current redshift */
    a_now = a_start;

    MAX_ITERATION = 2000000000;

    if (ThisTask == 0 && ithis == 0)
    {
        VERBOSE(2, "main loop\n");
    }

    /*    MAIN LOOP: ------------------------------------------------------------------ */
    for (int i = 0; i < MAX_ITERATION; i++)
    {
        if (i > 100000)
        {
            VERBOSE_ALL(2,
                        "\n\n\n !!! Warning: on Task %d, too many iterations for particles ithis=%d, rho=%e !!!\n\n\n",
                        ThisTask, ithis, SphP[ithis].Density);
        }

        /*    get current cooling time  */
        dt_in = dt;

        ifunc = energy_solver(advance_flag = 0, ithis, a_now, dt_in, rho_gas, T_gas_in, &T_gas_out, &t_cool,
                              &um_coolingfct);

        T_gas_in = T_gas_out;
        dt_in = dt;
        /*------- Compute rate of change in electron density. */
        ifunc = species_solver(advance_flag = 0, ithis, a_now, dt_in, rho_gas, T_gas_in, &elec_dot);

        /*---------  set time step. */
        dt_min = MAX_REAL_NUMBER;

        time1 = All.Epsilon * abs(t_cool);
        time2 = All.Epsilon * abs(SphP[ithis].elec * base.numden / elec_dot);

        if (All.ComovingIntegrationOn)
        {
            da = a_end - a_now;
            hubble_a = hubble_function(a_now);
            /*
               hubble_a = All.Hubble * sqrt(All.Omega0 / (a_now * a_now * a_now)
               + (1 - All.Omega0 - All.OmegaLambda) / (a_now * a_now) +
               All.OmegaLambda);
             */

            time3 = da / (a_now * hubble_a);
            time3 *= All.UnitTime_in_s;
            time3 /= All.HubbleParam; /* in second */
            time3 /= base.time;
        }
        else
        {
            time3 = (a_end - a_now) * All.UnitTime_in_s / base.time;
        }

        /* take the minimum of the new timescales */
        if (time1 < dt_min)
        {
            dt_min = time1;
        }

        if (time2 < dt_min)
        {
            dt_min = time2;
        }

        if (time3 < dt_min)
        {
            dt_min = time3;
        }

        dt = dt_min; /* actual time step to advance */

        /*    Update temperature    */
        dt_in = dt;

        ifunc = energy_solver(advance_flag = 1, ithis, a_now, dt_in, rho_gas, T_gas_in, &T_gas_out, &t_cool,
                              &um_coolingfct);

        /*    Update species */
        T_gas_in = T_gas_out;
        dt_in = dt;
        ifunc = species_solver(advance_flag = 1, ithis, a_now, dt_in, rho_gas, T_gas_in, &elec_dot);

        /* Note. When optical depths are used, radiative rates need to be re-evaluated here */

        /* Note. When optical depths are used, radiative rates need to be re-evaluated here */
#ifdef UM_SEN_H2_SOLOMON_PROCESS
        /* J_21 aca significa que solo estoy considerando el campo de radiacion que proviene del CMB */
        /* Si agregase una power law (nu^{-ALPHA}) para el campo de radiaci'on tendr'ia que sumar este al J_nu[] del CMB
         */

        ifunc = radiative_rates(ithis, a_now, 1);
#endif // UM_SEN_H2_SOLOMON_PROCESS

        if (All.ComovingIntegrationOn)
        {
            hubble_a = hubble_function(a_now);
            /*
               hubble_a = All.Hubble * sqrt(All.Omega0 / (a_now * a_now * a_now)
               + (1 - All.Omega0 - All.OmegaLambda) / (a_now * a_now) +
               All.OmegaLambda);
             */

            /* get dt to be in the system units */
            dt = dt * base.time; /* in seconds */
            dt /= All.UnitTime_in_s; /* system units */
            dt *= All.HubbleParam; /* h correction */

            a_now = a_now + a_now * hubble_a * dt;
        }
        else
        {

            dt = dt * base.time; /* in seconds */
            dt /= All.UnitTime_in_s; /* system units */

            a_now = a_now + dt;
        }

        /* stop if time = a_end */
        if (a_now >= a_end)
        {
            break;
        }
    }

    molecular_weight = Um_Compute_MeanMolecularWeight(ithis);

    egyspec_to_temp = All.UnitPressure_in_cgs / All.UnitDensity_in_cgs * (SphP[ithis].Gamma - 1.0) / BOLTZMANN *
                      PROTONMASS * molecular_weight;

    egyspec = T_gas_out / egyspec_to_temp;

    SphP[ithis].Entropy = egyspec * GAMMA_MINUS1 / pow(SphP[ithis].Density * a3inv, GAMMA_MINUS1);
    SphP[ithis].Pressure = (SphP[ithis].Gamma - 1.0) * egyspec * SphP[ithis].Density;
    SphP[ithis].t_cool = t_cool * 1.0e6; /* yrs */
    SphP[ithis].t_elec = SphP[ithis].elec * base.numden / elec_dot * 1.0e6; /* yrs */

    /* energy per unit mass (u=U/m) in internal units: */
    *um_energy = egyspec;

    return (i);
}

/**
 * @brief Computes the cooling time of gas and solves for its thermal energy.
 *
 * This function calculates the cooling time of gas based on its temperature and other properties.
 * It also updates the gas temperature and computes the thermal energy based on various cooling and heating processes,
 * including collisional excitations, ionizations, recombinations, Compton effects, and more.
 *
 * @param solve_it_flag Flag indicating whether to solve for the thermal energy.
 * @param ithis Index of the current particle or gas element.
 * @param current_a Current scale factor.
 * @param dt Time step for the calculation.
 * @param rho_gas Density of the gas.
 * @param T_gas Current temperature of the gas.
 * @param T_gas_out Pointer to store the updated gas temperature.
 * @param t_cool Pointer to store the computed cooling time.
 * @param um_coolingfct Pointer to store the computed cooling function.
 *
 * @return Number of iterations performed in the energy calculation loop.
 */
int energy_solver(int solve_it_flag, int ithis, double current_a, double dt, double rho_gas, double T_gas,
                  double *T_gas_out, double *t_cool, double *um_coolingfct)
{
    /* ----- Collisional excitation coeffs. */
    double ceHI, ceHeI, ceHeII;

    /* ----- Collisional ionization coeffs. */
    double ciHI, ciHeI, ciHeIS, ciHeII;

    /* ----- Recombination coeffs. */
    double reHII, reHeII1, reHeII2, reHeIII;

    /* ----- Compton cooling/heating and bremstrahlung. */
    double comp1, comp2, brem;

    int i_T, N_energy_iter;
    int i_energy = 0;
    double dt_half, dt_internal, t_count, mean_nden, energy = 0, edot = 0;
    double log_T, c_T, log_T0, log_T9, T_CMB;

    double TM, LT, LDL, T3, HDLR, HDLV, HDL, cool_rate, heat_rate;

    double LogT; // used for HD-cooling conditions

    log_T0 = log(T_start);
    log_T9 = log(T_end);

    /* initialize variables */
    ceHI = ceHeI = ceHeII = 0.;
    ciHI = ciHeI = ciHeIS = ciHeII = 0.;
    reHII = reHeII1 = reHeII2 = reHeIII = 0.;
    brem = 0.;
    LogT = log10(T_gas); // used for HD-cooling conditions
    edot_radHI = edot_radHeI = edot_radHeII = edot_rad = 0.;

    /*     compute total numberdensity in units of mh
       hence base.numden * HI gives the number density of HI  */

    base.numden = Um_Compute_basenumden(rho_gas, ithis);

    N_energy_iter = 1;

    if (solve_it_flag)
    {
        N_energy_iter = 2000000000;
    }

    dt_half = dt / 2.0;
    mean_nden = Um_Compute_mean_nden(ithis);

    /* ------- Init time counter. */
    t_count = 0.0;

    for (int i_energy = 1; i_energy <= N_energy_iter; i_energy++)
    {
        /* --------- Compute thermal energy from temperature and mass. */
        energy = (1.0 / (SphP[ithis].Gamma - 1.0)) * (mean_nden * base.numden) * (BOLTZMANN * (T_gas));

        /* --------- Compute temperture index.  */
        log_T = log(T_gas);
        if (log_T < log_T0)
        {
            log_T = log_T0;
        }
        if (log_T > log_T9)
        {
            log_T = log_T9;
        }

        i_T = (log_T - log_T0) / dlog_T;
        if (i_T < 1)
        {
            i_T = 1;
        }
        if (i_T > N_T - 2)
        {
            i_T = N_T - 2;
        }

        c_T = (T_gas - T[i_T - 1]) / (T[i_T] - T[i_T - 1]);

        /* --------- Interpolate temperature dependent coefficients. */
        ceHI = ceHIa[i_T] + c_T * (ceHIa[i_T + 1] - ceHIa[i_T]);
        ceHeI = ceHeIa[i_T] + c_T * (ceHeIa[i_T + 1] - ceHeIa[i_T]);
        ceHeII = ceHeIIa[i_T] + c_T * (ceHeIIa[i_T + 1] - ceHeIIa[i_T]);

        ciHI = ciHIa[i_T] + c_T * (ciHIa[i_T + 1] - ciHIa[i_T]);
        ciHeI = ciHeIa[i_T] + c_T * (ciHeIa[i_T + 1] - ciHeIa[i_T]);
        ciHeII = ciHeIIa[i_T] + c_T * (ciHeIIa[i_T + 1] - ciHeIIa[i_T]);
        ciHeIS = ciHeISa[i_T] + c_T * (ciHeISa[i_T + 1] - ciHeISa[i_T]);

        reHII = reHIIa[i_T] + c_T * (reHIIa[i_T + 1] - reHIIa[i_T]);
        reHeII1 = reHeII1a[i_T] + c_T * (reHeII1a[i_T + 1] - reHeII1a[i_T]);
        reHeII2 = reHeII2a[i_T] + c_T * (reHeII2a[i_T + 1] - reHeII2a[i_T]);
        reHeIII = reHeIIIa[i_T] + c_T * (reHeIIIa[i_T + 1] - reHeIIIa[i_T]);

        /* ========= Compton heating ========================================================= */

        comp1 = 5.65e-36;
        comp2 = T_CMB0;

        if (All.ComovingIntegrationOn)
        {
            comp1 = comp1 * pow(current_a, -4);
            comp2 = comp2 / current_a; // this is the temperature evolution: comp2 = T_CMB0
        }
        else
        {
            comp1 = comp2 = 0;
        }
        brem = brema[i_T] + c_T * (brema[i_T + 1] - brema[i_T]);

        /* ========= Cooling Functions ======================================================= */
        /* --------- Collisional excitations */
        edot_ceHI = -ceHI * SphP[ithis].HI * SphP[ithis].elec * base.numden * base.numden;
        edot_ceHeI =
            -ceHeI * SphP[ithis].HeII * SphP[ithis].elec * SphP[ithis].elec * base.numden * base.numden * base.numden;
        edot_ceHeII = -ceHeII * SphP[ithis].HeII * SphP[ithis].elec * base.numden * base.numden;

        /* --------- Collisional ionizations */
        edot_ciHI = -ciHI * SphP[ithis].HI * SphP[ithis].elec * base.numden * base.numden;
        edot_ciHeI = -ciHeI * SphP[ithis].HeI * SphP[ithis].elec * base.numden * base.numden;
        edot_ciHeII = -ciHeII * SphP[ithis].HeII * SphP[ithis].elec * base.numden * base.numden;
        edot_ciHeIS =
            -ciHeIS * SphP[ithis].HeII * SphP[ithis].elec * SphP[ithis].elec * base.numden * base.numden * base.numden;

        /* --------- Recombinations  */
        edot_reHI = -reHII * SphP[ithis].HII * SphP[ithis].elec * base.numden * base.numden;
        edot_reHeII1 = -reHeII1 * SphP[ithis].HeII * SphP[ithis].elec * base.numden * base.numden;
        edot_reHeII2 = -reHeII2 * SphP[ithis].HeII * SphP[ithis].elec * base.numden * base.numden;
        edot_reHeIII = -reHeIII * SphP[ithis].HeIII * SphP[ithis].elec * base.numden * base.numden;

        /* --------- Compton cooling or heating */
        edot_comp = -comp1 * (T_gas - comp2) * SphP[ithis].elec * base.numden;

        /* --------- Bremsstrahlung */
        edot_brem = -brem * (SphP[ithis].HII + SphP[ithis].HeII + SphP[ithis].HeIII) * SphP[ithis].elec * base.numden *
                    base.numden;

        /* --------- Cooling from H2, Galli-Palla 1998 */
        TM = max(T_gas, 13.); /* no cooling below 13 Kelvin ... */
        TM = min(TM, 1.e5); /* fixes numerics */
        LT = log10(TM);
        /*    low density limit from Galli and Palla  */
        LDL = pow(10, -103.0 + 97.59 * LT - 48.05 * LT * LT + 10.80 * LT * LT * LT - 0.9032 * LT * LT * LT * LT);

        /* high density limit from Hollenbach Mckee 1979    */
        T3 = TM / 1000.;
        HDLR = ((9.5e-22 * pow(T3, 3.76)) / (1. + 0.12 * pow(T3, 2.1)) * exp(-pow(0.13 / T3, 3)) +
                3.e-24 * exp(-0.51 / T3));

        HDLV = (6.7e-19 * exp(-5.86 / T3) + 1.6e-18 * exp(-11.7 / T3));
        HDL = (HDLR + HDLV) / (SphP[ithis].HI * base.numden);

        /*    cooling rate per molecule [erg/s]  */
        cool_rate = SphP[ithis].HI * base.numden * HDL / (1. + (HDL / LDL));

        /* heating by CMB */
        T_CMB = T_CMB0 / current_a;
        TM = max(T_CMB, 13.);
        TM = min(TM, 1.e5);
        LT = log10(TM);
        LDL = pow(10, -103.0 + 97.59 * LT - 48.05 * LT * LT + 10.80 * LT * LT * LT - 0.9032 * LT * LT * LT * LT);
        T3 = TM / 1000.;
        HDLR = ((9.5e-22 * pow(T3, 3.76)) / (1. + 0.12 * pow(T3, 2.1)) * exp(-pow(0.13 / T3, 3)) +
                3.e-24 * exp(-0.51 / T3));
        HDLV = (6.7e-19 * exp(-5.86 / T3) + 1.6e-18 * exp(-11.7 / T3));
        HDL = (HDLR + HDLV) / (SphP[ithis].HI * base.numden);
        heat_rate = SphP[ithis].HI * base.numden * HDL / (1. + (HDL / LDL));
        edot_H2 = -SphP[ithis].H2I * base.numden * (cool_rate - heat_rate);

        /* e-H2II cooling and H-H2II cooling from Galli & Palla 1998 */
        edot_elecH2II = -3.5e-27 * exp(-800. / T_gas) * T_gas * T_gas * SphP[ithis].elec * SphP[ithis].H2II *
                        base.numden * base.numden;
        edot_HydH2II = -1.e-28 * exp(-650. / T_gas) * T_gas * T_gas * SphP[ithis].HI * SphP[ithis].H2II * base.numden *
                       base.numden;

        /* ------ HD cooling, from Lipovka, Nunez-Lopez & Avila-Reese 2005 */
        if (LogT < log10(T_start))
        {
            LogT = log10(T_start);
        }
        if (LogT > log10(T_end))
        {
            LogT = log10(T_end);
        }

        edot_HD = -pow(10., (-42.45906 + 21.90083 * LogT - 10.1954 * LogT * LogT + 2.19788 * LogT * LogT * LogT -
                             0.17286 * LogT * LogT * LogT * LogT)) *
                  SphP[ithis].HD * base.numden * SphP[ithis].HI * base.numden;

#if defined(UM_METAL_COOLING) && (defined(LT_METAL_COOLING) || defined(LT_METAL_COOLING_WAL)) &&                       \
    defined(UM_MET_IN_NONEQ_COOLING)
        /* ---------- Cooling from metals (CII, OI, SiII, FeII) ------------- */
        VERBOSE_IF(ithis == 0, 2, "Add um_metal_line_cooling: ");

#ifdef UM_e_MET_IMPACTS
        edot_met = -abs(um_metal_line_cooling(T_gas, rho_gas, LT_NMet, SphP[ithis].elec * base.numden));
#else // UM_e_MET_IMPACTS
        edot_met = -abs(um_metal_line_cooling(T_gas, rho_gas, LT_NMet));
#endif // UM_e_MET_IMPACTS

        VERBOSE_IF(ithis == 0, 2, "lambda_met = %e (edot_met in energy_solver)\n", edot_met);

#endif // defined (UM_METAL_COOLING) && (defined (LT_METAL_COOLING) || defined(LT_METAL_COOLING_WAL)) && defined
       // (UM_MET_IN_NONEQ_COOLING)

        /* ========= Heating functions ============================================================== */
        /* --------- Photoinization heating */

        edot_radHI = piHI * SphP[ithis].HI * base.numden;
        edot_radHeI = piHeI * SphP[ithis].HeI * base.numden;
        edot_radHeII = piHeII * SphP[ithis].HeII * base.numden;
        edot_rad = edot_radHI + edot_radHeI + edot_radHeII;

        // interpolated values for epsH0[erg/s], epsHe0[erg/s], epsHep[erg/s] (from cooling.c)
        // correspond to: edot_radHI, edot_radHeI, edot_rad_HeII
        // passed via variables (defined in allvars.h and allvars.c):
        //  double epsH0uvb;
        //  double epsHe0uvb;
        //  double epsHepuvb;
        //
        if (epsH0uvb > 0)
            edot_rad += epsH0uvb * SphP[ithis].HI; // multiply by fraction (eps in erg/s) - as in cooling.c}

        if (epsHe0uvb > 0)
            edot_rad += epsHe0uvb * SphP[ithis].HeI;

        if (epsHepuvb > 0)
            edot_rad += epsHepuvb * SphP[ithis].HeII;

#ifdef UM_SEN_H2_SOLOMON_PROCESS
        edot_solH2I = solH2I * SphP[ithis].H2I * base.numden;
#endif // UM_SEN_H2_SOLOMON_PROCESS

        /* ========= Compute total cooling/heating  ================================================== */

        // cooling from H, He, Bremsstrahlung
        edot = edot_ceHI + edot_ceHeI + edot_ceHeII + edot_ciHI + edot_ciHeI + edot_ciHeII + edot_ciHeIS + edot_reHI +
               edot_reHeII1 + edot_reHeII2 + edot_reHeIII + edot_brem;

        // cooling from H2
        edot += edot_H2 + edot_elecH2II + edot_HydH2II;

        // cooling from HD
        edot += edot_HD;

#if defined(UM_METAL_COOLING) && (defined(LT_METAL_COOLING) || defined(LT_METAL_COOLING_WAL)) &&                       \
    defined(UM_MET_IN_NONEQ_COOLING)
        // cooling from fine-structure transitions
        edot += edot_met;
#endif // defined (UM_METAL_COOLING) && (defined (LT_METAL_COOLING) || defined(LT_METAL_COOLING_WAL)) && defined
       // (UM_MET_IN_NONEQ_COOLING)

#ifdef UM_SEN_H2_SOLOMON_PROCESS
        edot += edot_solH2I;
#endif // UM_SEN_H2_SOLOMON_PROCESS

        // Cooling/Heating from the Compton effect
        edot += edot_comp;

        // Heating from radiation
        edot += edot_rad;

        /* --------------------- Finish here ---------------------- */
        *t_cool = (energy / edot) / base.time; /* in units of million years */

        if (solve_it_flag)
        {

            /* ----------- Compute internal time step.
               compare the cooling time,
               the time remaining,
               the given half time */

            dt_internal = abs(All.Epsilon * (*t_cool));

            if (abs(dt - t_count) < dt_internal)
                dt_internal = abs(dt - t_count);

            if (dt_half < dt_internal)
                dt_internal = dt_half;

            /* ----------- Advance gas temperature
               NOTE this update is only accurate for constant density during
               the temperature update!  */

            T_gas = T_gas + (dt_internal / (*t_cool)) * T_gas;

            if (T_gas < T[1])
                T_gas = T[1]; /* temperature boundary */

            if (T_gas > T[N_T - 2])
                T_gas = T[N_T - 2];

            t_count = t_count + dt_internal;

            if (dt - t_count < (dt / N_energy_iter))
                goto LOOP_OUT;
        }
    }

LOOP_OUT:;
    if (solve_it_flag && (i_energy >= N_energy_iter))
    {
        VERBOSE_ALL(2, "ENERGY i_energy exceeds %d for particle %d on PE %d\n", N_energy_iter, ithis, ThisTask);
        VERBOSE_ALL(2, "T_gas=%g, t_cool=%g, energy=%g, edot=%g \n", T_gas, *t_cool, energy, edot);
    }

    *T_gas_out = T_gas;
    *um_coolingfct = edot;
    return i_energy;
}

/**
 * @brief Solves the rate equations for various species in a gas medium.
 *
 * This function integrates the rate equations for different species and calculates the rate of change
 * of the electron density. It updates the number densities of hydrogen, helium, deuterium, and their ions
 * based on the provided physical conditions and rate coefficients.
 *
 * @param solve_it A flag indicating whether to perform the integration (non-zero) or directly compute
 *                 the electron density rate (zero).
 * @param ithis Index of the current particle or cell in the simulation.
 * @param a_now Current scale factor or time.
 * @param dt Time step for the integration.
 * @param rho_gas Gas density.
 * @param T_gas Gas temperature.
 * @param elec_dot Pointer to a variable where the computed rate of change of electron density will be stored.
 *
 * @return Integer status code (0 for success, non-zero for errors).
 */
int species_solver(int solve_it, int ithis, double a_now, double dt, double rho_gas, double T_gas, double *elec_dot)
{
    int i_rate = 0, N_rate_iter, i_T;
    double dt_half, dt_internal, t_count, Ccoef, Dcoef, denom;
    double HIp, HIIp, HeIp, HeIIp, HeIIIp, elecp;
    double HeHIIp;
    double HDp, DIp, DIIp;
    double log_T, c_T, log_T0, log_T9;
    double k1, k2, k3, k4, k5, k6;
    double k7, k8, k9, k10, k11, k12;
    double k14, k16, k17, k18;
    double k19, k20, k21;
    double kHD1, kHD2, kHD3, kHD4, kHD5, kHD6, kHD7, kHD8;
    double kHeHII1, kHeHII2;
    log_T0 = log(T_start);
    log_T9 = log(T_end);
    N_rate_iter = 2000000000;
    /*     compute total numberdensity in units of mh
       hence base.numden*HI gives the number density of HI   */
    base.numden = Um_Compute_basenumden(rho_gas, ithis);
    /* ----- Initialize rate coeffs. */
    k1 = k2 = k3 = k4 = k5 = k6 = k7 = k8 = k9 = k10 = k11 = k12 = k14 = k16 = k17 = k18 = k19 = k20 = k21 = 0;
    kHD1 = kHD2 = kHD3 = kHD4 = kHD5 = kHD6 = kHD7 = kHD8 = 0;
    kHeHII1 = kHeHII2 = kHeHII3 = 0;
    /* ------- Init time counter. */
    t_count = 0.0;
    /* ------- Compute temperature index. */
    log_T = log(T_gas);
    if (log_T < log_T0)
    {
        log_T = log_T0;
    }

    if (log_T > log_T9)
    {
        log_T = log_T9;
    }

    /*          i_T = MIN0(N_T-1,
       &               MAX0(1,ifIX((log_T-log_T0)/dlog_T)+1)) */

    i_T = (log_T - log_T0) / dlog_T;
    if (i_T < 1)
    {
        i_T = 1;
    }

    if (i_T > N_T - 2)
    {
        i_T = N_T - 2;
    }

    c_T = (T_gas - T[i_T]) / (T[i_T + 1] - T[i_T]);
    /* ------- Interpolate coefficients. */
    if (k1_flag)
    {
        k1 = k1a[i_T] + c_T * (k1a[i_T + 1] - k1a[i_T]);
    }
    if (k2_flag)
    {
        k2 = k2a[i_T] + c_T * (k2a[i_T + 1] - k2a[i_T]);
    }
    if (k3_flag)
    {
        k3 = k3a[i_T] + c_T * (k3a[i_T + 1] - k3a[i_T]);
    }
    if (k4_flag)
    {
        k4 = k4a[i_T] + c_T * (k4a[i_T + 1] - k4a[i_T]);
    }
    if (k5_flag)
    {
        k5 = k5a[i_T] + c_T * (k5a[i_T + 1] - k5a[i_T]);
    }
    if (k6_flag)
    {
        k6 = k6a[i_T] + c_T * (k6a[i_T + 1] - k6a[i_T]);
    }
    if (k7_flag)
    {
        k7 = k7a[i_T] + c_T * (k7a[i_T + 1] - k7a[i_T]);
    }
    if (k8_flag)
    {
        k8 = k8a[i_T] + c_T * (k8a[i_T + 1] - k8a[i_T]);
    }
    if (k9_flag)
    {
        k9 = k9a[i_T] + c_T * (k9a[i_T + 1] - k9a[i_T]);
    }
    if (k10_flag)
    {
        k10 = k10a[i_T] + c_T * (k10a[i_T + 1] - k10a[i_T]);
    }
    if (k11_flag)
    {
        k11 = k11a[i_T] + c_T * (k11a[i_T + 1] - k11a[i_T]);
    }
    if (k12_flag)
    {
        k12 = k12a[i_T] + c_T * (k12a[i_T + 1] - k12a[i_T]);
    }
    if (k14_flag)
    {
        k14 = k14a[i_T] + c_T * (k14a[i_T + 1] - k14a[i_T]);
    }
    if (k16_flag)
    {
        k16 = k16a[i_T] + c_T * (k16a[i_T + 1] - k16a[i_T]);
    }
    if (k17_flag)
    {
        k17 = k17a[i_T] + c_T * (k17a[i_T + 1] - k17a[i_T]);
    }
    if (k18_flag)
    {
        k18 = k18a[i_T] + c_T * (k18a[i_T + 1] - k18a[i_T]);
    }
    if (k19_flag)
    {
        k19 = k19a[i_T] + c_T * (k19a[i_T + 1] - k19a[i_T]);
    }
    if (k20_flag)
    {
        k20 = k20a[i_T] + c_T * (k20a[i_T + 1] - k20a[i_T]);
    }
    if (k21_flag)
    {
        k21 = k21a[i_T] + c_T * (k21a[i_T + 1] - k21a[i_T]);
    }

    kHD1 = kHD1a[i_T] + c_T * (kHD1a[i_T + 1] - kHD1a[i_T]);
    kHD2 = kHD2a[i_T] + c_T * (kHD2a[i_T + 1] - kHD2a[i_T]);
    kHD3 = kHD3a[i_T] + c_T * (kHD3a[i_T + 1] - kHD3a[i_T]);
    kHD4 = kHD4a[i_T] + c_T * (kHD4a[i_T + 1] - kHD4a[i_T]);
    kHD5 = kHD5a[i_T] + c_T * (kHD5a[i_T + 1] - kHD5a[i_T]);
    kHD6 = kHD6a[i_T] + c_T * (kHD6a[i_T + 1] - kHD6a[i_T]);
    kHD7 = kHD7a[i_T] + c_T * (kHD7a[i_T + 1] - kHD7a[i_T]);
    kHeHII1 = kHeHII1a[i_T] + c_T * (kHeHII1a[i_T + 1] - kHeHII1a[i_T]);
    kHeHII2 = kHeHII2a[i_T] + c_T * (kHeHII2a[i_T + 1] - kHeHII2a[i_T]);
    /* not kHeHII3  */
    if (!solve_it)
    {
        *elec_dot = k1 * SphP[ithis].HI * SphP[ithis].elec * base.numden * base.numden +
                    k3 * SphP[ithis].HeI * SphP[ithis].elec * base.numden * base.numden +
                    k5 * SphP[ithis].HeII * SphP[ithis].elec * base.numden * base.numden -
                    k2 * SphP[ithis].HII * SphP[ithis].elec * base.numden * base.numden -
                    k4 * SphP[ithis].HeII * SphP[ithis].elec * base.numden * base.numden -
                    k6 * SphP[ithis].HeIII * SphP[ithis].elec * base.numden * base.numden -
                    kHD7 * SphP[ithis].DII * SphP[ithis].elec * base.numden * base.numden
                    /* check base.vol is properly updated */
                    + (k24 * SphP[ithis].HI * base.numden / base.vol + k25 * SphP[ithis].HeII * base.numden / base.vol +
                       k26 * SphP[ithis].HeI * base.numden / base.vol + kHD8 * SphP[ithis].DI * base.numden / base.vol);
        goto LEAVE2;
    }

    /* ------- Integrate rate equations. */
    for (int i_rate = 1; i_rate <= N_rate_iter; i_rate++)
    {
        dt_half = dt / 2.;
        /* ------- Compute electron density rate of change. */
        *elec_dot = k1 * SphP[ithis].HI * SphP[ithis].elec * base.numden * base.numden +
                    k3 * SphP[ithis].HeI * SphP[ithis].elec * base.numden * base.numden +
                    k5 * SphP[ithis].HeII * SphP[ithis].elec * base.numden * base.numden -
                    k2 * SphP[ithis].HII * SphP[ithis].elec * base.numden * base.numden -
                    k4 * SphP[ithis].HeII * SphP[ithis].elec * base.numden * base.numden -
                    k6 * SphP[ithis].HeIII * SphP[ithis].elec * base.numden * base.numden -
                    kHD7 * SphP[ithis].DII * SphP[ithis].elec * base.numden * base.numden +
                    (k24 * SphP[ithis].HI * base.numden / base.vol + k25 * SphP[ithis].HeII * base.numden / base.vol +
                     k26 * SphP[ithis].HeI * base.numden / base.vol + kHD8 * SphP[ithis].DI * base.numden / base.vol);
        dt_internal = abs(All.Epsilon * SphP[ithis].elec / (*elec_dot) * base.numden / base.vol);
        if (dt - t_count < dt_internal)
        {
            dt_internal = dt - t_count;
        }
        if (dt_half < dt_internal)
        {
            dt_internal = dt_half;
        }

        if (H2_flag)
        {

            /* HM non-equilibrium treatment

               Ccoef = k7*SphP[ithis].HI*SphP[ithis].elec*base.kunit*base.numden*base.numden;

               Dcoef = (k8+k17)*SphP[ithis].HI*base.kunit*base.numden
               + (k18+k19)*SphP[ithis].HII*base.kunit*base.numden
               + k16*SphP[ithis].elec*base.kunit*base.numden
               + k27/base.time;


               SphP[ithis].HM = ( Ccoef*dt_internal*base.time + SphP[ithis].HM*base.numden )
               / ( 1. + Dcoef*dt_internal*base.time ) / base.numden;


               Ccoef = k9 * SphP[ithis].HI * SphP[ithis].HII*base.kunit*base.numden*base.numden
               + k12 * SphP[ithis].H2I * SphP[ithis].HII*base.kunit*base.numden*base.numden
               + k19 * SphP[ithis].HM  * SphP[ithis].HII*base.kunit*base.numden*base.numden
               +kHeHII2 * SphP[ithis].HI * SphP[ithis].HeHII*base.kunit*base.numden*base.numden

               + k29 /base.time * SphP[ithis].H2I * base.numden;

               Dcoef = (k10*SphP[ithis].HI  + k20*SphP[ithis].elec + k21*SphP[ithis].HM) * base.kunit*base.numden
               + (k28+k30)/base.time;

               SphP[ithis].H2II =  ( Ccoef*dt_internal*base.time + SphP[ithis].H2II*base.numden )
               / ( 1. + Dcoef*dt_internal*base.time ) / base.numden;

               Ccoef =  ( k8 * SphP[ithis].HM   * SphP[ithis].HI
               + k10 * SphP[ithis].H2II * SphP[ithis].HI
               + k21 * SphP[ithis].H2II * SphP[ithis].HM )*base.kunit*base.numden*base.numden
               + ( kHD3 * SphP[ithis].HD * SphP[ithis].HI
               + kHD4 * SphP[ithis].HD * SphP[ithis].HII)*base.kunit*base.numden*base.numden
               ;

               Dcoef = ( k11 * SphP[ithis].HI + k12 * SphP[ithis].HII + k14 * SphP[ithis].elec )*base.kunit*base.numden
               + (k29+k31)/base.time
               + ( kHD1 * SphP[ithis].DI + kHD2 * SphP[ithis].DII )*base.kunit*base.numden
               ;

               SphP[ithis].H2I = ( Ccoef*dt_internal*base.time + SphP[ithis].H2I*base.numden )
               / ( 1. + Dcoef*dt_internal*base.time ) / base.numden;
             */

            /* HM, H2II, equilibrium treatment */
            denom = (k8 + k17) * SphP[ithis].HI + (k18 + k19) * SphP[ithis].HII + k16 * SphP[ithis].elec +
                    k27 / (base.time * base.kunit * base.numden);
            SphP[ithis].HM = (k7 * SphP[ithis].HI * SphP[ithis].elec) / denom;
            SphP[ithis].H2II = (k9 * SphP[ithis].HI * SphP[ithis].HII + k12 * SphP[ithis].H2I * SphP[ithis].HII +
                                k19 * SphP[ithis].HM * SphP[ithis].HII + kHeHII2 * SphP[ithis].HI * SphP[ithis].HeHII +
                                k29 * SphP[ithis].H2I / (base.time * base.kunit * base.numden)) /
                               (k10 * SphP[ithis].HI + k20 * SphP[ithis].elec + k21 * SphP[ithis].HM +
                                (k28 + k30) / (base.time * base.kunit * base.numden));
            Ccoef = (k8 * SphP[ithis].HM * SphP[ithis].HI + k10 * SphP[ithis].H2II * SphP[ithis].HI +
                     k21 * SphP[ithis].H2II * SphP[ithis].HM) *
                        base.kunit * base.numden * base.numden +
                    (kHD3 * SphP[ithis].HD * SphP[ithis].HI + kHD4 * SphP[ithis].HD * SphP[ithis].HII) * base.kunit *
                        base.numden * base.numden;
            Dcoef = (k11 * SphP[ithis].HI + k12 * SphP[ithis].HII + k14 * SphP[ithis].elec) * base.kunit * base.numden +
                    (k29 + k31) / base.time +
                    (kHD1 * SphP[ithis].DI + kHD2 * SphP[ithis].DII) * base.kunit * base.numden;
            SphP[ithis].H2I = (Ccoef * dt_internal * base.time + SphP[ithis].H2I * base.numden) /
                              (1. + Dcoef * dt_internal * base.time) / base.numden;
        }

        if (H_flag)
        {
            /* ----------- HI */
            Ccoef = k2 * SphP[ithis].HII * SphP[ithis].elec * base.kunit * base.numden * base.numden +
                    (kHD1 * SphP[ithis].DI * SphP[ithis].H2I + kHD5 * SphP[ithis].HII * SphP[ithis].DI) * base.kunit *
                        base.numden * base.numden;
            Dcoef = k1 * SphP[ithis].elec * base.kunit * base.numden + k24 / base.time +
                    (kHD3 * SphP[ithis].HD + kHD6 * SphP[ithis].DII + k8 * SphP[ithis].HM + k10 * SphP[ithis].H2II) *
                        base.kunit * base.numden +
                    kHeHII2 * SphP[ithis].HeHII * base.kunit * base.numden;
            HIp = (Ccoef * dt_internal * base.time + SphP[ithis].HI * base.numden) /
                  (1. + Dcoef * dt_internal * base.time) / base.numden;
            /* ----------- HII */
            Ccoef = k1 * HIp * SphP[ithis].elec * base.kunit * base.numden * base.numden +
                    k24 / base.time * HIp * base.numden +
                    (kHD2 * SphP[ithis].DII * SphP[ithis].H2I + kHD6 * HIp * SphP[ithis].DII) * base.kunit *
                        base.numden * base.numden +
                    kHeHII3 / base.time * SphP[ithis].HeHII * base.numden;
            Dcoef = k2 * SphP[ithis].elec * base.kunit * base.numden +
                    (kHD4 * SphP[ithis].HD + kHD5 * SphP[ithis].DI) * base.kunit * base.numden +
                    kHeHII1 * SphP[ithis].HeI * base.kunit * base.numden;
            HIIp = (Ccoef * dt_internal * base.time + SphP[ithis].HII * base.numden) /
                   (1. + Dcoef * dt_internal * base.time) / base.numden;
        }

        if (He_flag)
        {
            /* ----------- HeI */
            Ccoef = k4 * SphP[ithis].HeII * SphP[ithis].elec * base.kunit * base.numden * base.numden +
                    (kHeHII2 * SphP[ithis].HeHII * HIp) * base.kunit * base.numden * base.numden +
                    kHeHII3 / base.time * SphP[ithis].HeHII * base.numden;
            Dcoef = k3 * SphP[ithis].elec * base.kunit * base.numden + k26 / base.time +
                    kHeHII1 * HIIp * base.kunit * base.numden;
            HeIp = (Ccoef * dt_internal * base.time + SphP[ithis].HeI * base.numden) /
                   (1. + Dcoef * dt_internal * base.time) / base.numden;
            /* ----------- HeII */
            Ccoef = k3 * HeIp * SphP[ithis].elec * base.kunit * base.numden * base.numden +
                    k6 * SphP[ithis].HeIII * SphP[ithis].elec * base.kunit * base.numden * base.numden +
                    k26 / base.time * HeIp * base.numden;
            Dcoef = (k4 * SphP[ithis].elec + k5 * SphP[ithis].elec) * base.kunit * base.numden + k25 / base.time;
            HeIIp = (Ccoef * dt_internal * base.time + SphP[ithis].HeII * base.numden) /
                    (1. + Dcoef * dt_internal * base.time) / base.numden;
            /* ----------- HeIII */
            Ccoef = k5 * HeIIp * SphP[ithis].elec * base.kunit * base.numden * base.numden +
                    k25 / base.time * HeIIp * base.numden;
            Dcoef = k6 * SphP[ithis].elec * base.kunit * base.numden;
            HeIIIp = (Ccoef * dt_internal * base.time + SphP[ithis].HeIII * base.numden) /
                     (1. + Dcoef * dt_internal * base.time) / base.numden;
        }

        /* ---------HD */
        Ccoef = (kHD1 * SphP[ithis].DI * SphP[ithis].H2I + kHD2 * SphP[ithis].DII * SphP[ithis].H2I) * base.kunit *
                base.numden * base.numden;
        Dcoef = (kHD3 * HIp + kHD4 * HIIp) * base.kunit * base.numden;
        HDp = (Ccoef * dt_internal * base.time + SphP[ithis].HD * base.numden) /
              (1. + Dcoef * dt_internal * base.time) / base.numden;
        /* ---------DI */
        Ccoef = (kHD3 * HDp * HIp + kHD6 * HIp * SphP[ithis].DII + kHD7 * SphP[ithis].DII * SphP[ithis].elec) *
                base.kunit * base.numden * base.numden;
        Dcoef = (kHD1 * SphP[ithis].H2I + kHD5 * HIIp) * base.kunit * base.numden + kHD8 / base.time;
        DIp = (Ccoef * dt_internal * base.time + SphP[ithis].DI * base.numden) /
              (1. + Dcoef * dt_internal * base.time) / base.numden;
        /* ---------DII */
        Ccoef = (kHD4 * HDp * HIIp + kHD5 * HIIp * DIp) * base.kunit * base.numden * base.numden +
                kHD8 / base.time * DIp * base.numden;
        Dcoef = (kHD2 * SphP[ithis].H2I + kHD6 * HIp + kHD7 * SphP[ithis].elec) * base.kunit * base.numden;
        DIIp = (Ccoef * dt_internal * base.time + SphP[ithis].DII * base.numden) /
               (1. + Dcoef * dt_internal * base.time) / base.numden;
        /////////////////////////////////////////////////////////////
        Ccoef = k1 * HIp * SphP[ithis].elec * base.kunit * base.numden * base.numden +
                k24 / base.time * HIp * base.numden +
                (kHD2 * SphP[ithis].DII * SphP[ithis].H2I + kHD6 * HIp * SphP[ithis].DII) * base.kunit * base.numden *
                    base.numden +
                kHeHII3 / base.time * SphP[ithis].HeHII * base.numden;
        Dcoef = k2 * SphP[ithis].elec * base.kunit * base.numden +
                (kHD4 * SphP[ithis].HD + kHD5 * SphP[ithis].DI) * base.kunit * base.numden +
                kHeHII1 * SphP[ithis].HeI * base.kunit * base.numden;
        HIIp = (Ccoef * dt_internal * base.time + SphP[ithis].HII * base.numden) /
               (1. + Dcoef * dt_internal * base.time) / base.numden;
        /////////////////////////////////////////////////////////////
        /* ---------HeHII */
        Ccoef = kHeHII1 * HeIp * HIIp * base.kunit * base.numden * base.numden;
        Dcoef = kHeHII2 * HIp * base.kunit * base.numden + kHeHII3 / base.time;
        HeHIIp = (Ccoef * dt_internal * base.time + SphP[ithis].HeHII * base.numden) /
                 (1. + Dcoef * dt_internal * base.time) / base.numden;
        /* --------- elec */
        Ccoef = k24 / base.time * HIp * base.numden + k25 / base.time * HeIIp * base.numden +
                k26 / base.time * HeIp * base.numden + kHD8 / base.time * DIp * base.numden;
        Dcoef = -(k1 * HIp - k2 * HIIp + k3 * HeIp - k6 * HeIIIp + k5 * HeIIp - k4 * HeIIp - kHD7 * DIIp) * base.kunit *
                base.numden;
        elecp = (Ccoef * dt_internal * base.time + SphP[ithis].elec * base.numden) /
                (1. + Dcoef * dt_internal * base.time) / base.numden;
        SphP[ithis].HI = HIp;
        SphP[ithis].HII = HIIp;
        SphP[ithis].HeI = HeIp;
        SphP[ithis].HeII = HeIIp;
        SphP[ithis].HeIII = HeIIIp;
        SphP[ithis].elec = elecp;
        SphP[ithis].HD = HDp;
        SphP[ithis].DI = DIp;
        SphP[ithis].DII = DIIp;
        SphP[ithis].HeHII = HeHIIp;
        /* fix possible numerical approximations: */
        if (SphP[ithis].HeII < 0)
        {
            SphP[ithis].HeII = 0;
        }

        if (SphP[ithis].DI < 0)
        {
            SphP[ithis].DI = 0;
        }

        if (SphP[ithis].DII < 0)
        {
            SphP[ithis].DII = 0;
        }

        /* --------- Undertake mass conservation corrections. */

#if defined(UM_METAL_COOLING) && (defined(LT_METAL_COOLING) || defined(LT_METAL_COOLING_WAL)) &&                       \
    defined(UM_MET_IN_NONEQ_COOLING) && !defined(UM_DISABLE_MET_NONEQ_CORRECTIONS)
        Um_met_corrections(ithis);
#else // defined (UM_METAL_COOLING) && (defined (LT_METAL_COOLING) || defined(LT_METAL_COOLING_WAL)) && defined
      // (UM_MET_IN_NONEQ_COOLING) && !defined(UM_DISABLE_MET_NONEQ_CORRECTIONS)
        Std_corrections(ithis);
#endif // defined (UM_METAL_COOLING) && (defined (LT_METAL_COOLING) || defined(LT_METAL_COOLING_WAL)) && defined
       // (UM_MET_IN_NONEQ_COOLING) && !defined(UM_DISABLE_MET_NONEQ_CORRECTIONS)
        t_count = t_count + dt_internal;
        if (abs(dt - t_count) < 1.0e-3 * dt)
        {
            goto LEAVE;
        }
    }

LEAVE:;
    if (i_rate >= N_rate_iter)
    {
        VERBOSE_ALL(2, "RATE i_rate exceeds %d for particle %d on PE %d\n", N_rate_iter, ithis, ThisTask);
        VERBOSE_ALL(2, "dt=%g, t_count%g \n", dt, t_count);
        VERBOSE_ALL(2, "particle mass=%g \n", P[ithis].Mass);
    }

LEAVE2:;
    return i_rate;
}

#ifndef UM_DISABLE_RADIATION
/**
 * @brief Computes radiative rate coefficients and column densities for a given radiation field.
 *
 * This function calculates column densities and radiative rate coefficients based on the
 * provided radiation field and geometry. It takes into account various factors such as
 * shielding processes, different radiation transfer schemes, and specific radiation fields.
 *
 * @param ithis Index of the current particle or entity for which the rates are computed.
 * @param a_now Scale factor of the universe at the current time.
 * @param J_21 Amplitude of the radiation field in units of 10^-21 erg/s/cm^2/Hz/sr.
 *
 * @return An integer indicating the status.
 */
int radiative_rates(int ithis, double a_now, double J_21)
{
    /* Compute column densities, radiative rate coefficients for a given
       radiation field and geometry.

       J_21:  amplitude of radiation field  */
    /*----- Declare other values. */
    double delta_nu, c0, c1;
    double tau;
    double T_eff;
    int j;
#ifdef UM_SEN_H2_SOLOMON_PROCESS
    /* Simple shielding */
    double f_shield;
#endif // UM_SEN_H2_SOLOMON_PROCESS
    /*----- Now compute column densities */
    /*----- Set units. */
    double sigma_unit = 1.0;
    double tau_unit = base.len * base.numden * sigma_unit;
#ifndef RADTRANSFER
    /*----- Now compute J_nu from column densities.  */
    /*------- Do optically thin case. */
    if (opt_thin_flag)
    {
        for (j = 0; j < N_nu; j++)
        {
            J_nu[j] = J0_nu[j];
        }
    }
    else
    {
        for (j = 0; j < N_nu; j++)
        {
            tau = 0.0;
            if (H_flag)
            {
                tau = tau + SphP[ithis].HI * sigma24[j];
            }

            if (He_flag)
            {
                tau = tau + SphP[ithis].HeI * sigma26[j] + SphP[ithis].HeII * sigma25[j];
            }

            if (H2_flag)
            {
                tau = tau + SphP[ithis].HM * sigma27[j] + SphP[ithis].H2I * (sigma29[j] + sigma31[j]) +
                      SphP[ithis].H2II * (sigma28[j] + sigma30[j]);
            }
            /*------- Compute J_nu for this frequency and radius. */
            if (tau > 1.0e-40)
            {
                J_nu[j] = J_21 * radiation_field(a_now) * J0_nu[j] * exp(-tau_unit * tau);
            }
            else
            {
                J_nu[j] = J_21 * radiation_field(a_now) * J0_nu[j];
            }
        }
    }
#else // RADTRANSFER
    rt_get_J_nu(ithis);
#endif // RADTRANSFER
    /*----- Compute radiatve coefficients. */
    k24 = k25 = k26 = k27 = k28 = k29 = k30 = k31 = piHI = piHeI = piHeII = kHeHII3 = kHD8 = tiny;
#ifdef UM_SEN_H2_SOLOMON_PROCESS
    k31sol = solH2I = tiny;
#endif // UM_SEN_H2_SOLOMON_PROCESS
    /* if no radiation then leave */
    if (!rad_flag)
    {
        goto LOOP_OUT;
    }

    /*----- Compute k values integrating over frequency. */
    for (j = 1; j < N_nu; j++)
    {
        delta_nu = (nu[j] - nu[j - 1]) * eV_to_Hz;
        c0 = base.J * J_nu[j] / (nu[j] * ELECTRONVOLT_IN_ERGS);
        c1 = base.J * J_nu[j - 1] / (nu[j - 1] * ELECTRONVOLT_IN_ERGS);
        if (k24_flag)
        {
            k24 = k24 + (0.5 * 4. * M_PI) * (c0 * sigma24[j] + c1 * sigma24[j - 1]) * delta_nu;
        }
        if (k25_flag)
        {
            k25 = k25 + (0.5 * 4. * M_PI) * (c0 * sigma25[j] + c1 * sigma25[j - 1]) * delta_nu;
        }
        if (k26_flag)
        {
            k26 = k26 + (0.5 * 4. * M_PI) * (c0 * sigma26[j] + c1 * sigma26[j - 1]) * delta_nu;
        }
        if (k27_flag)
        {
            k27 = k27 + (0.5 * 4. * M_PI) * (c0 * sigma27[j] + c1 * sigma27[j - 1]) * delta_nu;
        }
        if (k28_flag)
        {
            k28 = k28 + (0.5 * 4. * M_PI) * (c0 * sigma28[j] + c1 * sigma28[j - 1]) * delta_nu;
        }
        if (k29_flag)
        {
            k29 = k29 + (0.5 * 4. * M_PI) * (c0 * sigma29[j] + c1 * sigma29[j - 1]) * delta_nu;
        }
        if (k30_flag)
        {
            k30 = k30 + (0.5 * 4. * M_PI) * (c0 * sigma30[j] + c1 * sigma30[j - 1]) * delta_nu;
        }
        if (k31_flag)
        {
            k31 = k31 + (0.5 * 4. * M_PI) * (c0 * sigma31[j] + c1 * sigma31[j - 1]) * delta_nu;
        }
        if (nu[j - 1] >= e24)
        {
            c0 = base.J * J_nu[j] * (nu[j] - e24) / nu[j];
            c1 = base.J * J_nu[j - 1] * (nu[j - 1] - e24) / nu[j - 1];
            piHI = piHI + (0.5 * 4. * M_PI) * (c0 * sigma24[j] + c1 * sigma24[j - 1]) * delta_nu;
        }

        if (nu[j - 1] >= e26)
        {
            c0 = base.J * J_nu[j] * (nu[j] - e26) / nu[j];
            c1 = base.J * J_nu[j - 1] * (nu[j - 1] - e26) / nu[j - 1];
            piHeI = piHeI + (0.5 * 4. * M_PI) * (c0 * sigma26[j] + c1 * sigma26[j - 1]) * delta_nu;
        }

        if (nu[j - 1] >= e25)
        {
            c0 = base.J * J_nu[j] * (nu[j] - e25) / nu[j];
            c1 = base.J * J_nu[j - 1] * (nu[j - 1] - e25) / nu[j - 1];
            piHeII = piHeII + (0.5 * 4. * M_PI) * (c0 * sigma25[j] + c1 * sigma25[j - 1]) * delta_nu;
        }
    }
    kHD8 = k24; // same as HI (as in Galli&Palla98): no available data !!!
#ifdef UM_SEN_H2_SOLOMON_PROCESS
    /* Abundance rate associated to Solomon-dissociation for H2I (Abel et al. 1997) */
    /* ([J_LW]=[erg/Hz/s/sr/cm^2]) */
    k31sol = 1.1e8 * 4 * M_PI * get_JLW(a_now); /* [s^-1] */
    /* Heating associated to Solomon-dissociation rate for H2I (Abel et al. 1997) */
    /* (This should be in physical units [as e.g. piHI, etc.]: [erg/s/cm^3]) */
    /* (To compute the heating contribution only the Solomon process' rate is used [k31sol]) */
    solH2I = 6.4e-13 * SphP[ithis].H2I * base.numden * k31sol; /* [erg cm^-3 s^-1] */
#endif // UM_SEN_H2_SOLOMON_PROCESS
#ifndef UM_DISABLE_CMB
    /* LTE dissociation rate for CMB run, fit by Galli-Palla (1998) */
    T_eff = T_CMB0 / a_now;
    k28 = 1.63e7 * exp(-32400.0 / T_eff);
    kHeHII3 = 0.68 * exp(-22750 / T_eff) * pow(T_eff, 1.5);
#endif // UM_DISABLE_CMB
    k24 = k24 * base.time;
    k25 = k25 * base.time;
    k26 = k26 * base.time;
    k27 = k27 * base.time;
    k28 = k28 * base.time;
    k29 = k29 * base.time;
    k30 = k30 * base.time;
    k31 = k31 * base.time;
#ifdef UM_SEN_H2_SOLOMON_PROCESS
    k31sol = k31sol * base.time;
#endif // UM_SEN_H2_SOLOMON_PROCESS
    kHeHII3 = kHeHII3 * base.time;
    kHD8 = kHD8 * base.time;
    // Add UVB:
    // interpolated values for gJH0[s^-1], gJHe0[s^-1], gJHep[s^-1]  (from cooling.c)
    // correspond to k24, k26 and k25, respectively
    // epsH0[erg/s], epsHe0[erg/s], epsHep[erg/s] are needed for heating (edot_rad)
    // values passed via
    //   double gJH0uvb, gJHe0uvb, gJHepuvb;
    //   double epsH0uvb, epsHe0uvb, epsHepuvb;
    // (defined in allvars.h and allvars.c)
    //
    if (k24_flag > 0 && gJH0uvb > 0)
    {
        /* Compute HI shielding by using UM median approximation for
           corrections as fct of n_H (HI+HII) at all z:
         */
        /* Note: HIshielding defined in allvars.h(c) and
           initialized to 1 in begrun.c
         */
#ifdef UM_HI_SHIELD_MED /* UM z-median fit */
        HIshielding = 0.985 * pow(1. + pow((SphP[ithis].HI + SphP[ithis].HII) * base.numden / 0.006, 1.9), -2) +
                      0.015 * pow(1. + (SphP[ithis].HI + SphP[ithis].HII) * base.numden / 0.006, -0.72);
#endif // UM_HI_SHIELD_MED
        k24 += HIshielding * gJH0uvb * base.time; // units,  k24 -> gJH0uvb
    }

    if (k25_flag > 0 && gJHepuvb > 0)
    {
        k25 += gJHepuvb * base.time; // units,  k25 ->  gJHepuvb
    }
    if (k26_flag > 0 && gJHe0uvb > 0)
    {
        k26 += gJHe0uvb * base.time; // units,  k26 ->  gJHe0uvb
    }

    /*if(ThisTask == 0 && J_nu[40] > 0)
       {
       printf("Radiative rates k24=%g k25=%g k26=%g k27=%g k28=%g k29=%g k30=%g k31=%g \n",
       k24, k25, k26, k27, k28, k29, k30, k31);
       }
     */

#ifdef H2_SHIELD
    /*----- Compute rate using Bruce Draine's */
    /*----- fitting formula. */
    /*    ------- Approximate energy density by J_21 */
    /*    ------- which is at 912A. */
    U_1000A = (base.J * J_21 * radiation_field(a_now) * pow(1000. / 912., ALPHA)) / (1000. * 1.e-10);
    /*    --------- Compute average f_shield over all angles. */
    f_shield_avg = 0.0;
    x = (SphP[ithis].H2I * base.numden * base.len) / 1.e14;
    f_shield = 1.0;
    if (x >= 1.0)
    {
        f_shield = pow(x, -0.75);
    }

    k31 = 0.15 * 3.e-10 * (U_1000A / 4.e-14) * f_shield * base.time;
#endif // H2_SHIELD
/* Sebas' shielding */
#ifdef UM_SEN_H2_SOLOMON_PROCESS
    f_shield = 1.0;
    /* Compute self-shielding with two simple different prescriptions */
#ifdef UM_SEN_H2_SHIELD_I
    /* Compute rate using Bruce Draine's fitting formula. */
    x = (SphP[ithis].H2I * base.numden * base.len) / 1.e14;
    if (x >= 1.0)
    {
        f_shield = pow(x, -0.75);
    }
    k31sol *= f_shield;
#endif // UM_SEN_H2_SHIELD_I
       // #ifdef UM_SEN_H2_SHIELD_II
       //   /* Compute rate using Machacek et al. (2001) fitting formula */
       // #endif // UM_SEN_H2_SHIELD_II
    /* Summing both CMB and LW resulting rates for the reaction rate 31 */
    /* Lo puedo sumar porque todo esta en las mismas unidades */
    k31 = k31 + k31sol * f_shield;
#endif // UM_SEN_H2_SOLOMON_PROCESS
/* Closing Sebas' shielding */
#ifdef UM_LJ_H2_SHIELD
    double a3inv, rho_gas, molecular_weight, egyspec_to_temp, egyspec, T_gas, x, f_shield;
    if (All.ComovingIntegrationOn)
    {
        a3inv = 1 / (a_now * a_now * a_now);
        rho_gas = All.UnitDensity_in_cgs * SphP[ithis].Density * All.HubbleParam * All.HubbleParam *
                  a3inv; /* physical gram / cm^3 */
    }
    else
    {
        a3inv = 1;
        rho_gas = All.UnitDensity_in_cgs * SphP[ithis].Density;
    }

    /* here mean molecular weight must be taken into account */
    base.numden = Um_Compute_basenumden(rho_gas, ithis);
    //  mean_nden = Um_Compute_mean_nden(ithis);
    /* this returns SphP[].Um_MeanMolecularWeight: */
    molecular_weight = Um_Compute_MeanMolecularWeight(ithis);
    /* unit conversion */
    egyspec_to_temp = All.UnitPressure_in_cgs / All.UnitDensity_in_cgs * (SphP[ithis].Gamma - 1.0) / BOLTZMANN *
                      PROTONMASS * molecular_weight;
    egyspec = SphP[ithis].Entropy / GAMMA_MINUS1 * pow(SphP[ithis].Density * a3inv, GAMMA_MINUS1);
    T_gas = egyspec * egyspec_to_temp;
    /* Draine & Bertoldi (1996)'s corrections, along the Jeans lenght, Lj = 3.35e19 sqrt*(T/n)
       x = N2/1.e14 = n xH2 Lj / 1.e14 =  xH2 3.35e19 sqrt*(nT) / 1.e14
     */
    x = (SphP[ithis].H2I * 3.35e19 * sqrt(base.numden * T_gas)) / 1.e14;
    f_shield = 1.0;
    if (x >= 1.0)
    {
        f_shield = pow(x, -0.75);
    }

    k31 *= f_shield;
    //   if(ThisTask == 0 && ithis == 0)
    //     printf("\n\n\n :::  SphP[ithis].H2I=%g, base.numden=%g, T_gas=%g, x=%g, f_shield=%g k31=%g\n\n",
    //     SphP[ithis].H2I, base.numden, T_gas, x, f_shield, k31);
    // endrun(12345);
#endif // UM_LJ_H2_SHIELD
LOOP_OUT:;
    return j;
}

/**
 * @brief Returns the radiation field strength as a function of the scale factor.
 *
 * This function provides the intensity of the radiation field at a given scale factor.
 * Currently, it returns a constant value of 1.0, but it is designed to be flexible for
 * incorporating more complex radiation field models in the future.
 *
 * @param a Scale factor of the universe at the current time.
 *
 * @return The radiation field strength, currently a constant value of 1.0.
 */
double radiation_field(double a)
{
    /*    ----- variables  */
    double res = 1.0;
    return (res);
}
#endif // UM_DISABLE_RADIATION

/**
 * @brief Initializes chemical and physical parameters.
 *
 * This function sets up various coefficients, initializes tables for reaction rates,
 * and configures radiation-related parameters based on the compilation flags.
 * It also handles specific processes like the H2-Solomon process if the relevant
 * flags are defined.
 *
 * @return An integer status code indicating the result of the initialization.
 *         Typically, this will be the return value from the `chem_tabulate` function.
 */
int InitChem(void)
{
    int ifunc;
    /* ----- Set various coefs. */
    base.len = 1.0;
    base.time = 1.e6 * SEC_PER_YEAR;
    base.vol = base.len * base.len * base.len;
#ifndef UM_DISABLE_CMB
    base.J = 1.0;
#else // UM_DISABLE_CMB
#ifdef RADTRANSFER
    base.J = 1.0;
#else // RADTRANSFER
    base.J = 1.E-21;
#endif // RADTRANSFER
#endif // UM_DISABLE_CMB
    base.kunit = base.vol / base.time;
#ifndef UM_DISABLE_RADIATION
    /* ----- Initialize tables for reaction rates,
       ----- set up powerlaw radiation field, and frequency array  */
    ifunc = init_rad(All.Time);
#endif // UM_DISABLE_RADIATION
    /* ----- cross-sections, etc ... */
    ifunc = chem_tabulate();
    /* Ask umberto what is that ?? */
    // #ifndef UM_DISABLE_RADIATION
    // ifunc = radiative_rates(1, All.Time, 1);
    // #endif // UM_DISABLE_RADIATION
#ifdef UM_SEN_H2_SOLOMON_PROCESS
    load_JLW();
#endif // UM_SEN_H2_SOLOMON_PROCESS
    return (ifunc);
}

/* probably this part should be parallelized in the near future */
/**
 * @brief Initializes and populates chemical reaction rate tables.
 *
 * This function sets up and computes various chemical reaction rates, cross-sections, and other necessary parameters
 * over a specified temperature range. It calculates values for reaction rates, including temperature-dependent factors,
 * and stores them in arrays. The function handles different cases based on compilation flags, such as `UM_OLD_RATES`
 * and `UM_OLD_RATES_HG97`, to not account for updated rate expressions and additional calculations.
 *
 * @return Integer status code indicating the success of the tabulation process. returns 0 on success.
 */
int chem_tabulate(void)
{
    /* ----- Declare other values. */
    double log_T, log_T_eV, T_eV;
    double log_T_eV2, log_T_eV3, log_T_eV4, log_T_eV5;
    double log_T_eV6, log_T_eV7, log_T_eV8, log_T_eV9;
#ifndef UM_OLD_RATES
    double log_T2, log_T3, log_T4;
    double log_T5, log_T6, log_T7;
#endif // UM_OLD_RATES
#ifndef UM_OLD_RATES_HG97
    double den;
#endif // UM_OLD_RATES_HG97

#ifndef UM_DISABLE_RADIATION
    double dum;
#endif // UM_DISABLE_RADIATION
    /* ----- Initialize tables for reaction rates,
       ----- cross-sections, etc ... */
    /* ----- Set endpoints of T grid in K.
       ----- Might want to contract this range. */
    /* ----- Compute temperature grid. */

    dlog_T = (log(T_end) - log(T_start)) / (N_T - 1);

    for (int i = 0; i < N_T; i++)
    {
        log_T = log(T_start) + i * dlog_T;
        T[i] = exp(log_T);
    }

    /* ----- Initialize arrays. */
    for (int i = 0; i < N_T; i++)
    {
        k1a[i] = tiny;
        k2a[i] = tiny;
        k3a[i] = tiny;
        k4a[i] = tiny;
        k5a[i] = tiny;
        k6a[i] = tiny;
        k7a[i] = tiny;
        k8a[i] = tiny;
        k9a[i] = tiny;
        k10a[i] = tiny;
        k11a[i] = tiny;
        k12a[i] = tiny;
        k13a[i] = tiny;
        k14a[i] = tiny;
        k15a[i] = tiny;
        k16a[i] = tiny;
        k17a[i] = tiny;
        k18a[i] = tiny;
        k19a[i] = tiny;
        k20a[i] = tiny;
        k21a[i] = tiny;
        kHD1a[i] = tiny;
        kHD2a[i] = tiny;
        kHD3a[i] = tiny;
        kHD4a[i] = tiny;
        kHD5a[i] = tiny;
        kHD6a[i] = tiny;
        kHD7a[i] = tiny;
        kHeHII1a[i] = tiny;
        kHeHII2a[i] = tiny;
        kHeHII3a[i] = tiny;
        /* HeH++ */
        ceHIa[i] = tiny;
        ceHeIa[i] = tiny;
        ceHeIIa[i] = tiny;
        ciHIa[i] = tiny;
        ciHeIa[i] = tiny;
        ciHeISa[i] = tiny;
        ciHeIIa[i] = tiny;
        reHIIa[i] = tiny;
        reHeII1a[i] = tiny;
        reHeII2a[i] = tiny;
        reHeIIIa[i] = tiny;
        brema[i] = tiny;
    }

    /* ----- Fill in tables over the range T_start to T_end. */
    for (int i = 0; i < N_T; i++)
    {
        /* ------- Compute various values of T.  */
        log_T = log(T[i]);
#ifndef UM_OLD_RATES
        log_T2 = log_T * log_T;
        log_T3 = log_T * log_T2;
        log_T4 = log_T * log_T3;
        log_T5 = log_T * log_T4;
        log_T6 = log_T * log_T5;
        log_T7 = log_T * log_T6;
#endif // UM_OLD_RATES
        T_eV = T[i] / eV_to_K;
        log_T_eV = log(T_eV);
        log_T_eV2 = log_T_eV * log_T_eV;
        log_T_eV3 = log_T_eV * log_T_eV2;
        log_T_eV4 = log_T_eV * log_T_eV3;
        log_T_eV5 = log_T_eV * log_T_eV4;
        log_T_eV6 = log_T_eV * log_T_eV5;
        log_T_eV7 = log_T_eV * log_T_eV6;
        log_T_eV8 = log_T_eV * log_T_eV7;
        log_T_eV9 = log_T_eV * log_T_eV8;

        if (T_eV > 0.8)
        {
            k1a[i] = exp(-32.71396786375 + 13.53655609057 * log_T_eV - 5.739328757388 * log_T_eV2 +
                         1.563154982022 * log_T_eV3 - 0.2877056004391 * log_T_eV4 + 0.03482559773736999 * log_T_eV5 -
                         0.00263197617559 * log_T_eV6 + 0.0001119543953861 * log_T_eV7 - 2.039149852002e-6 * log_T_eV8);

            k3a[i] =
                exp(-44.09864886561001 + 23.91596563469 * log_T_eV - 10.75323019821 * log_T_eV2 +
                    3.058038757198 * log_T_eV3 - 0.5685118909884001 * log_T_eV4 + 0.06795391233790001 * log_T_eV5 -
                    0.005009056101857001 * log_T_eV6 + 0.0002067236157507 * log_T_eV7 - 3.649161410833e-6 * log_T_eV8);

            k4a[i] =
                1.54e-9 * (1. + 0.3 / exp(8.099328789667 / T_eV)) / (exp(40.49664394833662 / T_eV) * pow(T_eV, 1.5)) +
                3.92e-13 / pow(T_eV, 0.6353);

            k5a[i] =
                exp(-68.71040990212001 + 43.93347632635 * log_T_eV - 18.48066993568 * log_T_eV2 +
                    4.701626486759002 * log_T_eV3 - 0.7692466334492 * log_T_eV4 + 0.08113042097303 * log_T_eV5 -
                    0.005324020628287001 * log_T_eV6 + 0.0001975705312221 * log_T_eV7 - 3.165581065665e-6 * log_T_eV8);
        }
        else
        {
            k1a[i] = tiny;
            k3a[i] = tiny;
            k4a[i] = 3.92e-13 / pow(T_eV, 0.6353);
            k5a[i] = tiny;
        }

        if (T[i] > 5500.0)
        {
            k2a[i] = exp(-28.61303380689232 - 0.7241125657826851 * log_T_eV - 0.02026044731984691 * log_T_eV2 -
                         0.002380861877349834 * log_T_eV3 - 0.0003212605213188796 * log_T_eV4 -
                         0.00001421502914054107 * log_T_eV5 + 4.989108920299513e-6 * log_T_eV6 +
                         5.755614137575758e-7 * log_T_eV7 - 1.856767039775261e-8 * log_T_eV8 -
                         3.071135243196595e-9 * log_T_eV9);
        }
        else
        {
            k2a[i] = k4a[i];
        }

#ifndef UM_OLD_RATES
        k6a[i] = 2 * k2a[i] * T_eV / 4; /* Abel et al. 1997, Yoshida et al. 2006 */
#else // UM_OLD_RATES
        k6a[i] = 3.36e-10 / sqrt(T[i]) / pow(T[i] / 1.e3, 0.2) / (1 + pow(T[i] / 4.e6, 0.7));
#endif // UM_OLD_RATES
#ifndef UM_OLD_RATES
        k7a[i] = 1.4e-18 * pow(T[i], 0.928) / exp(T[i] / 16200); /* Yoshida et al. 2006, Galli & Palla 1998 */
#else // UM_OLD_RATES
        k7a[i] = 6.77e-15 * pow(T_eV, 0.8779);
#endif // UM_OLD_RATES
#ifndef UM_OLD_RATES
        k8a[i] = 4.e-9 / pow(T[i], 0.17); /* Galli & Palla 1998, Yoshida et al. 2006 */
#else // UM_OLD_RATES
        if (T_eV > 0.1)
        {
            k8a[i] = exp(-20.06913897587003 + 0.2289800603272916 * log_T_eV + 0.03599837721023835 * log_T_eV2 -
                         0.004555120027032095 * log_T_eV3 - 0.0003105115447124016 * log_T_eV4 +
                         0.0001073294010367247 * log_T_eV5 - 8.36671960467864e-6 * log_T_eV6 +
                         2.238306228891639e-7 * log_T_eV7);
        }
        else
        {
            k8a[i] = 1.43e-9;
        }
#endif // UM_OLD_RATES

#ifndef UM_OLD_RATES
        k9a[i] = pow(10., -19.38 - 1.523 * log10(T[i]) + 1.118 * log10(T[i]) * log10(T[i]) -
                              0.1269 * log10(T[i]) * log10(T[i]) *
                                  log10(T[i])); /* Galli & Palla 1998, Yoshida et al. 2006 */
#else // UM_OLD_RATES
        k9a[i] = 1.85e-23 * pow(T[i], 1.8);
        if (T[i] > 6.7e3)
        {
            k9a[i] = 5.81e-16 * pow(T[i] / 56200, -0.6657 * log10(T[i] / 56200));
        }
#endif // UM_OLD_RATES

        k10a[i] = 6.0e-10;
        if (T_eV > 0.3)
        {
            k11a[i] = 1.0670825e-10 * pow(T_eV, 2.012) /
                      (exp(4.463 / T_eV) * pow(1 + 0.2472 * T_eV, 3.512)); /* Abel et al. 1997 */
        }
        else
        {
            k11a[i] = tiny;
        }

#ifndef UM_OLD_RATES
        if (T_eV > 0.1)
#else // UM_OLD_RATES
        if (T_eV > 0.3)
#endif // UM_OLD_RATES
        {
#ifndef UM_OLD_RATES
            k12a[i] =
                exp(-21237.15 / T[i]) * (-3.3232183e-7 + 3.3735382e-7 * log_T - 1.4491368e-7 * log_T2 +
                                         3.4172805e-8 * log_T3 - 4.7813720e-9 * log_T4 + 3.9731542e-10 * log_T5 -
                                         1.8171411e-11 * log_T6 + 3.5311932e-13 * log_T7); /* Savin et al. 2004a,b */
#else // UM_OLD_RATES
            k12a[i] = exp(-24.24914687731536 + 3.400824447095291 * log_T_eV - 3.898003964650152 * log_T_eV2 +
                          2.045587822403071 * log_T_eV3 - 0.5416182856220388 * log_T_eV4 +
                          0.0841077503763412 * log_T_eV5 - 0.007879026154483455 * log_T_eV6 +
                          0.0004138398421504563 * log_T_eV7 - 9.36345888928611e-6 * log_T_eV8);
#endif // UM_OLD_RATES
        }
        else
        {
            k12a[i] = tiny;
        }

        if (T_eV > 0.3)
        {
#ifndef UM_OLD_RATES
            k14a[i] = 3.73e-9 * pow(T[i], 0.1121) / exp(99430 / T[i]); /* Stibbe & Tennyson 1999, Yoshida et al. 2006 */
#else // UM_OLD_RATES
            k14a[i] = 4.38e-10 * exp(-102000.0 / T[i]) * pow(T[i], 0.35);
#endif // UM_OLD_RATES
        }
        else
        {
            k14a[i] = tiny;
        }
        k13a[i] = 0.0;
        k15a[i] = 0.0;
        if (T_eV > 0.04)
        {
            k16a[i] =
                exp(-18.01849334273 + 2.360852208681 * log_T_eV - 0.2827443061704 * log_T_eV2 +
                    0.01623316639567 * log_T_eV3 - 0.03365012031362999 * log_T_eV4 + 0.01178329782711 * log_T_eV5 -
                    0.001656194699504 * log_T_eV6 + 0.0001068275202678 * log_T_eV7 - 2.631285809207e-6 * log_T_eV8);
        }
        else
        {
            k16a[i] = tiny;
        }

        if (T_eV > 0.1)
        {
            k17a[i] =
                exp(-20.37260896533324 + 1.139449335841631 * log_T_eV - 0.1421013521554148 * log_T_eV2 +
                    0.00846445538663 * log_T_eV3 - 0.0014327641212992 * log_T_eV4 + 0.0002012250284791 * log_T_eV5 +
                    0.0000866396324309 * log_T_eV6 - 0.00002585009680264 * log_T_eV7 + 2.4555011970392e-6 * log_T_eV8 -
                    8.06838246118e-8 * log_T_eV9);
        }
        else
        {
            k17a[i] = 2.56e-9 * pow(T_eV, 1.78186);
        }

#ifndef UM_OLD_RATES
        k18a[i] = 6.3e-8 + 5.7e-6 / sqrt(T[i]) - 9.2e-11 * sqrt(T[i]) + 4.4e-13 * T[i];
        /* Peterson et al. 1971, Galli & Palla 1998, Yoshida et al. 2006 */
        if (k18a[i] < 0)
        {
            k18a[i] = 0;
        }
#else // UM_OLD_RATES
        k18a[i] = 6.5e-9 / sqrt(T_eV);
#endif // UM_OLD_RATES

        k19a[i] = 1.0e-8 * pow(T[i], -0.4);

        if (T[i] > 1.0e4)
        {
            k19a[i] = 4.0e-4 * pow(T[i], -1.4) * exp(-15100.0 / T[i]);
        }

#ifndef UM_OLD_RATES
        k20a[i] = 2.e-7 / sqrt(T[i]); /* Galli & Palla 1998, Yoshida et al. 2006 */
#else // UM_OLD_RATES
        k20a[i] = 5.56396e-8 / pow(T_eV, 0.6035);
#endif // UM_OLD_RATES
        k21a[i] = 4.64e-8 / sqrt(T_eV);
        /* initializes array elements: already done... */
        // kHD1a[i] = kHD2a[i] = kHD3a[i] = kHD4a[i] = kHD5a[i] = kHD6a[i] = kHD7a[i] = kHD8a[i] = 0.;
        /* Wang & Stancil 2002 */
        /* DI + H2I -> HD + HI */
        kHD1a[i] = 9.0e-11 * exp(-3876 / T[i]);
        /* Wang & Stancil 2002 */
        /* DII + H2I -> HD + HII */
        kHD2a[i] = 1.6e-9;
        /* Stancil, Lepp & Dalgarno 1998 */
        /* HD + HI -> DI + H2 */
        kHD3a[i] = 3.2e-11 * exp(-3624 / T[i]);
        /* Stancil, Lepp & Dalgarno 1998 */
        /* HD + HII -> DII + H2 */
        kHD4a[i] = 1.e-9 * exp(-464 / T[i]);
        /* Savin 2002  */
        /* HII + DI -> HI + DII */
        kHD5a[i] = 2.00e-10 * pow(T[i], 0.402) * exp(-37.1 / T[i]) - 3.31e-17 * pow(T[i], 1.48);
        if (kHD5a[i] < 0)
        {
            kHD5a[i] = 0;
        }
        if (T[i] < 1)
        {
            kHD5a[i] = 0;
        }

        /* Savin 2002  */
        /* HI + DII -> HII + DI */
        kHD6a[i] = 2.06e-10 * pow(T[i], 0.396) * exp(-33.0 / T[i]) + 2.03e-9 * pow(T[i], -0.332);
        // kHD6a[i]= 3.7e-10 * pow(T[i],0.28) * exp(-43/T[i]); old Galli&Palla98
        /* Galli&Palla98 */
        /* DII   + e   ->   DI + p     D recombination */
        kHD7a[i] = k2a[i]; // same as HI (Galli&  Palla, 1998)
        /* for HeH++ */

        if (T[i] > 1000.)
        {
            kHeHII1a[i] = 3.45e-16 * pow(T[i], -1.06);
        }
        else
        {
            kHeHII1a[i] = 7.6e-18 * pow(T[i], -0.5);
        }

        kHeHII2a[i] = 9.1e-10;
        /* not kHeHII3a */
    }

    /* ----- Convert to base.kunits. */
    for (int i = 0; i < N_T; i++)
    {
        k1a[i] = k1a[i] / base.kunit;
        k2a[i] = k2a[i] / base.kunit;
        k3a[i] = k3a[i] / base.kunit;
        k4a[i] = k4a[i] / base.kunit;
        k5a[i] = k5a[i] / base.kunit;
        k6a[i] = k6a[i] / base.kunit;
        k7a[i] = k7a[i] / base.kunit;
        k8a[i] = k8a[i] / base.kunit;
        k9a[i] = k9a[i] / base.kunit;
        k10a[i] = k10a[i] / base.kunit;
        k11a[i] = k11a[i] / base.kunit;
        k12a[i] = k12a[i] / base.kunit;
        k13a[i] = k13a[i] / base.kunit;
        k14a[i] = k14a[i] / base.kunit;
        k15a[i] = k15a[i] / base.kunit;
        k16a[i] = k16a[i] / base.kunit;
        k17a[i] = k17a[i] / base.kunit;
        k18a[i] = k18a[i] / base.kunit;
        k19a[i] = k19a[i] / base.kunit;
        k20a[i] = k20a[i] / base.kunit;
        k21a[i] = k21a[i] / base.kunit;
        kHD1a[i] = kHD1a[i] / base.kunit;
        kHD2a[i] = kHD2a[i] / base.kunit;
        kHD3a[i] = kHD3a[i] / base.kunit;
        kHD4a[i] = kHD4a[i] / base.kunit;
        kHD5a[i] = kHD5a[i] / base.kunit;
        kHD6a[i] = kHD6a[i] / base.kunit;
        kHD7a[i] = kHD7a[i] / base.kunit;
        /* not kHD8 */
        kHeHII1a[i] = kHeHII1a[i] / base.kunit;
        kHeHII2a[i] = kHeHII2a[i] / base.kunit;
        /* not kHeHII3a */
    }

    int k1b = 0, k2b = 0, k3b = 0, k4b = 0, k5b = 0, k6b = 0;
    for (int i = 0; i < N_T; i++)
    {
        k1b += k1a[i];
        k2b += k2a[i];
        k3b += k3a[i];
        k4b += k4a[i];
        k5b += k5a[i];
        k6b += k6a[i];
    }

    int i = 0;
    for (i = 0; i < N_T; i++)
    {
        /* ------- Collisional excitations (Black 1981; Cen 1992) */
        ceHIa[i] = 7.5e-19 * exp(-118348. / T[i]) / (1. + sqrt(T[i] / 1.e5));
        ceHeIa[i] = 9.1e-27 * exp(-13179. / T[i]) * pow(T[i], -0.1687) / (1. + sqrt(T[i] / 1.e5));
        ceHeIIa[i] = 5.54e-17 * exp(-473638. / T[i]) * pow(T[i], -0.397) / (1. + sqrt(T[i] / 1.e5));
        ciHeISa[i] = 5.01e-27 * pow(T[i], -0.1687) / (1. + sqrt(T[i] / 1.e5)) * exp(-55338. / T[i]);
        /* ------- Collisional ionizations (Tom's polynomial fits) */
        ciHIa[i] = 2.18e-11 * k1a[i] * base.kunit;
        ciHeIa[i] = 3.94e-11 * k3a[i] * base.kunit;
        ciHeIIa[i] = 8.72e-11 * k5a[i] * base.kunit;
        /* ------- Recombinations */
#ifndef UM_OLD_RATES
#ifndef UM_OLD_RATES_HG97
        /* case B recombination cooling rates - Hui&Gnedin1997 */
        den = pow(315614 / T[i] / 2.250, 0.376);
        den = pow(1. + den, 3.720);
        reHIIa[i] = 3.435e-30 * T[i] * pow(315614 / T[i], 1.970) / den;
#else // UM_OLD_RATES_HG97
        reHIIa[i] = 6.5e-27 * sqrt(T[i]) * pow(T[i] / 1000.0, -0.2) / (1.0 + pow(T[i] / 1.e6, 0.7));
#endif // UM_OLD_RATES_HG97
#else // UM_OLD_RATES
        /* (Cen 1992) */
        reHIIa[i] = 8.70e-27 * sqrt(T[i]) * pow(T[i] / 1000.0, -0.2) / (1.0 + pow(T[i] / 1.e6, 0.7));
#endif // UM_OLD_RATES
#ifndef UM_OLD_RATES_HG97
        /* case B recombination cooling rates - Hui&Gnedin1997 */
        reHeII1a[i] = 1.73958e-30 * T[i] * pow(570670 / T[i], 0.750);
#else // UM_OLD_RATES_HG97
        reHeII1a[i] = 1.55e-26 * pow(T[i], 0.3647);
#endif // UM_OLD_RATES_HG97
        /* dielectric recombination */
        reHeII2a[i] = 1.24e-13 * pow(T[i], -1.5) * exp(-470000 / T[i]) * (1. + 0.3 * exp(-94000. / T[i]));
#ifndef UM_OLD_RATES_HG97
        /* case B recombination cooling rates - Hui&Gnedin1997 */
        den = pow(1263030 / T[i] / 2.250, 0.376);
        den = pow(1. + den, 3.720);
        reHeIIIa[i] = 8 * 3.435e-30 * T[i] * pow(1263030 / T[i], 1.970) / den;
#else // UM_OLD_RATES_HG97
        reHeIIIa[i] = 3.48e-26 * sqrt(T[i]) * pow(T[i] / 1000.0, -0.2) / (1.0 + pow(T[i] / 4.e6, 0.7));
#endif // UM_OLD_RATES_HG97
        /* ------- Bremsstrahlung (Black 1981; Spitzer & Hart 1979) */
        brema[i] = 1.43e-27 * sqrt(T[i]) * (1.1 + 0.34 * exp(-pow(5.5 - log10(T[i]), 2.0) / 3.0));
    }

#ifndef UM_DISABLE_RADIATION
    /* ----- Compute cross-sections. */
    for (i = 0; i < N_nu; i++)
    {

        /* photoionisation heating, HI */
        if (nu[i] > e24)
        {
            dum = sqrt(nu[i] / e24 - 1.0);
            sigma24[i] =
                6.3e-18 * pow(e24 / nu[i], 4) * exp(4.0 - 4.0 * atan(dum) / dum) / (1 - exp(-2.0 * M_PI / dum));
        }
        else
        {
            sigma24[i] = 0.0;
        }

        /* photoionisation heating, HeII */
        if (nu[i] > e25)
        {
            dum = sqrt(nu[i] / e25 - 1.0);
            sigma25[i] =
                1.58e-18 * pow(e25 / nu[i], 4) * exp(4.0 - 4.0 * atan(dum) / dum) / (1 - exp(-2.0 * M_PI / dum));
        }
        else
        {
            sigma25[i] = 0.0;
        }

        /* photoionisation heating HeI */
        if (nu[i] > e26)
        {
            sigma26[i] = 7.42e-18 * (1.66 * pow(nu[i] / e26, -2.05) - 0.66 * pow(nu[i] / e26, -3.05));
        }
        else
        {
            sigma26[i] = 0.0;
        }
        /* photodissociation heating, HM */
        if (nu[i] > e27)
        {
            sigma27[i] = 2.11e-16 * pow(nu[i] - e27, 1.5) / pow(nu[i], 3);
        }
        else
        {
            sigma27[i] = 0.0;
        }
        /* photodissociation heating, H2II */
        if (nu[i] > e28a && nu[i] <= e28b)
        {
            sigma28[i] = pow(10, -40.97 + 6.03 * nu[i] - 0.504 * nu[i] * nu[i] + 1.387e-2 * nu[i] * nu[i] * nu[i]);
        }
        else if (nu[i] > e28b && nu[i] < e28c)
        {
            sigma28[i] = pow(10, -30.26 + 2.79 * nu[i] - 0.184 * nu[i] * nu[i] + 3.535e-3 * nu[i] * nu[i] * nu[i]);
        }
        else
        {
            sigma28[i] = 0.0;
        }
        /* photodissociation heating, H2I */
        if (nu[i] > e29a && nu[i] <= e29b)
        {
            sigma29[i] = 6.2e-18 * nu[i] - 9.4e-17;
        }
        else if (nu[i] > e29b && nu[i] <= e29c)
        {
            sigma29[i] = 1.4e-18 * nu[i] - 1.48e-17;
        }
        else if (nu[i] > e29c)
        {
            sigma29[i] = 2.5e-14 * pow(nu[i], -2.71);
        }
        else
        {
            sigma29[i] = 0.0;
        }

        /* photodissociation heating, H2II */
        if (nu[i] >= e30a && nu[i] < e30b)
        {
            sigma30[i] =
                pow(10.0, -16.926 - 4.528e-2 * nu[i] + 2.238e-4 * nu[i] * nu[i] + 4.245e-7 * nu[i] * nu[i] * nu[i]);
        }
        else
        {
            sigma30[i] = 0.0;
        }

        /*  photodissociation heating, H2I */
        if (nu[i] > e28b && nu[i] < e24)
            sigma31[i] = 3.71e-18;
        else
            sigma31[i] = 0.0;
    }
#endif // UM_DISABLE_RADIATION

    return (i);
}

#ifndef UM_DISABLE_RADIATION
/**
 * @brief Initializes the frequency grid and radiation field.
 *
 * This function sets up the frequency grid for radiation and computes the intrinsic radiation
 * field values. It also initializes boundary points and sorts the frequency grid. If the
 * Cosmic Microwave Background (CMB) option is enabled, it computes the CMB thermal radiation
 * field. Finally, it writes the computed spectrum to a file if certain conditions are met.
 *
 * @param atime The current cosmological scale factor.
 *
 * @return An integer value representing the number of frequencies initialized. In this implementation,
 *         the return value is the last index of the frequency array after initialization and sorting.
 */
int init_rad(double atime)
{
    /* ----- Initialize nu grid  and radiation field.  */

    /* ----- Declare other values.  */
    double log_nu, eps, nu_bound[N_nu_bound];
    char spec_name[200];
    FILE *fd;
#ifndef UM_DISABLE_CMB
    double T_eff, hc;
#endif // UM_DISABLE_CMB
    /* ----- Compute base nu grid (in eV) and intrinsic J0_nu.  */
    /*        base.J = 1.e-21  ! now in cloud9.f  */
    dlog_nu = (log(nu_end) - log(nu_start)) / (N_nu_base - 1);
    for (int i = 0; i < N_nu_base; i++)
    {
        log_nu = log(nu_start) + i * dlog_nu;
        nu[i] = exp(log_nu);
    }

    /* ----- Set transition points.  */
    nu_bound[0] = e24;
    nu_bound[1] = e25;
    nu_bound[2] = e26;
    nu_bound[3] = e27;
    nu_bound[4] = e28a;
    nu_bound[5] = e28b;
    nu_bound[6] = e28c;
    nu_bound[7] = e29a;
    nu_bound[8] = e29b;
    nu_bound[9] = e29c;
    nu_bound[10] = e30a;
    nu_bound[11] = e30b;
    /* ----- Append boundary points to nu.  */
    eps = 1.e-5;
    for (int i = 0; i < N_nu_bound; i++)
    {
        int j = i * N_pts_bound + N_nu_base - 1;
        nu[j + 1] = nu_bound[i] * (1. - 4. * eps);
        nu[j + 2] = nu_bound[i] * (1. - 2. * eps);
        nu[j + 3] = nu_bound[i] * (1. - 1. * eps);
        nu[j + 4] = nu_bound[i];
        nu[j + 5] = nu_bound[i] * (1. + 1. * eps);
        nu[j + 6] = nu_bound[i] * (1. + 2. * eps);
        nu[j + 7] = nu_bound[i] * (1. + 4. * eps);
    }

    /* ----- Sort nu.  */
    (void)heap_sort(N_nu, nu - 1);
#ifndef UM_DISABLE_CMB
    /* CMB thermal radiation */
    T_eff = T_CMB0 / atime;
    hc = C_LIGHT * PLANCK;
    for (int i = 0; i < N_nu; i++)
    {
        if (nu[i] == 0)
        {
            J0_nu[i] = 0;
        }
        else
        {
            J0_nu[i] = 2.0 * pow(nu[i] * ELECTRONVOLT_IN_ERGS, 3) / (hc * hc) /
                       (exp(nu[i] * ELECTRONVOLT_IN_ERGS / (BOLTZMANN * T_eff)) - 1.0);
            if (J0_nu[i] < 1.0e-60)
            {
                J0_nu[i] = 0.0;
            }
        }
    }
#endif // UM_DISABLE_CMB

    int i = 0;
    if (ThisTask == 0 && All.NumCurrentTiStep == 0)
    {
        sprintf(spec_name, "spectrum.J0.%4.4d", All.NumCurrentTiStep);
        fd = fopen(spec_name, "w");
        for (i = 0; i < N_nu; i++)
        {
            fprintf(fd, "%g %g \n", nu[i], J0_nu[i]);
        }
        fclose(fd);
    }

    return (i);
}

/**
 * @brief Performs a sort on an array of doubles.
 *
 * This function sorts an array of doubles in ascending order using the heap sort algorithm.
 * Heap sort first builds a max-heap from the array and then repeatedly extracts the maximum
 * element from the heap and places it at the end of the sorted portion of the array.
 *
 * @param n The number of elements in the array to be sorted.
 * @param ra Array of doubles to be sorted. The array is sorted in-place.
 *
 * @return An integer value indicating the success of the operation.
 *         In this implementation, the function always returns 1.
 */
int heap_sort(unsigned long n, double ra[])
{
    int ifunc = 1;
    unsigned long i, ir, j, l;
    double rra;
    if (n < 2)
    {
        return n;
    }
    l = (n >> 1) + 1;
    ir = n;
    for (;;)
    {
        if (l > 1)
        {
            rra = ra[--l];
        }
        else
        {
            rra = ra[ir];
            ra[ir] = ra[1];
            if (--ir == 1)
            {
                ra[1] = rra;
                break;
            }
        }
        i = l;
        j = l + l;
        while (j <= ir)
        {
            if (j < ir && ra[j] < ra[j + 1])
            {
                j++;
            }
            if (rra < ra[j])
            {
                ra[i] = ra[j];
                i = j;
                j <<= 1;
            }
            else
                j = ir + 1;
        }
        ra[i] = rra;
    }

    return (ifunc);
}
#endif // UM_DISABLE_RADIATION

/* add some routines (by UM) */
/* -----  ( declarations are in the header and in proto.hpp )  ----- */
#if defined(UM_METAL_COOLING) && (defined(LT_METAL_COOLING) || defined(LT_METAL_COOLING_WAL)) &&                       \
    defined(UM_MET_IN_NONEQ_COOLING)
/**
 * @brief Computes the inverse density factor for metal calculations.
 *
 * This function calculates the inverse of the density of the gas by considering the
 * contributions of various metals and hydrogen. It is used for metal-related calculations
 * where the density needs to be inversely scaled based on the metal content.
 *
 * @param i Index of the particle or cell for which the inverse density is computed.
 *
 * @return The inverse density factor.
 */
double INLINE_FUNC Um_Compute_invden(int i)
{
    /* useful for the calculations with metals ... */

    um_mass = P[i].Mass;
    um_ZsPoint = SphP[i].Metals;
    double invmass = 1.0 / (double)um_mass;
    double sum_Zs = 0;
    for (int k = 0; k < LT_NMet; k++)
    {
        sum_Zs += um_ZsPoint[k];
    }

    double Hmass = (double)um_mass - sum_Zs;
    double denom = (Hmass * invmass) + (um_ZsPoint[Hel] * invmass) / 4;
    if (Carbon >= 0)
    {
        denom += (um_ZsPoint[Carbon] * invmass) / 12;
    }
    if (Oxygen >= 0)
    {
        denom += (um_ZsPoint[Oxygen] * invmass) / 16;
    }
    if (Silicon >= 0)
    {
        denom += (um_ZsPoint[Silicon] * invmass) / 28;
    }
    if (Iron >= 0)
    {
        denom += (um_ZsPoint[Iron] * invmass) / 56;
    }
    if (Nitrogen >= 0)
    {
        denom += (um_ZsPoint[Nitrogen] * invmass) / 14;
    }
    if (Magnesium >= 0)
    {
        denom += (um_ZsPoint[Magnesium] * invmass) / 24;
    }

    denom += (SphP[i].Metals[FillEl] * invmass) / dummy_average_atomic_mass;
    /* Metals[FillEl] accounts for the dummy elements */
    double invden = 1.0 / denom;
    return invden;
}
#endif // defined (UM_METAL_COOLING) && (defined (LT_METAL_COOLING) || defined(LT_METAL_COOLING_WAL)) && defined
       // (UM_MET_IN_NONEQ_COOLING)

#if defined(UM_METAL_COOLING) && (defined(LT_METAL_COOLING) || defined(LT_METAL_COOLING_WAL)) &&                       \
    defined(UM_MET_IN_NONEQ_COOLING) && !defined(UM_DISABLE_MET_NONEQ_CORRECTIONS)
/**
 * @brief Applies mass conservation corrections and updates species abundances.
 *
 * This function adjusts the abundances of various species to ensure mass conservation
 * and corrects the number fractions of hydrogen and helium based on the total mass and
 * metal content. It also updates the electron number density based on the corrected abundances.
 *
 * @param i Index of the particle or cell for which the corrections are applied.
 */
void Um_met_corrections(int i)
{
    /* --------- Undertake mass conservation corrections. */
    SphP[i].elec = 0.;
    um_mass = P[i].Mass;
    um_ZsPoint = SphP[i].Metals;
    double sum_Zs = 0;
    for (int k = 0; k < LT_NMet; k++)
    {
        sum_Zs += um_ZsPoint[k];
    }

    Hmass = (double)um_mass - sum_Zs;
    double invmass = 1.0 / (double)um_mass;
    double invden = Um_Compute_invden(i);
    double H_nf, He_nf;
    /* to convert mass fractions in number fractions:
       number_fr = mass_fr / atomic_mass_el * invden */
    if (Hmass > 0 && um_ZsPoint[Hel] > 0)
    {
        H_nf = (Hmass * invmass) * invden;
        He_nf = (um_ZsPoint[Hel] * invmass) / 4. * invden;
        if (Carbon >= 0)
        {
            SphP[i].elec += (um_ZsPoint[Carbon] * invmass) / 12 * invden;
        }
        // -> assumed ionized once

        if (Silicon >= 0)
        {
            SphP[i].elec += (um_ZsPoint[Silicon] * invmass) / 28 * invden;
        }
        // -> assumed ionized once

        if (Iron >= 0)
        {
            SphP[i].elec += (um_ZsPoint[Iron] * invmass) / 56 * invden;
        }
        // -> assumed ionized once
    }
    else
    {
        /* primordial number fractions: */
        H_nf = 0.9268;
        He_nf = 0.0731;
    }

    //  correction = H_nf / (SphP[i].HI + SphP[i].HII + SphP[i].HM + 2. * SphP[i].H2I + 2. * SphP[i].H2II);
    double correction = H_nf / (SphP[i].HI + SphP[i].HII + SphP[i].HM + 2. * SphP[i].H2I + 2. * SphP[i].H2II +
                                2. * SphP[i].DI + 2. * SphP[i].DII + 3. * SphP[i].HD);
    SphP[i].elec = 0.;
    if (H_flag)
    {
        SphP[i].HI *= correction;
        SphP[i].HII *= correction;
        SphP[i].elec += SphP[i].HII;
    }
    if (H2_flag)
    {
        SphP[i].HM *= correction;
        SphP[i].H2I *= correction;
        SphP[i].H2II *= correction;
        SphP[i].elec = SphP[i].elec - SphP[i].HM + SphP[i].H2II;
    }

    if (D_flag)
    {
        SphP[i].HD *= correction;
        SphP[i].DI *= correction;
        SphP[i].DII *= correction;
        SphP[i].elec = SphP[i].elec + SphP[i].DII;
    }

    if (He_flag)
    {
        correction = He_nf / (SphP[i].HeI + SphP[i].HeII + SphP[i].HeIII);
        SphP[i].HeI *= correction;
        SphP[i].HeII *= correction;
        SphP[i].HeIII *= correction;
        SphP[i].elec = SphP[i].elec + SphP[i].HeII + 2. * SphP[i].HeIII;
    }
    SphP[i].elec += SphP[i].HeHII;
    SphP[i].elec += +(um_ZsPoint[Carbon] * invmass) / 12 * invden // assumed ionized once
                    + (um_ZsPoint[Silicon] * invmass) / 28 * invden // assumed ionized once
                    + (um_ZsPoint[Iron] * invmass) / 56 * invden // assumed ionized once
        ;
    return;
}

#else // defined (UM_METAL_COOLING) && (defined (LT_METAL_COOLING) || defined(LT_METAL_COOLING_WAL)) && defined
      // (UM_MET_IN_NONEQ_COOLING) && !defined(UM_DISABLE_MET_NONEQ_CORRECTIONS)

/**
 * @brief Applies standard corrections to the abundances of various species.
 *
 * This function adjusts the abundances of hydrogen, helium, and deuterium species based on
 * specified correction factors. The corrections are applied to ensure that the total abundance
 * fractions are consistent with predefined number fractions.
 *
 * @param i Index of the particle or cell for which the corrections are applied.
 */
void Std_corrections(int i)
{
    // correction =
    // H_number_fraction / (SphP[i].HI + SphP[i].HII + SphP[i].HM + 2. * SphP[i].H2I + 2. * SphP[i].H2II);
    double correction = H_number_fraction / (SphP[i].HI + SphP[i].HII + SphP[i].HM + 2. * SphP[i].H2I +
                                             2. * SphP[i].H2II + 2. * SphP[i].DI + 2. * SphP[i].DII + 3. * SphP[i].HD);
    SphP[i].elec = 0;
    if (H_flag)
    {
        SphP[i].HI *= correction;
        SphP[i].HII *= correction;
        SphP[i].elec += SphP[i].HII;
    }
    if (H2_flag)
    {
        SphP[i].HM *= correction;
        SphP[i].H2I *= correction;
        SphP[i].H2II *= correction;
        SphP[i].elec = SphP[i].elec - SphP[i].HM + SphP[i].H2II;
    }

    if (D_flag)
    {
        SphP[i].HD *= correction;
        SphP[i].DI *= correction;
        SphP[i].DII *= correction;
        SphP[i].elec = SphP[i].elec + SphP[i].DII;
    }

    if (He_flag)
    {
        correction = He_number_fraction / (SphP[i].HeI + SphP[i].HeII + SphP[i].HeIII);
        SphP[i].HeI *= correction;
        SphP[i].HeII *= correction;
        SphP[i].HeIII *= correction;
        SphP[i].elec = SphP[i].elec + SphP[i].HeII + 2. * SphP[i].HeIII;
    }
    return;
}
#endif // defined (UM_METAL_COOLING) && (defined (LT_METAL_COOLING) || defined(LT_METAL_COOLING_WAL)) && defined
       // (UM_MET_IN_NONEQ_COOLING) && !defined(UM_DISABLE_MET_NONEQ_CORRECTIONS)

/**
 * @brief Computes the base number density of nuclei.
 *
 * This function calculates the number density of nuclei based on the gas density and composition,
 * considering contributions from various species and metals.
 *
 * @param rho_cgs Gas density in cgs units.
 * @param i Index of the particle or cell for which the computation is performed.
 *
 * @return The base number density of nuclei.
 */
double INLINE_FUNC Um_Compute_basenumden(double rho_cgs, int i)
{
    double denom = 4 * SphP[i].HeI + SphP[i].HII + 4 * SphP[i].HeIII + SphP[i].HI + 4 * SphP[i].HeII + SphP[i].HM +
                   2 * SphP[i].H2II + 2 * SphP[i].H2I;
    denom += 5. * SphP[i].HeHII;
    denom += 3. * SphP[i].HD + 2. * SphP[i].DI + 2. * SphP[i].DII;
#if defined(UM_METAL_COOLING) && (defined(LT_METAL_COOLING) || defined(LT_METAL_COOLING_WAL)) &&                       \
    defined(UM_MET_IN_NONEQ_COOLING)
    um_mass = P[i].Mass;
    um_ZsPoint = SphP[i].Metals;
    double invmass = 1.0 / (double)um_mass;
    double invden = Um_Compute_invden(i);
    if (Carbon >= 0)
    {
        denom += (um_ZsPoint[Carbon] * invmass) * invden;
    }
    if (Oxygen >= 0)
    {
        denom += (um_ZsPoint[Oxygen] * invmass) * invden;
    }
    if (Silicon >= 0)
    {
        denom += (um_ZsPoint[Silicon] * invmass) * invden;
    }
    if (Iron >= 0)
    {
        denom += (um_ZsPoint[Iron] * invmass) * invden;
    }
    if (Nitrogen >= 0)
    {
        denom += (um_ZsPoint[Nitrogen] * invmass) * invden;
    }
    if (Magnesium >= 0)
    {
        denom += (um_ZsPoint[Magnesium] * invmass) * invden;
    }
    denom += (SphP[i].Metals[FillEl] * invmass) * invden;
    /* Metals[FillEl] accounts for the dummy elements */
#endif // defined (UM_METAL_COOLING) && (defined (LT_METAL_COOLING) || defined(LT_METAL_COOLING_WAL))  && defined
       // (UM_MET_IN_NONEQ_COOLING)
    base.numden = rho_cgs / PROTONMASS / denom;
    return base.numden;
}

/**
 * @brief Computes the sum of nuclei in the gas.
 *
 * This function calculates the total number of nuclei in the gas by summing up contributions from
 * various species including hydrogen, helium, and metals.
 *
 * @param i Index of the particle or cell for which the computation is performed.
 *
 * @return The sum of nuclei in the gas.
 */
double INLINE_FUNC Um_Compute_sum_nuclei(int i)
{
    double sum_nuclei = 4 * SphP[i].HeI + SphP[i].HII + 4 * SphP[i].HeIII + SphP[i].HI + 4 * SphP[i].HeII + SphP[i].HM +
                        2 * SphP[i].H2II + 2 * SphP[i].H2I;
    sum_nuclei += 5. * SphP[i].HeHII;
    sum_nuclei += 3. * SphP[i].HD + 2. * SphP[i].DI + 2. * SphP[i].DII;
#if defined(UM_METAL_COOLING) && (defined(LT_METAL_COOLING) || defined(LT_METAL_COOLING_WAL)) &&                       \
    defined(UM_MET_IN_NONEQ_COOLING)
    um_mass = P[i].Mass;
    um_ZsPoint = SphP[i].Metals;
    double invmass = 1.0 / (double)um_mass;
    double invden = Um_Compute_invden(i);
    if (Carbon >= 0)
    {
        sum_nuclei += (um_ZsPoint[Carbon] * invmass) * invden;
    }
    if (Oxygen >= 0)
    {
        sum_nuclei += (um_ZsPoint[Oxygen] * invmass) * invden;
    }
    if (Silicon >= 0)
    {
        sum_nuclei += (um_ZsPoint[Silicon] * invmass) * invden;
    }
    if (Iron >= 0)
    {
        sum_nuclei += (um_ZsPoint[Iron] * invmass) * invden;
    }
    if (Nitrogen >= 0)
    {
        sum_nuclei += (um_ZsPoint[Nitrogen] * invmass) * invden;
    }
    if (Magnesium >= 0)
    {
        sum_nuclei += (um_ZsPoint[Magnesium] * invmass) * invden;
    }

    sum_nuclei += (SphP[i].Metals[FillEl] * invmass) * invden;
    /* Metals[FillEl] accounts for the dummy elements */
#endif // defined (UM_METAL_COOLING) && (defined (LT_METAL_COOLING) || defined(LT_METAL_COOLING_WAL)) && defined
       // (UM_MET_IN_NONEQ_COOLING)
    return sum_nuclei;
}

/**
 * @brief Computes the number density of nuclei based on the gas density and composition.
 *
 * This function calculates the number density of nuclei in the gas by dividing the gas density by
 * the sum of nuclei, adjusted for the mean molecular weight.
 *
 * @param rho_cgs Gas density in cgs units.
 * @param i Index of the particle or cell for which the computation is performed.
 *
 * @return The number density of nuclei in the gas.
 */
double INLINE_FUNC Um_Compute_n_den(double rho_cgs, int i)
{
    double denom = 4 * SphP[i].HeI + SphP[i].HII + 4 * SphP[i].HeIII + SphP[i].HI + 4 * SphP[i].HeII + SphP[i].HM +
                   2 * SphP[i].H2II + 2 * SphP[i].H2I;
    denom += 5. * SphP[i].HeHII;
    denom += 3 * SphP[i].HD + 2. * SphP[i].DI + 2. * SphP[i].DII;
#if defined(UM_METAL_COOLING) && (defined(LT_METAL_COOLING) || defined(LT_METAL_COOLING_WAL)) &&                       \
    defined(UM_MET_IN_NONEQ_COOLING)
    um_mass = P[i].Mass;
    um_ZsPoint = SphP[i].Metals;
    double invmass = 1.0 / (double)um_mass;
    double invden = Um_Compute_invden(i);
    if (Carbon >= 0)
    {
        denom += (um_ZsPoint[Carbon] * invmass) * invden;
    }
    if (Oxygen >= 0)
    {
        denom += (um_ZsPoint[Oxygen] * invmass) * invden;
    }
    if (Silicon >= 0)
    {
        denom += (um_ZsPoint[Silicon] * invmass) * invden;
    }
    if (Iron >= 0)
    {
        denom += (um_ZsPoint[Iron] * invmass) * invden;
    }
    if (Nitrogen >= 0)
    {
        denom += (um_ZsPoint[Nitrogen] * invmass) * invden;
    }
    if (Magnesium >= 0)
    {
        denom += (um_ZsPoint[Magnesium] * invmass) * invden;
    }
    denom += (SphP[i].Metals[FillEl] * invmass) * invden;
    /* Metals[FillEl] accounts for the dummy elements */
#endif // defined (UM_METAL_COOLING) && (defined (LT_METAL_COOLING) || defined(LT_METAL_COOLING_WAL)) && defined
       // (UM_MET_IN_NONEQ_COOLING)
    double n_den = rho_cgs / (PROTONMASS * denom);
    return n_den;
}

double INLINE_FUNC Um_Compute_mean_nden(int i)
{
    double mean_nden = SphP[i].HI + SphP[i].HII + SphP[i].elec + SphP[i].HeI + SphP[i].HeII + SphP[i].HeIII +
                       SphP[i].HM + SphP[i].H2II + SphP[i].H2I;
    mean_nden += SphP[i].HeHII;
    mean_nden += SphP[i].HD + SphP[i].DI + SphP[i].DII;
#if defined(UM_METAL_COOLING) && (defined(LT_METAL_COOLING) || defined(LT_METAL_COOLING_WAL)) &&                       \
    defined(UM_MET_IN_NONEQ_COOLING)
    um_mass = P[i].Mass;
    um_ZsPoint = SphP[i].Metals;
    double invmass = 1.0 / (double)um_mass;
    double invden = Um_Compute_invden(i);
    if (Carbon >= 0)
    {
        mean_nden += (um_ZsPoint[Carbon] * invmass) / 12 * invden;
    }
    if (Oxygen >= 0)
    {
        mean_nden += (um_ZsPoint[Oxygen] * invmass) / 16 * invden;
    }
    if (Silicon >= 0)
    {
        mean_nden += (um_ZsPoint[Silicon] * invmass) / 28 * invden;
    }
    if (Iron >= 0)
    {
        mean_nden += (um_ZsPoint[Iron] * invmass) / 56 * invden;
    }
    if (Nitrogen >= 0)
    {
        mean_nden += (um_ZsPoint[Nitrogen] * invmass) / 14 * invden;
    }
    if (Magnesium >= 0)
    {
        mean_nden += (um_ZsPoint[Magnesium] * invmass) / 24 * invden;
    }
    mean_nden += (SphP[i].Metals[FillEl] * invmass) / dummy_average_atomic_mass * invden;
    /* Metals[FillEl] accounts for the dummy elements */
#endif // defined (UM_METAL_COOLING) && (defined (LT_METAL_COOLING) || defined(LT_METAL_COOLING_WAL)) && defined
       // (UM_MET_IN_NONEQ_COOLING)
    return mean_nden;
}

double INLINE_FUNC Um_Compute_MeanMolecularWeight(int i)
{
    /* stores the mmw in SphP[i].Um_MeanMolecularWeight
       and sets um_mass, um_ZsPoint: */
    double sum_nuclei = Um_Compute_sum_nuclei(i);
    double mean_nden = Um_Compute_mean_nden(i);
    /* store the mean molecular weight in the structure SphP: */
    SphP[i].Um_MeanMolecularWeight = sum_nuclei / mean_nden;
    return SphP[i].Um_MeanMolecularWeight;
}

#if (defined(SFR) || defined(COOLING)) && !(defined(LT_METAL_COOLING) || defined(LT_METAL_COOLING_WAL))
double Um_DoCooling(double u_old, double rho, double dt, double *ne_guess, int i, int flag) // std SFR + um_chem
#else // (defined(SFR) || defined(COOLING)) && !(defined(LT_METAL_COOLING) || defined(LT_METAL_COOLING_WAL))
#ifndef LT_METAL_COOLING_WAL
double Um_DoCooling(double u_old, double rho, double dt, double *ne_guess, double Z, int i, int flag)
#else // LT_METAL_COOLING_WAL
double Um_DoCooling(double u_old, double rho, double dt, double *ne_guess, double *Zpoint, double Redshift, double DZ,
                    int i, int flag)
#endif // LT_METAL_COOLING_WAL
#endif // (defined(SFR) || defined(COOLING)) && !(defined(LT_METAL_COOLING) || defined(LT_METAL_COOLING_WAL))
{
    /* 'flag' check whether conditions for star formation are fulfilled.
     * - from cooling_and_starformation():
     * f=1  normal cooling
     * f=0  star formation
     */
    double um_u = 0;
#ifdef UM_METAL_COOLING
    double do_tcool;
#endif // UM_METAL_COOLING
    double temp = 0.;
    double u = u_old;
    /* stores the mmw in SphP[i].Um_MeanMolecularWeight
       and sets um_mass, um_ZsPoint: */
    Um_Compute_MeanMolecularWeight(i);
    /* compute temperature (in cgs units): */
    temp = (SphP[i].Gamma - 1.0) / BOLTZMANN * PROTONMASS * SphP[i].Um_MeanMolecularWeight * u_old *
           All.UnitPressure_in_cgs / All.UnitDensity_in_cgs;
    temp = abs(temp);
#ifndef UM_DO_NOT_CUT_MOLECULES
    if (temp >= T_SUP_INTERPOL_LIMIT)
    {
        /* set an arbitrary low value at high T: molecules are dissociated! */
        SphP[i].H2I = 1.e-20;
        SphP[i].H2II = 1.e-20;
        SphP[i].HD = 1.e-20;
        SphP[i].HeHII = 1.e-20;
    }
#endif // UM_DO_NOT_CUT_MOLECULES

#ifdef UM_METAL_COOLING
    if (temp >= T_SUP_INTERPOL_LIMIT)
    {
        /*
           printf(" temp >= T_SUP_INTERPOL_LIMIT  (T=%g, flag=%d, density=%g, pressure=%g, entropy=%g, P.ID=%d) ", temp,
           flag,  SphP[i].Density, SphP[i].Pressure, SphP[i].Entropy, P[i].ID);
         */

        *ne_guess = 1;
#ifndef LT_METAL_COOLING_WAL
        u = DoCoolingZ(u_old, rho, dt, ne_guess, Z, &temp);
        if ((do_tcool = GetCoolingTimeZ(u, rho, ne_guess, Z, &temp)) == 0)
        {
            do_tcool = 1e13;
        }
#else // LT_METAL_COOLING_WAL
        u = DoCoolingMET(u_old, rho, Zpoint, Redshift, DZ, dt, &temp);
        double Z = get_metallicity_solarunits(get_metallicity(i, Iron));
        if ((do_tcool = GetCoolingTimeMET(u, rho, Redshift, DZ, Zpoint, &temp)) == 0)
        {
            do_tcool = 1e13;
        }
#endif // LT_METAL_COOLING_WAL

        do_tcool *= All.UnitTime_in_s / All.HubbleParam; /* in sec */
        do_tcool /= SEC_PER_YEAR; /* in yrs */
        SphP[i].t_elec = SphP[i].t_cool = do_tcool * 0; /* in yrs */
        /*  in this statement, 0 makes cooling time inefficient in timestep.c */
        /* u, u_old, rho, dt in internal units */
    }
    else
    {
        /*
           VERBOSE_ALL(2," temp < T_SUP_INTERPOL_LIMIT  (T=%g, flag=%d, density=%g, pressure=%g, entropy=%g, P.ID=%d) ",
           temp, flag,  SphP[i].Density, SphP[i].Pressure, SphP[i].Entropy, P[i].ID);
         */

        if (flag)
        {
            /* this excludes star forming particles from non-equilibrium module */
            /* flag=1 cooling, flag=0 star formation */
            /* "a_start" and "a_end" are set like in timestep.c */
            double a_start, a_end;
            int ti_step = P[i].TimeBin ? (1 << P[i].TimeBin) : 0;
            int tstart = P[i].Ti_begstep + ti_step / 2; /* midpoint of old step */
            int tend = P[i].Ti_begstep + ti_step + ti_step / 2; /* midpoint of new step */
            if (All.ComovingIntegrationOn)
            {
                a_start = All.TimeBegin * exp(tstart * All.Timebase_interval);
                a_end = All.TimeBegin * exp(tend * All.Timebase_interval);
            }
            else
            {
                a_start = tstart * All.Timebase_interval;
                a_end = tend * All.Timebase_interval;
            }

            (void)compute_abundances(1, i, a_start, a_end, &um_u); /* :set um_u */
            u = um_u; /* internal units */
            *ne_guess = SphP[i].elec;
        }
    }

#else // UM_METAL_COOLING

    if (flag)
    {
        // printf(" UM_METAL_COOLING off:  T=%g, flag=%d \n", temp, flag);
        //
        //  old scheme:
        //       a_start = All.TimeBegin * exp(P[i].Ti_begstep * All.Timebase_interval);
        //       a_end = All.TimeBegin * exp(P[i].Ti_endstep * All.Timebase_interval);
        double a_start, a_end;
        int ti_step = P[i].TimeBin ? (1 << P[i].TimeBin) : 0;
        int tstart = P[i].Ti_begstep + ti_step / 2; /* midpoint of old step */
        int tend = P[i].Ti_begstep + ti_step + ti_step / 2; /* midpoint of new step */
        if (All.ComovingIntegrationOn)
        {
            a_start = All.TimeBegin * exp(tstart * All.Timebase_interval);
            a_end = All.TimeBegin * exp(tend * All.Timebase_interval);
        }
        else
        {
            a_start = tstart * All.Timebase_interval;
            a_end = tend * All.Timebase_interval;
        }

        (void)compute_abundances(1, i, a_start, a_end, &um_u);
        /* :set um_u */
        u = um_u;
        *ne_guess = SphP[i].elec;
    }

#endif // UM_METAL_COOLING

    /* DtEntropy will be updated in lt_sfr.c -> Entropy is updated in timestep.c
       Pressure will be calculated at the usual places using Entropy and DtEntropy */
    return u; /* u  in internal units */
}

/**
 * @brief Computes the cooling time of the gas based on its specific internal energy and density.
 *
 * This function calculates the cooling time of the gas, taking into account the presence of metal cooling
 * and using either a detailed cooling time calculation or an approximation based on the stored cooling time.
 *
 * @param u Specific internal energy of the gas (in code units).
 * @param rho Density of the gas (in code units).
 * @param ne_guess Pointer to a variable where the electron density estimate will be stored.
 * #ifndef LT_METAL_COOLING_WAL
 * @param Z Metallicity of the gas.
 * #else
 * @param Zpoint Pointer to a variable where the metallicity point will be stored (used in some cooling models).
 * @param Redshift Redshift value of the gas (used in some cooling models).
 * @param DZ Redshift interval (used in some cooling models).
 * #endif // LT_METAL_COOLING_WAL
 *
 * @param i Index of the particle or cell for which the calculation is performed.
 *
 * @return The computed cooling time of the gas (in internal units).
 */
#if (defined(SFR) || defined(COOLING)) && !(defined(LT_METAL_COOLING) || defined(LT_METAL_COOLING_WAL))
double Um_GetCoolingTime(double u, double rho, double *ne_guess, int i) // std SFR + um_chem
#else // (defined(SFR) || defined(COOLING)) && !(defined(LT_METAL_COOLING) || defined(LT_METAL_COOLING_WAL))
#ifndef LT_METAL_COOLING_WAL
double Um_GetCoolingTime(double u, double rho, double *ne_guess, double Z, int i)
#else // LT_METAL_COOLING_WAL
double Um_GetCoolingTime(double u, double rho, double *ne_guess, double *Zpoint, double Redshift, double DZ, int i)
#endif // LT_METAL_COOLING_WAL
#endif // (defined(SFR) || defined(COOLING)) && !(defined(LT_METAL_COOLING) || defined(LT_METAL_COOLING_WAL))
{
    double cool_time;
#ifdef UM_METAL_COOLING
    double temp;
#endif // UM_METAL_COOLING
    /* stores the mmw in SphP[i].Um_MeanMolecularWeight
       and sets um_mass, um_ZsPoint: */
    Um_Compute_MeanMolecularWeight(i);
#ifdef UM_METAL_COOLING
    /* compute temperature (in cgs units): */
    temp = (SphP[i].Gamma - 1.0) / BOLTZMANN * PROTONMASS * SphP[i].Um_MeanMolecularWeight * u *
           All.UnitPressure_in_cgs / All.UnitDensity_in_cgs;
    temp = abs(temp);
    if (temp >= T_SUP_INTERPOL_LIMIT)
    {
        *ne_guess = 1;
#ifndef LT_METAL_COOLING_WAL
        cool_time = GetCoolingTimeZ(u, rho, ne_guess, Z, &temp); /*internal units */
#else // LT_METAL_COOLING_WAL
        cool_time = GetCoolingTimeMET(u, rho, Redshift, DZ, Zpoint, &temp);
#endif // LT_METAL_COOLING_WAL
    }
    else
    {
        /*
           a_start = All.TimeBegin * exp(P[i].Ti_begstep * All.Timebase_interval);
           a_end   = All.TimeBegin * exp(P[i].Ti_endstep * All.Timebase_interval);
           ifunc   = compute_abundances(1, i, a_start, a_end, &um_u);
         */
        *ne_guess = SphP[i].elec;
        cool_time = abs(SphP[i].t_cool); /* in yrs */
        cool_time *= SEC_PER_YEAR * All.HubbleParam / All.UnitTime_in_s;
        /* internal units */
    }

#else // UM_METAL_COOLING
    *ne_guess = SphP[i].elec;
    cool_time = abs(SphP[i].t_cool);
    cool_time *= SEC_PER_YEAR * All.HubbleParam / All.UnitTime_in_s;
#endif // UM_METAL_COOLING
    return cool_time;
}

#ifndef LT_METAL_COOLING_WAL
/**
 * @brief Computes the temperature and abundance ratios for a given set of parameters.
 *
 * This function calculates the temperature of the gas based on the specific internal energy
 * and then determines the abundance ratios of various species. It updates pointers to store
 * the electron density, neutral hydrogen density, and ionized helium density based on the
 * presence of metal cooling.
 *
 * @param u Specific internal energy of the gas (in code units).
 * @param rho Density of the gas (in code units).
 * @param ne_guess Pointer to a variable where the electron density estimate will be stored.
 * @param nH0_pointer Pointer to a variable where the neutral hydrogen density will be stored.
 * @param nHeII_pointer Pointer to a variable where the ionized helium density will be stored.
 * @param dt Time step (not used in the current function but included for interface consistency).
 * @param Z Metallicity of the gas (not used in the current function but included for interface consistency).
 * @param i Index of the particle or cell for which the calculation is performed.
 *
 * @return The computed temperature in cgs units.
 */
double Um_AbundanceRatios(double u, double rho, double *ne_guess, double *nH0_pointer, double *nHeII_pointer, double dt,
                          double Z, int i)
{
    double temp;
    /* stores the mmw in SphP[i].Um_MeanMolecularWeight
       and sets um_mass, um_ZsPoint: */
    Um_Compute_MeanMolecularWeight(i);
    /* compute temperature (in cgs units): */
    temp = (SphP[i].Gamma - 1.0) / BOLTZMANN * PROTONMASS * SphP[i].Um_MeanMolecularWeight * u *
           All.UnitPressure_in_cgs / All.UnitDensity_in_cgs;
    temp = abs(temp);
#ifdef UM_METAL_COOLING
    if (temp >= T_SUP_INTERPOL_LIMIT)
    {
        AbundanceRatios(u, rho, ne_guess, nH0_pointer, nHeII_pointer);
    }
    else
    {
        *ne_guess = SphP[i].elec;
        *nH0_pointer = SphP[i].HI;
        *nHeII_pointer = SphP[i].HeII;
    }
#else // UM_METAL_COOLING
    *ne_guess = SphP[i].elec;
    *nH0_pointer = SphP[i].HI;
    *nHeII_pointer = SphP[i].HeII;
#endif // UM_METAL_COOLING
    return temp;
}
#endif // LT_METAL_COOLING_WAL

#ifdef RADTRANSFER
/**
 * @brief Computes the specific intensity of the radiation field at various frequencies.
 *
 * This function calculates the specific intensity \( J_\nu \) of the radiation field
 * at different frequencies, given the photon number density obtained from the radiative
 * transfer calculations. The intensity \( J_\nu \) is computed in units of \( \text{erg} \, \text{s}^{-1} \,
 * \text{cm}^{-2} \, \text{Hz}^{-1} \).
 *
 * @param ithis The index of the particle or cell for which the radiation field is computed.
 */
void rt_get_J_nu(int ithis)
{
    double fac, a3inv;
    if (All.ComovingIntegrationOn)
    {
        a3inv = 1 / (All.Time * All.Time * All.Time);
    }
    else
    {
        a3inv = 1.0;
    }

    fac = All.UnitLength_in_cm / All.HubbleParam;
    fac = fac * fac * fac;
    fac = 1 / fac;
    fac *= a3inv;
    for (int j = 0; j < N_nu; j++)
    {
        double n_gamma = SphP[ithis].n_gamma[j] / P[ithis].Mass * fac; // physical cm^-3
        J_nu[j] = C_LIGHT / 4.0 / M_PI * n_gamma * PLANCK;
        if (J_nu[j] < 1e-60)
        {
            J_nu[j] = 0.0;
        }
    }
}
#endif // RADTRANSFER

/* Some routines by Sebastian E. Nuza*/
#ifdef UM_SEN_H2_SOLOMON_PROCESS
/**
 * @brief Loads the Lyman-Werner (LW) field data.
 *
 * This function reads Lyman-Werner field values from a text file (`LW_field.txt`)
 * and stores them in the global arrays `z_JLW` and `JLW`. The file is expected
 * to contain redshift values and corresponding LW field intensities in two columns.
 * The function reads each line of the file, parses the values, and populates
 * the arrays accordingly.
 */
void load_JLW(void)
{
    /* Opens the Lyman-Werner field bakcground */
    double aux1, aux2;
    FILE *fp;
    VERBOSE(2, "Reading Lyman-Werner field...\n");
    if (!(fp = fopen("LW_field.txt", "r")))
    {
        printf("Cannot find LW_field.txt!\n");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < JLWn; i++)
    {
        fscanf(fp, "%lf %lf\n", &aux1, &aux2);
        z_JLW[i] = aux1;
        JLW[i] = aux2;
        VERBOSE(2, "%d: z = %lf, J_LW = %e\n", i, z_JLW[i], JLW[i]);
    }
    fclose(fp);
}

/**
 * @brief Interpolates the Lyman-Werner (LW) field values based on the redshift.
 *
 * This function calculates the Lyman-Werner field intensity by interpolating
 * between tabulated values of the Lyman-Werner field at different redshifts.
 * The redshift is computed from the scale factor `a`, and the function
 * returns the interpolated Lyman-Werner field intensity.
 *
 * @param a Scale factor of the universe.
 *
 * @return Interpolated Lyman-Werner field intensity at the given scale factor `a`. The returned value is in units of
 * erg/s/Hz/sr/cm².
 */
double get_JLW(double a)
{
    /* Interpolates the Lyman-Werner field values with redshift */
    int ind;
    double c_JLW;
    double fout = 0.;
    double zmin = z_JLW[0];
    double zmax = z_JLW[JLWn - 1];
    double z = 1. / a - 1.;
    ind = (int)((JLWn - 1) * (z - zmin) / (zmax - zmin));
    if (z <= zmin || z >= zmax)
    {
        if (z <= zmin)
        {
            fout = JLW[0];
        }
        if (z >= zmax)
        {
            fout = JLW[JLWn - 1];
        }
    }
    else
    {
        c_JLW = (z - z_JLW[ind]) / (z_JLW[ind + 1] - z_JLW[ind]);
        fout = JLW[ind] + c_JLW * (JLW[ind + 1] - JLW[ind]);
    }
    // printf("zmin = %lf, zmax = %lf, z = %lf, c_JLW = %lf\n",zmin,zmax,z,c_JLW);
    // printf("J_LW(a_now) is: %e erg/s/Hz/sr/cm/cm\n",fout);
    return fout;
}

#endif // UM_SEN_H2_SOLOMON_PROCESS
#endif // UM_CHEMISTRY + !CHEMISTRY
