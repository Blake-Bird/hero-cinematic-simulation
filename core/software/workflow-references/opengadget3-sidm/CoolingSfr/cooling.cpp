/**
 * @file
 *
 * This file is part of the N-body/SPH code OpenGadget3.
 *
 * @brief COOLING Module Implementation.
 *
 * Within this file, all cooling modules implemented in the code are invoked: COOLING_KATZ,
 * LT_METAL_COOLING, LT_METAL_COOLING_WAL.
 * Below is a brief description of each module.
 *
 * - COOLING_KATZ
 *
 * This module follows the implementation described in KWH 95 (Katz, Weinberg, & Hernquist, 1995),
 * available at: https://arxiv.org/abs/astro-ph/9509107.
 *
 * The implementation is briefly summarized here:
 *
 * The cooling processes are calculated for a gas with primordial composition, where:
 *     - X = 0.76 (mass fraction of hydrogen)
 *     - Y = 0.24 (mass fraction of helium)
 *
 * The cooling processes considered for this gas include:
 *     - Collisional ionization of H0, He0, He+,
 *     - Standard recombination of H+, He+, He++,
 *     - Dielectronic recombination of He++,
 *     - Free-free emission (bremsstrahlung radiation)
 *
 * The formulae used are from Black 81, https://articles.adsabs.harvard.edu/pdf/1981MNRAS.197..553B, Table 3
 * using the modification introduced by Cen 92, https://articles.adsabs.harvard.edu/pdf/1992ApJS...78..341C, 2.1.1
 * - 2.1.3
 *
 * To use these formulas, it is necessary to know the abundances of the various atomic species in the gas.
 * To obtain these values, the following assumptions are made:
 *     - The gas is optically thin
 *     - The gas is in ionization equilibrium with the ultraviolet background (UVB) as described by HM01 (Haardt &
 * Madau, 2001, through TREECOOL file)
 *     - The gas is not in thermal equilibrium
 *
 * - COOLING_SUTHERLAND:  LT_METAL_COOLING
 * - COOLING_WAL: LT_METAL_COOLING_WAL
 * - COOLING_LUEDERS (incomplete)
 *
 * @author OpenGadget3-development team, in particular Volker Springel.
 *
 * @copyright Copyright (c) 2026, OpenGadget3-development team lead by Klaus Dolag and Stefano Borgani
 * (see change logs). All rights reserved.
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License v3 or later.
 * See <https://www.gnu.org/licenses/>.
 */

#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../CodeBase/allvars.hpp"
#include "../CodeBase/proto.hpp"
#include "../CodeBase/constants.hpp"
// #include "cooling.h"

#ifdef COOLING

// TODO does this construction serve a purpose? - it prints a message during compilation. But why?
// #define __STR(M) #M
// #define __XSTR(N) __STR(N)
// #pragma message "GADGET_COOLING set to  " __XSTR(GADGET_COOLING)
// #undef __STR
// #undef __XSTR

#ifdef LT_METAL_COOLING_WAL
extern double DZ, Redshift;
#ifdef GL_DUST_COOLING
double DL, DS = 0; // here defined only to allow compilation of cl_CoolingOnly
#endif // GL_DUST_COOLING
#endif // LT_METAL_COOLING_WAL

double XH = HYDROGEN_MASSFRAC; /* Hydrogen mass fraction. This value is usually set to 0.76. But if HYDROGEN_ONLY it is
                                  set to 1 */
double yhelium; /* y = helium number density (nHe0+nHep+nHepp) over Hydrogen number density (nH); y=Y/(4-4Y), where
                   Y=1-XH*/

struct global_cooling_data Cool;

/*
 * Table input (from file TREECOOL) for ionizing parameters (HM01 UVB)
 * These arrays store different photoionization and photoheating rates for Hydrogen and Helium.
 */
float inlogz[TABLESIZE]; /* log(1+z), where z is the redshift*/
float gH0[TABLESIZE]; /* Photoionization rate for neutral hydrogen (H0) in units of s^-1 */
float gHe[TABLESIZE]; /* Photoionization rate for neutral helium (He) in units of s^-1 */
float gHep[TABLESIZE]; /* Photoionization rate for singly ionized helium (He+) in units of s^-1 */
float eH0[TABLESIZE]; /* Photoheating rate for neutral hydrogen (H0) in units of s^-1 */
float eHe[TABLESIZE]; /* Photoheating rate for neutral helium (He) in units of s^-1 */
float eHep[TABLESIZE]; /* Photoheating rate for singly ionized helium (He+) in units of s^-1 */

int nheattab; /* Length of the table */

/**********************************************************************************************************************/
/*                                                      InitCool                                                      */
/**********************************************************************************************************************/

/**
 * @brief Allocates memory for cooling processes and rates parameters.
 *
 * Relevant papers:
 * Black 81, https://articles.adsabs.harvard.edu/pdf/1981MNRAS.197..553B, Table 3
 * Cen 92, https://articles.adsabs.harvard.edu/pdf/1992ApJS...78..341C, 2.1.1 - 2.1.3
 *
 * Each array is allocated with a size of (NCOOLTAB + 1) elements where
 * NCOOLTAB is the number of steps in the COOLing TABle
 *
 * Description of the parameters:
 * - Cool.BetaH0: Collisional excitacion values for neutral hydrogen (H0)
 * - Cool.BetaHep: Collisional excitacion values for singly ionized helium (He+)
 * - Cool.AlphaHp: Recombination Coefficients values for ionized hydrogen (H+)
 * - Cool.AlphaHep: Recombination Coefficients values for singly ionized helium (He+)
 * - Cool.Alphad: Dielectronic Recombination values for for doubly ionized helium (He++)
 * - Cool.AlphaHepp: Recombination Coefficients values for doubly ionized helium (He++)
 * - Cool.GammaeH0: Collisional ionization values for neutral hydrogen (H0)
 * - Cool.GammaeHe0: Collisional ionization values for neutral helium (He0)
 * - Cool.GammaeHep: Collisional ionization values for singly ionized helium (He+)
 * - Cool.Betaff: Bremmstrahlung cooling of all ions values
 */

static void InitCoolMemory(void)
{
    Cool.BetaH0 = (double *)mymalloc("Cool.BetaH0", (NCOOLTAB + 1) * sizeof(double));
    Cool.BetaHep = (double *)mymalloc("Cool.BetaHep", (NCOOLTAB + 1) * sizeof(double));
    Cool.AlphaHp = (double *)mymalloc("Cool.AlphaHp", (NCOOLTAB + 1) * sizeof(double));
    Cool.AlphaHep = (double *)mymalloc("Cool.AlphaHep", (NCOOLTAB + 1) * sizeof(double));
    Cool.Alphad = (double *)mymalloc("Cool.Alphad", (NCOOLTAB + 1) * sizeof(double));
    Cool.AlphaHepp = (double *)mymalloc("Cool.AlphaHepp", (NCOOLTAB + 1) * sizeof(double));
    Cool.GammaeH0 = (double *)mymalloc("Cool.GammaeH0", (NCOOLTAB + 1) * sizeof(double));
    Cool.GammaeHe0 = (double *)mymalloc("Cool.GammaeHe0", (NCOOLTAB + 1) * sizeof(double));
    Cool.GammaeHep = (double *)mymalloc("Cool.GammaeHep", (NCOOLTAB + 1) * sizeof(double));
    Cool.Betaff = (double *)mymalloc("Cool.Betaff", (NCOOLTAB + 1) * sizeof(double));
}

/**
 * @brief Makes Cooling Rates table to interpolate [erg s-1 cm-3]
 *
 * Set up interpolation tables in T for cooling rates given in KWH, ApJS, 105, 19
 *
 * Hydrogen, Helium III recombination rates and collisional ionization
 * cross-sections are updated
 */
// TODO FUnction has side effects, but only within cooling scope (except maybe XH, yhelium, not sure)
// TODO Function overrides HYDROGEN_ONLY (XH is not HYDROGEN_MASSFRAC)
// No dependencies
static void MakeCoolingTable(void)
{
#ifdef NEW_RATES
    double dE, P, A, X, K, U, T_eV;
    double b0, b1, b2, b3, b4, b5, c0, c1, c2, c3, c4, c5, y; /* used in Scholz-Walters fit */
    double E1s_2, Gamma1s_2s, Gamma1s_2p;
#endif // NEW_RATES

    XH = 0.76; // X = 0.76 (mass fraction of hydrogen)
    yhelium = (1 - XH) / (4 * XH); // yhelium = helium abundance / hydrogen abundance

    if (All.MinGasTemp > 0.0)
    {
        Cool.Tmin = log10(0.1 * All.MinGasTemp);
    }
    else
    {
        Cool.Tmin = 1.0;
    }

    Cool.deltaT = (Cool.Tmax - Cool.Tmin) / NCOOLTAB;

    for (int i = 0; i <= NCOOLTAB; i++)
    {
        Cool.BetaH0[i] = Cool.BetaHep[i] = Cool.Betaff[i] = Cool.AlphaHp[i] = Cool.AlphaHep[i] = Cool.AlphaHepp[i] =
            Cool.Alphad[i] = Cool.GammaeH0[i] = Cool.GammaeHe0[i] = Cool.GammaeHep[i] = 0;

        double T = pow(10.0, Cool.Tmin + Cool.deltaT * i);
        double Tfact = 1.0 / (1 + sqrt(T / 1.0e5));

        // TODO figure out who wrote this and slap them
        // eq. 15a Cen 92, Collisional Excitation H0
        if (118348 / T < 70)
        {
            Cool.BetaH0[i] = 7.5e-19 * exp(-118348 / T) * Tfact;
        }

#ifdef NEW_RATES
        // TODO turn into struct. magic numbers
        /* Scholtz-Walters 91 fit */
        if (T >= 2.0e3 && T < 1e8)
        {

            if (T >= 2.0e3 && T < 6.0e4)
            {
                b0 = -3.299613e1;
                b1 = 1.858848e1;
                b2 = -6.052265;
                b3 = 8.603783e-1;
                b4 = -5.717760e-2;
                b5 = 1.451330e-3;

                c0 = -1.630155e2;
                c1 = 8.795711e1;
                c2 = -2.057117e1;
                c3 = 2.359573;
                c4 = -1.339059e-1;
                c5 = 3.021507e-3;
            }
            else
            {
                if (T >= 6.0e4 && T < 6.0e6)
                {
                    b0 = 2.869759e2;
                    b1 = -1.077956e2;
                    b2 = 1.524107e1;
                    b3 = -1.080538;
                    b4 = 3.836975e-2;
                    b5 = -5.467273e-4;

                    c0 = 5.279996e2;
                    c1 = -1.939399e2;
                    c2 = 2.718982e1;
                    c3 = -1.883399;
                    c4 = 6.462462e-2;
                    c5 = -8.811076e-4;
                }
                else
                {
                    b0 = -2.7604708e3;
                    b1 = 7.9339351e2;
                    b2 = -9.1198462e1;
                    b3 = 5.1993362;
                    b4 = -1.4685343e-1;
                    b5 = 1.6404093e-3;

                    c0 = -2.8133632e3;
                    c1 = 8.1509685e2;
                    c2 = -9.4418414e1;
                    c3 = 5.4280565;
                    c4 = -1.5467120e-1;
                    c5 = 1.7439112e-3;
                }

                y = log(T);
                E1s_2 = 10.2; /* eV */

                Gamma1s_2s =
                    exp(b0 + b1 * y + b2 * y * y + b3 * y * y * y + b4 * y * y * y * y + b5 * y * y * y * y * y);
                Gamma1s_2p =
                    exp(c0 + c1 * y + c2 * y * y + c3 * y * y * y + c4 * y * y * y * y + c5 * y * y * y * y * y);

                T_eV = T / eV_to_K;
                Cool.BetaH0[i] = E1s_2 * ELECTRONVOLT_IN_ERGS * (Gamma1s_2s + Gamma1s_2p) * exp(-E1s_2 / T_eV);
            }
        }
#endif // NEW_RATES

        // eq. 15b Cen 92, Collisional Excitation Hepp
        if (473638 / T < 70)
        {
            Cool.BetaHep[i] = 5.54e-17 * pow(T, -0.397) * exp(-473638 / T) * Tfact;
        }
        // eq. 16 Cen 92 Bremmstrahlung cooling of all ions
        Cool.Betaff[i] = 1.43e-27 * sqrt(T) * (1.1 + 0.34 * exp(-(5.5 - log10(T)) * (5.5 - log10(T)) / 3));

#ifdef NEW_RATES
        Cool.AlphaHp[i] = 6.28e-11 * pow(T / 1000, -0.2) / (1. + pow(T / 1.0e6, 0.7)) / sqrt(T);
        Cool.AlphaHepp[i] = 3.36e-10 * pow(T / 1000, -0.2) / (1. + pow(T / 4.0e6, 0.7)) / sqrt(T);
#else // NEW_RATES
        Cool.AlphaHp[i] = 8.4e-11 * pow(T / 1000, -0.2) / (1. + pow(T / 1.0e6, 0.7)) / sqrt(T); /* old Cen92 fit */
        Cool.AlphaHepp[i] = 4. * Cool.AlphaHp[i]; /* old Cen92 fit */
#endif // NEW_RATES

        Cool.AlphaHep[i] = 1.5e-10 * pow(T, -0.6353);
        // eq. 28, Cen 92, Dielectronic Recombination
        if (470000 / T < 70)
        {
            Cool.Alphad[i] = 1.9e-3 * pow(T, -1.5) * exp(-470000 / T) * (1. + 0.3 * exp(-94000 / T));
        }

#ifdef NEW_RATES
        T_eV = T / eV_to_K;

        /* Voronov 97 fit */
        /* hydrogen */
        dE = 13.6;
        P = 0.0;
        A = 0.291e-7;
        X = 0.232;
        K = 0.39;

        U = dE / T_eV;
        Cool.GammaeH0[i] = A * (1.0 + P * sqrt(U)) * pow(U, K) * exp(-U) / (X + U);

        /* Helium */
        dE = 24.6;
        P = 0.0;
        A = 0.175e-7;
        X = 0.18;
        K = 0.35;

        U = dE / T_eV;
        Cool.GammaeHe0[i] = A * (1.0 + P * sqrt(U)) * pow(U, K) * exp(-U) / (X + U);

        /* Hellium II */
        dE = 54.4;
        P = 1.0;
        A = 0.205e-8;
        X = 0.265;
        K = 0.25;

        U = dE / T_eV;
        Cool.GammaeHep[i] = A * (1.0 + P * sqrt(U)) * pow(U, K) * exp(-U) / (X + U);

#else // NEW_RATES
        if (157809.1 / T < 70)
        {
            Cool.GammaeH0[i] = 5.85e-11 * sqrt(T) * exp(-157809.1 / T) * Tfact;
        }

        if (285335.4 / T < 70)
        {
            Cool.GammaeHe0[i] = 2.38e-11 * sqrt(T) * exp(-285335.4 / T) * Tfact;
        }

        if (631515.0 / T < 70)
        {
            Cool.GammaeHep[i] = 5.68e-12 * sqrt(T) * exp(-631515.0 / T) * Tfact;
        }
#endif // NEW_RATES
    } // end for
}

/**
 *
 * @brief  opens TREECOOL file and and saves the values across several arrays
 * (inlogz, gH0, gHe, gHep, eH0, eHe, eHep).
 * Each column in the file corresponds to one of these arrays.
 *
 */
static void ReadIonizeParams(const char *fname)
{
    FILE *fdcool;

    if (!(fdcool = fopen(fname, "r")))
    {
        ENDRUN_MAIN(" Cannot read ionization table in file `%s'\n", fname);
    }

    for (int i = 0; i < TABLESIZE; i++)
    {
        gH0[i] = 0;
    }

    for (int i = 0; i < TABLESIZE; i++)
    {
        if (fscanf(fdcool, "%g %g %g %g %g %g %g", &inlogz[i], &gH0[i], &gHe[i], &gHep[i], &eH0[i], &eHe[i],
                   &eHep[i]) == EOF)
        {
            break;
        }
    }

    fclose(fdcool);

    /*  nheattab is the number of entries in the table */
    nheattab = 0;
    for (int i = 0; i < TABLESIZE; i++)
    {
        if (gH0[i] != 0.0)
        {
            nheattab++;
        }
        else
        {
            break;
        }
    }

    VERBOSE(3, "\n\nread ionization table with %d entries in file `%s'.\n", nheattab, fname);
}

/**
 *
 * @brief  initializes the cooling module to use.
 *
 * #if GADGET_COOLING == COOLING_KATZ -> switch: COOLING, original implementation previously described
 *
 * #if GADGET_COOLING == COOLING_SUTHERLAND -> switch: LT_METAL_COOLING, reads and interpolates
 *                                             Sutherland and Dopita tables
 *
 * #if GADGET_COOLING == COOLING_WAL -> switch: LT_METAL_COOLING_WAL, reads and interpolates
 *                                      Wiersma tables
 *
 * #if GADGET_COOLING == COOLING_LUEDERS -> incomplete
 *
 * InitCoolMemory() -> allocates memory for cooling processes
 * cl_set_Tmin(4.0) -> it sets the minimum Temperature to 10^4 in read_cooling_tables_dummy()
 * cl_set_Tmax(8.0) -> it sets the maximum Temperature to 10^8 in read_cooling_tables_dummy()
 * MakeCoolingTable() -> Makes Cooling Rates table to interpolate
 * ReadIonizeParams(TREECOOL) -> loads TREECOOL values
 * All.Time = All.TimeBegin time of initial conditions of the simulation
 * set_cosmo_factors_for_current_time() -> This function sets cosmological factors used in simulations for the current
 * time. These factors are dependent on whether comoving integration is enabled. IonizeParams() -> Sets the values of
 * photoionizing parameters for a specific redshift based on data from the TREECOOL file.
 *
 */
void InitCool(void)
{
    VERBOSE(3, "Initializing cooling ...");

    InitCoolMemory();
    cl_set_Tmin(4.0);
    cl_set_Tmax(8.0);
    MakeCoolingTable();

    VERBOSE(3, "   reading TREECOOL tables ...");
    ReadIonizeParams("TREECOOL");

    All.Time = All.TimeBegin;
    set_cosmo_factors_for_current_time();
    IonizeParams();
#if GADGET_COOLING == COOLING_SUTHERLAND
    VERBOSE(3, "   reading S&D tables ...");
    read_cooling_tables("coolingtable");

#elif GADGET_COOLING == COOLING_WAL
    read_cooling_tables_dummy();
    VERBOSE(3, "   reading WAL tables ...");
    WalCool_Initialize();

#elif GADGET_COOLING == COOLING_LUEDERS
    VERBOSE(3, "   reading delauni sampled CLOUDI tables ...");
#endif // GADGET_COOLING == COOLING_LUEDERS
}

/**********************************************************************************************************************/
/*                                                  End of InitCool                                                   */
/**********************************************************************************************************************/

/**********************************************************************************************************************/
/*                                              Setter/Getter Functions                                               */
/**********************************************************************************************************************/
// As part of the refactoring of the cooling code all cooling code globals are consolidated into struct Cool
// Modifying Cool from outside this file is STRONGLY discouraged
// For backwards compatibility these setter and getter functions are provided

// Currently only used once, in lt_wal_cooling.c/read_cooling_tables_dummy()
void cl_set_Tmin(double value)
{
    Cool.Tmin = value;
}

// Currently only used once, in lt_wal_cooling.c/read_cooling_tables_dummy()
void cl_set_Tmax(double value)
{
    Cool.Tmax = value;
}

// These are used by lt_wal_cooling.c, mostly
void cl_set_ne(double value)
{
    Cool.ne = value;
}

void cl_set_necgs(double value)
{
    Cool.necgs = value;
}

void cl_set_nHcgs(double value)
{
    Cool.nHcgs = value;
}

// The follwoing functions are called exactly once, in lt_wal_cooling.c
// the variable Cool.bH0 is also only used once, btw! TODO can probably be made non global somehow
void cl_set_bH0(double value)
{
    Cool.bH0 = value;
}

void cl_set_bHep(double value)
{
    Cool.bHep = value;
}

void cl_set_bff(double value)
{
    Cool.bff = value;
}

void cl_set_aHp(double value)
{
    Cool.aHp = value;
}

void cl_set_aHep(double value)
{
    Cool.aHep = value;
}

void cl_set_aHepp(double value)
{
    Cool.aHepp = value;
}

void cl_set_ad(double value)
{
    Cool.ad = value;
}

void cl_set_geH0(double value)
{
    Cool.geH0 = value;
}

void cl_set_geHe0(double value)
{
    Cool.geHe0 = value;
}

void cl_set_geHep(double value)
{
    Cool.geHep = value;
}

void cl_set_gJH0ne(double value)
{
    Cool.gJH0ne = value;
}

void cl_set_gJHe0ne(double value)
{
    Cool.gJHe0ne = value;
}

void cl_set_gJHepne(double value)
{
    Cool.gJHepne = value;
}

void cl_set_nH0(double value)
{
    Cool.nH0 = value;
}

void cl_set_nHp(double value)
{
    Cool.nHp = value;
}

void cl_set_nHep(double value)
{
    Cool.nHep = value;
}

void cl_set_nHe0(double value)
{
    Cool.nHe0 = value;
}

void cl_set_nHepp(double value)
{
    Cool.nHepp = value;
}

/**********************************************************************************************************************/
/*                                           End of Setter/Getter Functions                                           */
/**********************************************************************************************************************/

/**********************************************************************************************************************/
/*                                                  Static Functions                                                  */
/**********************************************************************************************************************/
// These functions are only used in cooling.c

/*!
 * Calculates (heating rate-cooling rate)/n_h^2 in cgs units
 */
// #ifdef LT_METAL_COOLING
// double CoolingRateZ(double logT, double rho, double *nelec, double Z)
// #else
// double CoolingRateSTD(double logT, double rho, double *nelec)
// #endif
//{
#if !defined(LT_METAL_COOLING_WAL) || (GADGET_COOLING == COOLING_KATZ) || (GADGET_COOLING == COOLING_SUTHERLAND)
static double CoolingRate(CoolingRateArgs *args)
{
    //   unpack arguments
    //   TODO using directly would be faster
    double logT = args->logT;
    double rho = args->rho;
    double *nelec = args->nelec;

#ifdef LT_METAL_COOLING
    double Z = args->Z;
#endif // LT_METAL_COOLING

    double Lambda, Heat;
    double LambdaExc, LambdaIon, LambdaRec, LambdaFF, LambdaCmptn = 0.0;
    double LambdaExcH0, LambdaExcHep, LambdaIonH0, LambdaIonHe0, LambdaIonHep;
    double LambdaRecHp, LambdaRecHep, LambdaRecHepp, LambdaRecHepd;
    double redshift;
    double T;

#ifdef LT_METAL_COOLING
    double LambdaMet;
#endif // LT_METAL_COOLING

#if defined(UM_METAL_COOLING) && defined(UM_MET_IN_LT_COOLING)
    double LambdaMetalLines;
#endif // defined(UM_METAL_COOLING) && defined(UM_MET_IN_LT_COOLING)

    if (logT <= Cool.Tmin)
    {
        logT = Cool.Tmin + 0.5 * Cool.deltaT; /* floor at Tmin */
    }

    Cool.nHcgs = XH * rho / PROTONMASS; /* hydrogen number dens in cgs units */

    if (logT < Cool.Tmax)
    {
        find_abundances_and_rates(logT, rho, nelec);

        /* Compute cooling and heating rate (cf KWH Table 1) in units of nH**2 */
#ifdef LT_METAL_COOLING
        /* get cooling rate from sutherland and dopita tables */
        /* keeping abundances and rates calculations */

#if defined(UM_METAL_COOLING) && defined(UM_MET_IN_LT_COOLING)
        if ((Z >= ZMin) && (logT >= log10(UM_LT_INF_LIMIT)))
#else // defined(UM_METAL_COOLING) && defined(UM_MET_IN_LT_COOLING)
        if ((Z > ZMin) && (logT >= Cool.Tmin))
#endif // defined(UM_METAL_COOLING) && defined(UM_MET_IN_LT_COOLING)...
        {
            if ((LambdaMet = (GetMetalLambda(logT, Z) - GetMetalLambda(logT, ZMin)) * Cool.ne *
                             (Cool.nHp + Cool.nHep + Cool.nHepp)) < 0)
                LambdaMet = 0;
        }
        else
        {
            LambdaMet = 0;
        }
#endif // LT_METAL_COOLING

        T = pow(10.0, logT);

#if defined(UM_CHEMISTRY) && defined(UM_METAL_COOLING) && defined(UM_MET_IN_LT_COOLING)
#ifdef UM_e_MET_IMPACTS
        LambdaMetalLines = um_metal_line_cooling(T, rho, LT_NMet, ne * Cool.nHcgs) / (Cool.nHcgs * Cool.nHcgs);
#else // UM_e_MET_IMPACTS
        LambdaMetalLines = um_metal_line_cooling(T, rho, LT_NMet) / (Cool.nHcgs * Cool.nHcgs);
#endif // UM_e_MET_IMPACTS
#endif // defined(UM_CHEMISTRY) && defined(UM_METAL_COOLING) && defined(UM_MET_IN_LT_COOLING)

        LambdaExcH0 = Cool.bH0 * Cool.ne * Cool.nH0;
        LambdaExcHep = Cool.bHep * Cool.ne * Cool.nHep;
        LambdaExc = LambdaExcH0 + LambdaExcHep; /* excitation */

        LambdaIonH0 = 2.18e-11 * Cool.geH0 * Cool.ne * Cool.nH0;
        LambdaIonHe0 = 3.94e-11 * Cool.geHe0 * Cool.ne * Cool.nHe0;
        LambdaIonHep = 8.72e-11 * Cool.geHep * Cool.ne * Cool.nHep;
        LambdaIon = LambdaIonH0 + LambdaIonHe0 + LambdaIonHep; /* ionization */

        LambdaRecHp = 1.036e-16 * T * Cool.ne * (Cool.aHp * Cool.nHp);
        LambdaRecHep = 1.036e-16 * T * Cool.ne * (Cool.aHep * Cool.nHep);
        LambdaRecHepp = 1.036e-16 * T * Cool.ne * (Cool.aHepp * Cool.nHepp);
        LambdaRecHepd = 6.526e-11 * Cool.ad * Cool.ne * Cool.nHep;
        LambdaRec = LambdaRecHp + LambdaRecHep + LambdaRecHepp + LambdaRecHepd;

        LambdaFF = Cool.bff * (Cool.nHp + Cool.nHep + 4 * Cool.nHepp) * Cool.ne;

        Lambda = LambdaExc + LambdaIon + LambdaRec + LambdaFF;

#if defined(UM_CHEMISTRY) && defined(UM_METAL_COOLING) && defined(UM_MET_IN_LT_COOLING)
        Lambda += LambdaMetalLines;
#endif // defined(UM_CHEMISTRY) && defined(UM_METAL_COOLING) && defined(UM_MET_IN_LT_COOLING)

        if (All.ComovingIntegrationOn)
        {
            redshift = 1 / All.Time - 1;
            LambdaCmptn = 5.65e-36 * Cool.ne * (T - 2.73 * (1. + redshift)) * pow(1. + redshift, 4.) / Cool.nHcgs;

            Lambda += LambdaCmptn;
        }
        else
        {
            LambdaCmptn = 0;
        }
        Heat = 0;
        if (Cool.J_UV != 0)
        {
            Heat += (Cool.nH0 * Cool.epsH0 + Cool.nHe0 * Cool.epsHe0 + Cool.nHep * Cool.epsHep) / Cool.nHcgs;
        }

#ifdef LT_METAL_COOLING
        Lambda += LambdaMet;
#endif // LT_METAL_COOLING
    }
    else /* here we're outside of tabulated rates, T>Cool.Tmax K */
    {
        /* at high T (fully ionized); only free-free and Compton cooling are
           present. Assumes no heating. */

        Heat = 0;

        LambdaExcH0 = LambdaExcHep = LambdaIonH0 = LambdaIonHe0 = LambdaIonHep = LambdaRecHp = LambdaRecHep =
            LambdaRecHepp = LambdaRecHepd = 0;

        /* very hot: H and He both fully ionized */
        Cool.nHp = 1.0;
        Cool.nHep = 0;
        Cool.nHepp = yhelium;
        Cool.ne = Cool.nHp + 2.0 * Cool.nHepp;
        *nelec = Cool.ne; /* note: in units of the hydrogen number density */

        T = pow(10.0, logT);
        LambdaFF = 1.42e-27 * sqrt(T) * (1.1 + 0.34 * exp(-(5.5 - logT) * (5.5 - logT) / 3)) *
                   (Cool.nHp + 4 * Cool.nHepp) * Cool.ne;

        if (All.ComovingIntegrationOn)
        {
            redshift = 1 / All.Time - 1;
            /* add inverse Compton cooling off the microwave background */
            LambdaCmptn = 5.65e-36 * Cool.ne * (T - 2.73 * (1. + redshift)) * pow(1. + redshift, 4.) / Cool.nHcgs;
        }
        else
        {
            LambdaCmptn = 0;
        }

        Lambda = LambdaFF + LambdaCmptn;
    }
    return (Heat - Lambda);
}
#endif // LT_METAL_COOLING_WAL

// Called only once in cl_GetCoolingTimeAll
// function is fine, TODO check dependencies, write documentation
static double cl_GetCoolingRateFromUAll(coolingdata_in in, coolingdata_out *out)
{
    double lambda = 0;

#if GADGET_COOLING == COOLING_WAL
    out->Temperature = convert_u_to_tempMET(in.u_old, in.Redshift, in.DZ, &in.Metallicities[0]);
#ifdef GL_DUST_COOLING
    lambda = CoolingRateMET(out->Temperature, in.Redshift, in.DZ, in.DL, in.DS, &in.Metallicities[0]);
#else // GL_DUST_COOLING
    lambda = CoolingRateMET(out->Temperature, in.Redshift, in.DZ, &in.Metallicities[0]);
#endif // GL_DUST_COOLING
    return lambda;
#else // GADGET_COOLING == COOLING_WAL
    out->Temperature = convert_u_to_tempSTD(in.u_old, in.rho_cgs, &out->ne_guess);
#endif // GADGET_COOLING == COOLING_WAL

    CoolingRateArgs args;
    args.logT = log10(out->Temperature);
    args.rho = in.rho_cgs;
    args.nelec = &out->ne_guess;

#if GADGET_COOLING == COOLING_KATZ
    //  lambda = CoolingRateSTD(log10(out->Temperature), in.rho_cgs, &out->ne_guess);
    lambda = CoolingRate(&args);
#elif GADGET_COOLING == COOLING_SUTHERLAND
    //  lambda = CoolingRateZ(log10(out->Temperature), in.rho_cgs, &out->ne_guess, in.Z);
    args.Z = in.Z;
    lambda = CoolingRate(&args);
#elif GADGET_COOLING == COOLING_LUEDERS
// ???
#endif // GADGET_COOLING == COOLING_KATZ
    return lambda;
}

/**********************************************************************************************************************/
/*                                              End of Static Functions                                               */
/**********************************************************************************************************************/

/**********************************************************************************************************************/
/*                                                  Public Functions                                                  */
/**********************************************************************************************************************/

// Goal: These functions should serve as the public interface of the cooling code
// There is one function for each purpose that works differently under different circumstances
double cl_GetCooling()
{
    return 5;
}

double cl_GetCoolingRate()
{
    return 5;
}

double cl_get_CoolingTime()
{
    return 5;
}

/**********************************************************************************************************************/
/*                                              End of Public Functions                                               */
/**********************************************************************************************************************/

// called only once, in io.c
// No side effects, TODO check dependencies, what is DZ?
void particle2in_cooling(coolingdata_in *in, int i)
{
    double a3inv;
    if (All.ComovingIntegrationOn)
    {
        a3inv = 1 / (All.Time * All.Time * All.Time);
    }
    else
    {
        a3inv = 1.;
    }

    in->rho_cgs = SphP[i].Density * a3inv * All.HubbleParam * All.HubbleParam * All.UnitDensity_in_cgs;
    in->u_old = SphP[i].Entropy / GAMMA_MINUS1 * pow(SphP[i].Density * a3inv, GAMMA_MINUS1) * All.UnitPressure_in_cgs /
                All.UnitDensity_in_cgs;

#if defined(SFR) && !defined(LT_STELLAREVOLUTION)
    in->Z = MPP(i).Metallicity;
#elif defined(LT_STELLAREVOLUTION)
    in->Z = get_metallicity_solarunits(get_metallicity(i, Iron));
#endif // SFR && !LT_STELLAREVOLUTION

#if GADGET_COOLING == COOLING_WAL
    if (All.ComovingIntegrationOn)
    {
        in->Redshift = 1.0 / All.Time - 1; /* note that ComovingIntegration is alway on with wal cooling */
    }
    else
    {
        in->Redshift = 0.0;
    }

    //  get_cool_redshift(Redshift, &in->DZ);
    in->DZ = lt_cl_GetCoolRedshift(in->Redshift);

    double Metallicities[LT_NMet];
    memset(Metallicities, 0, sizeof(double) * LT_NMet);
    set_metallicities(i, &Metallicities[0], a3inv);

    for (int k = 0; k < LT_NMet; k++)
    {
        in->Metallicities[k] = Metallicities[k];
    }
#endif // GADGET_COOLING == COOLING_WAL
}

// Called only once, in io.c
// function is fine, TODO remove this comment, write documentation
double cl_GetCoolingTimeAll(coolingdata_in in, coolingdata_out *out)
{
    out->LambdaNet = cl_GetCoolingRateFromUAll(in, out);

    // Do we actually have heating due to UV background?
    if (out->LambdaNet >= 0)
    {
        return 0;
    }

    double nHcgs = XH * in.rho_cgs / PROTONMASS; /* hydrogen number dens in cgs units */
    double ratefact = nHcgs * nHcgs / in.rho_cgs;
    double coolingtime = in.u_old / (-ratefact * out->LambdaNet);
    coolingtime *= All.HubbleParam / All.UnitTime_in_s;

    return coolingtime;
}

// Called in lots of places, mostly cooling.c
// function is fine, TODO check dependencies, write documentation
double DoCooling(coolingdata_in in, coolingdata_out *out, double dt)
{
    double unew = in.u_old;
#if GADGET_COOLING == COOLING_KATZ
    unew = DoCoolingSTD(in.u_old, in.rho_cgs, dt, &out->ne_guess);
#elif GADGET_COOLING == COOLING_SUTHERLAND
    unew = DoCoolingZ(in.u_old, in.rho_cgs, dt, &out->ne_guess, in.Z, &out->Temperature);
#elif GADGET_COOLING == COOLING_WAL
#ifdef GL_DUST_COOLING
    unew = DoCoolingMET(in.u_old, in.rho_cgs, &in.Metallicities[0], in.DL, in.DS, in.Redshift, in.DZ, dt,
                        &out->Temperature);
#else // GL_DUST_COOLING
    unew = DoCoolingMET(in.u_old, in.rho_cgs, &in.Metallicities[0], in.Redshift, in.DZ, dt, &out->Temperature);
#endif // GL_DUST_COOLING
#elif GADGET_COOLING == COOLING_LUEDERS
    // do nothing?
#endif // GADGET_COOLING == COOLING_LUEDERS
    return unew;
}
/**
 * @brief Interpolates the UV background table (TREECOOL) to set the photoionization
 * and photoheating rates at the current redshift. If z exceeds the maximum redshift
 * of the table, the rates are set to zero since no UVB can be applied.
 *
 */
// TODO Has side effects to chemistry noneq code, but no dependencies. Mostly fine.
void IonizeParams(void)
{
    if (!All.ComovingIntegrationOn)
    {
        SetZeroIonization();
        return;
    }

    double redshift = 1 / All.Time - 1;
    double logz =
        log10(redshift + 1.0); // In TREECOOL the redshift is tabulated as log10(z+1), this conversion is then applied

    int ilow = 0;
    for (int i = 0; i < nheattab; i++)
    {
        if (inlogz[i] < logz)
        {
            ilow = i;
        }
        else
        {
            break; // It updates ilow while inlogz[i] < logz, then it stops once inlogz[i] ≥ logz
        }
    }

    double dzlow = logz - inlogz[ilow];
    double dzhi = inlogz[ilow + 1] - logz;

    VERBOSE(3, " UM check: at redshift %g/%g: %d/%g/%g\n", redshift, pow(10., inlogz[nheattab - 1]), nheattab,
            gH0[ilow], gH0[ilow + 1]);

    if (logz > inlogz[nheattab - 1] || gH0[ilow] == 0 || gH0[ilow + 1] == 0 || nheattab == 0)
    {
        SetZeroIonization();
#ifdef UM_CHEMISTRY
        gJH0uvb = gJHe0uvb = gJHepuvb = 0;
        epsH0uvb = epsHe0uvb = epsHepuvb = 0;
#endif // UM_CHEMISTRY
        return;
    }
    else
    {
        Cool.J_UV = 1.e-21; /* irrelevant as long as it's not 0 */
    }

#ifdef UM_CHEMISTRY_NO_UV_BACKGROUND_IN_COSMO_EVOLUTION
    Cool.J_UV = 0.; /* neglect UV background */
#endif // UM_CHEMISTRY_NO_UV_BACKGROUND_IN_COSMO_EVOLUTION
    // log-linear interpolation between the two bracketing redshifts
    Cool.gJH0 = JAMPL * pow(10., (dzhi * log10(gH0[ilow]) + dzlow * log10(gH0[ilow + 1])) / (dzlow + dzhi));
    Cool.gJHe0 = JAMPL * pow(10., (dzhi * log10(gHe[ilow]) + dzlow * log10(gHe[ilow + 1])) / (dzlow + dzhi));
    Cool.gJHep = JAMPL * pow(10., (dzhi * log10(gHep[ilow]) + dzlow * log10(gHep[ilow + 1])) / (dzlow + dzhi));
    Cool.epsH0 = JAMPL * pow(10., (dzhi * log10(eH0[ilow]) + dzlow * log10(eH0[ilow + 1])) / (dzlow + dzhi));
    Cool.epsHe0 = JAMPL * pow(10., (dzhi * log10(eHe[ilow]) + dzlow * log10(eHe[ilow + 1])) / (dzlow + dzhi));
    Cool.epsHep = JAMPL * pow(10., (dzhi * log10(eHep[ilow]) + dzlow * log10(eHep[ilow + 1])) / (dzlow + dzhi));

#ifdef UM_CHEMISTRY
    gJH0uvb = Cool.gJH0;
    gJHe0uvb = Cool.gJHe0;
    gJHepuvb = Cool.gJHep;
    epsH0uvb = Cool.epsH0;
    epsHe0uvb = Cool.epsHe0;
    epsHepuvb = Cool.epsHep;

    VERBOSE(3, " **** interpolated gJH0uvb = %g, gJHe0uvb = %g, gJHepuvb = %g \n", gJH0uvb, gJHe0uvb, gJHepuvb);
    VERBOSE(3, " **** interpolated epsH0uvb = %g, epsHe0uvb = %g, epsHepuvb = %g\n\n", epsH0uvb, epsHe0uvb, epsHepuvb);
#endif // UM_CHEMISTRY

    return;
}
/**
 * @brief The the photoionization and photoheating rates are set to zero (no UVB can be applied)
 */
void SetZeroIonization(void)
{
    Cool.gJHe0 = Cool.gJHep = Cool.gJH0 = 0;
    Cool.epsHe0 = Cool.epsHep = Cool.epsH0 = 0;
    Cool.J_UV = 0;
}

/**
 * @brief This function performs Only Cooling - it is used when Star formation is disabled
 */
// called only from run.c
// TODO Needs more work.
void cl_CoolingOnly(void)
{
    TIMER_PUSH(1, "cl_cooling_only");
    double ascale = 1;
#ifdef LT_METAL_COOLING_WAL
    if (All.ComovingIntegrationOn)
    {
        Redshift = 1.0 / ascale - 1;
    }
    else
    {
        Redshift = 0.0;
    }

    //  get_cool_redshift(Redshift, &DZ);
    DZ = lt_cl_GetCoolRedshift(Redshift);
    // number of metals (Config) < number of solar abundances (WAL Tables)
    ENDRUN_IF(LT_NMet < get_cool_n_el(), "1009");
    // load the two WAL Cooling Tables bracketing the current redshift
    WalCool_tables_load(Redshift);
#endif // LT_METAL_COOLING_WAL

    double a3inv, hubble_a = 0, time_hubble_a;
    if (All.ComovingIntegrationOn)
    {
        /* Factors for comoving integration of hydro */
        a3inv = 1 / (All.Time * All.Time * All.Time);
        hubble_a = hubble_function(All.Time);
        time_hubble_a = All.Time * hubble_a;
        ascale = All.Time;
    }
    else
    {
        a3inv = time_hubble_a = hubble_a = ascale = 1;
    }

    for (int i = FirstActiveParticle; i >= 0; i = NextActiveParticle[i])
    {
#ifdef BLACK_HOLES
        if (P[i].Type == 0 && P[i].Mass > 0)
#else // BLACK_HOLES
        if (P[i].Type == 0)
#endif // BLACK_HOLES
        {
            double dt = (P[i].TimeBin ? (1 << P[i].TimeBin) : 0) * All.Timebase_interval;
            /*  the actual time-step */

            double dtime = dt;
            if (All.ComovingIntegrationOn)
            {
                dtime = All.Time * dt / time_hubble_a;
            }

            // electron abundance (gives ionization state and mean molecular weight)
            double ne = SphP[i].elec;

            double fac_entr_to_u = pow(SphP[i].Density * a3inv, GAMMA_MINUS1) / GAMMA_MINUS1;
            double uold = max(All.MinEgySpec, (double)SphP[i].Entropy * fac_entr_to_u);

#ifdef LT_METAL_COOLING_WAL
            double Metallicities[LT_NMet];

            set_metallicities(i, &Metallicities[0], a3inv); /* note: sets the physical density too! */
            // double Zcool = get_metallicity_solarunits(get_metallicity(i, Iron));
#elif defined(LT_METAL_COOLING)
            double Zcool = get_metallicity_solarunits(get_metallicity(i, Iron));
#endif // LT_METAL_COOLING_WAL

            double unew;

#ifdef UM_CHEMISTRY
            double ti_step = P[i].TimeBin ? (1 << P[i].TimeBin) : 0;
            double tstart = P[i].Ti_begstep + ti_step / 2; /* midpoint of old step */
            double tend = P[i].Ti_begstep + ti_step + ti_step / 2; /* midpoint of new step */
            double a_start, a_end;
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

            /* flag=1: cooling */
#ifdef LT_METAL_COOLING_WAL
            unew = Um_DoCooling(uold, SphP[i].Density * a3inv, dtime, &ne, &Metallicities[0], Redshift, DZ, i, 1);
#elif defined(LT_METAL_COOLING)
            // 	  unew = Um_DoCooling(uold, SphP[i].Density * a3inv, dtime, &ne, Zcool, i, flag);
            unew = Um_DoCooling(uold, SphP[i].Density * a3inv, dtime, &ne, Zcool, i, 1);
#else // LT_METAL_COOLING_WAL
            unew = Um_DoCooling(uold, SphP[i].Density * a3inv, dtime, &ne, i, 1);
#endif // LT_METAL_COOLING_WAL

#else // UM_CHEMISTRY

#if !defined(RT_COOLING_PHOTOHEATING) && (defined(LT_METAL_COOLING_WAL) || defined(LT_METAL_COOLING))
            double Temperature;
#endif // !RT_COOLING_PHOTOHEATING && (LT_METAL_COOLING_WAL || LT_METAL_COOLING)
#ifndef RT_COOLING_PHOTOHEATING
#ifdef LT_METAL_COOLING_WAL
#ifdef GL_DUST_COOLING
            unew = DoCoolingMET(uold, SphP[i].Density * a3inv, &Metallicities[0], DL, DS, Redshift, DZ, dtime,
                                &Temperature);
#else // GL_DUST_COOLING
            unew = DoCoolingMET(uold, SphP[i].Density * a3inv, &Metallicities[0], Redshift, DZ, dtime, &Temperature);
#endif // GL_DUST_COOLING
#elif defined(LT_METAL_COOLING)
            unew = DoCoolingZ(uold, SphP[i].Density * a3inv, dtime, &ne, Zcool, &Temperature);
#else // LT_METAL_COOLING_WAL
            unew = DoCoolingSTD(uold, SphP[i].Density * a3inv, dtime, &ne);
#endif // LT_METAL_COOLING_WAL
#else // RT_COOLING_PHOTOHEATING
            unew = uold + dt * fac_entr_to_u * (rt_DoHeating(i, dt) + rt_DoCooling(i, dt));
#endif // RT_COOLING_PHOTOHEATING

#endif // UM_CHEMISTRY

#if GADGET_HYDRO == HYDRO_MFM
            SphP[i].InternalEnergy = unew;
#endif // GADGET_HYDRO == HYDRO_MFM
            SphP[i].Entropy = unew / fac_entr_to_u;
            SphP[i].elec = ne;

#if GADGET_HYDRO == HYDRO_MFM
            SphP[i].InternalEnergyPred = SphP[i].InternalEnergy;
            mfm_synchronize_entropy(i);
#endif // GADGET_HYDRO == HYDRO_MFM
            SphP[i].EntropyPred = SphP[i].Entropy;
            SphP[i].Pressure = get_pressure(i);
        }
    }
    TIMER_POP(1, "cl_cooling_only");
}

#ifndef LT_METAL_COOLING_WAL

#ifdef LT_METAL_COOLING
double GetMetalLambda(double, double);
#endif // LT_METAL_COOLING

/*!
 * returns new internal energy per unit mass.
 * Arguments are passed in code units, density is proper density.
 */
// TODO relatively little side effects? Needs work. Unify function header.
#ifdef LT_METAL_COOLING
double DoCoolingZ(double u_old, double rho, double dt, double *ne_guess, double Z, double *temp)
#else // LT_METAL_COOLING
double DoCoolingSTD(double u_old, double rho, double dt, double *ne_guess)
#endif // LT_METAL_COOLING
{
    double du;
    int iter = 0;

    double DoCool_u_old_input = u_old;
    double DoCool_rho_input = rho;
    double DoCool_dt_input = dt;
    double DoCool_ne_guess_input = *ne_guess;

    rho *= All.UnitDensity_in_cgs * All.HubbleParam * All.HubbleParam; /* convert to physical cgs units */
    u_old *= All.UnitPressure_in_cgs / All.UnitDensity_in_cgs;
    dt *= All.UnitTime_in_s / All.HubbleParam;

    Cool.nHcgs = XH * rho / PROTONMASS; /* hydrogen number dens in cgs units */
    double ratefact = Cool.nHcgs * Cool.nHcgs / rho;

    double u = u_old;
    double u_lower = u;
    double u_upper = u;

#ifdef LT_METAL_COOLING
    double LambdaNet = CoolingRateFromU(u, rho, ne_guess, Z, temp);
#else // LT_METAL_COOLING
    double LambdaNet = CoolingRateFromU(u, rho, ne_guess);
#endif // LT_METAL_COOLING

    /* bracketing */

    // heating
    if (u - u_old - ratefact * LambdaNet * dt < 0)
    {
        u_upper *= sqrt(1.1);
        u_lower /= sqrt(1.1);
#ifdef LT_METAL_COOLING
        while (u_upper - u_old - ratefact * CoolingRateFromU(u_upper, rho, ne_guess, Z, temp) * dt < 0)
#else // LT_METAL_COOLING
        while (u_upper - u_old - ratefact * CoolingRateFromU(u_upper, rho, ne_guess) * dt < 0)
#endif // LT_METAL_COOLING
        {
            u_upper *= 1.1;
            u_lower *= 1.1;
        }
    }

    if (u - u_old - ratefact * LambdaNet * dt > 0)
    {
        u_lower /= sqrt(1.1);
        u_upper *= sqrt(1.1);
#ifdef LT_METAL_COOLING
        while (u_lower - u_old - ratefact * CoolingRateFromU(u_lower, rho, ne_guess, Z, temp) * dt > 0)
#else // LT_METAL_COOLING
        while (u_lower - u_old - ratefact * CoolingRateFromU(u_lower, rho, ne_guess) * dt > 0)
#endif // LT_METAL_COOLING
        {
            u_upper /= 1.1;
            u_lower /= 1.1;
        }
    }

    do
    {
        u = 0.5 * (u_lower + u_upper);

#ifdef LT_METAL_COOLING
        LambdaNet = CoolingRateFromU(u, rho, ne_guess, Z, temp);
#else // LT_METAL_COOLING
        LambdaNet = CoolingRateFromU(u, rho, ne_guess);
#endif // LT_METAL_COOLING

        if (u - u_old - ratefact * LambdaNet * dt > 0)
        {
            u_upper = u;
        }
        else
        {
            u_lower = u;
        }

        du = u_upper - u_lower;

        iter++;
        if (iter >= (MAXITER - 10))
        {
            VERBOSE_ALL(3, "u= %g", u);
        }
    } while (abs(du / u) > 1.0e-6 && iter < MAXITER);

    ENDRUN_IF(iter >= MAXITER,
              "failed to converge in DoCooling()\nDoCool_u_old_input=%g\nDoCool_rho_input= %g\nDoCool_dt_input= "
              "%g\nDoCool_ne_guess_input= %g",
              DoCool_u_old_input, DoCool_rho_input, DoCool_dt_input, DoCool_ne_guess_input);

    u *= All.UnitDensity_in_cgs / All.UnitPressure_in_cgs; /* to internal units */

    return u;
}

/*!
 * returns cooling time.
 * NOTE: If we actually have heating, a cooling time of 0 is returned.
 */
// TODO fix header, figure out dependencies
#ifdef LT_METAL_COOLING
double GetCoolingTimeZ(double u_old, double rho, double *ne_guess, double Z, double *temp)
#else // LT_METAL_COOLING
double GetCoolingTimeSTD(double u_old, double rho, double *ne_guess)
#endif // LT_METAL_COOLING
{
    double coolingtime;

    // convert to physical cgs units
    rho *= All.UnitDensity_in_cgs * All.HubbleParam * All.HubbleParam;
    u_old *= All.UnitPressure_in_cgs / All.UnitDensity_in_cgs;

    Cool.nHcgs = XH * rho / PROTONMASS; /* hydrogen number dens in cgs units */
    double ratefact = Cool.nHcgs * Cool.nHcgs / rho;

    double u = u_old;

#ifdef LT_METAL_COOLING
    double LambdaNet = CoolingRateFromU(u, rho, ne_guess, Z, temp);
#else // LT_METAL_COOLING
    double LambdaNet = CoolingRateFromU(u, rho, ne_guess);
#endif // LT_METAL_COOLING

    /* bracketing */
    if (LambdaNet >= 0) /* ups, we have actually heating due to UV background */
    {
        return 0;
    }

    coolingtime = u_old / (-ratefact * LambdaNet);
    coolingtime *= All.HubbleParam / All.UnitTime_in_s;

    return coolingtime;
}

// called in too many places
/*!
 * this function determines the electron fraction, and hence the mean
 * molecular weight. With it arrives at a self-consistent temperature.
 * Element abundances and the rates for the emission are also computed
 */
double convert_u_to_tempSTD(double u, double rho, double *ne_guess)
{
    // TODO i think these intermediate variables can be safely removed
    double u_input = u;
    double rho_input = rho;
    double ne_input = *ne_guess;

    double mu = (1 + 4 * yhelium) / (1 + yhelium + *ne_guess);
    double temp = GAMMA_MINUS1 / BOLTZMANN * u * PROTONMASS * mu;

    double maxT = 0;
    double temp_old;
    int iter = 0;
    do
    {
        double ne_old = *ne_guess;

        find_abundances_and_rates(log10(temp), rho, ne_guess);
        temp_old = temp;

        mu = (1 + 4 * yhelium) / (1 + yhelium + *ne_guess);

        double temp_new = GAMMA_MINUS1 / BOLTZMANN * u * PROTONMASS * mu;

        maxT =
            max(maxT, temp_new / (1 + yhelium + *ne_guess) * abs((*ne_guess - ne_old) / (temp_new - temp_old + 1.0)));

        temp = temp_old + (temp_new - temp_old) / (1 + maxT);
        iter++;

        if (iter > (MAXITER - 10))
#if defined(LT_SMOOTH_Z) || defined(LT_METAL_COOLING)
            if (!ignore_failure_in_convert_u)
#endif // defined(LT_SMOOTH_Z) || defined(LT_METAL_COOLING)
                printf("-> temp= %g ne=%g\n", temp, *ne_guess);
    } while (abs(temp - temp_old) > 1.0e-3 * temp && iter < MAXITER);

    if (iter >= MAXITER)
    {
#if defined(LT_SMOOTH_Z) || defined(LT_METAL_COOLING)
        if (!ignore_failure_in_convert_u)
#endif // defined(LT_SMOOTH_Z) || defined(LT_METAL_COOLING)
        {
            ENDRUN_MAIN("failed to converge in convert_u_to_tempSTD()\nu_input= %g\nrho_input=%g\n ne_input=%g",
                        u_input, rho_input, ne_input);
            //      printf("DoCool_u_old_input=%g\nDoCool_rho_input= %g\nDoCool_dt_input= "
            //             "%g\nDoCool_ne_guess_input= %g\n",
            //             DoCool_u_old_input, DoCool_rho_input, DoCool_dt_input,
            //             DoCool_ne_guess_input);
        }
    }

    return temp;
}

/*!
 * this function computes the actual abundance ratios
 */
void find_abundances_and_rates(double logT, double rho, double *ne_guess)
{
    double logT_input = logT;
    double rho_input = rho;
    double ne_input = *ne_guess;

    if (logT <= Cool.Tmin) /* everything neutral */
    {
        Cool.nH0 = 1.0;
        Cool.nHe0 = yhelium;
        Cool.nHp = 0;
        Cool.nHep = 0;
        Cool.nHepp = 0;
        Cool.ne = 0;
        *ne_guess = 0;
        return;
    }

    if (logT >= Cool.Tmax) /* everything is ionized */
    {
        Cool.nH0 = 0;
        Cool.nHe0 = 0;
        Cool.nHp = 1.0;
        Cool.nHep = 0;
        Cool.nHepp = yhelium;
        Cool.ne = Cool.nHp + 2.0 * Cool.nHepp;
        *ne_guess = Cool.ne; /* note: in units of the hydrogen number density */
        return;
    }

    double t = (logT - Cool.Tmin) / Cool.deltaT;
    int j = (int)t;
    double fhi = t - j;
    double flow = 1 - fhi;

    if (*ne_guess == 0)
    {
        *ne_guess = 1.0;
    }

    Cool.nHcgs = XH * rho / PROTONMASS; /* hydrogen number dens in cgs units */

    Cool.ne = *ne_guess;
    double neold = Cool.ne;
    Cool.necgs = Cool.ne * Cool.nHcgs;

    /* evaluate number densities iteratively (cf KWH eqns 33-38) in units of nH */
    int niter = 0;
    do
    {
        niter++;

        Cool.aHp = flow * Cool.AlphaHp[j] + fhi * Cool.AlphaHp[j + 1];
        Cool.aHep = flow * Cool.AlphaHep[j] + fhi * Cool.AlphaHep[j + 1];
        Cool.aHepp = flow * Cool.AlphaHepp[j] + fhi * Cool.AlphaHepp[j + 1];
        Cool.ad = flow * Cool.Alphad[j] + fhi * Cool.Alphad[j + 1];
        Cool.geH0 = flow * Cool.GammaeH0[j] + fhi * Cool.GammaeH0[j + 1];
        Cool.geHe0 = flow * Cool.GammaeHe0[j] + fhi * Cool.GammaeHe0[j + 1];
        Cool.geHep = flow * Cool.GammaeHep[j] + fhi * Cool.GammaeHep[j + 1];

        if (Cool.necgs <= 1.e-25 || Cool.J_UV == 0)
        {
            Cool.gJH0ne = Cool.gJHe0ne = Cool.gJHepne = 0;
        }
        else
        {
            Cool.gJH0ne = Cool.gJH0 / Cool.necgs;
            Cool.gJHe0ne = Cool.gJHe0 / Cool.necgs;
            Cool.gJHepne = Cool.gJHep / Cool.necgs;
        }

        // Katz95 eqs. 33 and 34
        Cool.nH0 = Cool.aHp / (Cool.aHp + Cool.geH0 + Cool.gJH0ne);
        Cool.nHp = 1.0 - Cool.nH0; /* eqn (34) */

        if ((Cool.gJHe0ne + Cool.geHe0) <= SMALLNUM) /* no ionization at all */
        {
            Cool.nHep = 0.0;
            Cool.nHepp = 0.0;
            Cool.nHe0 = yhelium;
        }
        else
        {
            // Katz95 eqs. 35-37
            Cool.nHep = yhelium / (1.0 + (Cool.aHep + Cool.ad) / (Cool.geHe0 + Cool.gJHe0ne) +
                                   (Cool.geHep + Cool.gJHepne) / Cool.aHepp);
            Cool.nHe0 = Cool.nHep * (Cool.aHep + Cool.ad) / (Cool.geHe0 + Cool.gJHe0ne);
            Cool.nHepp = Cool.nHep * (Cool.geHep + Cool.gJHepne) / Cool.aHepp;
        }

        neold = Cool.ne;
        // Katz95 eq. 38
        Cool.ne = Cool.nHp + Cool.nHep + 2 * Cool.nHepp;

        Cool.necgs = Cool.ne * Cool.nHcgs;

        if (Cool.J_UV == 0)
        {
            break;
        }

        Cool.ne = 0.5 * (Cool.ne + neold);
        Cool.necgs = Cool.ne * Cool.nHcgs;

        if (abs(Cool.ne - neold) < 1.0e-4)
        {
            break;
        }

        if (niter > (MAXITER - 10))
            VERBOSE_ALL(3, "ne= %g  niter=%d", Cool.ne, niter);
    } while (niter < MAXITER);

    ENDRUN_IF(niter >= MAXITER,
              "no convergence reached in find_abundances_and_rates()\nlogT_input= %g  rho_input= %g  ne_input= %g\n",
              logT_input, rho_input, ne_input);
    //    printf("DoCool_u_old_input=%g\nDoCool_rho_input= %g\nDoCool_dt_input= "
    //           "%g\nDoCool_ne_guess_input= %g\n",
    //           DoCool_u_old_input, DoCool_rho_input, DoCool_dt_input,
    //           DoCool_ne_guess_input);

    Cool.bH0 = flow * Cool.BetaH0[j] + fhi * Cool.BetaH0[j + 1];
    Cool.bHep = flow * Cool.BetaHep[j] + fhi * Cool.BetaHep[j + 1];
    Cool.bff = flow * Cool.Betaff[j] + fhi * Cool.Betaff[j + 1];

    *ne_guess = Cool.ne;
}

/*!
 * this function first computes the self-consistent temperature
 * and abundance ratios, and then it calculates
 * (heating rate-cooling rate)/n_h^2 in cgs units
 * TODO is that description even accurate?
 */
// TODO Unify function headers
#ifdef LT_METAL_COOLING
double CoolingRateFromU(double u, double rho, double *ne_guess, double Z, double *temp)
#else // LT_METAL_COOLING
double CoolingRateFromU(double u, double rho, double *ne_guess)
#endif // LT_METAL_COOLING
{
#ifndef LT_METAL_COOLING
    double temp = convert_u_to_tempSTD(u, rho, ne_guess);
#endif // LT_METAL_COOLING
    CoolingRateArgs args;

#ifndef LT_METAL_COOLING
    args.logT = log10(temp);
#else // LT_METAL_COOLING
    args.logT = log10(*temp);
#endif // LT_METAL_COOLING
    args.rho = rho;
    args.nelec = ne_guess;

#ifdef LT_METAL_COOLING
    args.Z = Z;
#endif // LT_METAL_COOLING

    return CoolingRate(&args);

    // #ifdef LT_METAL_COOLING
    //   return CoolingRateZ(log10(temp), rho, ne_guess, Z);
    // #else
    //   return CoolingRateSTD(log10(temp), rho, ne_guess);
    // #endif
}

/*!
 * this function computes the self-consistent temperature
 * and abundance ratios
 */
// This function is fine, mostly just calls taht other function - TODO remove comment
double AbundanceRatios(double u, double rho, double *ne_guess, double *nH0_pointer, double *nHeII_pointer)
{
    //  DoCool_u_old_input = u;
    //  DoCool_rho_input = rho;
    //  DoCool_ne_guess_input = *ne_guess;

    rho *= All.UnitDensity_in_cgs * All.HubbleParam * All.HubbleParam; /* convert to physical cgs units */
    u *= All.UnitPressure_in_cgs / All.UnitDensity_in_cgs;

    double temp = convert_u_to_tempSTD(u, rho, ne_guess);

    *nH0_pointer = Cool.nH0;
    *nHeII_pointer = Cool.nHep;
    *ne_guess = Cool.ne;

    return temp;
}

#ifdef LT_METAL_COOLING
/*  M E T A L   C O O L I N G  */

double GetMetalLambda(double logT, double Zsol)
{
    double t, u, log_temp = logT;
    double x_1y, x_1y_1, xy_1;
    int Zi, Ti;

    /* note: Zsol is supposed to be the Iron abundance in solar units
       (anders and grevesse 1989: 2.62e-3)
     */

    if (getindex(&CoolZvalue[0], 0, ZBins - 1, &Zsol, &Zi) == -2)
    {
        return 0;
    }
    if (Zi == ZBins - 1)
    {
        t = 0;
    }
    else
    {
        t = (Zsol - CoolZvalue[Zi]) / (CoolZvalue[Zi + 1] - CoolZvalue[Zi]);
    }
    if (getindex(&CoolTvalue[0], 0, TBins - 1, &log_temp, &Ti) == -2)
    {
        return 0;
    }
    if (Ti == TBins - 1)
    {
        u = 0;
    }
    else
    {
        u = (log_temp - CoolTvalue[Ti]) / (CoolTvalue[Ti + 1] - CoolTvalue[Ti]);
    }

    if (t * u > 0)
    {
        x_1y_1 = CoolingTables[Zi + 1][Ti + 1];
    }
    else
    {
        x_1y_1 = 0;
    }

    if (t == 0)
    {
        x_1y = 0;
    }
    else
    {
        x_1y = CoolingTables[Zi + 1][Ti];
    }

    if (u == 0)
    {
        xy_1 = 0;
    }
    else
    {
        xy_1 = CoolingTables[Zi][Ti + 1];
    }

    return pow(10,
               (1 - t) * (1 - u) * CoolingTables[Zi][Ti] + t * (1 - u) * x_1y + t * u * x_1y_1 + (1 - t) * u * xy_1);
}
#endif // LT_METAL_COOLING

#endif // LT_METAL_COOLING_WAL

#endif // COOLING
