/**
 * @file
 *
 * This file is part of the N-body/SPH code OpenGadget3.
 *
 * @brief
 *
 * @author OpenGadget3-development team, in particular Umberto Maio.
 *
 * @copyright Copyright (c) 2026, OpenGadget3-development team lead by Klaus Dolag and Stefano Borgani
 * (see change logs). All rights reserved.
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License v3 or later.
 * See <https://www.gnu.org/licenses/>.
 */

/* UM: This file will be migrated */

/*=================== Cooling from metals (C+,Si+,O,Fe+) ===================*/

/*------------------------ [H/e-impact excitations] ------------------------*/

/*==========================================================================*/

/*  For the atomic data, see: Moore 1952 - Osterbrock 1989 -
    Hollenbach & McKee (HM) 1989 - Quinet et al. 1996 - Santoro & Shull 2005 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <boost/numeric/ublas/lu.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>

#include "../CodeBase/allvars.hpp"

#if defined(UM_METAL_COOLING) && (defined(LT_METAL_COOLING) || defined(LT_METAL_COOLING_WAL))

static void sq_system_solver(double *, double *, int, double *);
static void set_ndens_and_metallicity(float, int);
static double CII_cooling(double);
static double SiII_cooling(double);
static double OI_cooling(double);
static double OI_cooling_2level_approx(double);
static double FeII_cooling(double);
static double FeII_cooling_2level_approx(double);

/* abundances: */
static double H_ndens, He_ndens;
static double C_ndens, N_ndens, O_ndens, Mg_ndens, Si_ndens, Fe_ndens;
static double CII_ndens, OI_ndens, SiII_ndens, FeII_ndens;
static double dummy_average_atomic_mass = 40.;

static double tiny_value = 1.e-10;

/* interacting particles: */

#ifdef UM_H_MET_IMPACTS
static double n_field_H;
#endif // UM_H_MET_IMPACTS

#ifdef UM_e_MET_IMPACTS
static double n_field_e;
#endif // UM_e_MET_IMPACTS

/**
 * @brief Solves a square system of linear equations using LU decomposition.
 *
 * This function solves the system of linear equations represented as \[ A \mathbf{x} = \mathbf{b} \],
 * where \[ A \] is an \(n \times n\) matrix, \(\mathbf{b}\) is an \(n\)-dimensional vector, and \(\mathbf{x}\)
 * is the solution vector. The LU decomposition method from the Boost library is used for
 * solving the system.
 *
 * @param a_data Pointer to an array of size \(n \times n\) representing the matrix \(A\).
 * @param b_data Pointer to an array of size \(n\) representing the vector \(\mathbf{b}\).
 * @param n The dimension of the matrix and vector (both are \(n\)-dimensional).
 * @param sol Pointer to an array of size \(n\) where the solution vector \(\mathbf{x}\) will be stored.
 */
void sq_system_solver(double *a_data, double *b_data, int n, double sol[])
{
    boost::numeric::ublas::matrix<double> A(n, n);
    boost::numeric::ublas::vector<double> b(n);
    boost::numeric::ublas::permutation_matrix<std::size_t> pm(n);

    // Convert vector and matrix to ublas objects
    for (int i = 0; i < n; i++)
    {
        b(i) = b_data[i];
        for (int j = 0; j < n; j++)
        {
            A(i, j) = a_data[i * n + j];
        }
    }

    // Solve equation of type Ax=b via LU decomposition
    boost::numeric::ublas::lu_factorize(A, pm);
    boost::numeric::ublas::lu_substitute(A, pm, b);

    // Convert back
    for (int i = 0; i < n; i++)
    {
        sol[i] = b(i);
    }

    return;
}

/**
 * @brief Sets the number densities and metallicity of various elements.
 *
 * This function calculates the number densities of hydrogen, helium, and selected metals (carbon,
 * nitrogen, oxygen, magnesium, silicon, and iron) based on the total gas density (\p rho) and the
 * element mass fractions provided in the yields table. The function also initializes densities for
 * C II, Si II, Fe II, and O I based on the calculated number densities.
 *
 * The element mass fractions are used to compute the number densities, considering the atomic mass
 * of each element and the gas density. The densities are computed using predefined constants for
 * proton mass and other properties.
 *
 * @param rho The total gas density in units of mass per volume.
 * @param n The number of elements considered for mass fraction calculations.
 */
void set_ndens_and_metallicity(float rho, int n)
/*
 * note: FillEl collects the mass of all the elements present in the yields table
 *       but not explicitly tracked by the stellarevolution.
 *       Elements explicitly tracked are specified in metals.dat.
 */
{
    int k;
    double sum_Zs;

    /* LT_ADD */
    //  double FillEl_ndens;
    double Hmass, invmass;

    for (sum_Zs = 0, k = 0; k < n; k++)
    {
        sum_Zs += um_ZsPoint[k];
    }

    invmass = 1.0 / (double)um_mass;
    Hmass = (double)um_mass - sum_Zs;

    /* note :: FillEl gives the position of the Filling elements in the metal array. This position
     * is the same as that in metals.dat (look for the tag "Ej") */

    /* FillEl_ndens = rho * um_ZsPoint[FillEl] * invmass / dummy_average_atomic_mass; (comment by UM) */
    /* end of LT */

    /* compute ndens = rho * mass_fraction / mass_element */

    C_ndens = N_ndens = O_ndens = Mg_ndens = Si_ndens = Fe_ndens = 0.;
    H_ndens = rho * (Hmass * invmass) / (1. * PROTONMASS);
    He_ndens = rho * (um_ZsPoint[Hel] * invmass) / (4. * PROTONMASS);

    if (Carbon >= 0)
    {
        C_ndens = rho * (um_ZsPoint[Carbon] * invmass) / (12. * PROTONMASS);
    }

    if (Nitrogen >= 0)
    {
        N_ndens = rho * (um_ZsPoint[Nitrogen] * invmass) / (14. * PROTONMASS);
    }

    if (Oxygen >= 0)
    {
        O_ndens = rho * (um_ZsPoint[Oxygen] * invmass) / (16. * PROTONMASS);
    }

    if (Magnesium >= 0)
    {
        Mg_ndens = rho * (um_ZsPoint[Magnesium] * invmass) / (24. * PROTONMASS);
    }

    if (Silicon >= 0)
    {
        Si_ndens = rho * (um_ZsPoint[Silicon] * invmass) / (28. * PROTONMASS);
    }

    if (Iron >= 0)
    {
        Fe_ndens = rho * (um_ZsPoint[Iron] * invmass) / (56. * PROTONMASS);
    }

    /* Ansatz: C, Si, Fe ionized by the background radiation */

    CII_ndens = C_ndens;
    SiII_ndens = Si_ndens;
    FeII_ndens = Fe_ndens;
    OI_ndens = O_ndens;

    return;
}

/**
 * @brief Computes the cooling function for C II using a 2-level approximation.
 *
 * This function calculates the cooling rate of a gas due to C II transitions from
 * 2(J=3/2) to 1(J=1/2). The computation incorporates contributions from hydrogen and/or
 * electron impacts, depending on the defined preprocessor flags. The function uses transition
 * energies and Einstein coefficients from Santoro & Shull (2005), Wolfire et al. (1995), and
 * Hollenbach & McKee (1989).
 *
 * @param T_gas Temperature of the gas in Kelvin.
 *
 * @return The cooling rate in erg/cm³/s. Returns 0 if neither hydrogen nor electron impacts
 *         are considered.
 */
double CII_cooling(double T_gas)
{
#if (defined(UM_H_MET_IMPACTS)) || (defined(UM_e_MET_IMPACTS))

    double E21 = 1.259e-14;
    double A21 = 2.4e-6;
    int g2 = 4;
    int g1 = 2;
    double x2 = 0;

#ifdef UM_H_MET_IMPACTS
    double gammaH21 = 8e-10 * pow(T_gas / 100, 0.07);
    double gammaH12 = gammaH21 * g2 / g1 * exp(-E21 / (BOLTZMANN * T_gas));
#endif // UM_H_MET_IMPACTS
#ifdef UM_e_MET_IMPACTS
    double gammae21 = 2.8e-7 * pow(T_gas / 100, -0.5); /* HM89 */
    double gammae12 = gammae21 * g2 / g1 * exp(-E21 / (BOLTZMANN * T_gas));
#endif // UM_e_MET_IMPACTS

#if defined(UM_H_MET_IMPACTS) && !(defined(UM_e_MET_IMPACTS))
    x2 = gammaH12 / (gammaH21 + gammaH12 + A21 / n_field_H);
#endif // defined (UM_H_MET_IMPACTS) && !(defined (UM_e_MET_IMPACTS))
#if defined(UM_e_MET_IMPACTS) && !defined(UM_H_MET_IMPACTS)
    x2 = gammae12 / (gammae21 + gammae12 + A21 / n_field_e);
#endif // defined (UM_e_MET_IMPACTS) && !defined (UM_H_MET_IMPACTS)
#if defined(UM_H_MET_IMPACTS) && defined(UM_e_MET_IMPACTS)
    x2 = (gammaH12 + gammae12 * n_field_e / n_field_H) /
         (gammaH21 + gammaH12 + A21 / n_field_H + (gammae21 + gammae12) * n_field_e / n_field_H);
#endif // defined (UM_H_MET_IMPACTS) && defined (UM_e_MET_IMPACTS)

    return (A21 * E21 * x2 * CII_ndens); // by def. erg/cm3/s

#else // (defined (UM_H_MET_IMPACTS)) || (defined (UM_e_MET_IMPACTS))

    return 0;
#endif // (defined (UM_H_MET_IMPACTS)) || (defined (UM_e_MET_IMPACTS))
}

/**
 * @brief Computes the cooling function for Si II using a 2-level approximation.
 *
 * This function calculates the cooling rate of a gas due to Si II transitions from
 * 2(J=3/2) to 1(J=1/2). The computation incorporates contributions from hydrogen and/or
 * electron impacts, depending on the defined preprocessor flags. The function uses transition
 * energies and Einstein coefficients from Santoro & Shull (2005) and Hollenbach & McKee (1989).
 *
 * @param T_gas Temperature of the gas in Kelvin.
 *
 * @return The cooling rate in erg/cm³/s. Returns 0 if neither hydrogen nor electron impacts
 *         are considered.
 */
double SiII_cooling(double T_gas)
{
#if (defined(UM_H_MET_IMPACTS)) || (defined(UM_e_MET_IMPACTS))
    double E21 = 5.71e-14;
    double A21 = 2.1e-4;
    int g2 = 4;
    int g1 = 2;
    double x2 = 0;

#ifdef UM_H_MET_IMPACTS
    double gammaH21 = 8e-10 * pow(T_gas / 100, -0.07);
    double gammaH12 = gammaH21 * g2 / g1 * exp(-E21 / (BOLTZMANN * T_gas));
#endif // UM_H_MET_IMPACTS
#ifdef UM_e_MET_IMPACTS
    double gammae21 = 1.7e-6 * pow(T_gas / 100, -0.5);
    double gammae12 = gammae21 * g2 / g1 * exp(-E21 / (BOLTZMANN * T_gas));
#endif // v

#if defined(UM_H_MET_IMPACTS) && !(defined(UM_e_MET_IMPACTS))
    x2 = gammaH12 / (gammaH21 + gammaH12 + A21 / n_field_H);
#endif // defined (UM_H_MET_IMPACTS) && !(defined (UM_e_MET_IMPACTS))
#if defined(UM_e_MET_IMPACTS) && !(defined(UM_H_MET_IMPACTS))
    x2 = gammae12 / (gammae21 + gammae12 + A21 / n_field_e);
#endif // defined (UM_e_MET_IMPACTS) && !(defined (UM_H_MET_IMPACTS))
#if defined(UM_H_MET_IMPACTS) && defined(UM_e_MET_IMPACTS)
    x2 = (gammaH12 + gammae12 * n_field_e / n_field_H) /
         (gammaH21 + gammaH12 + A21 / n_field_H + (gammae21 + gammae12) * n_field_e / n_field_H);
#endif // defined (UM_H_MET_IMPACTS) && defined (UM_e_MET_IMPACTS)

    return (A21 * E21 * x2 * SiII_ndens); // by def. erg/cm3/s

#else // (defined (UM_H_MET_IMPACTS)) || (defined (UM_e_MET_IMPACTS))
    return 0;
#endif // (defined (UM_H_MET_IMPACTS)) || (defined (UM_e_MET_IMPACTS))
}

/**
 * @brief Computes the cooling function using a 2-level approximation.
 *
 * This function calculates the cooling rate of a gas using a simplified 2-level system
 * approximation. It considers the impacts of hydrogen and/or electron collisions based on
 * the defined preprocessor flags. The function uses transition energies and Einstein coefficients
 * from Hollenbach & McKee (1989) and Hollenbach & McKee (1989).
 *
 * @param T_gas Temperature of the gas in Kelvin.
 *
 * @return The cooling rate in erg/cm³/s. Returns 0 if neither hydrogen nor electron impacts
 *         are considered.
 *
 * @note cooling from OI :levels (1,2,3,4,5)=(3P2, 3P1, 3P0, 1D2, 1S0)
 */
double OI_cooling_2level_approx(double T_gas)
{
#if (defined(UM_H_MET_IMPACTS)) || (defined(UM_e_MET_IMPACTS))

    double E21 = 3.14e-14;
    double A21 = 8.9e-5; // A[i][j] from Osterbrock 1989
    int g1 = 5;
    int g2 = 3;
    double x2 = 0;

#ifdef UM_H_MET_IMPACTS
    double gammaH21 = 9.2e-11 * pow(T_gas / 100, 0.67);
    double gammaH12 = gammaH21 * g2 / g1 * exp(-E21 / (BOLTZMANN * T_gas));
#endif // UM_H_MET_IMPACTS
#ifdef UM_e_MET_IMPACTS
    double gammae21 = 1.4e-8; //~2.5e-10; (Bell '98): this is not crucial
    double gammae12 = gammae21 * g2 / g1 * exp(-E21 / (BOLTZMANN * T_gas));
#endif // UM_e_MET_IMPACTS

#if defined(UM_H_MET_IMPACTS) && !(defined(UM_e_MET_IMPACTS))
    x2 = gammaH12 / (gammaH21 + gammaH12 + A21 / n_field_H);
#endif // defined (UM_H_MET_IMPACTS) && !(defined (UM_e_MET_IMPACTS))
#if defined(UM_e_MET_IMPACTS) && !(defined(UM_H_MET_IMPACTS))
    x2 = gammae12 / (gammae21 + gammae12 + A21 / n_field_e);
#endif // defined (UM_e_MET_IMPACTS) && !(defined (UM_H_MET_IMPACTS))
#if defined(UM_H_MET_IMPACTS) && defined(UM_e_MET_IMPACTS)
    x2 = (gammaH12 + gammae12 * n_field_e / n_field_H) /
         (gammaH21 + gammaH12 + A21 / n_field_H + (gammae21 + gammae12) * n_field_e / n_field_H);
#endif //  defined (UM_H_MET_IMPACTS) && defined (UM_e_MET_IMPACTS)

    return (A21 * E21 * x2 * OI_ndens); // by def. erg/cm3/s

#else // (defined (UM_H_MET_IMPACTS)) || (defined (UM_e_MET_IMPACTS))
    return 0;
#endif // (defined (UM_H_MET_IMPACTS)) || (defined (UM_e_MET_IMPACTS))
}

/**
 * @brief Computes the cooling function for a given gas temperature.
 *
 * This function calculates the cooling rate of a gas using the population of energy levels
 * and their respective Einstein coefficients and transition energies. It accounts for impacts
 * from hydrogen and electron collisions if the corresponding preprocessor flags are set.
 *
 * The function builds a matrix based on the specified conditions and solves the system
 * of equations to determine the population of the energy levels. The cooling rate is then
 * computed using these populations and the transition parameters.
 *
 * @param T_gas Temperature of the gas in Kelvin.
 *
 * @return The cooling rate in erg/cm³/s. Returns 0 if neither hydrogen nor electron impacts
 *         are considered.
 */
double OI_cooling(double T_gas)
{
#if (defined(UM_H_MET_IMPACTS)) || (defined(UM_e_MET_IMPACTS))
    int n = 5; // number of levels
    int g[n]; // level degenarations
    double E[n][n]; // energy transitions
    double A[n][n]; // Einstein coefficients

#ifdef UM_H_MET_IMPACTS
    double gammaH[n][n]; // rates
#endif // UM_H_MET_IMPACTS
#ifdef UM_e_MET_IMPACTS
    double gammae[n][n]; // rates
#endif // UM_e_MET_IMPACTS
#if defined(UM_H_MET_IMPACTS) && defined(UM_e_MET_IMPACTS)
    double n_ratio = n_field_e / n_field_H;
#endif //  defined (UM_H_MET_IMPACTS) && defined(UM_e_MET_IMPACTS)

    double mtrx[n][n]; // population matrix with index 1-5
    double sum_column[n]; // sum in column in population matrix
    double b[n]; // known term vector
    double x_pop[n]; // fractional population of the levels

    /* set   b=( 1 0 0 0 0 )  */
    b[1] = 1.;
    for (int i = 2; i <= n; i++)
    {
        b[i] = 0.;
    }

    for (int i = 1; i <= n; i++)
    {
        for (int j = 1; j <= n; j++) /* initialization: */
        {
            E[i][j] = 0.;
            A[i][j] = 0.;
#ifdef UM_H_MET_IMPACTS
            gammaH[i][j] = 0.;
#endif // UM_H_MET_IMPACTS
#ifdef UM_e_MET_IMPACTS
            gammae[i][j] = 0.;
#endif // UM_e_MET_IMPACTS
            mtrx[i][j] = 0.;
        }
        g[i] = 0;
        sum_column[i] = 0;
    }

    /* atomic data... */
    E[2][1] = 3.14e-14;
    E[3][2] = 1.29e-14;
    E[4][3] = 3.143e-12;
    E[5][3] = 3.56e-12;
    E[3][1] = E[3][2] + E[2][1];
    E[4][2] = E[4][3] + E[3][2];
    E[4][1] = E[4][3] + E[3][1];
    E[5][2] = E[5][3] + E[3][2];
    E[5][1] = E[5][2] + E[2][1];
    E[5][4] = E[5][3] - E[4][3];
    A[2][1] = 8.9e-5; // A[i][j] from Osterbrock 1989
    A[3][1] = 1.0e-10;
    A[3][2] = 1.7e-5;
    A[4][1] = 6.3e-3;
    A[4][2] = 2.1e-3;
    A[4][3] = 7.3e-7;
    A[5][1] = 2.9e-4;
    A[5][2] = 7.3e-2;
    A[5][4] = 1.2;
    g[1] = 5; //    3P2, J = 2 (ground level)
    g[2] = 3; //    3P1, J = 1
    g[3] = 1; //    3P0, J = 0
    g[4] = 5; //    1D2, J = 2
    g[5] = 1; //    1S0, J = 0

#ifdef UM_H_MET_IMPACTS
    gammaH[2][1] = 9.2e-11 * pow(T_gas / 100, 0.67);
    gammaH[3][1] = 4.3e-11 * pow(T_gas / 100, 0.80);
    gammaH[3][2] = 1.1e-10 * pow(T_gas / 100, 0.44);
    gammaH[4][1] = gammaH[4][2] = gammaH[4][3] = 1.e-12; // less well determined
    gammaH[5][1] = gammaH[5][2] = gammaH[5][3] = 1.e-12; // assumed = 10^-12 cm3/s
    gammaH[1][2] = gammaH[2][1] * g[2] / g[1] * exp(-E[2][1] / (BOLTZMANN * T_gas));
    gammaH[1][3] = gammaH[3][1] * g[3] / g[1] * exp(-E[3][1] / (BOLTZMANN * T_gas));
    gammaH[2][3] = gammaH[3][2] * g[3] / g[2] * exp(-E[3][2] / (BOLTZMANN * T_gas));
    gammaH[1][4] = gammaH[4][1] * g[4] / g[1] * exp(-E[4][1] / (BOLTZMANN * T_gas));
    gammaH[2][4] = gammaH[4][2] * g[4] / g[2] * exp(-E[4][2] / (BOLTZMANN * T_gas));
    gammaH[3][4] = gammaH[4][3] * g[4] / g[3] * exp(-E[4][3] / (BOLTZMANN * T_gas));
    gammaH[1][5] = gammaH[5][1] * g[5] / g[1] * exp(-E[5][1] / (BOLTZMANN * T_gas));
    gammaH[2][5] = gammaH[5][2] * g[5] / g[2] * exp(-E[5][2] / (BOLTZMANN * T_gas));
    gammaH[3][5] = gammaH[5][3] * g[5] / g[3] * exp(-E[5][3] / (BOLTZMANN * T_gas));
#endif // UM_H_MET_IMPACTS

#ifdef UM_e_MET_IMPACTS
    gammae[2][1] = 1.4e-8;
    ; //~2.5e-10; (Bell '98): to change this is not crucial
    gammae[3][1] = 1.4e-8; //~2.6e-10; (Bell '98): to change this is not crucial
    gammae[3][2] = 5.0e-9; //~1.0e-11; (Bell '98): to change this is not crucial
    gammae[4][1] = gammae[4][2] = gammae[4][3] = 1.e-10; // less well determined
    gammae[5][1] = gammae[5][2] = gammae[5][3] = 1.e-10; // assumed ~1.e-10!!!!
    gammae[1][2] = gammae[2][1] * g[2] / g[1] * exp(-E[2][1] / (BOLTZMANN * T_gas));
    gammae[1][3] = gammae[3][1] * g[3] / g[1] * exp(-E[3][1] / (BOLTZMANN * T_gas));
    gammae[2][3] = gammae[3][2] * g[3] / g[2] * exp(-E[3][2] / (BOLTZMANN * T_gas));
    gammae[1][4] = gammae[4][1] * g[4] / g[1] * exp(-E[4][1] / (BOLTZMANN * T_gas));
    gammae[2][4] = gammae[4][2] * g[4] / g[2] * exp(-E[4][2] / (BOLTZMANN * T_gas));
    gammae[3][4] = gammae[4][3] * g[4] / g[3] * exp(-E[4][3] / (BOLTZMANN * T_gas));
    gammae[1][5] = gammae[5][1] * g[5] / g[1] * exp(-E[5][1] / (BOLTZMANN * T_gas));
    gammae[2][5] = gammae[5][2] * g[5] / g[2] * exp(-E[5][2] / (BOLTZMANN * T_gas));
    gammae[3][5] = gammae[5][3] * g[5] / g[3] * exp(-E[5][3] / (BOLTZMANN * T_gas));
#endif // UM_e_MET_IMPACTS

    /* build the matrix for the system to solve */
    for (int i = 1; i <= n; i++)
    {
        for (int j = 1; j <= n; j++)
        {
#if defined(UM_H_MET_IMPACTS) && !(defined(UM_e_MET_IMPACTS))
            mtrx[i][j] = gammaH[j][i] + A[j][i] / n_field_H;
#endif // defined (UM_H_MET_IMPACTS) && !(defined(UM_e_MET_IMPACTS))
#if defined(UM_e_MET_IMPACTS) && !(defined(UM_H_MET_IMPACTS))
            mtrx[i][j] = gammae[j][i] + A[j][i] / n_field_e;
#endif // defined (UM_e_MET_IMPACTS) && !(defined(UM_H_MET_IMPACTS))
#if defined(UM_e_MET_IMPACTS) && defined(UM_H_MET_IMPACTS)
            mtrx[i][j] = gammaH[j][i] + A[j][i] / n_field_H + gammae[j][i] * n_ratio;
#endif // defined (UM_e_MET_IMPACTS) && defined(UM_H_MET_IMPACTS)
        }
        VERBOSE_ALL(2, "1 - |  %g  %g  %g  %g  %g  | ", mtrx[i][1], mtrx[i][2], mtrx[i][3], mtrx[i][4], mtrx[i][5]);
    }

    // now, update the diagonal elements(= 0 so far) to -(sum in column)!
    for (int j = 1; j <= n; j++)
    {
        for (int i = 1; i <= n; i++)
        {
            sum_column[j] += mtrx[i][j];
        }
    }

    for (int i = 1; i <= n; i++)
    {
        mtrx[i][i] = -sum_column[i];
    }
    //...done!

    VERBOSE_ALL(2, "2 - |  %g  %g  %g  %g  %g  | ", mtrx[1][1], mtrx[1][2], mtrx[1][3], mtrx[1][4], mtrx[1][5]);
    VERBOSE_ALL(2, "2 - |  %g  %g  %g  %g  %g  | ", mtrx[2][1], mtrx[2][2], mtrx[2][3], mtrx[2][4], mtrx[2][5]);
    VERBOSE_ALL(2, "2 - |  %g  %g  %g  %g  %g  | ", mtrx[3][1], mtrx[3][2], mtrx[3][3], mtrx[3][4], mtrx[3][5]);
    VERBOSE_ALL(2, "2 - |  %g  %g  %g  %g  %g  | ", mtrx[4][1], mtrx[4][2], mtrx[4][3], mtrx[4][4], mtrx[4][5]);
    VERBOSE_ALL(2, "2 - |  %g  %g  %g  %g  %g  | ", mtrx[5][1], mtrx[5][2], mtrx[5][3], mtrx[5][4], mtrx[5][5]);
    VERBOSE_ALL(2, "\n");

    /*  now, set the condition "x1 + x2 + ... = 1",
       replacing one (the first) line of mtrx[][] with: ( 1  1 ... 1  1)
       because of the linear dependence: */
    mtrx[1][1] = 1.;
    mtrx[1][2] = 1.;
    mtrx[1][3] = 1.;
    mtrx[1][4] = 1.;
    mtrx[1][5] = 1.;

    VERBOSE_ALL(2, "3 - |  %g  %g  %g  %g  %g  | ", mtrx[1][1], mtrx[1][2], mtrx[1][3], mtrx[1][4], mtrx[1][5]);
    VERBOSE_ALL(2, "3 - |  %g  %g  %g  %g  %g  | ", mtrx[2][1], mtrx[2][2], mtrx[2][3], mtrx[2][4], mtrx[2][5]);
    VERBOSE_ALL(2, "3 - |  %g  %g  %g  %g  %g  | ", mtrx[3][1], mtrx[3][2], mtrx[3][3], mtrx[3][4], mtrx[3][5]);
    VERBOSE_ALL(2, "3 - |  %g  %g  %g  %g  %g  | ", mtrx[4][1], mtrx[4][2], mtrx[4][3], mtrx[4][4], mtrx[4][5]);
    VERBOSE_ALL(2, "3 - |  %g  %g  %g  %g  %g  | ", mtrx[5][1], mtrx[5][2], mtrx[5][3], mtrx[5][4], mtrx[5][5]);
    VERBOSE_ALL(2, "\n");

    /* find level populations */
    double mtrx_pass[n][n];
    double b_pass[n];
    double x_got[n];

    // matrix/vectors with index 0-4
    /*
       the following 3 assignements are necessary to be consistent with the routine sq_system_solver()
     */
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            mtrx_pass[i][j] = mtrx[i + 1][j + 1]; // 1. ok
        }
        b_pass[i] = b[i + 1]; // 2. ok
    }

    sq_system_solver((double *)mtrx_pass, b_pass, 5, x_got); //..find the solution

    for (int j = 0; j < n; j++)
    {
        x_pop[j + 1] = x_got[j]; // 3.
        VERBOSE_ALL(2, "   x_pop[%d] = %g ", j + 1, x_pop[j + 1]);
    }

    //  ENDRUN_MAIN("end here"); // try runs with 2-level approx
    /* compute the cooling fct */

    return (x_pop[2] * A[2][1] * E[2][1] + x_pop[3] * A[3][1] * E[3][1] + x_pop[3] * A[3][2] * E[3][2] +
            /* Santoro&Shull2005 do not include the following: */
            x_pop[4] * A[4][1] * E[4][1] + x_pop[4] * A[4][2] * E[4][2] + x_pop[4] * A[4][3] * E[4][3] +
            x_pop[5] * A[5][1] * E[5][1] + x_pop[5] * A[5][2] * E[5][2] + x_pop[5] * A[5][4] * E[5][4]) *
           OI_ndens; // erg/cm3/s
#else // #if (defined (UM_H_MET_IMPACTS)) || (defined (UM_e_MET_IMPACTS))
    return 0;
#endif // #if (defined (UM_H_MET_IMPACTS)) || (defined (UM_e_MET_IMPACTS))
}

/**
 * @brief Computes the approximate cooling rate for Fe II ions using a two-level model.
 *
 * This function calculates the Fe II cooling rate using an approximation based on two energy levels.
 * The cooling rate is computed depending on the presence of hydrogen or electron impacts, as indicated
 * by the preprocessor directives. The result is the cooling rate in erg/cm³/s.
 *
 * @param T_gas Temperature of the gas in Kelvin.
 * @return The approximate Fe II cooling rate in erg/cm³/s.
 *
 * @note cooling from Fe+ :level 6D(1,2,3,4,5)=6D(J=9/2,7/2,5/2,3/2,1/2)
 * data from Hollenbach&McKee 1989
 */
double FeII_cooling_2level_approx(double T_gas)
{
#if (defined(UM_H_MET_IMPACTS)) || (defined(UM_e_MET_IMPACTS))
    double E21 = 7.64e-14;
    double A21 = 2.13e-3;
    int g1 = 10;
    int g2 = 8;
    double x2 = 0;

#ifdef UM_H_MET_IMPACTS
    double gammaH21 = 9.5e-10;
    double gammaH12 = gammaH21 * g2 / g1 * exp(-E21 / (BOLTZMANN * T_gas));
#endif // UM_H_MET_IMPACTS
#ifdef UM_e_MET_IMPACTS
    double gammae21 = 1.8e-6 * pow(T_gas / 100, -0.5);
    double gammae12 = gammae21 * g2 / g1 * exp(-E21 / (BOLTZMANN * T_gas));
#endif // UM_e_MET_IMPACTS

#if defined(UM_H_MET_IMPACTS) && !(defined(UM_e_MET_IMPACTS))
    x2 = gammaH12 / (gammaH21 + gammaH12 + A21 / n_field_H);
#endif // defined (UM_H_MET_IMPACTS) && !(defined (UM_e_MET_IMPACTS))
#if defined(UM_e_MET_IMPACTS) && !(defined(UM_H_MET_IMPACTS))
    x2 = gammae12 / (gammae21 + gammae12 + A21 / n_field_e);
#endif // defined (UM_e_MET_IMPACTS) && !(defined (UM_H_MET_IMPACTS))
#if defined(UM_H_MET_IMPACTS) && defined(UM_e_MET_IMPACTS)
    x2 = (gammaH12 + gammae12 * n_field_e / n_field_H) /
         (gammaH21 + gammaH12 + A21 / n_field_H + (gammae21 + gammae12) * n_field_e / n_field_H);
#endif // defined (UM_H_MET_IMPACTS) && defined (UM_e_MET_IMPACTS)

    return (A21 * E21 * x2 * FeII_ndens); // by def. erg/cm3/s

#else // (defined (UM_H_MET_IMPACTS)) || (defined (UM_e_MET_IMPACTS))
    return 0;
#endif // (defined (UM_H_MET_IMPACTS)) || (defined (UM_e_MET_IMPACTS))
}

/**
 * @brief Computes the Fe II cooling rate using a detailed multi-level model.
 *
 * This function calculates the Fe II cooling rate using a detailed model considering multiple energy levels
 * and transitions. It accounts for both hydrogen and electron impacts if the relevant preprocessor directives
 * are defined. The result is the cooling rate in erg/cm³/s.
 *
 * @param T_gas Temperature of the gas in Kelvin.
 * @return The Fe II cooling rate in erg/cm³/s.
 */
double FeII_cooling(double T_gas)
{
#if (defined(UM_H_MET_IMPACTS)) || (defined(UM_e_MET_IMPACTS))
    int n = 5; // number of levels
    int g[n]; // level degenarations
    double E[n][n]; // energy transitions
    double A[n][n]; // Einstein coefficients

#ifdef UM_H_MET_IMPACTS
    double gammaH[n][n]; // rates
#endif // UM_H_MET_IMPACTS
#ifdef UM_e_MET_IMPACTS
    double gammae[n][n]; // rates
#endif // UM_e_MET_IMPACTS
#if defined(UM_H_MET_IMPACTS) && defined(UM_e_MET_IMPACTS)
    double n_ratio = n_field_e / n_field_H;
#endif // defined (UM_H_MET_IMPACTS) && defined(UM_e_MET_IMPACTS)

    double mtrx[n][n]; // population matrix with index 1-5 (consistent withthe data in Santoro&Shull 2005)
    double sum_column[n]; // sum in column in population matrix
    double b[n]; // known term vector
    double x_pop[n]; // fractional population of the levels

    /* set   b=( 1 0 0 0 0 )  */
    b[1] = 1.;
    for (int i = 2; i <= n; i++)
    {
        b[i] = 0.;
    }

    for (int i = 1; i <= n; i++)
    {
        for (int j = 1; j <= n; j++)
        {
            E[i][j] = 0.;
            A[i][j] = 0.;
#ifdef UM_H_MET_IMPACTS
            gammaH[i][j] = 0.;
#endif // UM_H_MET_IMPACTS
#ifdef UM_e_MET_IMPACTS
            gammae[i][j] = 0.;
#endif // UM_e_MET_IMPACTS
            mtrx[i][j] = 0.;
        }
        g[i] = 0;
        sum_column[i] = 0;
    }

    /* atomic data... */
    E[2][1] = 7.64e-14;
    E[3][2] = 5.62e-14;
    E[4][3] = 3.87e-14;
    E[5][4] = 2.27e-14;
    E[3][1] = E[3][2] + E[2][1];
    E[4][1] = E[4][3] + E[3][1];
    E[5][1] = E[5][4] + E[4][1];
    E[4][2] = E[4][3] + E[3][2];
    E[5][2] = E[5][4] + E[4][2];
    E[5][3] = E[5][4] + E[4][3];
    A[2][1] = 2.13e-3;
    A[3][2] = 1.57e-3;
    A[3][1] = 1.50e-9;
    A[4][3] = 7.18e-4;
    A[5][4] = 1.88e-4;
    g[1] = 10; //  J = 9/2    (ground level)
    g[2] = 8; //  J = 7/2
    g[3] = 6; //  J = 5/2
    g[4] = 4; //  J = 3/2
    g[5] = 2; //  J = 1/2

#ifdef UM_H_MET_IMPACTS
    gammaH[2][1] = 9.5e-10;
    gammaH[3][2] = 4.7e-10;
    gammaH[4][3] = 5.e-10; /* unknown datum: fiducial value */
    gammaH[5][4] = 5.e-10; /* unknown datum: fiducial value */
    gammaH[3][1] = 5.7e-10;
    gammaH[4][1] = 5.e-10; /* unknown datum: fiducial value */
    gammaH[5][1] = 5.e-10; /* unknown datum: fiducial value */
    gammaH[4][2] = 5.e-10; /* unknown datum: fiducial value */
    gammaH[5][2] = 5.e-10; /* unknown datum: fiducial value */
    gammaH[5][3] = 5.e-10; /* unknown datum: fiducial value */
    gammaH[1][2] = gammaH[2][1] * g[2] / g[1] * exp(-E[2][1] / (BOLTZMANN * T_gas));
    gammaH[2][3] = gammaH[3][2] * g[3] / g[2] * exp(-E[3][2] / (BOLTZMANN * T_gas));
    gammaH[3][4] = gammaH[4][3] * g[4] / g[3] * exp(-E[4][3] / (BOLTZMANN * T_gas));
    gammaH[4][5] = gammaH[5][4] * g[5] / g[4] * exp(-E[5][4] / (BOLTZMANN * T_gas));
    gammaH[1][3] = gammaH[3][1] * g[3] / g[1] * exp(-E[3][1] / (BOLTZMANN * T_gas));
    gammaH[1][4] = gammaH[4][1] * g[4] / g[1] * exp(-E[4][1] / (BOLTZMANN * T_gas));
    gammaH[1][5] = gammaH[5][1] * g[5] / g[1] * exp(-E[5][1] / (BOLTZMANN * T_gas));
    gammaH[2][4] = gammaH[4][2] * g[4] / g[2] * exp(-E[4][2] / (BOLTZMANN * T_gas));
    gammaH[2][5] = gammaH[5][2] * g[5] / g[2] * exp(-E[5][2] / (BOLTZMANN * T_gas));
    gammaH[3][5] = gammaH[5][3] * g[5] / g[3] * exp(-E[5][3] / (BOLTZMANN * T_gas));
#endif // UM_H_MET_IMPACTS

#ifdef UM_e_MET_IMPACTS
    gammae[2][1] = 1.8e-6 * pow(T_gas / 100, -0.5);
    gammae[3][2] = 8.7e-7 * pow(T_gas / 100, -0.5);
    gammae[4][3] = 1.e-5 * pow(T_gas, -0.5); /* unknown datum: fiducial value */
    gammae[5][4] = 1.e-5 * pow(T_gas, -0.5); /* unknown datum: fiducial value */
    gammae[3][1] = 1.8e-6 * pow(T_gas / 100, -0.5);
    gammae[4][1] = 1.e-5 * pow(T_gas, -0.5); /* unknown datum: fiducial value */
    gammae[5][1] = 1.e-5 * pow(T_gas, -0.5); /* unknown datum: fiducial value */
    gammae[4][2] = 1.e-5 * pow(T_gas, -0.5); /* unknown datum: fiducial value */
    gammae[5][2] = 1.e-5 * pow(T_gas, -0.5); /* unknown datum: fiducial value */
    gammae[5][3] = 1.e-5 * pow(T_gas, -0.5); /* unknown datum: fiducial value */
    gammae[1][2] = gammae[2][1] * g[2] / g[1] * exp(-E[2][1] / (BOLTZMANN * T_gas));
    gammae[2][3] = gammae[3][2] * g[3] / g[2] * exp(-E[3][2] / (BOLTZMANN * T_gas));
    gammae[3][4] = gammae[4][3] * g[4] / g[3] * exp(-E[4][3] / (BOLTZMANN * T_gas));
    gammae[4][5] = gammae[5][4] * g[5] / g[4] * exp(-E[5][4] / (BOLTZMANN * T_gas));
    gammae[1][3] = gammae[3][1] * g[3] / g[1] * exp(-E[3][1] / (BOLTZMANN * T_gas));
    gammae[1][4] = gammae[4][1] * g[4] / g[1] * exp(-E[4][1] / (BOLTZMANN * T_gas));
    gammae[1][5] = gammae[5][1] * g[5] / g[1] * exp(-E[5][1] / (BOLTZMANN * T_gas));
    gammae[2][4] = gammae[4][2] * g[4] / g[2] * exp(-E[4][2] / (BOLTZMANN * T_gas));
    gammae[2][5] = gammae[5][2] * g[5] / g[2] * exp(-E[5][2] / (BOLTZMANN * T_gas));
    gammae[3][5] = gammae[5][3] * g[5] / g[3] * exp(-E[5][3] / (BOLTZMANN * T_gas));
#endif // UM_e_MET_IMPACTS

    /* build the matrix for the system to solve */
    for (int i = 1; i <= n; i++)
    {
        for (int j = 1; j <= n; j++)
        {
#if defined(UM_H_MET_IMPACTS) && !(defined(UM_e_MET_IMPACTS))
            mtrx[i][j] = gammaH[j][i] + A[j][i] / n_field_H;
#endif // defined (UM_H_MET_IMPACTS) && !(defined(UM_e_MET_IMPACTS))
#if defined(UM_e_MET_IMPACTS) && !(defined(UM_H_MET_IMPACTS))
            mtrx[i][j] = gammae[j][i] + A[j][i] / n_field_e;
#endif // defined (UM_e_MET_IMPACTS) && !(defined(UM_H_MET_IMPACTS))
#if defined(UM_e_MET_IMPACTS) && defined(UM_H_MET_IMPACTS)
            mtrx[i][j] = gammaH[j][i] + A[j][i] / n_field_H + gammae[j][i] * n_ratio;
#endif // defined (UM_e_MET_IMPACTS) && defined(UM_H_MET_IMPACTS)
        }
    }

    // now, update the diagonal elements(= 0 so far) to -(sum in column)!
    for (int j = 1; j <= n; j++)
    {
        for (int i = 1; i <= n; i++)
        {
            sum_column[j] += mtrx[i][j];
        }
    }

    for (int i = 1; i <= n; i++)
    {
        mtrx[i][i] = -sum_column[i];
    }
    //...done!

    /*  now, set the condition "x1 + x2 + ... = 1",
       replacing one (the first) line of mtrx[][] with: ( 1  1 ... 1  1)
       bocouse of the linear dependence: */

    for (int j = 1; j <= n; j++)
    {
        mtrx[1][j] = 1.;
    }

    /* find level populations */

    double mtrx_pass[n][n];
    double b_pass[n];
    double x_got[n];

    // matrix/vectors with index 0-4
    /*
       the following 3 assignements are necessary to be consistent with
       the routine sq_system_solver()
     */
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            mtrx_pass[i][j] = mtrx[i + 1][j + 1]; // 1.
        }
        b_pass[i] = b[i + 1]; // 2.
    }

    sq_system_solver((double *)mtrx_pass, b_pass, 5, x_got); //..find the solution

    for (int j = 0; j < n; j++)
    {
        x_pop[j + 1] = x_got[j]; // 3.
    }

    /* compute cooling fct */

    return (x_pop[2] * A[2][1] * E[2][1] + x_pop[3] * A[3][2] * E[3][2] + x_pop[3] * A[3][1] * E[3][1] +
            x_pop[4] * A[4][3] * E[4][3] + x_pop[5] * A[5][4] * E[5][4]) *
           FeII_ndens; // erg/cm3/s
#else // (defined (UM_H_MET_IMPACTS)) || (defined (UM_e_MET_IMPACTS))
    return 0;
#endif // (defined (UM_H_MET_IMPACTS)) || (defined (UM_e_MET_IMPACTS))
}

/**
 * @brief Computes the metal line cooling rate for a gas.
 *
 * This function calculates the cooling rate  (in erg/cm3/s) due to metal lines based on the gas temperature,
 * density, and optionally, electron density. The cooling processes are enabled conditionally
 * based on preprocessor directives. The result is the sum of cooling contributions from various
 * metal lines.
 *
 * @param T_gas Temperature of the gas in Kelvin.
 * @param rho Density of the gas.
 * @param n Metallicity of the gas.
 * @param e_ndens Electron density (optional, used if UM_e_MET_IMPACTS is defined).
 *
 * @return The total metal line cooling rate.
 */
#ifdef UM_e_MET_IMPACTS
double um_metal_line_cooling(double T_gas, double rho, int n, double e_ndens)
#else // UM_e_MET_IMPACTS
double um_metal_line_cooling(double T_gas, double rho, int n)
#endif // UM_e_MET_IMPACTS
{
#if !(defined(UM_H_MET_IMPACTS)) && !(defined(UM_e_MET_IMPACTS))
    return 0;
#endif // !(defined (UM_H_MET_IMPACTS)) && !(defined (UM_e_MET_IMPACTS))

    /* initialization */
    double CoolSum = 0.;
    CII_ndens = 0.;
    SiII_ndens = 0.;
    FeII_ndens = 0.;
    OI_ndens = 0.;

    set_ndens_and_metallicity(rho, n);
    /* now, set the interacting particles: */
#ifdef UM_H_MET_IMPACTS
    n_field_H = H_ndens;
#endif // UM_H_MET_IMPACTS

#ifdef UM_e_MET_IMPACTS
    n_field_e = e_ndens;
#endif // UM_e_MET_IMPACTS

    if (T_gas < T_SUP_INTERPOL_LIMIT)
    {
#ifndef UM_DISABLE_CII_COOLING
        if (CII_ndens > tiny_value)
        {
            CoolSum += CII_cooling(T_gas);
        }
#endif // UM_DISABLE_CII_COOLING
#ifndef UM_DISABLE_SiII_COOLING
        if (SiII_ndens > tiny_value)
        {
            CoolSum += SiII_cooling(T_gas);
        }
#endif // UM_DISABLE_SiII_COOLING
#ifndef UM_DISABLE_OI_COOLING
        if (OI_ndens > tiny_value)
        {
            //      CoolSum += OI_cooling(T_gas);
            CoolSum += OI_cooling_2level_approx(T_gas);
            /* uses only first two levels with well known rates */
        }
#endif // UM_DISABLE_OI_COOLING
#ifndef UM_DISABLE_FeII_COOLING
        if (FeII_ndens > tiny_value)
        {
            /*      CoolSum += FeII_cooling(T_gas); */
            CoolSum += FeII_cooling_2level_approx(T_gas);
            /* uses only first two levels with well known rates */
        }
#endif // UM_DISABLE_FeII_COOLING
    }
    return CoolSum;
}

#endif // defined (UM_METAL_COOLING) && (defined (LT_METAL_COOLING) || defined(LT_METAL_COOLING_WAL))
