/**
 * @file
 *
 * This file is part of the N-body/SPH code OpenGadget3.
 *
 * @brief Darkeneergy functions for time dependent w.
 *
 * @author OpenGadget3-development team, in particular Klaus Dolag.
 *
 * @copyright Copyright (c) 2026, OpenGadget3-development team lead by Klaus Dolag and Stefano Borgani
 * (see change logs). All rights reserved.
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License v3 or later.
 * See <https://www.gnu.org/licenses/>.
 */

/* Using Dark Energy instead of a Cosmological constant can be archived by
 * replacing Lambda by Lambda * a^(-3*(1+w)) in the Hubble function.
 * So easy to see that w = -1 gives back a standard Cosmological Constant !
 * Also w = -1/3 gives Lambda / a^2 which then cancel within the Hubble
 * function and is then equal to the dynamics of a universe with Lambda = 0 !
 *
 * For a time varying w once has to replace Lambda * a^(-3*(1+w)) by
 * Lambda * exp(Integral(a,1,3*(1+w)/a))
 *
 * Once can now also read in colums for the change of the gravitational
 * constant and the correction by this to the hubble function.
 *
 * Additional once can read also an "external" hubble function from a
 * column of the dark energy file.
 *
 * Note that the first column is 'z+1' !
 *
 * Dark Energy does not alter the powerspectrum of initial conditions.
 * To get the same cluster for various values or functions of w, once
 * has do assign a new redshift to the initial cond. to match the
 * linear growth factors, so g(z=0)/g(z_ini) == g_w(z=0)/g_w(z_ini^new)
 * Also the initial velocities field has to be scaled by
 * (Hubble_w(z_ini^new)*Omega_w(z_ini^new)^0.6)/(Hubble(z_ini)*Omega(z_ini)^0.6)
 * where _w means the according functions including the terms for
 * Dark Energy.
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "../CodeBase/allvars.hpp"
#include "../CodeBase/proto.hpp"

#ifdef DARKENERGY
#ifdef TIMEDEPDE

#define ANZ_W_A_IN 4000
#define ANZ_W_A 50000

static double atab[ANZ_W_A_IN];
static double wtab[ANZ_W_A_IN];
static double intwtab[ANZ_W_A + 1];
static double intwatab[ANZ_W_A + 1];

#ifdef TIMEDEPGRAV
static double dHtab[ANZ_W_A_IN];
static double dGtab[ANZ_W_A_IN];
static double intdHtab[ANZ_W_A + 1];
static double intdGtab[ANZ_W_A + 1];
#endif // TIMEDEPGRAV

#ifdef EXTERNALHUBBLE
static double Htab[ANZ_W_A_IN];
static double intHtab[ANZ_W_A + 1];
#endif // EXTERNALHUBBLE

/**
 * @brief Initializes the time-dependent w.
 *
 * This function initializes the time-dependent w by reading data from a file specified by `All.DarkEnergyFile`.
 * It sets up a table with Exp(-Integral_a^1 (q+w(a'))/a' da' needed in the hubble function for time dependent w.
 */
void fwa_init(void)
{
    int count = 0, i;
    char buf[200], buf1[200], buf2[200];

#ifdef TIMEDEPGRAV
    char buf3[200], buf4[200];
#endif // TIMEDEPGRAV

#ifdef EXTERNALHUBBLE
    char buf5[200];
#endif // EXTERNALHUBBLE

    FILE *fd;
    MyFloat a_first, w_first, a, w, sum;

    VERBOSE(2, "initialize time dependent w ...\n");

    if ((fd = fopen(All.DarkEnergyFile, "r")))
    {
        VERBOSE(2, "\nreading w of a from file `%s'\n", All.DarkEnergyFile);
        atab[0] = -1.0; /* we have to extrapolate wtab[0] later ! */
        count = 1;
        while (!feof(fd) && count < ANZ_W_A_IN)
        {
            if (fgets(buf, 200, fd))
            {
#ifdef TIMEDEPGRAV
#ifdef EXTERNALHUBBLE
                if (sscanf(buf, "%s%s%s%s%s", buf1, buf2, buf3, buf4, buf5) < 5)
#else // EXTERNALHUBBLE
                if (sscanf(buf, "%s%s%s%s", buf1, buf2, buf3, buf4) < 4)
#endif // EXTERNALHUBBLE
#else // TIMEDEPGRAV
#ifdef EXTERNALHUBBLE
                if (sscanf(buf, "%s%s%s", buf1, buf2, buf5) < 3)
#else // EXTERNALHUBBLE
                if (sscanf(buf, "%s%s", buf1, buf2) < 2)
#endif // EXTERNALHUBBLE
#endif // TIMEDEPGRAV
                {
                    ENDRUN_MAIN("Wrong syntax in file `%s', line %d\n", All.DarkEnergyFile, count);
                }
                a = 1. / atof(buf1);
                if (a == 0.0 && count == 1)
                {
                    count--; /* w(0) present in file, so fill the first element ! */
                }
                atab[count] = a;
                wtab[count] = atof(buf2);
#ifdef TIMEDEPGRAV
                dHtab[count] = atof(buf3);
                dGtab[count] = atof(buf4);
#endif // TIMEDEPGRAV
#ifdef EXTERNALHUBBLE
                Htab[count] = atof(buf5);
#endif // EXTERNALHUBBLE
                count++;
            }
        }
        fclose(fd);

        ENDRUN_IF(count >= ANZ_W_A_IN + 1,
                  "File `%s' contains to many datapoints, increase ANZ_W_A_IN (count=%d,tab=%d)!\n", All.DarkEnergyFile,
                  count, ANZ_W_A_IN);

        ENDRUN_IF(count <= 2, "File `%s' has to less Data Points (%d) !\n", All.DarkEnergyFile, count);

        if (atab[0] < 0.) /* We still have to extrapolate w to a = 0 (w[0]) */
        {
            atab[0] = 0.;
            wtab[0] = wtab[1] - (wtab[2] - wtab[1]) / (atab[2] - atab[1]) * (atab[1] - atab[0]);
#ifdef TIMEDEPGRAV
            dHtab[0] = dHtab[1] - (dHtab[2] - dHtab[1]) / (atab[2] - atab[1]) * (atab[1] - atab[0]);
            dGtab[0] = dGtab[1] - (dGtab[2] - dGtab[1]) / (atab[2] - atab[1]) * (atab[1] - atab[0]);
#endif // TIMEDEPGRAV
#ifdef EXTERNALHUBBLE
            Htab[0] = Htab[1] - (Htab[2] - Htab[1]) / (atab[2] - atab[1]) * (atab[1] - atab[0]);
#endif // EXTERNALHUBBLE
        }

        /* Setp back if tables go bejond a=1 */
        while (atab[count - 1] > 1.0)
        {
            count--;
        }

        /* Calculate w(1) if needed */
        if (atab[count - 1] < 1.)
        {
            atab[count] = 1.0;
            wtab[count] = wtab[count - 1] + (wtab[count - 1] - wtab[count - 2]) / (atab[count - 1] - atab[count - 2]) *
                                                (1. - atab[count - 1]);
#ifdef TIMEDEPGRAV
            dHtab[count] = dHtab[count - 1] + (dHtab[count - 1] - dHtab[count - 2]) /
                                                  (atab[count - 1] - atab[count - 2]) * (1. - atab[count - 1]);
            dGtab[count] = dGtab[count - 1] + (dGtab[count - 1] - dGtab[count - 2]) /
                                                  (atab[count - 1] - atab[count - 2]) * (1. - atab[count - 1]);
#endif // TIMEDEPGRAV
#ifdef EXTERNALHUBBLE
            Htab[count] = Htab[count - 1] + (Htab[count - 1] - Htab[count - 2]) / (atab[count - 1] - atab[count - 2]) *
                                                (1. - atab[count - 1]);
#endif // EXTERNALHUBBLE
            /*            if(ThisTask ==0)
                          {
                            printf("%d %f %f %f\n",count,atab[count-2],atab[count-1],atab[count]);
                            printf("%d %f %f %f\n",count,wtab[count-2],wtab[count-1],wtab[count]);
                          }*/
            count++;
        }

        /* Now calculated the integral (starting from last to first to save Time !
         * Explicit asume that a[0]=0. and a[count-1]=1. , which is enshured by
         * the loading precedure !                                                  */

        /* Set todays values in the tables */
        intwtab[ANZ_W_A] = All.OmegaLambda;
        intwatab[ANZ_W_A] = wtab[count - 1];
#ifdef TIMEDEPGRAV
        intdHtab[ANZ_W_A] = dHtab[count - 1];
        intdGtab[ANZ_W_A] = dGtab[count - 1];
#endif // TIMEDEPGRAV
#ifdef EXTERNALHUBBLE
        intHtab[ANZ_W_A] = Htab[count - 1];
#endif // EXTERNALHUBBLE

        /* Place count on last entry in table */
        count--;

        a_first = atab[count]; /* Startinv value should be 1.0 ! */
        w_first = wtab[count]; /* Starting value from table ! */
        sum = 0.0; /* set int to 0.0 */
        for (i = ANZ_W_A - 1; i >= 1; i--)
        {
            a = (double)i / (double)ANZ_W_A;
            if (count > 1) /* Still inside the table */
            {
                while (atab[count - 1] > a && count > 0)
                {
                    sum += 0.5 * ((1. + w_first) / a_first + (1. + wtab[count - 1]) / atab[count - 1]) *
                           (a_first - atab[count - 1]);
                    count--;
                    a_first = atab[count];
                    w_first = wtab[count];
                }
                w = w_first - (wtab[count] - wtab[count - 1]) / (atab[count] - atab[count - 1]) * (a_first - a);
                sum += 0.5 * ((1. + w_first) / a_first + (1. + w) / a) * (a_first - a);
                w_first = w;
                a_first = a;
            }
            else
            {
                w = w_first - (wtab[count] - wtab[count - 1]) / (atab[count] - atab[count - 1]) * (a_first - a);
                sum += 0.5 * ((1. + w_first) / a_first + (1. + w) / a) * (a_first - a);
                w_first = w;
                a_first = a;
            }
            intwtab[i] = All.OmegaLambda * exp(3. * sum);
            intwatab[i] = wtab[count - 1] +
                          (wtab[count] - wtab[count - 1]) / (atab[count] - atab[count - 1]) * (a - atab[count - 1]);
#ifdef TIMEDEPGRAV
            intdHtab[i] = dHtab[count - 1] +
                          (dHtab[count] - dHtab[count - 1]) / (atab[count] - atab[count - 1]) * (a - atab[count - 1]);
            intdGtab[i] = dGtab[count - 1] +
                          (dGtab[count] - dGtab[count - 1]) / (atab[count] - atab[count - 1]) * (a - atab[count - 1]);
#endif // TIMEDEPGRAV
#ifdef EXTERNALHUBBLE
            intHtab[i] = Htab[count - 1] +
                         (Htab[count] - Htab[count - 1]) / (atab[count] - atab[count - 1]) * (a - atab[count - 1]);
#endif // EXTERNALHUBBLE
        }
        /* artificially define value for a=0 */
        intwtab[0] = intwtab[1];
        intwatab[0] = intwatab[1];
#ifdef TIMEDEPGRAV
        intdHtab[0] = intdHtab[1];
        intdGtab[0] = intdGtab[1];
#endif // TIMEDEPGRAV
#ifdef EXTERNALHUBBLE
        intHtab[0] = intHtab[1];
#endif // EXTERNALHUBBLE
    }
    else
    {
        ENDRUN_MAIN("File `%s' not found !\n", All.DarkEnergyFile);
    }
    VERBOSE(2, "Integrating w(a) finisched.\n");
}

/**
 * @brief Returns the integral of w(a) term for the actual time.
 *
 * This function calculates the integral of w(a) term for the actual time based on the scale factor 'a'.
 * It is needed in the calculation of the hubble function.
 *
 * @param a The scale factor.
 * @return The integral of w(a) term for the actual time.
 */
double INLINE_FUNC fwa(double a)
{
    double fwa = 0.0;

    int ai = a * ANZ_W_A;
    if (ai >= ANZ_W_A)
    {
        fwa = intwtab[ANZ_W_A];
    }
    else
    {
        fwa = intwtab[ai] + (intwtab[ai + 1] - intwtab[ai]) * (a * (double)ANZ_W_A - (double)ai);
    }
    /*   if(ThisTask==0) printf("%f %f %f %f\n",a,intwtab[ai],fwa,intwtab[ai+1]);*/
    return (fwa);
}
#endif // TIMEDEPDE

/**
 * @brief Returns the dark energy parameter as a function of the scale factor.
 *
 * This function returns the dark energy parameter as a function of the scale factor 'a'.
 * It is only needed for comoving integration.
 *
 * @param a The scale factor.
 * @return The dark energy parameter.
 */
double DarkEnergy_a(double a)
{
#ifdef TIMEDEPDE
    return fwa(a);
#else // TIMEDEPDE
    return (All.OmegaLambda * pow(a, -3. * (1 + All.DarkEnergyParam)));
#endif // TIMEDEPDE
}

/**
 * @brief Returns the dark energy parameter as a function of time.
 *
 * This function returns the dark energy parameter, which is constant over time and defined by `All.DarkEnergyParam`.
 * It is only needed for physical integration.
 *
 * @param Time The time parameter.
 * @return The dark energy parameter.
 */

/*
double DarkEnergy_t(double Time)
{
    return All.DarkEnergyParam;
}
*/

#ifdef TIMEDEPDE
/**
 * @brief Caclulate the interpolated equation of state parameter.
 *
 * This function calculates the interpolated equation of state parameter based on the scale factor 'a'.
 * This parameter is only used for information in the log files.
 *
 * @param a The scale factor.
 * @return The interpolated equation of state parameter.
 */
double INLINE_FUNC get_wa(double a)
{
    double fw = 0.0;

    int ai = a * ANZ_W_A;
    if (ai >= ANZ_W_A)
    {
        fw = intwatab[ANZ_W_A];
    }
    else
    {
        fw = intwatab[ai] + (intwatab[ai + 1] - intwatab[ai]) * (a * (double)ANZ_W_A - (double)ai);
    }
    return (fw);
}

#ifdef TIMEDEPGRAV
/**
 * @brief Calculate the interpolated correction for the Hubble function.
 *
 * This function calculates the interpolated correction for the Hubble function based on the scale factor 'a'.
 *
 * @param a The scale factor.
 * @return The interpolated correction for the Hubble function.
 */
double INLINE_FUNC dHfak(double a)
{
    double fdH = 0.0;

    int ai = a * ANZ_W_A;
    if (ai >= ANZ_W_A)
    {
        fdH = intdHtab[ANZ_W_A];
    }
    else
    {
        fdH = intdHtab[ai] + (intdHtab[ai + 1] - intdHtab[ai]) * (a * (double)ANZ_W_A - (double)ai);
    }
    return (fdH);
}

/**
 * @brief Calculate the interpolated correction for the gravitational constant.
 *
 * This function calculates the interpolated correction for the gravitational constant based on the scale factor 'a'.
 *
 * @param a The scale factor.
 * @return The interpolated correction for the gravitational constant.
 */
double INLINE_FUNC dGfak(double a)
{
    double fdG = 0.0;

    int ai = a * ANZ_W_A;
    if (ai >= ANZ_W_A)
    {
        fdG = intdGtab[ANZ_W_A];
    }
    else
    {
        fdG = intdGtab[ai] + (intdGtab[ai + 1] - intdGtab[ai]) * (a * (double)ANZ_W_A - (double)ai);
    }
    return (fdG);
}
#endif // TIMEDEPGRAV

#ifdef EXTERNALHUBBLE
/**
 * @brief Calculate the interpolated correction for the Hubble function if EXTERNALHUBBLE.
 *
 * This function calculates the interpolated correction for the Hubble function based on the scale factor 'a'.
 *
 * @param a The scale factor.
 * @return The interpolated correction for the Hubble function.
 */
double INLINE_FUNC hubble_function_external(double a)
{
    double H = 0.0;

    int ai = a * ANZ_W_A;
    if (ai >= ANZ_W_A)
    {
        H = intHtab[ANZ_W_A];
    }
    else
    {
        H = intHtab[ai] + (intHtab[ai + 1] - intHtab[ai]) * (a * (double)ANZ_W_A - (double)ai);
    }
    return (H);
}
#endif // EXTERNALHUBBLE
#endif // TIMEDEPDE
#endif // DARKENERGY

/**
 * @brief Calculates the Hubble function.
 *
 * This function calculates the Hubble function for a given scale factor 'a'.
 *
 * @param a The scale factor.
 * @return The value of the Hubble function.
 */
double INLINE_FUNC hubble_function(double a)
{
    double hubble_a;

#ifdef EXTERNALHUBBLE
    hubble_a = hubble_function_external(a);
#else // EXTERNALHUBBLE
    hubble_a = All.Omega0 / (a * a * a) +
               OMEGAK / (a * a)
#ifdef INCLUDE_RADIATION
               /*Note OMEGAR is defined to be 0 if INCLUDE_RADIATION is not on */
               + OMEGAR / (a * a * a * a)
#endif // INCLUDE_RADIATION
#ifdef DARKENERGY
               + DarkEnergy_a(a);
#else // DARKENERGY
               + All.OmegaLambda;
#endif // DARKENERGY
    hubble_a = All.Hubble * sqrt(hubble_a);
#endif // EXTERNALHUBBLE
#ifdef TIMEDEPGRAV
    hubble_a *= dHfak(a);
#endif // TIMEDEPGRAV
    return (hubble_a);
}
