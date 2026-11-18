/**
 * @file
 *
 * This file is part of the N-body/SPH code OpenGadget3.
 *
 * @brief
 *
 * @author OpenGadget3-development team, in particluar Luca Tornatore.
 *
 * @copyright Copyright (c) 2025, OpenGadget3-development team lead by Klaus Dolag and Stefano Borgani
 * (see change logs). All rights reserved.
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License v3 or later.
 * See <https://www.gnu.org/licenses/>.
 */

#include <stdlib.h>
#include <stdio.h>
#include <cmath>
#include <boost/math/quadrature/gauss_kronrod.hpp>

#ifndef LIBIMF_VARS_H
#define LIBIMF_VARS_H

/* /------------------------------------------------\
 * |                                                |
 * | IMF definition                                 |
 * |                                                |
 * | In this section we actually define what an IMF |
 * | is in this library.                            |
 * \------------------------------------------------/ */

#define LIMF_MAX(a, b) (((a) > (b)) ? (a) : (b))
#define LIMF_MIN(a, b) (((a) < (b)) ? (a) : (b))

typedef enum
{
    power_law,
    whatever
} IMF_SPEC;
extern const char *IMF_Spec_Labels[];
extern IMF_SPEC IMF_Spec;

#define IMF_NSPEC 9 /* defines how many fields must be specified in IMFs' format file for each IMF */
#define INC_BH 0
#define EXC_BH 1

typedef struct
{
    IMF_SPEC type;
    char *name;
    double (*IMFfunc_byMass)(double, void *);
    double (*IMFfunc_byNum)(double, void *);
    double (*IMFfunc_byEgy)(double, void *);
    double (*getp)(int, double *, double);

    /* Normalization constants; A contains the normalizations
     * for each segment of a general IMF
     */
    double Atot, *A;

    /* [Mm MU] is the mass range over which the IMF is defined */
    double MU, Mm;

    /* inf_lifetime = lifetime(MU)
     * sup_lifetime = lifetime(Mm)
     */
    double inf_lifetime, sup_lifetime;

    /* this array contains the parameters for the IMF */
    double *Params;

    /* This structure contains the slopes of a multi powerlaw IMF */
    struct
    {
        double *masses, *slopes;
    } Slopes;

    /* This strcture contains the kinetic energies of Sn as
     * a function of mass
     */
    struct
    {
        double *masses, *ekin;
    } EKin;

    /* This structure defines in which interval the stars will not
     * directly end in BH without becoming SN beforehand
     */
    struct
    {
        double *sup, *inf, *list;
    } notBH_ranges;

    int NSlopes, /* number of segment in the IMF */
        NParams, /* number of needed parameters  */
        YSet, /* the yields set */
        NEKin, /* how many bins for the kinetic energy */
        N_notBH_ranges; /* how many intervals for non-directly-BH */

    /* set if the IMF is timedependent */
    int timedep;
} IMF_Type;

extern IMF_Type *IMFs, IMFu;

/* How many IMFs have been defined */
extern int IMFs_dim;

extern int Nof_TimeDep_IMF;

/**
 * @brief Minimal drop-in replacement for GSL's gsl_function. Keeps the existing
 * `limf_F.function = ...; limf_F.params = ...;` call sites in libimf.c working
 * unchanged while the integration itself is done via limf_integration_qag() below.
 */
struct limf_integrand
{
    double (*function)(double, void *);
    void *params;
};

/**
 * @brief Adaptive Gauss-Kronrod integration (15-point rule, matching the GSL
 * key=1 rule this replaces) with the same call signature (minus the now-unneeded
 * workspace/limit argument) and 0-on-success return convention as GSL's
 * gsl_integration_qag(), backed by Boost.Math instead of GSL.
 */
inline int limf_integration_qag(const limf_integrand *F, double a, double b, double epsabs, double epsrel,
                                double *result, double *abserr)
{
    auto f = [F](double x) { return F->function(x, F->params); };
    *result = boost::math::quadrature::gauss_kronrod<double, 15>::integrate(f, a, b, 15, epsrel, abserr);
    return (*abserr > epsabs + epsrel * std::fabs(*result)) ? 1 : 0;
}

#define LIBIMF_ERR_IMF_FILE (-10)
#define LIBIMF_ERR_IMF_FILE_FORMAT (-11)
#define LIBIMF_ERR_IMF_ALLOCATE (-12)
#define LIBIMF_ERR_INTEGRATION (-100)

int read_imfs(char *);

double Renormalize_IMF(IMF_Type *, double, double, int);

void print_IMF(int, char *);
void write_IMF_info(int, FILE *);
void write_SF_info(int, FILE *);

double IntegrateIMF_byNum(double, double, IMF_Type *, int);
double IntegrateIMF_byMass(double, double, IMF_Type *, int);
double IntegrateIMF_byEgy(double, double, IMF_Type *);

double IMFevaluate_byMass_powerlaw(double, void *);
double IMFevaluate_byNum_powerlaw(double, void *);
double IMFevaluate_byEgy_powerlaw(double, void *);

double IMFevaluate_byMass_Larson(double, void *);
double IMFevaluate_byNum_Larson(double, void *);
double IMFevaluate_byEgy_Larson(double, void *);

int get_IMF_SlopeBin(double, IMF_Type *);

int not_in_BHrange(int, double *);

// typedef EXTERNIMF double (*(double, double, IMF_Type *, int));

typedef double(EXTERNALIMF)(double, void *);

extern EXTERNALIMF **externalIMFs_byMass;
extern EXTERNALIMF **externalIMFs_byNum;
extern EXTERNALIMF **externalIMFs_byEgy;
extern char **externalIMFs_names;

extern int NexternalIMFs;

int initialize_externalIMFs(int);
int set_externalIMF(int, char *, EXTERNALIMF *, EXTERNALIMF *, EXTERNALIMF *);
int search_externalIMF_name(char *);

#endif
