/**
 * @file
 *
 * This file is part of the N-body/SPH code OpenGadget3.
 *
 * @brief
 *
 * @author OpenGadget3-development team, in particular Cinthia Ragone Figueroa and Gian Luigi Granato.
 *
 * @copyright Copyright (c) 2026, OpenGadget3-development team lead by Klaus Dolag and Stefano Borgani
 * (see change logs). All rights reserved.
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License v3 or later.
 * See <https://www.gnu.org/licenses/>.
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <ctype.h>
#include "../../CodeBase/allvars.hpp"
#include "../../CodeBase/proto.hpp"

#if defined(GL_GIVENACC) && !defined(GL_GIVENGRAIN)
#error nonsense to use GL_GIVENACC without GL_GIVENGRAIN
#endif

#include "gl_cr_dust.hpp"

double dcondeffAGB[LT_NMet];
double dcondeffIa[LT_NMet];
double dcondeffII[LT_NMet];
// Indexes, NOTE here iniialized negative to recognize when element not present in
// metals.dat or dust.dat. See function read_dust
int ind_C = -1, ind_O = -1; // indexes in dcondeff*
// here "olivine" version Mg, Si, Fe
int ind_sil[n_sil] = {-1, -1, -1}; // indexes in dcondeff* of elements other than O entering in assumed silicate grain
double mu_sil[n_sil] = {24., 28., 56.}; // atomic weights
#ifdef GL_GIVENGRAIN //
int n_ato_sil[n_sil] = {1, 1, 1},
    n_ato_O = 4.; // number of atoms in assumed silicate c. e.g. 1 1 1 4 for O in standard olivine
double nummol_AGB, nummol_II, nummol_Ia; // number of "molecules" of assumed chemical compound forming silicate grains
double effsil_AGB; // these effsil_* are set equal to the condensation efficiencies for silicium in dust.dat
double effsil_Ia; // see read_dust
double effsil_II;
// NOTE that if GL_GIVENGRAIN is defined, only the C elements of the arrays dcondeffX are used.
#endif

double mu_O = 16.0, mu_C = 12.0; // mus are atomic weights
double dcondeffAGB_var[LT_NMet], dcondeffIa_var[LT_NMet], dcondeffII_var[LT_NMet];

double Aacc_C = 2502., Aacc_sil_O = 1565.,
       Aacc_sil[n_sil] = {733., 911., 2523.}; // coefficients for accretion timescales of gas on grains

// Aacc_* computed with the following piece of IDL code. Uses Eq. 20 of Hirashita & Kuo 2011. with
// nx=n*(M_x/M)*<mu>/mu_x
/*pro acc_fac, metal
  if (metal eq 'C') then begin
      fx=1.0   & mu=12. & s_sma=2.26
  endif
  if (metal eq 'O') then begin
      fx=0.371   & mu=16.    & s_sma=3.3
  endif
  if (metal eq 'Mg') then begin
      fx=0.141   & mu=24.3  & s_sma=3.3 ; Mg
  endif
  if (metal eq 'Si') then begin
      fx=0.163   & mu=28.11 & s_sma=3.3 ; Si
  endif
  if (metal eq 'Fe') then begin
      fx=0.32    & mu=55.9 & s_sma=3.3 ; Fe
  endif
  a=5e-3 * 1e-4
  HYDROGEN_MASSFRAC = 0.7
  mu_c = (4./ (3. * HYDROGEN_MASSFRAC + 1.) )
  S_big=0.3
  n=1e3
  T=50
  k=1.38D-16
  mh=1.66d-24

  mx=mu*mh
  coef=a*fx*s_sma*mu/(sqrt(mx)*n*mu_c*S_big) * sqrt(2*!Pi/k/T)

  coef_yr=coef/(365.25*24*3600)

  print, metal, coef_yr, coef_yr/3
end
*/

#define mu_h ((double)4. / (5. * HYDROGEN_MASSFRAC + 3.)) /* molecular weight of the hot phase */
#define mu_c ((double)4. / (3. * HYDROGEN_MASSFRAC + 1.)) /* molecular weight of the cold phase */
#define SOLAR_MASS_IN_CODE_UNITS (SOLAR_MASS / All.UnitMass_in_g)

#if !(defined(LT_STELLAREVOLUTION) && defined(GL_CR_DUST))
#error Dust evolution GL_DUST requires LT_STELLAREVOLUTION!
#endif

#if defined(LT_STELLAREVOLUTION) && defined(GL_CR_DUST)

#define YEAR_IN_CODE_UNITS (SEC_PER_YEAR / All.UnitTime_in_s)

/**
 * @brief The function computes the evolution of gas particles metals that are locked in large and small grains.
 *
 * This function computes MORE DETAILS IN THE FUTURE
 *
 * The function operates in a loop over active particles MORE DETAILS IN THE FUTURE
 */

/* START OF DUST EVOLUTION FUNCTION */
void evolve_dust(void)
{
    double dt, dtime, ascale = 1, hubble_a = 0, a3inv, time_hubble_a, rhomean;
    double taumax = 1e11; // to improve speed, timescales above a maximum are set to 0, and computations are avoided
    double facsn;
    double epssn = 0.1; // grain destruction efficiency in SN shocks
    double eta, Nsne, msw; // other quantities for grain destruction in SN shocks
    double Ztot, Zgas, Mgc, D, Dl, Ds, taush, tauco, tauac, tauspu_s, tauspu_l;
    double XCold, M_h, M_c, Vol_tot, T_h, n_g, xn, n_hot, dum;
    double Mdl[LT_NMetP], Mds[LT_NMetP], dtDl[LT_NMetP], dtDs[LT_NMetP];

    double T_d = 2.e6, w_d = 2.5; // constants entering into the formula for sputtering

    double fdense; // fraction of gas particles dense enough for accretion and coagulation.
    // following quantities are used to compute fdense depending on particle density (NOT with Muppi),
    // based on Wada & Norman 07 results
    double sig_pdf = 2.5; // sigma of assumed PDF for unresolved gas densities
    double n_0 = 2.12, arg_erf; // normalization of PDF and auxiliary argument to be put in erf

    double Metals_accretion;
    double mwk, T_wk;

    TIMER_PUSH(1, "COOLINGSFR");
    //******************************** VEDERE COME EVOLVE LA ISSUE
    // https://gitlab.lrz.de/KlausDolag/OpenGadget3/-/issues/94 */
    if (All.ComovingIntegrationOn)
    { /* Factors for comoving integration of hydro */
        a3inv = 1 / (All.Time * All.Time * All.Time);
        hubble_a = hubble_function(All.Time);
        time_hubble_a = All.Time * hubble_a;
        ascale = All.Time;
        rhomean = 3 * hubble_a * hubble_a / 8 / M_PI /
                  All.GravityConstantInternal; /* mean density at time a in unit's code */
    }
    else
    {
        a3inv = ascale = time_hubble_a = 1;
        rhomean = 1;
    }

    /* this is used to get the hot phase numerical density in the effective model */
    T_h = All.TempSupernova / All.FactorEVP;

    for (int i = FirstActiveParticle; i >= 0; i = NextActiveParticle[i])
    {
        /* SphP[i].d.Density -> density
           SphP[i].Temperature -> temperature
           P[i].Mass -> mass
           SphP[i].Metals[1...15] -> mass of metals
           SphP[i].numSnIa, SphP[i].numSnII -> number of supernovae
           SphP[i].DustL[1...15], SphP[i].DustS[1...15] -> large and small dust grains mass
         */

        if (P[i].Type == 0)
        {
            dt = (P[i].TimeBin ? (1 << P[i].TimeBin) : 0) * All.Timebase_interval;
            //******************************** VEDERE COME EVOLVE LA ISSUE
            // https://gitlab.lrz.de/KlausDolag/OpenGadget3/-/issues/94 */
            /*  the actual time-step */
            if (All.ComovingIntegrationOn)
                dtime = All.Time * dt / time_hubble_a / All.HubbleParam; // GL 150823
            else
                dtime = dt;

            XCold = 0;
            Ztot = 0.;
            Zgas = 0.;
            D = 0.;
            Dl = 0.;
            Ds = 0.;
            tauac = 0.;
            taush = 0.;
            tauco = 0.;
            tauspu_l = 0.;
            tauspu_s = 0.;
            Metals_accretion = 0.;
            Mgc = P[i].Mass;

            /* these are used to get the hot phase numerical density in the effective model */
            XCold = SphP[i].XColdCloud;
            Vol_tot = Mgc / SphP[i].Density / a3inv / All.HubbleParam / All.HubbleParam / All.HubbleParam;
            M_c = XCold * Mgc / All.HubbleParam;
            M_h = (1. - XCold) * Mgc / All.HubbleParam;
            T_wk = SphP[i].Temperature;
            n_g = SphP[i].Density * a3inv * All.UnitDensity_in_cgs / mu_h / PROTONMASS * All.HubbleParam *
                  All.HubbleParam;
#ifdef GL_GIVENACC
            // find the element that constraints accretion to preserve given grain atom proportions
            double wk, nummolwk;
            int ind_acc, ind_acc_sil;
            nummolwk = SphP[i].Metals[ind_O] / pow(mu_O, 1.5) / n_ato_O;
            ind_acc = ind_O;
            ind_acc_sil = -1;
            for (int j = 0; j < n_sil; j++)
            {
                wk = SphP[i].Metals[ind_sil[j]] / pow(mu_sil[j], 1.5) / n_ato_sil[j];
                if (wk < nummolwk)
                {
                    nummolwk = wk;
                    ind_acc = ind_sil[j];
                    ind_acc_sil = j;
                }
            }
#endif // GL_GIVENACC

            for (int j = 1; j < LT_NMetP; j++)
            {
                dtDl[j] = 0.;
                dtDs[j] = 0.;
                Ztot += SphP[i].Metals[j] + SphP[i].DustL[j] + SphP[i].DustS[j];
                Zgas += SphP[i].Metals[j];

                Mdl[j] = SphP[i].DustL[j];
                Mds[j] = SphP[i].DustS[j];
                if (P[i].Mass != 0)
                {
                    D += (Mdl[j] + Mds[j]);
                    Dl += Mdl[j];
                    Ds += Mds[j];
                } // CLOSES if (P[i].Mass != 0)
            } // CLOSES for(j=1;j<LT_NMet; j++)

            if (P[i].Mass != 0)
            {
                Ztot /= Mgc;
                Zgas /= Mgc;
                Dl /= Mgc;
                Ds /= Mgc;
                D /= Mgc;
            }

            if (D > 1. || Dl > 1. || Ds > 1. || Ztot > 1. || Zgas > 1)
            {
                VERBOSE(2, " DUST ERROR. Dl=%e, Ds=%e, D=%e Ztot=%g Zgas=%g Mg=%g Task %d part %d\n", Dl, Ds, D, Ztot,
                        Zgas, Mgc, ThisTask, i);
            }

            //========================= Calculation of timescales

            if (XCold > 0) // particle is multi-phase
            {
                compute_ism_properties_effective_model(M_h, M_c, Vol_tot, T_h, &dum, &dum, &dum, &n_hot, &xn);
                // Wada and Norman 07
                arg_erf = (log(xn / n_0) / sig_pdf - sig_pdf) / 1.4142;
                fdense /= 0.5 * (1.0 - erf(arg_erf));
                T_wk = T_h;
            }
            else // not multi-phase particle
            {
                xn = n_g;
            }

            // calculate shattering timescale
            if (Dl > 0 && xn < 1e3)
            {
                /* as detailed in Granato+21, we use a smooth transition of grain velocities as a function of xn^(-2/3),
                   assuming grain velocities v=10 km/s for xn<1 and v=0.1 km/s for xn=1000 */
                taush = 5.41e7 * 0.01 / Dl / xn; // equation B4  Aoyama+16
                if (xn > 1)
                {
                    taush = taush * pow(xn, 0.6666);
                }

                if (taush < taumax) // convert to code unit or nullify if too long
                {
                    taush = taush * YEAR_IN_CODE_UNITS;
                }
                else
                {
                    taush = 0;
                }
            }

            // calculate coagulation timescale
            if (Ds > 0 && XCold > 0. && fdense > 1e-8) // only for MP particles
            {
                double vco = 0.2; // typical velocity dispersion of small grains in [km/s].
                tauco = (0.1 / vco) * 2.71e5 * (0.01 / Ds) / fdense / XCold;
                if (tauco < taumax) // convert to code unit or nullify if too long
                {
                    tauco = tauco * YEAR_IN_CODE_UNITS;
                }
                else
                {
                    tauco = 0;
                }
            }

            // calculate sputtering timescale
            // inspired by a combination of Hirashita 2017 (eq 10) and Tsai & Mathews 1995 eq 14 and 15
            // adopting 0.05 micron as typical size of large grains, the mean for a power law -3.5
            // distribution between 0.03 and 0.25 micron
            if (T_wk > 3e7)
                T_wk = 3e7; // GLG190118
            double n_forspu;
            if (XCold <= 0.)
            {
                n_forspu = xn; // no multi-phase
            }
            else
            {
                n_forspu = n_hot; // multi-phase
            }
            tauspu_l =
                5 * 1.65e7 / 3. * (0.05 / 0.1) * 0.01 / n_forspu *
                (pow(T_d / T_wk, w_d) + 1); // factor 3 is because Tsai & Mathews give timescale for radius not mass
            tauspu_s = 0.1 * tauspu_l; // adopting for small grains a size 10 times smaller than for large ones
            if (tauspu_l < taumax) // convert to code unit or nullify if too long
            {
                tauspu_l = tauspu_l * YEAR_IN_CODE_UNITS;
            }
            else
            {
                tauspu_l = 0;
            }

            if (tauspu_s < taumax) // convert to code unit or nullify if too long
            {
                tauspu_s = tauspu_s * YEAR_IN_CODE_UNITS;
            }
            else
            {
                tauspu_s = 0;
            }

            // calculate factor for SNae distruction
            Nsne = SphP[i].numSnIa + SphP[i].numSnII;
            msw = 1360 * SOLAR_MASS_IN_CODE_UNITS / 4.; // mass swept by a single blast in code units Aoymama+2016 (15)
            // factor 4 because we take v_s=200 km/s
            eta = epssn * msw / (Mgc / All.HubbleParam);
            if (eta > epssn) // if msw > mgas
            {
                eta = epssn;
            }
            facsn = (1 - pow(1 - eta, Nsne));

            //  now compute variations due to the various channels.
            //  Computation are avoided when timescales arrive here still 0, or set again to 0 because too long

            for (int j = 1; j < LT_NMetP; j++)
            {

                if (taush > 0)
                {
                    mwk = dtime * Mdl[j] / taush;
                    if (mwk > SphP[i].DustL[j])
                        mwk = SphP[i].DustL[j]; // not enough large grains available to transfer to small
                    dtDs[j] += mwk;
                    dtDl[j] -= mwk;
                }

                if (tauco > 0)
                {
                    mwk = dtime * Mds[j] / tauco;
                    if (mwk > SphP[i].DustS[j])
                        mwk = SphP[i].DustS[j]; // not enough small grains available to transfer large
                    dtDl[j] += mwk;
                    dtDs[j] -= mwk;
                }

                // calculate accretion timescale
                if (XCold > 0. && SphP[i].Metals[j] > 0. && fdense > 1e-8)
                {
                    tauac = 1.0 * Mgc / fdense / XCold;
#ifdef GL_GIVENACC
                    if (j != ind_C)
                    {
                        if (ind_acc_sil > -1)
                        {
                            tauac = Aacc_sil[ind_acc_sil] * tauac / (SphP[i].Metals[ind_acc]);
                        }
                        else
                        {
                            tauac = Aacc_sil_O * tauac / (SphP[i].Metals[ind_acc]);
                        }
                    }
                    else
                    {
                        tauac = Aacc_C * tauac / (SphP[i].Metals[j]);
                    }
#else // GL_GIVENACC
                    tauac = tauac / (SphP[i].Metals[j]);
                    if (j != ind_C)
                    {
                        tauac = Aacc_sil[ind_Si] * tauac;
                    }
                    else
                    {
                        tauac = Aacc_C * tauac;
                    }
#endif // GL_GIVENACC
                    if (tauac < taumax) // convert to code unit or nullify if too long
                    {
                        tauac = tauac * YEAR_IN_CODE_UNITS;
                    }
                    else
                    {
                        tauac = 0;
                    }
                }
                else
                {
                    tauac = 0.;
                }

                if (tauac > 0)
                {
                    Metals_accretion = dtime * Mds[j] / tauac;
                    if (Metals_accretion > XCold * fdense * SphP[i].Metals[j])
                    // check to be done here, to avoid creating small grains with metals that do not exist
                    {
                        VERBOSE(2,
                                " DUST Task %d Part %d wants to accrete a mass %e of metal %d, but only "
                                "has %e\n",
                                ThisTask, i, Metals_accretion, j, SphP[i].Metals[j]);
                        Metals_accretion = XCold * fdense * SphP[i].Metals[j];
                    }
                    /* decreasing metals amount for metals accretion on small dust grain */
                    SphP[i].Metals[j] -= Metals_accretion;
                    dtDs[j] += Metals_accretion;
                }

                if (SphP[i].DustL[j] + dtDl[j] < 0.)
                {
                    VERBOSE(2, " DUST Task %d Part %d wants to decrease large dust %d to less than zero.\n", ThisTask,
                            i, j);
                    VERBOSE(2, " Present=%e decrement=%e dt=%e n=%e Temp=%e XCold=%e\n", SphP[i].DustL[j], dtDl[j],
                            dtime / YEAR_IN_CODE_UNITS, xn, T_wk, XCold);
                    SphP[i].DustL[j] = 0.0;
                }
                else
                    SphP[i].DustL[j] += dtDl[j];

                if (SphP[i].DustS[j] + dtDs[j] < 0.)
                {
                    VERBOSE(2, " DUST Task %d Part %d wants to decrease small dust %d to less than zero.\n", ThisTask,
                            i, j);
                    VERBOSE(2, " Present=%e decrement=%e dt=%e n=%e Temp=%e XCold=%e\n", SphP[i].DustS[j], dtDs[j],
                            dtime / YEAR_IN_CODE_UNITS, xn, T_wk, XCold);
                    SphP[i].DustS[j] = 0.0;
                }
                else
                    SphP[i].DustS[j] += dtDs[j];

                // We make SN destruction and sputtering after the rest of the processes.
                // Since these two processes retun metals to the gas phase we need to be sure that the
                // amount of metals returned to the gas phase is exactly the same as the destroyed dust.
                // We also add here the metals restitution by SN
                mwk = facsn * SphP[i].DustL[j];
                if (mwk > SphP[i].DustL[j])
                {
                    mwk = SphP[i].DustL[j];
                }
                SphP[i].DustL[j] -= mwk;
                SphP[i].Metals[j] += mwk;

                mwk = facsn * SphP[i].DustS[j];
                if (mwk > SphP[i].DustS[j])
                {
                    mwk = SphP[i].DustS[j];
                }
                SphP[i].DustS[j] -= mwk;
                SphP[i].Metals[j] += mwk;

                double Xhot;
                Xhot = 1. - XCold;
                if (tauspu_l > 0)
                {
                    mwk = dtime * SphP[i].DustL[j] * Xhot / tauspu_l;
                    if (mwk > Xhot * SphP[i].DustL[j])
                        mwk = Xhot * SphP[i].DustL[j]; //  enough large grains available
                    SphP[i].Metals[j] += mwk;
                    SphP[i].DustL[j] -= mwk;
                }

                if (tauspu_s > 0)
                {
                    mwk = dtime * SphP[i].DustS[j] * Xhot / tauspu_s;
                    if (mwk > Xhot * SphP[i].DustS[j])
                        mwk = Xhot * SphP[i].DustS[j]; //  enough small grains available
                    SphP[i].Metals[j] += mwk;
                    SphP[i].DustS[j] -= mwk;
                }

                if (SphP[i].Metals[j] < 0.0)
                {
                    /* note, this should not happen */
                    SphP[i].Metals[j] = 0.0;
                }

                if (D > 1. || Dl > 1. || Ds > 1. || Ztot > 1. || Zgas > 1)
                {
                    VERBOSE(2,
                            " DUST ERROR. Dl=%e, Ds=%e, dtDl=%e dtDs=%g Met=%g Ztot=%g Zgas=%g Mg=%g Xhot=%g tauS=%g "
                            "tauL=%g Task %d part %d j %d\n",
                            SphP[i].DustL[j], SphP[i].DustS[j], dtDl[j], dtDs[j], SphP[i].Metals[j], Ztot, Zgas, Mgc,
                            Xhot, tauspu_s, tauspu_l, ThisTask, i, j);
                }
            } // CLOSES for(j=1;j<LT_NMetP; j++)

            /* zeroing the sn count */
            SphP[i].numSnIa = 0.;
            SphP[i].numSnII = 0.;
        } // CLOSES if(P[i].Type==0)
    } // CLOSES for(i = FirstActiveParticle; i >= 0; i = NextActiveParticle[i])

    TIMER_POP(1, "COOLINGSFR");
    /* XXXX Maybe we want to add a CPU_DUST */

} // CLOSES evolve_dust()

void compute_ism_properties_effective_model(double M_h, double M_c, double Vol_tot, double T_h, double *fill_h,
                                            double *Rho_h, double *Rho_c, double *n_h, double *n_c)
{
    double Vol_h, Vol_c, Frac_h, Frac_c;

    Frac_h = M_h / (M_c + M_h);
    Frac_c = 1.0 - Frac_h;
    *fill_h = 1.0 / (1.0 + Frac_c / Frac_h * mu_h / mu_c * All.TempClouds / T_h);
    Vol_h = Vol_tot * *fill_h;
    Vol_c = Vol_tot * (1.0 - *fill_h);
    *Rho_h = M_h / Vol_h;
    *n_h = *Rho_h * All.UnitDensity_in_cgs / mu_h / PROTONMASS;
    *Rho_c = M_c / Vol_c;
    *n_c = *Rho_c * All.UnitDensity_in_cgs / mu_c / PROTONMASS;

    return;
}

#endif // CLOSES defined(LT_STELLAREVOLUTION) && defined(GL_CR_DUST)
