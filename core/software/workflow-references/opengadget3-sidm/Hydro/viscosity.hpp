/**
 * @file
 *
 * This file is part of the N-body/SPH code OpenGadget3.
 *
 * @brief Header file for extra viscosity and artificial conductivity routines.
 *
 * @author OpenGadget3-development team, in particular Klaus Dolag and Volker Springel.
 *
 * @copyright Copyright (c) 2026, OpenGadget3-development team lead by Klaus Dolag and Stefano Borgani
 * (see change logs). All rights reserved.
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License v3 or later.
 * See <https://www.gnu.org/licenses/>.
 */

#ifndef VISCOSITY_HPP
#define VISCOSITY_HPP

#include "../CodeBase/allvars.hpp"
#if !(defined ACC_INCLUDE_GRADIENTS) && defined(ACC_GRADIENTS) // we reinclude this same file if ACC is active
#include "../OpenACC/gpuallvars_branch.hpp"
#endif // !(defined ACC_INCLUDE_GRADIENTS) && defined(ACC_GRADIENTS)

#ifdef TIME_DEP_ART_VISC
#ifndef ACC_INCLUDE_GRADIENTS

#if TIME_DEP_ART_VISC == 0
/**
 * @brief Evaluates the artificial viscosity for a particle.
 *
 * This function calculates the artificial viscosity for a particle based on its divergence
 * velocity and curl velocity. It computes the sound speed over the smoothing length, and
 * determines the correction factor `f` based on the ratio of the absolute value of the divergence
 * velocity to the sum of the absolute value of the divergence velocity, curl velocity, and a small
 * fraction of the sound speed over the smoothing length. Then, it computes the change in alpha
 * (viscosity parameter) over time using the decay time, maximum signal velocity, smoothing length,
 * and correction factor `f`. If comoving integration is enabled, it adjusts the change in alpha
 * by the Hubble parameter and the square of the cosmic scale factor. Finally, it applies a source
 * term to the change in alpha based on the divergence velocity.
 *
 * @param i Index of the particle for which to evaluate the artificial viscosity.
 * @param fac_mu Factor for artificial viscosity.
 * @param hubble_a Hubble parameter multiplied by the cosmic scale factor.
 */
static inline void eval_art_visc(int i, MyAtLeastDouble hubble_a, MyAtLeastDouble hubble_a2)
{
    // #define TRID 334666   // Use this to get some usefull debug outputs
    MyAtLeastDouble cs_h;

    cs_h = sqrt(GAMMA * SphP[i].Pressure / SphP[i].Density) / (2 * P[i].Hsml);

    // Balsare 1995, Eq. 4 / Dolag+2005, Eq. 5 (4.12.2024: removed the / fac_mu as all should be in code units)
    MyAtLeastDouble f =
        abs(SphP[i].DivVel) / (abs(SphP[i].DivVel) + SphP[i].CurlVel + BALSARA_FAC_FOR_CSND_OVER_H * cs_h);

#ifdef NEW_VISC_LIMIT
    MyAtLeastDouble min_div_vel =
        sqrt(SphP[i].VelPred[0] * SphP[i].VelPred[0] + SphP[i].VelPred[1] * SphP[i].VelPred[1] +
             SphP[i].VelPred[2] * SphP[i].VelPred[2]) /
        P[i].Hsml * DIVV_PRECISION_VISC;
#else // NEW_VISC_LIMIT
    MyAtLeastDouble min_div_vel = 0.0;
#endif // NEW_VISC_LIMIT

    MyAtLeastDouble avisc_source = 0;
    if (SphP[i].DivVel < min_div_vel)
    {
        avisc_source = All.ViscositySourceScaling * av_source_scaling * f * max(min_div_vel, -SphP[i].DivVel);
    }

    // Dolag+2005, Eq. 9+10
    MyAtLeastDouble avisc_decay = (SphP[i].alpha - All.ViscosityAlphaMin) * SphP[i].MaxSignalVel /
                                  (All.ViscosityDecayLength * av_decay_scaling * P[i].Hsml);

    // Dolag+2005, Eq. 9
    SphP[i].Dtalpha = (avisc_source - avisc_decay) / hubble_a2;

#ifdef WINDS
    // wind particles should be non-viscous by definition
    if (SphP[i].DelayTime > 0)
    {
        SphP[i].Dtalpha = 0.0;
        SphP[i].alpha = All.ViscosityAlphaMin;
    }
#endif // WINDS

#ifdef TRID
    if (P[i].ID == TRID)
    {
        VERBOSE(1,
                "VISCDEBUG: cs_h=%g f=%g (divv=%g curlv=%g min_div_vel=%g) source/decay=%g/%g (alpha=%g Dtalpha=%g)\n",
                cs_h, f, SphP[i].DivVel, SphP[i].CurlVel, min_div_vel, avisc_source, avisc_decay, SphP[i].alpha,
                SphP[i].Dtalpha);
    }
#endif // TRID
}
#endif // TIME_DEP_ART_VISC == 0

#if (TIME_DEP_ART_VISC == 1) || (TIME_DEP_ART_VISC == 2) || (TIME_DEP_ART_VISC == 3)

static inline void eval_art_visc(int i, MyAtLeastDouble hubble_a, MyAtLeastDouble hubble_a2)
{
    /* We prepare the time-integration for the artificial viscosity as discussed in Beck+16a here. */
    MyAtLeastDouble cs;

    cs = sqrt(GAMMA * SphP[i].Pressure / SphP[i].Density);

    // apply final density division, Beck+16a, Eq. 12
    SphP[i].visc_R /= SphP[i].Density;

    // numerator of Beck+16a, Eq. 14  (but using ((1-R)/2)^4 as weight, which is closer to Price 2018)
    MyAtLeastDouble w = (1.0 - SphP[i].visc_R);
    MyAtLeastDouble f1 = 2 * w * w * w * w * SphP[i].DivVel;
    // needs to be squared
    f1 *= f1;

    // velicity curl squared in denominator of Beck+16a, Eq. 14
    // As f1 is weighted by 2^10 for converegnt flows, we added 2^8 as weight for f2
    MyAtLeastDouble f2 = SphP[i].CurlVel * SphP[i].CurlVel;

    // Ratio between strength of shock, to strength of shear, see Beck+16a, Eq. 14
    MyAtLeastDouble xi_i = 0.0;

    // check to be sure we don't divide by 0
    if ((f1 + f2) != 0)
    {
        // Beck+16a, Eq. 14 (but adding a cs/h term, as done in the balsara switch)
        MyAtLeastDouble f3 = cs / (2 * P[i].Hsml);
        f3 *= f3;
        xi_i = f1 / (f1 + f2);
    }

    // timestep between last time particle was active and now
    MyAtLeastDouble dt = (P[i].Ti_current - SphP[i].visc_Oldtime) * All.Timebase_interval; /* / hubble_a2; */

    // change in divV between the last and this timestep, see Beck+16a, used in Eq. 9+10
    MyAtLeastDouble dtdivv_i = 0.0;
    if (dt > 0)
    {
        dtdivv_i = (SphP[i].DivVel - SphP[i].visc_OldDvel) / dt;
    }

    // store current divV for next timestep
    SphP[i].visc_OldDvel = SphP[i].DivVel;
    SphP[i].visc_Oldtime = P[i].Ti_current;

    // directional shock indicator, Beck+16a, Eq. 13
    MyAtLeastDouble A_i = xi_i * max(-dtdivv_i, (MyAtLeastDouble)0);

    // numerator of Beck+16a, Eq. 15
    MyAtLeastDouble h2A = 2 * P[i].Hsml * 2 * P[i].Hsml * A_i;

#ifdef ART_VISC_DEBUG
    SphP[i].h2A = h2A;
#endif // ART_VISC_DEBUG

    // signal velocity squared in denominator of Beck+16a, Eq. 15
    MyAtLeastDouble vsig2 = cs * cs;
#if (TIME_DEP_ART_VISC == 3)
    vsig2 += pow((MyAtLeastDouble)SphP[i].CurlVel * 2 * P[i].Hsml, 2);
#endif // (TIME_DEP_ART_VISC == 3)
    // minimum artificial viscosity coefficient
    MyAtLeastDouble alpha_loc = 0;

    // This will add the hubble expansion as a typical velocity
    if (All.ComovingIntegrationOn)
    {
        vsig2 += hubble_a * 2 * P[i].Hsml * hubble_a * 2 * P[i].Hsml;
    }

    // check to avoid division by 0 and only apply to real shocks
    if (h2A != 0 && SphP[i].visc_R < -0.75)
    {
        // Beck+16a, Eq. 15
        // (Using prefactor like in the original viscosity to reach ArtBulkViscConst also for weak shocks)
        alpha_loc = All.ViscositySourceScaling * All.ArtBulkViscConst * h2A / (h2A + vsig2);
    }
    else
    {
        // fallback: minimum artificial viscosity coefficient
        alpha_loc = All.ViscosityAlphaMin * SphP[i].F_Balsara;
    }

#if ((TIME_DEP_ART_VISC == 3) || (TIME_DEP_ART_VISC == 2)) && defined(DIVB_TO_ALPHA)
    MyAtLeastDouble Bmag = 0;
    for (int k = 0; k < 3; k++)
    {
        Bmag += SphP[i].BPred[k] * SphP[i].BPred[k];
    }
    if (Bmag > 0)
    {
        Bmag = sqrt(Bmag);
        MyAtLeastDouble v_alfven = sqrt((SphP[i].BPred[0] * SphP[i].BPred[0] + SphP[i].BPred[1] * SphP[i].BPred[1] +
                                         SphP[i].BPred[2] * SphP[i].BPred[2]) *
                                        mu0_inv / SphP[i].Density);
        // Skalining with numngb because this should be order of the inverse accuracy divB can be estimated (e.g.
        // percentage at 100dreds of ngbs) Scaling with divB normalized to B devioded by kernel width (e.g. 2 times
        // hsml) Scaling with plasma beta, so if beta gets 1, divB errors of order a percent start to hurt -> drive up
        // the viscosity
        alpha_loc += DIVB_TO_ALPHA * All.DesNumNgb * All.ViscositySourceScaling * All.ArtBulkViscConst *
                     abs(SphP[i].divB) * 2 * P[i].Hsml / Bmag * (v_alfven * v_alfven) / (cs * cs);
        vsig2 += v_alfven * v_alfven;
    }
#endif // (TIME_DEP_ART_VISC == 2) && defined(DIVB_TO_ALPHA)
#if (TIME_DEP_ART_VISC == 3)
    if (alpha_loc < All.ViscosityAlphaMin * SphP[i].F_Balsara && SphP[i].DivVel < 0)
    {
        alpha_loc = All.ViscosityAlphaMin * SphP[i].F_Balsara;
    }
#else // (TIME_DEP_ART_VISC == 3)
    if (alpha_loc < All.ViscosityAlphaMin)
    {
        alpha_loc = All.ViscosityAlphaMin;
    }
#endif // (TIME_DEP_ART_VISC == 3)
    if (alpha_loc > All.ArtBulkViscConst)
    {
        alpha_loc = All.ArtBulkViscConst;
    }

    // check if current artificial viscosity coefficient is larger than the minimal one
    if (SphP[i].alpha > alpha_loc)
    {
#if (TIME_DEP_ART_VISC == 1)
        SphP[i].Dtalpha = 0.0;
        SphP[i].alpha = All.ViscosityAlphaMin;
#elif (TIME_DEP_ART_VISC == 2)
        // Cullen&Dehnen (2010), Eq. 16 (adapted)
        // Beck+16a, Eq. 16
        // SphP[i].Dtalpha = (alpha_loc - SphP[i].alpha) * SphP[i].MaxSignalVel / (2 * All.ViscosityDecayLength *
        // P[i].Hsml * hubble_a2); Alternative formulation to decay alpha over ViscosityDecayLength Courant timesteps
        // (works beter in cosmological simulations)
        SphP[i].Dtalpha = (alpha_loc - SphP[i].alpha) * All.CourantFac * 0.5 / (All.ViscosityDecayLength * dt);
#elif (TIME_DEP_ART_VISC == 3)
        if (SphP[i].alpha < 0.5 * All.ArtBulkViscConst) // At late times switch to exponential decay
        {
            SphP[i].Dtalpha = (alpha_loc - SphP[i].alpha) * All.CourantFac * 0.5 / (2 * All.ViscosityDecayLength * dt);
        }
        else // Begin with a quartic decay to avoid quick decay
        {
            MyAtLeastDouble beta = 4.0; // Use a quartic function
            // Analytic solution for a ptential decay
            SphP[i].Dtalpha = -All.ArtBulkViscConst * (beta / 2) *
                              pow(1 - SphP[i].alpha / (1.0001 * All.ArtBulkViscConst), (beta - 1) / beta) *
                              SphP[i].MaxSignalVel / (All.ViscosityDecayLength * P[i].Hsml);
        }
#endif // TIME_DEP_ART_VISC
    }
    else
    {
        // set to locally computed artificial viscosity coefficient
        SphP[i].alpha = alpha_loc;
        // enforce no change in alpha
        SphP[i].Dtalpha = 0.0;
    }

#ifdef WINDS
    // wind particles should be non-viscous by definition
    if (SphP[i].DelayTime > 0)
    {
        // set to minimal artificial viscosity coefficient
        SphP[i].Dtalpha = 0.0;
        SphP[i].alpha = All.ViscosityAlphaMin * SphP[i].F_Balsara;
    }
#endif // WINDS
}
#endif // (TIME_DEP_ART_VISC == 1) || (TIME_DEP_ART_VISC == 2) || (TIME_DEP_ART_VISC == 3)
#endif // ACC_INCLUDE_GRADIENTS
#endif // TIME_DEP_ART_VISC

#endif // VISCOSITY_HPP
