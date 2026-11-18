/**
 * @file
 *
 * This file is part of the N-body/SPH code OpenGadget3.
 *
 * @brief Helper functions for the One-Fluid model implementation of dust dynamics.
 * Implicit second-order integrator for gas-dust aerodynamic drag.
 *
 * @author OpenGadget3-development team, in particular Giovanni Tedeschi.
 *
 * @copyright Copyright (c) 2026, Klaus Dolag, and all contributing authors
 * (see change logs). All rights reserved.
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License v3 or later.
 * See <https://www.gnu.org/licenses/>.
 */

#include "../CodeBase/allvars.hpp"
#include "../CodeBase/proto.hpp"

#ifdef GT_DUST

/**
 * @brief Set the TotalDustFrac and TotalDeltaVel for a particle.
 *
 * This function sets the total physical dust fraction and the total
 * velocity difference, summed over all dust grain sizes, for a given particle.
 *
 * @param i Index of the particle.
 */
void compute_TotalDustFrac_and_TotalDeltaVel(int i)
{
    SphP[i].TotalDustFrac = 0.0;
    for (int j = 0; j < 3; j++)
        SphP[i].TotalDeltaVel[j] = 0.0;
    for (int idust = 0; idust < GT_NUM_DUST; idust++)
    {
        SphP[i].TotalDustFrac += dustfrac(SphP[i].DustFrac[idust]);
        for (int j = 0; j < 3; j++)
            SphP[i].TotalDeltaVel[j] += dustfrac(SphP[i].DustFrac[idust]) * SphP[i].DeltaVel[idust][j];
    }
    for (int j = 0; j < 3; j++)
    {
        if (SphP[i].TotalDustFrac > 0.0)
        {
            SphP[i].TotalDeltaVel[j] /= SphP[i].TotalDustFrac;
        }
        else
        {
            SphP[i].TotalDeltaVel[j] = 0.0;
        }
    }
}

/**
 * @brief Set the TotalDustFracPred and TotalDeltaVelPred for a particle.
 *
 * This function sets the predicted total physical dust fraction and the predicted
 * total velocity difference, summed over all dust grain sizes, for a given particle.
 *
 * @param i Index of the particle.
 */
void compute_TotalDustFracPred_and_TotalDeltaVelPred(int i)
{
    SphP[i].TotalDustFracPred = 0.0;
    for (int j = 0; j < 3; j++)
        SphP[i].TotalDeltaVelPred[j] = 0.0;
    for (int idust = 0; idust < GT_NUM_DUST; idust++)
    {
        SphP[i].TotalDustFracPred += dustfrac(SphP[i].DustFracPred[idust]);
        for (int j = 0; j < 3; j++)
            SphP[i].TotalDeltaVelPred[j] += dustfrac(SphP[i].DustFracPred[idust]) * SphP[i].DeltaVelPred[idust][j];
    }
    for (int j = 0; j < 3; j++)
    {
        if (SphP[i].TotalDustFracPred > 0.0)
        {
            SphP[i].TotalDeltaVelPred[j] /= SphP[i].TotalDustFracPred;
        }
        else
        {
            SphP[i].TotalDeltaVelPred[j] = 0.0;
        }
    }
}

/**
 * @brief Compute the relative kinetic energy for a particle.
 *
 * This function computes the relative kinetic energy for a given particle.
 * It is used to monitor energy conservation during the drag integration.
 *
 * @param i Index of the particle.
 */
MyAtLeastDouble get_relative_kinetic_energy(int i)
{
    MyAtLeastDouble Krel = 0.0;

    for (int j = 0; j < 3; j++)
    {
        Krel += -0.5 * P[i].Mass * pow(SphP[i].TotalDustFrac, 2) * pow(SphP[i].TotalDeltaVel[j], 2);
        for (int idust = 0; idust < GT_NUM_DUST; idust++)
        {
            Krel += 0.5 * P[i].Mass * dustfrac(SphP[i].DustFrac[idust]) * pow(SphP[i].DeltaVel[idust][j], 2);
        }
    }

    return Krel;
}

/**
 * @brief Integrate the dust aerodynamic drag for a particle.
 *
 * This function integrates the dust aerodynamic drag for all dust
 * grain sizes for a given particle over a certain time interval.
 * The integration is performed using an implicit second-order scheme.
 *
 * @param i Index of the particle.
 * @param tstart Start time of the integration.
 * @param tend End time of the integration.
 */
void integrate_dust_drag(int i, integertime tstart, integertime tend)
{
    MyFloat dt_dust = (tend - tstart) * All.Timebase_interval;

    compute_TotalDustFrac_and_TotalDeltaVel(i);

    MyAtLeastDouble Ktot_init = get_relative_kinetic_energy(i);

    MyFloat ts_i;
    MyFloat ts_max = 0.;
    for (int idust = 0; idust < GT_NUM_DUST; idust++)
    {
        ts_i = (SphP[i].Density * dustfrac(SphP[i].DustFrac[idust])) / SphP[i].DragCoefficient[idust];
        ts_max = std::max(ts_max, ts_i);
    }
    MyFloat gamma1;
    MyFloat gamma2;
    MyFloat beta1;
    MyFloat beta2;
    MyFloat b;
    if (dt_dust <= ts_max)
    {
        gamma1 = 1.0;
        gamma2 = 0.0;
        beta1 = -0.5;
        beta2 = 2. / 3.;
        b = 1.;
    }
    else
    {
        gamma1 = 1.;
        gamma2 = 1.;
        beta1 = 1.;
        beta2 = -1.;
        b = 0.;
    }

    MyFloat A1x = 0., A1y = 0., A1z = 0., B1 = 0, C1 = 0, D1 = 0;
    MyFloat A2x = 0., A2y = 0., A2z = 0., B2 = 0, C2 = 0, D2 = 0;

    MyFloat lambda = 0., delta1 = 0., delta2 = 0.;
    MyFloat eps = 0.;
    MyFloat alpha[GT_NUM_DUST];
    MyFloat rhod_vd[GT_NUM_DUST];
    MyFloat dust_momentum[GT_NUM_DUST][NUMDIMS];
    MyFloat gas_momentum[NUMDIMS];
    MyFloat K1[GT_NUM_DUST + 1][NUMDIMS];
    MyFloat K2[GT_NUM_DUST + 1][NUMDIMS];

    for (int idust = 0; idust < GT_NUM_DUST; idust++)
    {
        alpha[idust] = SphP[i].DragCoefficient[idust] / (SphP[i].Density * dustfrac(SphP[i].DustFrac[idust]));
        eps = dustfrac(SphP[i].DustFrac[idust]) / (1. - SphP[i].TotalDustFrac);
        for (int j = 0; j < NUMDIMS; j++)
        {
            gas_momentum[j] = SphP[i].Density * (1. - SphP[i].TotalDustFrac) *
                              (P[i].Vel[j] - SphP[i].TotalDustFrac * SphP[i].TotalDeltaVel[j]);
            dust_momentum[idust][j] =
                SphP[i].Density * dustfrac(SphP[i].DustFrac[idust]) *
                (P[i].Vel[j] - SphP[i].TotalDustFrac * SphP[i].TotalDeltaVel[j] + SphP[i].DeltaVel[idust][j]);
        }
        lambda = 1. / (1. + alpha[idust] * dt_dust *
                                (gamma1 + gamma2 + alpha[idust] * dt_dust * (gamma1 * gamma2 - beta1 * beta2)));
        delta1 = 1. / (1. + gamma1 * dt_dust * alpha[idust]);
        delta2 = 1. / (1. + gamma2 * dt_dust * alpha[idust]);

        A1x += alpha[idust] * dust_momentum[idust][0] * delta1 -
               beta1 * dt_dust * alpha[idust] * alpha[idust] * dust_momentum[idust][0] *
                   (1. + alpha[idust] * dt_dust * (gamma1 - beta2)) * delta1 * lambda;
        A2x += alpha[idust] * dust_momentum[idust][0] * delta2 -
               beta2 * dt_dust * alpha[idust] * alpha[idust] * dust_momentum[idust][0] *
                   (1. + alpha[idust] * dt_dust * (gamma2 - beta1)) * delta2 * lambda;

        A1y += alpha[idust] * dust_momentum[idust][1] * delta1 -
               beta1 * dt_dust * alpha[idust] * alpha[idust] * dust_momentum[idust][1] *
                   (1. + alpha[idust] * dt_dust * (gamma1 - beta2)) * delta1 * lambda;
        A2y += alpha[idust] * dust_momentum[idust][1] * delta2 -
               beta2 * dt_dust * alpha[idust] * alpha[idust] * dust_momentum[idust][1] *
                   (1. + alpha[idust] * dt_dust * (gamma2 - beta1)) * delta2 * lambda;

        A1z += alpha[idust] * dust_momentum[idust][2] * delta1 -
               beta1 * dt_dust * alpha[idust] * alpha[idust] * dust_momentum[idust][2] *
                   (1. + alpha[idust] * dt_dust * (gamma1 - beta2)) * delta1 * lambda;
        A2z += alpha[idust] * dust_momentum[idust][2] * delta2 -
               beta2 * dt_dust * alpha[idust] * alpha[idust] * dust_momentum[idust][2] *
                   (1. + alpha[idust] * dt_dust * (gamma2 - beta1)) * delta2 * lambda;

        B1 += alpha[idust] * eps * delta1 - beta1 * dt_dust * pow(alpha[idust], 2) * eps *
                                                (1. + alpha[idust] * dt_dust * (gamma1 - beta2)) * delta1 * lambda;
        B2 += alpha[idust] * eps * delta2 - beta2 * dt_dust * pow(alpha[idust], 2) * eps *
                                                (1. + alpha[idust] * dt_dust * (gamma2 - beta1)) * delta2 * lambda;

        C1 += alpha[idust] * eps * delta1 - dt_dust * pow(alpha[idust], 2) * eps *
                                                (gamma2 + alpha[idust] * dt_dust * (gamma1 * gamma2 - beta1 * beta2)) *
                                                delta1 * lambda;
        C2 += alpha[idust] * eps * delta2 - dt_dust * pow(alpha[idust], 2) * eps *
                                                (gamma1 + alpha[idust] * dt_dust * (gamma1 * gamma2 - beta1 * beta2)) *
                                                delta2 * lambda;

        D1 += gamma1 * dt_dust * alpha[idust] * eps * delta1 -
              beta1 * beta2 * pow(dt_dust, 2) * pow(alpha[idust], 2) * eps * delta1 * lambda;
        D2 += gamma2 * dt_dust * alpha[idust] * eps * delta2 -
              beta1 * beta2 * pow(dt_dust, 2) * pow(alpha[idust], 2) * eps * delta2 * lambda;
    }
    D1 += 1.;
    D2 += 1.;

    K1[0][0] = (beta1 * dt_dust * C1 * (A2x - B2 * gas_momentum[0]) - D2 * (A1x - B1 * gas_momentum[0])) /
               (beta1 * beta2 * dt_dust * dt_dust * C1 * C2 - D1 * D2);
    K2[0][0] = (beta2 * dt_dust * C2 * (A1x - B1 * gas_momentum[0]) - D1 * (A2x - B2 * gas_momentum[0])) /
               (beta1 * beta2 * dt_dust * dt_dust * C1 * C2 - D1 * D2);

    K1[0][1] = (beta1 * dt_dust * C1 * (A2y - B2 * gas_momentum[1]) - D2 * (A1y - B1 * gas_momentum[1])) /
               (beta1 * beta2 * dt_dust * dt_dust * C1 * C2 - D1 * D2);
    K2[0][1] = (beta2 * dt_dust * C2 * (A1y - B1 * gas_momentum[1]) - D1 * (A2y - B2 * gas_momentum[1])) /
               (beta1 * beta2 * dt_dust * dt_dust * C1 * C2 - D1 * D2);

    K1[0][2] = (beta1 * dt_dust * C1 * (A2z - B2 * gas_momentum[2]) - D2 * (A1z - B1 * gas_momentum[2])) /
               (beta1 * beta2 * dt_dust * dt_dust * C1 * C2 - D1 * D2);
    K2[0][2] = (beta2 * dt_dust * C2 * (A1z - B1 * gas_momentum[2]) - D1 * (A2z - B2 * gas_momentum[2])) /
               (beta1 * beta2 * dt_dust * dt_dust * C1 * C2 - D1 * D2);

    for (int idust = 0; idust < GT_NUM_DUST; idust++)
    {
        eps = dustfrac(SphP[i].DustFrac[idust]) / (1. - SphP[i].TotalDustFrac);
        lambda = 1. / (1. + alpha[idust] * dt_dust *
                                (gamma1 + gamma2 + alpha[idust] * dt_dust * (gamma1 * gamma2 - beta1 * beta2)));

        for (int j = 0; j < NUMDIMS; j++)
        {
            K1[idust + 1][j] =
                alpha[idust] * lambda *
                ((gas_momentum[j] * eps - dust_momentum[idust][j]) * (1. + alpha[idust] * dt_dust * (gamma2 - beta1)) +
                 K1[0][j] * eps * dt_dust * (gamma1 + alpha[idust] * dt_dust * (gamma1 * gamma2 - beta1 * beta2)) +
                 K2[0][j] * beta1 * eps * dt_dust);

            K2[idust + 1][j] =
                alpha[idust] * lambda *
                ((gas_momentum[j] * eps - dust_momentum[idust][j]) * (1. + alpha[idust] * dt_dust * (gamma1 - beta2)) +
                 K2[0][j] * eps * dt_dust * (gamma2 + alpha[idust] * dt_dust * (gamma1 * gamma2 - beta1 * beta2)) +
                 K1[0][j] * beta2 * eps * dt_dust);
        }
    }

    MyFloat gas_factor, dust_factor;
    for (int j = 0; j < NUMDIMS; j++)
    {
        gas_factor = dt_dust / SphP[i].Density / (1. - SphP[i].TotalDustFrac) * (b * K1[0][j] + (1. - b) * K2[0][j]);
        for (int idust = 0; idust < GT_NUM_DUST; idust++)
        {
            if (dustfrac(SphP[i].DustFrac[idust]) > 1e-20)
            {
                dust_factor = dt_dust / (SphP[i].Density) / dustfrac(SphP[i].DustFrac[idust]) *
                              (b * K1[idust + 1][j] + (1. - b) * K2[idust + 1][j]);
                SphP[i].DeltaVel[idust][j] += dust_factor - gas_factor;
                SphP[i].DeltaVelPred[idust][j] = SphP[i].DeltaVel[idust][j];
            }
        }
    }

    compute_TotalDustFrac_and_TotalDeltaVel(i);
    SphP[i].TotalDustFracPred = SphP[i].TotalDustFrac;
    for (int j = 0; j < NUMDIMS; j++)
    {
        SphP[i].TotalDeltaVelPred[j] = SphP[i].TotalDeltaVel[j];
    }

    MyAtLeastDouble Ktot_final = get_relative_kinetic_energy(i);

#if GADGET_HYDRO == HYDRO_SPH
    SphP[i].InternalEnergySPH += (Ktot_init - Ktot_final) / (P[i].Mass * (1. - SphP[i].TotalDustFrac));
    SphP[i].Entropy = SphP[i].InternalEnergySPH * get_gamma_minus1(i) /
                      pow(SphP[i].Density * (1. - SphP[i].TotalDustFrac), get_gamma_minus1(i));
    SphP[i].InternalEnergySPHPred = SphP[i].InternalEnergySPH;
    SphP[i].EntropyPred = SphP[i].Entropy;
#elif GADGET_HYDRO == HYDRO_MFM
    SphP[i].InternalEnergy += (Ktot_init - Ktot_final) / (P[i].Mass * (1. - SphP[i].TotalDustFrac));
    SphP[i].InternalEnergyPred = SphP[i].InternalEnergy;
    mfm_synchronize_entropy(i);
#endif // GADGET_HYDRO
}

#endif // GT_DUST