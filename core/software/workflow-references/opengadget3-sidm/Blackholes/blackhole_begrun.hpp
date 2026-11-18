/**
 * @file
 *
 * This file is part of the N-body/SPH code OpenGadget3.
 *
 * @brief This file contains the black hole related functions and structures.
 *
 * @author OpenGadget3-development team, in particular Klaus Dolag, Luca Sala and Volker Springel.
 *
 * @copyright Copyright (c) 2026, OpenGadget3-development team lead by Klaus Dolag and Stefano Borgani
 * (see change logs). All rights reserved.
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License v3 or later.
 * See <https://www.gnu.org/licenses/>.
 */

#ifndef BLACKHOLES_BEGRUN_HPP
#define BLACKHOLES_BEGRUN_HPP

#include "blackhole.hpp"
/**
 * @brief Checks and reports the settings and configuration for black hole features.
 *
 * This function validates and reports on the configuration of black hole-related settings. It performs a series
 * of checks to ensure that the black hole model settings are consistent and valid based on the compile-time
 * options and the runtime parameters.
 */
void check_and_report_settings_for_blackhole()
{
#ifndef BLACK_HOLES
    ENDRUN_IF(All.BlackHolesOn != 0,
              "Blackholes switched on in param file but BLACK_HOLES was not set at compile time");
#else // BLACK_HOLES
    // Check if On/Off settings are correct
    if (All.BlackHolesOn == 0)
    {
        VERBOSE(0, "BH: Blackhole model is available, but switched off\n");
    }
    else
    {
        if (All.BlackHolesOn == 1)
        {
            VERBOSE_ALL(2, "BH: Blackhole model is activated");
        }
        else
        {
            ENDRUN_MAIN("BH: Blackhole model is available, therefore BlackHolesOn has to be 0/1 !");
        }
        // Check if other needed modules are acailable
#ifndef FOF
        VERBOSE(0, "BH: WARNING: FOF was not set at compile time, therefore no new black holes will be seeded ! ");
#endif // FOF
#ifndef EVALPOTENTIAL
        ENDRUN_IF(All.BlackHoleRepositioningOn == 1,
                  "BH: Repositioning is switched on but needs EVALPOTENTIAL to be set at compiletime !");
        ENDRUN_IF(All.BlackHoleSeedStarMassFraction > 0, "BH: Seeding on Stars implies to seed on minimum binding "
                                                         "energy and  needs EVALPOTENTIAL to be set at compiletime !");
#endif // EVALPOTENTIAL

        // Check validity of On/Off switches
        ENDRUN_IF(All.BlackHoleSwallowGasOn < 0 || All.BlackHoleSwallowGasOn > 2,
                  "BH: BlackHoleSwallowGasOn has to be 0/1/2 !");
        ENDRUN_IF(All.BlackHoleDetails < 0 || All.BlackHoleDetails > 4,
                  "BH: BlackHoleDetails has to be 0,1,2,3 or 4 !");
        ENDRUN_IF(All.BlackHoleRepositioningOn < 0 || All.BlackHoleRepositioningOn > 1,
                  "BH: BlackHoleRepositioningOn has to be 0/1 !");
        ENDRUN_IF(All.BlackHoleHotColdAccretionOn < -1 || All.BlackHoleHotColdAccretionOn > 1,
                  "BH: BlackHoleHotColdAccretionOn has to be -1/0/1 !");
        ENDRUN_IF(All.BlackHoleVelocityInBondiOn < 0 || All.BlackHoleVelocityInBondiOn > 1,
                  "BH: BlackHoleVelocityInBondiOn has to be 0/1 !");
        ENDRUN_IF(All.BlackHoleVariableAccretionFactorOn < 0 || All.BlackHoleVariableAccretionFactorOn > 1,
                  "BH: BlackHoleVariableAccretionFactorOn has to be 0/1 !");
        ENDRUN_IF(All.BlackHoleRadiativeEfficiencyTreatment < 0 || All.BlackHoleRadiativeEfficiencyTreatment > 3,
                  "BH: BlackHoleRadiativeEfficiencyTreatment has to be 0, 1, 2 or 3 !");
        ENDRUN_IF(All.BlackHoleLimitFeedbackOn < 0 || All.BlackHoleLimitFeedbackOn > 3,
                  "BH: BlackHoleLimitFeedbackOn has to be 0/1/2/3 !");
        ENDRUN_IF(All.BlackHoleFrictionForceOn < 0 || All.BlackHoleFrictionForceOn > 2,
                  "BH: BlackHoleFrictionForceOn has to be 0/1/2 !");
#ifdef LB_PRESSURE_DEPENDENT_ACCRETION
        ENDRUN_IF((All.BlackHoleLimitFeedbackOn & 2) != 2,
                  "BH: LB_PRESSURE_DEPENDENT_ACCRETION is defined, but BlackHoleLimitFeedbackOn is 0 or 1 !");
#else // LB_PRESSURE_DEPENDENT_ACCRETION
        ENDRUN_IF((All.BlackHoleLimitFeedbackOn & 2) == 2,
                  "BH: BlackHoleLimitFeedbackOn implies pressure dependent limiter for accretion, but "
                  "LB_PRESSURE_DEPENDENT_ACCRETION is not defined!");
#endif // LB_PRESSURE_DEPENDENT_ACCRETION

        if (All.BlackHoleRadiativeEfficiencyTreatment == 2 && All.BlackHoleRadioModeTreshold != 0.05)
        {
            if (EXPERT_LEVEL == 0)
            {
                ENDRUN_MAIN("BH: The outflow model in Steinborn+15 assumes BlackHoleRadioModeTreshold = 0.05. "
                            "Please note that the BlackHoleRadioModeTreshold will be ignored.");
            }
            else
            {
                VERBOSE(0, "BH: WARNING: BlackHoleRadioModeTreshold will be ignored as the outflow model in "
                           "Steinborn+15 is using 0.05 in "
                           "any case!");
            }
        }

        // Warning for expert modules:
        if (All.BlackHoleFrictionForceOn == 0)
        {
            VERBOSE(0, "BH: WARNING: FrictionForce is turned off, check results carefully and use on own risk! ");
        }

        if (All.BlackHoleRepositioningOn == 1)
        {
            VERBOSE(0, "BH: WARNING: you are using BH Repositioning, which is an expert module, check results "
                       "carefully and use "
                       "on own risc ! ");
        }

        VERBOSE_IF(All.BlackHoleVariableAccretionFactorOn == 1, 0,
                   "BH: WARNING: using VariableAccretionFactor, check results carefully "
                   "and use on own risc ! ");

#ifndef MV_GM_AGNMUPPI
        if (All.BlackHoleThermalFeedbackOn == 0 && All.BlackHoleKineticFeedbackOn == 0)
        {
            if (EXPERT_LEVEL == 0)
            {
                ENDRUN_MAIN(
                    "BH: You are running with black holes but without any feedback, please switch to EXPERT_LEVEL "
                    ">= 1 !");
            }
            else
            {
                VERBOSE(0,
                        "BH: WARNING: You are running with black holes but without any feedback, use on own risc ! ");
            }
        }
#endif // MV_GM_AGNMUPPI

        // General warnings for expert modules:
        VERBOSE_IF(All.BlackHoleDetails >= 1, 0,
                   "BH: WARNING: Details are written by every CPU which can lead to file system problems for "
                   "very large number of MPI ranks ! ");
        VERBOSE_IF(All.BlackHoleVelocityInBondiOn == 0, 0,
                   "BH: WARNING: Not using VelocityInBondi is not recommended ! ");
        VERBOSE_IF(All.BlackHoleVelocityInBondiOn == 0, 0,
                   "BH: WARNING: Not using VelocityInBondi is not recommended ! ");
        VERBOSE_IF(All.BlackHoleIgnoreMomentum == 0 && All.BlackHoleRepositioningOn == 0, 0,
                   "BH: WARNING: accounting for Momentum and not Repositioning is not recommended !");
    }
#endif // BLACK_HOLES
}

/**
 * @brief Reports credits and references related to black hole treatments and configurations.
 *
 * This function outputs information and credits regarding the black hole treatment settings if black hole features
 * are enabled. It provides messages indicating which black hole-related models or
 * methods are currently in use, along with appropriate references for each setting.
 */
void report_credits_for_blackhole()
{
#ifdef BLACK_HOLES
    // Check if On/Off settings are correct
    VERBOSE_IF(All.BlackHolesOn == 1, 0,
               "##  BH: Black hole treatment is activated (refer to Springel et al. 2005, MNRAS, 361, "
               "776, https://ui.adsabs.harvard.edu/abs/2005MNRAS.361..776S/abstract)");

    // Give credits:
    VERBOSE_IF(All.BlackHoleFrictionForceOn == 1 || All.BlackHoleFrictionForceOn == 2, 0,
               "##  BH: Explicitly using dynamical friction force (refer to Damiano et al 2024, 2025, submitted to "
               "A&A, https://ui.adsabs.harvard.edu/abs/2024arXiv240312600D/abstract)");
    VERBOSE_IF(All.BlackHoleRepositioningOn == 0, 0,
               "##  BH: No pinning of the black hole tracers (refer to Hirschmann et al. 2014, MNRAS, 442, "
               "2304, https://ui.adsabs.harvard.edu/abs/2014MNRAS.442.2304H/abstract");
    VERBOSE_IF(All.BlackHoleRepositioningOn == 0, 0,
               "##  BH:                                       and Damiano et al 2024, submitted to A&A, "
               "https://ui.adsabs.harvard.edu/abs/2024arXiv240312600D/abstract)");
    VERBOSE_IF(All.BlackHoleRadiativeEfficiencyTreatment == 1 || All.BlackHoleRadiativeEfficiencyTreatment == 2, 0,
               "##  BH: Using mass and accreation dependent efficiencies (refer to Steinborn et al 2015, MNRAS, "
               "448, 1504, https://ui.adsabs.harvard.edu/abs/2015MNRAS.448.1504S/abstract)");
    VERBOSE_IF(All.BlackHoleRadiativeEfficiencyTreatment == 3, 0,
               "##  BH: Using mass and accreation dependent efficiencies (refer to Sala et al 2024, A&A, 685, "
               "92 https://ui.adsabs.harvard.edu/abs/2024A&A...685A..92S/abstract)");
    VERBOSE_IF(All.BlackHoleSeedStarMassFraction > 0, 0,
               "##  BH: Seeding black holes in stellar bodies (refer to Hirschmann et al. 2014, MNRAS, 442, "
               "2304, https://ui.adsabs.harvard.edu/abs/2014MNRAS.442.2304H/abstract)");
    VERBOSE_IF(All.BlackHoleHotColdAccretionOn == 1, 0,
               "##  BH: Using hot and cold gas for distinguished accreation chanels (refer to Steinborn et al "
               "2015, MNRAS, 448, 1504, https://ui.adsabs.harvard.edu/abs/2015MNRAS.448.1504S/abstract)");
    VERBOSE_IF(All.BlackHoleRadiativeEfficiencyTreatment == 2, 0,
               "##  BH: Using outflow efficiencies fit from observations (refer to Steinborn et al 2015, MNRAS, "
               "448, 1504, https://ui.adsabs.harvard.edu/abs/2015MNRAS.448.1504S/abstract)");
    VERBOSE_IF((All.BlackHoleLimitFeedbackOn & 2) == 2, 0,
               "##  BH: Applying preasure limiter for AGN feedback (refer to Vogelsberger et al. 2014, MNRAS, "
               "444, 1518, https://ui.adsabs.harvard.edu/abs/2014MNRAS.444.1518V/abstract)");
    VERBOSE_IF(All.BlackHoleRadioModeFeedbackBoost > 1, 0,
               "##  BH: Using radio mode feedback boost (refer to Fabjan et 2010, MNRAS, 401, 1670, "
               "https://ui.adsabs.harvard.edu/abs/2010MNRAS.401.1670F)");
    VERBOSE_IF(All.BlackHoleVariableAccretionFactorOn == 1, 0,
               "##  BH: Using resolution dependent bost factot for accreation (refer to Booth & Schaye 2013, "
               "NatSR, 1738, https://ui.adsabs.harvard.edu/abs/2014MNRAS.444.1518V/abstract)");
#endif // BLACK_HOLES
}

/**
 * @brief Generates and writes various black hole model data to output files.
 *
 * This function computes and outputs black hole model data into multiple files based on the
 * specified ranges and binning parameters. It produces the following files:
 *
 * - BH_MODEL_Mbh_Medd_MaxRadEff.txt: Contains columns for black hole mass,
 *   Eddington ratio (Mdot/Eddington), and maximum radiative efficiency.
 * - BH_MODEL_RadEff.txt: Contains columns for black hole mass and radiative efficiency as a function of Eddington
 * ratio.
 * - BH_MODEL_OutflowEff.txt: Contains columns for black hole mass and outflow efficiency as a function of Eddington
 * ratio.
 * - BH_MODEL_TotEff.txt: Contains columns for black hole mass and total efficiency as a function of Eddington ratio.
 * - BH_MODEL_TotFeedEff.txt: Contains columns for black hole mass and total feedback efficiency as a function of
 * Eddington ratio.
 */
void report_internal_model_for_blackhole()
{
#ifdef BLACK_HOLES
    FILE *fmodel;

    int N_BIN_BH_MASS = 20;
    MyAtLeastDouble MIN_BH_MASS = 5.0;
    MyAtLeastDouble MAX_BH_MASS = 9.0;
    int N_BIN_FEDD = 20;
    MyAtLeastDouble MIN_FEDD = -8.0;
    MyAtLeastDouble MAX_FEDD = 1.3;
    if (All.BlackHoleRadiativeEfficiencyTreatment == 3)
    {
        MIN_BH_MASS = 5.0;
        MAX_BH_MASS = 10.0;
        MIN_FEDD = -3.0; // In this model the variable is reinterpreted as mdot, in Log(Msun/yr)
        MAX_FEDD = 2;
    }
    MyFloat frac;
    MyFloat mbh;

    if (!(fmodel = fopen("BH_MODEL_Mbh_Medd_MaxRadEff.txt", "w")))
    {
        ENDRUN_MAIN("Could not open <BH_MODEL_Medd_Mbh.txt> for writing !");
    }
    fprintf(fmodel, "# Mbh/1e10Msun Medd MaxRadEff\n");
    for (int i = 0; i < N_BIN_BH_MASS; i++)
    {
        frac = (double)i / (N_BIN_BH_MASS - 1);
        mbh = pow(10.0, MIN_BH_MASS + frac * (MAX_BH_MASS - MIN_BH_MASS)) / 1e10;
        MyAtLeastDouble medd = Mdot_Eddington(mbh);
        MyAtLeastDouble MaxRadEff = BlackHoleMaximumRadiativeEfficiency(mbh);
        fprintf(fmodel, "%1.2e %1.2e %1.2e\n", mbh, medd, MaxRadEff);
    }
    fclose(fmodel);

    if (!(fmodel = fopen("BH_MODEL_RadEff.txt", "w")))
    {
        ENDRUN_MAIN("Could not open <BH_MODEL_RadEff.txt> for writing !");
    }
    (All.BlackHoleRadiativeEfficiencyTreatment == 3) ? fprintf(fmodel, "# Mbh(1e10Msun) mdot(Msun/yr) -> \n")
                                                     : fprintf(fmodel, "# Mbh(1e10Msun) f_edd -> \n");

    fprintf(fmodel, "         ");
    for (int j = 0; j < N_BIN_FEDD; j++)
    {
        frac = (double)j / (N_BIN_FEDD - 1);
        MyFloat fedd = pow(10.0, MIN_FEDD + frac * (MAX_FEDD - MIN_FEDD));
        fprintf(fmodel, "%1.2e ", fedd);
    }
    fprintf(fmodel, "\n");
    for (int i = 0; i < N_BIN_BH_MASS; i++)
    {
        frac = (double)i / (N_BIN_BH_MASS - 1);
        mbh = pow(10.0, MIN_BH_MASS + frac * (MAX_BH_MASS - MIN_BH_MASS)) / 1e10;
        fprintf(fmodel, "%1.2e ", mbh);
        for (int j = 0; j < N_BIN_FEDD; j++)
        {
            frac = (double)j / (N_BIN_FEDD - 1);
            MyFloat fedd = pow(10.0, MIN_FEDD + frac * (MAX_FEDD - MIN_FEDD));
            if (All.BlackHoleRadiativeEfficiencyTreatment == 3)
            {
                fedd /= (All.UnitMass_in_g / All.UnitTime_in_s / SOLAR_MASS *
                         SEC_PER_YEAR); // convert back Msun/yr in internal units to pass to the function.
            }

            MyAtLeastDouble RadEff = BlackHoleRadiativeEfficiency(fedd, mbh);
            fprintf(fmodel, "%1.2e ", RadEff);
        }
        fprintf(fmodel, "\n");
    }
    fclose(fmodel);

    if (!(fmodel = fopen("BH_MODEL_OutflowEff.txt", "w")))
    {
        ENDRUN_MAIN("Could not open <BH_MODEL_OutflowEff.txt> for writing !");
    }
    (All.BlackHoleRadiativeEfficiencyTreatment == 3) ? fprintf(fmodel, "# Mbh(1e10Msun) mdot(Msun/yr) -> \n")
                                                     : fprintf(fmodel, "# Mbh(1e10Msun) f_edd -> \n");
    fprintf(fmodel, "         ");
    for (int j = 0; j < N_BIN_FEDD; j++)
    {
        frac = (double)j / (N_BIN_FEDD - 1);
        MyFloat fedd = pow(10.0, MIN_FEDD + frac * (MAX_FEDD - MIN_FEDD));
        fprintf(fmodel, "%1.2e ", fedd);
    }
    fprintf(fmodel, "\n");
    for (int i = 0; i < N_BIN_BH_MASS; i++)
    {
        frac = (double)i / (N_BIN_BH_MASS - 1);
        mbh = pow(10.0, MIN_BH_MASS + frac * (MAX_BH_MASS - MIN_BH_MASS)) / 1e10;
        fprintf(fmodel, "%1.2e ", mbh);
        for (int j = 0; j < N_BIN_FEDD; j++)
        {
            frac = (double)j / (N_BIN_FEDD - 1);
            MyFloat fedd = pow(10.0, MIN_FEDD + frac * (MAX_FEDD - MIN_FEDD));
            if (All.BlackHoleRadiativeEfficiencyTreatment == 3)
            {
                fedd /= (All.UnitMass_in_g / All.UnitTime_in_s / SOLAR_MASS *
                         SEC_PER_YEAR); // convert back Msun/yr in internal units to pass to the function.
            }
            MyAtLeastDouble RadEff = BlackHoleOutflowEfficiency(fedd, mbh);
            fprintf(fmodel, "%1.2e ", RadEff);
        }
        fprintf(fmodel, "\n");
    }
    fclose(fmodel);

    if (!(fmodel = fopen("BH_MODEL_TotEff.txt", "w")))
    {
        ENDRUN_MAIN("Could not open <BH_MODEL_TotEff.txt> for writing !");
    }
    (All.BlackHoleRadiativeEfficiencyTreatment == 3) ? fprintf(fmodel, "# Mbh(1e10Msun) mdot(Msun/yr) -> \n")
                                                     : fprintf(fmodel, "# Mbh(1e10Msun) f_edd -> \n");
    fprintf(fmodel, "         ");
    for (int j = 0; j < N_BIN_FEDD; j++)
    {
        frac = (double)j / (N_BIN_FEDD - 1);
        MyFloat fedd = pow(10.0, MIN_FEDD + frac * (MAX_FEDD - MIN_FEDD));
        fprintf(fmodel, "%1.2e ", fedd);
    }
    fprintf(fmodel, "\n");
    for (int i = 0; i < N_BIN_BH_MASS; i++)
    {
        frac = (double)i / (N_BIN_BH_MASS - 1);
        mbh = pow(10.0, MIN_BH_MASS + frac * (MAX_BH_MASS - MIN_BH_MASS)) / 1e10;
        fprintf(fmodel, "%1.2e ", mbh);
        for (int j = 0; j < N_BIN_FEDD; j++)
        {
            frac = (double)j / (N_BIN_FEDD - 1);
            MyFloat fedd = pow(10.0, MIN_FEDD + frac * (MAX_FEDD - MIN_FEDD));
            if (All.BlackHoleRadiativeEfficiencyTreatment == 3)
            {
                fedd /= (All.UnitMass_in_g / All.UnitTime_in_s / SOLAR_MASS *
                         SEC_PER_YEAR); // convert back Msun/yr in internal units to pass to the function.
            }
            MyAtLeastDouble RadEff = BlackHoleTotalEfficiency(fedd, mbh);
            fprintf(fmodel, "%1.2e ", RadEff);
        }
        fprintf(fmodel, "\n");
    }
    fclose(fmodel);

    if (!(fmodel = fopen("BH_MODEL_TotFeedEff.txt", "w")))
    {
        ENDRUN_MAIN("Could not open <BH_MODEL_TotFeedEff.txt> for writing !");
    }
    (All.BlackHoleRadiativeEfficiencyTreatment == 3) ? fprintf(fmodel, "# Mbh mdot -> \n")
                                                     : fprintf(fmodel, "# Mbh f_edd -> \n");
    fprintf(fmodel, "         ");
    for (int j = 0; j < N_BIN_FEDD; j++)
    {
        frac = (double)j / (N_BIN_FEDD - 1);
        MyFloat fedd = pow(10.0, MIN_FEDD + frac * (MAX_FEDD - MIN_FEDD));
        fprintf(fmodel, "%1.2e ", fedd);
    }
    fprintf(fmodel, "\n");
    for (int i = 0; i < N_BIN_BH_MASS; i++)
    {
        frac = (double)i / (N_BIN_BH_MASS - 1);
        mbh = pow(10.0, MIN_BH_MASS + frac * (MAX_BH_MASS - MIN_BH_MASS)) / 1e10;
        fprintf(fmodel, "%1.2e ", mbh);
        for (int j = 0; j < N_BIN_FEDD; j++)
        {
            frac = (double)j / (N_BIN_FEDD - 1);
            MyFloat fedd = pow(10.0, MIN_FEDD + frac * (MAX_FEDD - MIN_FEDD));
            if (All.BlackHoleRadiativeEfficiencyTreatment == 3)
            {
                fedd /= (All.UnitMass_in_g / All.UnitTime_in_s / SOLAR_MASS *
                         SEC_PER_YEAR); // convert back Msun/yr in internal units to pass to the function.
            }

            MyAtLeastDouble RadEff = BlackHoleTotalFeedbackEfficiency(fedd, mbh);
            fprintf(fmodel, "%1.2e ", RadEff);
        }
        fprintf(fmodel, "\n");
    }
    fclose(fmodel);

#endif // BLACK_HOLES
}

#endif // BLACKHOLES_BEGRUN_HPP
