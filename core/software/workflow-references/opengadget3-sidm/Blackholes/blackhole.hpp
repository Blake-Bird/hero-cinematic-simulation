/**
 * @file
 *
 * This file is part of the N-body/SPH code OpenGadget3.
 *
 * @brief This header file contains functions and definitions related to the black hole module.
 *
 * @author OpenGadget3-development team, in particular Alice Damiano, Klaus Dolag, Luca Sala and Volker Springel.
 *
 * @copyright Copyright (c) 2026, OpenGadget3-development team lead by Klaus Dolag and Stefano Borgani
 * (see change logs). All rights reserved.
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License v3 or later.
 * See <https://www.gnu.org/licenses/>.
 */

#ifndef BLACKHOLES_HPP
#define BLACKHOLES_HPP

#ifdef BLACK_HOLES

extern MyAtLeastDouble hubble_a, ascale;
constexpr MyAtLeastDouble SALPETER_TIME_IN_S =
    (C_LIGHT * THOMSON) / (4 * M_PI * GRAVITY * PROTONMASS); /* Salpeter time (450 Myr) in seconds. */
constexpr MyAtLeastDouble BlackHoleSteinbornEfficiencyModelSlope =
    0.52; /* Slope for mass dependent efficiency (see Steinborn+ 2015) */
constexpr MyAtLeastDouble BlackHoleSteinbornEfficiencyModelNorm =
    0.089; /* Norm for mass dependent efficiency (see Steinborn+ 2015) */
constexpr MyAtLeastDouble BlackHoleSteinbornEfficiencyModelMaxValue =
    0.42; /* Maximum efficiency (see Steinborn+ 2015) */
constexpr MyAtLeastDouble BlackHoleSteinbornEfficiencyModelMinValue =
    0.057; /* Minimum efficiency (see Steinborn+ 2015) */
constexpr MyAtLeastDouble BlackHoleSalaEfficiencyModelNorm =
    4.16e-06; /* Norm for mdot-mbh fit for efficiency (model by L. Sala) */
constexpr MyAtLeastDouble BlackHoleSalaEfficiencyModelSlopeMdot =
    -0.33552319; /* Mdot exponent for mdot-mbh efficiency fit (model by L. Sala) */
constexpr MyAtLeastDouble BlackHoleSalaEfficiencyModelSlopeMbh =
    0.53549928; /* Mbh exponent for mdot-mbh efficiency fit (model by L. Sala) */
constexpr MyAtLeastDouble MinimumAllowedDimensionlessSpinValue =
    1e-3; /* Minimum value for dimensionless spin parameter (a) for black holes.*/

#define BHPOTVALUEINIT 1.0e30

/***********************************************************************************/
/************************** Helper functions for model parameters ******************/
/***********************************************************************************/

/**
 * @brief Computes the maximum radiative efficiency of a black hole based on its mass.
 *
 * This function calculates the radiative efficiency of a black hole using different models
 * depending on the configuration flags. The efficiency can be either constant, variable, or
 * set to a fiducial value based on specific conditions.
 *
 * @param bh_mass The mass of the black hole. This value is used in calculations when the
 *                variable efficiency model is selected.
 *
 * @return The computed radiative efficiency of the black hole.
 */
static inline MyAtLeastDouble BlackHoleMaximumRadiativeEfficiency(MyFloat bh_mass)
{
    if (All.BlackHoleRadiativeEfficiencyTreatment == 3)
    {
        return (MyAtLeastDouble)All.BlackHoleRadiativeEfficiency;
    }
    else if (All.BlackHoleRadiativeEfficiencyTreatment == 1 || All.BlackHoleRadiativeEfficiencyTreatment == 2)
    {
        MyAtLeastDouble efficiency = BlackHoleSteinbornEfficiencyModelNorm *
                                     pow(bh_mass * 100 / All.HubbleParam, BlackHoleSteinbornEfficiencyModelSlope);
        if (efficiency > BlackHoleSteinbornEfficiencyModelMaxValue) // theoretical maximum value (spin=1)
        {
            efficiency = BlackHoleSteinbornEfficiencyModelMaxValue;
        }
        if (efficiency < BlackHoleSteinbornEfficiencyModelMinValue)
        {
            efficiency = BlackHoleSteinbornEfficiencyModelMinValue;
        }
        return efficiency;
    }
    else
    {
        return (MyAtLeastDouble)All.BlackHoleRadiativeEfficiency;
    }
}

/**
 * @brief Computes the Eddington accretion rate of a black hole.
 *
 * This function calculates the Eddington accretion rate (i.e., the maximum accretion rate
 * allowed by radiation pressure) for a black hole based on its mass.
 *
 * @param bh_mass The mass of the black hole, which is used to compute the Eddington rate.
 *
 * @return The Eddington accretion rate for the given black hole mass.
 */
static inline MyAtLeastDouble Mdot_Eddington(MyFloat bh_mass)
{
    MyAtLeastDouble mdot_edd = (bh_mass / All.HubbleParam) / (SALPETER_TIME_IN_S / All.UnitTime_in_s) /
                               BlackHoleMaximumRadiativeEfficiency(bh_mass);
    return mdot_edd; /* This is in code units, and has the same units as BH_Mdot. */
}

/**
 * @brief Computes the outflow efficiency of a black hole based on its Eddington ratio and mass.
 *
 * This function calculates the outflow efficiency of a black hole according to a specific
 * model, which is active if `All.BlackHoleRadiativeEfficiencyTreatment` is set to `2`. The efficiency
 * depends on the Eddington ratio (`f_Edd`) and follows different formulas based on the value
 * of `f_Edd`.
 *
 * @param f_Edd The Eddington ratio of the black hole. This value influences the efficiency
 *              calculation.
 * @param bh_mass The mass of the black hole.
 *
 * @return The calculated outflow efficiency of the black hole. The value is determined
 *         based on the Eddington ratio and the specific model active.
 */
static inline MyAtLeastDouble BlackHoleOutflowEfficiency(MyFloat f_Edd, MyFloat bh_mass)
{
    if (All.BlackHoleRadiativeEfficiencyTreatment == 2)
    {
        if (f_Edd < 0.05)
        {
            MyAtLeastDouble beta = 0.5; // slope
            MyAtLeastDouble gamma = pow(10., -4.0) * pow(0.05, (-2.8431 - beta));
            return 0.1 - (gamma * 0.1 * pow(f_Edd, beta));
        }
        else
        {
            return (MyAtLeastDouble)pow(10.0, -4.0) * 0.1 * pow(f_Edd, -2.8431);
        }
        // The slope 2.8431 was estimated numerically: -5-log(0.05)/(log(0.05))
    }
    else
        return (MyAtLeastDouble)0;
}

/**
 * @brief Computes the radiative efficiency of a black hole based on its Eddington ratio and mass.
 *
 * This function calculates the radiative efficiency of a black hole using different models
 * depending on the configuration flag `All.BlackHoleRadiativeEfficiencyTreatment`. The efficiency
 * is calculated based on the Eddington ratio (`f_Edd`) and the black hole mass (`bh_mass`).
 *
 * @param f_Edd The Eddington ratio of the black hole, representing the ratio of the actual
 *              accretion rate to the Eddington rate.
 * @param bh_mass The mass of the black hole, used in some models to calculate the efficiency.
 *
 * @return The computed radiative efficiency of the black hole.
 */
static inline MyAtLeastDouble BlackHoleRadiativeEfficiency(MyFloat f_Edd, MyFloat bh_mass)
{
    if (All.BlackHoleRadiativeEfficiencyTreatment == 1 || All.BlackHoleRadiativeEfficiencyTreatment == 2)
    {
        if (f_Edd < 0.05)
        {
            MyAtLeastDouble beta = 0.5; // slope
            MyAtLeastDouble gamma = pow(10., -4.0) * pow(0.05, (-2.8431 - beta));
            return gamma * BlackHoleMaximumRadiativeEfficiency(bh_mass) * pow(f_Edd, beta);
        }
        else
        {
            return BlackHoleMaximumRadiativeEfficiency(bh_mass) * (1.0 - (pow(10.0, -4.0) * pow(f_Edd, -2.8431)));
        }
    }
    else if (All.BlackHoleRadiativeEfficiencyTreatment == 3)
    {
        MyAtLeastDouble mdot = f_Edd * Mdot_Eddington(bh_mass);
        MyAtLeastDouble mdotphys = mdot * All.UnitMass_in_g / All.UnitTime_in_s / SOLAR_MASS *
                                   SEC_PER_YEAR; // Convert in Msun/yr (the quantities are passed in internal units;
        MyAtLeastDouble mbh = bh_mass * All.UnitMass_in_g / SOLAR_MASS / All.HubbleParam; // Convert in Msun/yr
        MyAtLeastDouble er = BlackHoleSalaEfficiencyModelNorm * pow(mdotphys, BlackHoleSalaEfficiencyModelSlopeMdot) *
                             pow(mbh, BlackHoleSalaEfficiencyModelSlopeMbh); // A*mdot^alpha*M^beta;
        if (er >= 0.42)
        {
            return 0.42; // theoretical maximum value (spin=0.998)
        }
        else
        {
            return er;
        }
    }
    else
    {
        return (MyAtLeastDouble)All.BlackHoleRadiativeEfficiency;
    }
}

/**
 * @brief Computes the total efficiency of a black hole by summing outflow and radiative efficiencies.
 *
 * This function calculates the total black hole efficiency as the sum of two components:
 * the outflow efficiency and the radiative efficiency. This approach follows the methodology
 * described in Davis & Laor (2010) and Chelouche (2013).
 *
 * @param f_Edd The Eddington ratio of the black hole, which influences the outflow and radiative
 *              efficiency calculations.
 * @param bh_mass The mass of the black hole, used in the efficiency calculations.
 *
 * @return The total black hole efficiency, which is the sum of:
 *         - The outflow efficiency, computed by `BlackHoleOutflowEfficiency()`
 *         - The radiative efficiency, computed by `BlackHoleRadiativeEfficiency()`
 * @note
 * - The implementation follows the references provided:
 *   - Davis & Laor, 2010, astro-ph 1012.3213
 *   - Chelouche, 2013, astro-ph 1305:6507
 *   - Steinborn et al., 2015,  eps_r or eps_0 + eps_r in equation (12)
 */
static inline MyAtLeastDouble BlackHoleTotalEfficiency(MyFloat f_Edd, MyFloat bh_mass)
{
    return BlackHoleOutflowEfficiency(f_Edd, bh_mass) + BlackHoleRadiativeEfficiency(f_Edd, bh_mass);
    // std model: BlackHoleOutflowEfficiency = 0 and BlackHoleRadiativeEfficiency = eps_r  -> returns eps_r
}

/**
 * @brief Computes the feedback coupling efficiency of a black hole, used in some models, including a boost factor in
 * radio mode.
 *
 * This function computes the Eddington-ratio dependent feedback coupling efficiency, which
 * is used in some models, including a boost in radio mode. This coupling efficiency depends on BlackHoleFeedbackFactor,
 * BlackHoleRadioModeBoost, and the chosen Eddington ratio threshold.
 *
 * @param f_Edd The Eddington ratio of the black hole, which determines the threshold below which
 * BlackHoleFeedbackFactor needs to be multiplied by thhe boost factor.
 *
 * @return The coupling efficiency.
 */
static inline MyAtLeastDouble BlackHoleCouplingEfficiency(MyFloat f_Edd)
{
    MyAtLeastDouble feedback_factor = All.BlackHoleFeedbackFactor;
    if (All.BlackHoleRadiativeEfficiencyTreatment != 2)
        if (All.BlackHoleRadioModeFeedbackBoost > 1)
            if (f_Edd < All.BlackHoleRadioModeTreshold)
                feedback_factor *= All.BlackHoleRadioModeFeedbackBoost;

    return feedback_factor;
}

/**
 * @brief Computes the total feedback efficiency of a black hole, incorporating a feedback boost factor.
 *
 * This function calculates the total feedback efficiency by combining the outflow efficiency
 * and the radiative efficiency, with an additional feedback factor that can be boosted in
 * radio mode. The calculation is based on the model described in Steinborn et al. (2015),
 * equation (9).
 *
 * @param f_Edd The Eddington ratio of the black hole, which influences the outflow and radiative
 *              efficiency calculations.
 * @param bh_mass The mass of the black hole, used in the efficiency calculations.
 *
 * @return The total feedback efficiency.
 *
 * @note
 * - The implementation follows the approach described in:
 *   - Steinborn et al., 2015, eps_r * eps_f (where eps_f can be boosted in radio mode) or eps_0 + eps_r * eps_f in
 * equation (9)
 */
static inline MyAtLeastDouble BlackHoleTotalFeedbackEfficiency(MyFloat f_Edd, MyFloat bh_mass)
{
    return BlackHoleOutflowEfficiency(f_Edd, bh_mass) +
           BlackHoleRadiativeEfficiency(f_Edd, bh_mass) * BlackHoleCouplingEfficiency(f_Edd);
    // std model: BlackHoleOutflowEfficiency = 0 and BlackHoleRadiativeEfficiency = eps_r  ->  returns eps_r * eps_f
}

/**
 * @brief Calculates the Bondi accretion rate and adjusts it based on various conditions and factors.
 *
 * This function computes the Bondi accretion rate for a black hole given the density, gas velocity,
 * and entropy of the surrounding medium. The calculation incorporates several factors including
 * the black hole's velocity, a variable accretion factor, and adjustments based on feedback mechanisms
 * and pressure-dependent accretion models.
 *
 * @param n The index of the black hole particle in the simulation, used to access particle-specific properties.
 * @param Rho The density of the surrounding gas.
 * @param GasVel Array containing the velocity components of the surrounding gas [vx, vy, vz].
 * @param Entropy The entropy of the surrounding gas.
 * @param AccretionFactor A base factor for accretion, potentially modified by other conditions.
 * @param dt The time step used for integration, required for pressure-dependent adjustments.
 * @param mdot Pointer to a variable where the computed Bondi accretion rate will be stored.
 */
static inline void Get_Bondi_Parameter(const int n, const MyFloat Rho, const MyFloat GasVel[3], const MyFloat Entropy,
                                       const double AccretionFactor, MyAtLeastDouble dt, MyAtLeastDouble *mdot,
                                       std::ostringstream &bh_log)
{
    MyAtLeastDouble rho = Rho, bhvel;
    if (All.BlackHoleVelocityInBondiOn)
    {
        bhvel =
            sqrt(pow(P[n].Vel[0] - GasVel[0], 2) + pow(P[n].Vel[1] - GasVel[1], 2) + pow(P[n].Vel[2] - GasVel[2], 2));
    }
    else
    {
        bhvel = 0;
    }

    if (All.ComovingIntegrationOn)
    {
        bhvel /= All.Time;
        rho /= pow(All.Time, 3);
    }
    MyAtLeastDouble soundspeed = sqrt(GAMMA * Entropy * pow(rho, GAMMA_MINUS1));

    MyAtLeastDouble accretionfactor = AccretionFactor;
    if (All.BlackHoleVariableAccretionFactorOn == 1)
    {
#ifdef SFR
        double rho_threshold;
        rho_threshold = All.PhysDensThresh;

        if (rho > rho_threshold)
        {
            accretionfactor = pow(rho / rho_threshold, All.BlackHoleVariableAccretionSlope);
        }
        else
        {
            accretionfactor = 1.0;
        }
#else // SFR
        accretionfactor = 1.0;
#endif // SFR
    }

    MyAtLeastDouble norm = pow((pow(soundspeed, 2) + pow(bhvel, 2)), 1.5);
    if (norm > 0)
    {
        *mdot = 4. * M_PI * accretionfactor * All.GravityConstantInternal * All.GravityConstantInternal *
                pow(BPP(n).BH_Mass, 2) * (rho) / norm;
    }
    else
    {
        *mdot = 0;
    }

    if (All.BlackHoleDetails >= 1)
    {
        bh_log << " mdot=" << (double)(*mdot) << " rho=" << (double)rho << " csnd=" << (double)soundspeed
               << " bhvel=" << (double)bhvel << " ";
    }

    // Nitya adding check for another variable, that's causing issue with medium workload

#if defined(LB_PRESSURE_DEPENDENT_ACCRETION) && defined(LT_NMet)
    if ((All.BlackHoleLimitFeedbackOn & 2) == 2)
    {
        MyAtLeastDouble M_feedback = BPP(n).BH_SurroundingGasMass;
        MyAtLeastDouble Lambda_input =
            All.BlackHoleFeedbackFactor *
            BlackHoleRadiativeEfficiency(*mdot / Mdot_Eddington(BPP(n).BH_Mass), BPP(n).BH_Mass) * C_LIGHT * C_LIGHT *
            4. * M_PI * All.BlackHoleAccretionFactor * GRAVITY * GRAVITY * BPP(n).BH_Mass * BPP(n).BH_Mass *
            PROTONMASS * PROTONMASS * All.UnitMass_in_g /
            (pow(soundspeed * All.UnitVelocity_in_cm_per_s, 3) * M_feedback); // richtige Formel!
        MyAtLeastDouble Redshift, DZ;

        if (All.ComovingIntegrationOn)
        {
            Redshift = 1.0 / All.Time - 1;
            get_cool_redshift(Redshift, &DZ);
        }
        else
        {
            Redshift = 0;
            DZ = 0;
        }

        MyAtLeastDouble Temperature = BPP(n).BH_SurroundingTemperature / rho;
        MyAtLeastDouble Metallicity[LT_NMet];
        Metallicity[0] = BPP(n).BH_SurroundingMetallicity / rho;
        int q;
        for (q = 1; q < LT_NMet; q++)
        {
            Metallicity[q] = 0.0;
        }
        MyAtLeastDouble u_eq =
            GetUFromLambda(Lambda_input, rho, &Metallicity[0], Redshift, DZ, dt, &Temperature, 1e-12, 1e-6);
        MyAtLeastDouble PhysDensThresh = BPP(n).BH_SfrThreshold / rho;
        MyAtLeastDouble P_ref = (GAMMA_MINUS1)*PhysDensThresh * u_eq * All.UnitPressure_in_cgs;
        MyAtLeastDouble P_external = BPP(n).BH_SurroundingGasPressure / rho;
        MyAtLeastDouble factor = P_external * P_external / (P_ref * P_ref);
        if (P_external < P_ref)
        {
            VERBOSE_ALL(1,
                        "WARNING: Plimiter: Lambda=%g, u_eq=%g, GAMMA_MINUS1=%g, PysDensThresh=%g, Metallicity=%g, "
                        "Temperature=%g ",
                        (double)Lambda_input, (double)u_eq, GAMMA_MINUS1, (double)PhysDensThresh, Metallicity[0],
                        (double)Temperature);
            VERBOSE_ALL(1, "P_ref=%g, P_ext=%g, factor=%g ", (double)P_ref, (double)P_external, (double)factor);
            *mdot *= P_external * P_external / (P_ref * P_ref);
            if (All.BlackHoleDetails >= 1)
            {
                fprintf(FdBlackHolesDetails, " PextLim=%g ", (double)P_external * P_external / (P_ref * P_ref));
            }
        }
    }
#endif // defined(LB_PRESSURE_DEPENDENT_ACCRETION) && defined(LT_NMet)
}

/**
 * @brief Computes the radius of the innermost stable circular orbit for a Kerr black hole.
 *
 * This function calculates the radius of the innermost stable circular orbit for a Kerr black hole
 * as a function of the black hole's spin parameter.
 *
 * @param a The dimensionless spin parameter of the black hole, which ranges from -1 (extremal retrograde)
 *          to +1 (extremal prograde). For a Schwarzschild black hole, this parameter is 0.
 *
 * @return The radius in geometric units of the innermost stable circular orbit for the given spin parameter.
 */
static inline MyAtLeastDouble calc_risco(const MyAtLeastDouble a)
{
    MyAtLeastDouble z1 = 1. + cbrt(1. - a * a) * (cbrt(1. + a) + cbrt(1. - a));
    MyAtLeastDouble z2 = sqrt(3 * a * a + z1 * z1);
    if (a <= 0.)
    {
        return 3. + z2 + sqrt((3. - z1) * (3. + z1 + 2 * z2));
    }
    else
    {
        return 3. + z2 - sqrt((3. - z1) * (3. + z1 + 2 * z2));
    }
}

/**
 * @brief Finds neighboring black holes within a specified search radius using a tree-based data structure.
 *
 * This function traverses a spatial tree to identify black holes that are within a specified search radius
 * from a given center point.
 *
 * @param searchcenter Array of length 3 representing the coordinates of the search center.
 * @param hsml The smoothing length (search radius) within which to find black holes.
 * @param target The index of the target black hole particle for which neighbors are being sought.
 * @param startnode Pointer to the index of the current node in the tree. This is updated by the function.
 * @param mode The mode of operation:
 *             - `0`: Normal mode
 *             - `1`: Special mode, used for top-level node checks.
 * @param exportflag Array indicating the task (process) to which each node should be exported.
 * @param exportnodecount Array counting the number of nodes to be exported for each task.
 * @param exportindex Array mapping export indices for nodes.
 * @param ngblist Array to store the indices of neighboring black holes found within the search radius.
 *
 * @return The number of neighboring black holes found within the specified radius. Returns `-1` if the
 *         buffer is full and export cannot proceed.
 */
static inline int ngb_treefind_blackhole_threads(MyLongDouble searchcenter[3], MyFloat hsml, int target, int *startnode,
                                                 int mode, int *exportflag, int *exportnodecount, int *exportindex,
                                                 int *ngblist)
{
    struct NODE *current;
#ifdef PERIODIC
    MyLongDouble xtmp;
#endif // PERIODIC
    int numngb = 0;
    int no = *startnode;

    while (no >= 0)
    {
        if (no < All.MaxPart) /* single particle */
        {
            int p = no;
            no = Nextnode[no];

            if (All.BlackHoleRepositioningOn == 0 && All.BlackHoleFrictionForceOn == 0)
            {
                if (P[p].Type != 0 && P[p].Type != 5)
                {
                    continue;
                }
            }

            MyAtLeastDouble dist = hsml;
            MyAtLeastDouble dx = NGB_PERIODIC_LONG_X(P[p].Pos[0] - searchcenter[0]);
            if (dx > dist)
            {
                continue;
            }
            MyAtLeastDouble dy = NGB_PERIODIC_LONG_Y(P[p].Pos[1] - searchcenter[1]);
            if (dy > dist)
            {
                continue;
            }
            MyAtLeastDouble dz = NGB_PERIODIC_LONG_Z(P[p].Pos[2] - searchcenter[2]);
            if (dz > dist)
            {
                continue;
            }
            if (dx * dx + dy * dy + dz * dz > dist * dist)
            {
                continue;
            }

            ngblist[numngb++] = p;
        }
        else
        {
            if (no >= All.MaxPart + MaxNodes) /* pseudo particle */
            {
                ENDRUN_IF(mode == 1, "Ending run in ngb_treefind_blackhole_threads: mode == 1");

                int task;
                if (exportflag[task = DomainTask[no - (All.MaxPart + MaxNodes)]] != target)
                {
                    exportflag[task] = target;
                    exportnodecount[task] = NODELISTLENGTH;
                }
                if (exportnodecount[task] == NODELISTLENGTH)
                {
                    int exitFlag = 0, nexp;
#ifdef _OPENMP
#pragma omp critical(_nexport_)
#endif // _OPENMP
                    {
                        if (Nexport >= All.BunchSize)
                        {
                            BufferFullFlag = 1;
                            exitFlag = 1;
                        }
                        else
                        {
                            nexp = Nexport;
                            Nexport++;
                        }
                    }

                    if (exitFlag)
                    {
                        return -1;
                    }

                    exportnodecount[task] = 0;
                    exportindex[task] = nexp;
                    DataIndexTable[nexp].Task = task;
                    DataIndexTable[nexp].Index = target;
                    DataIndexTable[nexp].IndexGet = nexp;
                }

                DataNodeList[exportindex[task]].NodeList[exportnodecount[task]++] =
                    DomainNodeIndex[no - (All.MaxPart + MaxNodes)];

                if (exportnodecount[task] < NODELISTLENGTH)
                {
                    DataNodeList[exportindex[task]].NodeList[exportnodecount[task]] = -1;
                }

                no = Nextnode[no - MaxNodes];
                continue;
            }

            current = &Nodes[no];

            if (mode == 1)
            {
                if (current->u.d.bitflags & (1 << BITFLAG_TOPLEVEL)) /* we reached a top-level node again, which means
                                                                        that we are done with the branch */
                {
                    *startnode = -1;
                    return numngb;
                }
            }

            no = current->u.d.sibling; /* in case the node can be discarded */

            MyAtLeastDouble dist = hsml + 0.5 * current->len;
            ;
            MyAtLeastDouble dx = NGB_PERIODIC_LONG_X(current->center[0] - searchcenter[0]);
            if (dx > dist)
            {
                continue;
            }
            MyAtLeastDouble dy = NGB_PERIODIC_LONG_Y(current->center[1] - searchcenter[1]);
            if (dy > dist)
            {
                continue;
            }
            MyAtLeastDouble dz = NGB_PERIODIC_LONG_Z(current->center[2] - searchcenter[2]);
            if (dz > dist)
            {
                continue;
            }
            /* now test against the minimal sphere enclosing everything */
            dist += FACT1 * current->len;
            if (dx * dx + dy * dy + dz * dz > dist * dist)
            {
                continue;
            }

            no = current->u.d.nextnode; /* ok, we need to open the node */
        }
    }

    *startnode = -1;
    return numngb;
}

/*static inline int ngb_treefind_blackhole(MyLongDouble searchcenter[3], MyFloat hsml, int target, int *startnode, int
mode, int *nexport, int *nsend_local)
{
  struct NODE *current;
#ifdef PERIODIC
  MyLongDouble xtmp;
#endif
  int nexport_save = *nexport;
  int numngb = 0;
  int no = *startnode;

  while(no >= 0)
    {
      if(no < All.MaxPart)	// single particle
    {
      int p = no;
      no = Nextnode[no];

      if(All.BlackHoleRepositioningOn == 0)
        if(P[p].Type != 0 && P[p].Type != 5)
          continue;

      MyAtLeastDouble dist = hsml;
      MyAtLeastDouble dx = NGB_PERIODIC_LONG_X(P[p].Pos[0] - searchcenter[0]);
      if(dx > dist)
        continue;
      MyAtLeastDouble dy = NGB_PERIODIC_LONG_Y(P[p].Pos[1] - searchcenter[1]);
      if(dy > dist)
        continue;
      MyAtLeastDouble dz = NGB_PERIODIC_LONG_Z(P[p].Pos[2] - searchcenter[2]);
      if(dz > dist)
        continue;
      if(dx * dx + dy * dy + dz * dz > dist * dist)
        continue;

      Ngblist[numngb++] = p;
    }
      else
    {
      if(no >= All.MaxPart + MaxNodes)	// pseudo particle
        {
        ENDRUN_IF(mode == 1,"Ending run in ngb_treefind_blackhole: mode == 1");

          int task;
          if(Exportflag[task = DomainTask[no - (All.MaxPart + MaxNodes)]] != target)
        {
          Exportflag[task] = target;
          Exportnodecount[task] = NODELISTLENGTH;
        }

          if(Exportnodecount[task] == NODELISTLENGTH)
        {
          if(*nexport >= All.BunchSize)
            {
              *nexport = nexport_save;
              for(task = 0; task < NTask; task++)
            nsend_local[task] = 0;
              for(no = 0; no < nexport_save; no++)
            nsend_local[DataIndexTable[no].Task]++;
              return -1;
            }
          Exportnodecount[task] = 0;
          Exportindex[task] = *nexport;
          DataIndexTable[*nexport].Task = task;
          DataIndexTable[*nexport].Index = target;
          DataIndexTable[*nexport].IndexGet = *nexport;
          *nexport = *nexport + 1;
          nsend_local[task]++;
        }

          DataNodeList[Exportindex[task]].NodeList[Exportnodecount[task]++] =
        DomainNodeIndex[no - (All.MaxPart + MaxNodes)];

          if(Exportnodecount[task] < NODELISTLENGTH)
        DataNodeList[Exportindex[task]].NodeList[Exportnodecount[task]] = -1;

          no = Nextnode[no - MaxNodes];
          continue;
        }

      current = &Nodes[no];

      if(mode == 1)
        {
          if(current->u.d.bitflags & (1 << BITFLAG_TOPLEVEL))	// we reached a top-level node again, which means that
we are done with the branch
        {
          *startnode = -1;
          return numngb;
        }
        }

      no = current->u.d.sibling;	// in case the node can be discarded

      MyAtLeastDouble dist = hsml + 0.5 * current->len;;
      MyAtLeastDouble dx = NGB_PERIODIC_LONG_X(current->center[0] - searchcenter[0]);
      if(dx > dist)
        continue;
      MyAtLeastDouble dy = NGB_PERIODIC_LONG_Y(current->center[1] - searchcenter[1]);
      if(dy > dist)
        continue;
      MyAtLeastDouble dz = NGB_PERIODIC_LONG_Z(current->center[2] - searchcenter[2]);
      if(dz > dist)
        continue;
      // now test against the minimal sphere enclosing everything
      dist += FACT1 * current->len;
      if(dx * dx + dy * dy + dz * dz > dist * dist)
        continue;

      no = current->u.d.nextnode;	// ok, we need to open the node
    }
    }

  *startnode = -1;
  return numngb;
}*/

#endif // BLACK_HOLES
#endif // BLACKHOLES_HPP
