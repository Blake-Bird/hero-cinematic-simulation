/**
 * @file
 *
 * This file is part of the N-body/SPH code OpenGadget3.
 *
 * @brief This source file contains the definitions for the scattering modules.
 *
 * The scattering modules are classes that are inherited by the main scattering class.
 * They specify the detailed behaviour of the scattering and contain auxiliary functionality.
 *
 * @author OpenGadget3-development team, in particular Moritz S. Fischer, Yashraj Patil,
 * Cenanda Arido, and Marc Wiertel.
 *
 * @copyright Copyright (c) 2026, OpenGadget3-development team lead by Klaus Dolag and Stefano Borgani
 * (see change logs). All rights reserved.
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License v3 or later.
 * See <https://www.gnu.org/licenses/>.
 */

#ifndef MSIDM_SCATTERING_MODULES_CPP
#define MSIDM_SCATTERING_MODULES_CPP

#include "scattering_modules.hpp"
#include "tools.hpp"
// #include <chrono>

// #define SCATTERING_CRITICALTIMERS

using namespace scatter;
using namespace scatter::tools;
using namespace scatter::pxp_scattering;

//////////////////////////////////////////////////////////////////////////////////////////////////
// module_base
/**
 * @brief Method to initialize class.
 */
template <typename SCTOPT> void module_base<SCTOPT>::init(const nlohmann::json &param)
{
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Default definition for auxiliary_method1
 *
 * Default definition for methods that are executed in the main scattering routine directly
 * before the frequent interactions (small-angle scattering).
 *
 */
template <typename SCTOPT> void module_base<SCTOPT>::auxiliary_method1(scatter_inner_base &si)
{
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Default definition for auxiliary_method2
 *
 * Default definition for methods that are executed in the main scattering routine directly
 * after the frequent interactions (small-angle scattering) and before the rare interactions
 * (large-angle scattering).
 *
 */
template <typename SCTOPT> void module_base<SCTOPT>::auxiliary_method2(scatter_inner_base &si)
{
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Default definition for auxiliary_method3
 *
 * Default definition for methods that are executed in the main scattering routine directly
 * after the rare interactions (large-angle scattering).
 *
 */
template <typename SCTOPT> void module_base<SCTOPT>::auxiliary_method3(scatter_inner_base &si)
{
}

//////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////
// debug module
/**
 * @brief Method to initialize class.
 */
template <typename SCTOPT> void module_debug<SCTOPT>::init(const nlohmann::json &param)
{
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Method to debug position and velocity of particles.
 */
template <typename SCTOPT> void module_debug<SCTOPT>::debug_pos_and_vel(const scatter_inner_base &si) const
{
    const scattering_particle_ref &pi = si.pi;
    const scattering_particle_ref &pj = si.pj;

    // check for NaNs - only for debugging
    for (int loopc = 0; loopc < 3; ++loopc)
    {
#ifdef SCATTERING_DEBUG
        check_nan(pi.Vel[loopc], "Scattering pi Vel " + std::to_string(pi.ID));
        check_nan(pj.Vel[loopc], "Scattering pj Vel " + std::to_string(pj.ID));
        check_nan(pi.Pos[loopc], "Scattering pi Pos " + std::to_string(pi.ID));
        check_nan(pj.Pos[loopc], "Scattering pj Pos " + std::to_string(pj.ID));
#endif // SCATTERING_DEBUG
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////
// scatter count module
/**
 * @brief Constructor of global scatter count class.
 */
template <typename SCTOPT>
module_scattercount<SCTOPT>::module_scattercount() : scatter_outer_base<SCTOPT>(), scatter_count(0)
{
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Method to initialize class.
 */
template <typename SCTOPT> void module_scattercount<SCTOPT>::init(const nlohmann::json &param)
{
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Method to exchange global scatter count among MPI processes.
 */
template <typename SCTOPT> void module_scattercount<SCTOPT>::exchange_scattercount() const
{
    long long unsigned int sum_scatter_count = 0;
    MPI_Allreduce(&scatter_count, &sum_scatter_count, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    if constexpr (SCTOPT::do_sidm)
    {
        coutr(important) << "rSIDM_SCATTERCOUNT: Time = " << All.Time << " SumScatterCount = " << sum_scatter_count
                         << endl;
    }
    else if constexpr (SCTOPT::do_idm)
    {
        coutr(important) << "rIDM_SCATTERCOUNT: Time = " << All.Time << " SumScatterCount = " << sum_scatter_count
                         << endl;
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Method to increase the scatter count.
 */
template <typename SCTOPT> void module_scattercount<SCTOPT>::increase_global_scattercount(scatter_inner_base &si)
{
#pragma omp critical
    {
        ++module_scattercount<SCTOPT>::scatter_count;
        if (SCTOPT::do_printscatters)
        {
#ifdef SCATTERING_DEBUG
            cout(important) << "SCATTER: IDs=[" << si.pi.ID << ", " << si.pj.ID << "], v_rel = " << si.v_rel_abs
                            << endl;
#endif // SCATTERING_DEBUG
        }
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////
// individual scatter count module
/**
 * @brief Method to increase the individual scatter count of particles.
 */
template <typename SCTOPT>
void module_scattercount_indi<SCTOPT>::increase_individual_scattercount(scatter_inner_base &si) const
{
#ifdef SCATTERING_COUNT_INDI
    // increase scatter count for both particles
    si.pi.scatterCount++;
    si.pj.scatterCount++;
#endif // SCATTERING_COUNT_INDI
}

//////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////
// scatter once module
/**
 * @brief Method to allow particles to scatter only once.
 */
template <typename SCTOPT> bool module_scatter_once<SCTOPT>::interact_crit_count(scatter_inner_base &si) const
{
#ifdef SCATTERING_COUNT_INDI
    return !(si.pi.scatterCount >= 1 || si.pj.scatterCount >= 1);
#else
    return true;
#endif // SCATTERING_COUNT_INDI
}

//////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////
// id cut module
/**
 * @brief Method to select interacting particles based on their ID.
 *
 * Only particle pairs where exactly one particle has an ID smaller All.MSIDM_first_target_id
 * can interact.
 *
 */
template <typename SCTOPT> bool module_id_cut<SCTOPT>::interact_crit_id(scatter_inner_base &si) const
{
#ifdef ANGULAR_DEFLECTION_TEST_ON_FIXED_TARGET
    MyIDType ftid = All.MSIDM_first_target_id;
    return (si.pi.ID < ftid || si.pj.ID < ftid) && !(si.pi.ID < ftid && si.pj.ID < ftid);
#else // ANGULAR_DEFLECTION_TEST_ON_FIXED_TARGET
    return true;
#endif // ANGULAR_DEFLECTION_TEST_ON_FIXED_TARGET
}

//////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////
// frequent interaction module
/**
 * @brief Constructor for frequent (small-angle) scattering class.
 */
template <typename SCTOPT>
module_freq<SCTOPT>::module_freq()
    : scatter_outer_base<SCTOPT>(), freq_tfact(0.04), CPU_FREQSCATTER_DF(""), CPU_FREQSCATTER_RC(""),
      CPU_FREQSCATTER("")
{
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Method to initialize class.
 */
template <typename SCTOPT> void module_freq<SCTOPT>::init(const nlohmann::json &param)
{
    if constexpr (SCTOPT::do_timestep)
    {
        freq_tfact = param["freq_tfact"].get<double>();
        Expects(freq_tfact > 0.0, "freq_tfact is non-positive");
    }

    CPU_FREQSCATTER_DF = "f" + scatter_outer_base<SCTOPT>::name + "_scatter_df";
    CPU_FREQSCATTER_RC = "f" + scatter_outer_base<SCTOPT>::name + "_scatter_rc";
    CPU_FREQSCATTER = "f" + scatter_outer_base<SCTOPT>::name + "_scatter";
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Main method to compute frequent interactions.
 */
template <typename SCTOPT>
template <typename SIGMADM, typename RDISS, typename MASSFACT>
void module_freq<SCTOPT>::compute_freq_scat(scatter_inner_base &si, SIGMADM sigmaDm, RDISS r_diss, MASSFACT massfact)
{
#ifdef SCATTERING_CRITICALTIMERS
#pragma omp master
    {
        TIMER_PUSH(1, CPU_FREQSCATTER);
        TIMER_PUSH(1, CPU_FREQSCATTER_DF);
    }
#endif // SCATTERING_CRITICALTIMERS

    MyAtLeastDouble dv_drag_abs = compute_drag_force(si, sigmaDm, massfact);
    MyAtLeastDouble my_r_diss = r_diss(si.v_rel2);
    check_drag_force(si, dv_drag_abs * my_r_diss);

    apply_drag_force(si, dv_drag_abs * my_r_diss);

#ifdef SCATTERING_CRITICALTIMERS
#pragma omp master
    TIMER_POP(1, CPU_FREQSCATTER_DF);
#endif // SCATTERING_CRITICALTIMERS

    if constexpr (!SCTOPT::do_norandcomp)
    {
#ifdef SCATTERING_CRITICALTIMERS
#pragma omp master
        TIMER_PUSH(1, CPU_FREQSCATTER_RC);
#endif // SCATTERING_CRITICALTIMERS

        compute_random_comp(si, dv_drag_abs, my_r_diss);

#ifdef SCATTERING_CRITICALTIMERS
#pragma omp master
        TIMER_POP(1, CPU_FREQSCATTER_RC);
#endif // SCATTERING_CRITICALTIMERS
    }

#ifdef SCATTERING_CRITICALTIMERS
#pragma omp master
    TIMER_POP(1, CPU_FREQSCATTER);
#endif // SCATTERING_CRITICALTIMERS
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Method to compute factor for time step.
 */
template <typename SCTOPT>
template <typename SIGMADMVEL>
double module_freq<SCTOPT>::freq_time_fact(SIGMADMVEL sigmaDm_vel) const
{
    return freq_tfact / sigmaDm_vel();
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Method to compute drag force.
 */
template <typename SCTOPT>
template <typename SIGMADM, typename MASSFACT>
MyAtLeastDouble module_freq<SCTOPT>::compute_drag_force(const scatter_inner_base &si, SIGMADM sigmaDm,
                                                        MASSFACT massfact) const
{
#ifdef ANGULAR_DEFLECTION_TEST_ON_FIXED_TARGET
    //=============== angular deflection test   ================
    // we set the dv_drag_abs to 2x the normal dv_drag_abs because
    // we are dealing with massive particles in their lab frame without recoil
    //
    return 2.0 * nuDm(si, sigmaDm) * massfact(si) * si.Lambda_dt;
    //=============== angular deflection test ends ================
#else // ANGULAR_DEFLECTION_TEST_ON_FIXED_TARGET
    return nuDm(si, sigmaDm) * massfact(si) * si.Lambda_dt;
#endif // ANGULAR_DEFLECTION_TEST_ON_FIXED_TARGET
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Method to check that drag force has a reasonable strength.
 */
template <typename SCTOPT>
void module_freq<SCTOPT>::check_drag_force(const scatter_inner_base &si, const MyAtLeastDouble &dv_drag_abs) const
{
    if constexpr (!SCTOPT::do_fastmode)
    {
        MyAtLeastDouble vi_cms;
#ifdef SCATTERING_UNEQUAL_MASS
        vi_cms = si.v_rel_abs *
                 si.specific_reduced_mass; // si.specific_reduced_mass = si.pj.Mass / (si.pi.Mass + si.pj.Mass);
#else // SCATTERING_UNEQUAL_MASS
        vi_cms = si.v_rel_abs * 0.5;
#endif // SCATTERING_UNEQUAL_MASS
        if (vi_cms * 0.093 < dv_drag_abs)
        {
            if (vi_cms <= dv_drag_abs)
            {
                cout(warning) << "SCATTER warning: time step too large for frequent scattering" << endl;
            }
            else
            {
                cout(warning) << "SCATTER warning: relative velocity change due to frequent scattering\
                        implies an angular error larger than 3%."
                              << endl;
            }
            if constexpr (!SCTOPT::do_ignorewarning)
            {
                if constexpr (!SCTOPT::do_vmax_param)
                {
                    ENDRUN("This can be resolved by decreasing the scattering related timestep.\
                     A smaller value for FSIDM_tfact or FIDM_tfact might help.");
                }
                else
                {
                    ENDRUN("This can be resolved by decreasing the scattering related timestep.\
                     A smaller value for FSIDM_tfact or FIDM_tfact might help. If the problem occurs at\
                     the begining of the simulation consider increasing M" +
                           scatter_outer_base<SCTOPT>::name + "_VmaxInit.");
                }
            }
        }
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Method to apply drag force on particle pair.
 */
template <typename SCTOPT>
void module_freq<SCTOPT>::apply_drag_force(scatter_inner_base &si, const MyAtLeastDouble &dv_drag_abs)
{
    // add v_drag - here pi and pj become the intermediate values (only velocity)
    {
        const MyAtLeastDouble dv_drag_absDv_rel_abs = dv_drag_abs / si.v_rel_abs;
        MyAtLeastDouble dv_drag_c;
#ifndef ANGULAR_DEFLECTION_TEST_ON_FIXED_TARGET
        for (int loopc = 0; loopc < 3; ++loopc)
        {
            dv_drag_c = dv_drag_absDv_rel_abs * si.v_rel[loopc];
            si.pi.Vel[loopc] -= dv_drag_c;
#ifdef SCATTERING_UNEQUAL_MASS
            si.pj.Vel[loopc] += dv_drag_c * si.mass_ratio;
#else // SCATTERING_UNEQUAL_MASS
            si.pj.Vel[loopc] += dv_drag_c;
#endif // SCATTERING_UNEQUAL_MASS
        }
#else // ANGULAR_DEFLECTION_TEST_ON_FIXED_TARGET
        if (si.pi.ID < All.MSIDM_first_target_id)
        {
            for (int loopc = 0; loopc < 3; ++loopc)
            {
                dv_drag_c = dv_drag_absDv_rel_abs * si.v_rel[loopc];
                si.pi.Vel[loopc] -= dv_drag_c;
            }
        }
        if (si.pj.ID < All.MSIDM_first_target_id)
        {
            for (int loopc = 0; loopc < 3; ++loopc)
            {
                dv_drag_c = dv_drag_absDv_rel_abs * si.v_rel[loopc];
                si.pj.Vel[loopc] += dv_drag_c;
            }
        }
#endif // ANGULAR_DEFLECTION_TEST_ON_FIXED_TARGET
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Method to compute random component for frequent interactions.
 */
template <typename SCTOPT>
void module_freq<SCTOPT>::compute_random_comp(scatter_inner_base &si, const MyAtLeastDouble &dv_drag_abs,
                                              const MyAtLeastDouble &r_diss)
{
    //=============== compute random component ===============
    const MyAtLeastDouble phi = random_numbers<SCTOPT>::get_random_phi();
    const MyAtLeastDouble sin_phi = sin(phi);
    const MyAtLeastDouble cos_phi = cos(phi);

#ifdef SCATTERING_UNEQUAL_MASS
    const MyAtLeastDouble tvi_cms = 2.0 * si.specific_reduced_mass * si.v_rel_abs;
#else // SCATTERING_UNEQUAL_MASS
    const MyAtLeastDouble &tvi_cms = si.v_rel_abs;
#endif // SCATTERING_UNEQUAL_MASS

#ifdef ANGULAR_DEFLECTION_TEST_ON_FIXED_TARGET
    // we set the v_rel_abs to 2x the normal v_rel_abs because
    // we are dealing with massive particles in their lab frame without recoil
    const MyAtLeastDouble dv_rest2 = dv_drag_abs * (2.0 * tvi_cms - dv_drag_abs);
#else // ANGULAR_DEFLECTION_TEST_ON_FIXED_TARGET
    const MyAtLeastDouble dv_rest2 = dv_drag_abs * (tvi_cms - dv_drag_abs * r_diss * r_diss);
#endif // ANGULAR_DEFLECTION_TEST_ON_FIXED_TARGET

    MyAtLeastDouble dv_rand_abs = sqrt(dv_rest2);

    const vec3d<MyAtLeastDouble> n = si.v_rel / si.v_rel_abs; // normalised vector

    MyAtLeastDouble zeta = n[1] * n[1] + n[2] * n[2];
    MyAtLeastDouble dv_rand[3];

    if (abs(zeta) > 0.02)
    {
        dv_rand_abs /=
            sqrt(zeta); // with this it is no longer dv_rand_abs, but the division is needed for the next step
        dv_rand[0] = dv_rand_abs * -zeta * cos_phi;
        dv_rand[1] = dv_rand_abs * (-n[2] * sin_phi + n[0] * n[1] * cos_phi);
        dv_rand[2] = dv_rand_abs * (n[1] * sin_phi + n[0] * n[2] * cos_phi);
    }
    else // in case n points into the x-direction
    {
        zeta = n[0] * n[0] + n[2] * n[2];
        if (abs(zeta) > 0.02)
        {
            dv_rand_abs /= sqrt(zeta);
            dv_rand[0] = dv_rand_abs * (n[2] * sin_phi + n[0] * n[1] * cos_phi);
            dv_rand[1] = dv_rand_abs * -zeta * cos_phi;
            dv_rand[2] = dv_rand_abs * (-n[0] * sin_phi + n[1] * n[2] * cos_phi);
        }
        else
        {
            ENDRUN("Scattering direction is problematic.");
            dv_rand[0] = 0;
            dv_rand[1] = 0;
            dv_rand[2] = 0;
        }
    }

    // add v_rand - here pi and pj become the final (post-scatter) values (only velocity)
#ifndef ANGULAR_DEFLECTION_TEST_ON_FIXED_TARGET
    for (int loopc = 0; loopc < 3; ++loopc)
    {
        si.pi.Vel[loopc] += dv_rand[loopc];
#ifdef SCATTERING_UNEQUAL_MASS
        si.pj.Vel[loopc] -= dv_rand[loopc] * si.mass_ratio;
#else // SCATTERING_UNEQUAL_MASS
        si.pj.Vel[loopc] -= dv_rand[loopc];
#endif // SCATTERING_UNEQUAL_MASS
    }
#else // ANGULAR_DEFLECTION_TEST_ON_FIXED_TARGET
    if (si.pi.ID < All.MSIDM_first_target_id)
    {
        for (int loopc = 0; loopc < 3; ++loopc)
        {
            si.pi.Vel[loopc] += dv_rand[loopc];
        }
    }
    if (si.pj.ID < All.MSIDM_first_target_id)
    {
        for (int loopc = 0; loopc < 3; ++loopc)
        {
            si.pj.Vel[loopc] -= dv_rand[loopc];
        }
    }
#endif // ANGULAR_DEFLECTION_TEST_ON_FIXED_TARGET
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Method to compute factor used for drag force.
 */
template <typename SCTOPT>
template <typename SIGMADM>
MyAtLeastDouble module_freq<SCTOPT>::nuDm(const scatter_inner_base &si, SIGMADM sigmaDm) const
{
    return sigmaDm(si.v_rel_abs) * si.pj.Mass * si.v_rel2;
}

//////////////////////////////////////////////////////////////////////////////////////////////////
// rare interaction module
/**
 * @brief Constructor for rare (large-angle) scattering class.
 */
template <typename SCTOPT>
module_rare<SCTOPT>::module_rare() : scatter_outer_base<SCTOPT>(), rare_tfact(0.04), CPU_RARESCATTER("")
{
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Method to initialize class.
 */
template <typename SCTOPT> void module_rare<SCTOPT>::init(const nlohmann::json &param)
{
    if constexpr (SCTOPT::do_timestep)
    {
        rare_tfact = param["rare_tfact"].get<double>();
        Expects(rare_tfact > 0.0, "rare_tfact is non-positive");
    }

    CPU_RARESCATTER = "r" + scatter_outer_base<SCTOPT>::name + "_scatter";
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Main method to compute rare interactions.
 */
template <typename SCTOPT>
template <typename SIGMADM, typename ANGLE, typename MASSFACT, typename... FUNCS>
void module_rare<SCTOPT>::compute_rare_scat(scatter_inner_base &si, SIGMADM sigmaDm, ANGLE angle, MASSFACT massfact,
                                            FUNCS... funcs)
{
#ifdef SCATTERING_CRITICALTIMERS
#pragma omp master
    TIMER_PUSH(1, CPU_RARESCATTER);
#endif // SCATTERING_CRITICALTIMERS

    MyAtLeastDouble P_ij = compute_interact_prob(si, sigmaDm, massfact);

    check_interact_prob(P_ij);

    if (decide_if_interact(P_ij))
    {
        if constexpr (SCTOPT::do_isotropic)
        {
            const vec3d<MyAtLeastDouble> rnd_dir = compute_rnddir_isotropic(si);

            auxiliary_methods(funcs...);

            apply_scattering(si, rnd_dir);
        }
        else
        {
            const MyAtLeastDouble theta = compute_scatter_angle(angle, si.v_rel_abs);

            const vec3d<MyAtLeastDouble> rnd_dir = compute_rnddir(si, theta);

            auxiliary_methods(funcs...);

            apply_scattering(si, rnd_dir);
        }
    }

#ifdef SCATTERING_CRITICALTIMERS
#pragma omp master
    TIMER_POP(1, CPU_RARESCATTER);
#endif // SCATTERING_CRITICALTIMERS
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Method to compute factor for time step.
 */
template <typename SCTOPT>
template <typename SIGMADMVEL>
double module_rare<SCTOPT>::rare_time_fact(SIGMADMVEL sigmaDm_vel) const
{
    return rare_tfact / sigmaDm_vel();
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Method to compute interaction probability.
 */
template <typename SCTOPT>
template <typename SIGMADM, typename MASSFACT>
MyAtLeastDouble module_rare<SCTOPT>::compute_interact_prob(const scatter_inner_base &si, SIGMADM sigmaDm,
                                                           MASSFACT massfact) const
{
    return sigmaDm(si.v_rel_abs) * massfact(si) * si.v_rel_abs * si.Lambda_dt;
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Method to check that interaction probability has a reasonable strength.
 */
template <typename SCTOPT> void module_rare<SCTOPT>::check_interact_prob(const MyAtLeastDouble &P_ij) const
{
    if constexpr (!SCTOPT::do_fastmode)
    {
        // check if probability for scatter is too large
        if (P_ij > 0.268)
        {
            if (P_ij > 1.0)
            {
                cout(warning) << "SCATTER warning: time step too large for rare scattering (P>1)" << endl;
            }
            else
            {
                cout(warning) << "SCATTER warning: time step is large for rare scattering,\
                        probability for multiple scatterings >3%"
                              << endl;
            }
            if constexpr (!SCTOPT::do_ignorewarning)
            {
                if constexpr (!SCTOPT::do_vmax_param)
                {
                    ENDRUN("This can be resolved by decreasing the scattering related timestep.\
                     A smaller value for RSIDM_tfact or RIDM_tfact might help.");
                }
                else
                {
                    ENDRUN("This can be resolved by decreasing the scattering related timestep.\
                     A smaller value for RSIDM_tfact or RIDM_tfact might help. If the problem occurs at\
                     the begining of the simulation consider increasing M" +
                           scatter_outer_base<SCTOPT>::name + "_VmaxInit.");
                }
            }
        }
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Methode to decide if particles interact.
 */
template <typename SCTOPT> bool module_rare<SCTOPT>::decide_if_interact(const MyAtLeastDouble &P_ij)
{
    return (random_numbers<SCTOPT>::get_rnd_scatter_num() <= P_ij); // if true particles interact
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Method to compute scattering angle.
 */
template <typename SCTOPT>
template <typename ANGLE>
MyAtLeastDouble module_rare<SCTOPT>::compute_scatter_angle(ANGLE angle, const MyAtLeastDouble &v_rel)
{
    if constexpr (SCTOPT::do_sangle90)
    {
        // random number in [0,1]
        const MyAtLeastDouble theta = angle(v_rel, random_numbers<SCTOPT>::get_rnd_scatter_num());
        return std::min(theta, M_PI - theta);
    }
    else
    {
        // random number in [0,1]
        return angle(v_rel, random_numbers<SCTOPT>::get_rnd_scatter_num());
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Method to execute additional functions within the rare interaction computations.
 *
 * These functions are executed for pairs of particles that do interact before their
 * velocities are updated.
 *
 */
template <typename SCTOPT>
template <typename... FUNCS>
void module_rare<SCTOPT>::auxiliary_methods(FUNCS... funcs) const
{
    execute(funcs...);
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Method to compute scattering direction for rare scattering.
 */
template <typename SCTOPT>
vec3d<MyAtLeastDouble> module_rare<SCTOPT>::compute_rnddir(scatter_inner_base &si, const MyAtLeastDouble &theta)
{
    // phi symmetry -> random phi in [0,2pi]
    const MyAtLeastDouble phi = random_numbers<SCTOPT>::get_random_phi();

    // scatter direction in cms frame if particle came from e_z direction
    const MyAtLeastDouble sin_theta = sin(theta);
    const vec3d<MyAtLeastDouble> e = {sin_theta * cos(phi), sin_theta * sin(phi), cos(theta)};

    // unit vector in cms frame
    const vec3d<MyAtLeastDouble> e_cms = si.v_rel / si.v_rel_abs;

    // if particle only has z component => Rotation matrix reduces to unity matrix
    const bool bxy = (e_cms.x == 0 && e_cms.y == 0);
    const MyAtLeastDouble c = !bxy * (1 - e_cms.z) / (e_cms.x * e_cms.x + e_cms.y * e_cms.y + bxy);

    // rotated scatter direction
    const vec3d<MyAtLeastDouble> rnd_dir = {
        // x
        (e_cms.z + c * e_cms.y * e_cms.y) * e.x - e_cms.x * e_cms.y * c * e.y + e_cms.x * e.z,
        // y
        -e_cms.x * e_cms.y * c * e.x + (e_cms.z + c * e_cms.x * e_cms.x) * e.y + e_cms.y * e.z,
        // z
        -e_cms.x * e.x - e_cms.y * e.y + e_cms.z * e.z};

    return rnd_dir;
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Method to compute scattering direction for isotropic scattering.
 */
template <typename SCTOPT> vec3d<MyAtLeastDouble> module_rare<SCTOPT>::compute_rnddir_isotropic(scatter_inner_base &si)
{
    // compute random direction -- isotropic scattering
    vec3d<MyAtLeastDouble> rnd_dir;
    {
        MyAtLeastDouble &cos_theta = rnd_dir[2];
        cos_theta = random_numbers<SCTOPT>::get_rnd_cos_theta();
        const MyAtLeastDouble sin_theta = sqrt(1.0 - cos_theta * cos_theta);
        const MyAtLeastDouble phi = random_numbers<SCTOPT>::get_random_phi();
        rnd_dir[0] = sin_theta * cos(phi);
        rnd_dir[1] = sin_theta * sin(phi);
    }

    if constexpr (SCTOPT::do_sangle90)
    {
        MyAtLeastDouble same_direction = 0.0;
        for (int loopc = 0; loopc < 3; ++loopc)
        {
            MyAtLeastDouble v_cms_c = 0.5 * (si.pi.Vel[loopc] + si.pj.Vel[loopc]);
            same_direction += (si.pi.Vel[loopc] - v_cms_c) * rnd_dir[loopc];
        }
        if (same_direction < 0.0)
        {
            rnd_dir *= -1;
        }
    }

    return rnd_dir;
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Method to apply scattering to particles, i.e. update their velocities.
 */
template <typename SCTOPT>
void module_rare<SCTOPT>::apply_scattering(scatter_inner_base &si, const vec3d<MyAtLeastDouble> &rnd_dir) const
{
#ifdef SCATTERING_UNEQUAL_MASS
    const MyAtLeastDouble vi_cms = si.v_rel_abs * si.specific_reduced_mass;
    const MyAtLeastDouble D2m = 1.0 / (si.pi.Mass + si.pj.Mass);
#endif // SCATTERING_UNEQUAL_MASS

#ifndef ANGULAR_DEFLECTION_TEST_ON_FIXED_TARGET
    for (int loopc = 0; loopc < 3; ++loopc)
    {
#ifdef SCATTERING_UNEQUAL_MASS
        const MyAtLeastDouble v_cms_c = (si.pi.Mass * si.pi.Vel[loopc] + si.pj.Mass * si.pj.Vel[loopc]) * D2m;
        si.pi.Vel[loopc] = v_cms_c + vi_cms * rnd_dir[loopc];
        si.pj.Vel[loopc] = v_cms_c - vi_cms * si.mass_ratio * rnd_dir[loopc];
#else // SCATTERING_UNEQUAL_MASS
        const MyAtLeastDouble v_cms_c = 0.5 * (si.pi.Vel[loopc] + si.pj.Vel[loopc]);
        si.pi.Vel[loopc] = v_cms_c + 0.5 * si.v_rel_abs * rnd_dir[loopc];
        si.pj.Vel[loopc] = v_cms_c - 0.5 * si.v_rel_abs * rnd_dir[loopc];
#endif // SCATTERING_UNEQUAL_MASS
    }
#else // ANGULAR_DEFLECTION_TEST_ON_FIXED_TARGET
    if (si.pi.ID < All.MSIDM_first_target_id)
    {
        for (int loopc = 0; loopc < 3; ++loopc)
        {
            const MyAtLeastDouble v_cms_c = si.pj.Vel[loopc];
            si.pi.Vel[loopc] = v_cms_c + si.v_rel_abs * rnd_dir[loopc];
        }
    }
    if (si.pj.ID < All.MSIDM_first_target_id)
    {
        for (int loopc = 0; loopc < 3; ++loopc)
        {
            const MyAtLeastDouble v_cms_c = si.pi.Vel[loopc];
            si.pj.Vel[loopc] = v_cms_c - si.v_rel_abs * rnd_dir[loopc];
        }
    }
#endif // ANGULAR_DEFLECTION_TEST_ON_FIXED_TARGET
}

//////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////
// module frequent simple
/**
 * @brief Constructor for simple far-forward dominated scattering class.
 */
template <typename SCTOPT> module_freq_simple<SCTOPT>::module_freq_simple() : sigmaDm_f(1), sigmaDm_vel_f_ext(0)
{
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Method to initialize class
 */
template <typename SCTOPT> void module_freq_simple<SCTOPT>::init(const nlohmann::json &param)
{
    const MyAtLeastDouble unitSigmaDm = All.UnitLength_in_cm * All.UnitLength_in_cm / All.UnitMass_in_g;
    sigmaDm_f = param["sigmaDm_f"].get<double>() / unitSigmaDm;
    std::string crosssection = param["crosssection_f"].get<std::string>();

    // sigmaDm_f has to be in terms of the transfer cross-section
    if (crosssection == "transfer")
    {
        sigmaDm_f *= 1.0;
    }
    else if (crosssection == "transfer_norm")
    {
        sigmaDm_f *= 0.5;
    }
    else if (crosssection == "viscosity")
    {
        sigmaDm_f *= 0.5;
    }
    else if (crosssection == "viscosity_norm")
    {
        sigmaDm_f /= 3.0;
    }
    else if (crosssection == "total")
    {
        Expects(false, "the total cross-section cannot be used for frequent interactions");
    }
    else
    {
        Expects(false, "cross-section definition for frequent scattering unkown");
    }

    coutr(important) << scatter_outer_base<SCTOPT>::name << endl;
    coutr(important) << " frequent sigma_X/m in cm^2/g and code units" << endl;
    coutr(important) << "  transfer:        " << sigmaDm_f * unitSigmaDm << ", " << sigmaDm_f << endl;
    coutr(important) << "  transfer_norm:   " << sigmaDm_f * unitSigmaDm * 2.0 << ", " << sigmaDm_f * 2.0 << endl;
    coutr(important) << "  viscosity:       " << sigmaDm_f * unitSigmaDm * 2.0 << ", " << sigmaDm_f * 2.0 << endl;
    coutr(important) << "  viscosity_norm:  " << sigmaDm_f * unitSigmaDm * 3.0 << ", " << sigmaDm_f * 3.0 << endl;
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Method to initialize variables for the time step criterion.
 */
template <typename SCTOPT>
template <typename EXTFACT>
void module_freq_simple<SCTOPT>::init_timestep_crit_freq(EXTFACT ext_fact)
{
    vel_f_ext = ext_fact();
    sigmaDm_vel_f_ext = sigmaDm_f * vel_f_ext;
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Method that returns transfer cross-section per mass.
 */
template <typename SCTOPT>
MyAtLeastDouble module_freq_simple<SCTOPT>::calc_sigmaDm_f(const MyAtLeastDouble &v_rel) const
{
    return sigmaDm_f;
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Method that returns transfer cross-section per mass.
 */
template <typename SCTOPT>
template <typename VDEP>
MyAtLeastDouble module_freq_simple<SCTOPT>::calc_sigmaDm_f(VDEP vdep, const MyAtLeastDouble &v_rel) const
{
    if constexpr (SCTOPT::do_vdep)
    {
        return sigmaDm_f * vdep(v_rel / All.cf_atime);
    }
    else
    {
        return sigmaDm_f; // this is a constant cross-section
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Method that returns the maximum of cross-section times velocitiy.
 */
template <typename SCTOPT> MyFloat module_freq_simple<SCTOPT>::freq_sigmaDm_vel() const
{
    return sigmaDm_vel_f_ext * All.cf_atime;
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Method that returns the maximum of cross-section times velocitiy.
 */
template <typename SCTOPT>
template <typename VDEP>
MyFloat module_freq_simple<SCTOPT>::freq_sigmaDm_vel(VDEP vdep, const MyAtLeastDouble &v_max) const
{
    const MyAtLeastDouble vel = v_max / All.cf_atime;
    const bool v_below_ext = (vel < vel_f_ext);
    return (sigmaDm_vel_f_ext * All.cf_atime) * !v_below_ext + sigmaDm_f * vdep(vel) * v_max * v_below_ext;
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Method that returns the maximum of cross-section times velocitiy.
 */
template <typename SCTOPT> MyFloat module_freq_simple<SCTOPT>::freq_sigmaDm_vel(const MyAtLeastDouble &v_max) const
{
    if constexpr (SCTOPT::do_vdep) // this is a simplefied version and usually not used
    {
        const bool v_below_ext = (v_max < vel_f_ext * All.cf_atime);
        return (sigmaDm_vel_f_ext * All.cf_atime) * !v_below_ext + sigmaDm_f * v_max * v_below_ext;
    }
    else
    {
        return sigmaDm_f * v_max;
    }
    return 0.0; // to suppress compiler warning
}

//////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////
// module rare isotropic
/**
 * @brief Constructor for isotropic scattering class.
 */
template <typename SCTOPT> module_rare_isotropic<SCTOPT>::module_rare_isotropic() : sigmaDm_r(1), sigmaDm_vel_r_ext(0)
{
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Method to initialize class.
 */
template <typename SCTOPT> void module_rare_isotropic<SCTOPT>::init(const nlohmann::json &param)
{
    const MyAtLeastDouble unitSigmaDm = All.UnitLength_in_cm * All.UnitLength_in_cm / All.UnitMass_in_g;
    sigmaDm_r = param["sigmaDm_r"].get<double>() / unitSigmaDm;
    std::string crosssection = param["crosssection_r"].get<std::string>();

    // sigmaDm_r has to be in terms of the total cross-section
    if (crosssection == "total")
    {
        sigmaDm_r *= 1.0;
    }
    else if (crosssection == "transfer")
    {
        sigmaDm_r *= 2.0;
    }
    else if (crosssection == "transfer_norm")
    {
        sigmaDm_r *= 1.0;
    }
    else if (crosssection == "viscosity")
    {
        sigmaDm_r *= 3.0 / 2.0;
    }
    else if (crosssection == "viscosity_norm")
    {
        sigmaDm_r *= 1.0;
    }
    else
    {
        Expects(false, "cross-section definition for rare isotropic scattering unkown");
    }

    coutr(important) << scatter_outer_base<SCTOPT>::name << endl;
    coutr(important) << " isotropic sigma_X/m in cm^2/g and code units" << endl;
    coutr(important) << "  total:           " << sigmaDm_r * unitSigmaDm << ", " << sigmaDm_r << endl;
    coutr(important) << "  transfer:        " << sigmaDm_r * unitSigmaDm * 0.5 << ", " << sigmaDm_r * 0.5 << endl;
    coutr(important) << "  transfer_norm:   " << sigmaDm_r * unitSigmaDm << ", " << sigmaDm_r << endl;
    coutr(important) << "  viscosity:       " << sigmaDm_r * unitSigmaDm * 2.0 / 3.0 << ", " << sigmaDm_r * 2.0 / 3.0
                     << endl;
    coutr(important) << "  viscosity_norm:  " << sigmaDm_r * unitSigmaDm << ", " << sigmaDm_r << endl;
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Method to initialize variables for the time step criterion.
 */
template <typename SCTOPT>
template <typename EXTFACT>
void module_rare_isotropic<SCTOPT>::init_timestep_crit_rare(EXTFACT ext_fact)
{
    vel_r_ext = ext_fact();
    sigmaDm_vel_r_ext = sigmaDm_r * vel_r_ext;
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Method that returns total cross-section per mass.
 */
template <typename SCTOPT>
MyAtLeastDouble module_rare_isotropic<SCTOPT>::calc_sigmaDm_r(const MyAtLeastDouble &v_rel) const
{
    return sigmaDm_r;
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Method that returns total cross-section per mass.
 */
template <typename SCTOPT>
template <typename VDEP>
MyAtLeastDouble module_rare_isotropic<SCTOPT>::calc_sigmaDm_r(VDEP vdep, const MyAtLeastDouble &v_rel) const
{
    if constexpr (SCTOPT::do_vdep)
    {
        return sigmaDm_r * vdep(v_rel / All.cf_atime);
    }
    else
    {
        return sigmaDm_r; // this is a constant cross-section
    }
    return 0.0; // to suppress compiler warning
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Method that returns the maximum of cross-section times velocitiy.
 */
template <typename SCTOPT> MyFloat module_rare_isotropic<SCTOPT>::rare_sigmaDm_vel() const
{
    return sigmaDm_vel_r_ext * All.cf_atime;
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Method that returns the maximum of cross-section times velocitiy.
 */
template <typename SCTOPT>
template <typename VDEP>
MyFloat module_rare_isotropic<SCTOPT>::rare_sigmaDm_vel(VDEP vdep, const MyAtLeastDouble &v_max) const
{
    const MyAtLeastDouble vel = v_max / All.cf_atime;
    const bool v_below_ext = (vel < vel_r_ext);
    return (sigmaDm_vel_r_ext * All.cf_atime) * !v_below_ext + sigmaDm_r * vdep(vel) * v_max * v_below_ext;
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Method that returns the maximum of cross-section times velocitiy.
 */
template <typename SCTOPT> MyFloat module_rare_isotropic<SCTOPT>::rare_sigmaDm_vel(const MyAtLeastDouble &v_max) const
{
    if constexpr (SCTOPT::do_vdep)
    {
        const bool v_below_ext = (v_max < vel_r_ext * All.cf_atime);
        return (sigmaDm_vel_r_ext * All.cf_atime) * !v_below_ext + sigmaDm_r * v_max * v_below_ext;
    }
    else
    {
        return sigmaDm_r * v_max;
    }
    return 0.0; // to suppress compiler warning
}

//////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////
// module for equal masses
/**
 * @brief Constructor for equal-mass scattering class.
 */
template <typename SCTOPT>
module_equalmass<SCTOPT>::module_equalmass() : mass_ratio(1.0), specific_reduced_mass(0.5), mass_factor_freq(0.5)
{
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Method to initialize class.
 */
template <typename SCTOPT> void module_equalmass<SCTOPT>::init(const nlohmann::json &param)
{
    if (All.MassTable[1] == 0.0)
    {
        coutr(warning) << "Warning: MassTable[1] is zero. Do you want equal-mass scattering?" << endl;
        coutr(warning) << "If all DM particles have the same mass it should be safe to continue. (File: " << __FILE__
                       << ", Line: " << __LINE__ << ")" << endl;
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Method to return the frequent mass factor for equal-mass scattering.
 */
template <typename SCTOPT>
MyAtLeastDouble module_equalmass<SCTOPT>::get_mass_factor_freq(const scatter_inner_base &si) const
{
    return mass_factor_freq;
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Method to return the rare mass factor for equal-mass scattering.
 */
template <typename SCTOPT>
MyAtLeastDouble module_equalmass<SCTOPT>::get_mass_factor_rare(const scatter_inner_base &si) const
{
    return si.pi.Mass;
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Method to return the time step mass factor for equal-mass scattering.
 */
template <typename SCTOPT>
MyAtLeastDouble module_equalmass<SCTOPT>::get_mass_factor_timestep(const scattering_particle_ref &p) const
{
    return p.Mass;
}

//////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////
// unequal mass module
/**
 * @brief Constructor for unequal-mass scattering class.
 */
template <typename SCTOPT> module_unequalmass<SCTOPT>::module_unequalmass() : min_mass(0.0)
{
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Method to initialize class.
 */
template <typename SCTOPT> void module_unequalmass<SCTOPT>::init(const nlohmann::json &param)
{
    Expects(All.MassTable[1] == 0.0, "MassTable[1] is not zero");

    unsigned long int j = 0;
    while ((P[j].Type != type_dm) && (j < (unsigned long int)(NumPart - 1)))
    {
        ++j;
    }

    Expects(P[j].Type == type_dm, "no DM particle found");
    min_mass = P[j].Mass;

    for (unsigned long int i = j + 1; i < (unsigned long int)(NumPart); ++i)
    {
        if (P[i].Type == type_dm)
        {
            min_mass = std::min(P[i].Mass, min_mass);
        }
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Method to return frequent mass factor for unequal-mass scattering.
 */
template <typename SCTOPT>
MyAtLeastDouble module_unequalmass<SCTOPT>::get_mass_factor_freq(const scatter_inner_base &si) const
{
    MyAtLeastDouble mass_ratio = (si.pi.Mass > si.pj.Mass)
                                     ? (si.pi.Mass / si.pj.Mass)
                                     : (si.pj.Mass / si.pi.Mass); // heavier_particle / lighter_particle
    MyAtLeastDouble mass_factor_freq =
        (1.0 / (1.0 + mass_ratio)); // as in IDM: mu=1, f_bary=1, m_virt=heavier_particle, m_dm=lighter_particle
    return mass_factor_freq;
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Method to return rare mass factor for unequal-mass scattering.
 */
template <typename SCTOPT>
MyAtLeastDouble module_unequalmass<SCTOPT>::get_mass_factor_rare(const scatter_inner_base &si) const
{
    MyAtLeastDouble mass_factor_rare =
        (si.pi.Mass > si.pj.Mass) ? si.pj.Mass : si.pi.Mass; // select the lighter particle mass
    return mass_factor_rare;
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Method to return time step mass factor for unequal-mass scattering.
 */
template <typename SCTOPT>
MyAtLeastDouble module_unequalmass<SCTOPT>::get_mass_factor_timestep(const scattering_particle_ref &p) const
{
    return min_mass;
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Method for interaction criterion of unequal-mass particles.
 */
template <typename SCTOPT> bool module_unequalmass<SCTOPT>::interact_crit_mass(scatter_inner_base &si) const
{
    return (si.pi.Mass != si.pj.Mass);
}

//////////////////////////////////////////////////////////////////////////////////////////////////
// computing vmax
/**
 * @brief Method to initialize class.
 */
template <typename SCTOPT> void module_vmax<SCTOPT>::init(const nlohmann::json &param)
{
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Method to update maximum velocity a particle has seen.
 */
template <typename SCTOPT> void module_vmax<SCTOPT>::auxiliary_method1(scatter_inner_base &si)
{
#ifdef SCATTERING_VMAX
    si.pi.v_max = std::max(si.pi.v_max, si.v_rel_abs);
    si.pj.v_max = std::max(si.pj.v_max, si.v_rel_abs);
#endif // SCATTERING_VMAX
}

//////////////////////////////////////////////////////////////////////////////////////////////////
// computing hmin
/**
 * @brief Method to initialize class.
 */
template <typename SCTOPT> void module_hmin<SCTOPT>::init(const nlohmann::json &param)
{
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Method to update maximum velocity a particle has seen.
 */
template <typename SCTOPT> void module_hmin<SCTOPT>::auxiliary_method1(scatter_inner_base &si)
{
#ifdef SCATTERING_HMIN
    if constexpr (SCTOPT::do_hmin)
    {
        si.pi.h_min = std::min(si.pi.h_min, si.pj.Hsml);
        si.pj.h_min = std::min(si.pj.h_min, si.pi.Hsml);
    }
#endif // SCATTERING_HMIN
}

//////////////////////////////////////////////////////////////////////////////////////////////////
// velocity dependent scattering - model 0
/**
 * @brief Method to initialize class.
 */
template <typename SCTOPT> void module_vdep0<SCTOPT>::init(const nlohmann::json &param)
{
    Expects(SCTOPT::do_vdep, "do_vdep is not activated");
    vdep_w = param["vdep_w"].get<double>();
    vdep_alpha = param["vdep_alpha"].get<double>();
    vdep_beta = param["vdep_beta"].get<double>();

    vdep_Dw = 1.0 / vdep_w;
    vdep_Dbeta = 1.0 / vdep_beta;
    vdep_alphaDbeta = vdep_alpha * vdep_Dbeta;
    vdep_D1alpha = 1.0 / (1.0 + vdep_alpha);
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Method that gives the velocity dependence of the cross-section for model 0.
 */
template <typename SCTOPT> MyAtLeastDouble module_vdep0<SCTOPT>::sigmaDm_vdep_fact(const MyAtLeastDouble &v_rel) const
{
    return pow(1.0 + pow(v_rel * vdep_Dw, vdep_beta), vdep_alphaDbeta);
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Method that returns velocity at which sigma/m v becomes maximal.
 */
template <typename SCTOPT> MyAtLeastDouble module_vdep0<SCTOPT>::sigmaDm_vel_ext_fact() const
{
    return pow(vdep_alpha * vdep_D1alpha, vdep_alphaDbeta) * vdep_w * pow(-vdep_D1alpha, vdep_Dbeta);
}

//////////////////////////////////////////////////////////////////////////////////////////////////
// velocity dependent scattering - model 1
/**
 * @brief Method to initialize class.
 */
template <typename SCTOPT> void module_vdep1<SCTOPT>::init(const nlohmann::json &param)
{
    Expects(SCTOPT::do_vdep, "do_vdep is not activated");
    vdep_w = param["vdep_w"].get<double>();
    vdep_u = param["vdep_u"].get<double>();

    vdep_Du2 = 1.0 / (vdep_u * vdep_u);
    vdep_numer = 1.0 + vdep_w * vdep_w * vdep_Du2;
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Method that gives the velocity dependence of the cross-section for model 1.
 */
template <typename SCTOPT> MyAtLeastDouble module_vdep1<SCTOPT>::sigmaDm_vdep_fact(const MyAtLeastDouble &v_rel) const
{
    const MyAtLeastDouble vmw = v_rel - vdep_w;
    return vdep_numer / (1.0 + vmw * vmw * vdep_Du2);
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Method that returns velocity at which sigma/m v becomes maximal.
 */
template <typename SCTOPT> MyAtLeastDouble module_vdep1<SCTOPT>::sigmaDm_vel_ext_fact() const
{
    return sqrt(vdep_w * vdep_w + vdep_u * vdep_u);
}

//////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////
// velocity dependent scattering - model mv
/**
 * @brief Method to initialize class.
 */
template <typename SCTOPT> void module_vdep_mv<SCTOPT>::init(const nlohmann::json &param)
{
    Expects(SCTOPT::do_vdep, "do_vdep is not activated");
    vdep_w = param["vdep_w"].get<double>();
    vdep_Dw = 1.0 / vdep_w;
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Method that gives the velocity dependence of the viscosity cross-section for Moeller scattering.
 */
template <typename SCTOPT> MyAtLeastDouble module_vdep_mv<SCTOPT>::sigmaDm_vdep_fact(const MyAtLeastDouble &v_rel) const
{
    const MyAtLeastDouble x = v_rel * vdep_Dw;
    const MyAtLeastDouble x2 = x * x;
    const MyAtLeastDouble x4 = x2 * x2;
    const MyAtLeastDouble x6 = x4 * x2;
    const MyAtLeastDouble x8 = x4 * x4;
    const MyAtLeastDouble fact1 = 6.0 / (x8 + 2.0 * x6);
    const MyAtLeastDouble term1 = 2.0 * (5.0 + 5.0 * x2 + x4) * log(1.0 + x2);
    const MyAtLeastDouble term2 = 5.0 * (2.0 * x2 + x4);
    return std::min(fact1 * (term1 - term2),
                    MyAtLeastDouble(1.0)); // we take the minium to deal with potential round-off errors
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Method that returns velocity at which sigma/m v becomes maximal.
 */
template <typename SCTOPT> MyAtLeastDouble module_vdep_mv<SCTOPT>::sigmaDm_vel_ext_fact() const
{
    return 0.98447266 * vdep_w;
}

//////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////
// velocity dependent scattering - model rv
/**
 * @brief Method to initialize class.
 */
template <typename SCTOPT> void module_vdep_rv<SCTOPT>::init(const nlohmann::json &param)
{
    Expects(SCTOPT::do_vdep, "do_vdep is not activated");
    vdep_w = param["vdep_w"].get<double>();
    vdep_Dw = 1.0 / vdep_w;
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Method that gives the velocity dependence of the viscosity cross-section for Rutherford scattering.
 */
template <typename SCTOPT> MyAtLeastDouble module_vdep_rv<SCTOPT>::sigmaDm_vdep_fact(const MyAtLeastDouble &v_rel) const
{
    const MyAtLeastDouble x = v_rel * vdep_Dw;
    const MyAtLeastDouble x2 = x * x;
    const MyAtLeastDouble x4 = x2 * x2;
    const MyAtLeastDouble x6 = x4 * x2;
    const MyAtLeastDouble fact1 = 6.0 / x6;
    const MyAtLeastDouble term1 = (2.0 + x2) * log(1.0 + x2);
    const MyAtLeastDouble term2 = 2.0 * x2;
    return std::min(fact1 * (term1 - term2),
                    MyAtLeastDouble(1.0)); // we take the minium to deal with potential round-off errors
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Method that returns velocity at which sigma/m v becomes maximal.
 */
template <typename SCTOPT> MyAtLeastDouble module_vdep_rv<SCTOPT>::sigmaDm_vel_ext_fact() const
{
    return 0.89960937 * vdep_w;
}

#ifdef SCATTERING_RARE_DISSIPATION
//////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////
// module for rare dissipative scattering
/**
 * @brief Constructor for rare dissipation class.
 */
template <typename SCTOPT> module_rare_dissipation<SCTOPT>::module_rare_dissipation() : fact_dissipation(1.0)
{
}

/**
 * @brief Method to initialize class.
 */
template <typename SCTOPT> void module_rare_dissipation<SCTOPT>::init(const nlohmann::json &param)
{
    fact_dissipation = sqrt(1.0 - param["f_diss"].get<double>());
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Method that reduces relative velocity to model dissipation.
 */
template <typename SCTOPT>
void module_rare_dissipation<SCTOPT>::apply_dissipation_to_v_rel_abs(MyAtLeastDouble &v_rel_abs) const
{
    v_rel_abs *= fact_dissipation;
}
#endif // SCATTERING_RARE_DISSIPATION

//////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////
// module for frequent dissipative scattering
#ifdef SCATTERING_FREQ_DISSIPATION
/**
 * @brief Constructor for frequent dissipation class.
 */
template <typename SCTOPT>
module_freq_dissipation<SCTOPT>::module_freq_dissipation() : const_r_diss(1.0), freq_tfact(0.04)
{
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Method to initialize class.
 */
template <typename SCTOPT> void module_freq_dissipation<SCTOPT>::init(const nlohmann::json &param)
{
    const_r_diss = param["const_r_diss"].get<double>();
    freq_tfact = param["freq_tfact"].get<double>();
    Expects(freq_tfact > 0.0, "freq_tfact is non-positive");
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Method to define a velocity-dependent dissipation parameter (not implemented).
 */
template <typename SCTOPT>
MyAtLeastDouble module_freq_dissipation<SCTOPT>::get_r_diss(const MyAtLeastDouble &v_rel2) const
{
    return const_r_diss;
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Method to compute modification of time step due to dissipation (not ready for velocity dependence).
 */
template <typename SCTOPT> MyAtLeastDouble module_freq_dissipation<SCTOPT>::get_dissipation_factor_timestep() const
{
    return std::max(const_r_diss, freq_tfact * const_r_diss * const_r_diss);
}
#endif // SCATTERING_FREQ_DISSIPATION
//////////////////////////////////////////////////////////////////////////////////////////////////
// module for differential cross-section
/**
 * @brief Constructor for differential cross-section class.
 */
template <typename SCTOPT>
module_differential_cross_section<SCTOPT>::module_differential_cross_section() : scatter_outer_base<SCTOPT>()
{
    Expects(!SCTOPT::do_vdep, "do_vdep should not be activated");
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Method to initialize class.
 */
template <typename SCTOPT> void module_differential_cross_section<SCTOPT>::init(const nlohmann::json &param)
{
    dcs.init(scatter_outer_base<SCTOPT>::name, param);
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Method to call extension of the table for differential cross-sections.
 */
template <typename SCTOPT> void module_differential_cross_section<SCTOPT>::auxiliary_method1(scatter_inner_base &si)
{
    if constexpr (SCTOPT::do_extend_dcs_table)
    {
#ifdef SCATTERING_OPENMP
#pragma omp critical
        {
#endif // SCATTERING_OPENMP
            dcs.extend_table_if_needed(si.v_rel_abs / All.cf_atime);
#ifdef SCATTERING_OPENMP
        }
#endif // SCATTERING_OPENMP
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Method to obtain transfer cross-section of differential cross-section for frequent scattering.
 */
template <typename SCTOPT>
MyAtLeastDouble module_differential_cross_section<SCTOPT>::calc_sigmaDm_f(const MyAtLeastDouble &v_rel) const
{
    return dcs.interp_sigmaT<SCTOPT::do_ignorewarning>(v_rel / All.cf_atime); // is already thread-safe
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Method to obtain total cross-section of differential cross-section for rare scattering.
 */
template <typename SCTOPT>
MyAtLeastDouble module_differential_cross_section<SCTOPT>::calc_sigmaDm_r(const MyAtLeastDouble &v_rel) const
{
    return dcs.interp_sigma<SCTOPT::do_ignorewarning>(v_rel / All.cf_atime); // is already thread-safe
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Method to obtain scattering angle for differential cross-section.
 */
template <typename SCTOPT>
MyAtLeastDouble module_differential_cross_section<SCTOPT>::calc_angle_r(const MyAtLeastDouble &v_rel,
                                                                        const MyAtLeastDouble &rnd_num) const
{
    return dcs.interp_theta<SCTOPT::do_ignorewarning>(rnd_num, v_rel / All.cf_atime); // is already thread-safe
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Method for throwing an error when ill-configured.
 */
template <typename SCTOPT> MyFloat module_differential_cross_section<SCTOPT>::freq_sigmaDm_vel() const
{
    ENDRUN_MAIN("SCATTERING_VMAX should be turned on.");
    return 0.0;
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Method that returns the maximum of cross-section times velocitiy for frequent scattering.
 */
template <typename SCTOPT>
MyFloat module_differential_cross_section<SCTOPT>::freq_sigmaDm_vel(const MyAtLeastDouble &v_max) const
{
    return dcs.sigmaDm_vel_f_ext(v_max / All.cf_atime) * All.cf_atime;
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Method for throwing an error when ill-configured.
 */
template <typename SCTOPT> MyFloat module_differential_cross_section<SCTOPT>::rare_sigmaDm_vel() const
{
    ENDRUN_MAIN("SCATTERING_VMAX should be turned on.");
    return 0.0;
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Method that returns the maximum of cross-section times velocitiy for rare scattering.
 */
template <typename SCTOPT>
MyFloat module_differential_cross_section<SCTOPT>::rare_sigmaDm_vel(const MyAtLeastDouble &v_max) const
{
    return dcs.sigmaDm_vel_r_ext(v_max / All.cf_atime) * All.cf_atime;
}


//////////////////////////////////////////////////////////////////////////////////////////////////

#endif // MSIDM_SCATTERING_MODULES_CPP
