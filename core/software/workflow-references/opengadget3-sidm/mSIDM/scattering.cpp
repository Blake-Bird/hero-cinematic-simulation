/**
 * @file
 *
 * This file is part of the N-body/SPH code OpenGadget3.
 *
 * @brief Source file for the main scattering class.
 *
 * @author OpenGadget3-development team, in particular Moritz S. Fischer.
 *
 * @copyright Copyright (c) 2026, Klaus Dolag, and all contributing authors
 * (see change logs). All rights reserved.
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License v3 or later.
 * See <https://www.gnu.org/licenses/>.
 */

#include "scattering.hpp"
#include "scattering_types.hpp"
#include "tools.hpp"
#include "../CodeBase/allvars.hpp"
#include "../CodeBase/proto.hpp"
#include "scattering_base.cpp"
#include "scattering_modules.cpp"

// #define SCATTERING_CRITICALTIMERS

using namespace scatter::pxp_scattering;

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Constructor of the main scattering class.
 */
template <typename SCTOPT, typename LNUM, template <typename SCTOPTM> typename... MODULE>
Scattering<SCTOPT, LNUM, MODULE...>::Scattering(LNUM &calc_lambda_in)
    : scatter_outer<SCTOPT, MODULE...>(), so(*this), calc_lambda(calc_lambda_in)
{
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Destructor of the main scattering class.
 */
template <typename SCTOPT, typename LNUM, template <typename SCTOPTM> typename... MODULE>
Scattering<SCTOPT, LNUM, MODULE...>::~Scattering()
{
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Method for initialization of the scattering class.
 */
template <typename SCTOPT, typename LNUM, template <typename SCTOPTM> typename... MODULE>
void Scattering<SCTOPT, LNUM, MODULE...>::sec_const(const std::string &name_in, const nlohmann::json &param)
{
    scatter_outer_base<SCTOPT>::name = name_in;
    init_independent(param); // initializes all modules
    init_timestep_crit();
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Method for initialization of scattering modules.
 */
template <typename SCTOPT, typename LNUM, template <typename SCTOPTM> typename... MODULE>
void Scattering<SCTOPT, LNUM, MODULE...>::init_independent(const nlohmann::json &param)
{
    do_init([this, &param]() { MODULE<SCTOPT>::init(param); }...);
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Method for end of recursive initialization.
 */
template <typename SCTOPT, typename LNUM, template <typename SCTOPTM> typename... MODULE>
void Scattering<SCTOPT, LNUM, MODULE...>::do_init()
{
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Method for recursive initialization.
 */
template <typename SCTOPT, typename LNUM, template <typename SCTOPTM> typename... MODULE>
template <typename FUNC, typename... FUNCS>
void Scattering<SCTOPT, LNUM, MODULE...>::do_init(FUNC func, FUNCS... funcs)
{
    func();
    do_init(funcs...);
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Method for initialization of the time step criteria.
 */
template <typename SCTOPT, typename LNUM, template <typename SCTOPTM> typename... MODULE>
void Scattering<SCTOPT, LNUM, MODULE...>::init_timestep_crit()
{
    auto ext_fact = [this]() -> MyAtLeastDouble { return scatter_outer<SCTOPT, MODULE...>::sigmaDm_vel_ext_fact(); };

    so.init_timestep_crit_freq(ext_fact);
    so.init_timestep_crit_rare(ext_fact);
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief This is the main method to compute the scattering of two particles.
 */
template <typename SCTOPT, typename LNUM, template <typename SCTOPTM> typename... MODULE>
void Scattering<SCTOPT, LNUM, MODULE...>::operator()(scattering_particle_ref &pi_in, scattering_particle_ref &pj_in)
{
#ifdef SCATTERING_CRITICALTIMERS
#pragma omp master
    TIMER_PUSH(1, "m" + scatter_outer_base<SCTOPT>::name + "_scatter_prep");
#endif

    scatter_inner sct_pair(so, pi_in, pj_in);

    if constexpr (SCTOPT::do_interact_crit)
    {
        if (!sct_pair.interact_crit_pre())
        {
#ifdef SCATTERING_CRITICALTIMERS
#pragma omp master
            TIMER_POP(1, "m" + scatter_outer_base<SCTOPT>::name + "_scatter_prep");
#endif

            return;
        }
    }

    sct_pair.debug_pos_and_vel();

    sct_pair.compute_distance();

    if constexpr (SCTOPT::do_interact_crit)
    {
        if (!sct_pair.interact_crit_dist())
        {
#ifdef SCATTERING_CRITICALTIMERS
#pragma omp master
            TIMER_POP(1, "m" + scatter_outer_base<SCTOPT>::name + "_scatter_prep");
#endif

            return;
        }
    }

    sct_pair.compute_velocity();

    if constexpr (SCTOPT::do_interact_crit)
    {
        if (!sct_pair.interact_crit_vel())
        {
#ifdef SCATTERING_CRITICALTIMERS
#pragma omp master
            TIMER_POP(1, "m" + scatter_outer_base<SCTOPT>::name + "_scatter_prep");
#endif

            return;
        }
    }

    sct_pair.compute_mass_ratio();

    sct_pair.compute_lambda_dt();

#ifdef SCATTERING_CRITICALTIMERS
#pragma omp master
    TIMER_POP(1, "m" + scatter_outer_base<SCTOPT>::name + "_scatter_prep");
#endif

    sct_pair.auxiliary_methods1();

    sct_pair.compute_freq_scat();

    // note v_rel can be outdated, which would be the case for dissipative interactions, if needed it must be recomputed

    sct_pair.auxiliary_methods2();

    sct_pair.compute_rare_scat();

    sct_pair.auxiliary_methods3();
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Method for initialization of maximum velocity that a paritlce has seen.
 *
 * This method is executed at the beginning of a simulation.
 */
template <typename SCTOPT, typename LNUM, template <typename SCTOPTM> typename... MODULE>
void Scattering<SCTOPT, LNUM, MODULE...>::init_vmax(scattering_particle_ref &pi_in, scattering_particle_ref &pj_in)
{
    scatter_inner sct_pair(so, pi_in, pj_in);

    sct_pair.compute_velocity();

    sct_pair.auxiliary_methods1();
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/** time step functionality */
/**
 * @brief Method to evaluate time step criterion for scattering.
 */
template <typename SCTOPT, typename LNUM, template <typename SCTOPTM> typename... MODULE>
double Scattering<SCTOPT, LNUM, MODULE...>::calc_dt(const scattering_particle_ref &p) const
{
    const double &hsml = p.Hsml;

    if constexpr (!SCTOPT::do_fastmode)
    {
        Expects(hsml > 0.0, "hsml is non-positive");
    }

    double const_dt_fact;

    if constexpr (SCTOPT::do_hmin)
    {
#ifdef SCATTERING_HMIN
        /** when localhsmlrescaling is on it can happen that a particle has not seen any neighbour
            over the last time step, in this case we use the previous time step as a constraint for
            the scattering */
        if (p.h_min == FLT_MAX)
        {
            return calc_dt_wrapper(p.TimeBin, p.Ti_begstep);
        }

        const_dt_fact = 1.0 / (so.get_mass_factor_timestep(p) * so.get_dissipation_factor_timestep() *
                               calc_lambda(hsml, p.h_min, 0.0));
#else // SCATTERING_HMIN
        Expects(false, "SCATTERING_HMIN must be activated");
#endif // SCATTERING_HMIN
    }
    else
    {
        const_dt_fact = 1.0 / (so.get_mass_factor_timestep(p) * so.get_dissipation_factor_timestep() *
                               calc_lambda(hsml, hsml, 0.0));
    }

    auto freq_sigmaDm_vel = [this, &p]() -> double {
#ifdef SCATTERING_VMAX
        if constexpr (SCTOPT::do_vdep)
        {
            auto vdep = [this](const MyAtLeastDouble &v_rel_in) -> MyAtLeastDouble {
                return so.sigmaDm_vdep_fact(v_rel_in);
            };

            return so.freq_sigmaDm_vel(vdep, p.v_max);
        }
        else
        {
            return so.freq_sigmaDm_vel(p.v_max);
        }
#else // SCATTERING_VMAX
        return so.freq_sigmaDm_vel();
#endif // SCATTERING_VMAX
    };

    auto rare_sigmaDm_vel = [this, &p]() -> double {
#ifdef SCATTERING_VMAX
        if constexpr (SCTOPT::do_vdep)
        {
            auto vdep = [this](const MyAtLeastDouble &v_rel_in) -> MyAtLeastDouble {
                return so.sigmaDm_vdep_fact(v_rel_in);
            };

            return so.rare_sigmaDm_vel(vdep, p.v_max);
        }
        else
        {
            return so.rare_sigmaDm_vel(p.v_max);
        }
#else // SCATTERING_VMAX
        return so.rare_sigmaDm_vel();
#endif // SCATTERING_VMAX
    };

    double dt_scat = const_dt_fact * std::min(so.freq_time_fact(freq_sigmaDm_vel), so.rare_time_fact(rare_sigmaDm_vel));

    if (All.ComovingIntegrationOn)
    {
        const double a2 = All.cf_atime * All.cf_atime;
        dt_scat *= a2 * a2 / All.HubbleParam;
    }

#ifdef SCATTERING_ID0SPECIALTSTEP
    if constexpr (SCTOPT::do_id0specialtstep)
    {
        if (p.ID == 0)
        {
            dt_scat *= 0.02;
        }
    }
#endif // SCATTERING_ID0SPECIALTSTEP

#ifdef SCATTERING_VMAX
    p.v_max = 0.0;
#endif // SCATTERING_VMAX

#ifdef SCATTERING_HMIN
    if constexpr (SCTOPT::do_hmin)
    {
        p.h_min = FLT_MAX;
    }
#endif // SCATTERING_HMIN

#ifdef SCATTERING_DEBUG
    check_nan(dt_scat, "Error: scatter time step is nan.");
#endif // SCATTERING_DEBUG

    return dt_scat;
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Method to call exchange_scattercount.
 */
template <typename SCTOPT, typename LNUM, template <typename SCTOPTM> typename... MODULE>
void Scattering<SCTOPT, LNUM, MODULE...>::exchange_scatter_count()
{
    scatter_outer<SCTOPT, MODULE...>::exchange_scattercount();
}

//////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////
/** inner routine for scattering, acts on paris of particles */
/**
 * @brief Constructor of scatter_inner class.
 */
template <typename SCTOPT, typename LNUM, template <typename SCTOPTM> typename... MODULE>
Scattering<SCTOPT, LNUM, MODULE...>::scatter_inner::scatter_inner(Scattering &so_in, scattering_particle_ref &pi_in,
                                                                  scattering_particle_ref &pj_in)
    : scatter_inner_base(pi_in, pj_in
#if defined(SCATTERING_UNEQUAL_MASS) && !defined(SCATTERING_INDI_MASS_RATIO)
                         ,
                         so_in.mass_ratio, so_in.specific_reduced_mass
#endif // defined(SCATTERING_UNEQUAL_MASS) && !defined(SCATTERING_INDI_MASS_RATIO)
                         ),
      so(so_in)
{
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Method to call the debug functionality for the particle position and velocity.
 */
template <typename SCTOPT, typename LNUM, template <typename SCTOPTM> typename... MODULE>
void Scattering<SCTOPT, LNUM, MODULE...>::scatter_inner::debug_pos_and_vel()
{
    so.debug_pos_and_vel(si);
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Method to compute distance of particle pair.
 */
template <typename SCTOPT, typename LNUM, template <typename SCTOPTM> typename... MODULE>
void Scattering<SCTOPT, LNUM, MODULE...>::scatter_inner::compute_distance()
{
    // compute distance
    MyAtLeastDouble x_rel_c = NEAREST_X(pi.Pos[0] - pj.Pos[0]);
    MyAtLeastDouble x_rel2 = x_rel_c * x_rel_c;
    x_rel_c = NEAREST_Y(pi.Pos[1] - pj.Pos[1]);
    x_rel2 += x_rel_c * x_rel_c;
    x_rel_c = NEAREST_Z(pi.Pos[2] - pj.Pos[2]);
    x_rel2 += x_rel_c * x_rel_c;

    x_rel_abs = sqrt(x_rel2);
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Method to check for several interaction criteria.
 */
template <typename SCTOPT, typename LNUM, template <typename SCTOPTM> typename... MODULE>
bool Scattering<SCTOPT, LNUM, MODULE...>::scatter_inner::interact_crit_pre()
{
    return so.interact_crit_id(si) && so.interact_crit_count(si) && so.interact_crit_mass(si);
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Method to check if particles are close enough to interact.
 */
template <typename SCTOPT, typename LNUM, template <typename SCTOPTM> typename... MODULE>
bool Scattering<SCTOPT, LNUM, MODULE...>::scatter_inner::interact_crit_dist()
{
    // interaction criterion
    return (x_rel_abs < pi.Hsml + pj.Hsml);
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Method to check that relative velocity of particles in non-zero.
 */
template <typename SCTOPT, typename LNUM, template <typename SCTOPTM> typename... MODULE>
bool Scattering<SCTOPT, LNUM, MODULE...>::scatter_inner::interact_crit_vel()
{
    // interaction criterion
    return (v_rel_abs != 0);
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Method to call relative velocity computation.
 */
template <typename SCTOPT, typename LNUM, template <typename SCTOPTM> typename... MODULE>
void Scattering<SCTOPT, LNUM, MODULE...>::scatter_inner::compute_velocity()
{
    // calculate relative velocity
    scatter_inner_base::compute_velocity();
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Method to compute mass ratio in case of unequal-mass scattering.
 */
template <typename SCTOPT, typename LNUM, template <typename SCTOPTM> typename... MODULE>
void Scattering<SCTOPT, LNUM, MODULE...>::scatter_inner::compute_mass_ratio()
{
#if defined(SCATTERING_UNEQUAL_MASS) && defined(SCATTERING_INDI_MASS_RATIO)
    mass_ratio = pi.Mass / pj.Mass;
    specific_reduced_mass = pj.Mass / (pi.Mass + pj.Mass);
#endif // SCATTERING_INDI_MASS_RATIO
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Method to compute kernel overlap intergral times time step.
 */
template <typename SCTOPT, typename LNUM, template <typename SCTOPTM> typename... MODULE>
void Scattering<SCTOPT, LNUM, MODULE...>::scatter_inner::compute_lambda_dt()
{
    dt = get_dt();
    Lambda_dt = so.calc_lambda(pi.Hsml, pj.Hsml, x_rel_abs) * dt;
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Method to call of all scattering modules their auxiliary_method1.
 */
template <typename SCTOPT, typename LNUM, template <typename SCTOPTM> typename... MODULE>
void Scattering<SCTOPT, LNUM, MODULE...>::scatter_inner::auxiliary_methods1()
{
    execute([this]() { so.MODULE<SCTOPT>::auxiliary_method1(si); }...);
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Method to call of all scattering modules their auxiliary_method2.
 */
template <typename SCTOPT, typename LNUM, template <typename SCTOPTM> typename... MODULE>
void Scattering<SCTOPT, LNUM, MODULE...>::scatter_inner::auxiliary_methods2()
{
    execute([this]() { so.MODULE<SCTOPT>::auxiliary_method2(si); }...);
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Method to call of all scattering modules their auxiliary_method3.
 */
template <typename SCTOPT, typename LNUM, template <typename SCTOPTM> typename... MODULE>
void Scattering<SCTOPT, LNUM, MODULE...>::scatter_inner::auxiliary_methods3()
{
    execute([this]() { so.MODULE<SCTOPT>::auxiliary_method3(si); }...);
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Method for frequent scattering.
 */
template <typename SCTOPT, typename LNUM, template <typename SCTOPTM> typename... MODULE>
void Scattering<SCTOPT, LNUM, MODULE...>::scatter_inner::compute_freq_scat()
{
    auto sigmaDm = [this](const MyAtLeastDouble &v_rel_in) -> MyAtLeastDouble {
        if constexpr (SCTOPT::do_vdep) // unnessecary complicated - could pass reference to value instead of lambda
        {
            auto vdep = [this](const MyAtLeastDouble &v_rel_in2) -> MyAtLeastDouble {
                return so.sigmaDm_vdep_fact(v_rel_in2);
            };

            return so.calc_sigmaDm_f(vdep, v_rel_in);
        }
        else
        {
            return so.calc_sigmaDm_f(v_rel_in);
        }
        return 0.0; // to suppress compiler warning
    };

    auto r_diss = [&](const MyAtLeastDouble &v_rel2_in) -> MyAtLeastDouble { return so.get_r_diss(v_rel2_in); };

    auto massfact = [this](const scatter_inner_base &si_in) -> MyAtLeastDouble {
        return so.get_mass_factor_freq(si_in);
    };

    so.compute_freq_scat(si, sigmaDm, r_diss, massfact); // could add further functions as arguments
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Method for rare scattering.
 */
template <typename SCTOPT, typename LNUM, template <typename SCTOPTM> typename... MODULE>
void Scattering<SCTOPT, LNUM, MODULE...>::scatter_inner::compute_rare_scat()
{
    auto sigmaDm = [this](const MyAtLeastDouble &v_rel_in) -> MyAtLeastDouble {
        if constexpr (SCTOPT::do_vdep)
        {
            auto vdep = [this](const MyAtLeastDouble &v_rel_in2) -> MyAtLeastDouble {
                return so.sigmaDm_vdep_fact(v_rel_in2);
            };

            return so.calc_sigmaDm_r(vdep, v_rel_in);
        }
        else
        {
            return so.calc_sigmaDm_r(v_rel_in);
        }
        return 0.0; // to suppress compiler warning
    };

    auto angle = [this](const MyAtLeastDouble &v_rel_in, const MyAtLeastDouble &rnd_num_in) -> MyAtLeastDouble {
        return so.calc_angle_r(v_rel_in, rnd_num_in);
    };

    auto massfact = [this](const scatter_inner_base &si_in) -> MyAtLeastDouble {
        return so.get_mass_factor_rare(si_in);
    };

#ifdef SCATTERING_RARE_DISSIPATION
    auto dissip = [this]() -> void { so.apply_dissipation_to_v_rel_abs(si.v_rel_abs); };
#endif // SCATTERING_RARE_DISSIPATION

    auto glob_sctcnt = [this]() -> void { so.increase_global_scattercount(si); };

    auto indi_sctcnt = [this]() -> void { so.increase_individual_scattercount(si); };

#ifdef SCATTERING_RARE_DISSIPATION
    so.compute_rare_scat(si, sigmaDm, angle, massfact, dissip, glob_sctcnt, indi_sctcnt);
#else // SCATTERING_RARE_DISSIPATION
    so.compute_rare_scat(si, sigmaDm, angle, massfact, glob_sctcnt, indi_sctcnt);
#endif // SCATTERING_RARE_DISSIPATION
}

//////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef mSIDM
namespace scatter::pxp_scattering
{
/**
 * @brief Template specialization for mSIDM.
 *
 * This template specialization should be the same as the type definition
 * given in scattering_type.hpp.
 */
template class Scattering<sct_opt_msidm, LambdaNumerical<kernel_msidm>
#ifdef mSIDM_DEBUG
                          ,
                          module_debug
#endif // mSIDM_DEBUG
#ifdef rSIDM_SCATTERCOUNT
                          ,
                          module_scattercount
#endif // rSIDM_SCATTERCOUNT
#ifdef rSIDM_SCATTERCOUNT_INDI
                          ,
                          module_scattercount_indi
#endif // rSIDM_SCATTERCOUNT_INDI
#ifdef rSIDM_SCATTER_ONCE
                          ,
                          module_scatter_once
#endif // rSIDM_SCATTER_ONCE
#ifdef ANGULAR_DEFLECTION_TEST_ON_FIXED_TARGET
                          ,
                          module_id_cut
#endif // ANGULAR_DEFLECTION_TEST_ON_FIXED_TARGET
#ifdef fSIDM
                          ,
                          module_freq
#endif // fSIDM
#ifdef rSIDM
                          ,
                          module_rare
#endif // rSIDM
#ifdef fSIDM_SIMPLE
                          ,
                          module_freq_simple
#endif // fSIDM_SIMPLE
#ifdef rSIDM_ISOTROPIC
                          ,
                          module_rare_isotropic
#endif // rSIDM_ISOTROPIC
#ifdef mSIDM_EQUALMASS
                          ,
                          module_equalmass
#endif // mSIDM_EQUALMASS
#ifdef mSIDM_UNEQUALMASS
                          ,
                          module_unequalmass
#endif // mSIDM_UNEQUALMASS
#ifdef mSIDM_VMAX
                          ,
                          module_vmax
#endif // mSIDM_VMAX
#ifdef mSIDM_HMIN
                          ,
                          module_hmin
#endif // mSIDM_HMIN
#ifdef mSIDM_VDEP0
                          ,
                          module_vdep0
#endif // mSIDM_VDEP0
#ifdef mSIDM_VDEP1
                          ,
                          module_vdep1
#endif // mSIDM_VDEP1
#ifdef mSIDM_VDEP_MV
                          ,
                          module_vdep_mv
#endif // mSIDM_VDEP_MV
#ifdef mSIDM_VDEP_RV
                          ,
                          module_vdep_rv
#endif // mSIDM_VDEP_RV
#ifdef rSIDM_DISSIPATION
                          ,
                          module_rare_dissipation
#endif // rSIDM_DISSIPATION
#ifdef fSIDM_DISSIPATION
                          ,
                          module_freq_dissipation
#endif // fSIDM_DISSIPATION
#ifdef mSIDM_DIFFERENTIAL_CROSS_SECTION
                          ,
                          module_differential_cross_section
#endif // mSIDM_DIFFERENTIAL_CROSS_SECTION
                          >;
} // namespace scatter::pxp_scattering
#endif // mSIDM

//////////////////////////////////////////////////////////////////////////////////////////////////


//////////////////////////////////////////////////////////////////////////////////////////////////
