/**
 * @file
 *
 * This file is part of the N-body/SPH code OpenGadget3.
 *
 * @brief This header file contains the declarations for the scattering modules.
 *
 * The scattering modules are classes that are inherited by the main scattering class.
 * They specify the detailed behaviour of the scattering and contain auxiliary functionality.
 *
 * @author OpenGadget3-development team, in particular Moritz S. Fischer, Yashraj Patil, Cenanda Arido, and Marc
 * Wiertel.
 *
 * @copyright Copyright (c) 2026, OpenGadget3-development team lead by Klaus Dolag and Stefano Borgani
 * (see change logs). All rights reserved.
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License v3 or later.
 * See <https://www.gnu.org/licenses/>.
 */

#ifndef MSIDM_SCATTERING_MODULE_HPP
#define MSIDM_SCATTERING_MODULE_HPP

#include "scattering_base.hpp"
#include "scattering_data.hpp"
#include "scattering_options.hpp"
#define JSON_ASSERT(x)
#include "dcs_class.hpp"
#include <nlohmann/json.hpp>

//////////////////////////////////////////////////////////////////////////////////////////////////

namespace scatter::pxp_scattering
{
/**
 * @brief Base class for scattering modules.
 */
template <typename SCTOPT> class module_base : virtual public scatter_outer_base<SCTOPT>
{
  public:
    void init(const nlohmann::json &param);

    void auxiliary_method1(scatter_inner_base &si); // they are also executed in init_vmax
    void auxiliary_method2(scatter_inner_base &si);
    void auxiliary_method3(scatter_inner_base &si);
};
} // namespace scatter::pxp_scattering

//////////////////////////////////////////////////////////////////////////////////////////////////

namespace scatter::pxp_scattering
{
/**
 * Scattering module class for debug capabilities.
 */
template <typename SCTOPT> class module_debug : public module_base<SCTOPT>
{
  public:
    void init(const nlohmann::json &param);

    void debug_pos_and_vel(const scatter_inner_base &si) const;
};
} // namespace scatter::pxp_scattering

//////////////////////////////////////////////////////////////////////////////////////////////////

namespace scatter::pxp_scattering
{
/**
 * @brief Scattering module class for global counting of scattering events.
 */
template <typename SCTOPT> class module_scattercount : public module_base<SCTOPT>
{
  public:
    module_scattercount();

    void init(const nlohmann::json &param);

    void exchange_scattercount() const;

    void increase_global_scattercount(scatter_inner_base &si);

    // for test problem and debugging
    long long unsigned int scatter_count;
};
} // namespace scatter::pxp_scattering

//////////////////////////////////////////////////////////////////////////////////////////////////

namespace scatter::pxp_scattering
{
/**
 * @brief Scattering module class for counting scattering events per particle.
 */
template <typename SCTOPT> class module_scattercount_indi : public module_base<SCTOPT>
{
  public:
    void increase_individual_scattercount(scatter_inner_base &si) const;
};
} // namespace scatter::pxp_scattering

//////////////////////////////////////////////////////////////////////////////////////////////////

namespace scatter::pxp_scattering
{
/**
 * @brief Scattering module for test problems where particles are allowed to scatter only once.
 */
template <typename SCTOPT> class module_scatter_once : public module_base<SCTOPT>
{
  public:
    bool interact_crit_count(scatter_inner_base &si) const;
};
} // namespace scatter::pxp_scattering

//////////////////////////////////////////////////////////////////////////////////////////////////

namespace scatter::pxp_scattering
{
/**
 * @brief Scattering module to select particles based on their ID.
 */
template <typename SCTOPT> class module_id_cut : public module_base<SCTOPT>
{
  public:
    bool interact_crit_id(scatter_inner_base &si) const;
};
} // namespace scatter::pxp_scattering

//////////////////////////////////////////////////////////////////////////////////////////////////

namespace scatter::pxp_scattering
{
/**
 * @brief Scattering module for frequent small-angle scatterings.
 */
template <typename SCTOPT> class module_freq : public module_base<SCTOPT>
{
  public:
    module_freq();

    void init(const nlohmann::json &param);

    MyAtLeastDouble freq_tfact;

    template <typename SIGMADM, typename RDISS, typename MASSFACT>
    void compute_freq_scat(scatter_inner_base &si, SIGMADM sigmaDm, RDISS r_diss, MASSFACT massfact);

    template <typename SIGMADMVEL> double freq_time_fact(SIGMADMVEL sigmaDm_vel) const;

  private: // methods to be used within module_freq only
    template <typename SIGMADM, typename MASSFACT>
    MyAtLeastDouble compute_drag_force(const scatter_inner_base &si, SIGMADM sigmaDm, MASSFACT massfact) const;

    void check_drag_force(const scatter_inner_base &si, const MyAtLeastDouble &dv_drag_abs) const;

    void apply_drag_force(scatter_inner_base &si, const MyAtLeastDouble &dv_drag_abs);

    void compute_random_comp(scatter_inner_base &si, const MyAtLeastDouble &dv_drag_abs, const MyAtLeastDouble &r_diss);

    template <typename SIGMADM> MyAtLeastDouble nuDm(const scatter_inner_base &si, SIGMADM sigmaDm) const;

    //======= timer =======
  private:
    std::string CPU_FREQSCATTER_DF;
    std::string CPU_FREQSCATTER_RC;
    std::string CPU_FREQSCATTER;
};
} // namespace scatter::pxp_scattering

//////////////////////////////////////////////////////////////////////////////////////////////////

namespace scatter::pxp_scattering
{
/**
 * @brief Scattering module for rare large-angle scattering.
 */
template <typename SCTOPT> class module_rare : public module_base<SCTOPT>
{
  public:
    module_rare();

    void init(const nlohmann::json &param);

    MyAtLeastDouble rare_tfact;

    template <typename SIGMADM, typename ANGLE, typename MASSFACT, typename... FUNCS>
    void compute_rare_scat(scatter_inner_base &si, SIGMADM sigmaDm, ANGLE angle, MASSFACT massfact, FUNCS... funcs);

    template <typename SIGMADMVEL> double rare_time_fact(SIGMADMVEL sigmaDm_vel) const;

  private: // methods to be used within module_rare only
    template <typename SIGMADM, typename MASSFACT>
    MyAtLeastDouble compute_interact_prob(const scatter_inner_base &si, SIGMADM sigmaDM, MASSFACT massfact) const;

    void check_interact_prob(const MyAtLeastDouble &P_ij) const;

    bool decide_if_interact(const MyAtLeastDouble &P_ij);

    template <typename ANGLE>
    MyAtLeastDouble compute_scatter_angle(ANGLE angle, const MyAtLeastDouble &v_rel); // can use random_number generator

    template <typename... FUNCS>
    void auxiliary_methods(
        FUNCS... funcs) const; // can be used for additional physics such as dissipation or computing the scatter count

    vec3d<MyAtLeastDouble> compute_rnddir(scatter_inner_base &si, const MyAtLeastDouble &theta);

    vec3d<MyAtLeastDouble> compute_rnddir_isotropic(scatter_inner_base &si);

    void apply_scattering(scatter_inner_base &si, const vec3d<MyAtLeastDouble> &rnd_dir) const;

    //======= timer =======
  private:
    std::string CPU_RARESCATTER;
};
} // namespace scatter::pxp_scattering

//////////////////////////////////////////////////////////////////////////////////////////////////

namespace scatter::pxp_scattering
{
/**
 * @brief Scattering module for a simple far-forward dominated cross-section.
 */
template <typename SCTOPT> class module_freq_simple : public module_base<SCTOPT>
{
  public:
    module_freq_simple();

    void init(const nlohmann::json &param);

    MyAtLeastDouble sigmaDm_f; // this is the transfer cross-section
    MyAtLeastDouble vel_f_ext; // vel must be in terms of the proper motion
    MyAtLeastDouble sigmaDm_vel_f_ext; // vel must be in terms of the proper motion

    template <typename EXTFACT> void init_timestep_crit_freq(EXTFACT ext_fact);

    MyAtLeastDouble calc_sigmaDm_f(const MyAtLeastDouble &v_rel) const;

    template <typename VDEP> MyAtLeastDouble calc_sigmaDm_f(VDEP vdep, const MyAtLeastDouble &v_rel) const;

    // for time step
    MyFloat freq_sigmaDm_vel() const;

    template <typename VDEP> MyFloat freq_sigmaDm_vel(VDEP vdep, const MyAtLeastDouble &v_max) const;

    MyFloat freq_sigmaDm_vel(const MyAtLeastDouble &v_max) const;
};
} // namespace scatter::pxp_scattering

//////////////////////////////////////////////////////////////////////////////////////////////////

namespace scatter::pxp_scattering
{
/**
 * @brief Scattering module for an isotropic cross-section.
 */
template <typename SCTOPT> class module_rare_isotropic : public module_base<SCTOPT>
{
  public:
    module_rare_isotropic();

    void init(const nlohmann::json &param);

    MyAtLeastDouble sigmaDm_r; // this is the total cross-section
    MyAtLeastDouble vel_r_ext; // vel must be in terms of the proper motion
    MyAtLeastDouble sigmaDm_vel_r_ext; // vel must be in terms of the proper motion

    template <typename EXTFACT> void init_timestep_crit_rare(EXTFACT ext_fact);

    MyAtLeastDouble calc_sigmaDm_r(const MyAtLeastDouble &v_rel) const;

    template <typename VDEP> MyAtLeastDouble calc_sigmaDm_r(VDEP vdep, const MyAtLeastDouble &v_rel) const;

    // for time step
    MyFloat rare_sigmaDm_vel() const;

    template <typename VDEP> MyFloat rare_sigmaDm_vel(VDEP vdep, const MyAtLeastDouble &v_max) const;

    MyFloat rare_sigmaDm_vel(const MyAtLeastDouble &v_max) const;
};
} // namespace scatter::pxp_scattering

//////////////////////////////////////////////////////////////////////////////////////////////////

namespace scatter::pxp_scattering
{
/**
 * @brief Scattering module for equal-mass scattering.
 */
template <typename SCTOPT> class module_equalmass : public module_base<SCTOPT>
{
  public:
    module_equalmass();

    void init(const nlohmann::json &param);

    MyAtLeastDouble mass_ratio, specific_reduced_mass;
    MyAtLeastDouble mass_factor_freq;

    MyAtLeastDouble get_mass_factor_freq(const scatter_inner_base &si) const;

    MyAtLeastDouble get_mass_factor_rare(const scatter_inner_base &si) const;

    MyAtLeastDouble get_mass_factor_timestep(const scattering_particle_ref &p) const;
};
} // namespace scatter::pxp_scattering

//////////////////////////////////////////////////////////////////////////////////////////////////

namespace scatter::pxp_scattering
{
/**
 * @brief Scattering module for unequal-mass scattering.
 */
template <typename SCTOPT> class module_unequalmass : public module_base<SCTOPT>
{
  public:
    module_unequalmass();

    void init(const nlohmann::json &param);

    MyFloat min_mass;

    MyAtLeastDouble get_mass_factor_freq(const scatter_inner_base &si) const;

    MyAtLeastDouble get_mass_factor_rare(const scatter_inner_base &si) const;

    MyAtLeastDouble get_mass_factor_timestep(const scattering_particle_ref &p) const;

    bool interact_crit_mass(scatter_inner_base &si) const;
};
} // namespace scatter::pxp_scattering

//////////////////////////////////////////////////////////////////////////////////////////////////

namespace scatter::pxp_scattering
{
/**
 * @brief Scattering module to determine the maximum velocity that a particle sees.
 */
template <typename SCTOPT> class module_vmax : public module_base<SCTOPT>
{
  public:
    void init(const nlohmann::json &param);

    void auxiliary_method1(scatter_inner_base &si); // updates vmax
};
} // namespace scatter::pxp_scattering

//////////////////////////////////////////////////////////////////////////////////////////////////

namespace scatter::pxp_scattering
{
/**
 * @brief Scattering module to determine the minimum kernel size that a particle sees.
 */
template <typename SCTOPT> class module_hmin : public module_base<SCTOPT>
{
  public:
    void init(const nlohmann::json &param);

    void auxiliary_method1(scatter_inner_base &si); // updates hmin
};
} // namespace scatter::pxp_scattering

//////////////////////////////////////////////////////////////////////////////////////////////////

namespace scatter::pxp_scattering
{
/**
 * @brief Scattering module for the velocity-dependence given by Eq.2 in Fischer et al. 2024.
 */
template <typename SCTOPT> class module_vdep0 : public module_base<SCTOPT>
{
  public:
    void init(const nlohmann::json &param);

    MyAtLeastDouble vdep_w, vdep_Dw;
    MyAtLeastDouble vdep_alpha, vdep_beta;
    MyAtLeastDouble vdep_Dbeta, vdep_alphaDbeta;
    MyAtLeastDouble vdep_D1alpha;

    MyAtLeastDouble sigmaDm_vdep_fact(const MyAtLeastDouble &v_rel) const;

    // for initialization of time step criterion

    MyAtLeastDouble sigmaDm_vel_ext_fact() const;
};
} // namespace scatter::pxp_scattering

//////////////////////////////////////////////////////////////////////////////////////////////////

namespace scatter::pxp_scattering
{
/**
 * @brief Scattering module for a velocity-dependence with a simple resonance.
 */
template <typename SCTOPT> class module_vdep1 : public module_base<SCTOPT>
{
  public:
    void init(const nlohmann::json &param);

    MyAtLeastDouble vdep_w, vdep_numer;
    MyAtLeastDouble vdep_u, vdep_Du2;

    MyAtLeastDouble sigmaDm_vdep_fact(const MyAtLeastDouble &v_rel) const;

    // for initialization of time step criterion

    MyAtLeastDouble sigmaDm_vel_ext_fact() const;
};
} // namespace scatter::pxp_scattering

//////////////////////////////////////////////////////////////////////////////////////////////////

namespace scatter::pxp_scattering
{
/**
 * @brief Scattering module for the velocity-dependence of the viscosity cross-section for Moeller scattering.
 */
template <typename SCTOPT> class module_vdep_mv : public module_base<SCTOPT>
{
  public:
    void init(const nlohmann::json &param);

    MyAtLeastDouble vdep_w, vdep_Dw;

    MyAtLeastDouble sigmaDm_vdep_fact(const MyAtLeastDouble &v_rel) const;

    // for initialization of time step criterion

    MyAtLeastDouble sigmaDm_vel_ext_fact() const;
};
} // namespace scatter::pxp_scattering

//////////////////////////////////////////////////////////////////////////////////////////////////

namespace scatter::pxp_scattering
{
/**
 * @brief Scattering module for the velocity-dependence of the viscosity cross-section for Rutherford scattering.
 */
template <typename SCTOPT> class module_vdep_rv : public module_base<SCTOPT>
{
  public:
    void init(const nlohmann::json &param);

    MyAtLeastDouble vdep_w, vdep_Dw;

    MyAtLeastDouble sigmaDm_vdep_fact(const MyAtLeastDouble &v_rel) const;

    // for initialization of time step criterion

    MyAtLeastDouble sigmaDm_vel_ext_fact() const;
};
} // namespace scatter::pxp_scattering

#ifdef SCATTERING_RARE_DISSIPATION
//////////////////////////////////////////////////////////////////////////////////////////////////

namespace scatter::pxp_scattering
{
/**
 * @brief Scattering module for a simple model with dissipation.
 */
template <typename SCTOPT> class module_rare_dissipation : public module_base<SCTOPT>
{
  public:
    module_rare_dissipation();
    void init(const nlohmann::json &param);

    MyAtLeastDouble fact_dissipation;

    void apply_dissipation_to_v_rel_abs(MyAtLeastDouble &v_rel_abs) const;
};
} // namespace scatter::pxp_scattering
#endif // SCATTERING_RARE_DISSIPATION

//////////////////////////////////////////////////////////////////////////////////////////////////
#ifdef SCATTERING_FREQ_DISSIPATION
namespace scatter::pxp_scattering
{
template <typename SCTOPT> class module_freq_dissipation : public module_base<SCTOPT>
{
  public:
    module_freq_dissipation();
    void init(const nlohmann::json &param);

    MyAtLeastDouble const_r_diss;
    MyAtLeastDouble freq_tfact;

    MyAtLeastDouble get_r_diss(const MyAtLeastDouble &v_rel2) const;
    MyAtLeastDouble get_dissipation_factor_timestep() const;
};
} // namespace scatter::pxp_scattering
#endif // SCATTERING_FREQ_DISSIPATION
//////////////////////////////////////////////////////////////////////////////////////////////////

namespace scatter::pxp_scattering
{
/**
 * @brief Scattering module for a full differential cross-section.
 */
template <typename SCTOPT> class module_differential_cross_section : public module_base<SCTOPT>
{
  public:
    module_differential_cross_section();

    void init(const nlohmann::json &param);

    diff_cross_section_class dcs;

    void auxiliary_method1(scatter_inner_base &si); // extends table

    MyAtLeastDouble calc_sigmaDm_f(const MyAtLeastDouble &v_rel) const;

    MyAtLeastDouble calc_sigmaDm_r(const MyAtLeastDouble &v_rel) const;

    MyAtLeastDouble calc_angle_r(const MyAtLeastDouble &v_rel, const MyAtLeastDouble &rnd_num) const;

    MyFloat freq_sigmaDm_vel() const;

    MyFloat freq_sigmaDm_vel(const MyAtLeastDouble &v_max) const;

    MyFloat rare_sigmaDm_vel() const;

    MyFloat rare_sigmaDm_vel(const MyAtLeastDouble &v_max) const;
};
} // namespace scatter::pxp_scattering


//////////////////////////////////////////////////////////////////////////////////////////////////

#endif /* MSIDM_SCATTERING_MODULE_HPP */
