/**
 * @file
 *
 * This file is part of the N-body/SPH code OpenGadget3.
 *
 * @brief Header file for the main scattering class.
 *
 * @author OpenGadget3-development team, in particular Moritz S. Fischer.
 *
 * @copyright Copyright (c) 2026, OpenGadget3-development team lead by Klaus Dolag and Stefano Borgani
 * (see change logs). All rights reserved.
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License v3 or later.
 * See <https://www.gnu.org/licenses/>.
 */

#ifndef MSIDM_SCATTERING_HPP
#define MSIDM_SCATTERING_HPP

#include "../CodeBase/precision.hpp"
#include "../CodeBase/switches.hpp"

#include "kernel_integration.hpp"
#include "scattering_base.hpp"
#include "scattering_data.hpp"
#include "scattering_modules.hpp"
#include "scattering_options.hpp"
#define JSON_ASSERT(x)
#include <nlohmann/json.hpp>

//////////////////////////////////////////////////////////////////////////////////////////////////

namespace scatter::pxp_scattering
{
/**
 * @brief Main class for scattering.
 */
template <typename SCTOPT, typename LNUM, template <typename SCTOPTM> typename... MODULE>
class Scattering final : private scatter_outer<SCTOPT, MODULE...>
{
  public:
    Scattering(LNUM &calc_lambda_in);
    ~Scattering();
    void sec_const(const std::string &name_in, const nlohmann::json &param);

    //======= initialisation ========
  private:
    void init_independent(const nlohmann::json &param); // calls init() of all modules
    void init_timestep_crit();

    void do_init();

    template <typename FUNC, typename... FUNCS> void do_init(FUNC func, FUNCS... funcs);

    //======= scattering ========
  public:
    void operator()(scattering_particle_ref &pi_in, scattering_particle_ref &pj_in);

    void init_vmax(scattering_particle_ref &pi_in, scattering_particle_ref &pj_in);

    //======= scattering class ========
  public:
    class scatter_inner : private scatter_inner_base
    {
      public:
        scatter_inner(Scattering &so_in, scattering_particle_ref &pi_in, scattering_particle_ref &pj_in);

      private:
        Scattering &so;

      public:
        void debug_pos_and_vel();
        void compute_distance();
        bool interact_crit_pre();
        bool interact_crit_dist();
        bool interact_crit_vel();
        void compute_velocity();
        void compute_mass_ratio();
        void compute_lambda_dt();
        void auxiliary_methods1();
        void compute_freq_scat();
        void auxiliary_methods2();
        void compute_rare_scat();
        void auxiliary_methods3();
    };

    //======= reference members ========
  private:
    Scattering &so;
    LNUM &calc_lambda;

    //======= time step ========
  public:
    /** calc_dt() is called by get_timestep() (defined in Integrator/timestep.c) */
    double calc_dt(const scattering_particle_ref &p) const;

    //======== scatter count ========
  public:
    /** method for MPI exchange of scatter count  */
    void exchange_scatter_count();
};
} // namespace scatter::pxp_scattering

//////////////////////////////////////////////////////////////////////////////////////////////////

#endif /* MSIDM_SCATTERING_HPP */
