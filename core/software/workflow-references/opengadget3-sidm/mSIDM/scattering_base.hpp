/**
 * @file
 *
 * This file is part of the N-body/SPH code OpenGadget3.
 *
 * @brief Header file for the scattering base class.
 *
 * @author OpenGadget3-development team, in particular Moritz S. Fischer.
 *
 * @copyright Copyright (c) 2026, OpenGadget3-development team lead by Klaus Dolag and Stefano Borgani
 * (see change logs). All rights reserved.
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License v3 or later.
 * See <https://www.gnu.org/licenses/>.
 */

#ifndef MSIDM_SCATTERING_BASE_HPP
#define MSIDM_SCATTERING_BASE_HPP

#include "scattering_data.hpp"
#include <random>

#ifdef SCATTERING_OPENMP
#include <omp.h>
#endif // SCATTERING_OPENMP

//////////////////////////////////////////////////////////////////////////////////////////////////

namespace scatter::pxp_scattering
{
/**
 * @brief Small class for a three-dimensional vector.
 */
template <typename DTYPE> class vec3d
{
  public:
    vec3d();
    vec3d(std::initializer_list<DTYPE> inputs);
    vec3d(const vec3d<DTYPE> &input);

    DTYPE &x, &y, &z;

    DTYPE &operator[](const int &index);
    DTYPE operator[](const int &index) const;

    vec3d<DTYPE> &operator=(const vec3d<DTYPE> &a);
    vec3d<DTYPE> &operator+=(const vec3d<DTYPE> &a);
    vec3d<DTYPE> &operator-=(const vec3d<DTYPE> &a);
    vec3d<DTYPE> &operator*=(DTYPE a);
    vec3d<DTYPE> &operator/=(DTYPE a);
    vec3d<DTYPE> operator-(const vec3d<DTYPE> &a);
    vec3d<DTYPE> operator+(const vec3d<DTYPE> &a);
    vec3d<DTYPE> operator-(DTYPE d) const;
    vec3d<DTYPE> operator+(DTYPE d) const;
    vec3d<DTYPE> operator*(DTYPE d) const;
    vec3d<DTYPE> operator/(DTYPE d) const;

  private:
    DTYPE array[3];
};

/**
 * @brief Define ostream operator to print content of vector.
 */
template <typename DTYPE> std::ostream &operator<<(std::ostream &s, const vec3d<DTYPE> &v)
{
    s << "[" << v[0] << ", " << v[1] << ", " << v[2] << "]";
    return s;
};
} // namespace scatter::pxp_scattering

//////////////////////////////////////////////////////////////////////////////////////////////////

namespace scatter::pxp_scattering
{
/**
 * @brief Function to compute time step for comoving integration based on time step bin.
 */
double calc_dt_comoving(const short int &TimeBin, const integertime &Ti_begstep);

/**
 * @brief Function to compute time step for non-comoving integration based on time step bin.
 */
double calc_dt_lin(const short int &TimeBin, const integertime &Ti_begstep);

/**
 * @brief Wrapper function to compute time step for comoving and non-comoving integration.
 */
double calc_dt_wrapper(const short int &TimeBin, const integertime &Ti_begstep);

/**
 * @brief Base of inner scattering class (nested in Scattering).
 *
 * This class contains the properties of the pair of interacting particles.
 */
class scatter_inner_base
{
  public:
#if defined(SCATTERING_UNEQUAL_MASS) && !defined(SCATTERING_INDI_MASS_RATIO)
    scatter_inner_base(scattering_particle_ref &pi_in, scattering_particle_ref &pj_in, MyAtLeastDouble &mass_ratio_in,
                       MyAtLeastDouble &specific_reduced_mass_in);
#else // SCATTERING_UNEQUAL_MASS && !SCATTERING_INDI_MASS_RATIO
    scatter_inner_base(scattering_particle_ref &pi_in, scattering_particle_ref &pj_in);
#endif // SCATTERING_UNEQUAL_MASS && !SCATTERING_INDI_MASS_RATIO

    scatter_inner_base &si;
    scattering_particle_ref &pi, &pj;
#ifdef SCATTERING_UNEQUAL_MASS
#ifdef SCATTERING_INDI_MASS_RATIO
    MyAtLeastDouble mass_ratio;
    MyAtLeastDouble specific_reduced_mass;
#else // SCATTERING_INDI_MASS_RATIO
    MyAtLeastDouble &mass_ratio;
    MyAtLeastDouble &specific_reduced_mass;
#endif // SCATTERING_INDI_MASS_RATIO
#endif // SCATTERING_UNEQUAL_MASS
    MyAtLeastDouble x_rel_abs;
    MyAtLeastDouble Lambda_dt, dt;
    MyAtLeastDouble v_rel2, v_rel_abs;
    vec3d<MyAtLeastDouble> v_rel;

    void compute_velocity();

  public:
    /** get_dt = get_dt_full * get_dt_fact, get_dt takes into account how often a particle is
        considered per timestep, this depends on whether the particle is active or passive. */
    MyAtLeastDouble get_dt() const;
    MyAtLeastDouble get_dt_full() const;
    MyAtLeastDouble get_dt_fact() const;
};
} // namespace scatter::pxp_scattering

//////////////////////////////////////////////////////////////////////////////////////////////////

namespace scatter::pxp_scattering
{
/**
 * @brief Class for randome numbers; base of scatter_outer_base; accessible to all modules.
 */
template <typename SCTOPT> class random_numbers
{
  public:
    random_numbers();
    ~random_numbers();

    void init();

    unsigned int seed;
    std::mt19937 generator;
#ifdef SCATTERING_OPENMP
    omp_lock_t lock_generator; // OMP lock for the random number generator
#endif

    std::uniform_real_distribution<MyAtLeastDouble> random_phi;
    MyAtLeastDouble get_random_phi();

    std::uniform_real_distribution<MyAtLeastDouble> random_scatter_num;
    MyAtLeastDouble get_rnd_scatter_num();

    std::uniform_real_distribution<MyAtLeastDouble> random_cos_theta;
    MyAtLeastDouble get_rnd_cos_theta();
};
} // namespace scatter::pxp_scattering

//////////////////////////////////////////////////////////////////////////////////////////////////

namespace scatter::pxp_scattering
{
/**
 * @brief Declaration of the base class for scattering modules (virtual inheritance).
 *
 * The class contains methods that are meant to be overwritten by the modules
 * (scattering_modules.hpp) to specify the details of the pair-wise interaction.
 */
template <typename SCTOPT> class scatter_outer_base : public random_numbers<SCTOPT>
{
  public:
    scatter_outer_base();

    // methods for debugging
    void debug_pos_and_vel(const scatter_inner_base &si) const;

    // methods for interaction criterion
    bool interact_crit_count(scatter_inner_base &si) const;
    bool interact_crit_id(scatter_inner_base &si) const;
    bool interact_crit_mass(scatter_inner_base &si) const;

    // methods for scatter count
    void exchange_scattercount() const;

    // methods for frequent interactions
    template <typename SIGMADM, typename RDISS, typename MASSFACT>
    void compute_freq_scat(scatter_inner_base &si, SIGMADM sigmaDm, RDISS r_diss, MASSFACT massfact);

    MyAtLeastDouble calc_sigmaDm_f(const MyAtLeastDouble &v_rel) const;

    template <typename VDEP> MyAtLeastDouble calc_sigmaDm_f(VDEP vdep, const MyAtLeastDouble &v_rel) const;

    template <typename SIGMADMVEL> double freq_time_fact(SIGMADMVEL sigmaDm_vel) const;

    MyFloat freq_sigmaDm_vel() const;

    template <typename VDEP> MyFloat freq_sigmaDm_vel(VDEP vdep, const MyAtLeastDouble &v_max) const;

    MyFloat freq_sigmaDm_vel(const MyAtLeastDouble &v_max) const;

    // methods for rare interactions
    template <typename SIGMADM, typename ANGLE, typename... FUNCS>
    void compute_rare_scat(scatter_inner_base &si, SIGMADM sigmaDm, ANGLE angle, FUNCS... funcs);

    MyAtLeastDouble calc_sigmaDm_r(const MyAtLeastDouble &v_rel) const;

    template <typename VDEP> MyAtLeastDouble calc_sigmaDm_r(VDEP vdep, const MyAtLeastDouble &v_rel) const;

    MyAtLeastDouble calc_angle_r(const MyAtLeastDouble &v_rel, const MyAtLeastDouble &rnd_num) const;

    template <typename SIGMADMVEL> double rare_time_fact(SIGMADMVEL sigmaDm_vel) const;

    MyFloat rare_sigmaDm_vel() const;

    template <typename VDEP> MyFloat rare_sigmaDm_vel(VDEP vdep, const MyAtLeastDouble &v_max) const;

    MyFloat rare_sigmaDm_vel(const MyAtLeastDouble &v_max) const;

    // methods for velocity dependence
    MyAtLeastDouble sigmaDm_vdep_fact(const MyAtLeastDouble &v_rel) const;

    // methods for initialization of time step criterion
    MyAtLeastDouble sigmaDm_vel_ext_fact() const;

    template <typename EXTFACT> void init_timestep_crit_freq(EXTFACT ext_fact) const;

    template <typename EXTFACT> void init_timestep_crit_rare(EXTFACT ext_fact) const;

#ifdef SCATTERING_RARE_DISSIPATION
    // methods for dissipative scattering
    void apply_dissipation_to_v_rel_abs(MyAtLeastDouble &v_rel_abs) const;
#endif // SCATTERING_RARE_DISSIPATION
    MyAtLeastDouble get_r_diss(const MyAtLeastDouble &v_rel2) const;
    MyAtLeastDouble get_dissipation_factor_timestep() const;

    // methods for scattering counts
    void increase_global_scattercount(scatter_inner_base &si);
    void increase_individual_scattercount(scatter_inner_base &si) const;

    // methods for unequal masses
    MyAtLeastDouble get_mass_factor_freq(const scatter_inner_base &si) const;
    MyAtLeastDouble get_mass_factor_rare(const scatter_inner_base &si) const;
    MyAtLeastDouble get_mass_factor_timestep(const scattering_particle_ref &p) const;

  protected:
    std::string name;
};
} // namespace scatter::pxp_scattering

//////////////////////////////////////////////////////////////////////////////////////////////////

namespace scatter::pxp_scattering
{
/**
 * @brief Declaration of outer scattering class; base of Scattering.
 *
 * The scattering modules of scattering_modules.hpp enter here.
 */
template <typename SCTOPT, template <typename SCTOPTM> typename... MODULE>
class scatter_outer : virtual public scatter_outer_base<SCTOPT>, public MODULE<SCTOPT>...
{
  public:
    scatter_outer();
};
} // namespace scatter::pxp_scattering

//////////////////////////////////////////////////////////////////////////////////////////////////

namespace scatter::pxp_scattering
{
/**
 * @brief End of recursion for template that executes multiple functions after each other.
 */
void execute();

/**
 * @brief Template to execute multiple functions after each other.
 */
template <typename FUNC, typename... FUNCS> void execute(FUNC func, FUNCS... funcs);
} // namespace scatter::pxp_scattering

//////////////////////////////////////////////////////////////////////////////////////////////////

#endif /* MSIDM_SCATTERING_BASE_HPP */
