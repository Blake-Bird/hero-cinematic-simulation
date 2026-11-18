/**
 * @file
 *
 * This file is part of the N-body/SPH code OpenGadget3.
 *
 * @brief Source file for the scattering base class.
 *
 * @author OpenGadget3-development team, in particular Moritz S. Fischer.
 *
 * @copyright Copyright (c) 2026, OpenGadget3-development team lead by Klaus Dolag and Stefano Borgani
 * (see change logs). All rights reserved.
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License v3 or later.
 * See <https://www.gnu.org/licenses/>.
 */

#ifndef MSIDM_SCATTERING_BASE_CPP
#define MSIDM_SCATTERING_BASE_CPP

#include "scattering_base.hpp"
#include "../CodeBase/proto.hpp"

namespace s = scatter::pxp_scattering;
using namespace s;

//////////////////////////////////////////////////////////////////////////////////////////////////
// definitions for vec3d
/**
 * @brief Constructor of the vector class.
 */
template <typename DTYPE> vec3d<DTYPE>::vec3d() : x(array[0]), y(array[1]), z(array[2])
{
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Constructor of the vector class.
 */
template <typename DTYPE>
vec3d<DTYPE>::vec3d(std::initializer_list<DTYPE> inputs) : x(array[0]), y(array[1]), z(array[2])
{
    int loop = 0;
    for (auto element : inputs)
    {
        array[loop++] = DTYPE(element);
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Constructor of the vector class.
 */
template <typename DTYPE> vec3d<DTYPE>::vec3d(const vec3d<DTYPE> &input) : x(array[0]), y(array[1]), z(array[2])
{
    for (int loop = 0; loop < 3; ++loop)
    {
        array[loop] = input.array[loop];
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Definition of the [] operator for the vector class.
 */
template <typename DTYPE> DTYPE &vec3d<DTYPE>::operator[](const int &index)
{
    return array[index];
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Definition of the [] operator for the vector class for reading only.
 */
template <typename DTYPE> DTYPE vec3d<DTYPE>::operator[](const int &index) const
{
    return array[index];
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Definition of the = operator for the vector class.
 */
template <typename DTYPE> vec3d<DTYPE> &vec3d<DTYPE>::operator=(const vec3d<DTYPE> &a)
{
    for (short loopc = 0; loopc < 3; ++loopc)
    {
        array[loopc] = a.array[loopc];
    }
    return *this;
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Definition of the += operator for the vector class.
 */
template <typename DTYPE> vec3d<DTYPE> &vec3d<DTYPE>::operator+=(const vec3d<DTYPE> &a)
{
    for (short loopc = 0; loopc < 3; ++loopc)
    {
        array[loopc] += a.array[loopc];
    }
    return *this;
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Definition of the -= operator for the vector class.
 */
template <typename DTYPE> vec3d<DTYPE> &vec3d<DTYPE>::operator-=(const vec3d<DTYPE> &a)
{
    for (short loopc = 0; loopc < 3; ++loopc)
    {
        array[loopc] -= a.array[loopc];
    }
    return *this;
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Definition of the *= operator for the vector class.
 */
template <typename DTYPE> vec3d<DTYPE> &vec3d<DTYPE>::operator*=(DTYPE a)
{
    for (short loopc = 0; loopc < 3; ++loopc)
    {
        array[loopc] *= a;
    }
    return *this;
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Definition of the /= operator for the vector class.
 */
template <typename DTYPE> vec3d<DTYPE> &vec3d<DTYPE>::operator/=(DTYPE a)
{
    for (short loopc = 0; loopc < 3; ++loopc)
    {
        array[loopc] /= a;
    }
    return *this;
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Definition of the - operator for the vector class.
 */
template <typename DTYPE> vec3d<DTYPE> vec3d<DTYPE>::operator-(const vec3d<DTYPE> &a)
{
    vec3d<DTYPE> v;
    for (short loopc = 0; loopc < 3; ++loopc)
    {
        v.array[loopc] = array[loopc] - a.array[loopc];
    }
    return v;
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Definition of the + operator for the vector class.
 */
template <typename DTYPE> vec3d<DTYPE> vec3d<DTYPE>::operator+(const vec3d<DTYPE> &a)
{
    vec3d<DTYPE> v;
    for (short loopc = 0; loopc < 3; ++loopc)
    {
        v.array[loopc] = array[loopc] + a.array[loopc];
    }
    return v;
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Definition of the - operator for the vector class.
 */
template <typename DTYPE> vec3d<DTYPE> vec3d<DTYPE>::operator-(DTYPE d) const
{
    vec3d<DTYPE> v;
    for (short loopc = 0; loopc < 3; ++loopc)
    {
        v.array[loopc] = array[loopc] - d;
    }
    return v;
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Definition of the + operator for the vector class.
 */
template <typename DTYPE> vec3d<DTYPE> vec3d<DTYPE>::operator+(DTYPE d) const
{
    vec3d<DTYPE> v;
    for (short loopc = 0; loopc < 3; ++loopc)
    {
        v.array[loopc] = array[loopc] + d;
    }
    return v;
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Definition of the * operator for the vector class.
 */
template <typename DTYPE> vec3d<DTYPE> vec3d<DTYPE>::operator*(DTYPE d) const
{
    vec3d<DTYPE> v;
    for (short loopc = 0; loopc < 3; ++loopc)
    {
        v.array[loopc] = array[loopc] * d;
    }
    return v;
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Definition of the / operator for the vector class.
 */
template <typename DTYPE> vec3d<DTYPE> vec3d<DTYPE>::operator/(DTYPE d) const
{
    vec3d<DTYPE> v;
    for (short loopc = 0; loopc < 3; ++loopc)
    {
        v.array[loopc] = array[loopc] / d;
    }
    return v;
}

//////////////////////////////////////////////////////////////////////////////////////////////////
// functions to compute time steps
/**
 * @brief Function to compute time step for comoving integration based on time step bin.
 */
double s::calc_dt_comoving(const short int &TimeBin, const integertime &Ti_begstep)
{
    const integertime ti_step = (TimeBin > 0) * (integertime(1) << TimeBin);
    const integertime tstart = Ti_begstep; /* beginning of step */
    const integertime tend = Ti_begstep + ti_step; /* endpoint of step */

    // intergration over a dependence - better version
    return get_scattering_kick_factor(tstart, tend) * All.HubbleParam;
    // no intergration over a dependence - simple version
    // return All.Timebase_interval * ti_step * All.HubbleParam / (pow(All.Time,4) * hubble_function(A>
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Function to compute time step for non-comoving integration based on time step bin.
 */
double s::calc_dt_lin(const short int &TimeBin, const integertime &Ti_begstep)
{
    const integertime ti_step = (TimeBin > 0) * (integertime(1) << TimeBin);
    return ti_step * All.Timebase_interval;
}

//////////////////////////////////////////////////////////////////////////////////////////////////
// functions to compute time steps
/**
 * @brief Wrapper function to compute time step for comoving and non-comoving integration.
 */
double s::calc_dt_wrapper(const short int &TimeBin, const integertime &Ti_begstep)
{
    static double (*my_calc_dt)(const short int &, const integertime &) =
        All.ComovingIntegrationOn ? calc_dt_comoving : calc_dt_lin;
    return my_calc_dt(TimeBin, Ti_begstep);
}

//////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////
// definitions for scatter_inner_base
/**
 * @brief Constructor of the scatter_inner_base class.
 */
#if defined(SCATTERING_UNEQUAL_MASS) && !defined(SCATTERING_INDI_MASS_RATIO)
scatter_inner_base::scatter_inner_base(scattering_particle_ref &pi_in, scattering_particle_ref &pj_in,
                                       MyAtLeastDouble &mass_ratio_in, MyAtLeastDouble &specific_reduced_mass_in)
    : si(*this), pi(pi_in), pj(pj_in), mass_ratio(mass_ratio_in), specific_reduced_mass(specific_reduced_mass_in)
{
}
#else // SCATTERING_UNEQUAL_MASS && !SCATTERING_INDI_MASS_RATIO
scatter_inner_base::scatter_inner_base(scattering_particle_ref &pi_in, scattering_particle_ref &pj_in)
    : si(*this), pi(pi_in), pj(pj_in)
{
}
#endif // SCATTERING_UNEQUAL_MASS && !SCATTERING_INDI_MASS_RATIO

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Definition of the method to compute the relative velocity of interacting particles.
 */
void scatter_inner_base::compute_velocity()
{
    // calculate relative velocity
    v_rel2 = 0;
    for (short loopc = 0; loopc < 3; ++loopc)
    {
        v_rel[loopc] = pi.Vel[loopc] - pj.Vel[loopc];
        v_rel2 += v_rel[loopc] * v_rel[loopc];
    }
    v_rel_abs = sqrt(v_rel2);
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Method that computes the time step used for a particle pair.
 *
 * The time step depends on whether only one or both particles are active.
 * get_dt = get_dt_full * get_dt_fact, get_dt takes into account how often a particle is
 * considered per timestep, this depends on whether the particle is active or passive.
 */
MyAtLeastDouble scatter_inner_base::get_dt() const
{
    const MyAtLeastDouble pi_dt = calc_dt_wrapper(pi.TimeBin, pi.Ti_begstep);

    if (!TimeBinActive[pj.TimeBin]) // check whether pj is active
    {
        return pi_dt;
    }
    else
    {
        const MyAtLeastDouble pj_dt = calc_dt_wrapper(pj.TimeBin, pj.Ti_begstep);
#ifdef SCATTERING_DOACTIVEONCE
        return std::min(pi_dt, pj_dt);
#else // SCATTERING_DOACTIVEONCE
        if (TimeBinActive[pi.TimeBin])
        {
            return 0.5 * std::min(pi_dt, pj_dt);
        }
        else
        {
            return pj_dt;
        }
#endif // SCATTERING_DOACTIVEONCE
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Method that computes the full time step by which a particle pair needs to be advanced.
 */
MyAtLeastDouble scatter_inner_base::get_dt_full() const
{
    const MyAtLeastDouble pi_dt = calc_dt_wrapper(pi.TimeBin, pi.Ti_begstep);

    if (!TimeBinActive[pj.TimeBin]) // check whether pj is active
    {
        return pi_dt;
    }
    else
    {
        const MyAtLeastDouble pj_dt = calc_dt_wrapper(pj.TimeBin, pj.Ti_begstep);
        return std::min(pi_dt, pj_dt);
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Method that returns factor between get_dt and get_dt_full.
 */
MyAtLeastDouble scatter_inner_base::get_dt_fact() const
{
#ifdef SCATTERING_DOACTIVEONCE
    return 1.0;
#else // SCATTERING_DOACTIVEONCE
    if (!TimeBinActive[pj.TimeBin] || !TimeBinActive[pi.TimeBin])
    {
        return 1.0;
    }
    else
    {
        return 0.5;
    }
#endif // SCATTERING_DOACTIVEONCE
}

//////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////
// definitions for random_numbers
/**
 * @brief Constructor for random number class.
 */
template <typename SCTOPT>
random_numbers<SCTOPT>::random_numbers()
    : random_phi(0.0, 2.0 * M_PI), random_scatter_num(0.0, 1.0), random_cos_theta(-1.0, 1.0)
{
#ifdef SCATTERING_OPENMP
    omp_init_lock(&lock_generator);
#endif // SCATTERING_OPENMP
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Destructor for random number class.
 */
template <typename SCTOPT> random_numbers<SCTOPT>::~random_numbers()
{
#ifdef SCATTERING_OPENMP
    omp_destroy_lock(&lock_generator);
#endif // SCATTERING_OPENMP
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Method to initialize random number class.
 */
template <typename SCTOPT> void random_numbers<SCTOPT>::init()
{
#ifdef mSIDM
    if constexpr (SCTOPT::do_sidm)
    {
        seed = All.MSIDM_SEED + ThisTask;
    }
#endif // mSIDM

    generator.seed(seed);
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Method to draw a random azimuthal angle.
 */
template <typename SCTOPT> MyAtLeastDouble random_numbers<SCTOPT>::get_random_phi()
{
#ifdef SCATTERING_OPENMP
    omp_set_lock(&lock_generator);
    const MyAtLeastDouble phi = random_phi(generator);
    omp_unset_lock(&lock_generator);
    return phi;
#else // SCATTERING_OPENMP
    return random_phi(generator);
#endif // SCATTERING_OPENMP
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Method to draw a random number used to decide if particles interact.
 */
template <typename SCTOPT> MyAtLeastDouble random_numbers<SCTOPT>::get_rnd_scatter_num()
{
#ifdef SCATTERING_OPENMP
    omp_set_lock(&lock_generator);
    const MyAtLeastDouble num = random_scatter_num(generator);
    omp_unset_lock(&lock_generator);
    return num;
#else // SCATTERING_OPENMP
    return random_scatter_num(generator);
#endif // SCATTERING_OPENMP
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Method to draw a random value for cos(theta), to determine the scattering angle.
 */
template <typename SCTOPT> MyAtLeastDouble random_numbers<SCTOPT>::get_rnd_cos_theta()
{
#ifdef SCATTERING_OPENMP
    omp_set_lock(&lock_generator);
    const MyAtLeastDouble cos_theta = random_cos_theta(generator);
    omp_unset_lock(&lock_generator);
    return cos_theta;
#else // SCATTERING_OPENMP
    return random_cos_theta(generator);
#endif // SCATTERING_OPENMP
}

//////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////
// constructor for scatter_outer_base
/**
 * @brief Constructor for scatter_outer_base.
 */
template <typename SCTOPT> scatter_outer_base<SCTOPT>::scatter_outer_base() : name("")
{
}

//////////////////////////////////////////////////////////////////////////////////////////////////
// definitions for scatter_outer_base
/**
 * @brief Default definition for method to debug position and velocity of particle pair.
 */
template <typename SCTOPT> void scatter_outer_base<SCTOPT>::debug_pos_and_vel(const scatter_inner_base &si) const
{
}

//////////////////////////////////////////////////////////////////////////////////////////////////
// methods for interaction criterion
/**
 * @brief Default definition for the method to debug position and velocity of a particle pair.
 */
template <typename SCTOPT> bool scatter_outer_base<SCTOPT>::interact_crit_count(scatter_inner_base &si) const
{
    return true;
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Default definition for the method to select particles according to their ID.
 */
template <typename SCTOPT> bool scatter_outer_base<SCTOPT>::interact_crit_id(scatter_inner_base &si) const
{
    return true;
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Default definition for the method to select particles according to their mass.
 */
template <typename SCTOPT> bool scatter_outer_base<SCTOPT>::interact_crit_mass(scatter_inner_base &si) const
{
    return true;
}

//////////////////////////////////////////////////////////////////////////////////////////////////
// methods for scatter count
/**
 * @brief Default definition for the method to exchange scatter count.
 */
template <typename SCTOPT> void scatter_outer_base<SCTOPT>::exchange_scattercount() const
{
}

//////////////////////////////////////////////////////////////////////////////////////////////////
// methods for frequent scattering
/**
 * @brief Default definition for the method to compute frequent scatterings.
 */
template <typename SCTOPT>
template <typename SIGMADM, typename RDISS, typename MASSFACT>
void scatter_outer_base<SCTOPT>::compute_freq_scat(scatter_inner_base &si, SIGMADM sigmaDm, RDISS r_diss,
                                                   MASSFACT massfact)
{
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Default definition for the method to return the transfer cross-section.
 */
template <typename SCTOPT>
MyAtLeastDouble scatter_outer_base<SCTOPT>::calc_sigmaDm_f(const MyAtLeastDouble &v_rel) const
{
    ENDRUN_MAIN("Invalid configuration, cross-section for frequent interactions unkown.");
    return 0.0;
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Default definition for the method to return the transfer cross-section.
 */
template <typename SCTOPT>
template <typename VDEP>
MyAtLeastDouble scatter_outer_base<SCTOPT>::calc_sigmaDm_f(VDEP vdep, const MyAtLeastDouble &v_rel) const
{
    ENDRUN_MAIN("Invalid configuration, cross-section for frequent interactions unkown.");
    return 0.0;
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Default definition for the method to return the frequent time step factor.
 */
template <typename SCTOPT>
template <typename SIGMADMVEL>
double scatter_outer_base<SCTOPT>::freq_time_fact(SIGMADMVEL sigmaDm_vel) const
{
    return FLT_MAX;
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Default definition for the maximum of cross-section times velocitiy.
 */
template <typename SCTOPT> MyFloat scatter_outer_base<SCTOPT>::freq_sigmaDm_vel() const
{
    ENDRUN_MAIN("Invalid configuration, freq_sigmaDm_vel not specified.");
    return 0.0;
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Default definition for the maximum of cross-section times velocitiy.
 */
template <typename SCTOPT>
template <typename VDEP>
MyFloat scatter_outer_base<SCTOPT>::freq_sigmaDm_vel(VDEP vdep, const MyAtLeastDouble &v_max) const
{
    ENDRUN_MAIN("Invalid configuration, freq_sigmaDm_vel not specified.");
    return 0.0;
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Default definition for the maximum of cross-section times velocitiy.
 */
template <typename SCTOPT> MyFloat scatter_outer_base<SCTOPT>::freq_sigmaDm_vel(const MyAtLeastDouble &v_max) const
{
    ENDRUN_MAIN("Invalid configuration, freq_sigmaDm_vel not specified.");
    return 0.0;
}

//////////////////////////////////////////////////////////////////////////////////////////////////
// methods for rare scattering
/**
 * @brief Default definition for the method to compute rare scatterings.
 */
template <typename SCTOPT>
template <typename SIGMADM, typename ANGLE, typename... FUNCS>
void scatter_outer_base<SCTOPT>::compute_rare_scat(scatter_inner_base &si, SIGMADM sigmaDm, ANGLE angle, FUNCS... funcs)
{
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Default definition for the method to return the total cross-section.
 */
template <typename SCTOPT>
MyAtLeastDouble scatter_outer_base<SCTOPT>::calc_sigmaDm_r(const MyAtLeastDouble &v_rel) const
{
    ENDRUN_MAIN("Invalid configuration, cross-section for rare interactions unkown.");
    return 0.0;
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Default definition for the method to return the total cross-section.
 */
template <typename SCTOPT>
template <typename VDEP>
MyAtLeastDouble scatter_outer_base<SCTOPT>::calc_sigmaDm_r(VDEP vdep, const MyAtLeastDouble &v_rel) const
{
    ENDRUN_MAIN("Invalid configuration, cross-section for rare interactions unkown.");
    return 0.0;
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Default definition for the method to compute the scattering angle.
 */
template <typename SCTOPT>
MyAtLeastDouble scatter_outer_base<SCTOPT>::calc_angle_r(const MyAtLeastDouble &v_rel,
                                                         const MyAtLeastDouble &rnd_num) const
{
    ENDRUN_MAIN("Invalid configuration, scattering angle for rare interactions unkown.");
    return 0.0;
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Default definition for the method to return the rare time step factor.
 */
template <typename SCTOPT>
template <typename SIGMADMVEL>
double scatter_outer_base<SCTOPT>::rare_time_fact(SIGMADMVEL sigmaDm_vel) const
{
    return FLT_MAX;
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Default definition for the maximum of cross-section times velocitiy.
 */
template <typename SCTOPT> MyFloat scatter_outer_base<SCTOPT>::rare_sigmaDm_vel() const
{
    ENDRUN_MAIN("Invalid configuration, rare_sigmaDm_vel not specified.");
    return 0.0;
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Default definition for the maximum of cross-section times velocitiy.
 */
template <typename SCTOPT>
template <typename VDEP>
MyFloat scatter_outer_base<SCTOPT>::rare_sigmaDm_vel(VDEP vdep, const MyAtLeastDouble &v_max) const
{
    ENDRUN_MAIN("Invalid configuration, rare_sigmaDm_vel not specified.");
    return 0.0;
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Default definition for the maximum of cross-section times velocitiy.
 */
template <typename SCTOPT> MyFloat scatter_outer_base<SCTOPT>::rare_sigmaDm_vel(const MyAtLeastDouble &v_max) const
{
    ENDRUN_MAIN("Invalid configuration, rare_sigmaDm_vel not specified.");
    return 0.0;
}

//////////////////////////////////////////////////////////////////////////////////////////////////
// methods for velocity dependence
/**
 * @brief Default definition for the velocity dependence of the cross-section.
 */
template <typename SCTOPT>
MyAtLeastDouble scatter_outer_base<SCTOPT>::sigmaDm_vdep_fact(const MyAtLeastDouble &v_rel) const
{
    return 1.0;
}

//////////////////////////////////////////////////////////////////////////////////////////////////
// initialization of time step criterion
/**
 * @brief Default definition for the velocity at which sigma/m v becomes maximal.
 */
template <typename SCTOPT> MyAtLeastDouble scatter_outer_base<SCTOPT>::sigmaDm_vel_ext_fact() const
{
    return 0.0;
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Default definition for the method to initialize variables for the time step criterion.
 */
template <typename SCTOPT>
template <typename EXTFACT>
void scatter_outer_base<SCTOPT>::init_timestep_crit_freq(EXTFACT ext_fact) const
{
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Default definition for the method to initialize variables for the time step criterion.
 */
template <typename SCTOPT>
template <typename EXTFACT>
void scatter_outer_base<SCTOPT>::init_timestep_crit_rare(EXTFACT ext_fact) const
{
}

#ifdef SCATTERING_RARE_DISSIPATION
//////////////////////////////////////////////////////////////////////////////////////////////////
// methods for dissipative scattering
/**
 * @brief Default definition of method that reduces relative velocity to model dissipation.
 */
template <typename SCTOPT>
void scatter_outer_base<SCTOPT>::apply_dissipation_to_v_rel_abs(MyAtLeastDouble &v_rel_abs) const
{
}
#endif // SCATTERING_RARE_DISSIPATION

//////////////////////////////////////////////////////////////////////////////////////////////////

template <typename SCTOPT> MyAtLeastDouble scatter_outer_base<SCTOPT>::get_r_diss(const MyAtLeastDouble &v_rel2) const
{
    return 1.0;
}

//////////////////////////////////////////////////////////////////////////////////////////////////

template <typename SCTOPT> MyAtLeastDouble scatter_outer_base<SCTOPT>::get_dissipation_factor_timestep() const
{
    return 1.0;
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Default definition of the method to increase the scatter count.
 */
template <typename SCTOPT> void scatter_outer_base<SCTOPT>::increase_global_scattercount(scatter_inner_base &si)
{
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Default definition of the method to increase the individual scatter count of particles.
 */
template <typename SCTOPT>
void scatter_outer_base<SCTOPT>::increase_individual_scattercount(scatter_inner_base &si) const
{
}

//////////////////////////////////////////////////////////////////////////////////////////////////
// methods for unequal masses
/**
 * @brief Method to return the frequent mass factor for scattering.
 */
template <typename SCTOPT>
MyAtLeastDouble scatter_outer_base<SCTOPT>::get_mass_factor_freq(const scatter_inner_base &si) const
{
    ENDRUN_MAIN("get_mass_fact_freq has not been specified.");
    return 0.5; // case for equal masses
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Method to return the rare mass factor for scattering.
 */
template <typename SCTOPT>
MyAtLeastDouble scatter_outer_base<SCTOPT>::get_mass_factor_rare(const scatter_inner_base &si) const
{
    ENDRUN_MAIN("get_mass_fact_rare has not been specified.");
    return All.MassTable[1]; // case for equal masses
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Method to return the time step mass factor for scattering.
 */
template <typename SCTOPT>
MyAtLeastDouble scatter_outer_base<SCTOPT>::get_mass_factor_timestep(const scattering_particle_ref &p) const
{
    ENDRUN_MAIN("get_mass_fact_timestep has not been specified.");
    return p.Mass;
}

//////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////
// outer scattering class, base of Scattering
/**
 * @brief Constructor of the scatter_outer class.
 */
template <typename SCTOPT, template <typename SCTOPTM> typename... MODULE>
scatter_outer<SCTOPT, MODULE...>::scatter_outer() : MODULE<SCTOPT>()...
{
}

//////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////
// variadic template for executing multiple functions after each other
/**
 * @brief End of recursion for template that executes multiple functions after each other.
 */
void s::execute()
{
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Template to execute multiple functions after each other.
 */
template <typename FUNC, typename... FUNCS> void s::execute(FUNC func, FUNCS... funcs)
{
    func();
    execute(funcs...);
}

//////////////////////////////////////////////////////////////////////////////////////////////////

#endif /* MSIDM_SCATTERING_BASE_CPP */
