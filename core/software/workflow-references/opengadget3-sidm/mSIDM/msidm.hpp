/**
 * @file
 *
 * This file is part of the N-body/SPH code OpenGadget3.
 *
 * @brief Header file with declaration of the main SIDM class.
 *
 * The main SIDM class is declared here. Besides, the file contains a
 * declaration for a class to compute the maximum velocity that a particle
 * sees initially. In addition, data types for the particle exchange via MPI
 * communication are declared.
 *
 * @author OpenGadget3-development team, in particular Moritz S. Fischer.
 *
 * @copyright Copyright (c) 2026, OpenGadget3-development team lead by Klaus Dolag and Stefano Borgani
 * (see change logs). All rights reserved.
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License v3 or later.
 * See <https://www.gnu.org/licenses/>.
 */

#ifndef MSIDM_MSIDM_H
#define MSIDM_MSIDM_H

#include "scattering.hpp"
#include "execute_pxp.hpp"
#include "scattering_data.hpp"

#define LOCK_NEXPORT
#define UNLOCK_NEXPORT

#include "msidm_base.hpp"
#include "scattering_types.hpp"

//////////////////////////////////////////////////////////////////////////////////////////////////

typedef struct particle_data particle_data;

//////////////////////////////////////////////////////////////////////////////////////////////////

namespace scatter
{
/**
 * @brief Main class for the SIDM module.
 */
class MSIDM final : public execute_pxp<msidm_base>
{
  public:
    MSIDM();
    ~MSIDM();

  protected:
    /** This method contains the scattering physics. */
    void pxp_physics(msidmdata_in_ref &pi, const int &pj) override;

  public:
    /** This method is called in begrun() and initializes all relevant things for the
        self-interactions. */
    void init();

    /** This method is called in run() and initializes various things that are required when
        no restart files are used. */
    void init_calc();

    /** This method is called in get_timestep() and does the scattering of all active particles. */
    void do_full_kick();

    /** This method is called in get_timestep() and computes the SIDM timestep. */
    double calc_dt(particle_data &p) const;

    /** Method to check the parameters of the SIDM module. */
    void check_and_report_settings() const;

    /** Method to print relevant papers for the compiled version. */
    void report_credits() const;

  private:
    /** Instance to compute the kernel overlap. */
    LambdaNumerical<kernel_msidm> calc_lambda;

    /** Instance to do the scattering computations based on pairs of particles. */
    scatter::pxp_scattering::Scattering_msidm scatter;

    /** This method computes the maximum velocity that a particle sees or sets it to
        a specific value. */
    void init_vmax();

    /** Method to compute scatterings between host and satellite particles. */
    void subhalo_halo_interaction();

    /** Normalisation constant used for the subhalo-halo interaction. */
    double lambda_0;
};
} // namespace scatter

//////////////////////////////////////////////////////////////////////////////////////////////////

namespace scatter
{
/**
 * @brief Class to compute maximum velocity that a particles sees initially.
 */
class MSIDM_vmax final : public execute_pxp<msidm_base>
{
  public:
    MSIDM_vmax(pxp_scattering::Scattering_msidm &scatter_in);

  protected:
    /** This method computes the relative velocity of the particles and take the maximum */
    void pxp_physics(msidmdata_in_ref &pi, const int &pj) override;

  private:
    pxp_scattering::Scattering_msidm &scatter;
};
} // namespace scatter

//////////////////////////////////////////////////////////////////////////////////////////////////

namespace scatter
{
/**
 * @brief Data structure for MPI communication.
 */
class msidmdata_in : public pxp_scattering::scattering_particle
{
  public:
    msidmdata_in();
    msidmdata_in(const int place);
    void particle2in(const int place); // this is used to perpare particles to be send to other process
};
} // namespace scatter

//////////////////////////////////////////////////////////////////////////////////////////////////

namespace scatter
{
/**
 * @brief Data structure for MPI communication.
 */
class msidmdata_out
{
  public:
    msidmdata_out();
    msidmdata_out(const msidmdata_in &pi_in);
    void in2out(const msidmdata_in &pi_in); // this is used to copy data in preparation of sending guest particles back
    void out2particle(
        const int place) const; // this is used to copy particles to particle data array after receving them back

    MyFloat Vel[3];

#ifdef mSIDM_VMAX
    MyAtLeastDouble v_max;
#endif // mSIDM_VMAX

#ifdef SCATTERING_HMIN
    MyAtLeastDouble h_min;
#endif // SCATTERING_HMIN

#ifdef SCATTERING_COUNT_INDI
    int scatterCount;
#endif // SCATTERING_COUNT_INDI
};
} // namespace scatter

//////////////////////////////////////////////////////////////////////////////////////////////////

namespace scatter
{
/**
 * @brief Reference struct for scattering computation.
 */
class msidmdata_in_ref : public pxp_scattering::scattering_particle_ref
{
  public:
    msidmdata_in_ref(class msidmdata_in &p_in);
    msidmdata_in_ref(const int place);
    msidmdata_in_ref(particle_data &p_in);
};
} // namespace scatter

//////////////////////////////////////////////////////////////////////////////////////////////////

#endif /* MSIDM_MSIDM_H */
