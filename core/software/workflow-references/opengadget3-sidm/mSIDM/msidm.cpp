/**
 * @file
 *
 * This file is part of the N-body/SPH code OpenGadget3.
 *
 * @brief Source file with the definitions for the main SIDM module class.
 *
 * The main SIDM class is defined here. Besides, the file contains a class to
 * compute the maximum velocity that a particle sees initially. In addition,
 * data types for the particle exchange via MPI communication are defined.
 *
 * @author OpenGadget3-development team, in particular Moritz S. Fischer.
 *
 * @copyright Copyright (c) 2026, OpenGadget3-development team lead by Klaus Dolag and Stefano Borgani
 * (see change logs). All rights reserved.
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License v3 or later.
 * See <https://www.gnu.org/licenses/>.
 */

#include "msidm.hpp"

#include "../CodeBase/allvars.hpp"
#include "../CodeBase/proto.hpp"

#ifdef NODELISTLENGTH
#undef NODELISTLENGTH
#endif // NODELISTLENGTH

#define NODELISTLENGTH NODELISTLENGTH_PXP
#define DataNodeList scatter::pxp_nodelist::DataNodeList_pxp
#define DataNodeListIn scatter::pxp_nodelist::DataNodeListIn_pxp
#define DataNodeListGet scatter::pxp_nodelist::DataNodeListGet_pxp
#include "../CodeBase/greentree.hpp"
#include "../System/communication.hpp"
#define JSON_ASSERT(x)
#include "tools.hpp"
#include <nlohmann/json.hpp>

//////////////////////////////////////////////////////////////////////////////////////////////////

using namespace scatter;
using namespace scatter::tools;

scatter::msidm_isactive_class msidm_isactive;

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Constructor of the main class of the SIDM module.
 */
scatter::MSIDM::MSIDM()
    : msidm_base(Module_mSIDM, NeighbourSearchCriteria_mSIDM), calc_lambda("mSIDM_itable"), scatter(calc_lambda),
      lambda_0(0.0)
{
    std::cout.clear();
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Destructor of the main class of the SIDM module.
 */
scatter::MSIDM::~MSIDM()
{
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief This method is called in begrun() and initializes all relevant things for the self-interactions.
 */
void scatter::MSIDM::init()
{
    TIMER_PUSH(1, "mSIDM");
    TIMER_PUSH(1, "mSIDM_init");

    coutr(info) << "Initialize mSIDM:" << endl;

    calc_lambda.setup();

#ifdef mSIDM_SSHI
    lambda_0 = calc_lambda(1.0, 1.0, 0.0);
#endif // mSIDM_SSHI

    nlohmann::json params{};

#if defined(fSIDM) && defined(mSIDM_TIMESTEP)
    params["freq_tfact"] = All.FSIDM_tfact;
#endif // fSIDM && mSIDM_TIMESTEP
#if defined(rSIDM) && defined(mSIDM_TIMESTEP)
    params["rare_tfact"] = All.RSIDM_tfact;
#endif // rSIDM && mSIDM_TIMESTEP
#ifdef fSIDM_SIMPLE
    params["sigmaDm_f"] = All.FSIDM_CrossSectionPerMass_in_cgs;
    params["crosssection_f"] = All.FSIDM_CrossSectionDef;
#endif // fSIDM_SIMPLE
#ifdef rSIDM_ISOTROPIC
    params["sigmaDm_r"] = All.RSIDM_CrossSectionPerMass_in_cgs;
    params["crosssection_r"] = All.RSIDM_CrossSectionDef;
#endif // rSIDM_ISOTROPIC
#if defined(mSIDM_VDEP0) || defined(mSIDM_VDEP1) || defined(mSIDM_VDEP_MV) || defined(mSIDM_VDEP_RV)
    params["vdep_w"] = All.MSIDM_vdep_w;
#endif // mSIDM_VDEP0 || mSIDM_VDEP1 || mSIDM_VDEP_MV || mSIDM_VDEP_RV
#ifdef mSIDM_VDEP0
    params["vdep_alpha"] = All.MSIDM_vdep_alpha;
    params["vdep_beta"] = All.MSIDM_vdep_beta;
#endif // mSIDM_VDEP0
#ifdef mSIDM_VDEP1
    params["vdep_u"] = All.MSIDM_vdep_u;
#endif // mSIDM_VDEP1
#ifdef rSIDM_DISSIPATION
    params["f_diss"] = All.RSIDM_f_diss;
#endif // rSIDM_DISSIPATION
#ifdef fSIDM_DISSIPATION
    params["const_r_diss"] = All.FSIDM_const_r_diss;
#endif // fSIDM_DISSIPATION
#ifdef mSIDM_DIFFERENTIAL_CROSS_SECTION
    params["crosssection"] = std::string(All.MSIDM_CrossSection);
    params["params"] = std::string(All.MSIDM_DCS_params);

    params["thetaC_fwd"] = All.MSIDM_thetaC_fwd;
    params["thetaC_bwd"] = All.MSIDM_thetaC_bwd;
    params["v_min"] = All.MSIDM_v_min;
    params["v_max"] = All.MSIDM_v_max;
    params["len_xh"] = All.MSIDM_len_xh;
    params["len_v"] = All.MSIDM_len_v;
    params["reserve_fact"] = All.MSIDM_reserve_fact;
#endif // mSIDM_DIFFERENTIAL_CROSS_SECTION
    scatter.sec_const("SIDM", params);

    std::cout.clear();

    TIMER_POP(1, "mSIDM_init");
    TIMER_POP(1, "mSIDM");
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Method for inital computations after parameter file and ICs are read.
 *
 * This method is called in run() and initializes various things that are required when
 * no restart files are used.
 */
void scatter::MSIDM::init_calc()
{
    TIMER_PUSH(1, "mSIDM");

#if defined(mSIDM_VMAX) || defined(mSIDM_HMIN)
    init_vmax();
#endif // mSIDM_VMAX || mSIDM_HMIN

    std::cout.clear();

    TIMER_POP(1, "mSIDM");
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Method to initialize maximum velocity that a particle has seen.
 *
 * This method computes the maximum velocity that a particle sees or sets it to
 * a specific value
 */
void scatter::MSIDM::init_vmax()
{
    for (int i = 0; i < NumPart; ++i)
    {
#ifdef mSIDM_VMAX
        P[i].mSIDM_v_max = 0.0;
#endif // mSIDM_VMAX
#ifdef mSIDM_HMIN
        P[i].mSIDM_h_min = FLT_MAX;
#endif // mSIDM_HMIN
    }

    coutr(info) << "Start mSIDM initial velocity calculation" << endl;

    MSIDM_vmax my_vmax(scatter);
    my_vmax.all();

    coutr(info) << "mSIDM initial velocity calculation done" << endl;
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief This function is called in run() and does the scattering of all active particles.
 */
void scatter::MSIDM::do_full_kick()
{
    TIMER_PUSH(1, "mSIDM");

#if defined(fSIDM) && defined(rSIDM)
    coutr(info) << "Start fSIDM and rSIDM scattering" << endl;
#elif defined(fSIDM)
    coutr(info) << "Start fSIDM scattering" << endl;
#elif defined(rSIDM)
    coutr(info) << "Start rSIDM scattering" << endl;
#endif // rSIDM

    all(); // does the scattering and updates the velocities

    subhalo_halo_interaction(); // SSHI: scattering-induced subhalo-halo interaction

    scatter.exchange_scatter_count();

#if defined(fSIDM) && defined(rSIDM)
    coutr(info) << "fSIDM and rSIDM scattering done" << endl;
#elif defined(fSIDM)
    coutr(info) << "fSIDM scattering done" << endl;
#elif defined(rSIDM)
    coutr(info) << "rSIDM scattering done" << endl;
#endif // rSIDM

    std::cout.clear();

    TIMER_POP(1, "mSIDM");
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief This method is called in get_timestep() and computes the SIDM timestep.
 */
double scatter::MSIDM::calc_dt(particle_data &p) const
{
    const double dt = scatter.calc_dt(msidmdata_in_ref(p));

    std::cout.clear();

    return dt;
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief This method contains the scattering physics.
 */
void scatter::MSIDM::pxp_physics(msidmdata_in_ref &pi, const int &place)
{
    msidmdata_in_ref pj(place);
    scatter(pi, pj);
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Method to check the parameters of the SIDM module.
 */
void scatter::MSIDM::check_and_report_settings() const
{
    TIMER_PUSH(1, "mSIDM");

#ifdef fSIDM_SIMPLE
    ENDRUN_IF(All.FSIDM_CrossSectionPerMass_in_cgs < 0.0, "FSIDM_CrossSectionPerMass_in_cgs is not\
                                                                       allowed to be negative");
#endif // fSIDM_SIMPLE

#ifdef rSIDM_ISOTROPIC
    ENDRUN_IF(All.RSIDM_CrossSectionPerMass_in_cgs < 0.0, "RSIDM_CrossSectionPerMass_in_cgs is not\
                                                                       allowed to be negative");
#endif // rSIDM_ISOTROPIC

#ifdef mSIDM_VDEP0
    ENDRUN_IF(All.MSIDM_vdep_w <= 0.0, "MSIDM_vdep_w must be larger than zero");
    ENDRUN_IF(All.MSIDM_vdep_alpha >= 0.0, "MSIDM_vdep_alpha must be smaller than zero");
    ENDRUN_IF(All.MSIDM_vdep_beta <= 0.0, "MSIDM_vdep_beta must be larger than zero");
#endif // mSIDM_VDEP0

#ifdef mSIDM_VDEP1
    ENDRUN_IF(All.MSIDM_vdep_w <= 0.0, "MSIDM_vdep_w must be larger than zero");
    ENDRUN_IF(All.MSIDM_vdep_u <= 0.0, "MSIDM_vdep_u must be larger than zero");
#endif // mSIDM_VDEP1

#ifdef mSIDM_TIMESTEP
#ifdef fSIDM
    ENDRUN_IF(All.FSIDM_tfact <= 0.0, "FSIDM_tfact must be larger than zero");
#endif // fSIDM
#ifdef rSIDM
    ENDRUN_IF(All.RSIDM_tfact <= 0.0, "RSIDM_tfact must be larger than zero");
#endif // rSIDM
#ifdef mSIDM_VDEP0
    ENDRUN_IF(All.MSIDM_vdep_alpha >= -1.0,
              "MSIDM_vdep_alpha must be smaller than minus one for the time step criterion");
#endif // mSIDM_VDEP0
#endif // mSIDM_TIMESTEP


#ifndef ADAPTGRAVSOFT
    ENDRUN_IF(All.DesNumNgb_pt1 <= 0, "DesNumNgb_pt1 must be larger than zero");
    ENDRUN_IF(All.MaxNumNgbDeviation_pt1 < 0, "MaxNumNgbDeviation_pt1 is not allowed to be negative");
#endif // ADAPTGRAVSOFT

#if defined(rSIDM_SANGLE90) || defined(mSIDM_DIFFERENTIAL_CROSS_SECTION)
    bool is_symmetric = true;
#ifdef mSIDM_UNEQUALMASS
    is_symmetric = false;
#endif // mSIDM_UNEQUALMASS
#endif // rSIDM_SANGLE90 || mSIDM_DIFFERENTIAL_CROSS_SECTION

    //  Check for valid values in SIDM DCS parameters
#ifdef mSIDM_DIFFERENTIAL_CROSS_SECTION
    ENDRUN_IF(All.MSIDM_v_min < 0, "SIDM DCS | MSIDM_v_min must be non-negative.");
    ENDRUN_IF(All.MSIDM_v_min >= All.MSIDM_v_max, "SIDM DCS | MSIDM_v_max must be larger than MSIDM_v_min.");
    ENDRUN_IF(All.MSIDM_thetaC_fwd < 0 || All.MSIDM_thetaC_fwd > M_PI / 2,
              "MSIDM_thetaC_fwd must be in the range [0, pi/2]");
    ENDRUN_IF(All.MSIDM_thetaC_bwd < 0 || All.MSIDM_thetaC_bwd > M_PI / 2,
              "MSIDM_thetaC_bwd must be in the range [0, pi/2]");

    if (static_cast<std::string>(All.MSIDM_CrossSection) == "rutherford")
    {
        is_symmetric = false;
    }

    if (is_symmetric) // symmetric cross-sections
    {
#if !defined(rSIDM_SANGLE90) && defined(fSIDM) && defined(rSIDM)
        ENDRUN_IF(All.MSIDM_thetaC_bwd > 0, "You may want to turn on rSIDM_SANGLE90.");
#endif // rSIDM_SANGLE90
    }
    else // asymmetric cross-sections
    {
        ENDRUN_IF(All.MSIDM_thetaC_bwd != 0, "MSIDM_thetaC_bwd must be zero for asymmetric cross-sections.");
    }
#endif // mSIDM_DIFFERENTIAL_CROSS_SECTION

#ifdef rSIDM_SANGLE90
    ENDRUN_IF(!is_symmetric, "It may not make sense to use rSIDM_SANGLE90 for an asymmetric cross-section.");
#endif // rSIDM_SANGLE90

    if ((All.HubbleParam != 1.0) && (All.ComovingIntegrationOn == 0))
    {
        coutr(important) << "The SIDM module assumes for non-comoving integration that the units are not per h. You "
                            "may want to set HubbleParam = 1."
                         << endl;
    }

    // Additional checks not diectly related to SIDM
    beyond_scatter_checks();

    std::cout.clear();

    TIMER_POP(1, "mSIDM");
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Method to print relevant papers for the compiled version.
 */
void scatter::MSIDM::report_credits() const
{
    TIMER_PUSH(1, "mSIDM");

    //========== Give credits ============
    VERBOSE(0, "##  SIDM: The SIDM module is activated (refer to Fischer et al. 2026, arXiv, "
               "https://ui.adsabs.harvard.edu/abs/2026arXiv260310107F/abstract)");

    VERBOSE(0, "##  SIDM: fSIDM or rSIDM is activated (refer to Fischer et al. 2021, MNRAS, 505, 851, "
               "https://ui.adsabs.harvard.edu/abs/2021MNRAS.505..851F/abstract)");
#ifdef mSIDM_VDEP
    VERBOSE(0, "##  SIDM: velocity dependent SIDM is on (refer to Fischer et al. 2024, MNARS, 529, 2327, "
               "https://ui.adsabs.harvard.edu/abs/2024MNRAS.529.2327F/abstract)");
#endif // mSIDM_VDEP
#ifdef mSIDM_TIMESTEP
#if defined(mSIDM_VDEP) || defined(mSIDM_DIFFERENTIAL_CROSS_SECTION)
    VERBOSE(0, "##  SIDM: SIDM timestep for velocity-dependent scattering is on (refer to Fischer et al. 2024, MNRAS, "
               "529, 2327, https://ui.adsabs.harvard.edu/abs/2024MNRAS.529.2327F/abstract)");
#else // mSIDM_VDEP || mSIDM_DIFFERENTIAL_CROSS_SECTION
    VERBOSE(0, "##  SIDM: SIDM timestep for velocity-independent scattering is on (refer to Fischer et al. 2022, "
               "MNRAS, 510, 4080, https://ui.adsabs.harvard.edu/abs/2022MNRAS.510.4080F/abstract)");
    VERBOSE(
        0,
        "##         Note that the current implementation slightly deviates from the paper above, see the code "
        "release paper (Fischer et al. 2026, arXiv, https://ui.adsabs.harvard.edu/abs/2026arXiv260310107F/abstract)");
#endif // mSIDM_VDEP || mSIDM_DIFFERENTIAL_CROSS_SECTION
#endif // mSIDM_TIMESTEP
    if (All.ComovingIntegrationOn == 1)
    {
        VERBOSE(0, "##  SIDM: comoving integration is on (refer to Fischer et al. 2022, MNRAS, 516, 1923, "
                   "https://ui.adsabs.harvard.edu/abs/2022MNRAS.516.1923F/abstract)");
    }
#ifdef mSIDM_SSHI
    VERBOSE(0, "##  SIDM: scattering-induced subhalo–halo interaction (SSHI) is on (Klemmer et al. 2026, "
               "https://ui.adsabs.harvard.edu/abs/2026arXiv260319362K/abstract).");
    VERBOSE(0, "##  SIDM: For SSHI, please also refer to Garrison-Kimmel et al. 2013 "
               "(https://ui.adsabs.harvard.edu/abs/2013MNRAS.433.3539G/abstract) "
               "as the implementation of the Eddington Inversion is based on SpherIC "
               "(https://bitbucket.org/mtryan83/spheric-public/src/public/).");
    VERBOSE(0, "## SIDM: Our SSHI method is inspired by the work of Zeng et al. 2022 "
               "(https://ui.adsabs.harvard.edu/abs/2022MNRAS.513.4845Z/abstract), but differs in some details.");
#endif // mSIDM_SSHI
#ifdef mSIDM_DISSIPATION
    VERBOSE(0, "##  SIDM: dissipation is on (Schmidt et al. 2026, "
               "https://ui.adsabs.harvard.edu/abs/2026arXiv260619428S/abstract).");
#endif // mSIDM_DISSIPATION
#ifdef rSIDM_DISSIPATION
    VERBOSE(0, "##  SIDM: note rSIDM dissipation is implemented the same was as by Shen et al. 2021 "
               "(https://ui.adsabs.harvard.edu/abs/2021MNRAS.506.4421S/abstract).");
#endif // rSIDM_DISSIPATION
#ifdef mSIDM_NEWPARALLEL
    VERBOSE(0, "##  SIDM: new parallelisation is on (refer to Fischer et al. 2024, MNARS, 529, 2327, "
               "https://ui.adsabs.harvard.edu/abs/2024MNRAS.529.2327F/abstract)");
#endif // mSIDM_NEWPARALLEL
#ifdef mSIDM_OPENMP
    VERBOSE(0, "##  SIDM: SIDM OpenMP parallelisation is on (refer to Fischer et al. 2026, arXiv, "
               "https://ui.adsabs.harvard.edu/abs/2026arXiv260310107F/abstract).");
#endif // mSIDM_OPENMP
#ifdef mSIDM_DIFFERENTIAL_CROSS_SECTION
    VERBOSE(0, "##  SIDM: the module for differential cross-sections has been mainly developed by Marc Wiertel "
               "(refer to Fischer et al. 2026, arXiv, https://ui.adsabs.harvard.edu/abs/2026arXiv260310107F/abstract)");
#if defined(rSIDM) && defined(fSIDM)
    VERBOSE(0, "##  SIDM: combination of rSIDM and fSIDM is on (refer to Arido et al. 2025, A&A, 694, A297, "
               "https://ui.adsabs.harvard.edu/abs/2025A%26A...694A.297A/abstract)");
#endif // rSIDM && fSIDM
#endif // mSIDM_DIFFERENTIAL_CROSS_SECTION
#ifdef mSIDM_UNEQUALMASS
    VERBOSE(0, "##  SIDM: unequal-mass scattering is on (refer to Patil & Fischer 2025, "
               "https://ui.adsabs.harvard.edu/abs/2025arXiv250606272P/abstract)");
#endif // mSIDM_UNEQUALMASS
    TIMER_POP(1, "mSIDM");
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Method to compute scatterings between host and satellite particles.
 */
void scatter::MSIDM::subhalo_halo_interaction()
{
#ifdef mSIDM_SSHI
    const int &Nngb = All.DesNumNgb_pt1;

    const double a = lambda_0 * static_cast<double>(Nngb);

#ifdef mSIDM_OPENMP
#pragma omp parallel for
    for (int il = 0; il < NActivePart; ++il)
    {
        unsigned int n = ActiveParticleList[il];
#else // mSIDM_OPENMP
    for (int n = FirstActiveParticle; n >= 0; n = NextActiveParticle[n])
    {
#endif // mSIDM_OPENMP
        if (P[n].Type == type_dm)
        {
            double radius = 0;
            for (int loopc = 0; loopc < 3; ++loopc)
            {
                radius += P[n].Pos[loopc] * P[n].Pos[loopc];
            }
            radius = sqrt(radius);

            const double hsml_old = P[n].Hsml;
            const double b = a * P[n].Mass / analytic_potentials.rho(radius);
            const double new_hsml = pow(b, 1.0 / 3.0);

            P[n].Hsml = new_hsml;

            for (int k = 0; k < Nngb; ++k)
            {
                msidmdata_in vi(n); // create virtual particle by copying particle n

                analytic_potentials.get_vel(vi.Vel, radius); // sample velocity of the virtual particle

                msidmdata_in_ref pi(vi);

                pxp_physics(pi, n); // do the scattering
            }

            P[n].Hsml = hsml_old;
        }
    }
#endif // mSIDM_SSHI
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Constructor of the class to determine vmax at the beginning of the simulation.
 */
scatter::MSIDM_vmax::MSIDM_vmax(pxp_scattering::Scattering_msidm &scatter_in)
    : msidm_base(Module_mSIDM, NeighbourSearchCriteria_mSIDM), scatter(scatter_in)
{
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief This method computes the relative velocity of the particles and take the maximum.
 */
void scatter::MSIDM_vmax::pxp_physics(msidmdata_in_ref &pi, const int &place)
{
    msidmdata_in_ref pj(place);

    scatter.init_vmax(pi, pj);
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Constructor of a data class for MPI communication.
 */
scatter::msidmdata_in::msidmdata_in()
{
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Constructor of a data class for MPI communication.
 */
scatter::msidmdata_in::msidmdata_in(const int place)
{
    particle2in(place);
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Method to copy particle data to class.
 */
void scatter::msidmdata_in::particle2in(const int place)
{
    const particle_data &p_in = P[place];
    for (int k = 0; k < 3; ++k)
    {
        Pos[k] = p_in.Pos[k];
        Vel[k] = p_in.Vel[k];
    }
#ifdef TYPE_FOR_DATASTRUCT
    Type = p_in.Type;
#endif // TYPE_FOR_DATASTRUCT
#ifdef ADAPTGRAVSOFT
    Hsml = p_in.AGS_Hsml;
#else // ADAPTGRAVSOFT
    Hsml = p_in.Hsml;
#endif // ADAPTGRAVSOFT
    Mass = p_in.Mass;
    TimeBin = p_in.TimeBin;
    Ti_begstep = p_in.Ti_begstep;

#ifdef mSIDM_VMAX
    v_max = p_in.mSIDM_v_max;
#endif // mSIDM_VMAX

#ifdef mSIDM_HMIN
    h_min = p_in.mSIDM_h_min;
#endif // mSIDM_HMIN

#if defined(mSIDM_DOACTIVEONCE) || defined(mSIDM_DEBUG) || defined(ANGULAR_DEFLECTION_TEST_ON_FIXED_TARGET)
    ID = p_in.ID;
#endif // mSIDM_DOACTIVEONCE || mSIDM_DEBUG || ANGULAR_DEFLECTION_TEST_ON_FIXED_TARGET

#ifdef rSIDM_SCATTERCOUNT_INDI
    scatterCount = p_in.scatterCount;
#endif // rSIDM_SCATTERCOUNT_INDI
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Constructor of a data class for MPI communication.
 */
scatter::msidmdata_out::msidmdata_out()
{
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Constructor of a data class for MPI communication.
 */
scatter::msidmdata_out::msidmdata_out(const msidmdata_in &pi_in)
{
    in2out(pi_in);
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Method to copy data between data types for communication.
 */
void scatter::msidmdata_out::in2out(const msidmdata_in &pi_in)
{
    for (int k = 0; k < 3; ++k)
    {
        Vel[k] = pi_in.Vel[k];
    }
#ifdef mSIDM_VMAX
    v_max = pi_in.v_max;
#endif // mSIDM_VMAX

#ifdef mSIDM_HMIN
    h_min = pi_in.h_min;
#endif // mSIDM_HMIN

#ifdef rSIDM_SCATTERCOUNT_INDI
    scatterCount = pi_in.scatterCount;
#endif // rSIDM_SCATTERCOUNT_INDI
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Method to copy data from class to global particle data.
 */
void scatter::msidmdata_out::out2particle(const int place) const
{
    particle_data &p_out = P[place];
    for (int k = 0; k < 3; ++k)
    {
        p_out.Vel[k] = Vel[k];
    }

#ifdef mSIDM_VMAX
    p_out.mSIDM_v_max = std::max(p_out.mSIDM_v_max, v_max);
#endif // mSIDM_VMAX

#ifdef mSIDM_HMIN
    p_out.mSIDM_h_min = std::min(p_out.mSIDM_h_min, h_min);
#endif // mSIDM_HMIN

#ifdef rSIDM_SCATTERCOUNT_INDI
    p_out.scatterCount = scatterCount;
#endif // rSIDM_SCATTERCOUNT_INDI
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Constructor of reference class for msidmdata_in.
 */
scatter::msidmdata_in_ref::msidmdata_in_ref(class msidmdata_in &p_in)
    : scattering_particle_ref(p_in.Pos, p_in.Vel, p_in.Hsml, p_in.Mass, p_in.TimeBin, p_in.Ti_begstep
#ifdef SCATTERING_VMAX
                              ,
                              p_in.v_max
#endif // SCATTERING_VMAX
#ifdef SCATTERING_HMIN
                              ,
                              p_in.h_min
#endif // SCATTERING_HMIN
#ifdef SCATTERING_ID
                              ,
                              p_in.ID
#endif // SCATTERING_ID
#ifdef TYPE_FOR_DATASTRUCT
                              ,
                              p_in.Type
#endif // TYPE_FOR_DATASTRUCT
#ifdef SCATTERING_COUNT_INDI
                              ,
                              p_in.scatterCount
#endif // SCATTERING_COUNT_INDI
      )
{
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Constructor of reference class for msidmdata_in.
 */
scatter::msidmdata_in_ref::msidmdata_in_ref(const int place)
    : scattering_particle_ref(P[place].Pos, P[place].Vel
#ifdef ADAPTGRAVSOFT
                              ,
                              P[place].AGS_Hsml
#else // ADAPTGRAVSOFT
                              ,
                              P[place].Hsml
#endif // ADAPTGRAVSOFT
                              ,
                              P[place].Mass, P[place].TimeBin, P[place].Ti_begstep
#ifdef SCATTERING_VMAX
                              ,
                              P[place].mSIDM_v_max
#endif // SCATTERING_VMAX
#ifdef SCATTERING_HMIN
                              ,
                              P[place].mSIDM_h_min
#endif // SCATTERING_HMIN
#ifdef SCATTERING_ID
                              ,
                              P[place].ID
#endif // SCATTERING_ID
#ifdef TYPE_FOR_DATASTRUCT
                              ,
                              P[place].Type
#endif // TYPE_FOR_DATASTRUCT
#ifdef SCATTERING_COUNT_INDI
                              ,
                              P[place].scatterCount
#endif // SCATTERING_COUNT_INDI
      )
{
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Constructor of reference class for msidmdata_in.
 */
scatter::msidmdata_in_ref::msidmdata_in_ref(particle_data &p_in)
    : scattering_particle_ref(p_in.Pos, p_in.Vel
#ifdef ADAPTGRAVSOFT
                              ,
                              p_in.AGS_Hsml
#else // ADAPTGRAVSOFT
                              ,
                              p_in.Hsml
#endif // ADAPTGRAVSOFT
                              ,
                              p_in.Mass, p_in.TimeBin, p_in.Ti_begstep
#ifdef SCATTERING_VMAX
                              ,
                              p_in.mSIDM_v_max
#endif // SCATTERING_VMAX
#ifdef SCATTERING_HMIN
                              ,
                              p_in.mSIDM_h_min
#endif // SCATTERING_HMIN
#ifdef SCATTERING_ID
                              ,
                              p_in.ID
#endif // SCATTERING_ID
#ifdef TYPE_FOR_DATASTRUCT
                              ,
                              p_in.Type
#endif // TYPE_FOR_DATASTRUCT
#ifdef SCATTERING_COUNT_INDI
                              ,
                              p_in.scatterCount
#endif // SCATTERING_COUNT_INDI
      )
{
}

//////////////////////////////////////////////////////////////////////////////////////////////////
