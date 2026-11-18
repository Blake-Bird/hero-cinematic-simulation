/**
 * @file
 *
 * This file is part of the N-body/SPH code OpenGadget3.
 *
 * @brief Classes for the computation of different analytic potentials and sampling random velocities from an NFW halo
 * using the Eddington Inversion.
 *
 * @author OpenGadget3-development team, in particular Moritz S. Fischer and David Klemmer.
 *
 * @copyright Copyright (c) 2026, OpenGadget3-development team lead by Klaus Dolag and Stefano Borgani
 * (see change logs). All rights reserved.
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License v3 or later.
 * See <https://www.gnu.org/licenses/>.
 */

#ifndef ANALYTIC_POTENTIALS_HPP
#define ANALYTIC_POTENTIALS_HPP

#include "../CodeBase/precision.hpp"
#include <omp.h>
#include <random>

#ifdef mSIDM_SSHI
#ifndef ANALYTIC_POTENTIALS_VELOCITIES
#define ANALYTIC_POTENTIALS_VELOCITIES
#endif // ANALYTIC_POTENTIALS_VELOCITIES
#endif // mSIDM_SSHI

#ifdef ANALYTIC_POTENTIALS_VELOCITIES
/**
 * @brief Class to sample random velocities from an NFW halo using the Eddington Inversion.
 *
 * This class samples 3d velocities as a function of radius for an NFW halo. It was build based
 * on the public code SpherIC (https://bitbucket.org/mtryan83/spheric-public/src/public/).
 */
class EddingtonInversion
{
  public:
    EddingtonInversion(const int NGRIDR_in, const MyAtLeastDouble &rho, const MyAtLeastDouble &r);
    ~EddingtonInversion();

    /**
     * @brief This method initializes all relevant things for the Eddington inversion.
     */
    void init();

    /**
     * @brief Public method to sample the velocity vector.
     *
     * @param Vel array to return the sampled velocity vector.
     * @param r radius at which the velocity vector is sampled.
     */
    void get_vel(MyFloat (&Vel)[3], const MyAtLeastDouble r);

    /**
     * @brief Method that computes the density.
     *
     * @param r radius at which the density is evaluated.
     */
    MyAtLeastDouble rho(const MyAtLeastDouble r) const;

  private:
    const double &halo_rho_s, &halo_r_s;
    double rinner;

    double *Gridr;
    double *logr;
    double *logPot;
    double *fE;

    const int NGRIDR;
    const double FACTORINNER;
    const double FACTOROUTER;
    const double TOLLININT;

    /**
     * @brief Computes grid point and prints them.
     */
    void grid();

    /**
     * @brief Computes potential at grid points.
     */
    void pot();

    /**
     * @brief Computes distribution function for grid points.
     */
    void fe();

    /**
     * @brief Finds grid point.
     */
    int locate(int n, const double grid[], const double x) const;

    /**
     * @brief Computes linear interpolation.
     */
    double lininterpolation(int n, const double gridx[], const double gridy[], const double x) const;

    /**
     * @brief Method that evaluates the potential.
     *
     * @param r radius at which the potential is computed.
     */
    double Potential(const double r) const;

    /**
     * @brief Method for trapez computation used in integration.
     */
    double trapez(const double k, const int n) const;

    /**
     * @brief Method for integration.
     */
    double integral(const double j) const;

    /**
     * @brief Computes distribution function.
     */
    double E_distribution(const double r);

    /**
     * @brief Method for normalised density profile.
     *
     * @params r radius at which the profile is evaluated.
     */
    double tau(const double r) const;

    /**
     * @brief Computes eta.
     */
    double eta(const double r) const;

    /**
     * @brief Computes derivative of eta.
     */
    double detadr(const double r) const;

    /**
     * @brief Computes derivative of density.
     */
    double drhodr(const double r) const;

    /**
     * @brief Computes second-order derviative of density.
     */
    double d2rhodr2(const double r) const;

    /**
     * @brief Computes second-order derivative of density with respect to the potential.
     */
    double d2rhodPhi2(const double r) const;

    /**
     * @brief Computes enclosed mass.
     */
    double NFW_M(const double r) const;

    /**
     * @brief Computes escape velocity.
     */
    double vescape(const double r) const;

    /*
     * @brief Method to draw random numbers.
     */
    double rand01();

    /**
     * @brief Interpolates distribution function at specific energy.
     */
    double f1(const double E) const;

    /**
     * @brief Interpolates distribution function at specific radius.
     */
    double f2(const double r) const;

    unsigned int seed;

    std::mt19937 generator;

    std::normal_distribution<MyFloat> random_vel;

    omp_lock_t lock_generator; // OMP lock for the random number generator
};
#endif // ANALYTIC_POTENTIALS_VELOCITIES

/**
 * @brief Class containing functions for the computation of different analytic potentials.
 *
 * This class calculates and applies accelerations due to various components of an analytic potential.
 */
extern class ANALYTIC_POTENTIAL final
{
  public:
    ANALYTIC_POTENTIAL();
    ~ANALYTIC_POTENTIAL();

    /**
     * @brief This method is called in begrun() and initializes all relevant things for the analytic potentials.
     */
    void init();

    /**
     * @brief Applies the acceleration due to the analytic profiles.
     */
    void apply_accelerations() const;

#ifdef ANALYTIC_POTENTIALS_VELOCITIES
    /**
     * @brief Sampels velocity vectors from an NFW profile.
     *
     * @param Vel array to return the sampled velocity vector.
     * @param r radius at which the velocity vector is sampled.
     */
    void get_vel(MyFloat (&Vel)[3], const MyAtLeastDouble r);

    /**
     * @brief Computed the density of an NFW halo.
     *
     * @param r radius at which the density is computed.
     */
    MyAtLeastDouble rho(const MyAtLeastDouble r) const;
#endif // ANALYTIC_POTENTIALS_VELOCITIES

  private:
    const int &do_halo, &do_bulge, &do_thick_disk, &do_thin_disk, &do_disk1, &do_disk2, &do_king;
    const double &halo_rho_s, &halo_r_s;
    const double &bulge_M_h, &bulge_a_h;
    const double &thick_disk_M_d, &thick_disk_a_d, &thick_disk_b_d;
    const double &thin_disk_M_d, &thin_disk_a_d, &thin_disk_b_d;
    const double &disk1_M_d, &disk1_a_d, &disk1_b_d;
    const double &disk2_M_d, &disk2_a_d, &disk2_b_d;
    const double &king_rho_0, &king_r_c;

#ifdef ANALYTIC_POTENTIALS_VELOCITIES
    EddingtonInversion SampleVelocity;
#endif // ANALYTIC_POTENTIALS_VELOCITIES

    /**
     * @brief Applies the acceleration due to an NFW profile.
     *
     * @param rho_s Scale density of the NFW profile.
     * @param r_s Scale radius of the NFW profile.
     */
    void apply_nfw(const MyAtLeastDouble rho_s, const MyAtLeastDouble r_s) const;

    /**
     * @brief Applies the acceleration due to a Hernquist profile.
     *
     * @param M_h Total mass of the Hernquist profile.
     * @param a_h Scale radius of the Hernquist profile.
     */
    void apply_hern(const MyAtLeastDouble M_h, const MyAtLeastDouble a_h) const;

    /**
     * @brief Applies the acceleration due to a disk component.
     *
     * @param M_d Total mass of the disk.
     * @param a_d Scale radius along the disk's major axis.
     * @param b_d Scale radius along the disk's minor axis.
     */
    void apply_disk(const MyAtLeastDouble M_d, const MyAtLeastDouble a_d, const MyAtLeastDouble b_d) const;

    /**
     * @brief Applies the acceleration due to a King profile.
     *
     * @param rho_0 Density parameter of the King model.
     * @param r_c Scale radius of the King model.
     */
    void apply_king(const MyAtLeastDouble rho_0, const MyAtLeastDouble r_c) const;
} analytic_potentials;

#endif // ANALYTIC_POTENTIALS_HPP
