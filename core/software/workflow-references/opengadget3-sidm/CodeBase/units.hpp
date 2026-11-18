/**
 * @file
 *
 * This file is part of the N-body/SPH code OpenGadget3.
 *
 * @brief This file contains the declaration of the class used for comoving factors.
 *
 * @author OpenGadget3-development team, in particular Frederick Groth and Ludwig Böss.
 *
 * @copyright Copyright (c) 2026, OpenGadget3-development team lead by Klaus Dolag and Stefano Borgani
 * (see change logs). All rights reserved.
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License v3 or later.
 * See <https://www.gnu.org/licenses/>.
 */

#ifndef OPENGADGET3_COSMO_HPP
#define OPENGADGET3_COSMO_HPP

#include "../CodeBase/precision.hpp"
#include <cmath>

namespace OpenGadget3
{
/**
 * @brief Basic unit class to simplify unit conversions throughout the code.
 *
 * To avoid errors because of wrong conversions, this class should be used instead of individual definitions of the
 * conversion at each place.
 */
class Units
{
  public:
    // general time variables
    double atime, a2inv, a3inv, afac1, afac2, afac3, hubble_a;
    // units
    double UnitLength_in_cm, UnitVelocity_in_cm_per_s, UnitMass_in_g, UnitTime_in_s, UnitDensity_in_cgs,
        UnitPressure_in_cgs, UnitCoolingRate_in_cgs, UnitEnergy_in_cgs, UnitLength_remove_h0, UnitVelocity_remove_h0,
        UnitMass_remove_h0, UnitTime_remove_h0, UnitDensity_remove_h0, UnitPressure_remove_h0,
        UnitCoolingRate_remove_h0, UnitEnergy_remove_h0;
#ifdef MAGNETIC
    double UnitMagneticField_in_gauss, UnitMagneticField_remove_h0;
#endif // MAGNETIC
    // comoving-physical conversion factors
    MyFloat timestep_to_physical, length_to_physical, volume_to_comoving, mass_to_physical, density_to_physical,
        pressure_to_physical, velocity_to_physical, accel_to_physical, velocity_to_comoving;
#ifdef MAGNETIC
    MyFloat bfld_to_physical, bfld_to_comoving;
#ifdef DIVBCLEANING_DEDNER
    MyFloat cleaning_phi_to_physical, cleaning_phi_to_comoving;
#endif // DIVBCLEANING_DEDNER
#endif // MAGNETIC

    /**
     * @brief Constructor.
     */
    Units(){};

    /**
     * @brief initialize all conversion factors from (physical) code units to cgs.
     */
    void init_cgs_factors(double l_unit, double v_unit, double m_unit)
    {
        UnitLength_in_cm = l_unit;
        UnitVelocity_in_cm_per_s = v_unit;
        UnitMass_in_g = m_unit;
        UnitTime_in_s = UnitLength_in_cm / UnitVelocity_in_cm_per_s;
        UnitDensity_in_cgs = UnitMass_in_g / pow(UnitLength_in_cm, 3);
        UnitPressure_in_cgs = UnitMass_in_g / UnitLength_in_cm / pow(UnitTime_in_s, 2);
        UnitCoolingRate_in_cgs = UnitPressure_in_cgs / UnitTime_in_s;
        UnitEnergy_in_cgs = UnitMass_in_g * pow(UnitLength_in_cm, 2) / pow(UnitTime_in_s, 2);
#ifdef MAGNETIC
        UnitMagneticField_in_gauss = 1 / sqrt(UnitTime_in_s * UnitTime_in_s * UnitLength_in_cm / UnitMass_in_g);
#endif // MAGNETIC
    }

    /**
     * @brief initialize all conversion factors from units including hubble factors to units without it.
     */
    void init_h0_factors(double HubbleParam)
    {
        UnitLength_remove_h0 = 1 / HubbleParam;
        UnitVelocity_remove_h0 = 1;
        UnitMass_remove_h0 = 1 / HubbleParam;
        UnitTime_remove_h0 = UnitLength_remove_h0 / UnitVelocity_remove_h0;
        UnitDensity_remove_h0 = UnitMass_remove_h0 / pow(UnitLength_remove_h0, 3);
        UnitPressure_remove_h0 = UnitMass_remove_h0 / UnitLength_remove_h0 / pow(UnitTime_remove_h0, 2);
        UnitCoolingRate_remove_h0 = UnitPressure_remove_h0 / UnitTime_remove_h0;
        UnitEnergy_remove_h0 = UnitMass_remove_h0 * pow(UnitLength_remove_h0, 2) / pow(UnitTime_remove_h0, 2);
#ifdef MAGNETIC
        UnitMagneticField_remove_h0 =
            1 / sqrt(UnitTime_remove_h0 * UnitTime_remove_h0 * UnitLength_remove_h0 / UnitMass_remove_h0);
#endif // MAGNETIC
    }

    /**
     * @brief initialize the comoving factors used for conversion.
     *
     * This function should be called once per timestep to make sure the later conversion is done correctly.
     */
    virtual void init_cosmo_values(double time) = 0;
    /**
     * @brief initialize the conversion factors from comoving code units to physical code units.
     */
    virtual void set_comoving_physical_conversion_factors() = 0;
};

/**
 * @brief Unit class for the non comoving case to simplify unit conversions throughout the code.
 *
 * To avoid errors because of wrong conversions, this class should be used instead of individual definitions of the
 * conversion at each place.
 */
class UnitsNoncomoving final : public Units
{
  public:
    /**
     * @brief Constructor.
     */
    UnitsNoncomoving(){};

    /**
     * @brief initialize the comoving factors used for conversion.
     *
     * This function should be called once per timestep to make sure the later conversion is done correctly.
     * For the non comoving case, these are all set to one.
     */
    void init_cosmo_values(double time) override
    {
        atime = 1.0;
        a2inv = 1.0;
        a3inv = 1.0;
        afac1 = 1.0;
        afac2 = 1.0;
        afac3 = 1.0;
        hubble_a = 1.0;
        set_comoving_physical_conversion_factors();
    }
    /**
     * @brief initialize the conversion factors from comoving code units to physical code units.
     *
     * For the non comoving case, these are all set to one.
     */
    void set_comoving_physical_conversion_factors() override
    {
        timestep_to_physical = 1;
        length_to_physical = 1;
        volume_to_comoving = 1;
        mass_to_physical = 1;
        density_to_physical = 1;
        pressure_to_physical = 1;
        velocity_to_physical = 1;
        accel_to_physical = 1;
        velocity_to_comoving = 1;
#ifdef MAGNETIC
        bfld_to_physical = 1;
        bfld_to_comoving = 1;
#ifdef DIVBCLEANING_DEDNER
        cleaning_phi_to_physical = 1;
        cleaning_phi_to_comoving = 1;
#endif // DIVBCLEANING_DEDNER
#endif // MAGNETIC
    }
};

/**
 * @brief Unit class for the comoving case to simplify unit conversions throughout the code.
 *
 * To avoid errors because of wrong conversions, this class should be used instead of individual definitions of the
 * conversion at each place.
 */
class UnitsComoving final : public Units
{
  public:
    /**
     * @brief Constructor.
     */
    UnitsComoving(){};

    /**
     * @brief initialize the comoving factors used for conversion.
     *
     * This function should be called once per timestep to make sure the later conversion is done correctly.
     */
    void init_cosmo_values(double time) override;

    /**
     * @brief initialize the conversion factors from comoving code units to physical code units.
     */
    void set_comoving_physical_conversion_factors() override
    {
        /* Comoving timestep to physical timestep */
        timestep_to_physical = 1 / hubble_a;

        /* Comoving length to physical length */
        length_to_physical = atime;
        /* Physical volume to comoving volume */
        volume_to_comoving = a3inv;

        /* Comoving mass to physical mass (doesn't do anything) */
        mass_to_physical = 1;

        /* Comoving density to physical density */
        density_to_physical = a3inv;

        /* Comoving pressure to physical pressure */
        // ToDo: Account for different gammas!
        pressure_to_physical = a3inv / afac1;


        /* Comoving velocity to physical velocity */
        velocity_to_physical = 1 / atime;

        /* Comoving (gravitation) acceleration to physical acceleration */
        accel_to_physical = a2inv;

        /* Physical velocity to comoving velocity */
        velocity_to_comoving = atime;

#ifdef MAGNETIC
        bfld_to_physical = a2inv;
        bfld_to_comoving = atime * atime;
#ifdef DIVBCLEANING_DEDNER
        cleaning_phi_to_physical = a3inv;

        cleaning_phi_to_comoving = atime * atime * atime;
#endif // DIVBCLEANING_DEDNER
#endif // MAGNETIC
    }
};
} // namespace OpenGadget3

#endif // OPENGADGET3_COSMO_HPP
