/**
 * @file
 *
 * This file is part of the N-body/SPH code OpenGadget3.
 *
 * @brief This header file defines physical constants used in the code. The constants are required in CGS
 * (Centimeter-Gram-Second) units.
 *
 * @author OpenGadget3-development team, in particular Klaus Dolag and Volker Springel.
 *
 * @copyright Copyright (c) 2026, OpenGadget3-development team lead by Klaus Dolag and Stefano Borgani
 * (see change logs). All rights reserved.
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License v3 or later.
 * See <https://www.gnu.org/licenses/>.
 */

#ifndef CONSTANTS_HPP
#define CONSTANTS_HPP

namespace constants
{
constexpr MyAtLeastDouble SEC_PER_MEGAYEAR = 3.15576e13;
constexpr MyAtLeastDouble SEC_PER_YEAR = SEC_PER_MEGAYEAR / 1e6; /*seconds per Julian year*/

constexpr MyAtLeastDouble OSCILLATOR_STRENGTH = 0.41615;
constexpr MyAtLeastDouble OSCILLATOR_STRENGTH_HeII = 0.41615;
constexpr MyAtLeastDouble HUBBLE = 3.2407793e-18; /* in h/sec */
constexpr MyAtLeastDouble LYMAN_ALPHA = 1215.6e-8; /* 1215.6 Angstroem */
constexpr MyAtLeastDouble LYMAN_ALPHA_HeII = 303.8e-8; /* 303.8 Angstroem */

constexpr MyAtLeastDouble DIVV_PRECISION_COND =
    0.05; /* used when comparing |v|/h with div(v) for time dependent art conduction*/
constexpr MyAtLeastDouble DIVV_PRECISION_VISC =
    0.2; /* used when comparing |v|/h with div(v) for time dependent art viscosity*/
#ifdef FIRST_ORDER_GRADIENTS
constexpr MyAtLeastDouble BALSARA_FAC_FOR_CSND_OVER_H = 0.0001; /* used when adding csnd/h for balsara switch f */
#else // FIRST_ORDER_GRADIENTS
constexpr MyAtLeastDouble BALSARA_FAC_FOR_CSND_OVER_H = 0.01; /* used when adding csnd/h for balsara switch f */
#endif // FIRST_ORDER_GRADIENTS

#ifndef OLD_PHYSICAL_CONSTANTS

/* ... UPDATED physical constants (in cgs units) from https://physics.nist.gov/cuu/Constants/Table/allascii.txt and
 * https://pdg.lbl.gov/2024/reviews/rpp2024-rev-astrophysical-constants.pdf*/
constexpr MyAtLeastDouble GRAVITY = 6.67430e-8;
constexpr MyAtLeastDouble GRAVITY_HEL =
    1.32712440041279419e26; /* G*Msun in cm^3 s^-2 ("heliocentric gravitational constant" */
constexpr MyAtLeastDouble SOLAR_MASS = GRAVITY_HEL / GRAVITY;
/* defining the solar mass this way means that an internal computation of
   GRAVITY*SOLAR_MASS only has a relative uncertainty of 6e-11 */
constexpr MyAtLeastDouble SOLAR_LUM = 3.828e33; /*exact; definition of "nominal solar luminosity"*/
constexpr MyAtLeastDouble RAD_CONST = 7.56573325028e-15; /*exact, but rounded*/
constexpr MyAtLeastDouble AVOGADRO = 6.02214076e23; /*exact*/
constexpr MyAtLeastDouble BOLTZMANN = 1.380649e-16; /*exact*/
constexpr MyAtLeastDouble GAS_CONST = 8.31446261815324e7; /*exact, but rounded*/
constexpr MyAtLeastDouble C_LIGHT = 2.99792458e10; /*exact*/
constexpr MyAtLeastDouble PLANCK = 6.62607015e-27; /*exact*/
constexpr MyAtLeastDouble CM_PER_MPC = 3.085677567e24; /*exact; follows from exact definition of AU, but is rounded*/
constexpr MyAtLeastDouble CM_PER_KPC = CM_PER_MPC / 1e3;
constexpr MyAtLeastDouble CM_PER_PC = CM_PER_MPC / 1e6;
constexpr MyAtLeastDouble PROTONMASS = 1.67262192369e-24;
constexpr MyAtLeastDouble ELECTRONMASS = 9.1093837015e-28;
constexpr MyAtLeastDouble THOMSON = 6.6524587321e-25; /* Thomson cross-section in cm^2*/
constexpr MyAtLeastDouble ELECTRONCHARGE = 4.803204712570263e-10; /*exact*/
constexpr MyAtLeastDouble ELECTRONVOLT_IN_ERGS = 1.602176634e-12; /*exact*/
constexpr MyAtLeastDouble eV_to_K = 11604.5;
constexpr MyAtLeastDouble eV_to_Hz = 2.417989242e14;
constexpr MyAtLeastDouble tiny = 0.0;
constexpr MyAtLeastDouble ALPHA = 1.0;
constexpr MyAtLeastDouble ALPHA_FINE_STRUCT =
    7.2973525693e-3; /* fine structure constant (dimensionless), often approximated as alpha=1/137 */

#else // OLD_PHYSICAL_CONSTANTS

/* previously used physical constants */
constexpr MyAtLeastDouble gravity = 6.672e-8;
constexpr MyAtLeastDouble solar_mass = 1.989e33;
constexpr MyAtLeastDouble solar_lum = 3.826e33;
constexpr MyAtLeastDouble rad_const = 7.565e-15;
constexpr MyAtLeastDouble avogadro = 6.0222e23;
constexpr MyAtLeastDouble boltzmann = 1.38066e-16;
constexpr MyAtLeastDouble gas_const = 8.31425e7;
constexpr MyAtLeastDouble c_light = 2.9979e10;
constexpr MyAtLeastDouble planck = 6.6262e-27;
constexpr MyAtLeastDouble cm_per_mpc = 3.085678e24;
constexpr MyAtLeastDouble protonmass = 1.6726e-24;
constexpr MyAtLeastDouble electronmass = 9.10953e-28;
constexpr MyAtLeastDouble thomson = 6.65245e-25;
constexpr MyAtLeastDouble electroncharge = 4.8032e-10;
constexpr MyAtLeastDouble electronvolt_in_ergs = 1.60217733e-12;

#endif // OLD_PHYSICAL_CONSTANTS

#ifdef MAGNETIC
#ifdef MU0_UNITY
constexpr MyAtLeastDouble mu0 = 1.0;
constexpr MyAtLeastDouble mu0_inv = 1.0;
#else // MU0_UNITY
constexpr MyAtLeastDouble mu0 = 12.566370614; /* 4pi */
constexpr MyAtLeastDouble mu0_inv = 0.079577472; /* 1/4pi */
#endif // MU0_UNITY
#endif // MAGNETIC
} // namespace constants

#endif // CONSTANTS_HPP
