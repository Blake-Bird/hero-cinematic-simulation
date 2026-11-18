
/**
 * @file
 *
 * This file is part of the N-body/SPH code OpenGadget3.
 *
 * @brief Header for auxiliary routines for the modeling of decoupling of gas phases
 *
 * @author OpenGadget3-development team, in particular J. Badía and C. Scannapieco
 *
 * @copyright Copyright (c) 2026, OpenGadget3-development team lead by Klaus Dolag and Stefano Borgani
 * (see change logs). All rights reserved.
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License v3 or later.
 * See <https://www.gnu.org/licenses/>.
 */

#ifdef CS_DECOUPLING
void cs_copy_densities(void);
void cs_find_low_density_tail(void);
int cs_compare_density_values(const void *a, const void *b);
#endif // CS_DECOUPLING
