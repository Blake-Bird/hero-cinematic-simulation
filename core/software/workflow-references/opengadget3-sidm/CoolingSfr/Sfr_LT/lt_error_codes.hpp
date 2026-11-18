/**
 * @file
 *
 * This file is part of the N-body/SPH code OpenGadget3.
 *
 * @brief
 *
 * @author OpenGadget3-development team, in particular Luca Tornatore.
 *
 * @copyright Copyright (c) 2026, OpenGadget3-development team lead by Klaus Dolag and Stefano Borgani
 * (see change logs). All rights reserved.
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License v3 or later.
 * See <https://www.gnu.org/licenses/>.
 */

#define LT_ERR_SF_FILE 10001
/* error in opening the IMF file.
 * files: lt_utils.c
 */

#define LT_ERR_SF 10005
/* the number of IMFs in IMF file does not match with the one defined at compile-time
 * files: lt_utils.c
 */

#define LT_ERR_SF_FILE_FORMAT 10002
/* error in the imf file
 * files: lt_utils.c
 */

#define LT_ERR_SF_ALLOCATE 10003
/* error when allocating memory for dfs
 * files: lt_utils.c
 */

#define LT_ERR_IMF_ALLOCATE 10004
/* error when allocating memory for imfs
 * files: lt_utils.c
 */

#define LT_ERR_IMF_NOTPOWERLAW 10005
/* error when reading imf in case it is not a power_law
 * since the support for this case is incomplete
 * files: lt_utils.c
 */

#define LT_ERR_LIFET_ALLOCATE 10006
/* error when allocating memory for lifetime boundaries
 * files: lt_sn.c
 */

#define LT_ERR_MISMATCH_ShL_TH_BHranges 10104
/* the defined mass threshold for a star to be considered
 * short living and releasing its energy promptly falls
 * in a mass range where the fate of stars is to go in BH.
 * files: lt_sn.c
 */

#define LT_ERROR_INTEGRATION_ERROR 10900

#define LT_ERR_NOT_ENOUGH_PARAM_FOR_SF 19000

#define LT_ERR_ABUNDANCES_NUM_MISMATCH 90010
