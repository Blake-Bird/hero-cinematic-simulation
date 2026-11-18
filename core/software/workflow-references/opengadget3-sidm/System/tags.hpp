/**
 * @file
 *
 * This file is part of the N-body/SPH code OpenGadget3.
 *
 * @brief
 *
 * @author OpenGadget3-development team, in particular Klaus Dolag and Volker Springel.
 *
 * @copyright Copyright (c) 2026, OpenGadget3-development team lead by Klaus Dolag and Stefano Borgani
 * (see change logs). All rights reserved.
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License v3 or later.
 * See <https://www.gnu.org/licenses/>.
 */

#ifndef TAGS_HPP
#define TAGS_HPP

#define TAG_N 10 /*!< Various tags used for labelling MPI messages */
#define TAG_HEADER 11
#define TAG_PDATA 12
#define TAG_SPHDATA 13
#define TAG_KEY 14
#define TAG_DMOM 15
#define TAG_NODELEN 16
#define TAG_HMAX 17
#define TAG_GRAV_A 18
#define TAG_GRAV_B 19
#define TAG_DIRECT_A 20
#define TAG_DIRECT_B 21
#define TAG_HYDRO_A 22
#define TAG_HYDRO_B 23
#define TAG_NFORTHISTASK 24
#define TAG_PERIODIC_A 25
#define TAG_PERIODIC_B 26
#define TAG_PERIODIC_C 27
#define TAG_PERIODIC_D 28
#define TAG_NONPERIOD_A 29
#define TAG_NONPERIOD_B 30
#define TAG_NONPERIOD_C 31
#define TAG_NONPERIOD_D 32
#define TAG_POTENTIAL_A 33
#define TAG_POTENTIAL_B 34
#define TAG_DENS_A 35
#define TAG_DENS_B 36
#define TAG_LOCALN 37
#define TAG_BH_A 38
#define TAG_BH_B 39
#define TAG_SMOOTH_A 40
#define TAG_SMOOTH_B 41
#define TAG_ENRICH_A 42
#define TAG_CONDUCT_A 43
#define TAG_CONDUCT_B 44
#define TAG_FOF_A 45
#define TAG_FOF_B 46
#define TAG_FOF_C 47
#define TAG_FOF_D 48
#define TAG_FOF_E 49
#define TAG_FOF_F 50
#define TAG_FOF_G 51
#define TAG_HOTNGB_A 52
#define TAG_HOTNGB_B 53
#define TAG_SWAP 54
#define TAG_PM_FOLD 55
#define TAG_FOF_M 56
#define TAG_FOF_VX 1057
#define TAG_FOF_VY 1058
#define TAG_FOF_VZ 1059

#define TAG_SUBFIND_IO 57

#ifdef BG_SFR
#define TAG_STARDATA 60
#endif // BG_SFR

#define TAG_PDATA_SPH 70
#define TAG_KEY_SPH 71

#ifdef ADAPTGRAVSOFT
#define TAG_AGS_DENS_A 74
#define TAG_AGS_DENS_B 75
#endif // ADAPTGRAVSOFT

#ifdef WRITE_KEY_FILES
#define TAG_WRT_OFF 80
#endif // WRITE_KEY_FILES

#ifdef SFR
#define TAG_PDATA_STARS 100
#define TAG_KEY_STARS (TAG_PDATA_STARS + 1)
#define TAG_METDATA (TAG_KEY_STARS + 1)
#define TAG_SE (TAG_METDATA + 1)
#define TAG_TRCK (TAG_SE + 1)
#define TAG_CCRIT (TAG_TRCK + 1)
#endif // SFR

#ifdef BLACK_HOLES
#define TAG_PDATA_BHS 120
#define TAG_KEY_BHS 121
#define TAG_BHDATA 122
#endif // BLACK_HOLES

#ifdef KD_ALTERNATIVE_GROUP_SORT
#define TAG_ALTSORT_GRP 130
#define TAG_ALTSORT_SUB 131
#define TAG_ALTSORT_IDS 132
#endif // KD_ALTERNATIVE_GROUP_SORT


#ifdef AXION_DM
#define TAG_AX_DENS_A 150
#define TAG_AX_DENS_B 151
#define TAG_AX_HYDRO_A 152
#define TAG_AX_HYDRO_B 153
#define TAG_PDATA_AX 154
#define TAG_KEY_AX 155
#define TAG_AXDATA 156
#endif // AXION_DM

#if GADGET_HYDRO == HYDRO_MFM
#define TAG_GRAD_A 160
#define TAG_GRAD_B 161
#define TAG_FLUX_A 162
#define TAG_FLUX_B 163
#define TAG_SLOPE_A 164
#define TAG_SLOPE_B 165
#endif // GADGET_HYDRO == HYDRO_MFM

#ifdef PXP
#define TAG_PXP_A 166
#define TAG_PXP_B 167
#endif // PXP


#endif // TAGS_HPP
