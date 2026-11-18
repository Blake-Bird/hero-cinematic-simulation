/**
 * @file
 *
 * This file is part of the N-body/SPH code OpenGadget3.
 *
 * @brief
 *
 * @author OpenGadget3-development team, in particular Cinthia Ragone Figueroa and Gian Luigi Granato.
 *
 * @copyright Copyright (c) 2026, OpenGadget3-development team lead by Klaus Dolag and Stefano Borgani
 * (see change logs). All rights reserved.
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License v3 or later.
 * See <https://www.gnu.org/licenses/>.
 */

#ifdef GL_CR_DUST
// present list from metals.dat
// 0  1 2  3 4 5  6  7 8  9  10 11 12 13 14
// He C Ca O N Ne Mg S Si Fe Na Al Ar Ni Ej
// they aree  16 elements. the last is for H.
// XXXXX NOTE that if GL_GIVENGRAIN is defined, only the C elements of the arrays dcondeffX are used.
// while for silicate see effsil_??? below
extern double dcondeffAGB[LT_NMet];
extern double dcondeffIa[LT_NMet];
extern double dcondeffII[LT_NMet];

// XXXXXX NOTE that if GL_GIVENGRAIN is defined, only the C elements of the arrays dcondeffX are used.
// other auxiliary quantities
extern int ind_C, ind_O; // indices in dcondeff*
// here "olivine" version Mg, Si, Fe
#define n_sil 3 // n_sil is the number of elements other than O entering in assumed silicate grain
extern int ind_sil[n_sil]; // indeces in dcondeff* of elements other than O entering in assumed silicate grain
extern double mu_sil[n_sil]; // atomic weights
#ifdef GL_GIVENGRAIN
extern int n_ato_sil[n_sil], n_ato_O; // number of atoms in assumed silicate c. e.g. 1 1 1 4 for O in standard olivine
extern double nummol_AGB, nummol_II,
    nummol_Ia; // number of "molecules" of assumed chemical compound forming silicate grains
extern double effsil_AGB; // these effsil_* are set equal to the condensation efficiencies for silicium in dust.dat
extern double effsil_Ia; // see read_dust
extern double effsil_II;
#endif
extern double mu_O, mu_C; // mus are atomic weights
extern double dcondeffAGB_var[LT_NMet], dcondeffIa_var[LT_NMet], dcondeffII_var[LT_NMet];
#endif
