/**
 * @file
 *
 * This file is part of the N-body/SPH code OpenGadget3.
 *
 * @brief This file contains the functions for the io.
 *
 * @author OpenGadget3-development team, in particular Klaus Dolag, Geray S. Karademir and Volker Springel.
 *
 * @copyright Copyright (c) 2026, OpenGadget3-development team lead by Klaus Dolag and Stefano Borgani
 * (see change logs). All rights reserved.
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License v3 or later.
 * See <https://www.gnu.org/licenses/>.
 */

#ifndef IO_HPP
#define IO_HPP

#include <cstdint>

enum iofields
{
    // reference: old order for first blocks:
    IO_POS = 0,
    IO_VEL = 1,
    IO_ID = 2,
    IO_MASS = 3,
    IO_SECONDORDERMASS = 4,
    IO_U = 5,
    IO_RHO = 6,
    IO_NE = 7,
    IO_NH = 8,
    IO_HSML = 9,
    IO_SFR = 10,
    IO_AGE = 11,
    IO_HSMS = 12,
    IO_ACRS = 13,
    IO_Z = 14,
    IO_BHMASS = 15,

    // basic particle properties
    IO_ACCEL = 100,
    IO_TSTP = 101,
    IO_POT = 102,
    IO_PHIDOT = 103,
    IO_DTENTR = 104,
    IO_DENN = 105,
    IO_MAXSIGNALVEL = 106,
    IO_TRUENGB = 107,
    IO_WNGB = 108,

    // conduction
    IO_ACVC = 200,
    IO_GRADU = 201,
    IO_GRADENERGY = 202,
    IO_GRADPRESSURE = 203,

    // viscosity
    IO_ABVC = 300,
    IO_ABVR = 301,
    IO_ABVH = 302,
    IO_STRESSDIAG = 303,
    IO_STRESSOFFDIAG = 304,
    IO_STRESSBULK = 305,
    IO_SHEARCOEFF = 306,
    IO_PRESSURE_ANISO = 307,
    IO_ENERGY_VISC = 308,

    // turbulence
    IO_VRMS = 400,
    IO_VBULK = 401,
    IO_VRAD = 402,
    IO_VTAN = 403,
    IO_VDIV = 404,
    IO_VROT = 405,
    IO_VORT = 406,
    IO_GRADIENT_ORDER = 407,
    IO_GRADIENT_DISTANCEMATRIX = 408,
    IO_GRADIENT_VELOCITYMATRIX = 409,
    IO_DPP = 410,

    // magnetic field
    IO_BFLD = 500,
    IO_DBDT = 501,
    IO_DIVB = 502,
    IO_AMDC = 503,
    IO_PHI = 504,
    IO_XPHI = 505,
    IO_GRADPSI = 506,
    IO_GRADPHI = 507,
    IO_ROTB = 508,
    IO_SROTB = 509,
    IO_BCRV = 510,
    IO_FACLO = 511,

    // chemistry
    IO_HII = 700,
    IO_HeI = 701,
    IO_HeII = 702,
    IO_HeIII = 703,
    IO_H2I = 704,
    IO_H2II = 705,
    IO_HM = 706,
    IO_HD = 707,
    IO_DI = 708,
    IO_DII = 709,
    IO_HeHII = 710,

    // cooling
    IO_COOLRATE = 800,
    IO_CONDRATE = 801,
    IO_COOLTIME = 802,

    // LT model
    IO_iMass = 900,
    IO_Zs = 901,
    IO_ContribII = 902,
    IO_ContribIa = 903,
    IO_ContribAGB = 904,
    IO_ZAGE = 905,
    IO_ZAGE_LLV = 906,
    IO_CLDX = 907,
    IO_HTEMP = 908,
    IO_TEMP = 909,
    IO_CONTRIB = 910,
    IO_ZSMOOTH = 911,
    IO_allZSMOOTH = 912,
    IO_DELAYTIME = 914,

    // muppi
    IO_FB_M_H = 1000,
    IO_FB_M_C = 1001,
    IO_FB_M_MO = 1002,
    IO_FB_E_H = 1003,
    IO_FB_M_SF = 1004,
    IO_FB_MF = 1005,
    IO_FB_NMF = 1006,
    IO_FB_EOUT = 1007,
    IO_FB_EREC = 1008,
    IO_FB_CLOCK = 1009,
    IO_FB_E_TOT_0 = 1010,
    IO_FB_TDYN = 1011,
    IO_FB_GRADRHO = 1012,
    IO_FB_TCOOL = 1013,
    IO_FB_EKINREC = 1014,
    IO_FB_TSTARTMP = 1015,

    /* GM if GM_STARDENSITY  is on, records stellar density, hsml and grav potential if EVALPOTENTIAL is also on */
    IO_RHOSTAR = 1100,
    IO_HSMLSTAR = 1101, /* used if GM_STARDENSITY_FIXAPERTURE is NOT defined */
    IO_NEIGHSTAR = 1102, /* used if GM_STARDENSITY_FIXAPERTURE IS defined */

    // dust
    IO_DUSTL = 1200, /* used for GL_CR_DUST */
    IO_DUSTS = 1201,
    IO_DM_AVE_VEL = 1202, /* used if GM_DMVELS and GM_DMVELS_OUTPUT are defined */
    IO_DM_AVE_SIG = 1203,
    IO_DM_NPART = 1204,
    IO_DVEL = 1205,
    IO_DFRC = 1206,
    IO_DRAG = 1207,

    // black holes
    IO_BHMDOT = 1300,
    IO_BHPROGS = 1301,
    IO_BHMBUB = 1302,
    IO_BHMINI = 1303,
    IO_BHMRAD = 1304,
    IO_BHSPIN = 1305,
    IO_BHETADISK = 1306,
    IO_ACRB = 1307,
    IO_BH_Kin_ConeAxis = 1308, /* for 'PB_BH_KIN_EJECT_PART_INSIDE_CONE' option */
    IO_DECOUPLE_TIME_AGN = 1309, /* for 'PB_BH_KIN_FEED_DECOUPLE_WIND' option */

    // shock finder
    IO_MACH = 1400,
    IO_SHSPEED = 1401,
    IO_SHCOMPRESS = 1402,
    IO_SHNORMAL = 1403,
    IO_SHRHOUP = 1404,
    IO_SHPRESUP = 1405,
    IO_SHPRESDOWN = 1406,
    IO_SHVELUP = 1407,
    IO_SHVELDOWN = 1408,
    IO_MACH_ALFVEN = 1409,
    IO_SHALFVENUP = 1410,
    IO_SHALFVENDOWN = 1411,
    IO_SHOBLIQUITY = 1412,
    IO_SHEDISS = 1413, /* SHOCKFINDER_MFM: per-cell energy dissipation rate */
    IO_SHCAND = 1414, /* SHOCKFINDER_MFM: candidate-cell flag */
    IO_SHPRIM = 1415, /* SHOCKFINDER_MFM: primary-cell flag */
    IO_SHSIGNALUP = 1416, /* upstream signal speed the Mach number is measured against */

    // spectral CRs
    IO_LMBCR_pNORM = 1500,
    IO_LMBCR_eNORM = 1501,
    IO_LMBCR_pSLOPE = 1502,
    IO_LMBCR_eSLOPE = 1503,
    IO_LMBCR_pCUT = 1504,
    IO_LMBCR_eCUT = 1505,
    IO_LMBCR_ePRESSURE = 1506,
    IO_LMBCR_pPRESSURE = 1507,
    IO_LMBCR_eSYNCHROTRON = 1508,
    IO_CRP_pmax = 1509,
    IO_RHO_OLD = 1510,

    // adaptive grav softening
    IO_AGS_SOFT = 1600,
    IO_AGS_DENS = 1601,
    IO_AGS_ZETA = 1602,
    IO_AGS_OMEGA = 1603,
    IO_AGS_CORR = 1604,
    IO_AGS_NGBS = 1605,

    // MFM
    IO_MFM_GRADIENTS_VX = 1700,
    IO_MFM_GRADIENTS_VY = 1701,
    IO_MFM_GRADIENTS_VZ = 1702,
    IO_MFM_GRADIENTS_RHO = 1703,
    IO_MFM_GRADIENTS_U = 1704,
    IO_MFM_GRADIENTS_BX = 1705,
    IO_MFM_GRADIENTS_BY = 1706,
    IO_MFM_GRADIENTS_BZ = 1707,
    IO_MFM_GRADIENTS_PHI = 1708,

    // subfind
    IO_DMHSML = 1800, /* for 'SUBFIND_RESHUFFLE_CATALOGUE' option */
    IO_DMDENSITY = 1801,
    IO_DMVELDISP = 1802,
    IO_MAINHALO = 1803, /* Rank in the last subfind output */

    IO_LASTENTRY = 2999 /* This should be kept - it signals the end of the list */
};

enum siofields
{
    SIO_GLEN,
    SIO_GOFF,
    SIO_MTOT,
    SIO_GPOS,
    SIO_MMEA,
    SIO_RMEA,
    SIO_MCRI,
    SIO_RCRI,
    SIO_MTOP,
    SIO_RTOP,
    SIO_M500,
    SIO_R500,
    SIO_M5CC,
    SIO_R5CC,
    SIO_M25K,
    SIO_R25K,
    SIO_DMEA,
    SIO_DCRI,
    SIO_DTOP,
    SIO_D500,
    SIO_D5CC,
    SIO_D25K,
    SIO_MGAS,
    SIO_MSTR,
    SIO_TGAS,
    SIO_LGAS,
    SIO_YGAS,
    SIO_NCON,
    SIO_MCON,
    SIO_BGPOS,
    SIO_BGMTOP,
    SIO_BGRTOP,
    SIO_NSUB,
    SIO_FSUB,
    SIO_SLEN,
    SIO_SOFF,
    SIO_PFOF,
    SIO_MSUB,
    SIO_SPOS,
    SIO_SVEL,
    SIO_SCM,
    SIO_SPIN,
    SIO_DSUB,
    SIO_VMAX,
    SIO_RVMAX,
    SIO_RHMS,
    SIO_SHMR,
    SIO_MBID,
    SIO_GRNR,
    SIO_SMST,
    SIO_SLUM,
    SIO_SLATT,
    SIO_SLOBS,
    SIO_DUST,
    SIO_SAGE,
    SIO_SZ,
    SIO_SSFR,
    SIO_SSYNCHROTRON,
    SIO_MHI,
    SIO_SSANG,
    SIO_SBVAL,
    SIO_SZGAS,
    SIO_SZCGAS,
    SIO_PPOS,
    SIO_PVEL,
    SIO_PTYP,
    SIO_PMAS,
    SIO_PAGE,
    SIO_PID,

    SIO_LASTENTRY
};

int io_compare_P_GrNr_SubNr(const void *a, const void *b);
void write_file(char *fname, int readTask, int lastTask);
void distribute_file(int nfiles, int firstfile, int firsttask, int lasttask, int *filenr, int *master, int *last);
void find_block(char *label, FILE *fd);
void fill_write_buffer(enum iofields blocknr, int *pindex, int pc, int type);
double F1_Omega(double);
double F2_Omega(double);
void check_omega(void);
int find_files(char *fname);
unsigned int get_particles_in_block(enum iofields blocknr, unsigned int *typelist);

#ifdef OUTPUT_IN_FP16
float float_from_log10_half(uint16_t h);
#endif // OUTPUT_IN_FP16

#ifdef HAVE_HDF5
#include <hdf5.h>
hid_t GetHDF5Datatype(int dtype);
void write_header_attributes_in_hdf5(hid_t handle);
void read_header_attributes_in_hdf5(char *fname);
void write_parameters_attributes_in_hdf5(hid_t handle);
void write_units_attributes_in_hdf5(hid_t handle);
void write_constants_attributes_in_hdf5(hid_t handle);
#endif // HAVE_HDF5

// SUBFIND & IO
int io_compare_P_ID(const void *a, const void *b);
int io_compare_P_GrNr_ID(const void *a, const void *b);

#ifdef TEST_COPY_BOX_PERIODICALLY
void empty_read_buffer(enum iofields blocknr, int offset, int pc, int type, int ncopy);
void read_file(char *fname, int readTask, int lastTask, int ncopy);
#else // TEST_COPY_BOX_PERIODICALLY
void empty_read_buffer(enum iofields blocknr, int offset, int pc, int type);
void read_file(char *fname, int readTask, int lastTask);
#endif // TEST_COPY_BOX_PERIODICALLY

void swap_Nbyte(char *data, int n, int m);
void swap_header(void);

#ifdef WRITE_KEY_FILES
int key_compare(const void *a, const void *b);
int key_list_compare(const void *a, const void *b);
void write_key_index_file(int num, int filenr);
#endif // WRITE_KEY_FILES

#endif // IO_HPP
