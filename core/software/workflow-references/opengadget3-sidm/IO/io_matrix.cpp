/**
 * @file
 *
 * This file is part of the N-body/SPH code OpenGadget3.
 *
 * @brief This file contains functions related to the io-matrix.
 *
 * @author OpenGadget3-development team, in particular Geray S. Karademir.
 *
 * @copyright Copyright (c) 2026, OpenGadget3-development team lead by Klaus Dolag and Stefano Borgani
 * (see change logs). All rights reserved.
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License v3 or later.
 * See <https://www.gnu.org/licenses/>.
 */

#include <cstring>

#include "../CodeBase/allvars.hpp"
#include "io_matrix.hpp"

namespace io_matrix
{
/**
 * @brief enum to assign different data types to use case
 */
enum Typekey
{
    TYPE_INT = 0,
    TYPE_FLOAT = 1,
    TYPE_LONG_LONG = 2,
    TYPE_FLOAT16 = 3
};

/**
 * @brief return datatype indicator for int
 */
int datatype_in_block_int()
{
    return Typekey::TYPE_INT; /* native int */
}

/**
 * @brief return datatype indicator for float
 */
int datatype_in_block_float()
{
    return Typekey::TYPE_FLOAT; /* native MyOutputFloat */
}

#ifdef OUTPUT_IN_FP16
/**
 * @brief return datatype indicator for float16
 */
int datatype_in_block_float16()
{
    return Typekey::TYPE_FLOAT16;
}

#define FLOAT_OR_FP16 uint16_t
#else // OUTPUT_IN_FP16
#define FLOAT_OR_FP16 MyOutputFloat
#endif // OUTPUT_IN_FP16

/**
 * @brief return datatype indicator for float, or float16 if OUTPUT_IN_FP16 is set
 */
int datatype_in_block_float_or_fp16()
{
#ifdef OUTPUT_IN_FP16
    return datatype_in_block_float16();
#else // OUTPUT_IN_FP16
    return datatype_in_block_float();
#endif // OUTPUT_IN_FP16
}

/**
 * @brief return datatype indicator for int or longlong if LONGIDS
 */
int datatype_in_block_int_or_longlong()
{
#ifdef LONGIDS
    return Typekey::TYPE_LONG_LONG; /* native long long */
#else // LONGIDS
    return Typekey::TYPE_INT; /* native int */
#endif // LONGIDS
}

/**
 * @brief struct containing multiple number counts
 */
struct BlockCounts
{
    unsigned int nall;
    unsigned int ngas;
    unsigned int nstars;
    unsigned int ntot_withmasses;
};

/**
 * @brief Computes particle block counts and initializes type list flags.
 *
 * @param typelist Output array marking active particle types.
 * @return Aggregated block counts (gas, stars, total, etc.).
 */
BlockCounts block_main(unsigned int *typelist)
{
    BlockCounts counts_struct{};

    counts_struct.ngas = header.npart[0];
    counts_struct.nstars = header.npart[4];

    for (int i = 0; i < 6; i++)
    {
        typelist[i] = 0;

        if (header.npart[i] > 0)
        {
            counts_struct.nall += header.npart[i];
            typelist[i] = 1;
        }

        if (All.MassTable[i] == 0)
        {
            counts_struct.ntot_withmasses += header.npart[i];
        }
    }

    return counts_struct;
}

/**
 * @brief Returns total number of particles.
 *
 * @param typelist Output type selection mask.
 * @return Total particle count.
 */
int block_all_particles(unsigned int *typelist)
{
    BlockCounts counts_struct = block_main(typelist);
    return counts_struct.nall;
}

/**
 * @brief Returns number of gas particles.
 *
 * @param typelist Output type selection mask.
 * @return Gas particle count.
 */
int block_gas(unsigned int *typelist)
{
    BlockCounts counts_struct = block_main(typelist);

    for (int i = 1; i < 6; i++)
    {
        typelist[i] = 0;
    }

    return counts_struct.ngas;
}

/**
 * @brief Returns number of star particles.
 *
 * @param typelist Output type selection mask.
 * @return Star particle count.
 */
int block_stars(unsigned int *typelist)
{
    BlockCounts counts_struct = block_main(typelist);

    for (int i = 0; i < 6; i++)
    {
        if (i != 4)
        {
            typelist[i] = 0;
        }
    }

    return counts_struct.nstars;
}

/**
 * @brief Returns number of particles with non-zero mass entries.
 *
 * @param typelist Output type selection mask.
 * @return Count of particles with stored masses.
 */
int block_mass(unsigned int *typelist)
{
    BlockCounts counts_struct = block_main(typelist);

    for (int i = 0; i < 6; i++)
    {
        typelist[i] = 0;
        if (All.MassTable[i] == 0 && header.npart[i] > 0)
        {
            typelist[i] = 1;
        }
    }
    return counts_struct.ntot_withmasses;
}

/**
 * @brief Returns combined gas and star particle count.
 *
 * @param typelist Output type selection mask.
 * @return Gas + star particle count.
 */
int block_gas_and_stars(unsigned int *typelist)
{
    BlockCounts counts_struct = block_main(typelist);

    for (int i = 0; i < 6; i++)
    {
        if (i != 0 && i != 4)
        {
            typelist[i] = 0;
        }
    }
    return counts_struct.ngas + counts_struct.nstars;
}

/**
 * @brief Returns number of black hole particles.
 *
 * @param typelist Output type selection mask.
 * @return Black hole particle count.
 */
int block_blackhole(unsigned int *typelist)
{
    for (int i = 0; i < 6; i++)
    {
        if (i != 5)
        {
            typelist[i] = 0;
        }
    }
    typelist[5] = (header.npart[5] > 0) ? 1 : 0;
    return header.npart[5];
}

/**
 * @brief Returns star and (optionally) black hole particle count.
 *
 * @param typelist Output type selection mask.
 * @return Combined star (and BH if enabled) particle count.
 */
int block_blackholes_andor_stars(unsigned int *typelist)
{
    BlockCounts counts_struct = block_main(typelist);

    for (int i = 0; i < 6; i++)
#ifdef BLACK_HOLES
        if (i != 4 && i != 5)
        {
            typelist[i] = 0;
        }
    return counts_struct.nstars + header.npart[5];
#else // BLACK_HOLES
        if (i != 4)
        {
            typelist[i] = 0;
        }
    return counts_struct.nstars;
#endif // BLACK_HOLES
}

#ifdef KD_OUTPUT_NGBS
/**
 * @brief Returns neighbour-related particle count depending on compilation options.
 *
 * @param typelist Output type selection mask.
 * @return Number of neighbour-contributing particles.
 */
int block_NGB(unsigned int *typelist)
{
    BlockCounts counts_struct = block_main(typelist);

    unsigned int nngb = counts_struct.ngas;
    for (int i = 1; i < 4; i++)
    {
        typelist[i] = 0;
    }
#ifdef LT_STELLAREVOLUTION
    nngb += header.npart[4];
#else // LT_STELLAREVOLUTION
    typelist[4] = 0;
#endif // LT_STELLAREVOLUTION
#ifdef BLACK_HOLES
    nngb += header.npart[5];
#else // BLACK_HOLES
    typelist[5] = 0;
#endif // BLACK_HOLES
    return nngb;
}
#endif // KD_OUTPUT_NGBS

/**
 * @brief Returns smoothed metallicity-related particle count.
 *
 * @param typelist Output type selection mask.
 * @return Number of contributing particles.
 */
int block_ZSMOOTH(unsigned int *typelist)
{
    BlockCounts counts_struct = block_main(typelist);

    for (int i = 0; i < 6; i++)
#ifndef LT_SMOOTHZ_IN_IMF_SWITCH
        if (i != 0)
#else // LT_SMOOTHZ_IN_IMF_SWITCH
        if (i != 0 && i != 4)
#endif // LT_SMOOTHZ_IN_IMF_SWITCH
        {
            typelist[i] = 0;
        }
#ifndef LT_SMOOTHZ_IN_IMF_SWITCH
    return counts_struct.ngas;
#else // LT_SMOOTHZ_IN_IMF_SWITCH
    return counts_struct.ngas + counts_struct.nstars;
#endif // LT_SMOOTHZ_IN_IMF_SWITCH
}

#if defined(SUBFIND_RESHUFFLE_CATALOGUE) && defined(SUBFIND)
/**
 * @brief Returns number of selected dark matter particles for SUBFIND reshuffling.
 *
 * @param typelist Output type selection mask.
 * @return Number of selected particles.
 */
int block_DM_nsel(unsigned int *typelist)
{
    BlockCounts counts_struct = block_main(typelist);

    unsigned int nsel = 0;
    for (int i = 0; i < 6; i++)
    {
        if (((1 << i) & (FOF_PRIMARY_LINK_TYPES)))
        {
            nsel += header.npart[i];
        }
        else
        {
            typelist[i] = 0;
        }
    }
    return nsel;
}
#endif // SUBFIND_RESHUFFLE_CATALOGUE & SUBFIND

typedef int (*BlockFunc)(unsigned int *typelist);
typedef int (*DTypeInBlock)();

/**
 * @brief Describes a single I/O dataset field used for read/write operations.
 *
 * This structure defines the metadata required to map simulation data fields
 * to their corresponding representations.
 *
 * It includes the dataset name, short I/O label, dimensionality, and the
 * input/output byte sizes for type conversion during I/O operations.
 */
typedef struct
{
    const char *dataset_name; // get_dataset_name
    const char *label; // get_Tab_IO_Label (4 chars)
    int values_per_element; // get_values_per_blockelement
    size_t bytes_input; // get_bytes_per_blockelement - input
    size_t bytes_output; // get_bytes_per_blockelement - output
    BlockFunc particles_in_block; // get_particles_in_block
    DTypeInBlock datatype_in_block; // datatype_in_block
    bool read_from_ic; // read from ic
    bool read_restartfile; // read if restarting from restartfile
    bool read_restart_snap; // read if restarting from snapshot file
    bool write_to_snap; // write to snapshot
    bool read_from_spectra; // read if restarting from spectra file
} IOFieldInfo;

static void apply_read_from_ic_policy(IOFieldInfo *fields
)
{
    for (enum iofields blocknr = (enum iofields)0; blocknr < IO_LASTENTRY; blocknr = (enum iofields)(blocknr + 1))
    {
#ifndef CHEMISTRY
        if (blocknr > IO_U && blocknr != IO_BFLD
#ifdef READ_HSML
            && blocknr != IO_HSML
#endif // READ_HSML
#ifdef GT_DUST
            && blocknr != IO_DFRC && blocknr != IO_DVEL && blocknr != IO_DRAG
#endif // GT_DUST
#ifdef LMB_SPECTRAL_CRs_FROM_ICS
            && blocknr != IO_LMBCR_pNORM && blocknr != IO_LMBCR_eNORM && blocknr != IO_LMBCR_pSLOPE &&
            blocknr != IO_LMBCR_eSLOPE && blocknr != IO_LMBCR_pCUT && blocknr != IO_LMBCR_eCUT
#endif // LMB_SPECTRAL_CRs_FROM_ICS
#ifdef READ_RHO_OLD
            && blocknr != IO_RHO_OLD
#endif // READ_RHO_OLD
#ifdef READ_MACH
            && blocknr != IO_MACH
#endif // READ_MACH
#if defined(BHMASS_IN_ICS) && defined(BLACK_HOLES)
            && blocknr != IO_BHMASS
#endif // defined(BHMASS_IN_ICS) && defined(BLACK_HOLES)
#if defined(BHSPIN_IN_ICS) && defined(BLACK_HOLES)
            && blocknr != IO_BHSPIN
#endif // defined(BHSPIN_IN_ICS) && defined(BLACK_HOLES)
#if defined(LT_Zs_IN_IC) && defined(LT_STELLAREVOLUTION)
            && blocknr != IO_Zs
#endif // LT_Zs_IN_IC && LT_STELLAREVOLUTION
        )
#else // CHEMISTRY
        if (blocknr > IO_HM)
#endif // CHEMISTRY
        {
            fields[blocknr].read_from_ic = false; /* ignore all other blocks in initial conditions */
        }
    }

#if defined(UM_CHEMISTRY) && defined(UM_CHEMISTRY_INISET)
    fields[IO_NE].read_from_ic = false;
    fields[IO_NH].read_from_ic = false;
    fields[IO_HII].read_from_ic = false;
    fields[IO_HM].read_from_ic = false;
    fields[IO_HeI].read_from_ic = false;
    fields[IO_HeII].read_from_ic = false;
    fields[IO_HeIII].read_from_ic = false;
    fields[IO_H2I].read_from_ic = false;
    fields[IO_H2II].read_from_ic = false;
    fields[IO_HD].read_from_ic = false;
    fields[IO_DI].read_from_ic = false;
    fields[IO_DII].read_from_ic = false;
    fields[IO_HeHII].read_from_ic = false;
#endif // UM_CHEMISTRY && UM_CHEMISTRY_INISET

#ifdef NO_UTHERM_IN_IC_FILE
    fields[IO_U].read_from_ic = false;
#endif // NO_UTHERM_IN_IC_FILE

#ifdef BINISET
    fields[IO_BFLD].read_from_ic = false;
#endif // BINISET


#ifdef LT_STELLAREVOLUTION
#ifdef LT_TRACK_CONTRIBUTES
    if (FdTrck == 0x0)
    {
        fields[IO_CONTRIB].read_from_ic = false;
    }
#endif // LT_TRACK_CONTRIBUTES
#endif // LT_STELLAREVOLUTION
}

/**
 * @brief Metadata for an I/O field used in snapshot input/output.
 *
 * Defines dataset identification, I/O label, dimensionality, and
 * input/output byte sizes for conversion between in-memory and file formats.
 */
static const IOFieldInfo *get_io_fields(
)
{
    static IOFieldInfo fields[IO_LASTENTRY] = {}; // initialise all with NULL
    /* always present */
    fields[IO_POS] = {"Coordinates",
                      "POS ",
                      3,
                      sizeof(MyInputFloat),
                      sizeof(MyOutputFloat),
                      block_all_particles,
                      datatype_in_block_float,
                      true,
                      true,
                      true,
                      true,
                      false};
    fields[IO_VEL] = {"Velocities",
                      "VEL ",
                      3,
                      sizeof(MyInputFloat),
                      sizeof(MyOutputFloat),
                      block_all_particles,
                      datatype_in_block_float,
                      true,
                      true,
                      true,
                      true,
                      false};
    fields[IO_ID] = {"ParticleIDs",
                     "ID  ",
                     1,
                     sizeof(MyIDType),
                     sizeof(MyIDType),
                     block_all_particles,
                     datatype_in_block_int_or_longlong,
                     true,
                     true,
                     true,
                     true,
                     false};
    fields[IO_MASS] = {"Masses",
                       "MASS",
                       1,
                       sizeof(MyInputFloat),
                       sizeof(FLOAT_OR_FP16),
                       block_mass,
                       datatype_in_block_float_or_fp16,
                       true,
                       true,
                       true,
                       true,
                       true};
    fields[IO_U] = {"InternalEnergy",
                    "U   ",
                    1,
                    sizeof(MyInputFloat),
                    sizeof(MyOutputFloat),
                    block_gas,
                    datatype_in_block_float,
                    true,
                    true,
                    true,
                    true,
                    false};
    fields[IO_RHO] = {"Density",
                      "RHO ",
                      1,
                      sizeof(MyInputFloat),
                      sizeof(MyOutputFloat),
                      block_gas,
                      datatype_in_block_float,
                      true,
                      true,
                      true,
                      true,
                      false};
    fields[IO_HSML] = {"SmoothingLength",
                       "HSML",
                       1,
                       sizeof(MyInputFloat),
                       sizeof(MyOutputFloat),
                       block_gas,
                       datatype_in_block_float,
                       true,
                       true,
                       true,
                       true,
                       false};
    fields[IO_ABVC] = {"ArtificialViscosity",
                       "ABVC",
                       1,
                       sizeof(MyInputFloat),
                       sizeof(MyOutputFloat),
                       block_gas,
                       datatype_in_block_float,
                       true,
                       true,
                       true,
                       true,
                       false};

    /* NE, NH: present when cooling is on, or when chemistry is compiled in */
#if defined(CHEMISTRY) || defined(UM_CHEMISTRY)
    fields[IO_NE] = {"ElectronAbund",
                     "NE  ",
                     1,
                     sizeof(MyInputFloat),
                     sizeof(MyOutputFloat),
                     block_gas,
                     datatype_in_block_float,
                     true,
                     true,
                     true,
                     true,
                     false};
    fields[IO_NH] = {"NeutralHydrogenAbundance",
                     "NH  ",
                     1,
                     sizeof(MyInputFloat),
                     sizeof(MyOutputFloat),
                     block_gas,
                     datatype_in_block_float,
                     true,
                     true,
                     true,
                     true,
                     false};
#else // defined(CHEMISTRY) || defined(UM_CHEMISTRY)
    if (All.CoolingOn)
    {
        fields[IO_NE] = {"ElectronAbund",
                         "NE  ",
                         1,
                         sizeof(MyInputFloat),
                         sizeof(MyOutputFloat),
                         block_gas,
                         datatype_in_block_float,
                         true,
                         true,
                         true,
                         true,
                         false};
        fields[IO_NH] = {"NeutralHydrogenAbundance",
                         "NH  ",
                         1,
                         sizeof(MyInputFloat),
                         sizeof(MyOutputFloat),
                         block_gas,
                         datatype_in_block_float,
                         true,
                         true,
                         true,
                         true,
                         false};
    }
#endif // defined(CHEMISTRY) || defined(UM_CHEMISTRY)

#if defined(CHEMISTRY) || defined(UM_CHEMISTRY)
    fields[IO_HII] = {"HII",
                      "HII ",
                      1,
                      sizeof(MyInputFloat),
                      sizeof(MyOutputFloat),
                      block_gas,
                      datatype_in_block_float,
                      true,
                      true,
                      true,
                      true,
                      false};
    fields[IO_HeI] = {"HeI",
                      "HeI ",
                      1,
                      sizeof(MyInputFloat),
                      sizeof(MyOutputFloat),
                      block_gas,
                      datatype_in_block_float,
                      true,
                      true,
                      true,
                      true,
                      false};
    fields[IO_HeII] = {"HeII",
                       "HeII",
                       1,
                       sizeof(MyInputFloat),
                       sizeof(MyOutputFloat),
                       block_gas,
                       datatype_in_block_float,
                       true,
                       true,
                       true,
                       true,
                       false};
    fields[IO_HeIII] = {"HeIII",
                        "He3 ",
                        1,
                        sizeof(MyInputFloat),
                        sizeof(MyOutputFloat),
                        block_gas,
                        datatype_in_block_float,
                        true,
                        true,
                        true,
                        true,
                        false};
    fields[IO_H2I] = {"H2I",
                      "H2I ",
                      1,
                      sizeof(MyInputFloat),
                      sizeof(MyOutputFloat),
                      block_gas,
                      datatype_in_block_float,
                      true,
                      true,
                      true,
                      true,
                      false};
    fields[IO_H2II] = {"H2II",
                       "H2II",
                       1,
                       sizeof(MyInputFloat),
                       sizeof(MyOutputFloat),
                       block_gas,
                       datatype_in_block_float,
                       true,
                       true,
                       true,
                       true,
                       false};
    fields[IO_HM] = {
        "HM", "HM  ", 1,    sizeof(MyInputFloat), sizeof(MyOutputFloat), block_gas, datatype_in_block_float, true, true,
        true, true,   false};
#endif // defined(CHEMISTRY) || defined(UM_CHEMISTRY)

#ifdef UM_CHEMISTRY
    fields[IO_HD] = {"HD  ",
                     "HD  ",
                     1,
                     sizeof(MyInputFloat),
                     sizeof(MyOutputFloat),
                     block_gas,
                     datatype_in_block_float,
                     true,
                     true,
                     true,
                     true,
                     false};
    fields[IO_DI] = {"DI  ",
                     "DI  ",
                     1,
                     sizeof(MyInputFloat),
                     sizeof(MyOutputFloat),
                     block_gas,
                     datatype_in_block_float,
                     true,
                     true,
                     true,
                     true,
                     false};
    fields[IO_DII] = {"DII ",
                      "DII ",
                      1,
                      sizeof(MyInputFloat),
                      sizeof(MyOutputFloat),
                      block_gas,
                      datatype_in_block_float,
                      true,
                      true,
                      true,
                      true,
                      false};
    fields[IO_HeHII] = {"HeHp",
                        "HeHp",
                        1,
                        sizeof(MyInputFloat),
                        sizeof(MyOutputFloat),
                        block_gas,
                        datatype_in_block_float,
                        true,
                        true,
                        true,
                        true,
                        false};
#endif // UM_CHEMISTRY

#ifdef WINDS
    fields[IO_DELAYTIME] = {"DelayTime",
                            "DETI",
                            1,
                            sizeof(MyInputFloat),
                            sizeof(MyOutputFloat),
                            block_gas,
                            datatype_in_block_float,
                            true,
                            true,
                            false,
                            true,
                            false};
#endif // WINDS

#ifdef PB_BH_KIN_FEED_DECOUPLE_WIND
    fields[IO_DECOUPLE_TIME_AGN] = {"Decoupling Time of AGN Wind",
                                    "DTBH",
                                    1,
                                    sizeof(MyInputFloat),
                                    sizeof(MyOutputFloat),
                                    block_gas,
                                    datatype_in_block_float,
                                    true,
                                    true,
                                    true,
                                    true,
                                    false};
#endif // PB_BH_KIN_FEED_DECOUPLE_WIND

#ifdef SFR
    if (All.StarformationOn)
    {
        fields[IO_SFR] = {"StarFormationRate",
                          "SFR ",
                          1,
                          sizeof(MyInputFloat),
                          sizeof(MyOutputFloat),
                          block_gas,
                          datatype_in_block_float,
                          true,
                          true,
                          true,
                          true,
                          false};
    }
#endif // SFR

#ifdef SFR
    if (All.StarformationOn)
    {
        fields[IO_AGE] = {"StellarFormationTime",
                          "AGE ",
                          1,
                          sizeof(MyInputFloat),
                          sizeof(MyOutputFloat),
                          block_blackholes_andor_stars,
                          datatype_in_block_float,
                          true,
                          true,
                          true,
                          true,
                          false};
    }
#endif // SFR

#ifdef SUBFIND
    fields[IO_HSMS] = {"StellarSmoothingLength",
                       "HSMS",
                       1,
                       sizeof(MyInputFloat),
                       sizeof(MyOutputFloat),
                       block_stars,
                       datatype_in_block_float,
                       false,
                       false,
                       false,
                       true,
                       false};
#endif // SUBFIND

#ifdef LT_STELLAREVOLUTION
    fields[IO_ACRS] = {"StellarSpreadingLength",
                       "ACRS",
                       1,
                       sizeof(MyInputFloat),
                       sizeof(MyOutputFloat),
                       block_stars,
                       datatype_in_block_float,
                       true,
                       true,
                       true,
                       true,
                       false};
#endif // LT_STELLAREVOLUTION

#if defined(SFR) && !defined(LT_STELLAREVOLUTION)
    if (All.StarformationOn)
    {
        fields[IO_Z] = {"Metallicity",
                        "Z   ",
                        1,
                        sizeof(MyInputFloat),
                        sizeof(FLOAT_OR_FP16),
                        block_gas_and_stars,
                        datatype_in_block_float_or_fp16,
                        true,
                        true,
                        true,
                        true,
                        false};
    }
#endif // SFR && !LT_STELLAREVOLUTION

#if defined(OUTPUTPOTENTIAL) || defined(SUBFIND_RESHUFFLE_AND_POTENTIAL)
    fields[IO_POT] = {"Potential",
                      "POT ",
                      1,
                      sizeof(MyInputFloat),
                      sizeof(MyOutputFloat),
                      block_all_particles,
                      datatype_in_block_float,
                      false,
                      false,
                      false,
                      true,
                      false};
#endif // defined(OUTPUTPOTENTIAL) || defined(SUBFIND_RESHUFFLE_AND_POTENTIAL)

#ifdef PHIDOT
    fields[IO_PHIDOT] = {"DPotentialDt",
                         "PHID",
                         1,
                         sizeof(MyInputFloat),
                         sizeof(MyOutputFloat),
                         block_gas,
                         datatype_in_block_float,
                         true,
                         true,
                         true,
                         true,
                         false};
#endif // PHIDOT

#ifdef OUTPUTACCELERATION
    fields[IO_ACCEL] = {"Acceleration",
                        "ACCE",
                        3,
                        sizeof(MyInputFloat),
                        sizeof(MyOutputFloat),
                        block_all_particles,
                        datatype_in_block_float,
                        true,
                        true,
                        false,
                        true,
                        false};
#endif // OUTPUTACCELERATION

#ifdef OUTPUTCHANGEOFENTROPY
    fields[IO_DTENTR] = {"RateOfChangeOfEntropy",
                         "ENDT",
                         1,
                         sizeof(MyInputFloat),
                         sizeof(MyOutputFloat),
                         block_gas,
                         datatype_in_block_float,
                         true,
                         true,
                         true,
                         true,
                         false};
#endif // OUTPUTCHANGEOFENTROPY

#ifdef OUTPUTSTRESS
    fields[IO_STRESSDIAG] = {"DiagonalStressTensor",
                             "STRD",
                             3,
                             sizeof(MyInputFloat),
                             sizeof(MyOutputFloat),
                             block_gas,
                             datatype_in_block_float,
                             true,
                             true,
                             true,
                             true,
                             false};
    fields[IO_STRESSOFFDIAG] = {"OffDiagonalStressTensor",
                                "STRO",
                                3,
                                sizeof(MyInputFloat),
                                sizeof(MyOutputFloat),
                                block_gas,
                                datatype_in_block_float,
                                true,
                                true,
                                true,
                                true,
                                false};
#endif // OUTPUTSTRESS

#ifdef OUTPUTBULKSTRESS
    fields[IO_STRESSBULK] = {"BulkStressTensor",
                             "STRB",
                             1,
                             sizeof(MyInputFloat),
                             sizeof(MyOutputFloat),
                             block_gas,
                             datatype_in_block_float,
                             true,
                             true,
                             true,
                             true,
                             false};
#endif // OUTPUTBULKSTRESS

#ifdef OUTPUTSHEARCOEFF
    fields[IO_SHEARCOEFF] = {"ShearCoefficient",
                             "SHCO",
                             1,
                             sizeof(MyInputFloat),
                             sizeof(MyOutputFloat),
                             block_gas,
                             datatype_in_block_float,
                             true,
                             true,
                             true,
                             true,
                             false};
#endif // OUTPUTSHEARCOEFF

#ifdef OUTPUT_PRESSURE_ANISO
    fields[IO_PRESSURE_ANISO] = {"PressureAnisotropy",
                                 "PANI",
                                 1,
                                 sizeof(MyInputFloat),
                                 sizeof(MyOutputFloat),
                                 block_gas,
                                 datatype_in_block_float,
                                 true,
                                 true,
                                 true,
                                 true,
                                 false};
#endif // OUTPUT_PRESSURE_ANISO

#ifdef OUTPUT_ENERGY_VISC
    fields[IO_ENERGY_VISC] = {"ViscousHeating",
                              "VISE",
                              1,
                              sizeof(MyInputFloat),
                              sizeof(MyOutputFloat),
                              block_gas,
                              datatype_in_block_float,
                              true,
                              true,
                              true,
                              true,
                              false};
#endif // OUTPUT_ENERGY_VISC

#ifdef OUTPUTTIMESTEP
    fields[IO_TSTP] = {"TimeStep",
                       "TSTP",
                       1,
                       sizeof(MyInputFloat),
                       sizeof(MyOutputFloat),
                       block_all_particles,
                       datatype_in_block_float,
                       false,
                       false,
                       false,
                       true,
                       false};
#endif // OUTPUTTIMESTEP

#ifdef MAGNETIC
    fields[IO_BFLD] = {"MagneticField",
                       "BFLD",
                       3,
                       sizeof(MyInputFloat),
                       sizeof(MyOutputFloat),
                       block_gas,
                       datatype_in_block_float,
                       true,
                       true,
                       true,
                       true,
                       false};
    fields[IO_DIVB] = {"DivergenceOfMagneticField",
                       "DIVB",
                       1,
                       sizeof(MyInputFloat),
                       sizeof(MyOutputFloat),
                       block_gas,
                       datatype_in_block_float,
                       true,
                       true,
                       true,
                       true,
                       false};
#endif // MAGNETIC

#if defined(MAGNETIC) && defined(OUTPUT_MAGNETIC_CURVATURE)
    fields[IO_BCRV] = {"MagneticFieldCurvature",
                       "BCRV",
                       3,
                       sizeof(MyInputFloat),
                       sizeof(MyOutputFloat),
                       block_gas,
                       datatype_in_block_float,
                       true,
                       true,
                       true,
                       true,
                       false};
#endif // defined(MAGNETIC) && defined(OUTPUT_MAGNETIC_CURVATURE)

#if defined(MFM_FACE_CLOSURE_RESIDUAL)
    fields[IO_FACLO] = {"FaceClosureResidual",
                        "FACL",
                        1,
                        sizeof(MyInputFloat),
                        sizeof(MyOutputFloat),
                        block_gas,
                        datatype_in_block_float,
                        true,
                        true,
                        true,
                        true,
                        false};
#endif // defined(MFM_FACE_CLOSURE_RESIDUAL)

#ifdef TIME_DEP_ART_COND
    fields[IO_GRADU] = {"GradientOfU",
                        "GRDU",
                        3,
                        sizeof(MyInputFloat),
                        sizeof(MyOutputFloat),
                        block_gas,
                        datatype_in_block_float,
                        false,
                        false,
                        false,
                        true,
                        false};
    fields[IO_ACVC] = {"ArtificialConductivity",
                       "ACVC",
                       1,
                       sizeof(MyInputFloat),
                       sizeof(MyOutputFloat),
                       block_gas,
                       datatype_in_block_float,
                       true,
                       true,
                       true,
                       true,
                       false};
#endif // TIME_DEP_ART_COND

#ifdef DBOUTPUT
    fields[IO_DBDT] = {"RateOfChangeOfMagneticField",
                       "DBDT",
                       3,
                       sizeof(MyInputFloat),
                       sizeof(MyOutputFloat),
                       block_gas,
                       datatype_in_block_float,
                       true,
                       true,
                       true,
                       true,
                       false};
#endif // DBOUTPUT

#ifdef JD_VTURB
    fields[IO_VRMS] = {"RMSVelocity",
                       "VRMS",
                       1,
                       sizeof(MyInputFloat),
                       sizeof(MyOutputFloat),
                       block_gas,
                       datatype_in_block_float,
                       true,
                       true,
                       true,
                       true,
                       false};
    fields[IO_VBULK] = {"BulkVelocity",
                        "VBLK",
                        3,
                        sizeof(MyInputFloat),
                        sizeof(MyOutputFloat),
                        block_gas,
                        datatype_in_block_float,
                        true,
                        true,
                        true,
                        true,
                        false};
#endif // JD_VTURB

#ifdef JD_DECOMPOSE_VTURB
    fields[IO_VRAD] = {"RMSRadialVelocity",
                       "VRAD",
                       1,
                       sizeof(MyInputFloat),
                       sizeof(MyOutputFloat),
                       block_gas,
                       datatype_in_block_float,
                       true,
                       true,
                       true,
                       true,
                       false};
    fields[IO_VTAN] = {"RMSTangentialVelocity",
                       "VTAN",
                       1,
                       sizeof(MyInputFloat),
                       sizeof(MyOutputFloat),
                       block_gas,
                       datatype_in_block_float,
                       true,
                       true,
                       true,
                       true,
                       false};
#endif // JD_DECOMPOSE_VTURB

#if defined(KD_MAIN_HALO) && defined(SUBFIND)
    fields[IO_MAINHALO] = {"MainHaloIndex",       "MHIX", 1,    sizeof(int), sizeof(int), block_all_particles,
                           datatype_in_block_int, true,   true, false,       true,        false};
#endif // defined(KD_MAIN_HALO) && defined(SUBFIND)

#ifdef KD_OUTPUT_NGBS
    fields[IO_TRUENGB] = {"TrueNumberOfNeighbours", "TNGB", 1,    sizeof(int), sizeof(int), block_NGB,
                          datatype_in_block_int,    true,   true, true,        true,        false};
    fields[IO_WNGB] = {"WeigthedNumberOfNeighbours",
                       "NGB ",
                       1,
                       sizeof(MyInputFloat),
                       sizeof(MyOutputFloat),
                       block_NGB,
                       datatype_in_block_float,
                       true,
                       true,
                       true,
                       true,
                       false};
#endif // KD_OUTPUT_NGBS

#if defined(JD_DPP) || defined(LMB_DPP)
    fields[IO_DPP] = {"MagnetosReaccCoefficient",
                      "DPP ",
                      1,
                      sizeof(MyInputFloat),
                      sizeof(MyOutputFloat),
                      block_gas,
                      datatype_in_block_float,
                      true,
                      true,
                      true,
                      true, // write_to_snap
                      false}; // read_from_spectra
#endif // defined(JD_DPP) || defined(LMB_DPP)

#ifdef OUTPUT_DIV_CURL
    fields[IO_VDIV] = {"VelocityDivergence",
                       "VDIV",
                       1,
                       sizeof(MyInputFloat),
                       sizeof(MyOutputFloat),
                       block_gas,
                       datatype_in_block_float,
                       true,
                       true,
                       true,
                       true,
                       false};
    fields[IO_VROT] = {"VelocityCurl",
                       "VROT",
                       1,
                       sizeof(MyInputFloat),
                       sizeof(MyOutputFloat),
                       block_gas,
                       datatype_in_block_float,
                       true,
                       true,
                       true,
                       true,
                       false};
#endif // OUTPUT_DIV_CURL

#ifdef OUTPUT_VORTICITY
    fields[IO_VORT] = {"Vorticity",
                       "VORT",
                       3,
                       sizeof(MyInputFloat),
                       sizeof(MyOutputFloat),
                       block_gas,
                       datatype_in_block_float,
                       true,
                       true,
                       true,
                       true,
                       false};
#endif // OUTPUT_VORTICITY

#ifdef OUTPUT_GRADIENT_ORDER
    fields[IO_GRADIENT_ORDER] = {"GradientOrder",
                                 "GORD",
                                 1,
                                 sizeof(MyInputFloat),
                                 sizeof(MyOutputFloat),
                                 block_gas,
                                 datatype_in_block_float,
                                 true,
                                 true,
                                 true,
                                 true,
                                 false};
#endif // OUTPUT_GRADIENT_ORDER

#ifdef OUTPUT_GRADIENT_MATRICES
    fields[IO_GRADIENT_DISTANCEMATRIX] = {"GradientDistanceMatrix",
                                          "GRDX",
                                          9,
                                          sizeof(MyInputFloat),
                                          sizeof(MyOutputFloat),
                                          block_gas,
                                          datatype_in_block_float,
                                          true,
                                          true,
                                          true,
                                          true,
                                          false};
    fields[IO_GRADIENT_VELOCITYMATRIX] = {"GradientVelocityMatrix",
                                          "GRDV",
                                          9,
                                          sizeof(MyInputFloat),
                                          sizeof(MyOutputFloat),
                                          block_gas,
                                          datatype_in_block_float,
                                          true,
                                          true,
                                          true,
                                          true,
                                          false};
#endif // OUTPUT_GRADIENT_MATRICES

#if (TIME_DEP_ART_VISC > 0) && defined(ART_VISC_DEBUG)
    fields[IO_ABVR] = {"ArtificialViscosityR",
                       "ABVR",
                       1,
                       sizeof(MyInputFloat),
                       sizeof(MyOutputFloat),
                       block_gas,
                       datatype_in_block_float,
                       true,
                       true,
                       true,
                       true,
                       false};
    fields[IO_ABVH] = {"ArtificialViscosityh2A",
                       "ABVH",
                       1,
                       sizeof(MyInputFloat),
                       sizeof(MyOutputFloat),
                       block_gas,
                       datatype_in_block_float,
                       true,
                       true,
                       true,
                       true,
                       false};
#endif // (TIME_DEP_ART_VISC > 0) && defined(ART_VISC_DEBUG)

#ifdef TIME_DEP_MAGN_DISP
    fields[IO_AMDC] = {"ArtificialMagneticDissipatio",
                       "AMDC",
                       1,
                       sizeof(MyInputFloat),
                       sizeof(MyOutputFloat),
                       block_gas,
                       datatype_in_block_float,
                       true,
                       true,
                       true,
                       true,
                       false};
#endif // TIME_DEP_MAGN_DISP

#if defined(OUTPUTDEDNER) || defined(DIVBCLEANING_TRICCO)
    fields[IO_PHI] = {"DivBcleaningFunctionPhi",
                      "PHI ",
                      1,
                      sizeof(MyInputFloat),
                      sizeof(MyOutputFloat),
                      block_gas,
                      datatype_in_block_float,
                      true,
                      true,
                      true,
                      true,
                      false};
#endif // defined(OUTPUTDEDNER) || defined(DIVBCLEANING_TRICCO)

#if defined(MAGNETIC) && defined(SFR)
    fields[IO_XPHI] = {"ColdGasFraction_PHI",
                       "XPHI",
                       1,
                       sizeof(MyInputFloat),
                       sizeof(MyOutputFloat),
                       block_gas,
                       datatype_in_block_float,
                       true,
                       true,
                       true,
                       true,
                       false};
#endif // defined(MAGNETIC) && defined(SFR)

#ifdef DIVBCLEANING_TRICCO
    fields[IO_GRADPSI] = {"DivBcleaningFunctionGradPsi",
                          "GPSI",
                          3,
                          sizeof(MyInputFloat),
                          sizeof(MyOutputFloat),
                          block_gas,
                          datatype_in_block_float,
                          true,
                          true,
                          true,
                          true,
                          false};
#endif // DIVBCLEANING_TRICCO

#ifdef OUTPUTDEDNER
    fields[IO_GRADPHI] = {"DivBcleaningFunctionGradPhi",
                          "GPHI",
                          3,
                          sizeof(MyInputFloat),
                          sizeof(MyOutputFloat),
                          block_gas,
                          datatype_in_block_float,
                          true,
                          true,
                          true,
                          true,
                          false};
#endif // OUTPUTDEDNER

#ifdef OUTPUT_ROTB
    fields[IO_ROTB] = {"RotationB",
                       "ROTB",
                       3,
                       sizeof(MyInputFloat),
                       sizeof(MyOutputFloat),
                       block_gas,
                       datatype_in_block_float,
                       true,
                       true,
                       true,
                       true,
                       false};
#endif // OUTPUT_ROTB

#ifdef OUTPUT_SROTB
    fields[IO_SROTB] = {"SmoothedRotationB",
                        "SRTB",
                        3,
                        sizeof(MyInputFloat),
                        sizeof(MyOutputFloat),
                        block_gas,
                        datatype_in_block_float,
                        true,
                        true,
                        true,
                        true,
                        false};
#endif // OUTPUT_SROTB

#ifdef OUTPUTCOOLRATE
    fields[IO_COOLRATE] = {"CoolingRate",
                           "COOR",
                           1,
                           sizeof(MyInputFloat),
                           sizeof(MyOutputFloat),
                           block_gas,
                           datatype_in_block_float,
                           true,
                           true,
                           false,
                           true,
                           false};
#endif // OUTPUTCOOLRATE

#ifdef OUTPUTCONDRATE
    fields[IO_CONDRATE] = {"ConductionRate",
                           "CONR",
                           1,
                           sizeof(MyInputFloat),
                           sizeof(MyOutputFloat),
                           block_gas,
                           datatype_in_block_float,
                           true,
                           true,
                           true,
                           true,
                           false};
#endif // OUTPUTCONDRATE

#ifdef OUTPUTCOOLTIME
    fields[IO_COOLTIME] = {"CoolingTime",
                           "COOT",
                           1,
                           sizeof(MyInputFloat),
                           sizeof(MyOutputFloat),
                           block_gas,
                           datatype_in_block_float,
                           true,
                           true,
                           false,
                           true,
                           false};
#endif // OUTPUTCOOLTIME

#ifdef OUTPUTDENSNORM
    fields[IO_DENN] = {"Denn",
                       "DENN",
                       1,
                       sizeof(MyInputFloat),
                       sizeof(MyOutputFloat),
                       block_gas,
                       datatype_in_block_float,
                       true,
                       true,
                       true,
                       true,
                       false};
#endif // OUTPUTDENSNORM

#ifdef BLACK_HOLES
    fields[IO_BHMASS] = {"BH_Mass",
                         "BHMA",
                         1,
                         sizeof(MyInputFloat),
                         sizeof(MyOutputFloat),
                         block_blackhole,
                         datatype_in_block_float,
                         true,
                         true,
                         true,
                         true,
                         false};
    fields[IO_ACRB] = {"BH_AccreationLength",
                       "ACRB",
                       1,
                       sizeof(MyInputFloat),
                       sizeof(MyOutputFloat),
                       block_blackhole,
                       datatype_in_block_float,
                       true,
                       true,
                       true,
                       true,
                       false};
    fields[IO_BHMDOT] = {"BH_Mdot",
                         "BHMD",
                         1,
                         sizeof(MyInputFloat),
                         sizeof(MyOutputFloat),
                         block_blackhole,
                         datatype_in_block_float,
                         true,
                         true,
                         true,
                         true,
                         false};
    fields[IO_BHPROGS] = {"BH_NProgs",           "BHPC", 1,    sizeof(int), sizeof(int), block_blackhole,
                          datatype_in_block_int, true,   true, true,        true,        false};
    fields[IO_BHMBUB] = {"BH_Mass_bubbles",
                         "BHMB",
                         1,
                         sizeof(MyInputFloat),
                         sizeof(MyOutputFloat),
                         block_blackhole,
                         datatype_in_block_float,
                         true,
                         true,
                         true,
                         true,
                         false};
    fields[IO_BHMINI] = {"BH_Mass_ini",
                         "BHMI",
                         1,
                         sizeof(MyInputFloat),
                         sizeof(MyOutputFloat),
                         block_blackhole,
                         datatype_in_block_float,
                         true,
                         true,
                         true,
                         true,
                         false};
    fields[IO_BHMRAD] = {"BH_Mass_radio",
                         "BHMR",
                         1,
                         sizeof(MyInputFloat),
                         sizeof(MyOutputFloat),
                         block_blackhole,
                         datatype_in_block_float,
                         true,
                         true,
                         true,
                         true,
                         false};
#endif // BLACK_HOLES

#if defined(BLACK_HOLES) && defined(PB_BH_KIN_EJECT_PART_INSIDE_CONE)
    fields[IO_BH_Kin_ConeAxis] = {"BH_Unit_Vector_for_AGN_Wind_Kick",
                                  "BHWI",
                                  3,
                                  sizeof(MyInputFloat),
                                  sizeof(MyOutputFloat),
                                  block_blackhole,
                                  datatype_in_block_float,
                                  true,
                                  true,
                                  true,
                                  true,
                                  false};
#endif // defined(BLACK_HOLES) && defined(PB_BH_KIN_EJECT_PART_INSIDE_CONE)

#ifdef LS_BLACK_HOLES_SPIN_EVOLUTION
    fields[IO_BHSPIN] = {"BH_Spin",
                         "BHSP",
                         3,
                         sizeof(MyInputFloat),
                         sizeof(MyOutputFloat),
                         block_blackhole,
                         datatype_in_block_float,
                         false,
                         false,
                         false,
                         true,
                         false};
    fields[IO_BHETADISK] = {"BH_AccretionDiskEfficiency",
                            "BHED",
                            1,
                            sizeof(MyInputFloat),
                            sizeof(MyOutputFloat),
                            block_blackhole,
                            datatype_in_block_float,
                            false,
                            false,
                            false,
                            true,
                            false};
#endif // LS_BLACK_HOLES_SPIN_EVOLUTION

#ifdef OUTPUTMAXSIGVEL
    fields[IO_MAXSIGNALVEL] = {"MaximumSignalVelocity",
                               "VSIG",
                               1,
                               sizeof(MyInputFloat),
                               sizeof(MyOutputFloat),
                               block_gas,
                               datatype_in_block_float,
                               true,
                               true,
                               true,
                               true,
                               false};
#endif // OUTPUTMAXSIGVEL

#ifdef SHOCKFINDER
    fields[IO_MACH] = {"MachNumber",
                       "MACH",
                       1,
                       sizeof(MyInputFloat),
                       sizeof(MyOutputFloat),
                       block_gas,
                       datatype_in_block_float,
                       true,
                       true,
                       true,
                       true,
                       false};
    fields[IO_SHSPEED] = {"ShockSpeed",
                          "SHSP",
                          1,
                          sizeof(MyInputFloat),
                          sizeof(MyOutputFloat),
                          block_gas,
                          datatype_in_block_float,
                          true,
                          true,
                          true,
                          true,
                          false};
    fields[IO_SHCOMPRESS] = {"ShockCompressionRatio",
                             "SHCP",
                             1,
                             sizeof(MyInputFloat),
                             sizeof(MyOutputFloat),
                             block_gas,
                             datatype_in_block_float,
                             true,
                             true,
                             true,
                             true,
                             false};
    fields[IO_SHNORMAL] = {"ShockNormal",
                           "SHNR",
                           3,
                           sizeof(MyInputFloat),
                           sizeof(MyOutputFloat),
                           block_gas,
                           datatype_in_block_float,
                           true,
                           true,
                           true,
                           true,
                           false};
#endif // SHOCKFINDER


#if defined(SHOCKFINDER) && defined(SHOCKFINDER_RECONSTRUCT)
    fields[IO_SHSIGNALUP] = {"ShockUpstrSignalSpeed",
                             "SHSU",
                             1,
                             sizeof(MyInputFloat),
                             sizeof(MyOutputFloat),
                             block_gas,
                             datatype_in_block_float,
                             true,
                             true,
                             true,
                             true,
                             false};
    fields[IO_SHRHOUP] = {"ShockUpstrDensity",
                          "SHRH",
                          1,
                          sizeof(MyInputFloat),
                          sizeof(MyOutputFloat),
                          block_gas,
                          datatype_in_block_float,
                          true,
                          true,
                          true,
                          true,
                          false};
    fields[IO_SHPRESUP] = {"ShockUpstrPressure",
                           "SHPU",
                           1,
                           sizeof(MyInputFloat),
                           sizeof(MyOutputFloat),
                           block_gas,
                           datatype_in_block_float,
                           true,
                           true,
                           true,
                           true,
                           false};
    fields[IO_SHPRESDOWN] = {"ShockDownstrPressure",
                             "SHPD",
                             1,
                             sizeof(MyInputFloat),
                             sizeof(MyOutputFloat),
                             block_gas,
                             datatype_in_block_float,
                             true,
                             true,
                             true,
                             true,
                             false};
    fields[IO_SHVELUP] = {"ShockUpstrVelocity",
                          "SHVU",
                          1,
                          sizeof(MyInputFloat),
                          sizeof(MyOutputFloat),
                          block_gas,
                          datatype_in_block_float,
                          true,
                          true,
                          true,
                          true,
                          false};
    fields[IO_SHVELDOWN] = {"ShockDownstrVelocity",
                            "SHVD",
                            1,
                            sizeof(MyInputFloat),
                            sizeof(MyOutputFloat),
                            block_gas,
                            datatype_in_block_float,
                            true,
                            true,
                            true,
                            true,
                            false};
#endif // defined(SHOCKFINDER) && defined(SHOCKFINDER_RECONSTRUCT)

#if defined(SHOCKFINDER) && defined(MAGNETIC)
    fields[IO_MACH_ALFVEN] = {"AlfvenMachNumber",
                              "MALF",
                              1,
                              sizeof(MyInputFloat),
                              sizeof(MyOutputFloat),
                              block_gas,
                              datatype_in_block_float,
                              true,
                              true,
                              true,
                              true,
                              false};
    fields[IO_SHOBLIQUITY] = {"ShockObliquity",
                              "SHOB",
                              1,
                              sizeof(MyInputFloat),
                              sizeof(MyOutputFloat),
                              block_gas,
                              datatype_in_block_float,
                              true,
                              true,
                              true,
                              true,
                              false};
#endif // defined(SHOCKFINDER) && defined(MAGNETIC)

#if defined(SHOCKFINDER) && defined(SHOCKFINDER_RECONSTRUCT) && defined(MAGNETIC)
    fields[IO_SHALFVENUP] = {"ShockUpstrAlfven",
                             "SHAU",
                             1,
                             sizeof(MyInputFloat),
                             sizeof(MyOutputFloat),
                             block_gas,
                             datatype_in_block_float,
                             true,
                             true,
                             true,
                             true,
                             false};
#endif // defined(SHOCKFINDER) && defined(SHOCKFINDER_RECONSTRUCT) && defined(MAGNETIC)


    if (header.flag_ic_info == FLAG_SECOND_ORDER_ICS)
    {
        fields[IO_SECONDORDERMASS] = {"2lpt-mass",
                                      "SOMA",
                                      1,
                                      sizeof(MyInputFloat),
                                      sizeof(MyOutputFloat),
                                      block_all_particles,
                                      datatype_in_block_float,
                                      true,
                                      true,
                                      true,
                                      false,
                                      false};
    }

#if defined(SUBFIND_RESHUFFLE_CATALOGUE) && defined(SUBFIND)
    fields[IO_DMHSML] = {"DM Hsml", "DMHS", 1,    sizeof(float), sizeof(float), block_DM_nsel, datatype_in_block_float,
                         true,      true,   true, true,          false};
    fields[IO_DMDENSITY] = {
        "DM Density", "DMDE", 1,    sizeof(float), sizeof(float), block_DM_nsel, datatype_in_block_float,
        true,         true,   true, true,          false};
    fields[IO_DMVELDISP] = {"DM Velocity Dispersion", "DMVD", 1,    sizeof(float), sizeof(float), block_DM_nsel,
                            datatype_in_block_float,  true,   true, true,          true,          false};
#endif // defined(SUBFIND_RESHUFFLE_CATALOGUE) && defined(SUBFIND)

#ifdef LT_STELLAREVOLUTION
    fields[IO_Zs] = {"Mass of Metals",
                     "Zs  ",
                     LT_NMetP,
                     sizeof(MyInputFloat),
                     sizeof(MyOutputFloat),
                     block_gas_and_stars,
                     datatype_in_block_float,
                     true,
                     true,
                     true,
                     true,
                     false};
    fields[IO_iMass] = {"SSPInitialMass",
                        "iM  ",
                        1,
                        sizeof(MyInputFloat),
                        sizeof(MyOutputFloat),
                        block_stars,
                        datatype_in_block_float,
                        true,
                        true,
                        true,
                        true,
                        false};
#endif // LT_STELLAREVOLUTION

#ifdef LT_TRACK_CONTRIBUTES
    fields[IO_ContribII] = {"Contribution by SNII",
                            "CII ",
                            LT_NMetP,
                            sizeof(MyInputFloat),
                            sizeof(MyOutputFloat),
                            block_gas_and_stars,
                            datatype_in_block_float,
                            true,
                            true,
                            true,
                            true,
                            false};
    fields[IO_ContribIa] = {"Contribution by SNIa",
                            "CIa ",
                            LT_NMetP,
                            sizeof(MyInputFloat),
                            sizeof(MyOutputFloat),
                            block_gas_and_stars,
                            datatype_in_block_float,
                            true,
                            true,
                            true,
                            true,
                            false};
    fields[IO_ContribAGB] = {"Contribution by AGB",
                             "CAGB",
                             LT_NMetP,
                             sizeof(MyInputFloat),
                             sizeof(MyOutputFloat),
                             block_gas_and_stars,
                             datatype_in_block_float,
                             true,
                             true,
                             true,
                             true,
                             false};
    fields[IO_CONTRIB] = {"TrackContributes",
                          "TRCK",
                          1,
                          sizeof(Contrib),
                          sizeof(Contrib),
                          block_gas_and_stars,
                          true,
                          true,
                          true,
                          true,
                          false};
#endif // LT_TRACK_CONTRIBUTES

#ifdef LT_ZAGE
    fields[IO_ZAGE] = {"Metallicity-averaged time",
                       "ZAge",
                       1,
                       sizeof(MyInputFloat),
                       sizeof(MyOutputFloat),
                       block_gas_and_stars,
                       datatype_in_block_float,
                       true,
                       true,
                       true,
                       true,
                       false};
#endif // LT_ZAGE

#ifdef LT_ZAGE_LLV
    fields[IO_ZAGE_LLV] = {"long-living-Metallicity-averaged time",
                           "ZAlv",
                           1,
                           sizeof(MyInputFloat),
                           sizeof(MyOutputFloat),
                           block_gas_and_stars,
                           datatype_in_block_float,
                           true,
                           true,
                           true,
                           true,
                           false};
#endif // LT_ZAGE_LLV

#if defined(LT_STELLAREVOLUTION) && !defined(GM_MUPPI)
    fields[IO_CLDX] = {"CloudFraction",
                       "CLDX",
                       1,
                       sizeof(MyInputFloat),
                       sizeof(MyOutputFloat),
                       block_gas,
                       datatype_in_block_float,
                       true,
                       true,
                       true,
                       true,
                       false};
    fields[IO_HTEMP] = {"HotPhaseTemperature",
                        "HOTT",
                        1,
                        sizeof(MyInputFloat),
                        sizeof(MyOutputFloat),
                        block_gas,
                        datatype_in_block_float,
                        true,
                        true,
                        true,
                        true,
                        false};
#endif // defined(LT_STELLAREVOLUTION) && !defined(GM_MUPPI)

#ifdef COOLING
    fields[IO_TEMP] = {"Temperature",
                       "TEMP",
                       1,
                       sizeof(MyInputFloat),
                       sizeof(MyOutputFloat),
                       block_gas,
                       datatype_in_block_float,
                       true,
                       true,
                       true,
                       true,
                       false};
#endif // COOLING

#if defined(LT_SMOOTH_Z) && !defined(LT_SMOOTH_ALLMETALS)
    fields[IO_ZSMOOTH] = {"smoothed metallicity",
                          "ZsMT",
                          1,
                          sizeof(MyInputFloat),
                          sizeof(MyOutputFloat),
                          block_ZSMOOTH,
                          datatype_in_block_float,
                          true,
                          true,
                          true,
                          true,
                          false};
#endif // defined(LT_SMOOTH_Z) && !defined(LT_SMOOTH_ALLMETALS)

#ifdef LT_SMOOTH_ALLMETALS
    fields[IO_allZSMOOTH] = {"smoothed metallicities (all elements)",
                             "ZSMT",
                             LT_NMetP,
                             sizeof(MyInputFloat),
                             sizeof(MyOutputFloat),
                             block_ZSMOOTH,
                             datatype_in_block_float,
                             true,
                             true,
                             true,
                             true,
                             false};
#endif // LT_SMOOTH_ALLMETALS


#ifdef MV_OUTPUT_MMOL
    fields[IO_FB_M_MO] = {"Molecularmass",
                          "MMOL",
                          1,
                          sizeof(MyInputFloat),
                          sizeof(MyOutputFloat),
                          block_gas,
                          datatype_in_block_float,
                          true,
                          true,
                          true,
                          true,
                          false};
#endif // MV_OUTPUT_MMOL


#ifdef GT_DUST
    fields[IO_DVEL] = {"DeltaVelocities",
                       "DVEL",
                       3 * GT_NUM_DUST,
                       sizeof(MyInputFloat),
                       sizeof(MyOutputFloat),
                       block_all_particles,
                       datatype_in_block_float,
                       true,
                       true,
                       true,
                       true,
                       false};
    fields[IO_DFRC] = {"DustFractions",
                       "DFRC",
                       GT_NUM_DUST,
                       sizeof(MyInputFloat),
                       sizeof(MyOutputFloat),
                       block_gas,
                       datatype_in_block_float,
                       true,
                       true,
                       true,
                       true,
                       false};
    fields[IO_DRAG] = {"DragCoefficient",
                       "DRAG",
                       GT_NUM_DUST,
                       sizeof(MyInputFloat),
                       sizeof(MyOutputFloat),
                       block_gas,
                       datatype_in_block_float,
                       true,
                       true,
                       true,
                       true,
                       false};
#endif // GT_DUST

#ifdef GM_DMVELS_OUTPUT
    fields[IO_DM_AVE_VEL] = {"DMaveVel",
                             "DMVE",
                             1,
                             sizeof(MyInputFloat),
                             sizeof(MyOutputFloat),
                             block_gas,
                             datatype_in_block_float,
                             true,
                             true,
                             true,
                             true,
                             false};
    fields[IO_DM_AVE_SIG] = {"DMaveSig",
                             "DMSI",
                             1,
                             sizeof(MyInputFloat),
                             sizeof(MyOutputFloat),
                             block_gas,
                             datatype_in_block_float,
                             true,
                             true,
                             true,
                             true,
                             false};
    fields[IO_DM_NPART] = {"DMnumpart", "DMNP", 1,    sizeof(int), sizeof(int), block_gas, datatype_in_block_float,
                           true,        true,   true, true,        false};
#endif // GM_DMVELS_OUTPUT

#ifdef LMB_CR_PROTONS
    fields[IO_LMBCR_pNORM] = {"LMBCRpNormalization",
                              "CRpN",
                              LMB_CR_PROTONS,
                              sizeof(MyInputFloat),
                              sizeof(MyOutputFloat),
                              block_gas,
                              datatype_in_block_float,
                              true,
                              true,
                              true,
                              true,
                              false};
    fields[IO_LMBCR_pSLOPE] = {"LMBCRpSlope",
                               "CRpS",
                               LMB_CR_PROTONS,
                               sizeof(MyInputFloat),
                               sizeof(MyOutputFloat),
                               block_gas,
                               datatype_in_block_float,
                               true,
                               true,
                               true,
                               true,
                               false};
    fields[IO_LMBCR_pCUT] = {"LMBCRpCut",
                             "CRpC",
                             1,
                             sizeof(MyInputFloat),
                             sizeof(MyOutputFloat),
                             block_gas,
                             datatype_in_block_float,
                             true,
                             true,
                             true,
                             true,
                             false};
#endif // LMB_CR_PROTONS

#ifdef LMB_CR_ELECTRONS
    fields[IO_LMBCR_eNORM] = {"LMBCReNormalization",
                              "CReN",
                              LMB_CR_ELECTRONS,
                              sizeof(MyInputFloat),
                              sizeof(MyOutputFloat),
                              block_gas,
                              datatype_in_block_float,
                              true,
                              true,
                              true,
                              true,
                              false};
    fields[IO_LMBCR_eSLOPE] = {"LMBCReSlope",
                               "CReS",
                               LMB_CR_ELECTRONS,
                               sizeof(MyInputFloat),
                               sizeof(MyOutputFloat),
                               block_gas,
                               datatype_in_block_float,
                               true,
                               true,
                               true,
                               true,
                               false};
    fields[IO_LMBCR_eCUT] = {"LMBCReCut",
                             "CReC",
                             1,
                             sizeof(MyInputFloat),
                             sizeof(MyOutputFloat),
                             block_gas,
                             datatype_in_block_float,
                             true,
                             true,
                             true,
                             true,
                             false};
#endif // LMB_CR_ELECTRONS


#if defined(LMB_CR_PROTONS) && defined(LMB_CRs_CALCULATE_P_MAX)
    fields[IO_CRP_pmax] = {"LMBCRpPmax",
                           "CRpm",
                           1,
                           sizeof(MyInputFloat),
                           sizeof(MyOutputFloat),
                           block_gas,
                           datatype_in_block_float,
                           true,
                           true,
                           true,
                           true,
                           false};
#endif // defined(LMB_CR_PROTONS) && defined(LMB_CRs_CALCULATE_P_MAX)

#ifdef READ_RHO_OLD
    fields[IO_RHO_OLD] = {"DensityOld",
                          "RHOO",
                          1,
                          sizeof(MyInputFloat),
                          sizeof(MyOutputFloat),
                          block_gas,
                          datatype_in_block_float,
                          true,
                          true,
                          true,
                          true,
                          false};
#endif // READ_RHO_OLD

#ifdef LMB_CR_OUTPUT_SYNCHROTRON
    fields[IO_LMBCR_eSYNCHROTRON] = {"LMBCReSynchrotron",
                                     "SYNE",
                                     8,
                                     sizeof(MyInputFloat),
                                     sizeof(MyOutputFloat),
                                     block_gas,
                                     datatype_in_block_float,
                                     true,
                                     true,
                                     true,
                                     true,
                                     false};
#endif // LMB_CR_OUTPUT_SYNCHROTRON

#if defined(ADAPTGRAVSOFT) && defined(AGS_OUTPUTGRAVSOFT)
    fields[IO_AGS_SOFT] = {"AGS-Softening",
                           "AGSH",
                           1,
                           sizeof(MyInputFloat),
                           sizeof(MyOutputFloat),
                           block_all_particles,
                           datatype_in_block_float,
                           false,
                           false,
                           false,
                           true,
                           false};
#endif // defined(ADAPTGRAVSOFT) && defined(AGS_OUTPUTGRAVSOFT)

#if defined(ADAPTGRAVSOFT) && defined(AGS_OUTPUTGRAVNUMDENS)
    fields[IO_AGS_DENS] = {"AGS-Density",
                           "AGSD",
                           1,
                           sizeof(MyInputFloat),
                           sizeof(MyOutputFloat),
                           block_all_particles,
                           datatype_in_block_float,
                           false,
                           false,
                           false,
                           true,
                           false};
#endif // defined(ADAPTGRAVSOFT) && defined(AGS_OUTPUTGRAVNUMDENS)

#if defined(ADAPTGRAVSOFT) && defined(AGS_OUTPUTZETA)
    fields[IO_AGS_ZETA] = {"AGS-Zeta",
                           "AGSZ",
                           1,
                           sizeof(MyInputFloat),
                           sizeof(MyOutputFloat),
                           block_all_particles,
                           datatype_in_block_float,
                           false,
                           false,
                           false,
                           true,
                           false};
#endif // defined(ADAPTGRAVSOFT) && defined(AGS_OUTPUTZETA)

#if defined(ADAPTGRAVSOFT) && defined(AGS_OUTPUTOMEGA)
    fields[IO_AGS_OMEGA] = {"AGS-Omega",
                            "AGSO",
                            1,
                            sizeof(MyInputFloat),
                            sizeof(MyOutputFloat),
                            block_all_particles,
                            datatype_in_block_float,
                            false,
                            false,
                            false,
                            true,
                            false};
#endif // defined(ADAPTGRAVSOFT) && defined(AGS_OUTPUTOMEGA)

#if defined(ADAPTGRAVSOFT) && defined(AGS_OUTPUTCORR)
    fields[IO_AGS_CORR] = {"AGS-Correction",
                           "AGSC",
                           1,
                           sizeof(MyInputFloat),
                           sizeof(MyOutputFloat),
                           block_all_particles,
                           datatype_in_block_float,
                           false,
                           false,
                           false,
                           true,
                           false};
#endif // defined(ADAPTGRAVSOFT) && defined(AGS_OUTPUTCORR)

#if defined(ADAPTGRAVSOFT) && defined(AGS_OUTPUTNGBS)
    fields[IO_AGS_NGBS] = {"AGS-Neighbours",        "AGSN", 1,     sizeof(int), sizeof(int), block_all_particles,
                           datatype_in_block_float, false,  false, false,       true,        false};
#endif // defined(ADAPTGRAVSOFT) && defined(AGS_OUTPUTNGBS)

#if defined(CONDUCTION_INCLUDEMAGNETIC) || defined(AA_OUTPUT_GRADS)
    fields[IO_GRADENERGY] = {"Internal Energy Gradient",
                             "GRDU",
                             3,
                             sizeof(MyInputFloat),
                             sizeof(MyOutputFloat),
                             block_gas,
                             datatype_in_block_float,
                             true,
                             true,
                             true,
                             true,
                             false};
#endif // defined(CONDUCTION_INCLUDEMAGNETIC) || defined(AA_OUTPUT_GRADS)

#ifdef AA_OUTPUT_GRADS
    fields[IO_GRADPRESSURE] = {"Pressure Gradient",
                               "GRDP",
                               3,
                               sizeof(MyInputFloat),
                               sizeof(MyOutputFloat),
                               block_gas,
                               datatype_in_block_float,
                               true,
                               true,
                               true,
                               true,
                               false};
#endif // AA_OUTPUT_GRADS

#if (GADGET_HYDRO == HYDRO_MFM) && defined(MFM_OUTPUT_GRADIENTS)
    fields[IO_MFM_GRADIENTS_VX] = {"Gradient of x-velocity for MFM",
                                   "MGVX",
                                   mfm_numdims,
                                   sizeof(MyInputFloat),
                                   sizeof(MyOutputFloat),
                                   block_gas,
                                   datatype_in_block_float,
                                   true,
                                   true,
                                   true,
                                   true,
                                   false};
    fields[IO_MFM_GRADIENTS_VY] = {"Gradient of y-velocity for MFM",
                                   "MGVY",
                                   mfm_numdims > 1 ? mfm_numdims : 0,
                                   sizeof(MyInputFloat),
                                   sizeof(MyOutputFloat),
                                   block_gas,
                                   datatype_in_block_float,
                                   true,
                                   true,
                                   true,
                                   true,
                                   false};
    fields[IO_MFM_GRADIENTS_VZ] = {"Gradient of z-velocity for MFM",
                                   "MGVZ",
                                   mfm_numdims == 3 ? mfm_numdims : 0,
                                   sizeof(MyInputFloat),
                                   sizeof(MyOutputFloat),
                                   block_gas,
                                   datatype_in_block_float,
                                   true,
                                   true,
                                   true,
                                   true,
                                   false};
    fields[IO_MFM_GRADIENTS_RHO] = {"Gradient of density for MFM",
                                    "MGRH",
                                    mfm_numdims,
                                    sizeof(MyInputFloat),
                                    sizeof(MyOutputFloat),
                                    block_gas,
                                    datatype_in_block_float,
                                    true,
                                    true,
                                    true,
                                    true,
                                    false};
    fields[IO_MFM_GRADIENTS_U] = {"Gradient of specific internal energy for MFM",
                                  "MGU ",
                                  mfm_numdims,
                                  sizeof(MyInputFloat),
                                  sizeof(MyOutputFloat),
                                  block_gas,
                                  datatype_in_block_float,
                                  true,
                                  true,
                                  true,
                                  true,
                                  false};
#endif // (GADGET_HYDRO == HYDRO_MFM) && defined(MFM_OUTPUT_GRADIENTS)


#if (GADGET_HYDRO == HYDRO_MFM) && defined(MFM_OUTPUT_GRADIENTS) && defined(MFM_MAGNETIC) &&                           \
    defined(DIVBCLEANING_DEDNER)
    fields[IO_MFM_GRADIENTS_PHI] = {"Gradient of phi dedner cleaning function MFM",
                                    "MGPH",
                                    mfm_numdims,
                                    sizeof(MyInputFloat),
                                    sizeof(MyOutputFloat),
                                    block_gas,
                                    datatype_in_block_float,
                                    true,
                                    true,
                                    true,
                                    true,
                                    false};
#endif // (GADGET_HYDRO == HYDRO_MFM) && defined(MFM_OUTPUT_GRADIENTS) && defined(MFM_MAGNETIC) &&
       // defined(DIVBCLEANING_DEDNER)

#ifdef GL_CR_DUST
    fields[IO_DUSTL] = {"Mass of large dust grains",
                        "DSTL",
                        LT_NMetP,
                        sizeof(MyInputFloat),
                        sizeof(MyOutputFloat),
                        block_gas,
                        datatype_in_block_float,
                        true,
                        true,
                        true,
                        true,
                        false};
    fields[IO_DUSTS] = {"Mass of small dust grains",
                        "DSTS",
                        LT_NMetP,
                        sizeof(MyInputFloat),
                        sizeof(MyOutputFloat),
                        block_gas,
                        datatype_in_block_float,
                        true,
                        true,
                        true,
                        true,
                        false};
#endif // GL_CR_DUST

#ifdef GM_STARDENSITY
    fields[IO_RHOSTAR] = {"StellarDensityAroundStars",
                          "RHOS",
                          1,
                          sizeof(MyInputFloat),
                          sizeof(MyOutputFloat),
                          block_stars,
                          datatype_in_block_float,
                          false,
                          false,
                          false};
#ifndef GM_STARDENSITY_FIXAPERTURE
    fields[IO_HSMLSTAR] = {"SmoothingLengthForStellarDensity",
                           "SMTS",
                           1,
                           sizeof(MyInputFloat),
                           sizeof(MyOutputFloat),
                           block_stars,
                           datatype_in_block_float,
                           false,
                           false,
                           false,
                           true,
                           false};
#else // GM_STARDENSITY_FIXAPERTURE
    fields[IO_NEIGHSTAR] = {"NumberOfNeighboursForStellarDensity",
                            "NEIS",
                            1,
                            sizeof(MyInputFloat),
                            sizeof(MyOutputFloat),
                            block_stars,
                            datatype_in_block_float,
                            false,
                            false,
                            false,
                            true,
                            false};
#endif // GM_STARDENSITY_FIXAPERTURE
#endif // GM_STARDENSITY

    // apply the IC-reading policy special settings
    apply_read_from_ic_policy(fields
    );

    return fields;
};

static const IOFieldInfo *io_fields = nullptr;

/**
 * @brief Builds (or rebuilds) the io_fields metadata table for the given run.
 *
 * get_io_fields() is only correct once FdMuppi has actually been opened, which
 * happens at runtime inside read_file(). Call this from there instead of relying
 * on static initialization, which would run before FdMuppi exists.
 *
 * @param FdMuppi Auxiliary MUPPI file handle (nullptr if not present/enabled).
 */
void init_io_fields(
)
{
    io_fields = get_io_fields(
    );
}

/**
 * @brief Retrieve dataset name for a given I/O field.
 * @param blocknr Index of the I/O field.
 * @param buf Output buffer to store the dataset name (null-terminated).
 */
void get_dataset_name(enum iofields blocknr, char *buf)
{
    strcpy(buf, io_fields[blocknr].dataset_name ?: "default");
}

/**
 * @brief Get 4-byte label for a given I/O field.
 * @param blocknr Index of the I/O field.
 * @param label Output buffer (at least 4 bytes) for the label.
 */
void get_Tab_IO_Label(enum iofields blocknr, char *label)
{
    memcpy(label, io_fields[blocknr].label, 4);
}

/**
 * @brief Return number of values per block element.
 * @param blocknr Index of the I/O field.
 * @return Number of values per element.
 */
int get_values_per_blockelement(enum iofields blocknr)
{
    return io_fields[blocknr].values_per_element;
}

/**
 * @brief Compute byte size per block element.
 * @param blocknr Index of the I/O field.
 * @param mode Selects byte size: nonzero for input, zero for output.
 * @return Total bytes per element.
 */
int get_bytes_per_blockelement(enum iofields blocknr, int mode)
{
    return io_fields[blocknr].values_per_element *
           (mode ? (int)io_fields[blocknr].bytes_input : (int)io_fields[blocknr].bytes_output);
}

/**
 * @brief Check whether an I/O field is defined/present.
 * @param blocknr I/O field identifier.
 * @return Non-zero if the field is present, 0 otherwise.
 */
int blockpresent(enum iofields blocknr)
{
    return io_fields[blocknr].dataset_name != NULL;
}

/**
 * @brief Returns the number of particles in a given block.
 *
 * This function calculates the total number of particles in a specified block, using information from
 * the header structure. It also updates an array to indicate which particle types are present in the block.
 *
 * @param blocknr The block number (of type `enum iofields`) for which the particle count is to be determined.
 * @return The total number of particles present in the specified block.
 */
unsigned int get_particles_in_block(enum iofields blocknr, unsigned int *typelist)
{
    return io_fields[blocknr].particles_in_block(typelist);
}

/**
 * @brief Determines the data type used for a specified block.
 *
 * This function returns an integer representing the data type of the elements within a block,
 * based on the block type provided. The function uses a switch statement to map each block type
 * to a specific data type identifier.
 *
 * @param blocknr The block number (of type `enum iofields`) for which the data type is to be determined.
 *
 * @return An integer indicating the data type of the block elements:
 *         - 0 for `native int`
 *         - 1 for `native MyOutputFloat`
 *         - 2 for `native long long` (when `LONGIDS` is defined)
 */
int get_datatype_in_block(enum iofields blocknr)
{
    return io_fields[blocknr].datatype_in_block();
}

/**
 * @brief Determines whether a given I/O field should be read or writtenunder a specific restart mode.
 *
 * This function inspects the configuration flags of the specified data block
 * (`blocknr`) and returns whether the block should participate in I/O for the
 * provided `RestartFlag`. The mapping is:
 *
 * * RestartFlag == 0 : Read from initial conditions (`read_from_ic`)
 * * RestartFlag == 1 : Read from restart file (`read_restartfile`)
 * * RestartFlag == 2,3 : Read from restart snapshot (`read_restart_snap`)
 * * RestartFlag == 4 : Write to snapshot (`write_to_snap`)
 * * RestartFlag == 5,6 : Read from spectra input (`read_from_spectra`)
 *
 * Any unsupported `RestartFlag` value triggers a fatal error via ENDRUN_ANY.
 *
 * @param blocknr     Index of the I/O field (enum iofields).
 * @param RestartFlag Mode selector controlling I/O behavior.
 *
 * @return int Returns 1 if the block should be processed for the given mode, otherwise 0.
 */
int check_if_read_or_write_block(enum iofields blocknr, int RestartFlag)
{
    switch (RestartFlag)
    {
    case 0:
        return io_fields[blocknr].read_from_ic ? 1 : 0;
    case 1:
        return io_fields[blocknr].read_restartfile ? 1 : 0;
    case 2:
    case 3:
        return io_fields[blocknr].read_restart_snap ? 1 : 0;
    case 4:
        return io_fields[blocknr].write_to_snap ? 1 : 0;
    case 5:
    case 6:
        return io_fields[blocknr].read_from_spectra ? 1 : 0;
    default:
        ENDRUN("Restartflag has unsupported value of %d.", RestartFlag);
        return 0;
    }
}

} // namespace io_matrix
