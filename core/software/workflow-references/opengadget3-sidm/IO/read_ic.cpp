/**
 * @file
 *
 * This file is part of the N-body/SPH code OpenGadget3.
 *
 * @brief This file contains functions for the reading of ics.
 *
 * @author OpenGadget3-development team, in particular Klaus Dolag and Volker Springel.
 *
 * @copyright Copyright (c) 2026, OpenGadget3-development team lead by Klaus Dolag and Stefano Borgani
 * (see change logs). All rights reserved.
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License v3 or later.
 * See <https://www.gnu.org/licenses/>.
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <errno.h>
#include <unistd.h>
#include <string>
#include <iostream>

#include "../CodeBase/allvars.hpp"
#include "../CodeBase/proto.hpp"
#include "io_matrix.hpp"
#include "io.hpp"


int swap_file = 8;

#ifdef LT_STELLAREVOLUTION
double time_age, this_age;
int Zs_present = 0, ti_step, IMFi;

#ifdef LT_ZAGE
int ZAge_present = 0;
#endif // LT_ZAGE
#ifdef LT_ZAGE_LLV
int ZAge_llv_present = 0;
#endif // LT_ZAGE_LLV
#endif // LT_STELLAREVOLUTION

int count = 0;

/**
 * @brief This function returns the name of the node
 *
 * Return name of node or 'unknown' if it cant be determined.
 */
static std::string get_hostname()
{
    char host[256] = {};
    if (::gethostname(host, sizeof(host) - 1) != 0)
    {
        return "unknown";
    }
    return host;
}

/**
 * @brief This function reads the header of the IC file.
 *
 * This function reads initial conditions that are in the default file format
 * of Gadget, i.e. snapshot files can be used as input files.  However, when a
 * snapshot file is used as input, not all the information in the header is
 * used: THE STARTING TIME NEEDS TO BE SET IN THE PARAMETERFILE.
 * Alternatively, the code can be started with restartflag==2, then snapshots
 * from the code can be used as initial conditions-files without having to
 * change the parameterfile.  For gas particles, only the internal energy is
 * read, the density and mean molecular weight will be recomputed by the code.
 * When InitGasTemp>0 is given, the gas temperature will be initialzed to this
 * value assuming a mean colecular weight either corresponding to complete
 * neutrality, or full ionization.
 *
 * @param fname the filename of the IC file
 */
void read_ic(char *fname)
{
    int filenr, masterTask, lastTask, groupMaster;
    char buf[500];

    TIMER_PUSH(1, "READIC");

#ifdef RESCALEVINI
    if (RestartFlag == 0)
    {
        VERBOSE(2, stdout, "\nRescaling v_ini !  ");
    }
#endif // RESCALEVINI

    NumPart = 0;
    N_gas = 0;
    All.TotNumPart = 0;
    N_stars = 0;
    N_BHs = 0;

    int num_files = find_files(fname);
    int rest_files = num_files;

    VERBOSE(2, "num_files = %i ", num_files);

    while (rest_files > NTask)
    {
        sprintf(buf, "%s.%d", fname, ThisTask + (rest_files - NTask));
        if (All.ICFormat == 3)
        {
            sprintf(buf, "%s.%d.hdf5", fname, ThisTask + (rest_files - NTask));
        }

        int ngroups = NTask / All.NumFilesWrittenInParallel;
        if ((NTask % All.NumFilesWrittenInParallel))
        {
            ngroups++;
        }
        groupMaster = (ThisTask / ngroups) * ngroups;

        for (int gr = 0; gr < ngroups; gr++)
        {
            if (ThisTask == (groupMaster + gr)) /* ok, it's this processor's turn */
            {
                read_file(buf, ThisTask, ThisTask);
            }
            MPI_Barrier(MYMPI_COMM_WORLD);
        }
        rest_files -= NTask;
    }

    if (rest_files > 0)
    {
        distribute_file(rest_files, 0, 0, NTask - 1, &filenr, &masterTask, &lastTask);

        if (num_files > 1)
        {
            sprintf(buf, "%s.%d", fname, filenr);
            if (All.ICFormat == 3)
            {
                sprintf(buf, "%s.%d.hdf5", fname, filenr);
            }
        }
        else
        {
            sprintf(buf, "%s", fname);
            if (All.ICFormat == 3)
            {
                sprintf(buf, "%s.hdf5", fname);
            }
        }

        int ngroups = rest_files / All.NumFilesWrittenInParallel;
        if ((rest_files % All.NumFilesWrittenInParallel))
        {
            ngroups++;
        }

        for (int gr = 0; gr < ngroups; gr++)
        {
            if ((filenr / num_files) == gr) /* ok, it's this processor's turn */
            {
                read_file(buf, masterTask, lastTask);
            }
            MPI_Barrier(MYMPI_COMM_WORLD);
        }
    }

    myfree(CommBuffer);

    // Re arrange particles (e.g. bring gas to the front)
    // First step: Find the last gas partilce
    int index_gas_to_be = 0;
    while (index_gas_to_be < N_gas)
    {
        if (P[index_gas_to_be].Type != 0)
        {
            break;
        }
        index_gas_to_be++;
    }
    // Second step: Find the next gas particle which has to be moved
    int index_gas_to_move = index_gas_to_be;
    while (index_gas_to_move < NumPart)
    {
        if (P[index_gas_to_move].Type == 0)
        {
            break;
        }
        index_gas_to_move++;
    }

    struct particle_data psave;

    // Third step: Move all gas particles tio fromt
    while (index_gas_to_be < N_gas && index_gas_to_move < NumPart)
    {
        // repair pointers for extra data for stars and BHs
#ifdef SFR
        if (P[index_gas_to_be].Type == 4)
        {
            MetP[P[index_gas_to_be].pt.MetID].PID = index_gas_to_move;
        }
#endif // SFR
#ifdef BLACK_HOLES
        if (P[index_gas_to_be].Type == 5)
        {
            BHP[P[index_gas_to_be].pt.BHID].PID = index_gas_to_move;
        }
#endif // BLACK_HOLES
       // swap particles
        psave = P[index_gas_to_be];
        P[index_gas_to_be] = P[index_gas_to_move];
        P[index_gas_to_move] = psave;

        // find next free spot for gas particles
        while (index_gas_to_be < N_gas)
        {
            if (P[index_gas_to_be].Type != 0)
            {
                break;
            }
            index_gas_to_be++;
        }
        // find next gas particle to be move
        while (index_gas_to_move < NumPart)
        {
            if (P[index_gas_to_move].Type == 0)
            {
                break;
            }
            index_gas_to_move++;
        }
    }

    // here we do a sanity check if there unexpected gas particles left
    while (index_gas_to_move < NumPart)
    {
        if (P[index_gas_to_move].Type == 0)
        {
            VERBOSE_ALL(0, "Task %d: SOMETHING WRONG ! Seems to be gas at %d, N_gas=%d, N_tot=%d, type=%d count=%d ",
                        ThisTask, index_gas_to_move, N_gas, NumPart, P[index_gas_to_move].Type, count);
            for (int i = 0; i < N_gas; i++)
            {
                ENDRUN_IF(P[i].Type != 0, "Task %d: EVEN MORE WRONG ! Seems to be a non gas at %d, N_gas=%d type=%d  ",
                          ThisTask, i, N_gas, P[i].Type);
            }
            ENDRUN_MAIN("Error: unexpected gas particles left.");
        }
        index_gas_to_move++;
    }

    if (header.flag_ic_info != FLAG_SECOND_ORDER_ICS)
    {
        /* this makes sure that masses are initialized in the case that the mass-block
           is empty for this particle type */
        for (int i = 0; i < NumPart; i++)
        {
            if (All.MassTable[P[i].Type] != 0)
            {
                P[i].Mass = All.MassTable[P[i].Type];
            }
        }
    }

#ifdef GENERATE_GAS_IN_ICS
    int count = 0;

    if (RestartFlag == 0)
    {
        header.flag_entropy_instead_u = 0;

        check_particles();

        for (int i = 0; i < NumPart; i++)
        {
            if (P[i].Type == 1)
            {
                count++;
            }

            ENDRUN_IF(
                P[i].Type == 0,
                "Task %d: ICs already contain gas particles but flag GENERATE_GAS_IN_ICS is set. We better stop. ",
                ThisTask);
        }

        if (count != NumPart)
        {
            VERBOSE_ALL(2, "Task=%d not only type 1 partcles %d/%d !! ", ThisTask, count, NumPart);
        }

        ENDRUN_IF(NumPart + count > All.MaxPart, "Task=%d ends up getting more particles (%d) than allowed (%d) ",
                  ThisTask, NumPart + count, All.MaxPart);

        ENDRUN_IF(N_gas + count > All.MaxPartSph, "Task=%d ends up getting more SPH particles (%d) than allowed (%d) ",
                  ThisTask, N_gas + count, All.MaxPartSph);

        double fac = All.OmegaBaryon / All.Omega0;
        double rho = All.Omega0 * 3 * All.Hubble * All.Hubble / (8 * M_PI * All.GravityConstantInternal);

        for (int i = 0; i < count; i++)
        {
            P[i + count] = P[i];
            P[i + count].ID += 300000000000;
            P[i].Type = 0;

            double d = pow(P[i].Mass / rho, 1.0 / 3);
            double a = 0.5 * All.OmegaBaryon / All.Omega0 * d;
            double b = 0.5 * (All.Omega0 - All.OmegaBaryon) / All.Omega0 * d;

            P[i].Mass = All.MassTable[1] * fac;
            P[i + count].Mass = All.MassTable[1] * (1 - fac);
            P[i + count].Pos[0] += a;
            P[i + count].Pos[1] += a;
            P[i + count].Pos[2] += a;
            P[i].Pos[0] -= b;
            P[i].Pos[1] -= b;
            P[i].Pos[2] -= b;
        }

        NumPart += count;
        N_gas += count;

        All.MassTable[0] = 0;
        All.MassTable[1] *= (1 - fac);

#ifdef PEDANTIC_CHECK
        if (ThisTask == 0)
        {
            VERBOSE(2, " Checking Particles after adding gas %d/%d/%d ", NumPart, N_gas, NumPart - N_gas);
            int count_gas = 0, count_dm = 0;
            for (int i = 0; i < NumPart; i++)
            {
                if (P[i].Type == 0 && i >= N_gas)
                {
                    VERBOSE(2, "  Gas paricle at %d >= %d !! ", i, N_gas);
                }
                if (P[i].Type == 0 && P[i].Mass != P[0].Mass)
                {
                    VERBOSE(2, "  Gas mass wrong (%g/%g) !! ", P[i].Mass, P[0].Mass);
                }
                if (P[i].Type == 1 && P[i].Mass != P[N_gas].Mass)
                {
                    VERBOSE(2, "  DM mass wrong (%g/%g) !! ", P[i].Mass, P[N_gas].Mass);
                }
                if (P[i].Type == 0)
                {
                    count_gas++;
                }
                if (P[i].Type == 1)
                {
                    count_dm++;
                }
            }
            VERBOSE(2, " Checking Particles after adding gas %d/%d/%d done ... ", NumPart, count_gas, count_dm);
        }
#endif // PEDANTIC_CHECK
    }
#endif // GENERATE_GAS_IN_ICS

#ifdef BLACK_HOLES
    if (RestartFlag == 0)
    {
        All.MassTable[5] = 0;
    }
#endif // BLACK_HOLES

#ifdef SFR
    if (RestartFlag == 0)
    {
        if (All.MassTable[4] == 0 && All.MassTable[0] > 0)
        {
            All.MassTable[0] = 0;
            All.MassTable[4] = 0;
        }
    }
#endif // SFR

#ifdef LT_STELLAREVOLUTION
    All.Time_Age = get_age(All.Time);

    /* note: initialization of timings for stellar evolution occurs in init.c */
    for (int i = 0; i < NumPart; i++)
    {
        if (P[i].Type == 0)
        {
            if (!Zs_present)
            {
                //              memset(SphP[i].Metals, 0, sizeof(float) * LT_NMetP);
                SphP[i].Metals[Hel] = P[i].Mass * (1 - HYDROGEN_MASSFRAC);
            }

#ifdef GL_CR_DUST
            if (RestartFlag == 0)
            {
                memset(SphP[i].DustL, 0, sizeof(float) * LT_NMetP);
                memset(SphP[i].DustS, 0, sizeof(float) * LT_NMetP);
            }

            SphP[i].numSnIa = 0.; /* XXXX may be a problem when restarting from snapshots */
            SphP[i].numSnII = 0.;
#endif // GL_CR_DUST
        }
        else if (P[i].Type == 4)
        {
            // MetP[N_star_idx].PID = i;
            // P[i].pt.MetID = N_star_idx;

            if (!Zs_present)
            {
                memset(MPP(i).Metals, 0, sizeof(float) * LT_NMetP);
                MPP(i).Metals[Hel] = P[i].Mass * (1 - HYDROGEN_MASSFRAC);
            }
        }
    }

    if (!Zs_present)
    /* metal array is not present; we initialize Helium to the cosmic fraction */
    {
        VERBOSE(2, "Helium Fraction initialized to the Cosmic Value %8.5g ", 1 - HYDROGEN_MASSFRAC);
    }

#ifdef PEDANTIC_CHECK
    if (ThisTask == 0)
    {
        VERBOSE(2, " Checking Particles after setting metals %d/%d/%d ", NumPart, N_gas, NumPart - N_gas);
        int count_gas = 0, count_dm = 0;
        for (int i = 0; i < NumPart; i++)
        {
            if (P[i].Type == 0 && i >= N_gas)
            {
                VERBOSE(2, "  Gas paricle at %d >= %d !! ", i, N_gas);
            }
            if (P[i].Type == 0 && P[i].Mass != P[0].Mass)
            {
                VERBOSE(2, "  Gas mass wrong (%g/%g) !! ", P[i].Mass, P[0].Mass);
            }
            if (P[i].Type == 1 && P[i].Mass != P[N_gas].Mass)
            {
                VERBOSE(2, "  Gas mass wrong (%g/%g) !! ", P[i].Mass, P[N_gas].Mass);
            }
            if (P[i].Type == 0)
            {
                count_gas++;
            }
            if (P[i].Type == 1)
            {
                count_dm++;
            }
        }
        VERBOSE(2, " Checking Particles after setting metals %d/%d/%d done ... ", NumPart, count_gas, count_dm);
    }
#endif // PEDANTIC_CHECK
#endif // LT_STELLAREVOLUTION

    double u_init = (1.0 / GAMMA_MINUS1) * (BOLTZMANN / PROTONMASS) * max(All.InitGasTemp, 1e-20);
    u_init *= All.UnitMass_in_g / All.UnitEnergy_in_cgs; /* unit conversion */

    double molecular_weight;
    if (All.InitGasTemp > 1.0e4) /* assuming FULL ionization */
    {
        molecular_weight = 4 / (8 - 5 * (1 - HYDROGEN_MASSFRAC));
    }
    else /* assuming NEUTRAL GAS */
    {
        molecular_weight = 4 / (1 + 3 * HYDROGEN_MASSFRAC);
    }

    u_init /= molecular_weight;

    All.InitGasU = u_init;

    if (RestartFlag == 0)
    {
        for (int i = 0; i < N_gas; i++)
        {
            if (SphP[i].Entropy == 0)
            {
                SphP[i].Entropy = All.InitGasU;
            }
        }
    }

    //  for(i = 0; i < N_gas; i++)
    //    SphP[i].Entropy = max(All.MinEgySpec, SphP[i].Entropy);

#ifdef PEDANTIC_CHECK
    if (ThisTask == 0)
    {
        VERBOSE(2, " Checking Particles after setting gas entropy %d/%d/%d ", NumPart, N_gas, NumPart - N_gas);
        int count_gas = 0, count_dm = 0;
        for (int i = 0; i < NumPart; i++)
        {
            if (P[i].Type == 0 && i >= N_gas)
            {
                VERBOSE(2, "  Gas paricle at %d >= %d !! ", i, N_gas);
            }
            if (P[i].Type == 0 && P[i].Mass != P[0].Mass)
            {
                VERBOSE(2, "  Gas mass wrong (%g/%g) !! ", P[i].Mass, P[0].Mass);
            }
            if (P[i].Type == 1 && P[i].Mass != P[N_gas].Mass)
            {
                VERBOSE(2, "  Gas mass wrong (%g/%g) !! ", P[i].Mass, P[N_gas].Mass);
            }
            if (P[i].Type == 0)
            {
                count_gas++;
            }
            if (P[i].Type == 1)
            {
                count_dm++;
            }
        }
        VERBOSE(2, " Checking Particles after setting gas entropy %d/%d/%d done ... ", NumPart, count_gas, count_dm);
    }
#endif // PEDANTIC_CHECK

    MPI_Barrier(MYMPI_COMM_WORLD);

    VERBOSE(2, "reading done. ");
    VERBOSE(2, "Total number of particles :  %llu/%d ", All.TotNumPart, NumPart);
    VERBOSE(2, "Task %d: Total number of particles : %d ", ThisTask, NumPart);
    VERBOSE(2, "Task %d: Total number of Gas part. : %d ", ThisTask, N_gas);
    VERBOSE(2, "Task %d: Total number of Star part.: %d ", ThisTask, N_stars);
    VERBOSE(2, "Task %d: Total number of BH part.  : %d \n", ThisTask, N_BHs);

#ifdef PEDANTIC_CHECK
    if (ThisTask == 0)
    {
        VERBOSE(2, " Checking Particles after reading %d/%d/%d ", NumPart, N_gas, NumPart - N_gas);
        int count_gas = 0, count_dm = 0;
        for (int i = 0; i < NumPart; i++)
        {
            if (P[i].Type == 0 && i >= N_gas)
            {
                VERBOSE(2, "  Gas paricle at %d >= %d !! ", i, N_gas);
            }
            if (P[i].Type == 0 && P[i].Mass != P[0].Mass)
            {
                VERBOSE(2, "  Gas mass wrong (%g/%g) !! ", P[i].Mass, P[0].Mass);
            }
            if (P[i].Type == 1 && P[i].Mass != P[N_gas].Mass)
            {
                VERBOSE(2, "  Gas mass wrong (%g/%g) !! ", P[i].Mass, P[N_gas].Mass);
            }
            if (P[i].Type == 0)
            {
                count_gas++;
            }
            if (P[i].Type == 1)
            {
                count_dm++;
            }
        }
        VERBOSE(2, " Checking Particles after reading %d/%d/%d done ... ", NumPart, count_gas, count_dm);
    }
#endif // PEDANTIC_CHECK

    TIMER_POP(1, "READIC");
}

/**
 * @brief Empty the read buffer.
 *
 * This function reads out the buffer that was filled with particle data.
 *
 * @param blocknr the block number
 * @param offset the offset
 * @param pc the particle count
 * @param type the particle type
 * @param offset_gas the offset for gas particles
 * @param offset_stars the offset for star particles
 * @param offset_bhs the offset for BH particles
 */
void empty_read_buffer(enum iofields blocknr, int offset, int pc, int type, int offset_gas, int offset_stars,
                       int offset_bhs)
{
    MyInputFloat *fp;
    MyIDType *ip;

    int vt, vpb;
    char *cp;

    fp = (MyInputFloat *)CommBuffer;
#if (defined(SUBFIND_RESHUFFLE_CATALOGUE) && defined(SUBFIND)) || defined(GM_MUPPI)
    float *fp_single = (float *)CommBuffer;
#endif // (defined(SUBFIND_RESHUFFLE_CATALOGUE) && defined(SUBFIND)) || defined(GM_MUPPI)
    ip = (MyIDType *)CommBuffer;


    if (blocknr != IO_DMHSML && blocknr != IO_DMDENSITY && blocknr != IO_DMVELDISP)
    {
        cp = (char *)CommBuffer;
        vt = io_matrix::get_datatype_in_block(blocknr);
        vpb = io_matrix::get_values_per_blockelement(blocknr);
        if (vt == 2)
        {
            swap_Nbyte(cp, pc * vpb, 8);
        }
        else
        {
#ifdef INPUT_IN_DOUBLEPRECISION
            if (vt == 1)
            {
                swap_Nbyte(cp, pc * vpb, 8);
            }
            else
#endif // INPUT_IN_DOUBLEPRECISION
                swap_Nbyte(cp, pc * vpb, 4);
        }
    }

    switch (blocknr)
    {
        // Note that we use the reading of the postion array to also initialize the type
        // as well as setting the pointers for the extra data structures for stars and BHs
    case IO_POS: /* positions */
        for (int n = 0; n < pc; n++)
        {
            for (int k = 0; k < 3; k++)
            {
                P[offset + n].Pos[k] = *fp++;
            }
        }

#if defined(DS_SHIFT_BOX) && defined(PERIODIC)
        for (int n = 0; n < pc; n++)
        {
            for (int k = 0; k < 3; k++)
            {
                P[offset + n].Pos[k] += All.BoxSize / 2;
            }
        }
#endif // defined(DS_SHIFT_BOX) && defined(PERIODIC)

        for (int n = 0; n < pc; n++)
        {
            P[offset + n].Type = type; /* initialize type here as well */
            if (type == 0)
            {
                count++;
            }

#ifdef SFR
            if (P[offset + n].Type == 4) /* initialize pointer to extra informations for stars */
            {
                MetP[offset_stars + n].PID = offset + n;
                P[offset + n].pt.MetID = offset_stars + n;
            }
#endif // SFR
#ifdef BLACK_HOLES
            if (P[offset + n].Type == 5) /* initialize pointer to extra informations for BHs */
            {
                BHP[offset_bhs + n].PID = offset + n;
                P[offset + n].pt.BHID = offset_bhs + n;
            }
#endif // BLACK_HOLES
        }
        break;

    case IO_VEL: /* velocities */
        for (int n = 0; n < pc; n++)
        {
            for (int k = 0; k < 3; k++)
#ifdef RESCALEVINI
                /* scaling v to use same IC's for different cosmologies */
                if (RestartFlag == 0)
                {
                    P[offset + n].Vel[k] = (*fp++) * All.VelIniScale;
                }
                else
                {
                    P[offset + n].Vel[k] = *fp++;
                }
#else // RESCALEVINI
                P[offset + n].Vel[k] = *fp++;
#endif // RESCALEVINI
        }
        break;

    case IO_ID: /* particle ID */
        for (int n = 0; n < pc; n++)
        {
            P[offset + n].ID = *ip++;
        }
        break;

    case IO_MASS: /* particle mass */
        for (int n = 0; n < pc; n++)
        {
            P[offset + n].Mass = *fp++;
        }
        break;

    case IO_SECONDORDERMASS:
        for (int n = 0; n < pc; n++)
        {
            P[offset + n].OldAcc =
                P[offset + n].Mass; /* use this to temporarily store the masses in the 2plt IC case */
            P[offset + n].Mass = *fp++;
        }
        break;

    case IO_U: /* temperature */
        for (int n = 0; n < pc; n++)
        {
#ifdef JD_RELAX_CLUSTERS
            SphP[offset_gas + n].U = *fp;
#endif // JD_RELAX_CLUSTERS
            SphP[offset_gas + n].Entropy = *fp++;
        }
        break;

    case IO_RHO: /* density */
        for (int n = 0; n < pc; n++)
        {
            SphP[offset_gas + n].Density = *fp++;
        }
        break;

    case IO_NE: /* electron abundance */
#if defined(COOLING) || defined(CHEMISTRY) || defined(UM_CHEMISTRY)
        for (int n = 0; n < pc; n++)
        {
            SphP[offset_gas + n].elec = *fp++;
        }
#endif // defined(COOLING) || defined(CHEMISTRY) || defined(UM_CHEMISTRY)
        break;

#if defined(CHEMISTRY) || defined(UM_CHEMISTRY)
    case IO_NH: /* neutral hydrogen abundance */
        for (int n = 0; n < pc; n++)
        {
            SphP[offset_gas + n].HI = *fp++;
        }
        break;

    case IO_HII: /* ionized hydrogen abundance */
        for (int n = 0; n < pc; n++)
        {
            SphP[offset_gas + n].HII = *fp++;
        }
        break;

    case IO_HeI: /* neutral Helium */
        for (int n = 0; n < pc; n++)
        {
            SphP[offset_gas + n].HeI = *fp++;
        }
        break;

    case IO_HeII: /* ionized Heluum */
        for (int n = 0; n < pc; n++)
        {
            SphP[offset_gas + n].HeII = *fp++;
        }
        break;

    case IO_HeIII: /* double ionised Helium */
        for (int n = 0; n < pc; n++)
        {
            SphP[offset_gas + n].HeIII = *fp++;
        }
        break;

    case IO_H2I: /* H2 molecule */
        for (int n = 0; n < pc; n++)
        {
            SphP[offset_gas + n].H2I = *fp++;
        }
        break;

    case IO_H2II: /* ionised H2 molecule */
        for (int n = 0; n < pc; n++)
        {
            SphP[offset_gas + n].H2II = *fp++;
        }
        break;

    case IO_HM: /* H minus */
        for (int n = 0; n < pc; n++)
        {
            SphP[offset_gas + n].HM = *fp++;
        }
        break;

    case IO_HeHII: /* HeH+ */
#ifdef UM_CHEMISTRY
        for (int n = 0; n < pc; n++)
        {
            SphP[offset_gas + n].HeHII = *fp++;
        }
#endif // UM_CHEMISTRY
        break;

    case IO_HD: /* HD */
#ifdef UM_CHEMISTRY
        for (int n = 0; n < pc; n++)
        {
            SphP[offset_gas + n].HD = *fp++;
        }
#endif // UM_CHEMISTRY
        break;

    case IO_DI: /* D */
#ifdef UM_CHEMISTRY
        for (int n = 0; n < pc; n++)
        {
            SphP[offset_gas + n].DI = *fp++;
        }
#endif // UM_CHEMISTRY
        break;

    case IO_DII: /* D plus */
#ifdef UM_CHEMISTRY
        for (int n = 0; n < pc; n++)
        {
            SphP[offset_gas + n].DII = *fp++;
        }
#endif // UM_CHEMISTRY
        break;

#else //  defined(CHEMISTRY) || defined(UM_CHEMISTRY)
    case IO_NH: /* neutral hydrogen abundance */
    case IO_HII: /* ionized hydrogen abundance */
    case IO_HeI: /* neutral Helium */
    case IO_HeII: /* ionized Heluum */
    case IO_HeIII: /* double ionised Helium */
    case IO_H2I: /* H2 molecule */
    case IO_H2II: /* ionised H2 molecule */
    case IO_HM: /* H minus */
    case IO_HeHII: /* HeH+ */
    case IO_HD: /* HD */
    case IO_DI: /* D */
    case IO_DII: /* D plus  */
        break;
#endif //  defined(CHEMISTRY) || defined(UM_CHEMISTRY)

    case IO_HSML: /* SPH smoothing length */
        for (int n = 0; n < pc; n++)
        {
            P[offset + n].Hsml = *fp++;
        }
        break;

    case IO_DELAYTIME:
#ifdef WINDS
        for (int n = 0; n < pc; n++)
        {
            SphP[offset_gas + n].DelayTime = *fp++;
        }
#endif // WINDS
        break;

    case IO_DECOUPLE_TIME_AGN:
#ifdef PB_BH_KIN_FEED_DECOUPLE_WIND
        for (int n = 0; n < pc; n++)
        {
            SphP[offset + n].DecoupleTime_AGN = *fp++;
        }
#endif // PB_BH_KIN_FEED_DECOUPLE_WIND
        break;

    case IO_AGE: /* Age of stars */
#ifdef SFR
        for (int n = 0; n < pc; n++)
        {
            if (P[offset + n].Type == 4)
            {
                MPP(offset + n).StellarAge = *fp++;
            }
#ifdef BLACK_HOLES
            if (P[offset + n].Type == 5)
            {
                BPP(offset + n).StellarAge = *fp++;
            }
#endif // BLACK_HOLES
        }
#endif // SFR
        break;

    case IO_Z: /* Gas and star metallicity */
#if defined(SFR) && !defined(LT_STELLAREVOLUTION)
        for (int n = 0; n < pc; n++)
        {
            MPP(offset + n).Metallicity = *fp++;
        }
#endif // SFR && !LT_STELLAREVOLUTION
        break;

    case IO_VRMS: /* Turbulence on kernel scale */
#ifdef JD_VTURB
        for (int n = 0; n < pc; n++)
        {
            SphP[offset_gas + n].Vrms = *fp++;
        }
#endif // JD_VTURB
        break;

    case IO_VBULK:
#ifdef JD_VTURB
        for (int n = 0; n < pc; n++)
        {
            for (int k = 0; k < 3; k++)
            {
                SphP[offset_gas + n].Vbulk[k] = *fp++;
            }
        }
#endif // JD_VTURB
        break;

    case IO_VTAN:
#ifdef JD_DECOMPOSE_VTURB
        for (int n = 0; n < pc; n++)
        {
            SphP[offset_gas + n].Vtan = *fp++;
        }
#endif // JD_DECOMPOSE_VTURB
        break;

    case IO_VRAD:
#ifdef JD_DECOMPOSE_VTURB
        for (int n = 0; n < pc; n++)
        {
            SphP[offset_gas + n].Vrad = *fp++;
        }
#endif // JD_DECOMPOSE_VTURB
        break;

    case IO_VDIV:
#ifdef JD_VTURB
        for (int n = 0; n < pc; n++)
        {
            SphP[offset_gas + n].DivVel = *fp++;
        }
#endif // JD_VTURB
        break;

    case IO_VROT:
#ifdef JD_VTURB
        for (int n = 0; n < pc; n++)
        {
            SphP[offset_gas + n].CurlVel = *fp++;
        }
#endif // JD_VTURB
        break;

    case IO_TRUENGB:
#ifdef JD_VTURB
        for (int n = 0; n < pc; n++)
        {
            P[offset + n].TrueNGB = *fp++;
        }
#endif // JD_VTURB
        break;

    case IO_DPP:
#ifdef JD_DPP
        for (int n = 0; n < pc; n++)
        {
            SphP[offset_gas + n].Dpp = *fp++;
        }
#endif // JD_DPP
        break;

    case IO_BFLD: /* Magnetic field */
#ifdef MAGNETIC
        for (int n = 0; n < pc; n++)
        {
            for (int k = 0; k < 3; k++)
            {
                SphP[offset_gas + n].BPred[k] = *fp++;
            }
            SphP[offset_gas + n].divB = 0;
#ifdef MAGNETICZERO
            for (int k = 0; k < 3; k++)
            {
                SphP[offset_gas + n].BPred[k] = 0.0;
            }
#endif // MAGNETICZERO
#ifdef DIVBCLEANING_TRICCO
            SphP[offset_gas + n].Psi = 0;
            SphP[offset_gas + n].PsiPred = 0;
#endif // DIVBCLEANING_TRICCO
#ifdef DIVBCLEANING_DEDNER
            SphP[offset_gas + n].Phi = 0;
            SphP[offset_gas + n].PhiPred = 0;
#endif // DIVBCLEANING_DEDNER
        }
#endif // MAGNETIC
        break;

    case IO_LMBCR_pNORM:
#if (defined(LMB_SPECTRAL_CRs_FROM_ICS) && defined(LMB_CR_PROTONS))
        for (int n = 0; n < pc; n++)
            for (int k = 0; k < LMB_CR_PROTONS; k++)
            {
                SphP[offset_gas + n].CRpNorm[k] = *fp++;
                SphP[offset_gas + n].CRpNorm[k] = pow(10.0, SphP[offset_gas + n].CRpNorm[k]);
            }
#endif // (defined(LMB_SPECTRAL_CRs_FROM_ICS) && defined(LMB_CR_PROTONS))
        break;

    case IO_LMBCR_eNORM:
#if (defined(LMB_SPECTRAL_CRs_FROM_ICS) && defined(LMB_CR_ELECTRONS))
        for (int n = 0; n < pc; n++)
        {
            for (int k = 0; k < LMB_CR_ELECTRONS; k++)
            {
                SphP[offset_gas + n].CReNorm[k] = *fp++;
                SphP[offset_gas + n].CReNorm[k] = pow(10.0, SphP[offset_gas + n].CReNorm[k]);
            }
        }
#endif // (defined(LMB_SPECTRAL_CRs_FROM_ICS) && defined(LMB_CR_ELECTRONS))
        break;

    case IO_LMBCR_pSLOPE:
#if (defined(LMB_SPECTRAL_CRs_FROM_ICS) && defined(LMB_CR_PROTONS))
        for (int n = 0; n < pc; n++)
        {
            for (int k = 0; k < LMB_CR_PROTONS; k++)
            {
                SphP[offset_gas + n].CRpSlope[k] = *fp++;
            }
        }
#endif // (defined(LMB_SPECTRAL_CRs_FROM_ICS) && defined(LMB_CR_PROTONS))
        break;

    case IO_LMBCR_eSLOPE:
#if (defined(LMB_SPECTRAL_CRs_FROM_ICS) && defined(LMB_CR_ELECTRONS))
        for (int n = 0; n < pc; n++)
        {
            for (int k = 0; k < LMB_CR_ELECTRONS; k++)
            {
                SphP[offset_gas + n].CReSlope[k] = *fp++;
            }
        }
#endif // (defined(LMB_SPECTRAL_CRs_FROM_ICS) && defined(LMB_CR_ELECTRONS))
        break;

    case IO_LMBCR_pCUT:
#if (defined(LMB_SPECTRAL_CRs_FROM_ICS) && defined(LMB_CR_PROTONS))
        for (int n = 0; n < pc; n++)
        {
            SphP[offset_gas + n].CRpCut = *fp++;
        }
#endif // (defined(LMB_SPECTRAL_CRs_FROM_ICS) && defined(LMB_CR_PROTONS))
        break;

    case IO_LMBCR_eCUT:
#if (defined(LMB_SPECTRAL_CRs_FROM_ICS) && defined(LMB_CR_ELECTRONS))
        for (int n = 0; n < pc; n++)
        {
            SphP[offset_gas + n].CReCut = *fp++;
        }
#endif // (defined(LMB_SPECTRAL_CRs_FROM_ICS) && defined(LMB_CR_ELECTRONS))
        break;

    case IO_LMBCR_pPRESSURE:
#ifdef LMB_SPECTRAL_CRs_PRESSURE_FROM_ICS
        for (int n = 0; n < pc; n++)
        {
            SphP[offset_gas + n].CRpPressure = *fp++;
        }
#endif // LMB_SPECTRAL_CRs_PRESSURE_FROM_ICS
        break;

    case IO_LMBCR_ePRESSURE:
#ifdef LMB_SPECTRALCRs_PRESSURE_FROM_ICS
        for (int n = 0; n < pc; n++)
        {
            SphP[offset_gas + n].CRePressure = *fp++;
        }
#endif // LMB_SPECTRALCRs_PRESSURE_FROM_ICS
        break;

    case IO_RHO_OLD:
#ifdef READ_RHO_OLD
        for (int n = 0; n < pc; n++)
        {
            SphP[offset_gas + n].DensityOld = *fp++;
        }
#endif // READ_RHO_OLD
        break;

    case IO_MACH:
#ifdef READ_MACH
        for (int n = 0; n < pc; n++)
        {
            SphP[offset_gas + n].Shock_Mach = *fp++;
        }
#endif // READ_MACH
        break;

    case IO_BHMASS:
#ifdef BLACK_HOLES
        for (int n = 0; n < pc; n++)
        {
            BPP(offset + n).BH_Mass = *fp++;
        }
#endif // BLACK_HOLES
        break;

    case IO_BHMDOT:
#ifdef BLACK_HOLES
        for (int n = 0; n < pc; n++)
        {
            BPP(offset + n).BH_Mdot = *fp++;
        }
#endif // BLACK_HOLES
        break;

    case IO_BH_Kin_ConeAxis:
#if (defined(BLACK_HOLES) && defined(PB_BH_KIN_EJECT_PART_INSIDE_CONE))
        for (int n = 0; n < pc; n++)
        {
            for (int k = 0; k < 3; k++)
            {
                P[offset + n].UnitVector_AGN_Wind[k] = (float)*fp++;
            }
        }
#endif // BLACK_HOLES && PB_BH_KIN_EJECT_PART_INSIDE_CONE
        break;

    case IO_BHPROGS:
#ifdef BLACK_HOLES
        for (int n = 0; n < pc; n++)
        {
            BPP(offset + n).BH_CountProgs = *fp++;
        }
#endif // BLACK_HOLES
        break;

    case IO_BHMRAD:
        break;

    case IO_BHSPIN:
#if defined(BLACK_HOLES) && defined(LS_BLACK_HOLES_SPIN_EVOLUTION)
        for (int n = 0; n < pc; n++)
        {
            for (int k = 0; k < 3; k++)
            {
                BPP(offset + n).BH_Spin[k] = (float)*fp++;
            }
        }
#endif // defined(BLACK_HOLES) && defined(LS_BLACK_HOLES_SPIN_EVOLUTION)
        break;

    case IO_DUSTL:

#if defined(LT_STELLAREVOLUTION) && defined(GL_CR_DUST)
        if (type == 0)
        {
            for (int n = 0; n < pc; n++)
            {
                for (int k = 0; k < LT_NMetP; k++)
                {
                    SphP[offset_gas + n].DustL[k] = (float)*fp++;
                }
            }
        }
#endif // defined(LT_STELLAREVOLUTION) && defined(GL_CR_DUST)
        break;

    case IO_DUSTS:
#if defined(LT_STELLAREVOLUTION) && defined(GL_CR_DUST)
        if (type == 0)
        {
            for (int n = 0; n < pc; n++)
            {
                for (int k = 0; k < LT_NMetP; k++)
                {
                    SphP[offset_gas + n].DustS[k] = (float)*fp++;
                }
            }
        }
#endif // defined(LT_STELLAREVOLUTION) && defined(GL_CR_DUST)
        break;

    case IO_Zs:
#ifdef LT_STELLAREVOLUTION
        if (type == 4)
        {
            for (int n = 0; n < pc; n++)
            {
                for (int k = 0; k < LT_NMetP; k++)
                {
                    MPP(offset + n).Metals[k] = (float)*fp++;
                }
            }
        }
        else if (type == 0)
        {
            for (int n = 0; n < pc; n++)
            {
                for (int k = 0; k < LT_NMetP; k++)
                {
                    SphP[offset_gas + n].Metals[k] = (float)*fp++;
                }
            }
        }
#endif // LT_STELLAREVOLUTION
        break;

    case IO_ZAGE:
#ifdef LT_ZAGE
        if (type == 4)
        {
            for (int n = 0; n < pc; n++)
            {
                MPP(offset + n).ZAge = *fp++;
            }
        }
        else if (type == 0)
        {
            for (int n = 0; n < pc; n++)
            {
                /* note this is not the weight that was used when the snapshot has been written */
                SphP[offset_gas + n].ZAgeW = get_metalmass(SphP[offset_gas + n].Metals);
#ifndef LT_LOGZAGE
                SphP[offset_gas + n].ZAge = *fp++ * SphP[offset_gas + n].ZAgeW;
#else // LT_LOGZAGE
                if (SphP[offset_gas + n].ZAgeW > 0)
                {
                    SphP[offset_gas + n].ZAge = log10(*fp++ * SphP[offset_gas + n].ZAgeW);
                }
                else
                {
                    SphP[offset_gas + n].ZAge = 0;
                }
#endif // LT_LOGZAGE
            }
        }
#endif // LT_ZAGE
        break;

    case IO_ZAGE_LLV:
#ifdef LT_ZAGE_LLV
        if (type == 4)
        {
            for (int n = 0; n < pc; n++)
            {
                MPP(offset + n).ZAge_llv = *fp++;
            }
        }
        else if (type == 0)
        {
            for (int n = 0; n < pc; n++)
            {
                /* note this is not the weight that was used when the snapshot has been written */
                SphP[offset_gas + n].ZAgeW_llv = SphP[offset_gas + n].Metals[Iron];
#ifndef LT_LOGZAGE
                SphP[offset_gas + n].ZAge_llv = *fp++ * SphP[offset_gas + n].ZAgeW_llv;
#else // LT_LOGZAGE
                if (SphP[offset_gas + n].ZAgeW_llv > 0)
                {
                    SphP[offset_gas + n].ZAge_llv = log10(*fp++ * SphP[offset_gas + n].ZAgeW_llv);
                }
                else
                {
                    SphP[offset_gas + n].ZAge_llv = 0;
                }
#endif // LT_LOGZAGE
            }
        }
#endif // LT_ZAGE_LLV
        break;

    case IO_iMass:
#ifdef LT_STELLAREVOLUTION
        if (type == 4)
        {
            for (int n = 0; n < pc; n++)
            {
                MPP(offset + n).iMass = *fp++;
            }
        }
#endif // LT_STELLAREVOLUTION
        break;

    case IO_CONTRIB:
#if defined(LT_STELLAREVOLUTION) && defined(LT_TRACK_CONTRIBUTES)
        if (type == 4)
        {
            for (int n = 0; n < pc; n++)
            {
                MPP(offset + n).contrib = *contribp++;
            }
        }
        else if (type == 0)
            for (int n = 0; n < pc; n++)
            {
                SphP[offset_gas + n].contrib = *contribp++;
            }
#endif // defined(LT_STELLAREVOLUTION) && defined(LT_TRACK_CONTRIBUTES)
        break;

    case IO_DMHSML:
#if defined(SUBFIND_RESHUFFLE_CATALOGUE) && defined(SUBFIND)
        for (int n = 0; n < pc; n++)
        {
            P[offset + n].DM_Hsml = *fp_single++;
        }
#endif // defined(SUBFIND_RESHUFFLE_CATALOGUE) && defined(SUBFIND)
        break;

    case IO_DMDENSITY:
#if defined(SUBFIND_RESHUFFLE_CATALOGUE) && defined(SUBFIND)
        for (int n = 0; n < pc; n++)
        {
            P[offset + n].u.DM_Density = *fp_single++;
        }
#endif // defined(SUBFIND_RESHUFFLE_CATALOGUE) && defined(SUBFIND)
        break;

    case IO_DMVELDISP:
#if defined(SUBFIND_RESHUFFLE_CATALOGUE) && defined(SUBFIND)
        for (int n = 0; n < pc; n++)
        {
            P[offset + n].v.DM_VelDisp = *fp_single++;
        }
#endif //  defined(SUBFIND_RESHUFFLE_CATALOGUE) && defined(SUBFIND)
        break;

    case IO_FB_M_H: /* particle hot mass */
        break;

    case IO_FB_M_C: /* particle cold mass */
        break;

    case IO_FB_M_MO: /* particle molecular mass */
        break;

    case IO_FB_E_H: /* particle hot energy */
        break;

    case IO_FB_M_SF: /* particle SF mass */
        break;

    case IO_FB_MF: /* number of time steps in multiphase */
        break;

    case IO_FB_NMF: /* number of multiphase stages */
        break;

    case IO_FB_EOUT: /* Energy output from MP particles */
        break;

    case IO_FB_EREC: /* Energy received from  MP particles */
        break;

    case IO_FB_CLOCK: /* time to still pass in MP */
        break;

    case IO_FB_E_TOT_0: /* multi phase hot energy time i-1 */

    case IO_FB_TDYN: /* cold phase dynamical time */

    case IO_CLDX:
#ifdef LT_STELLAREVOLUTION
        for (int n = 0; n < pc; n++)
        {
            SphP[offset_gas + n].XColdCloud = *fp++;
        }
#endif // LT_STELLAREVOLUTION
        break;

    case IO_HTEMP:
#ifdef LT_STELLAREVOLUTION
        for (int n = 0; n < pc; n++)
        {
            SphP[offset_gas + n].Temperature = *fp++;
        }
#endif // LT_STELLAREVOLUTION
        break;

    case IO_TEMP:
#ifdef LT_METAL_COOLING_WAL
        for (int n = 0; n < pc; n++)
        {
            SphP[offset_gas + n].Temperature = *fp++;
        }
#endif // LT_METAL_COOLING_WAL
        break;

#ifdef GT_DUST
    case IO_DVEL: /* relative velocities */
        for (int n = 0; n < pc; n++)
        {
            for (int idust = 0; idust < GT_NUM_DUST; idust++)
            {
                for (int k = 0; k < 3; k++)
                {
                    SphP[offset_gas + n].DeltaVel[idust][k] = *fp++;
                }
            }
        }
        break;

    case IO_DFRC: /* dust fraction */
        for (int n = 0; n < pc; n++)
        {
            for (int idust = 0; idust < GT_NUM_DUST; idust++)
            {
                SphP[offset_gas + n].DustFrac[idust] = invdustfrac(*fp++);
            }
        }
        break;

    case IO_DRAG:
        for (int n = 0; n < pc; n++)
        {
            for (int idust = 0; idust < GT_NUM_DUST; idust++)
            {
                SphP[offset_gas + n].DragCoefficient[idust] = *fp++;
            }
        }
        break;
#endif // GT_DUST

        /* the other input fields (if present) are not needed to define the
           initial conditions of the code */

    case IO_SFR:
    case IO_ZSMOOTH:
    case IO_allZSMOOTH:
    case IO_POT:
    case IO_ACCEL:
    case IO_DTENTR:
    case IO_STRESSDIAG:
    case IO_STRESSOFFDIAG:
    case IO_STRESSBULK:
    case IO_SHEARCOEFF:
    case IO_PRESSURE_ANISO:
    case IO_TSTP:
    case IO_DBDT:
    case IO_DIVB:
    case IO_ABVC:
    case IO_COOLRATE:
    case IO_CONDRATE:
    case IO_COOLTIME:
    case IO_DENN:
    case IO_AMDC:
    case IO_PHI:
    case IO_XPHI:
    case IO_GRADPHI:
    case IO_ROTB:
    case IO_SROTB:
    case IO_HSMS:
    case IO_ACRS:
    case IO_ACRB:
    case IO_AGS_SOFT:
    case IO_AGS_DENS:
    case IO_AGS_ZETA:
    case IO_AGS_OMEGA:
    case IO_AGS_CORR:
    case IO_AGS_NGBS:
        break;

    case IO_LASTENTRY:
        ENDRUN_MAIN("Last entry reached");
        break;
    default:
        VERBOSE(1, "Warning in empty read buffer: case unknown");
    }
}

/**
 * @brief Validates the consistency of the inital time between the parameter file and the initial conditions (IC) file.
 *
 * This function checks if the initial time specified in the parameter file (`All.TimeBegin`) matches
 * the initial time from the IC file (`TimeBegin_from_IC`). If there is a mismatch, the program
 * terminates with an error message indicating the discrepancy. If `All.TimeBegin` is negative, it
 * instead sets the value to the time from the IC file.
 *
 * @note The function assumes that `All.TimeBegin` is non-negative. If `All.TimeBegin` is negative, the value is set
 * instead.
 *
 * @throws Terminates the program with an error message if the initial time does not match.
 */
void check_initial_time(double TimeBegin_from_IC)
{
    if ((All.TimeBegin >= 0) && fabs(All.TimeBegin - TimeBegin_from_IC) > 1e-7)
    {
#if (EXPERT_LEVEL < 42)
        ENDRUN("Time of IC file does not agree with the TimeBegin in the parameter file.\nIC time = %g, parameter file "
               "TimeBegin = %g\nPlease check your parameter file and/or IC file.",
               TimeBegin_from_IC, All.TimeBegin);
#else // EXPERT_LEVEL
        VERBOSE(0,
                "\nWarning!\nTime of IC file does not agree with the TimeBegin in the parameter file.\nIC time = %g, "
                "parameter file "
                "TimeBegin = %g\n",
                TimeBegin_from_IC, All.TimeBegin);
#endif // EXPERT_LEVEL
    }
    // else do nothing - everything is fine
}

/**
 * @brief Validates the consistency of the box size between the parameter file and the initial conditions (IC) file.
 *
 * This function checks if the box size specified in the parameter file (`All.BoxSize`) matches
 * the box size from the IC file (`Boxsize_from_IC`). If there is a mismatch, the program
 * terminates with an error message indicating the discrepancy. If `All.BoxSize` is zero, it
 * instead sets the value to the size from the IC file.
 *
 * @note The function assumes that `All.BoxSize` is non-zero. If `All.BoxSize` is zero, the value is set instead.
 *
 * @throws Terminates the program with an error message if the box sizes do not match.
 */
void check_boxsize(double Boxsize_from_IC)
{
    if ((All.BoxSize != 0) && fabs(All.BoxSize - Boxsize_from_IC) > 1e-5)
    {
#if (EXPERT_LEVEL < 42)
        ENDRUN("Boxsize of IC file does not agree with the one in the parameter file.\nIC boxsize = %g, parameter file "
               "boxsize = %g\nPlease check your parameter file and/or IC file.",
               Boxsize_from_IC, All.BoxSize);
#else // EXPERT_LEVEL
        VERBOSE(0,
                "\nWarning!\nBoxsize of IC file does not agree with the one in the parameter file.\nIC boxsize = %g, "
                "parameter file "
                "boxsize = %g\n",
                Boxsize_from_IC, All.BoxSize);
#endif // EXPERT_LEVEL
    }
    else
    {
        // set the box size based in the IC header
        All.BoxSize = Boxsize_from_IC;
    }
}

/**
 * @brief Validates the consistency of the cosmo factors between the parameter file and the initial conditions (IC)
 * file.
 *
 * This function checks if the cosmological factors specified in the parameter file (`All.Omega0`, `All.OmegaLambda`,
 * `All.HubbleParam`) match the cosmo factors from the IC file (`Omega0_from_IC`, `OmegaLambda_from_IC`,
 * `HubbleParam_from_IC`). If there is a mismatch, the program terminates with an error message indicating the
 * discrepancy.
 *
 * @note The function assumes that `All.ComovingIntegrationOn` is true. If `All.ComovingIntegrationOn` is false, the
 * check is omitted.
 *
 * @throws Terminates the program with an error message if the cosmo factors do not match.
 */
void check_cosmo_factors(double Omega0_from_IC, double OmegaLambda_from_IC, double HubbleParam_from_IC)
{
    if ((All.ComovingIntegrationOn != 0) && fabs(All.Omega0 - Omega0_from_IC) > 1e-7 &&
        fabs(All.OmegaLambda - OmegaLambda_from_IC) > 1e-7 && fabs(All.HubbleParam - HubbleParam_from_IC) > 1e-7)
    {
#if (EXPERT_LEVEL < 42)
        ENDRUN(
            "Cosmo factors of IC file do not agree with the ones in the parameter file.\nIC/parameter file: Omega0 = "
            "(%g/%g), OmegaLambda = (%g/%g), HubbleParam = (%g,%g)\nPlease check your parameter file and/or IC file.",
            Omega0_from_IC, All.Omega0, OmegaLambda_from_IC, All.OmegaLambda, HubbleParam_from_IC, All.HubbleParam);
#else // EXPERT_LEVEL
        VERBOSE(0,
                "\nCosmo factors of IC file do not agree with the ones in the parameter file.\nIC/parameter file: "
                "Omega0 = (%g/%g), OmegaLambda = (%g/%g), HubbleParam = (%g,%g)\n",
                Omega0_from_IC, All.Omega0, OmegaLambda_from_IC, All.OmegaLambda, HubbleParam_from_IC, All.HubbleParam);
#endif // EXPERT_LEVEL
    }
}

/**
 * @brief Reads and discards an integer value from a file stream.
 *
 * This function performs a read operation on the given file descriptor
 * and overwrites the provided block size variable with the value read
 * from the file. The original value of @p blksize is not used.
 *
 * @param blksize Pointer to an integer variable that will receive
 *                the value read from the file.
 * @param fd Pointer to the file stream to read from.
 */
inline void skip_read(unsigned int *blksize, FILE *fd)
{
    my_fread(blksize, sizeof(int), 1, fd);
}

/**
 * @brief Opens an input file and reads the snapshot header on the reading task.
 *
 * This function is responsible for opening the initial condition or restart file,
 * reading the header information, and initializing format-specific I/O handles.
 * All operations are performed exclusively on the designated reading task
 * (`readTask`); other tasks return immediately.
 *
 * Supported input formats:
 * - ICFormat 1/2: binary Gadget-style files with block markers
 * - ICFormat 3: HDF5-based files (if HAVE_HDF5 is enabled)
 *
 * Binary (ICFormat 1/2) workflow:
 * - Opens the file (`fd`)
 * - Optionally opens auxiliary MUPPI file (`FdMuppi`)
 * - For ICFormat 2: reads and logs the header block label
 * - Reads raw header and determines endianness (`swap_file`)
 * - Verifies block size consistency (must be 256 bytes)
 * - Applies byte swapping via `swap_header()`
 * - Optionally patches header layout (PATCH_IO)
 *
 * HDF5 (ICFormat 3) workflow:
 * - Reads header attributes via `read_header_attributes_in_hdf5()`
 * - Opens the HDF5 file handle (`hdf5_file`)
 * - Opens per-type groups (`/PartType0` ... `/PartType5`) for types present
 *
 * @param[in] readTask Task responsible for file I/O.
 * @param[in] fname Name of the input file.
 * @param[out] fd File pointer for binary input (ICFormat 1/2).
 *
 * @param[out] hdf5_file HDF5 file handle (ICFormat 3).
 * @param[out] hdf5_grp  Array of HDF5 group handles per particle type.
 *
 * @param[out] FdMuppi File pointer for auxiliary MUPPI data (if enabled).
 * @param[out] fd_muppi_back Backup of original file pointer when switching streams.
 * @param[out] fmuppiname Buffer storing the derived MUPPI filename.
 *
 * @param[out] blksize1 First block size marker read from the file (binary formats).
 */
static void open_and_read_header(int readTask, char *fname, FILE *&fd,
#ifdef HAVE_HDF5
                                 hid_t &hdf5_file, hid_t hdf5_grp[6],
#endif // HAVE_HDF5
                                 unsigned int &blksize1)
{
    if (ThisTask != readTask)
    {
        return;
    }

    if (All.ICFormat == 1 || All.ICFormat == 2)
    {
        if (!(fd = fopen(fname, "r")))
        {
            const int err = errno;
            const std::string host = get_hostname();
            ENDRUN("Can't open file `%s' for reading initial conditions. "
                   "errno=%d (%s), ThisTask=%d, host=%s\n",
                   fname, err, strerror(err), ThisTask, host.c_str());
        }


        if (All.ICFormat == 2)
        {
            unsigned int blksize2_hdr;
            char label[4];
            int nextblock;
            skip_read(&blksize1, fd);
            swap_file = blksize1;
            my_fread(&label, sizeof(char), 4, fd);
            my_fread(&nextblock, sizeof(int), 1, fd);
            swap_Nbyte((char *)&nextblock, 1, 4);
            VERBOSE_ALL(2, "Reading header => '%c%c%c%c' (%d byte)", label[0], label[1], label[2], label[3], nextblock);
            skip_read(&blksize2_hdr, fd);
        }

        // Read raw header and detect byte-swap from block markers
        unsigned int blksize2;
        skip_read(&blksize1, fd);
        if (All.ICFormat == 1 && blksize1 != 256)
        {
            swap_file = 1;
        }

        my_fread(&header, sizeof(header), 1, fd);
        skip_read(&blksize2, fd);
        swap_Nbyte((char *)&blksize1, 1, 4);
        swap_Nbyte((char *)&blksize2, 1, 4);

        if (blksize1 != 256 || blksize2 != 256)
        {
            ENDRUN("Incorrect header format. "
                   "(Fill[] in header probably wrong size — must total 256 bytes.)");
        }

        swap_header();
        check_cosmo_factors(header.Omega0, header.OmegaLambda, header.HubbleParam);

#ifdef PATCH_IO
        // Remap type 0/1/2 slots for MagneticumDM layout
        VERBOSE(2, "Task %d: WARNING, PATCHING HEADER FOR MagneticumDM simulation!!!", ThisTask);
        header.mass[2] = header.mass[1];
        header.mass[1] = header.mass[0];
        header.mass[0] = 0;
        header.npart[2] = header.npart[1];
        header.npart[1] = header.npart[0];
        header.npart[0] = 0;
        header.npartTotal[2] = header.npartTotal[1];
        header.npartTotal[1] = header.npartTotal[0];
        header.npartTotal[0] = 0;
        header.npartTotalHighWord[2] = header.npartTotalHighWord[1];
        header.npartTotalHighWord[1] = header.npartTotalHighWord[0];
        header.npartTotalHighWord[0] = 0;
#endif // PATCH_IO
    }

#ifdef HAVE_HDF5
    if (All.ICFormat == 3)
    {
        char buf[500];
        read_header_attributes_in_hdf5(fname);
        hdf5_file = H5Fopen(fname, H5F_ACC_RDONLY, H5P_DEFAULT);
        for (int type = 0; type < 6; type++)
        {
            if (header.npart[type] > 0)
            {
                sprintf(buf, "/PartType%d", type);
                hdf5_grp[type] = H5Gopen2(hdf5_file, buf, H5P_DEFAULT);
            }
        }
    }
#endif // HAVE_HDF5
}

/**
 * @brief Broadcasts file header and related metadata from the reading task.
 *
 * This function distributes the snapshot/restart file header (`header`) and
 * associated metadata from the designated reading task (`readTask`) to all
 * other tasks in the range [readTask, lastTask] using point-to-point MPI
 * communication.
 *
 * @param[in] readTask Task responsible for reading and broadcasting the header.
 * @param[in] lastTask Last task participating in the broadcast.
 *
 * @param[in,out] FdMuppi File pointer associated with MUPPI I/O (if GM_MUPPI enabled).
 */
static void broadcast_header(int readTask, int lastTask
)
{
    MPI_Status status;

    if (ThisTask == readTask)
    {
        for (int task = readTask + 1; task <= lastTask; task++)
        {
            MPI_Ssend(&header, sizeof(header), MPI_BYTE, task, TAG_HEADER, MYMPI_COMM_WORLD);
            MPI_Ssend(&swap_file, 1, MPI_INT, task, TAG_SWAP, MYMPI_COMM_WORLD);
        }
    }
    else
    {
        MPI_Recv(&header, sizeof(header), MPI_BYTE, readTask, TAG_HEADER, MYMPI_COMM_WORLD, &status);
        MPI_Recv(&swap_file, 1, MPI_INT, readTask, TAG_SWAP, MYMPI_COMM_WORLD, &status);
    }

    // Validate precision consistency between binary and build flags
#ifdef INPUT_IN_DOUBLEPRECISION
    ENDRUN_IF(header.flag_doubleprecision == 0, "Problem: Code compiled with INPUT_IN_DOUBLEPRECISION, "
                                                "but input files are in single precision!");
#else // INPUT_IN_DOUBLEPRECISION
    ENDRUN_IF(header.flag_doubleprecision, "Problem: Code not compiled with INPUT_IN_DOUBLEPRECISION, "
                                           "but input files are in double precision!");
#endif // INPUT_IN_DOUBLEPRECISION
}

/**
 * @brief Initializes global particle counts and allocates memory.
 *
 * This function prepares global simulation state after reading the file
 * header. It computes total particle counts, derives per-task memory limits,
 * and performs memory allocation for particle storage and communication buffers.
 *
 * The initialization proceeds in several stages:
 *
 * 1. **Global particle counts**
 *    - For single-file snapshots (`header.num_files <= 1`), per-file counts
 *      (`header.npart`) are copied into global totals (`header.npartTotal`)
 *    - Total particle numbers are reconstructed using low/high 32-bit words
 *      (`header.npartTotal` and `header.npartTotalHighWord`)
 *    - Global counters are stored in:
 *        - All.TotNumPart (all particle types)
 *        - All.TotN_gas   (gas particles, type 0)
 *        - All.TotN_stars (stars, type 4; if SFR enabled)
 *
 * 2. **Optional particle augmentation**
 *    - If `GENERATE_GAS_IN_ICS` is enabled and starting from ICs
 *      (`RestartFlag == 0`), additional gas particles are generated
 *      from type-1 particles and included in global counts
 *
 * 3. **Mass table initialization**
 *    - Copies per-type masses from `header.mass[]` into `All.MassTable[]`
 *
 * 4. **Per-task memory limits**
 *    - Computes maximum particle capacities per task:
 *        - All.MaxPart    (total particles)
 *        - All.MaxPartSph (gas particles)
 *    - Based on global totals, number of MPI tasks (`NTask`), and
 *      allocation factor (`All.PartAllocFactor`)
 *    - Optional overrides:
 *        - INHOMOG_GASDISTR_HINT: sets MaxPartSph = MaxPart
 *        - LT_STELLAREVOLUTION: computes All.MaxPartMet (stellar particles)
 *        - BLACK_HOLES: computes All.MaxPartBH (BH particles)
 *
 * 5. **Memory allocation**
 *    - Calls allocate_memory() to allocate particle data structures
 *    - Allocates communication buffer (`CommBuffer`) of size
 *      `All.BufferSize` MB
 *
 * 6. **Time initialization (restart runs)**
 *    - For restart modes (`RestartFlag >= 2`), initializes:
 *        - All.Time and All.TimeBegin from header
 *        - Cosmological factors via set_cosmo_factors_for_current_time()
 *
 * 7. **Additional physics initialization**
 *    - If stellar evolution is enabled, computes `time_age`
 */
static void init_global_counts_and_allocate()
{
    // Single-file snapshots: copy per-file counts into npartTotal
    if (header.num_files <= 1)
    {
        for (int i = 0; i < 6; i++)
        {
            header.npartTotal[i] = header.npart[i];
#ifdef SFR
            header.npartTotalHighWord[i] = 0;
#endif // SFR
        }
    }

    // Accumulate global counts
    All.TotN_gas = header.npartTotal[0] + (((long long)header.npartTotalHighWord[0]) << 32);
#ifdef SFR
    All.TotN_stars = header.npartTotal[4] + (((long long)header.npartTotalHighWord[4]) << 32);
#endif // SFR
    All.TotNumPart = 0;
    for (int i = 0; i < 6; i++)
    {
        All.TotNumPart += header.npartTotal[i];
        All.TotNumPart += (((long long)header.npartTotalHighWord[i]) << 32);
    }

#ifdef GENERATE_GAS_IN_ICS
    if (RestartFlag == 0)
    {
        VERBOSE(2, "  Generating space for GAS particles...");
        All.TotN_gas += header.npartTotal[1] + (((long long)header.npartTotalHighWord[1]) << 32);
        All.TotNumPart += header.npartTotal[1] + (((long long)header.npartTotalHighWord[1]) << 32);
    }
#endif // GENERATE_GAS_IN_ICS

    // Copy mass table
    for (int i = 0; i < 6; i++)
    {
        All.MassTable[i] = header.mass[i];
    }

    // Per-task storage limits
    All.MaxPart = (int)(All.PartAllocFactor * (All.TotNumPart / NTask));
    All.MaxPartSph = (int)(All.PartAllocFactor * (All.TotN_gas / NTask));
    VERBOSE(2, "Task %d/%d/%d: MaxPart %d, MaxSphP %d (%lld,%lld)", ThisTask, 0 /*readTask not in scope — log only*/, 0,
            All.MaxPart, All.MaxPartSph, All.TotNumPart, All.TotN_gas);

#ifdef INHOMOG_GASDISTR_HINT
    All.MaxPartSph = All.MaxPart;
#endif // INHOMOG_GASDISTR_HINT

#ifdef SFR
    All.MaxPartMet = (All.TotN_stars == 0)
                         ? (int)(All.PartAllocFactor * (All.TotN_gas / NTask) * All.SFfactor * All.Generations)
                         : (int)(All.PartAllocFactor *
                                 (All.TotN_stars / NTask + (All.TotN_gas / NTask) * All.SFfactor * All.Generations));
    VERBOSE(2, "       MaxStarPart %d (%lld,%lld,%g,%d)", All.MaxPartMet, All.TotN_gas, All.TotN_stars, All.SFfactor,
            All.Generations);
#endif // SFR

#ifdef BLACK_HOLES
    All.MaxPartBH =
        (All.TotBHs == 0)
            ? (int)(All.PartAllocFactor * (All.TotN_gas / NTask) * All.BlackHoleFormationFactor)
            : (int)(All.PartAllocFactor * (All.TotBHs / NTask + (All.TotN_gas / NTask) * All.BlackHoleFormationFactor));
    VERBOSE(2, "       MaxBlackHolePart %d (%lld,%d,%g)", All.MaxPartBH, All.TotN_gas, All.TotBHs,
            All.BlackHoleFormationFactor);
#endif // BLACK_HOLES

    allocate_memory();

    size_t bytes;
    if (!(CommBuffer = mymalloc("CommBuffer", bytes = All.BufferSize * 1024 * 1024)))
    {
        ENDRUN_MAIN("Failed to allocate memory for `CommBuffer' (%g MB).", bytes / (1024.0 * 1024.0));
    }

    if (RestartFlag >= 2)
    {
        All.Time = All.TimeBegin = header.time;
        set_cosmo_factors_for_current_time();
    }

#ifdef LT_STELLAREVOLUTION
    time_age = get_age(All.Time);
#endif // LT_STELLAREVOLUTION
}

/**
 * @brief Logs a summary of the particle content of the current input file.
 *
 * @param[in] readTask Task responsible for reading and logging.
 * @param[in] lastTask Last task participating in the data distribution.
 * @param[in] fname Name of the input file being processed.
 */
static void log_file_summary(int readTask, int lastTask, const char *fname)
{
    if (ThisTask != readTask)
    {
        return;
    }

    unsigned int n_in_file = 0;
    for (int i = 0; i < 6; i++)
    {
        n_in_file += header.npart[i];
    }

    long long total = 0;
    for (int i = 0; i < 6; i++)
    {
        total += header.npartTotal[i];
        total += (((long long)header.npartTotalHighWord[i]) << 32);
    }
    VERBOSE(2, "Total Number of Particles: %llu", (unsigned long long)total);

    VERBOSE(2,
            "\nreading file `%s' on task=%d (contains %u particles.)\n"
            "distributing this file to tasks %d-%d\n"
            "Type 0 (gas):   %8u  (tot=%6d%09d) masstab=%g\n"
            "Type 1 (halo):  %8u  (tot=%6d%09d) masstab=%g\n"
            "Type 2 (disk):  %8u  (tot=%6d%09d) masstab=%g\n"
            "Type 3 (bulge): %8u  (tot=%6d%09d) masstab=%g\n"
            "Type 4 (stars): %8u  (tot=%6d%09d) masstab=%g\n"
            "Type 5 (bndry): %8u  (tot=%6d%09d) masstab=%g\n",
            fname, ThisTask, n_in_file, readTask, lastTask, // lastTask assumed global/accessible
            header.npart[0], (int)(header.npartTotal[0] / 1000000000), (int)(header.npartTotal[0] % 1000000000),
            All.MassTable[0], header.npart[1], (int)(header.npartTotal[1] / 1000000000),
            (int)(header.npartTotal[1] % 1000000000), All.MassTable[1], header.npart[2],
            (int)(header.npartTotal[2] / 1000000000), (int)(header.npartTotal[2] % 1000000000), All.MassTable[2],
            header.npart[3], (int)(header.npartTotal[3] / 1000000000), (int)(header.npartTotal[3] % 1000000000),
            All.MassTable[3], header.npart[4], (int)(header.npartTotal[4] / 1000000000),
            (int)(header.npartTotal[4] % 1000000000), All.MassTable[4], header.npart[5],
            (int)(header.npartTotal[5] / 1000000000), (int)(header.npartTotal[5] % 1000000000), All.MassTable[5]);
}

/**
 * @brief Computes per-task particle counts and validates memory capacity.
 *
 * This function determines how many particles of each type from the input
 * file will be assigned to the calling MPI task, based on an even distribution
 * across the task range [readTask, lastTask]. It mirrors the load-balancing
 * logic used during actual data distribution.
 *
 * For each particle type:
 * - The number of particles assigned to this task is computed as either
 *   ⌊N/ntask⌋ or ⌈N/ntask⌉ depending on the remainder distribution
 * - The total number of incoming particles (`nall`) is accumulated
 *
 * @param[in] readTask Index of the first task participating in file reading.
 * @param[in] lastTask Index of the last task participating in file reading.
 */
static void compute_local_particle_counts(int readTask, int lastTask)
{
    int ntask = lastTask - readTask + 1;
    unsigned int nall = 0;

    for (int type = 0; type < 6; type++)
    {
        unsigned int n_in_file = header.npart[type];
        unsigned int n_for_this_task = n_in_file / ntask;
        if ((ThisTask - readTask) < (long int)(n_in_file % ntask))
        {
            n_for_this_task++;
        }

        if (type == 0 && (long int)(N_gas + n_for_this_task) > All.MaxPartSph)
            ENDRUN_MAIN("Not enough space on task=%d for SPH particles "
                        "(space for %d, need at least %d)",
                        ThisTask, All.MaxPartSph, N_gas + n_for_this_task);

        nall += n_for_this_task;
    }

    ENDRUN_IF(NumPart + nall > (unsigned int)All.MaxPart,
              "Not enough space on task=%d (space for %d, need at least %d)", ThisTask, All.MaxPart, NumPart + nall);
}

/**
 * @brief Reads and distributes particle data blocks across MPI tasks.
 *
 * This function is the core I/O routine for loading particle data from
 * initial condition or restart files. It iterates over all defined I/O
 * blocks (IOfields) and, for each enabled block, performs a block-wise
 * read on the designated reading task (`readTask`) followed by an
 * MPI-based distribution to all tasks in the range [readTask, lastTask].
 *
 * The routine supports multiple input formats:
 * - ICFormat 1/2: binary Gadget-style files with block markers
 * - ICFormat 3: HDF5-based files (if HAVE_HDF5 is enabled)
 *
 * For each block:
 * - The number of particles per type is determined
 * - Data is streamed in chunks limited by `All.BufferSize`
 * - The reading task loads data into a communication buffer
 * - Data is distributed via MPI_Ssend / MPI_Recv
 * - Each task unpacks its portion into local particle arrays
 *
 * Particle data is assigned in a load-balanced manner consistent with
 * the domain decomposition: each task receives either ⌊N/ntask⌋ or
 * ⌈N/ntask⌉ elements per particle type.
 *
 * @param[in] readTask Task responsible for reading from disk.
 * @param[in] lastTask Last task participating in the data distribution.
 * @param[in,out] fd File pointer for binary input (may be reassigned internally).
 *
 * @param[in] hdf5_grp Array of HDF5 group handles for particle types [0..5]
 *                     (used only if ICFormat == 3 and HAVE_HDF5 is defined).
 *
 * @param[in,out] FdMuppi File pointer for MUPPI-related blocks (if enabled).
 * @param[in] fd_muppi_back Backup file pointer to restore after MUPPI reads.
 *
 * @param[in] ntask Number of MPI tasks participating in the read operation.
 *
 * @warning Mismatched block sizes in binary mode will trigger a fatal error.
 * @warning Incorrect datatype assumptions (e.g. ID width) may cause read failures.
 */
static void read_data_blocks(int readTask, int lastTask, FILE *&fd,
#ifdef HAVE_HDF5
                             hid_t hdf5_grp[6],
#endif // HAVE_HDF5
                             int ntask)
{
    MPI_Status status;
    char buf[500];
    unsigned int blksize1, blksize2;

    for (int bnr = 0; bnr < IO_LASTENTRY; bnr++)
    {
        enum iofields blocknr = (enum iofields)bnr;
        if (blocknr == IO_LASTENTRY)
        {
            break;
        }

#if defined(GM_DMVELS) && defined(GM_DMVELS_OUTPUT)
        if (blocknr == IO_DM_AVE_VEL || blocknr == IO_DM_AVE_SIG || blocknr == IO_DM_NPART)
        {
            continue;
        }
#endif // GM_DMVELS && GM_DMVELS_OUTPUT

        if (!io_matrix::check_if_read_or_write_block(blocknr, RestartFlag))
        {
            continue;
        }

        if (ThisTask == readTask && (ThisTask == 0 || VERBOSE_LEVEL > 2))
        {
            io_matrix::get_dataset_name(blocknr, buf);
            VERBOSE_ALL(2, "Task %d: reading block %d (%s)", ThisTask, bnr, buf);
        }

        int bytes_per_blockelement = io_matrix::get_bytes_per_blockelement(blocknr, 1);
        size_t blockmaxlen = (size_t)((All.BufferSize * 1024 * 1024) / bytes_per_blockelement);
        unsigned int typelist[6];
        unsigned int npart = io_matrix::get_particles_in_block(blocknr, typelist);

        if (ThisTask == readTask && (ThisTask == 0 || VERBOSE_LEVEL > 2))
        {
            VERBOSE_ALL(2, "Task %d: reading %u [%u,%u,%u,%u,%u,%u]", ThisTask, npart, typelist[0], typelist[1],
                        typelist[2], typelist[3], typelist[4], typelist[5]);
        }

        if (npart == 0)
        {
            continue;
        }

        bool is_dm_derived = (blocknr == IO_DMHSML || blocknr == IO_DMDENSITY || blocknr == IO_DMVELDISP);

        // ---- open block in binary file (readTask only) ----
        if (ThisTask == readTask && !is_dm_derived)
        {
#ifdef LT_TRACK_CONTRIBUTES
            if (blocknr == IO_CONTRIB)
            {
                fd = FdTrck;
            }
#endif // LT_TRACK_CONTRIBUTES
            if (All.ICFormat == 2)
            {
                if (ThisTask == readTask && blocknr == IO_POS && (ThisTask == 0 || VERBOSE_LEVEL > 2))
                {
                    VERBOSE_ALL(2, "Task %d:   Reading POS, npart=%d", ThisTask, npart);
                }

                char label[4];
                io_matrix::get_Tab_IO_Label(blocknr, label);
                find_block(label, fd);
            }
            if (All.ICFormat == 1 || All.ICFormat == 2)
            {
                skip_read(&blksize1, fd);
            }
        }

        // ---- per-type MPI scatter-and-fill loop ----
        size_t nread = 0;
        for (unsigned int type = 0, offset = NumPart, offset_gas = N_gas, offset_stars = N_stars, offset_bhs = N_BHs;
             type < 6; type++)
        {
            unsigned int n_in_file = header.npart[type];
#ifdef HAVE_HDF5
            int pcsum = 0;
#endif // HAVE_HDF5
            if (typelist[type] == 0)
            {
                // Block doesn't include this type; skip over its share
                unsigned int n_skip =
                    n_in_file / ntask + ((ThisTask - readTask) < (long int)(n_in_file % ntask) ? 1 : 0);
                offset += n_skip;
                continue;
            }

            for (int task = readTask; task <= lastTask; task++)
            {
                unsigned int n_for_this_task = n_in_file / ntask;
                if ((task - readTask) < (long int)(n_in_file % ntask))
                {
                    n_for_this_task++;
                }

                if (task == ThisTask)
                {
                    ENDRUN_IF(NumPart + n_for_this_task > (unsigned int)All.MaxPart, "too many particles. %d %d %d",
                              NumPart, n_for_this_task, All.MaxPart);
                }

                // Verbose offset logging for IO_POS
                if (ThisTask == readTask && blocknr == IO_POS && (ThisTask == 0 || VERBOSE_LEVEL > 2))
                {
                    VERBOSE_ALL(2, "Task %d:    type = %d, offset = %d, n_for_this_task = %d", ThisTask, type, offset,
                                n_for_this_task);
                    if (type == 0)
                    {
                        VERBOSE_ALL(2, "Task %d:    offset_gas = %d, N_gas = %d", ThisTask, offset_gas, N_gas);
                    }
                    if (type == 1 || type == 2 || type == 3)
                    {
                        VERBOSE_ALL(2, "Task %d:    offset_dm = %d, NumPart = %d", ThisTask, offset, NumPart);
                    }
#ifdef SFR
                    if (type == 4)
                    {
                        VERBOSE_ALL(2, "Task %d:    offset_stars = %d, N_stars = %d", ThisTask, offset_stars, N_stars);
                    }
#endif // SFR
#ifdef BLACK_HOLES
                    if (type == 5)
                    {
                        VERBOSE_ALL(2, "Task %d:    offset_bhs = %d, N_BHs = %d", ThisTask, offset_bhs, N_BHs);
                    }
#endif // BLACK_HOLES
                }

                do
                {
                    unsigned int pc = std::min(n_for_this_task, (unsigned int)blockmaxlen);

                    // readTask reads from disk (binary or HDF5)
                    if (ThisTask == readTask)
                    {
                        if (All.ICFormat == 1 || All.ICFormat == 2)
                        {
                            if (!is_dm_derived)
                            {
                                my_fread(CommBuffer, bytes_per_blockelement, pc, fd);
                                nread += pc;
                            }
                            else
                            {
                                nread += pc;
                            }
                        }

#ifdef HAVE_HDF5
                        if (All.ICFormat == 3 && pc > 0)
                        {
                            hsize_t dims[2], count[2], start[2];

                            io_matrix::get_dataset_name(blocknr, buf);
                            hid_t hdf5_dataset = H5Dopen2(hdf5_grp[type], buf, H5P_DEFAULT);

                            dims[0] = header.npart[type];
                            dims[1] = io_matrix::get_values_per_blockelement(blocknr);
                            int rank = (dims[1] == 1) ? 1 : 2;

                            hid_t hdf5_dataspace_in_file = H5Screate_simple(rank, dims, NULL);
                            dims[0] = pc;
                            hid_t hdf5_dataspace_in_memory = H5Screate_simple(rank, dims, NULL);

                            start[0] = pcsum;
                            start[1] = 0;
                            count[0] = pc;
                            count[1] = io_matrix::get_values_per_blockelement(blocknr);
                            pcsum += pc;

                            H5Sselect_hyperslab(hdf5_dataspace_in_file, H5S_SELECT_SET, start, NULL, count, NULL);
                            hid_t hdf5_datatype = GetHDF5Datatype(io_matrix::get_datatype_in_block(blocknr));
                            H5Dread(hdf5_dataset, hdf5_datatype, hdf5_dataspace_in_memory, hdf5_dataspace_in_file,
                                    H5P_DEFAULT, CommBuffer);

                            H5Tclose(hdf5_datatype);
                            H5Sclose(hdf5_dataspace_in_memory);
                            H5Sclose(hdf5_dataspace_in_file);
                            H5Dclose(hdf5_dataset);
                        }
#endif // HAVE_HDF5
                    }

                    // Scatter buffer to target task
                    if (ThisTask == readTask && task != readTask && pc > 0)
                    {
                        MPI_Ssend(CommBuffer, bytes_per_blockelement * pc, MPI_BYTE, task, TAG_PDATA, MYMPI_COMM_WORLD);
                    }

                    if (ThisTask != readTask && task == ThisTask && pc > 0)
                    {
                        MPI_Recv(CommBuffer, bytes_per_blockelement * pc, MPI_BYTE, readTask, TAG_PDATA,
                                 MYMPI_COMM_WORLD, &status);
                    }
       // Target task unpacks buffer into particle arrays
                    if (ThisTask == task)
                    {
                        empty_read_buffer(blocknr, offset, pc, type, offset_gas, offset_stars, offset_bhs);
                        offset += pc;
                        if (type == 0)
                        {
                            offset_gas += pc;
                        }
                        if (type == 4)
                        {
                            offset_stars += pc;
                        }
                        if (type == 5)
                        {
                            offset_bhs += pc;
                        }
                    }

                    n_for_this_task -= pc;
                } while (n_for_this_task > 0);
            } // for task
        } // for type

        // ---- close block in binary file and verify markers (readTask only) ----
        if (ThisTask == readTask && !is_dm_derived && (All.ICFormat == 1 || All.ICFormat == 2))
        {
            skip_read(&blksize2, fd);


            swap_Nbyte((char *)&blksize1, 1, 4);
            swap_Nbyte((char *)&blksize2, 1, 4);

            if (blksize1 != blksize2)
            {
                VERBOSE_ALL(0, "incorrect block-sizes detected!");
                VERBOSE_ALL(0, "Task=%d  blocknr=%d  blksize1=%u  blksize2=%u", ThisTask, bnr, blksize1, blksize2);
                VERBOSE_ALL(0, "nread = %llu", (unsigned long long)nread);
                if (blocknr == IO_ID)
                {
                    VERBOSE_ALL(0, "Possible mismatch of 32-bit and 64-bit IDs "
                                   "in IC file vs OpenGadget3 compilation!");
                }
                ENDRUN_MAIN("Error in block-sizes.");
            }
        }
    } // for bnr
}

/**
 * @brief Distributes particle counts from a file across MPI tasks and updates local counters.
 *
 * This function computes how many particles of each type from the current input file
 * should be assigned to the calling task, based on a contiguous task range
 * [readTask, lastTask]. The distribution is performed as evenly as possible,
 * with any remainder particles assigned to the lower-indexed tasks.
 *
 * @param[in] readTask Index of the first task reading from the file.
 * @param[in] lastTask Index of the last task reading from the file.
 *
 * @note Assumes ThisTask is within the range [readTask, lastTask].
 * @note Uses integer division to ensure load-balanced distribution across tasks.
 * @note Relies on global variables (NumPart, N_gas, N_stars, N_BHs, header).
 */
static void update_local_particle_counts(int readTask, int lastTask)
{
    int ntask = lastTask - readTask + 1;

    for (int type = 0; type < 6; type++)
    {
        unsigned int n_in_file = header.npart[type];
        unsigned int n_for_this_task = n_in_file / ntask;
        if ((ThisTask - readTask) < (long int)(n_in_file % ntask))
            n_for_this_task++;

        NumPart += n_for_this_task;
        if (type == 0)
        {
            N_gas += n_for_this_task;
        }
#ifdef SFR
        if (type == 4)
        {
            N_stars += n_for_this_task;
        }
#endif // SFR
#ifdef BLACK_HOLES
        if (type == 5)
        {
            N_BHs += n_for_this_task;
        }
#endif // BLACK_HOLES
    }
}

/**
 * @brief Closes input file handles on the designated reading task.
 *
 * This function releases file resources associated with initial condition input,
 * but only on the task responsible for reading (`readTask`).
 *
 * @param[in] readTask Task responsible for file I/O.
 * @param[in] fd File pointer to the input file (used for binary formats).
 *
 * @param[in] hdf5_file HDF5 file handle (used when ICFormat == 3).
 * @param[in] hdf5_grp  Array of HDF5 group handles for particle types [0..5].
 */
static void close_input_files(int readTask, FILE *fd
#ifdef HAVE_HDF5
                              ,
                              hid_t hdf5_file, hid_t hdf5_grp[6]
#endif // HAVE_HDF5
)
{
    if (ThisTask != readTask)
    {
        return;
    }

    if (All.ICFormat == 1 || All.ICFormat == 2)
    {
        fclose(fd);
    }
#ifdef HAVE_HDF5
    if (All.ICFormat == 3)
    {
        for (int type = 5; type >= 0; type--)
        {
            if (header.npart[type] > 0)
            {
                H5Gclose(hdf5_grp[type]);
            }
        }
        H5Fclose(hdf5_file);
    }
#endif // HAVE_HDF5
}

/**
 * @brief Reads a snapshot file.
 *
 * This function reads a snapshot file and distributes the data it contains
 * to tasks 'readTask' to 'lastTask'.
 *
 * @param fname Name of the snapshot file.
 * @param readTask Task that reads the snapshot file.
 * @param lastTask Last task that reads the snapshot file.
 */
void read_file(char *fname, int readTask, int lastTask)
{
    FILE *fd = nullptr;

#ifdef HAVE_HDF5
    hid_t hdf5_file = 0, hdf5_grp[6] = {};
#endif // HAVE_HDF5


    unsigned int blksize1 = 0;

    // Open file and read header (readTask only)
    open_and_read_header(readTask, fname, fd,
#ifdef HAVE_HDF5
                         hdf5_file, hdf5_grp,
#endif // HAVE_HDF5
                         blksize1);

    // broadcast header to sibling tasks; validate precision
    broadcast_header(readTask, lastTask
    );

    // Sanity-check TimeBegin here after broadcasting information to all ranks
    check_initial_time(header.time);

    // Sanity-check box size against the parameter file.
    if (All.PeriodicBoundariesOn == 1)
    {
        check_boxsize(header.BoxSize);
    }

    // first-file initialisation: global counts + memory alloc
    if (All.TotNumPart == 0)
    {
        init_global_counts_and_allocate();
    }

    // log_file_summary  (readTask only)
    log_file_summary(readTask, lastTask, fname);

    // verify local storage capacity
    compute_local_particle_counts(readTask, lastTask);

    // build io_fields metadata table now that FdMuppi is actually open
    io_matrix::init_io_fields(
    );

    read_data_blocks(readTask, lastTask, fd,
#ifdef HAVE_HDF5
                     hdf5_grp,
#endif // HAVE_HDF5
                     lastTask - readTask + 1);

    update_local_particle_counts(readTask, lastTask);

    close_input_files(readTask, fd
#ifdef HAVE_HDF5
                      ,
                      hdf5_file, hdf5_grp
#endif // HAVE_HDF5
    );
}

/**
 * @brief Find the number of files a snapshot is distributed on
 *
 * This function determines on how many files a given snapshot is distributed.
 *
 * @param fname Filename of the snapshot
 */
int find_files(char *fname)
{
    FILE *fd;
    char buf[200], buf1[200];
    int dummy;

    sprintf(buf, "%s.%d", fname, 0);
    sprintf(buf1, "%s", fname);

    if (All.ICFormat == 3)
    {
        sprintf(buf, "%s.%d.hdf5", fname, 0);
        sprintf(buf1, "%s.hdf5", fname);
    }

    VERBOSE(2, "Trying to read file %s in format %d ", buf, All.ICFormat);

#ifndef HAVE_HDF5
    ENDRUN_IF(All.ICFormat == 3, "Code wasn't compiled with HDF5 support enabled! ");
#endif // HAVE_HDF5

    header.num_files = 0;

    if (ThisTask == 0)
    {
        if ((fd = fopen(buf, "r")))
        {
            if (All.ICFormat == 1 || All.ICFormat == 2)
            {
                if (All.ICFormat == 2)
                {
                    my_fread(&dummy, sizeof(dummy), 1, fd);
                    swap_file = dummy;
                    my_fread(&dummy, sizeof(dummy), 1, fd);
                    my_fread(&dummy, sizeof(dummy), 1, fd);
                    my_fread(&dummy, sizeof(dummy), 1, fd);
                }

                my_fread(&dummy, sizeof(dummy), 1, fd);
                if (All.ICFormat == 1)
                {
                    if (dummy == 256)
                    {
                        swap_file = 8;
                    }
                    else
                    {
                        swap_file = 1;
                    }
                }
                my_fread(&header, sizeof(header), 1, fd);
                swap_header();

#ifdef PATCH_IO
                VERBOSE(1, "Task %d: WARNING, PATCHING HEADER FOR MagneticumDM simulation !!! ", ThisTask);
                header.mass[2] = header.mass[1];
                header.mass[1] = header.mass[0];
                header.mass[0] = 0;
                header.npart[2] = header.npart[1];
                header.npart[1] = header.npart[0];
                header.npart[0] = 0;
                header.npartTotal[2] = header.npartTotal[1];
                header.npartTotal[1] = header.npartTotal[0];
                header.npartTotal[0] = 0;
                header.npartTotalHighWord[2] = header.npartTotalHighWord[1];
                header.npartTotalHighWord[1] = header.npartTotalHighWord[0];
                header.npartTotalHighWord[0] = 0;
#endif // PATCH_IO
                my_fread(&dummy, sizeof(dummy), 1, fd);
            }
            fclose(fd);

#ifdef HAVE_HDF5
            if (All.ICFormat == 3)
            {
                read_header_attributes_in_hdf5(buf);
            }
#endif // HAVE_HDF5
        }
    }

    MPI_Bcast(&swap_file, 1, MPI_INT, 0, MYMPI_COMM_WORLD);
    MPI_Bcast(&header, sizeof(header), MPI_BYTE, 0, MYMPI_COMM_WORLD);

    if (header.num_files > 0)
    {
        return header.num_files;
    }

    if (ThisTask == 0)
    {
        if ((fd = fopen(buf1, "r")))
        {
            if (All.ICFormat == 1 || All.ICFormat == 2)
            {
                if (All.ICFormat == 2)
                {
                    my_fread(&dummy, sizeof(dummy), 1, fd);
                    swap_file = dummy;
                    my_fread(&dummy, sizeof(dummy), 1, fd);
                    my_fread(&dummy, sizeof(dummy), 1, fd);
                    my_fread(&dummy, sizeof(dummy), 1, fd);
                }

                my_fread(&dummy, sizeof(dummy), 1, fd);
                if (All.ICFormat == 1)
                {
                    if (dummy == 256)
                    {
                        swap_file = 8;
                    }
                    else
                    {
                        swap_file = 1;
                    }
                }
                my_fread(&header, sizeof(header), 1, fd);
                swap_header();

#ifdef PATCH_IO
                VERBOSE(2, "Task %d: WARNING, PATCHING HEADER FOR MagneticumDM simulation !!! ", ThisTask);
                header.mass[2] = header.mass[1];
                header.mass[1] = header.mass[0];
                header.mass[0] = 0;
                header.npart[2] = header.npart[1];
                header.npart[1] = header.npart[0];
                header.npart[0] = 0;
                header.npartTotal[2] = header.npartTotal[1];
                header.npartTotal[1] = header.npartTotal[0];
                header.npartTotal[0] = 0;
                header.npartTotalHighWord[2] = header.npartTotalHighWord[1];
                header.npartTotalHighWord[1] = header.npartTotalHighWord[0];
                header.npartTotalHighWord[0] = 0;
#endif // PATCH_IO
                my_fread(&dummy, sizeof(dummy), 1, fd);
            }
            fclose(fd);

#ifdef HAVE_HDF5
            if (All.ICFormat == 3)
            {
                read_header_attributes_in_hdf5(buf1);
            }
#endif // HAVE_HDF5
            header.num_files = 1;
        }
    }

    MPI_Bcast(&swap_file, 1, MPI_INT, 0, MYMPI_COMM_WORLD);
    MPI_Bcast(&header, sizeof(header), MPI_BYTE, 0, MYMPI_COMM_WORLD);

    if (header.num_files > 0)
    {
        return header.num_files;
    }

    VERBOSE(0, "\nCan't find initial conditions file.");
    VERBOSE(0, "neither as '%s'\nnor as '%s' ", buf, buf1);

    ENDRUN_MAIN("Failed to read initial conditions file.");
    return 0;
}

/**
 * @brief Distribute files to processors
 *
 * This function assigns a certain number of files to processors, such that
 * each processor is exactly assigned to one file, and the number of cpus per
 * file is as homogenous as possible. The number of files may at most be
 * equal to the number of processors.
 *
 * @param nfiles Number of files to distribute
 * @param firstfile First file number
 * @param firsttask First task number
 * @param lasttask Last task number
 * @param filenr Pointer to the file number assigned to the current task
 * @param master Pointer to the master task number
 * @param last Pointer to the last task number
 */
void distribute_file(int nfiles, int firstfile, int firsttask, int lasttask, int *filenr, int *master, int *last)
{

    if (nfiles > 1)
    {
        int ntask = lasttask - firsttask + 1;

        int filesleft = (int)((((double)(ntask / 2)) / ntask) * nfiles);
        if (filesleft <= 0)
        {
            filesleft = 1;
        }
        if (filesleft >= nfiles)
        {
            filesleft = nfiles - 1;
        }

        int filesright = nfiles - filesleft;

        int tasksleft = ntask / 2;
        distribute_file(filesleft, firstfile, firsttask, firsttask + tasksleft - 1, filenr, master, last);
        distribute_file(filesright, firstfile + filesleft, firsttask + tasksleft, lasttask, filenr, master, last);
    }
    else
    {
        if (ThisTask >= firsttask && ThisTask <= lasttask)
        {
            *filenr = firstfile;
            *master = firsttask;
            *last = lasttask;
        }
    }
}

#ifdef HAVE_HDF5
/**
 * @brief Read header attributes in HDF5 format
 *
 * This function reads the header attributes from an HDF5 file.
 *
 * @param fname Filename of the HDF5 file
 */
void read_header_attributes_in_hdf5(char *fname)
{
    hid_t hdf5_file, hdf5_headergrp, hdf5_attribute;

    hdf5_file = H5Fopen(fname, H5F_ACC_RDONLY, H5P_DEFAULT);

    hdf5_headergrp = H5Gopen2(hdf5_file, "/Header", H5P_DEFAULT);

    hdf5_attribute = H5Aopen_name(hdf5_headergrp, "NumPart_ThisFile");
    H5Aread(hdf5_attribute, H5T_NATIVE_INT, header.npart);
    H5Aclose(hdf5_attribute);

    hdf5_attribute = H5Aopen_name(hdf5_headergrp, "NumPart_Total");
    H5Aread(hdf5_attribute, H5T_NATIVE_UINT, header.npartTotal);
    H5Aclose(hdf5_attribute);

    hdf5_attribute = H5Aopen_name(hdf5_headergrp, "NumPart_Total_HighWord");
    H5Aread(hdf5_attribute, H5T_NATIVE_UINT, header.npartTotalHighWord);
    H5Aclose(hdf5_attribute);

    hdf5_attribute = H5Aopen_name(hdf5_headergrp, "MassTable");
    H5Aread(hdf5_attribute, H5T_NATIVE_DOUBLE, header.mass);
    H5Aclose(hdf5_attribute);

    hdf5_attribute = H5Aopen_name(hdf5_headergrp, "Time");
    H5Aread(hdf5_attribute, H5T_NATIVE_DOUBLE, &header.time);
    H5Aclose(hdf5_attribute);

    hdf5_attribute = H5Aopen_name(hdf5_headergrp, "BoxSize");
    H5Aread(hdf5_attribute, H5T_NATIVE_DOUBLE, &header.BoxSize);
    H5Aclose(hdf5_attribute);

    hdf5_attribute = H5Aopen_name(hdf5_headergrp, "NumFilesPerSnapshot");
    H5Aread(hdf5_attribute, H5T_NATIVE_INT, &header.num_files);
    H5Aclose(hdf5_attribute);

    hdf5_attribute = H5Aopen_name(hdf5_headergrp, "Flag_IC_Info");
    H5Aread(hdf5_attribute, H5T_NATIVE_INT, &header.flag_ic_info);
    H5Aclose(hdf5_attribute);

    hdf5_attribute = H5Aopen_name(hdf5_headergrp, "Flag_DoublePrecision");
    H5Aread(hdf5_attribute, H5T_NATIVE_INT, &header.flag_doubleprecision);
    H5Aclose(hdf5_attribute);

    H5Gclose(hdf5_headergrp);
    H5Fclose(hdf5_file);
}
#endif // HAVE_HDF5

/**
 * @brief Swap endianess of data
 *
 * This function swaps the endianess of a given data array.
 *
 * @param data Pointer to the data
 * @param n Number of elements to swap
 * @param m Size of single element to swap
 *
 * @note: int,float = 4, double = 8
 */
void swap_Nbyte(char *data, int n, int m)
{
    char old_data[16];

    if (swap_file != 8)
    {
        for (int j = 0; j < n; j++)
        {
            memcpy(&old_data[0], &data[j * m], m);
            for (int i = 0; i < m; i++)
            {
                data[j * m + i] = old_data[m - i - 1];
            }
        }
    }
}

/**
 * @brief Swap header endianess
 *
 * This function swaps the endianess of the header.
 */
void swap_header()
{
    swap_Nbyte((char *)&header.npart, 6, 4);
    swap_Nbyte((char *)&header.mass, 6, 8);
    swap_Nbyte((char *)&header.time, 1, 8);
    swap_Nbyte((char *)&header.redshift, 1, 8);
    swap_Nbyte((char *)&header.flag_sfr, 1, 4);
    swap_Nbyte((char *)&header.flag_feedback, 1, 4);
    swap_Nbyte((char *)&header.npartTotal, 6, 4);
    swap_Nbyte((char *)&header.flag_cooling, 1, 4);
    swap_Nbyte((char *)&header.num_files, 1, 4);
    swap_Nbyte((char *)&header.BoxSize, 1, 8);
    swap_Nbyte((char *)&header.Omega0, 1, 8);
    swap_Nbyte((char *)&header.OmegaLambda, 1, 8);
    swap_Nbyte((char *)&header.HubbleParam, 1, 8);
    swap_Nbyte((char *)&header.flag_stellarage, 1, 4);
    swap_Nbyte((char *)&header.flag_metals, 1, 4);
    swap_Nbyte((char *)&header.npartTotalHighWord, 6, 4);
    swap_Nbyte((char *)&header.flag_entropy_instead_u, 1, 4);
    swap_Nbyte((char *)&header.flag_doubleprecision, 1, 4);
    swap_Nbyte((char *)&header.flag_ic_info, 1, 4);
    swap_Nbyte((char *)&header.lpt_scalingfactor, 1, 4);
}

/**
 * @brief Find a block in a snapshot file
 *
 * This function searches for a block in a snapshot file.
 *
 * @param label 4 byte identifier for block
 * @param fd File handle
 */
void find_block(char *label, FILE *fd)
{
    unsigned int blocksize = 0, blksize;
    char blocklabel[5] = {"    "};

    rewind(fd);

#ifdef LT_TRACK_CONTRIBUTES
    int blcksize;
    if (strcmp(label, "TRCK") == 0)
    {
        fread(&blcksize, sizeof(int), 1, fd);
        blcksize += sizeof(int);
        fseeko(fd, blcksize, SEEK_CUR);
    }
#endif // LT_TRACK_CONTRIBUTES

    while (!feof(fd) && blocksize == 0)
    {
        skip_read(&blksize, fd);
        swap_file = blksize;
        swap_Nbyte((char *)&blksize, 1, 4);

        if (blksize != 8)
        {
            ENDRUN("Incorrect Format (blksize=%u)!\n", blksize);
        }
        else
        {
            my_fread(blocklabel, 4 * sizeof(char), 1, fd);
            my_fread(&blocksize, sizeof(int), 1, fd);
            swap_Nbyte((char *)&blocksize, 1, 4);

            /*
               VERBOSE_ALL(2,"Searching <%c%c%c%c>, found Block <%s> with %d bytes\n",
               label[0],label[1],label[2],label[3],blocklabel,blocksize);
             */
            skip_read(&blksize, fd);
            if (strncmp(label, blocklabel, 4) != 0)
            {
                fseeko(fd, blocksize, 1);
                blocksize = 0;
            }
        }
    }
    if (feof(fd))
    {
        ENDRUN_MAIN("Block '%c%c%c%c' not found ! ", label[0], label[1], label[2], label[3]);
    }
}
