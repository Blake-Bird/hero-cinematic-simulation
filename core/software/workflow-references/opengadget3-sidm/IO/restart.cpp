/**
 * @file
 *
 * This file is part of the N-body/SPH code OpenGadget3.
 *
 * @brief This file contains functions for the restarting.
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
#include <cstdio>
#include <math.h>
#include <errno.h>
#include <fstream>

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/file.h>
#include <unistd.h>

#include "../CodeBase/allvars.hpp"
#include "../CodeBase/proto.hpp"
#include "../CodeBase/domain.hpp"
#include "io.hpp"

static FILE *fd;

static void in(int *x, int modus);
static void byten(void *x, size_t n, int modus);

int old_MaxPart = 0, new_MaxPart;

/**
 * @brief Joins a base path and a subpath into a new path.
 *
 * This function appends the subpath to the base path, ensuring there is
 * exactly one '/' character separating them. If the base is empty, the
 * subpath is returned with a '/' prefix.
 *
 * @param base The base path string
 * @param subpath The subpath to append
 * @return A new path string combining base and subpath
 */
std::string create_path(const std::string &base, const std::string &subpath)
{
    // if string is empty
    if (base.empty())
    {
        return "/" + subpath;
    }

    // if string ends with / then add subpath directly. Else add / and subpath
    if (base.back() == '/')
    {
        return base + subpath;
    }
    else
    {
        return base + "/" + subpath;
    }
}

/**
 * @brief Restarts the simulation.
 *
 * This function reads or writes the restart files.
 * Each processor writes its own restart file, with the
 * I/O being done in parallel. To avoid congestion of the disks
 * you can tell the program to restrict the number of files
 * that are simultaneously written to NumFilesWrittenInParallel.
 *
 * If modus>0 the restart()-routine reads,
 * if modus==0 it writes a restart file.
 *
 * @param modus The mode of operation: 0 for writing, >0 for reading.
 */
void restart(int modus)
{
    if (modus > 0)
    {
        VERBOSE(0, "Restarting the code with modus=%d", modus);
    }

    char buf[200], buf_bak[200];
    double save_PartAllocFactor, save_MaxTreeAllocFactor;
    int nprocgroup, masterTask;
    struct global_data_all_processes all_task0;
    int nmulti = Multipledomains;

#ifdef SFR
    double safe_SFfactor;
    int buffer;
#endif // SFR

#if defined(LT_STELLAREVOLUTION)
    int save_NumFilesPerSnapshot, save_NumFilesWrittenInParallel;
    double save_LLv_Step_Prec, save_SnII_Step_Prec;
    double save_SofteningGasMaxPhys, save_SofteningHaloMaxPhys, save_SofteningDiskMaxPhys, save_SofteningBulgeMaxPhys,
        save_SofteningStarsMaxPhys, save_SofteningBndryMaxPhys;
    double save_MinChemTimeStep, save_MinChemSpreadL, save_MaxChemSpreadL;
#endif // LT_STELLAREVOLUTION

#ifdef BLACK_HOLES
    int bhbuffer;
#endif // BLACK_HOLES

    // Path/name for restart dir
    std::string restart_dir = create_path(All.OutputDir, "restartfiles");

    // Create directory for restart files if necessary
    if (ThisTask == 0 && modus == 0)
    {
        sprintf(buf, "%s", restart_dir.c_str());
        mkdir(buf, 02755);
    }
    MPI_Barrier(MYMPI_COMM_WORLD);

#ifndef SWITCH_TIMER_OFF
    restart_timer(modus);
#endif // SWITCH_TIMER_OFF

    TIMER_PUSH(0, "IO");
    if (modus == 0)
    {
        TIMER_PUSH(0, "RESTART_WRITE");
    }
    else if (modus == 1)
    {
        TIMER_PUSH(0, "RESTART_READ");
    }

    // Create the restart file names and their bakup file names
    sprintf(buf, "%s/%s.%d", restart_dir.c_str(), All.RestartFile, ThisTask);
    sprintf(buf_bak, "%s/%s.%d.bak", restart_dir.c_str(), All.RestartFile, ThisTask);

    // Check if the number of processors exceeds the limit for parallel file writing
    ENDRUN_IF(NTask < All.NumFilesWrittenInParallel,
              "Fatal error.\nNumber of processors must be a smaller or equal than `NumFilesWrittenInParallel'. ");
    // Determine the number of processors in each group for parallel I/O
    nprocgroup = NTask / All.NumFilesWrittenInParallel;

    if ((NTask % All.NumFilesWrittenInParallel))
    {
        nprocgroup++;
    }

    masterTask = (ThisTask / nprocgroup) * nprocgroup;
    // Loop over processor groups for file operations
    for (int groupTask = 0; groupTask < nprocgroup; groupTask++)
    {
        // Move old restart files to .bak files
        if (ThisTask == (masterTask + groupTask))
        {
            // when writing, create the .bak files
            if (modus == 0)
            {
#ifdef NOCALLSOFSYSTEM
                char buf_mv[500];
                /* move old restart files to .bak files */
                sprintf(buf_mv, "mv %s %s", buf, buf_bak);
                int ret = system(buf_mv);
#else // NOCALLSOFSYSTEM
                if (rename(buf, buf_bak) != 0)
                {
                    VERBOSE(0, "WARNING: Could not back up restart file %s\n%s ", buf, strerror(errno));
                }
#endif // NOCALLSOFSYSTEM
            }
        }
    }
    // Loop over processor groups for reading or writing restart files
    for (int groupTask = 0; groupTask < nprocgroup; groupTask++)
    {
        // Processor-specific operations
        if (ThisTask == (masterTask + groupTask)) /* ok, it's this processor's turn */
        {
            // Open restart file for reading or writing
            // When open for reading, per default take the normal ones but if not present,
            // check if the back files are available
            if (modus > 0)
            {
                if (!(fd = fopen(buf, "r")))
                {
                    if (!(fd = fopen(buf_bak, "r")))
                    {
                        ENDRUN("Restart file '%s' nor '%s' found. ", buf, buf_bak);
                    }
                }
            }
            else
            {
                if (!(fd = fopen(buf, "w")))
                {
                    ENDRUN("Restart file '%s' cannot be opened. ", buf);
                }
            }
            save_MaxTreeAllocFactor = All.MaxTreeAllocFactor;

            // Save relevant data if writing restart file
            save_PartAllocFactor = All.PartAllocFactor;
            save_MaxTreeAllocFactor = All.MaxTreeAllocFactor;

#ifdef SFR
            safe_SFfactor = All.SFfactor;
#endif // SFR

#ifdef LT_STELLAREVOLUTION
            // Save relevant data for stellar evolution
            save_NumFilesPerSnapshot = All.NumFilesPerSnapshot;
            save_NumFilesWrittenInParallel = All.NumFilesWrittenInParallel;

            save_SofteningGasMaxPhys = All.SofteningGasMaxPhys;
            save_SofteningHaloMaxPhys = All.SofteningHaloMaxPhys;
            save_SofteningDiskMaxPhys = All.SofteningDiskMaxPhys;
            save_SofteningBulgeMaxPhys = All.SofteningBulgeMaxPhys;
            save_SofteningStarsMaxPhys = All.SofteningStarsMaxPhys;
            save_SofteningBndryMaxPhys = All.SofteningBndryMaxPhys;

            save_SnII_Step_Prec = All.SnII_Step_Prec;
            save_LLv_Step_Prec = All.LongLiving_Step_Prec;

            save_MinChemTimeStep = All.MinChemTimeStep;
            save_MinChemSpreadL = All.MinSpreadLength;
            save_MaxChemSpreadL = All.MaxChemSpreadL;

            if (!modus)
            {
                *(float *)&buffer = (float)All.Time;
            }
            in(&buffer, modus);
            if (!modus)
            {
                buffer = sizeof(struct global_data_all_processes);
            }
            in(&buffer, modus);
            if (modus && buffer != sizeof(struct global_data_all_processes))
            {
                ENDRUN("in file <%s> :: sizes of the current All structure and of the stored All structure are "
                       "different! (%d vs %d bytes) ",
                       buf, (int)buffer, (int)sizeof(struct global_data_all_processes));
            }
#endif // LT_STELLAREVOLUTION

            /* common data  */
            byten(&All, sizeof(struct global_data_all_processes), modus);

#ifdef LT_STELLAREVOLUTION
            // Restore relevant data for stellar evolution
            All.NumFilesPerSnapshot = save_NumFilesPerSnapshot;
            All.NumFilesWrittenInParallel = save_NumFilesWrittenInParallel;

            All.SnII_Step_Prec = save_SnII_Step_Prec;
            All.LongLiving_Step_Prec = save_LLv_Step_Prec;
            All.MinChemTimeStep = save_MinChemTimeStep;

            All.SofteningGasMaxPhys = save_SofteningGasMaxPhys;
            All.SofteningHaloMaxPhys = save_SofteningHaloMaxPhys;
            All.SofteningDiskMaxPhys = save_SofteningDiskMaxPhys;
            All.SofteningBulgeMaxPhys = save_SofteningBulgeMaxPhys;
            All.SofteningStarsMaxPhys = save_SofteningStarsMaxPhys;
            All.SofteningBndryMaxPhys = save_SofteningBndryMaxPhys;

            All.MinSpreadLength = save_MinChemSpreadL;
            All.MaxChemSpreadL = save_MaxChemSpreadL;
#endif // LT_STELLAREVOLUTION

            if (ThisTask == 0 && modus > 0)
            {
                all_task0 = All;
            }
            // Broadcast data from task 0 if reading restart files
            if (modus > 0 && groupTask == 0) /* read */
            {
                MPI_Bcast(&all_task0, sizeof(struct global_data_all_processes), MPI_BYTE, 0, MYMPI_COMM_WORLD);
            }

            // Continue reading data based on modus
            if (modus)
            {
                if (All.MaxTreeAllocFactor != save_MaxTreeAllocFactor)
                {
                    VERBOSE(2, "MaxTreeAllocFactor changed: %f/%f , adapting bounds ... ", All.MaxTreeAllocFactor,
                            save_PartAllocFactor);

                    All.MaxTreeAllocFactor = save_MaxTreeAllocFactor;
                }
                if (All.PartAllocFactor != save_PartAllocFactor)
                {
                    old_MaxPart = All.MaxPart; /* old MaxPart */

                    VERBOSE(2, "PartAllocFactor changed: %f/%f , adapting bounds ... ", All.PartAllocFactor,
                            save_PartAllocFactor);

                    All.PartAllocFactor = save_PartAllocFactor;
                    All.MaxPart = (int)(All.PartAllocFactor * (All.TotNumPart / NTask));
                    All.MaxPartSph = (int)(All.PartAllocFactor * (All.TotN_gas / NTask));
#ifdef INHOMOG_GASDISTR_HINT
                    All.MaxPartSph = All.MaxPart;
#endif // INHOMOG_GASDISTR_HINT

                    new_MaxPart = All.MaxPart;

#ifdef SFR
                    VERBOSE(2, "All.TotNumPart=%llu, All.TotN_gas=%llu, All.TotN_stars=%llu  ",
                            (unsigned long long)All.TotNumPart, (unsigned long long)All.TotN_gas,
                            (unsigned long long)All.TotN_stars);
                    if (All.TotN_stars == 0)
                    {
                        All.MaxPartMet = All.PartAllocFactor * (All.TotN_gas / NTask) * safe_SFfactor * All.Generations;
                    }
                    else
                    {
                        All.MaxPartMet =
                            All.PartAllocFactor *
                            (All.TotN_stars / NTask + (All.TotN_gas / NTask) * safe_SFfactor * All.Generations);
                    }
                    VERBOSE(2, "All.MaxPart=%llu, All.MaxPartSph=%llu, All.MaxPartMet=%llu  ",
                            (unsigned long long)All.MaxPart, (unsigned long long)All.MaxPartSph,
                            (unsigned long long)All.MaxPartMet);
                    All.SFfactor = safe_SFfactor;
#endif // SFR

#ifdef BLACK_HOLES
                    if (All.TotBHs == 0)
                    {
                        All.MaxPartBH = All.PartAllocFactor * (All.TotN_gas / NTask) * All.BlackHoleFormationFactor;
                    }
                    else
                    {
                        All.MaxPartBH = All.PartAllocFactor *
                                        (All.TotBHs / NTask + (All.TotN_gas / NTask) * All.BlackHoleFormationFactor);
                    }
#endif // BLACK_HOLES
                    save_PartAllocFactor = -1;
                }

#ifdef SFR
                VERBOSE(2, "SFfactor changed: %f/%f , adapting bounds ... ", All.SFfactor, safe_SFfactor);

                if (All.SFfactor != safe_SFfactor)
                {
                    All.MaxPartMet = (All.MaxPartMet / All.SFfactor) * safe_SFfactor;
                    All.SFfactor = safe_SFfactor;
                }
#endif // SFR

                ENDRUN_IF(all_task0.Time != All.Time,
                          "The restart file on task=%d is not consistent with the one on task=0 ", ThisTask);

                allocate_memory();
            }

            in(&NumPart, modus);

#ifdef LT_STELLAREVOLUTION
            if (!modus)
            {
                buffer = sizeof(struct particle_data);
            }
            in(&buffer, modus);
            if (modus && buffer != sizeof(struct particle_data))
            {
                ENDRUN("in file <%s> :: sizes of the current particle data and of the stored particle data are "
                       "different! (%d vs %d bytes) ",
                       buf, buffer, (int)sizeof(struct particle_data));
            }
#endif // LT_STELLAREVOLUTION

            ENDRUN_IF(NumPart > All.MaxPart,
                      " Error: It seems you have reduced(!) 'PartAllocFactor' below the value of %g needed to load the "
                      "restart "
                      "file. ",
                      NumPart / (((double)All.TotNumPart) / NTask));

            if (modus) /* read */
            {
                if (old_MaxPart)
                {
                    All.MaxPart = old_MaxPart; /* such that tree is still valid */
                }
            }

            /* Particle data  */
            byten(&P[0], (size_t)NumPart * sizeof(struct particle_data), modus);

            in(&N_gas, modus);
#ifdef LT_STELLAREVOLUTION
            if (!modus)
            {
                buffer = sizeof(struct sph_particle_data);
            }
            in(&buffer, modus);
            if (modus && buffer != sizeof(struct sph_particle_data))
            {
                ENDRUN("in file <%s> :: sizes of the current sph particle data and of the stored sph particle data "
                       "are different! (%d vs %d bytes) ",
                       buf, buffer, (int)sizeof(struct sph_particle_data));
            }
#endif // LT_STELLAREVOLUTION
            if (N_gas > 0)
            {
                ENDRUN_IF(N_gas > All.MaxPartSph,
                          "Fatal Error:\nSPH: it seems you have reduced(!) 'PartAllocFactor' below the value of %g "
                          "needed to load the restart file. ",
                          N_gas / (((double)All.TotN_gas) / NTask));
                /* Sph-Particle data  */
                byten(&SphP[0], (size_t)N_gas * sizeof(struct sph_particle_data), modus);
            }

#ifdef AR_SOA_LITE
#ifndef AR_SOA_LITE_FROM_RESTART_AOS
            // unfortunately we must clone the byten for each property
            byten(SphPSoa.wakeup, (size_t)N_gas * sizeof(short int), modus);
#endif // AR_SOA_LITE_FROM_RESTART_AOS
#endif // AR_SOA_LITE
#ifdef AR_SOA_LITE_FROM_RESTART_AOS // if you want to restart with soa-lite active
            for (int i = 0; i < N_gas; i++)
            {
                SPHP(i, wakeup) = SphP[i].wakeup;
            }
#endif // AR_SOA_LITE_FROM_RESTART_AOS

            /* write state of random number generator */
            char rng_state_buf[RNG_STATE_BUFSIZE];
            if (!modus)
            {
                rng_state_to_buffer(rng_state_buf);
            }
            byten(rng_state_buf, RNG_STATE_BUFSIZE, modus);
            if (modus)
            {
                rng_state_from_buffer(rng_state_buf);
            }
            byten(&SelRnd, sizeof(SelRnd), modus);

            /* write flags for active timebins */
            byten(TimeBinActive, TIMEBINS * sizeof(int), modus);

            /* now store relevant data for tree */
#if defined(SFR) || defined(FG_MAKE_OUTSIDE_HIGHRES_DM) || defined(FG_MAKE_MAXVEL_DM)
            in(&Particles_converted, modus);
#endif // defined(SFR) || defined(FG_MAKE_OUTSIDE_HIGHRES_DM) || defined(FG_MAKE_MAXVEL_DM)

#ifdef SFR
            in(&N_stars, modus);
            if (!modus)
            {
                buffer = sizeof(struct met_particle_data);
            }
            in(&buffer, modus);
            if (modus && buffer != sizeof(struct met_particle_data))
            {
                ENDRUN("in file <%s> :: sizes of the current (star) met particle data and of the stored (star) met "
                       "particle data are different! (%d vs %d bytes) ",
                       buf, buffer, (int)sizeof(struct met_particle_data));
            }

            if (N_stars > 0)
            {
                if (N_stars > All.MaxPartMet)
                {
                    ENDRUN("MET: it seems you have reduced(!) 'PartAllocFactor' below the value of %g needed to "
                           "load the restart file. ",
                           N_stars / (((double)All.TotN_stars) / NTask));
                }
                /* Sph-Particle data  */
                byten(&MetP[0], (size_t)N_stars * sizeof(struct met_particle_data), modus);
            }
#endif // SFR
#ifdef LT_STELLAREVOLUTION
            byten(&All.MetalsCheckSum, sizeof(long double) * LT_NMetP, modus);
#endif // LT_STELLAREVOLUTION

#ifdef BLACK_HOLES
            in(&N_BHs, modus);
            if (!modus)
            {
                bhbuffer = sizeof(struct bh_particle_data);
            }
            in(&bhbuffer, modus);
            if (modus && bhbuffer != sizeof(struct bh_particle_data))
            {
                ENDRUN("in file <%s> :: sizes of the current bh particle data and of the stored bh particle data "
                       "are different! (%d vs %d bytes) ",
                       buf, bhbuffer, (int)sizeof(struct bh_particle_data));
            }

            if (N_BHs > 0)
            {
                if (N_BHs > All.MaxPartBH)
                {
                    ENDRUN("BH: it seems you have reduced(!) 'PartAllocFactor' below the value of %g needed to "
                           "load the restart file. ",
                           N_BHs / (((double)All.TotBHs) / NTask));
                }
                /* Sph-Particle data  */
                byten(&BHP[0], (size_t)N_BHs * sizeof(struct bh_particle_data), modus);
            }
#endif // BLACK_HOLES

            /* now store relevant data for tree */
            in(&nmulti, modus);
            if (modus != 0 && nmulti != Multipledomains)
            {
                VERBOSE(2,
                        "Looks like you changed MULTIPLEDOMAINS from %d to %d.\nWe will need to discard tree stored in "
                        "restart files and construct a new one. ",
                        nmulti, Multipledomains);

                /* In this case we must do a new domain decomposition! */
            }
            else
            {
#ifndef NOTREEINRESTARTFILE
                in(&NTopleaves, modus);
                in(&NTopnodes, modus);

                if (modus) /* read */
                {
                    domain_allocate();
                    force_treeallocate((int)(All.TreeAllocFactor * All.MaxPart) + NTopnodes, All.MaxPart);
                }

                in(&Numnodestree, modus);

                ENDRUN_IF(Numnodestree > MaxNodes,
                          "Tree storage: it seems you have reduced(!) 'PartAllocFactor' below the value needed to load "
                          "the restart file (task=%d). "
                          "Numnodestree=%d  MaxNodes=%d ",
                          ThisTask, Numnodestree, MaxNodes);

                byten(Nodes_base, Numnodestree * sizeof(struct NODE), modus);
                byten(Extnodes_base, Numnodestree * sizeof(struct extNODE), modus);

                byten(Father, NumPart * sizeof(int), modus);
                byten(Nextnode, NumPart * sizeof(int), modus);
                byten(Nextnode + All.MaxPart, NTopnodes * sizeof(int), modus);

                byten(DomainStartList, NTask * Multipledomains * sizeof(int), modus);
                byten(DomainEndList, NTask * Multipledomains * sizeof(int), modus);
                byten(TopNodes, NTopnodes * sizeof(struct topnode_data), modus);
                byten(DomainTask, NTopnodes * sizeof(int), modus);
                byten(DomainNodeIndex, NTopleaves * sizeof(int), modus);

                byten(DomainCorner, 3 * sizeof(double), modus);
                byten(DomainCenter, 3 * sizeof(double), modus);
                byten(&DomainLen, sizeof(double), modus);
                byten(&DomainFac, sizeof(double), modus);
#endif // NOTREEINRESTARTFILE
            }

            fclose(fd);
        }
        else /* wait inside the group */
        {
            if (modus > 0 && groupTask == 0) /* read */
            {
                MPI_Bcast(&all_task0, sizeof(struct global_data_all_processes), MPI_BYTE, 0, MYMPI_COMM_WORLD);
            }
        }

        MPI_Barrier(MYMPI_COMM_WORLD);
    }

#ifdef DEBUG
    // Additional checke if DEBUG is activated
    if (modus != 0)
    {
        VERBOSE(2, "Check particle integrety after restart ...");
#ifdef _OPENMP
#pragma omp parallel for
#endif // _OPENMP
        for (int i = 0; i < NumPart; i++)
        {
            if (P[i].Type > 5 || P[i].Type < 0)
            {
                ENDRUN("Task=%d:  P[i=%d].Type=%d", ThisTask, i, P[i].Type);
            }
#ifdef LT_STELLAREVOLUTION
            if (P[i].Type == 4)
            {
                if (MetP[P[i].pt.MetID].PID != i)
                {
                    ENDRUN("Task=%d:  error in cross-indexes for star-particle %d ID %llu MetID %d PID[MetID] %d",
                           ThisTask, i, (unsigned long long)P[i].ID, P[i].pt.MetID, MetP[P[i].pt.MetID].PID);
                }
            }
#endif // LT_STELLAREVOLUTION

#ifdef BLACK_HOLES
            if (P[i].Type == 5)
            {
                if (BHP[P[i].pt.BHID].PID != i)
                {
                    ENDRUN("Task=%d:  error in cross-indexes for bh-particle %d ID %llu", ThisTask, i,
                           (unsigned long long)P[i].ID);
                }
            }
#endif // BLACK_HOLES
        }
    }
#endif // DEBUG

    // Additional operations in case domains have changed
    if (modus != 0)
    {
#ifndef NOTREEINRESTARTFILE
        if (nmulti != Multipledomains) /* in this case we must force a domain decomposition */
#endif // NOTREEINRESTARTFILE
        {
            VERBOSE(2, "Doing extra domain decomposition ");

            if (All.ComovingIntegrationOn)
            {
                init_drift_table();
            }

            domain_Decomposition(0, 0);
            if (TreeReconstructFlag)
            {
                VERBOSE(3, "Tree construction.  (presently allocated=%g MB) ", AllocatedBytes / (1024.0 * 1024.0));
                force_treebuild(NumPart, NULL);

                TreeReconstructFlag = 0;

                VERBOSE(4, "Tree construction done. ");
            }
        }
    }

    if (modus == 0)
    {
        TIMER_POP(0, "RESTART_WRITE");
    }
    else if (modus == 1)
    {
        TIMER_POP(0, "RESTART_READ");
    }
    TIMER_POP(0, "IO");
}

#ifndef SWITCH_TIMER_OFF
/**
 * @brief Copies the timer restart file to a backup file.
 *
 * This function copies the timer restart file to a backup file.
 * It takes the input filename and adds a .bak extension to it.
 *
 * @param source_fname Name of the file to be copied
 */
void copy_timer_restart_to_bak(std::string &source_fname)
{
    VERBOSE(3, "Creating backup files for timer restart file: %s", source_fname.c_str());
    // Add .bak extension
    std::string backup_file = source_fname + ".bak";
    // rename file to backup file
    if (std::rename(source_fname.c_str(), backup_file.c_str()) == 0)
    {
        VERBOSE(2, "File renamed successfully from %s to %s .", source_fname.c_str(), backup_file.c_str());
    }
    else
    {
        ENDRUN("Error renaming timer restart file from %s to %s.\n", source_fname.c_str(), backup_file.c_str());
    }
    VERBOSE(4, "Done creating backup files for timer restart file: %s", source_fname.c_str());
}

/**
 * @brief Checks if a file exists.
 *
 * This function attempts to open the specified file and checks
 * whether it can be successfully accessed.
 *
 * @param filename The name (and optionally path) of the file to check.
 * @return `true` if the file exists and can be opened; `false` otherwise.
 *
 * @note This method relies on the ability to open the file using
 *       std::ifstream. It does not verify permissions beyond readability.
 *       Hidden or locked files might affect the result.
 */
bool fileExists(const std::string &filename)
{
    std::ifstream file(filename);
    return file.good(); // Returns true if the file can be opened
}

/**
 * @brief Writes or reads Timer class data for restart.
 *
 * This function is responsible for writing or reading Timer class data for restart purposes.
 * If modus is 0, it writes Timer class data to files. If modus is non-zero, it reads Timer class data from files.
 *
 * @param modus Flag indicating the mode of operation. If modus is 0, data is written; otherwise, data is read.
 */
void restart_timer(int modus)
{
    VERBOSE(3, "Writing/Restarting Timers.");
    // When writing, create the backup files
    if (modus == 0)
    {
        // Create output string stream
        std::ostringstream oss;
        oss << All.OutputDir << "/restartfiles/" << All.RestartFile;
        std::string file_base = oss.str();

        if (ThisTask == 0)
        {
            // Write GlobalTimer to file
            std::ostringstream oss_file;
            oss_file << ".GlobalTimers." << ThisTask;
            std::string filePath = file_base + oss_file.str();
            // check if its the first time to write the file/does it already exist
            if (fileExists(filePath))
            {
                // Backup the timer restart file
                copy_timer_restart_to_bak(filePath);
            }
            else
            {
                VERBOSE(2, "Timer restart file does not exist. Not creating backup file.");
            }
            GlobalTimer.serializeToFile(filePath);

            // Write timer_write_map to file for proper restart of balance file
            std::ostringstream oss_file_balance;
            oss_file_balance << ".BalanceMap." << ThisTask;
            filePath = file_base + oss_file_balance.str();

            // check if its the first time to write the file/does it already exist
            if (fileExists(filePath))
            {
                // Backup the timer restart file
                copy_timer_restart_to_bak(filePath);
            }
            else
            {
                VERBOSE(2, "Timer restart file does not exist for Task %i. Not creating backup file.", ThisTask);
            }
            TimestepTimer.write_balance_to_file(filePath);
        }
#ifdef TIMER_DETAILS
        oss << ".Timers." << ThisTask;
        std::string filePath = oss.str();
        // check if its the first time to write the file/does it already exist
        if (fileExists(filePath))
        {
            // Backup the timer restart file
            copy_timer_restart_to_bak(filePath);
        }
        else
        {
            VERBOSE(2, "Timer restart file does not exist for Task %i. Not creating backup file.", ThisTask);
        }
        TimestepTimer.serializeToFile(filePath);
#endif // TIMER_DETAILS
    }
    else
    {
        if (ThisTask == 0)
        {
            // reset timer to make sure timer is equal to values from restart file
            TIMER_RESET(GlobalTimer);

            // Create string streams
            std::ostringstream restart_iss_global;

            // Get file names
            restart_iss_global << All.OutputDir << "/restartfiles/" << All.RestartFile << ".GlobalTimers." << ThisTask;
            std::string restart_fname = restart_iss_global.str();
            int no_file_availiable = 0;
            // check if general restart files exist
            if (fileExists(restart_fname))
            {
                VERBOSE(2, "Reading Timer restart file: %s", restart_fname.c_str());
            }
            else
            {
                VERBOSE(0, "GlobalTimer restart file for Task %i not found. Trying backup file.", ThisTask);
                restart_fname += ".bak";

                if (fileExists(restart_fname))
                {
                    VERBOSE(2, "Reading Timer restart file: %s", restart_fname.c_str());
                }
                else
                {
                    VERBOSE(2, "Can't find backup file %s - starting from empty timers.", restart_fname.c_str());
                    no_file_availiable = 1;
                }
            }

            if (no_file_availiable == 0)
            {
                GlobalTimer.deserializeFromFile(restart_fname);
            }
            // Create string streams
            std::ostringstream restart_balance_iss;
            restart_balance_iss << All.OutputDir << "/restartfiles/" << All.RestartFile << ".BalanceMap." << ThisTask;
            restart_fname = restart_balance_iss.str();
            no_file_availiable = 0;
            // check if general restart files exist
            if (fileExists(restart_fname))
            {
                VERBOSE(2, "Reading Timer restart file: %s", restart_fname.c_str());
            }
            else
            {
                VERBOSE(0, "BalanceTimer restart file for Task %i not found. Trying backup file.", ThisTask);
                restart_fname += ".bak";

                if (fileExists(restart_fname))
                {
                    VERBOSE(2, "Reading Timer restart file: %s", restart_fname.c_str());
                }
                else
                {
                    VERBOSE(2, "Can't find backup file %s - starting from empty timers.", restart_fname.c_str());
                    no_file_availiable = 1;
                }
            }

            if (no_file_availiable == 0)
            {
                TimestepTimer.read_timer_write_map(restart_fname);
            }
        }
#ifdef TIMER_DETAILS
        // Create string streams
        std::ostringstream restart_iss_task;
        // Get file names
        restart_iss_task << All.OutputDir << "/restartfiles/" << All.RestartFile << ".Timers." << ThisTask;
        std::string restart_task_fname = restart_iss_task.str();
        int no_file_availiable = 0;
        // check if general restart files exist
        if (fileExists(restart_task_fname))
        {
            VERBOSE(2, "Reading Timer restart file: %s", restart_task_fname.c_str());
        }
        else
        {
            VERBOSE(0, "TimestepTimer restart file for Task %i not found. Trying backup file.", ThisTask);
            restart_task_fname += ".bak";

            if (fileExists(restart_task_fname))
            {
                VERBOSE(2, "Reading Timer restart file: %s", restart_task_fname.c_str());
            }
            else
            {
                VERBOSE(2, "Can't find backup file %s - starting from empty timers.", restart_task_fname.c_str());
                no_file_availiable = 1;
            }
        }

        if (no_file_availiable == 0)
        {
            TimestepTimer.deserializeFromFile(restart_task_fname);
        }

#endif // TIMER_DETAILS
    }
    VERBOSE(4, "Done Writing/Restarting Timers.");
}
#endif // SWITCH_TIMER_OFF

/**
 * @brief Reads or writes n bytes.
 *
 * This function reads or writes n bytes of data, depending on the specified mode.
 *
 * @param x Pointer to the memory location.
 * @param n Number of bytes to read or write.
 * @param modus Flag indicating the mode of operation. If modus is non-zero, it reads data; otherwise, it writes data.
 */
void byten(void *x, size_t n, int modus)
{
    if (modus)
    {
        my_fread(x, n, 1, fd);
    }
    else
    {
        my_fwrite(x, n, 1, fd);
    }
}

/**
 * @brief Reads or writes one integer.
 *
 * This function reads or writes one integer, depending on the specified mode.
 *
 * @param x Pointer to the integer.
 * @param modus Flag indicating the mode of operation. If modus is non-zero, it reads data; otherwise, it writes data.
 */
void in(int *x, int modus)
{
    if (modus)
    {
        my_fread(x, 1, sizeof(int), fd);
    }
    else
    {
        my_fwrite(x, 1, sizeof(int), fd);
    }
}
