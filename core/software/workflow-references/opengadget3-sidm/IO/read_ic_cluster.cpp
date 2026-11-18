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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>

#include "../CodeBase/allvars.hpp"
#include "../CodeBase/proto.hpp"
#include "io.hpp"

#ifdef SWAP_ENDIAN_READIC_FORMAT4
/**
 * @brief Swap the endianess of a data block.
 *
 * This routine swaps the endianess of a data block. It is used to
 *
 * @param data Pointer to the data block.
 * @param n Number of elements to swap.
 * @param m Size of a single element to swap.
 *
 * @note int,float = 4, double = 8
 */
void swap_Nbyte_format4(char *data, int n, int m)
{
    char old_data[16];

    for (int j = 0; j < n; j++)
    {
        memcpy(&old_data[0], &data[j * m], m);
        for (int i = 0; i < m; i++)
        {
            data[j * m + i] = old_data[m - i - 1];
        }
    }
}
#endif // SWAP_ENDIAN_READIC_FORMAT4

/**
 * @brief Read in initial conditions for a cluster.
 *
 * This is a special-purpose read-in routine for initial conditions
 * produced with Bepi Tormen's initial conditions generator ZIC.
 * Using this routine requires that you know pretty well what you're
 * doing...
 * Be aware that there are unit conversion factor `massfac', `posfac',
 * and `velfac', that have to be set appropriately.
 * Also note that there is a boundary for the intermediate resolution
 * zone set by hand below (to a value of 24000.0 in this example).
 *
 * @param fname The filename of the initial conditions file.
 */
void read_ic_cluster(char *fname)
{
#define BLOCKSIZE 500000
    FILE *fd = 0;
    float *fp, dummy[BLOCKSIZE][3];
    int groupsize;
    MPI_Status status;
    char buf[100];
    double massfac = 0, posfac = 0, velfac = 0;
    int nhr = 0, nlr = 0;
    int dummy4byte;

    /* Felix: start */
    char head_c[120]; /* one element of the string head array */
    double head_r[64]; /* the numerical head array */
    float rcut = 0, xcm = 0, ycm = 0, zcm = 0;
    /* Felix: end */

#define SKIP my_fread(&dummy4byte, sizeof(dummy4byte), 1, fd);

    /* Below, Bepi's new format is assumed !!!!!! */
    /* for the old one, the HR particle mass 'pmhr' has to be set
     * by hand!
     */
    if (ThisTask == 0)
    {
        for (int i = 0; i < 5; i++)
        {
            All.MassTable[i] = 0;
        }

        if (!(fd = fopen(fname, "r")))
        {
            ENDRUN("Can't open file `%s'. ", fname);
        }
        VERBOSE_ALL(2, "READING FILE '%s'.... ", fname);

        /* Bepi: start */
        /* read string header */
        SKIP;
        for (int j = 0; j < 16; j++)
        {
            my_fread(&head_c, 120, 1, fd);
            VERBOSE_ALL(2, "j, head_c[j]= %i  %s ", j, head_c);
        }
        SKIP;

        /* read numeric header */
        SKIP;
        my_fread(&head_r, sizeof(double), 64, fd);
        SKIP;

#ifdef SWAP_ENDIAN_READIC_FORMAT4
        swap_Nbyte_format4((char *)&head_r, 64, sizeof(double));
#endif // SWAP_ENDIAN_READIC_FORMAT4

        for (int j = 0; j < 64; j++)
        {
            VERBOSE_ALL(2, "j, head_r[j]= %i  %f ", j, head_r[j]);
        }

        /* assign numeric header to variables */
        nhr = (int)head_r[0];
        nlr = (int)head_r[1];
        /*
           a0 = 1. / (1. + head_r[2]);
         */
        float pmhr = head_r[3];
        posfac = 1000. * head_r[4] * 3.085678e21 / All.UnitLength_in_cm;
        massfac = head_r[6] / 1.e10 * 1.989e43 / All.UnitMass_in_g;
        velfac = 100.0 * head_r[4] * 1e5 / All.UnitVelocity_in_cm_per_s;
        rcut = 1. / head_r[17] * posfac;

        VERBOSE_ALL(2, ">>>>>>>>>>>>> Bepi: from Zic header I read:  ");
        VERBOSE_ALL(2, "massfac = %g ", massfac);
        VERBOSE_ALL(2, "posfac  = %g ", posfac);
        VERBOSE_ALL(2, "velfac  = %g ", velfac);
        VERBOSE_ALL(2, "rcut    = %g ", rcut);
        VERBOSE_ALL(2, "nhr     = %i ", nhr);
        VERBOSE_ALL(2, "nlr     = %i ", nlr);
        VERBOSE_ALL(2, ">>>>>>>>>>>>> end Bepi  ");

        /* Bepi: end */

        /* Felix start
         *
         * For twolevel resimulations the HR box is NOT in the center of the simulation
         * anymore. Therefore the automatic second shell particle searching with rcut
         * is not valid in this case.
         *
         * xcm are set outside the ThisTask loop and are
         */
        if (head_r[33] > 0.5)
        {
            VERBOSE_ALL(2, "########################################################## ");
            VERBOSE_ALL(2, "########################################################## ");
            VERBOSE_ALL(2, " This run is a twolevel run! The HR box is NOT centered on ");
            VERBOSE_ALL(2, " zero anymore. Have to use values SET BY HAND IN THE CODE! ");
            VERBOSE_ALL(2, " For later runs: add center of HR box into zic.dat file!!! ");
            VERBOSE_ALL(2, "########################################################## ");
            VERBOSE_ALL(2, "########################################################## ");

            xcm = -0.1546277106E-01 * posfac;
            ycm = -0.2029071748E-01 * posfac;
            zcm = 0.2206191421E-02 * posfac;
            /* rboxtwo */
            rcut = 1. / 45.41310883 * posfac;

            VERBOSE_ALL(2, "########################################################## ");
            VERBOSE_ALL(2, "########################################################## ");
            VERBOSE_ALL(2, " xcm = %e \n ycm = %e \n zcm = %e \n rcut = %e\n", xcm, ycm, zcm, rcut);
            VERBOSE_ALL(2, "########################################################## ");
            VERBOSE_ALL(2, "########################################################## ");
        }
        else
        {
#ifdef PERIODIC
            xcm = All.BoxSize / 2;
            ycm = All.BoxSize / 2;
            zcm = All.BoxSize / 2;
#else // PERIODIC
            xcm = 0.0;
            ycm = 0.0;
            zcm = 0.0;
#endif // PERIODIC
        }

        VERBOSE_ALL(2, "########################################################## ");
        VERBOSE_ALL(2, "########################################################## ");
        VERBOSE_ALL(2, "Blocksize set to 10000000 for HR and 1000000 for LR !!!!!! ");
        VERBOSE_ALL(2, "########################################################## ");
        VERBOSE_ALL(2, "########################################################## ");
        /* Felix end */

        All.MassTable[1] = pmhr * massfac; /* high-res particles */

        VERBOSE_ALL(2, "All.MassTable[1]=%g ", All.MassTable[1]);
        VERBOSE_ALL(2, "file contains %d HR and %d LR particles. ", nhr, nlr);

        All.TotN_gas = 0;
        All.TotNumPart = nhr + nlr;

        VERBOSE_ALL(2, "\nN_sph: %d\nN_halo: %d\nN_disk: %d \n", 0, nhr, nlr);
        fclose(fd);
    }

    MPI_Bcast(&posfac, 1, MPI_DOUBLE, 0, MYMPI_COMM_WORLD);
    MPI_Bcast(&velfac, 1, MPI_DOUBLE, 0, MYMPI_COMM_WORLD);
    MPI_Bcast(&massfac, 1, MPI_DOUBLE, 0, MYMPI_COMM_WORLD);
    MPI_Bcast(&rcut, 1, MPI_FLOAT, 0, MYMPI_COMM_WORLD);
    MPI_Bcast(&xcm, 1, MPI_FLOAT, 0, MYMPI_COMM_WORLD);
    MPI_Bcast(&ycm, 1, MPI_FLOAT, 0, MYMPI_COMM_WORLD);
    MPI_Bcast(&zcm, 1, MPI_FLOAT, 0, MYMPI_COMM_WORLD);

    MPI_Bcast(&All, sizeof(struct global_data_all_processes), MPI_BYTE, 0, MYMPI_COMM_WORLD);

    All.MaxPart = (int)(All.PartAllocFactor * (All.TotNumPart / NTask)); /* sets the maximum number of particles that
                                        may reside on a processor */

    allocate_memory();

    int pc = 0;
    int id = 1;
    int NumPart = 0;

    for (int files = 0; files <= 0; files++) /* only one file here */
    {
        int nhr_blocks, nlr_blocks;
        if (ThisTask == 0)
        {
            sprintf(buf, "%s", fname);

            if (!(fd = fopen(buf, "r")))
            {
                ENDRUN("Can't open file `%s'. ", buf);
            }

            /* Bepi: start */
            /* read string header */
            SKIP;
            for (int j = 0; j < 16; j++)
            {
                my_fread(&head_c, 120, 1, fd);
            }
            SKIP;

            /* read numeric header */
            SKIP;
            my_fread(&head_r, sizeof(double), 64, fd);
            SKIP;

#ifdef SWAP_ENDIAN_READIC_FORMAT4
            swap_Nbyte_format4((char *)&head_r, 64, sizeof(double));
#endif // SWAP_ENDIAN_READIC_FORMAT4

            nhr_blocks = nhr / BLOCKSIZE + 1;
            nlr_blocks = nlr / BLOCKSIZE + 1;
            VERBOSE_ALL(2, "nhr %i, nhr_blocks %i, nlr %i, nlr_blocks %i ", nhr, nhr_blocks, nlr, nlr_blocks);
        }

        MPI_Bcast(&nhr_blocks, 1, MPI_INT, 0, MYMPI_COMM_WORLD);
        MPI_Bcast(&nlr_blocks, 1, MPI_INT, 0, MYMPI_COMM_WORLD);

        for (int blocks = 0; blocks < (nhr_blocks + nlr_blocks); blocks++)
        {
            int type, npart, n_in_file;
            if (ThisTask == 0)
            {
                SKIP;
                my_fread(&npart, sizeof(int), 1, fd);
                SKIP;

#ifdef SWAP_ENDIAN_READIC_FORMAT4
                swap_Nbyte_format4((char *)&npart, 1, sizeof(int));
#endif // SWAP_ENDIAN_READIC_FORMAT4
                n_in_file = npart;
                VERBOSE_ALL(2, "n_in_file: %i, blocks: %i ", n_in_file, blocks);

                if (blocks < nhr_blocks)
                {
                    type = 1;
                }
                else
                {
                    type = 2;
                }
            }

            MPI_Bcast(&n_in_file, 1, MPI_INT, 0, MYMPI_COMM_WORLD); /* receive type of particle and total number */
            MPI_Bcast(&type, 1, MPI_INT, 0, MYMPI_COMM_WORLD);

            int n_for_this_task = n_in_file / NTask;
            if (ThisTask < (n_in_file % NTask))
            {
                n_for_this_task++;
            }

            for (int subblock = 0; subblock < 3; subblock++)
            {
                if (type == 1 && subblock == 2)
                {
                    continue; /* HR part's have no mass array */
                }

                if (ThisTask == 0)
                {
                    SKIP;
                }

                int pc_here = pc;

                for (int pr = 0; pr < NTask; pr++) /* go through all processes, note: pr is the receiving process */
                {
                    if (ThisTask == 0 || ThisTask == pr)
                    {
                        int n = n_for_this_task; /* number of particles for this process */

                        if (ThisTask == 0 && pr > 0)
                        {
                            MPI_Recv(&n, 1, MPI_INT, pr, TAG_N, MYMPI_COMM_WORLD, &status);
                        }

                        if (ThisTask == pr && pr > 0)
                        {
                            MPI_Send(&n, 1, MPI_INT, 0, TAG_N, MYMPI_COMM_WORLD);
                        }

                        int left = n;
                        while (left > 0)
                        {
                            if (left > BLOCKSIZE) /* restrict transmission size to buffer length */
                            {
                                groupsize = BLOCKSIZE;
                            }
                            else
                            {
                                groupsize = left;
                            }

                            if (ThisTask == 0)
                            {
                                if (subblock < 2)
                                {
                                    VERBOSE_ALL(2, "Reading subblock %i, pr %i, left%i, n %i, blocks %i ", subblock, pr,
                                                left, n, blocks);
                                    my_fread(&dummy[0][0], sizeof(float), 3 * groupsize, fd);
#ifdef SWAP_ENDIAN_READIC_FORMAT4
                                    swap_Nbyte_format4((char *)&dummy[0][0], 3 * groupsize, sizeof(float));
#endif // SWAP_ENDIAN_READIC_FORMAT4
                                }
                                else
                                {
                                    VERBOSE_ALL(2, "Reading subblock %i ", subblock);
                                    my_fread(&dummy[0][0], sizeof(float), groupsize, fd);
#ifdef SWAP_ENDIAN_READIC_FORMAT4
                                    swap_Nbyte_format4((char *)&dummy[0][0], groupsize, sizeof(float));
#endif // SWAP_ENDIAN_READIC_FORMAT4
                                }
                            }

                            if (ThisTask == 0 && pr != 0)
                            {
                                if (subblock < 2)
                                {
                                    MPI_Ssend(&dummy[0][0], 3 * groupsize, MPI_FLOAT, pr, TAG_PDATA, MYMPI_COMM_WORLD);
                                }
                                else
                                {
                                    MPI_Ssend(&dummy[0][0], groupsize, MPI_FLOAT, pr, TAG_PDATA, MYMPI_COMM_WORLD);
                                }
                            }

                            if (ThisTask != 0 && pr != 0)
                            {
                                if (subblock < 2)
                                {
                                    MPI_Recv(&dummy[0][0], 3 * groupsize, MPI_FLOAT, 0, TAG_PDATA, MYMPI_COMM_WORLD,
                                             &status);
                                }
                                else
                                {
                                    MPI_Recv(&dummy[0][0], groupsize, MPI_FLOAT, 0, TAG_PDATA, MYMPI_COMM_WORLD,
                                             &status);
                                }
                            }

                            if (ThisTask == pr)
                            {
                                fp = &dummy[0][0];
                                for (int i = 0; i < groupsize; i++)
                                {
                                    if (subblock == 0)
                                    {
                                        P[pc_here].Type = type;
                                        P[pc_here].ID = id; /* now set ID */
                                        for (int k = 0; k < 3; k++)
                                        {
                                            P[pc_here].Pos[k] = dummy[i][k];
                                            /*VERBOSE_ALL(2,"P[pc_here].Pos[k]  = %f ", P[pc_here].Pos[k]); */
                                        }
                                        pc_here++;
                                        id++;
                                    }

                                    if (subblock == 1)
                                    {
                                        for (int k = 0; k < 3; k++)
                                            P[pc_here].Vel[k] = dummy[i][k];
                                        pc_here++;
                                    }
                                    if (subblock == 2)
                                    {
                                        P[pc_here].Mass = fp[i];
                                        pc_here++;
                                    }
                                }
                            }

                            left -= groupsize;
                        }
                    }

                    MPI_Barrier(MYMPI_COMM_WORLD);
                    MPI_Bcast(&id, 1, MPI_INT, pr, MYMPI_COMM_WORLD);
                }

                if (ThisTask == 0)
                {
                    SKIP;
                }
            }

            pc += n_for_this_task;
            NumPart += n_for_this_task;
        }

        if (ThisTask == 0)
        {
            fclose(fd);
        }
    }

    MPI_Barrier(MYMPI_COMM_WORLD);

    VERBOSE(2, "\nreading done.\n ");

#ifdef RESCALEVINI
    VERBOSE(2, stdout, "\nRescaling v_ini ! \n");
#endif // RESCALEVINI

    /* now convert the units */
    double sqr_a = sqrt(All.Time);

    int counttype2 = 0, counttype3 = 0;

    for (int i = 0; i < NumPart; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            P[i].Pos[j] = P[i].Pos[j] * posfac; /* here in units of kpc/h */
            P[i].Vel[j] = P[i].Vel[j] * velfac; /* comoving velocity xdot on km/sec */
#ifdef RESCALEVINI
            P[i].Vel[j] *= All.VelIniScale; /* scaling v to use same IC's for
                               different cosmologies */
#endif // RESCALEVINI
            P[i].Vel[j] *= sqr_a; /* transform to velocity variable u */
        }

        if (P[i].Type == 1)
        {
            P[i].Mass = All.MassTable[1];
        }
        else
        {
            P[i].Mass *= massfac;

            double r2 = (P[i].Pos[0] - xcm) * (P[i].Pos[0] - xcm) + (P[i].Pos[1] - ycm) * (P[i].Pos[1] - ycm) +
                        (P[i].Pos[2] - zcm) * (P[i].Pos[2] - zcm);

            if (sqrt(r2) > rcut) /* boundary of inner LR zone */
            {
                P[i].Type = 3;
                counttype3++;
            }
            else
            {
                counttype2++;
            }
        }
    }

    VERBOSE_ALL(2, "Task: %d has %d particles, and %d of type 3 ", ThisTask, NumPart, counttype3);
    int counttype_disk, counttype_bulge;
    MPI_Reduce(&counttype2, &counttype_disk, 1, MPI_INT, MPI_SUM, 0, MYMPI_COMM_WORLD);
    MPI_Reduce(&counttype3, &counttype_bulge, 1, MPI_INT, MPI_SUM, 0, MYMPI_COMM_WORLD);
    MPI_Bcast(&counttype_disk, 1, MPI_INT, 0, MYMPI_COMM_WORLD);
    MPI_Bcast(&counttype_bulge, 1, MPI_INT, 0, MYMPI_COMM_WORLD);

    int sumnumpart;
    MPI_Allreduce(&NumPart, &sumnumpart, 1, MPI_INT, MPI_SUM, MYMPI_COMM_WORLD);
    All.TotNumPart = sumnumpart;

    VERBOSE(2, "particles loaded: %d \n ", (int)All.TotNumPart);
    VERBOSE(2, "particles of type 2: %d  ", counttype_disk);
    VERBOSE(2, "particles of type 3: %d  (together %d)  \n ", counttype_bulge, counttype_bulge + counttype_disk);
    VERBOSE(2, "\n Collisionless particles   :  %d ", (int)(All.TotNumPart - All.TotN_gas));
    VERBOSE(2, "Baryonic particles        :  %d ", (int)All.TotN_gas);
    VERBOSE(2, "                             --------- ");
    VERBOSE(2, "Total number of particles :  %d \n", (int)All.TotNumPart);

#undef BLOCKSIZE
}
