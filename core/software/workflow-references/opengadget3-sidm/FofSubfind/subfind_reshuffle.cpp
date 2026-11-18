/**
 * @file
 *
 * This file is part of the N-body/SPH code OpenGadget3.
 *
 * @brief Reshuffling in Subfind.
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
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "../CodeBase/allvars.hpp"
#include "../CodeBase/proto.hpp"

#if defined(SUBFIND) && defined(SUBFIND_RESHUFFLE_CATALOGUE)
#include "subfind.hpp"
#include "fof.hpp"

static int table_read = 0, table_read_voronoi = 0;

static int Nfiles, NfilesVoronoi;

static long long *NumPartPerFile;
static long long *NumPartPerFileVoronoi;

/**
 * @brief Structure to store particle IDs and group numbers.
 *
 * This structure is used to store particle IDs and their corresponding group numbers.
 */
static struct id_list
{
    MyIDType ID;
    MyIDType GrNr;
};

/**
 * @brief Reads the table of HSML values from files.
 *
 * This function reads the table of HSML (smoothing lengths) values from files for further processing.
 */
void read_hsml_table(void)
{
    int i, dummy, nhsml;
    long long ntot;
    char fname[1000];
    FILE *fd;

    Nfiles = 1;
    for (int i = 0; i < Nfiles; i++)
    {
        sprintf(fname, "%s/hsmldir_%03d/%s_%03d.%d", All.OutputDir, RestartSnapNum, "hsml", RestartSnapNum, i);
        if (!(fd = fopen(fname, "r")))
        {
            ENDRUN("can't open file `%s`", fname);
        }

        my_fread(&nhsml, sizeof(int), 1, fd);
        my_fread(&dummy, sizeof(int), 1, fd);
        my_fread(&ntot, sizeof(long long), 1, fd);
        my_fread(&Nfiles, sizeof(int), 1, fd);

        if (i == 0)
        {
            NumPartPerFile = mymalloc("NumPartPerFile", (Nfiles + 1) * sizeof(long long));
        }

        NumPartPerFile[i] = nhsml;
    }

    long long sum = 0;
    for (int i = 0; i < Nfiles; i++)
    {
        long long n = NumPartPerFile[i];

        NumPartPerFile[i] = sum;

        sum += n;
    }

    NumPartPerFile[Nfiles] = sum;

    table_read = 1;
}

/**
 * @brief Determines the appropriate HSML file and the number of particles to read from it.
 *
 * This function calculates the file number and the number of particles to read from the HSML files based on the skip
 * count and the total number of particles.
 *
 * @param nskip Number of particles to skip.
 * @param count Number of values to read.
 * @param filenr Pointer to store the file number to read from.
 * @param n_to_read Pointer to store the number of particles to read from the file.
 * @param n_to_skip Pointer to store the number of particles to skip within the file.
 */
void get_hsml_file(long long nskip, int count, int *filenr, int *n_to_read, int *n_to_skip)
{
    int i;
    for (int i = 0; i < Nfiles; i++)
    {
        if (nskip >= NumPartPerFile[i] && nskip < NumPartPerFile[i + 1])
        {
            break;
        }
    }

    ENDRUN_IF(i >= Nfiles, "Index larger then number of files.");

    *filenr = i;
    *n_to_skip = (int)(nskip - NumPartPerFile[i]);

    int nrest = (NumPartPerFile[i + 1] - NumPartPerFile[i] - *n_to_skip);

    if (count <= nrest)
    {
        *n_to_read = count;
    }
    else
    {
        *n_to_read = nrest;
    }
}

/**
 * @brief Reads the preexisting Subhalo indices from files.
 *
 * This function reads preexisting Subhalo indices from files and assigns them to particles.
 *
 * @note This function assumes that the number of files in the Subhalo catalogues matches the number of MPI tasks.
 */
void read_subfind_ids(void)
{
    FILE *fd;
    double t0, t1;
    char fname[500];
    int i, j;
    MyIDType *ids;
    int *list_of_nids;
    int ngrp;
    int Nids, nprocgroup, masterTask, groupTask;
    unsigned long long nid_previous;
    int fof_compare_P_SubNr(const void *a, const void *b);
    long long TotNids;
    long long *NumIdsPerFile;
    static struct id_list *ID_list;

    VERBOSE(1, "\nTrying to read preexisting Subhalo indices...  (presently allocated=%g MB)",
            AllocatedBytes / (1024.0 * 1024.0));

    if (ThisTask == 0)
    {
        Nfiles = 1;
        for (int i = 0; i < Nfiles; i++)
        {
            sprintf(fname, "%s/groups_%03d/%s_%03d.%d", All.OutputDir, RestartSnapNum, "subhalo_ids", RestartSnapNum,
                    i);
            if (!(fd = fopen(fname, "r")))
            {
                ENDRUN("can't read file `%s`", fname);
            }

            my_fread(&Ngroups, sizeof(int), 1, fd);
            my_fread(&TotNgroups, sizeof(int), 1, fd);
            my_fread(&Nids, sizeof(int), 1, fd);
            my_fread(&TotNids, sizeof(long long), 1, fd);
            my_fread(&Nfiles, sizeof(int), 1, fd);
            my_fread(&nid_previous, sizeof(unsigned int), 1, fd); /* this is the number of IDs in previous files */
            fclose(fd);

            if (i == 0)
            {
                MPI_Bcast(&Nfiles, 1, MPI_INT, 0, MYMPI_COMM_WORLD);
                NumIdsPerFile = mymalloc("NumIdsPerFile", (Nfiles) * sizeof(long long));
            }

            NumIdsPerFile[i] = Nids;
        }
    }
    else
    {
        MPI_Bcast(&Nfiles, 1, MPI_INT, 0, MYMPI_COMM_WORLD);
        NumIdsPerFile = mymalloc("NumIdsPerFile", (Nfiles) * sizeof(long long));
    }

    MPI_Bcast(NumIdsPerFile, Nfiles * sizeof(long long), MPI_BYTE, 0, MYMPI_COMM_WORLD);

    /* start reading of group catalogue */

    if (ThisTask == 0)
    {
        sprintf(fname, "%s/groups_%03d/%s_%03d.%d", All.OutputDir, RestartSnapNum, "subhalo_ids", RestartSnapNum, 0);
        if (!(fd = fopen(fname, "r")))
        {
            ENDRUN("can't read file `%s`", fname);
        }

        my_fread(&Ngroups, sizeof(int), 1, fd);
        my_fread(&TotNgroups, sizeof(int), 1, fd);
        my_fread(&Nids, sizeof(int), 1, fd);
        my_fread(&TotNids, sizeof(long long), 1, fd);
        my_fread(&Nfiles, sizeof(int), 1, fd);
        fclose(fd);
    }

    MPI_Bcast(&Nfiles, 1, MPI_INT, 0, MYMPI_COMM_WORLD);
    MPI_Bcast(&TotNgroups, 1, MPI_INT, 0, MYMPI_COMM_WORLD);
    MPI_Bcast(&TotNids, sizeof(long long), MPI_BYTE, 0, MYMPI_COMM_WORLD);

    t0 = second();

    if (NTask != Nfiles)
    {
        VERBOSE(1, "number of files (%d) in subhalo catalogues does not match MPI-Tasks, I'm working around this.",
                Nfiles);

        ID_list = mymalloc("ID_list", (TotNids / NTask + NTask) * sizeof(struct id_list));

        int filenr, target, ngroups, nids, nsend, stored;

        int *nids_to_get = mymalloc("nids_to_get", NTask * sizeof(NTask));
        int *nids_obtained = mymalloc("nids_obtained", NTask * sizeof(NTask));

        for (int i = 0; i < NTask; i++)
        {
            nids_obtained[i] = 0;
        }

        for (int i = 0; i < NTask - 1; i++)
        {
            nids_to_get[i] = (int)(TotNids / NTask);
        }
        nids_to_get[NTask - 1] = (int)(TotNids - (NTask - 1) * (TotNids / NTask));

        Nids = nids_to_get[ThisTask];

        if (ThisTask == 0)
        {
            /**** now ids ****/
            for (int filenr = 0; filenr < Nfiles; filenr++)
            {
                sprintf(fname, "%s/groups_%03d/%s_%03d.%d", All.OutputDir, RestartSnapNum, "subhalo_ids",
                        RestartSnapNum, filenr);
                if (!(fd = fopen(fname, "r")))
                {
                    ENDRUN("can't read file `%s`", fname);
                }

                VERBOSE(2, "reading '%s'", fname);

                my_fread(&ngroups, sizeof(int), 1, fd);
                my_fread(&TotNgroups, sizeof(int), 1, fd);
                my_fread(&nids, sizeof(int), 1, fd);
                my_fread(&TotNids, sizeof(long long), 1, fd);
                my_fread(&Nfiles, sizeof(int), 1, fd);
                my_fread(&nid_previous, sizeof(int), 1, fd); /* this is the number of IDs in previous files */

                /* let's now fix nid_previous in case there has been a 32-bit overflow... */
                nid_previous = 0;
                for (int i = 0; i < filenr; i++)
                {
                    nid_previous += NumIdsPerFile[i];
                }

                struct id_list *tmpID_list = mymalloc("tmpID_list", nids * sizeof(struct id_list));

                ids = mymalloc("ids", nids * sizeof(MyIDType));

                my_fread(ids, sizeof(MyIDType), nids, fd);

                for (int i = 0; i < nids; i++)
                {
                    tmpID_list[i].ID = ids[i];
                    tmpID_list[i].GrNr = nid_previous + i;
                }

                myfree(ids);

                fclose(fd);

                target = 0;
                stored = 0;
                while (nids > 0)
                {
                    while (nids_to_get[target] == 0)
                    {
                        target++;
                    }

                    if (nids > nids_to_get[target])
                    {
                        nsend = nids_to_get[target];
                    }
                    else
                    {
                        nsend = nids;
                    }

                    if (target == 0)
                    {
                        memcpy(&ID_list[nids_obtained[target]], &tmpID_list[stored], nsend * sizeof(struct id_list));
                    }
                    else
                    {
                        MPI_Send(&nsend, 1, MPI_INT, target, TAG_HEADER, MYMPI_COMM_WORLD);
                        MPI_Send(&tmpID_list[stored], nsend * sizeof(struct id_list), MPI_BYTE, target, TAG_SPHDATA,
                                 MYMPI_COMM_WORLD);
                    }

                    nids_to_get[target] -= nsend;
                    nids_obtained[target] += nsend;
                    nids -= nsend;
                    stored += nsend;
                }

                myfree(tmpID_list);
            }
        }
        else
        {
            while (nids_to_get[ThisTask])
            {
                MPI_Recv(&nsend, 1, MPI_INT, 0, TAG_HEADER, MYMPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(&ID_list[nids_obtained[ThisTask]], nsend * sizeof(struct id_list), MPI_BYTE, 0, TAG_SPHDATA,
                         MYMPI_COMM_WORLD, MPI_STATUS_IGNORE);

                nids_to_get[ThisTask] -= nsend;
                nids_obtained[ThisTask] += nsend;
            }
        }

        myfree(nids_obtained);
        myfree(nids_to_get);
    }
    else
    {
        /* read routine can constinue in parallel */

        nprocgroup = NTask / All.NumFilesWrittenInParallel;
        if ((NTask % All.NumFilesWrittenInParallel))
        {
            nprocgroup++;
        }
        masterTask = (ThisTask / nprocgroup) * nprocgroup;
        for (int groupTask = 0; groupTask < nprocgroup; groupTask++)
        {
            if (ThisTask == (masterTask + groupTask)) /* ok, it's this processor's turn */
            {
                sprintf(fname, "%s/groups_%03d/%s_%03d.%d", All.OutputDir, RestartSnapNum, "subhalo_ids",
                        RestartSnapNum, ThisTask);
                if (!(fd = fopen(fname, "r")))
                {
                    ENDRUN("can't read file `%s`", fname);
                }

                VERBOSE(2, "reading '%s'", fname);

                my_fread(&Ngroups, sizeof(int), 1, fd);
                my_fread(&TotNgroups, sizeof(int), 1, fd);
                my_fread(&Nids, sizeof(int), 1, fd);
                my_fread(&TotNids, sizeof(long long), 1, fd);
                my_fread(&Nfiles, sizeof(int), 1, fd);
                my_fread(&nid_previous, sizeof(unsigned int), 1, fd); /* this is the number of IDs in previous files */

                /* let's now fix nid_previous in case there has been a 32-bit overflow... */
                nid_previous = 0;
                for (int i = 0; i < ThisTask; i++)
                {
                    nid_previous += NumIdsPerFile[i];
                }

                ID_list = mymalloc("ID_list", Nids * sizeof(struct id_list));
                ids = mymalloc("ids", Nids * sizeof(MyIDType));

                my_fread(ids, sizeof(MyIDType), Nids, fd);

                for (int i = 0; i < Nids; i++)
                {
                    ID_list[i].ID = ids[i];
                    ID_list[i].GrNr = nid_previous + i;
                }

                myfree(ids);

                fclose(fd);
            }

            MPI_Barrier(MYMPI_COMM_WORLD); /* wait inside the group */
        }
    }

    t1 = second();
    VERBOSE(2, "reading  took %g sec", timediff(t0, t1));
    t0 = second();

    qsort(P, NumPart, sizeof(struct particle_data), io_compare_P_ID);
    qsort(ID_list, Nids, sizeof(struct id_list), subfind_reshuffle_compare_ID_list_ID);

    for (int i = 0; i < NumPart; i++)
    {
        P[i].GrNr = TotNids + 1; /* will mark particles that are not in any group */
    }

    t1 = second();
    VERBOSE(2, "sorting took %g sec", timediff(t0, t1));

    list_of_nids = mymalloc("list_of_nids", NTask * sizeof(int));
    MPI_Allgather(&Nids, 1, MPI_INT, list_of_nids, 1, MPI_INT, MYMPI_COMM_WORLD);

    static struct id_list *recv_ID_list;

    t0 = second();

    int matches = 0;

    /* exchange  data */
    for (int ngrp = 0; ngrp < (1 << PTask); ngrp++)
    {
        int sendTask = ThisTask;
        int recvTask = ThisTask ^ ngrp;

        if (recvTask < NTask)
        {
            if (list_of_nids[sendTask] > 0 || list_of_nids[recvTask] > 0)
            {
                if (ngrp == 0)
                {
                    recv_ID_list = ID_list;
                }
                else
                {
                    recv_ID_list = mymalloc("recv_ID_list", list_of_nids[recvTask] * sizeof(struct id_list));

                    /* get the particles */
                    MPI_Sendrecv(ID_list, Nids * sizeof(struct id_list), MPI_BYTE, recvTask, TAG_DENS_A, recv_ID_list,
                                 list_of_nids[recvTask] * sizeof(struct id_list), MPI_BYTE, recvTask, TAG_DENS_A,
                                 MYMPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }
                int j = 0;
                for (int i = 0; i < list_of_nids[recvTask]; i++)
                {
                    while (j < NumPart - 1 && P[j].ID < recv_ID_list[i].ID)
                    {
                        j++;
                    }

                    if (recv_ID_list[i].ID == P[j].ID)
                    {
                        P[j].GrNr = recv_ID_list[i].GrNr;
                        matches++;
                    }
                }

                if (ngrp != 0)
                {
                    myfree(recv_ID_list);
                }
            }
        }
    }

    long long totlen;

    sumup_large_ints(1, &matches, &totlen);
    ENDRUN_IF(totlen != TotNids, "total number of IDs doesn't match");

    t1 = second();
    VERBOSE(2, "assigning GrNr to P[] took %g sec\n", timediff(t0, t1));

    MPI_Barrier(MYMPI_COMM_WORLD);

    myfree(list_of_nids);

    myfree(ID_list);
    myfree(NumIdsPerFile);
}

/**
 * @brief Compares two elements based on their ID values for reshuffling.
 *
 * This function is used as a comparison function for sorting elements of type `struct id_list`
 * based on their ID values. It returns an integer value indicating the relative ordering
 * of the elements.
 *
 * @param a Pointer to the first element to compare.
 * @param b Pointer to the second element to compare.
 * @return -1 if ID of 'a' is less than ID of 'b', 1 if ID of 'a' is greater than ID of 'b',
 *         and 0 if the IDs are equal.
 */
int subfind_reshuffle_compare_ID_list_ID(const void *a, const void *b)
{
    if (((struct id_list *)a)->ID < ((struct id_list *)b)->ID)
    {
        return -1;
    }

    if (((struct id_list *)a)->ID > ((struct id_list *)b)->ID)
    {
        return +1;
    }

    return 0;
}

/**
 * @brief Reads the HSML table in Voronoi format.
 *
 * This function reads the HSML table in Voronoi format from the appropriate files
 * and initializes necessary variables for further processing.
 */
void read_hsml_table_voronoi(void)
{
    int dummy, nhsml;
    long long ntot;
    char fname[1000];
    FILE *fd;

    NfilesVoronoi = 1;
    for (int i = 0; i < NfilesVoronoi; i++)
    {
        sprintf(fname, "%s/hsmldir_%03d/%s_%03d.%d", All.OutputDir, RestartSnapNum, "voronoi_hsml", RestartSnapNum, i);
        if (!(fd = fopen(fname, "r")))
        {
            ENDRUN("can't open file `%s`", fname);
        }

        my_fread(&nhsml, sizeof(int), 1, fd);
        my_fread(&dummy, sizeof(int), 1, fd);
        my_fread(&ntot, sizeof(long long), 1, fd);
        my_fread(&NfilesVoronoi, sizeof(int), 1, fd);

        if (i == 0)
        {
            NumPartPerFileVoronoi = mymalloc("	NumPartPerFileVoronoi", (NfilesVoronoi + 1) * sizeof(long long));
        }

        NumPartPerFileVoronoi[i] = nhsml;
    }

    long long sum = 0;
    for (int i = 0; i < NfilesVoronoi; i++)
    {
        long long n = NumPartPerFileVoronoi[i];

        NumPartPerFileVoronoi[i] = sum;

        sum += n;
    }

    NumPartPerFileVoronoi[NfilesVoronoi] = sum;

    table_read_voronoi = 1;
}

/**
 * @brief Determines the file and positions to read HSML values from in Voronoi format.
 *
 * This function calculates the file number and positions to read HSML values from
 * based on the input parameters.
 *
 * @param nskip Number of values to skip.
 * @param count Number of values to read.
 * @param filenr Pointer to the variable to store the file number.
 * @param n_to_read Pointer to the variable to store the number of values to read.
 * @param n_to_skip Pointer to the variable to store the number of values to skip within the file.
 */
void get_hsml_file_voronoi(long long nskip, int count, int *filenr, int *n_to_read, int *n_to_skip)
{
    int i = 0;
    for (i = 0; i < NfilesVoronoi; i++)
    {
        if (nskip >= NumPartPerFileVoronoi[i] && nskip < NumPartPerFileVoronoi[i + 1])
        {
            break;
        }
    }

    ENDRUN_IF(i >= NfilesVoronoi, "Index larger then number of files for Voronoi.");

    *filenr = i;
    *n_to_skip = (int)(nskip - NumPartPerFileVoronoi[i]);

    int nrest = (NumPartPerFileVoronoi[i + 1] - NumPartPerFileVoronoi[i] - *n_to_skip);

    if (count <= nrest)
    {
        *n_to_read = count;
    }
    else
    {
        *n_to_read = nrest;
    }
}

/**
 * @brief Reads HSML files in Voronoi format.
 *
 * This function reads HSML files in Voronoi format and stores the values in the provided array.
 *
 * @param Values Pointer to the array where the read values will be stored.
 * @param count Number of values to read.
 * @param blocknr Enum indicating the block number.
 * @param nskip Number of values to skip.
 */
void read_hsml_files_voronoi(float *Values, int count, enum iofields blocknr, long long nskip)
{
    char fname[1000];
    FILE *fd;
    float *tmp;
    int n_to_read, n_to_skip, filenr, dummy, ntask, nhsml;
    long long ntot;

    while (count > 0)
    {
        if (table_read_voronoi == 0)
        {
            read_hsml_table_voronoi();
        }

        get_hsml_file_voronoi(nskip, count, &filenr, &n_to_read, &n_to_skip);

        sprintf(fname, "%s/hsmldir_%03d/%s_%03d.%d", All.OutputDir, RestartSnapNum, "voronoi_hsml", RestartSnapNum,
                filenr);
        if (!(fd = fopen(fname, "r")))
        {
            ENDRUN_IF("can't open file `%s`", fname);
        }

        my_fread(&nhsml, sizeof(int), 1, fd);
        my_fread(&dummy, sizeof(int), 1, fd);
        my_fread(&ntot, sizeof(long long), 1, fd);
        my_fread(&ntask, sizeof(int), 1, fd);

        if (blocknr == IO_DMDENSITY_V)
        {
            fseeko(fd, nhsml * sizeof(float), SEEK_CUR); /* skip hsml */
        }

        fseeko(fd, n_to_skip * sizeof(float), SEEK_CUR);

        tmp = mymalloc("tmp", n_to_read * sizeof(float));
        my_fread(tmp, sizeof(float), n_to_read, fd);

        for (int i = 0; i < n_to_read; i++)
        {
            Values[i] = tmp[i];
        }

        myfree(tmp);

        fclose(fd);

        Values += n_to_read;
        count -= n_to_read;
        nskip += n_to_read;
    }
}
#endif // defined(SUBFIND) && defined(SUBFIND_RESHUFFLE_CATALOGUE)
