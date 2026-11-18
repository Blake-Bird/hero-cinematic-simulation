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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <ctype.h>
#include <math.h>
#include <mpi.h>

#include "../../CodeBase/allvars.hpp"
#include "../../CodeBase/proto.hpp"

#include "lt_error_codes.hpp"
#include "gl_cr_dust.hpp"

#ifdef LT_STELLAREVOLUTION

/* | ------------------------------------------------------------------------------------------------------ |
 * |
 * |  COOLING
 * | ------------------------------------------------------------------------------------------------------ |
 */

#ifdef LT_METAL_COOLING

int find_cooling_files(char *basename, int *nlines)
{
    char *comment_string = "!#;\0";

    FILE *file;

    int i, nline;

    char buff[500], line[1000], *c;

    nline = 0;
    for (i = 0;; i++)
    {
        sprintf(buff, "%s.%d", basename, i);
        if ((file = fopen(buff, "r")) != 0x0)
        {
            if (i == 0)
            {
                while ((c = fgets(line, 999, file)) != 0x0)
                {
                    if (strchr(comment_string, line[0]) != 0x0)
                        continue;
                    nline++;
                }
            }
            fclose(file);
        }
        else
            break;
    }

    *nlines = nline;
    return i;
}

void reading_thresholds_for_thermal_instability()
{
    FILE *file;

    char buff[500];

    int i;

    printf("reading temperatures for thermal instability thresholds..\n");

    sprintf(buff, "thermalinstability_onset.dat");
    if ((file = fopen(buff, "r")) != 0x0)
    {
        for (i = 0; i < ZBins; i++)
            fscanf(file, "%lf", &ThInst_onset[i]);
    }
    else
    {
        printf("file for thermal inistability thresholds not found: using 10^5 K for all the %d metallicities", ZBins);
        for (i = 0; i < ZBins; i++)
            ThInst_onset[i] = 5;
    }

    /*   for(i = 0; i < ZBins; i++) */
    /*     ThInst_onset[i] = pow(10, ThInst_onset[i]); */

    return;
}

void read_cooling_tables(char *basename)
{
    char comment_string[5];

    FILE *file;

    char buff[500], line[1000];

    char *c;

    int i, j;

    double Z, T, L;

    double *swap_point;

    double InfoGroup[6];

    if (ThisTask == 0)
    {
        sprintf(comment_string, "!#;");
        ENDRUN_IF((ZBins = find_cooling_files(basename, &TBins)) == 0,
                  "Error: no cooling tables found! better to stop here.");
    }

    VERBOSE(0, "Reading cooling tables ...");

    MPI_Bcast(&ZBins, 1, MPI_INT, 0, MYMPI_COMM_WORLD);
    MPI_Bcast(&TBins, 1, MPI_INT, 0, MYMPI_COMM_WORLD);

    CoolTvalue = (double *)mymalloc("CoolTvalue", (TBins + ZBins + TBins * ZBins) * sizeof(double));
    memset(CoolTvalue, 0, (TBins + ZBins + TBins * ZBins) * sizeof(double));

    CoolingTables = (double **)mymalloc("CoolingTables", ZBins * sizeof(double *));
    memset(CoolingTables, 0, ZBins * sizeof(double *));

    CoolZvalue = CoolTvalue + TBins;
    CoolingTables[0] = CoolZvalue + ZBins;
    for (i = 1; i < ZBins; i++)
        CoolingTables[i] = CoolingTables[i - 1] + TBins;

    VERBOSE(0, "Reading %d cooling tables, each with %d temperature data\n", ZBins, TBins);

    TMin = 1e9;
    TMax = 0;
    ZMin = 10;
    ZMax = 0;

    if (ThisTask == 0)
    {
        for (i = 0; i < ZBins; i++)
        {
            sprintf(buff, "%s.%d", basename, i);
            file = fopen(buff, "r");

            j = 0;
            Z = 1e9;
            while ((c = fgets(line, 999, file)) != 0x0)
            {
                if (strchr(comment_string, line[0]) != 0x0)
                {
                    if ((c = strcasestr(line, "name of tables ")) != 0x0)
                        if (i == 0)
                            printf("%s\n", &line[(int)(c - line)]);
                    if ((c = strcasestr(line, "Z = ")) != 0x0)
                    {
                        sscanf(&line[(int)(c - line)], "%*s = %lf %*s\n", &Z);
                        printf("reading data for Z = %8.6e\n", Z);
                        if (ZMin > Z)
                            ZMin = Z;
                        if (ZMax < Z)
                            ZMax = Z;
                        CoolZvalue[i] = Z;
                    }
                    continue;
                }

                ENDRUN_IF(sscanf(line, "%lf %*f %*f %*f %*f %lf %*s", &T, &L) != 2,
                          "Error in reading line %d of table %d\n", j, i);

                if (i == 0)
                {
                    CoolTvalue[j] = T;
                    if (TMin > T)
                    {
                        TMin = T;
                    }
                    if (TMax < T)
                    {
                        TMax = T;
                    }
                }
                CoolingTables[i][j] = L;
                j++;
                if (feof(file))
                {
                    break;
                }
            }

            ENRUN_IF(Z == 1e9, "This file %s has not a defined metallicity. interrupt.\n", buff);
            ENDRUN_IF(j < TBins, "this file %s has not %d data lines. interrupt.\n", buff, TBins);

            fclose(file);
        }

        for (i = 0; i < ZBins; i++)
            for (j = i + 1; j < ZBins; j++)
                if (CoolZvalue[j] < CoolZvalue[j - 1])
                {
                    Z = CoolZvalue[j - 1];
                    CoolZvalue[j - 1] = CoolZvalue[j];
                    CoolZvalue[j] = Z;

                    swap_point = CoolingTables[j - 1];
                    CoolingTables[j - 1] = CoolingTables[j];
                    CoolingTables[j] = swap_point;
                }

        *(int *)&InfoGroup[0] = TBins;
        *(int *)&InfoGroup[1] = ZBins;
        InfoGroup[2] = ZMax;
        InfoGroup[3] = ZMin;
        InfoGroup[4] = TMax;
        InfoGroup[5] = TMin;
    }

    MPI_Bcast(InfoGroup, 6 * sizeof(double), MPI_BYTE, 0, MYMPI_COMM_WORLD);

    if (ThisTask != 0)
    {
        TBins = *(int *)&InfoGroup[0];
        ZBins = *(int *)&InfoGroup[1];
        ZMax = InfoGroup[2];
        ZMin = InfoGroup[3];
        TMax = InfoGroup[4];
        TMin = InfoGroup[5];

        CoolTvalue = (double *)mymalloc("CoolTvalue", (TBins + ZBins + TBins * ZBins) * sizeof(double));
        memset(CoolTvalue, 0, (TBins + ZBins + TBins * ZBins) * sizeof(double));
    }

    MPI_Bcast(CoolTvalue, (TBins + ZBins + TBins * ZBins) * sizeof(double), MPI_BYTE, 0, MYMPI_COMM_WORLD);

    ThInst_onset = (double *)mymalloc("ThInst_onset", ZBins * sizeof(double));

    if (ThisTask > 0)
    {
        CoolZvalue = CoolTvalue + TBins;
        CoolingTables[0] = CoolZvalue + ZBins;
        for (i = 1; i < ZBins; i++)
        {
            CoolingTables[i] = CoolingTables[i - 1] + TBins;
        }

        for (i = 1; i < ZBins; i++)
        {
            CoolingTables[i] = CoolingTables[i - 1] + TBins;
        }
    }

    /* note : for S&D 1993 */
    /*   if(ThisTask == 0) */
    /*     reading_thresholds_for_thermal_instability(); */
    /*   MPI_Bcast(&ThInst_onset[0], ZBins * sizeof(double), MPI_BYTE, 0, MYMPI_COMM_WORLD); */
    /*   ThInst_onset[0] = 5; */
    /*   ThInst_onset[1] = 5; */
    /*   ThInst_onset[2] = 5; */
    /*   ThInst_onset[3] = 5.3; */
    /*   ThInst_onset[4] = 5.3; */
    /*   ThInst_onset[5] = 5.3; */
    /*   ThInst_onset[6] = 5.3; */
    /*   ThInst_onset[7] = 5.3; */

    if (ThisTask == 11)
    {
        for (i = 0; i < ZBins; i++)
        {

            sprintf(buff, "coolingtable.%d.%d", ThisTask, i);
            file = fopen(buff, "w");

            for (j = 0; j < TBins; j++)
                fprintf(file, "%10.8e %10.8e\n", CoolTvalue[j], CoolingTables[i][j]);

            fclose(file);
        }
    }

    MPI_Barrier(MYMPI_COMM_WORLD);
    if (ThisTask == 0)
        printf("\n\n");
    return;
}

#else

void read_cooling_tables(char *basename)
{
    int i;

    ZBins = 1;
    TBins = 1;
    ZMin = ZMax = -4.0;
    TMin = 4.0;
    TMax = 8.0;

    if (ThisTask == 0)
        printf("Setting dummy cooling tables to zero ...\n");

    CoolTvalue = (double *)mymalloc("CoolTvalue", (TBins + ZBins + TBins * ZBins) * sizeof(double));
    memset(CoolTvalue, 0, (TBins + ZBins + TBins * ZBins) * sizeof(double));

    CoolingTables = (double **)mymalloc("CoolingTables", ZBins * sizeof(double *));
    memset(CoolingTables, 0, ZBins * sizeof(double *));

    CoolZvalue = CoolTvalue + TBins;
    CoolingTables[0] = CoolZvalue + ZBins;
    for (i = 1; i < ZBins; i++)
        CoolingTables[i] = CoolingTables[i - 1] + TBins;
}

#endif

#if defined(GL_CR_DUST)

void read_dust()
{
    /*
    Reads in file dust.dat the condensatiion efficiencies for elements entering into grains,
    and set indiciees of those elements in the Metnames array:
    Possible dust.dat follow, with its comments

    # lines starting with # are ignored
    # condensation efficiencies in dust for the relevant elements due to the 3 channels.
    # last columns is the number of atoms of that element entering in grain composition
    # element AGB SNIa SNII
    # element order is irrelevant
    # NOTE that in GIVENGRAIN case, only Si values are relevant and  used for Mg and Fe as well
    C    0.300000000000000  0.030000000000000 0.3000000000000000000 1
    O    0.300000000000000  0.030000000000000 0.3000000000000000000 4
    Mg   0.300000000000000  0.030000000000000 0.3000000000000000000 1
    Si   0.300000000000000  0.030000000000000 0.3000000000000000000 1
    Fe   0.300000000000000  0.030000000000000 0.3000000000000000000 1
    */

    char ele[LT_NMet][3];
    double cAGB[LT_NMet], cIa[LT_NMet], cII[LT_NMet];
    int nato[LT_NMet];

    MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);

    if (ThisTask == 0)
    {

        FILE *file = fopen("dust.dat", "r");
        ENDRUN_IF(!file, "I can't open dust data input file dust.dat");

        char line[256]; // Buffer per leggere una riga dal file

        int irig = 0;

        while (irig < LT_NMet && fgets(line, sizeof(line), file))
        {
            // Controlla se la riga è un commento (inizia con #)
            if (line[0] == '#')
            {
                continue; // Salta la riga
            }

            if (sscanf(line, "%s %lf %lf %lf %d", ele[irig], &cAGB[irig], &cIa[irig], &cII[irig], &nato[irig]) == 5)
            {
                irig++; // Incrementa solo se la riga è valida
            }
        }
        fclose(file);

        for (int i = 0; i < irig; i++)
        {
            printf("%s %.2f %.2f %.2f %d\n", ele[i], cAGB[i], cIa[i], cII[i], nato[i]);
            fflush(stdout);
        }

        // Array per memorizzare gli indici
        int indices[LT_NMet];

        // Trova gli indici
        for (int i = 0; i < irig; i++)
        {
            indices[i] = -1; // Inizializza con -1 (non trovato)
            for (int j = 0; j < LT_NMet; j++)
            {
                if (strcmp(ele[i], MetNames[j]) == 0)
                {
                    if (strcmp("C", ele[i]) == 0)
                    { // ind_C is set
                        ind_C = j;
                    }
                    if (strcmp("O", ele[i]) == 0)
                    { // ind_O is set
                        ind_O = j;
                    }
                    if (strcmp("Mg", ele[i]) == 0)
                    { // ind_sil[0] is set
                        ind_sil[0] = j;
                    }
                    if (strcmp("Si", ele[i]) == 0)
                    { // ind_sil[1] is set and condensation efficiencies for GIVENGRAIN case
                        ind_sil[1] = j;
                        effsil_AGB = cAGB[i];
                        effsil_Ia = cIa[i];
                        effsil_II = cII[i];
                    }
                    if (strcmp("Fe", ele[i]) == 0)
                    { // ind_sil[2] is set
                        ind_sil[2] = j;
                    }

                    indices[i] = j; // Salva l'indice di MetNames dove c'è corrispondenza
                    break; // Esci dal loop interno una volta trovato
                }
            }
        }

        ENDRUN_IF(ind_sil[1] < 0, "you don't trace Mg, or its condensation efficiencies not present in dust.dat");
        ENDRUN_IF(ind_sil[1] < 0, "you don't trace Si, or its condensation efficiencies not present in dust.dat");
        ENDRUN_IF(ind_sil[1] < 0, "you don't trace Fe, or its condensation efficiencies not present in dust.dat");
        ENDRUN_IF(ind_C < 0, "you don't trace C, or its condensation efficiencies not present in dust.dat");
        ENDRUN_IF(ind_O < 0, "you don't trace O, or its condensation efficiencies not present in dust.dat");

        // DEBUG
        /*
                printf("Indici trovati:\n");
                for (int i = 0; i < irig; i++) {
                    printf("ele[%d] (%s) si trova in MetNames[%d]\n", i, ele[i], indices[i]);
                }
                fflush(stdout);
        */
        for (int i = 0; i < LT_NMet; i++)
        {
            dcondeffAGB[i] = 0.0;
            dcondeffIa[i] = 0.0;
            dcondeffII[i] = 0.0;
        }

        for (int i = 0; i < irig; i++)
        {
            dcondeffAGB[indices[i]] = cAGB[i];
            dcondeffIa[indices[i]] = cIa[i];
            dcondeffII[indices[i]] = cII[i];
        }
    }

    MPI_Bcast(dcondeffAGB, LT_NMet, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(dcondeffIa, LT_NMet, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(dcondeffII, LT_NMet, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Bcast(ind_sil, n_sil, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&ind_C, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&ind_O, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // debug
    /*
            for (int i = 0; i < LT_NMet; i++) {
                printf("hola task %d AGB %f metal %s\n", ThisTask, dcondeffAGB[i], MetNames[i]);
                printf("hola task %d Ia  %f metal %s\n", ThisTask, dcondeffIa[i], MetNames[i]);
                printf("hola task %d II  %f metal %s\n", ThisTask, dcondeffII[i], MetNames[i]);
            }

            printf("task %d got ind_sil: ", ThisTask);
            for (int i = 0; i < n_sil; i++) {
                printf("%d ", ind_sil[i]);
            }
            printf("\n");
            printf("task %d got ind_C %d\n", ThisTask, ind_C);
            printf("task %d got ind_O %d\n", ThisTask, ind_O);
            fflush(stdout);
    */

    // exit(0);
}

#endif // (GL_CR_DUST)

/* | ------------------------------------------------------------------------------------------------------ |
 * |
 * |  METALS
 * | ------------------------------------------------------------------------------------------------------ |
 */

void read_metals(void)
{
#define NAME_SIZE 5

    char s[200], name[200];
    char *names;
    int j;

    FILE *file;

#ifdef UM_METAL_COOLING
    char *PT_Symbols;

    float *PT_Masses;

    int NPT;
#endif

    Hel = -1;
    Iron = -1;
    Oxygen = -1;
    FillEl = -1;

    /* :: by UM :: */
#ifdef UM_CHEMISTRY // UM_METAL_COOLING
    Carbon = -1;
    Magnesium = -1;
    Silicon = -1;
    Nitrogen = -1;
    Neon = -1;
    Sulphur = -1;
    Calcium = -1;
    Sodium = -1;
    Aluminium = -1;
    Argon = -1;
    Nickel = -1;
#endif

    sprintf(s, "MetNames");
    names = (char *)mymalloc(s, LT_NMet * NAME_SIZE);

    for (j = 0; j < LT_NMet; j++)
        MetNames[j] = &names[j * NAME_SIZE];

    if (ThisTask == 0)
    {
        ENDRUN_IF((file = fopen("metals.dat", "r")) == NULL,
                  "I can't open metals data input file:"
                  "%s\n",
                  "metals.dat");

        for (j = 0; j < LT_NMet; j++)
        {

            ENDRUN_IF(feof(file), "something wrong with <metals.dat>\n");

            fgets(s, 200, file);
            if ((sscanf(s, "%[a-zA-Z]s %lg", &name[0], &MetSolarValues[j])) == 1)
                printf("it seems that in <metals.dat> no solar abundance is present for element %s\n", name);

            // sprintf(s, "MetNames_%02d", j);
            // MetNames[j] = (char *) mymalloc(s, strlen(name) + 2);
            strcpy(MetNames[j], name);
            if (strcmp(name, "He") == 0)
            {
                Hel = j;
            }
#ifdef UM_CHEMISTRY // UM_METAL_COOLING
            else if (strcmp(name, "C") == 0)
                Carbon = j;
            else if (strcmp(name, "N") == 0)
                Nitrogen = j;
            else if (strcmp(name, "Mg") == 0)
                Magnesium = j;
            else if (strcmp(name, "Si") == 0)
                Silicon = j;
            else if (strcmp(name, "Ne") == 0)
                Neon = j;
            else if (strcmp(name, "S") == 0)
                Sulphur = j;
            else if (strcmp(name, "Ca") == 0)
                Calcium = j;
            else if (strcmp(name, "Na") == 0)
                Sodium = j;
            else if (strcmp(name, "Al") == 0)
                Aluminium = j;
            else if (strcmp(name, "Ar") == 0)
                Argon = j;
            else if (strcmp(name, "Ni") == 0)
                Nickel = j;
#endif // UM_CHEMISTRY
            else if (strcmp(name, "O") == 0)
                Oxygen = j;
            else if (strcmp(name, "Fe") == 0)
                Iron = j;
            else if (strcmp(name, "Ej") == 0)
                FillEl = j;
        }

        Hyd = LT_NMet - 1;
        strcpy(MetNames[Hyd], "H");

#ifdef UM_CHEMISTRY
        printf("\n:: Hyd %d, Hel %d, Carbon %d, Nitrogen %d, Magnesium %d, Silicon %d, Neon %d, Sulphur %d, Calcium "
               "%d, Sodium %d, Aluminium %d, Argon %d, Nickel %d, Oxygen %d, Iron %d, FillEl %d\n",
               Hyd, Hel, Carbon, Nitrogen, Magnesium, Silicon, Neon, Sulphur, Calcium, Sodium, Aluminium, Argon, Nickel,
               Oxygen, Iron, FillEl);
        printf(":: -1 means -> not present!\n\n");
#endif // UM_CHEMISTRY

        ENDRUN_IF(FillEl == -1, "you don't have specified the FillEl position.. better to do\n");

        fclose(file);

#ifdef LT_METAL_COOLING
        if (Iron == -1)
        {
            if (Oxygen >= 0)
            {
                VERBOSE(0, "you don't trace IRON. metal cooling will be calculated inferring X_Fe from X_O\n");
            }
            else
            {
                ENDRUN_MAIN("you don't trace neither IRON nor OXYGEN: metal cooling cannot be used\n");
            }
        }
#endif // LT_METAL_COOLING
    } /* close ThisTask = 0 */

    MPI_Bcast(names, sizeof(char) * LT_NMet * NAME_SIZE, MPI_BYTE, 0, MYMPI_COMM_WORLD);

    MPI_Bcast(&Iron, sizeof(int), MPI_BYTE, 0, MYMPI_COMM_WORLD);
    MPI_Bcast(&Oxygen, sizeof(int), MPI_BYTE, 0, MYMPI_COMM_WORLD);
    MPI_Bcast(&Hel, sizeof(int), MPI_BYTE, 0, MYMPI_COMM_WORLD);
    MPI_Bcast(&FillEl, sizeof(int), MPI_BYTE, 0, MYMPI_COMM_WORLD);
#ifdef UM_METAL_COOLING
    MPI_Bcast(&Carbon, sizeof(int), MPI_BYTE, 0, MYMPI_COMM_WORLD);
    MPI_Bcast(&Nitrogen, sizeof(int), MPI_BYTE, 0, MYMPI_COMM_WORLD);
    MPI_Bcast(&Magnesium, sizeof(int), MPI_BYTE, 0, MYMPI_COMM_WORLD);
    MPI_Bcast(&Silicon, sizeof(int), MPI_BYTE, 0, MYMPI_COMM_WORLD);
#endif // UM_METAL_COOLING

    return;
}

/* | ------------------------------------------------------------------------------------------------------ |
 * |
 * |  YIELDS
 * | ------------------------------------------------------------------------------------------------------ |
 */

#define LL 1500

double *DataSpace;

int read_yields_file(FILE *file, char *DataSpaceName, int *Zbins_dim, int *Mbins_dim, double **Zbins, double **Mbins,
                     double *Yields[LT_NMet])
{
    char s[LL];
    int nonproc = 0;

    /*   read the number of metal bins (1 for metal
       independent yields) */
    do
    {
        fgets(s, LL, file);
        if ((strcasestr(s, "nonproc on") != NULL) || (strcasestr(s, "non proc on") != NULL) ||
            (strcasestr(s, "nonprocon") != NULL))
            nonproc = 1;
    } while (strchr("%# \n", s[0]) != 0x0);
    /*    allocate space for metal bins and read/store them */

    *Zbins_dim = atoi(s);
    *Zbins = (double *)mymalloc("Zbins", *Zbins_dim * sizeof(double));
    memset(*Zbins, 0, *Zbins_dim * sizeof(double));
    if (*Zbins_dim > 1)
    {
        do
        {
            fgets(s, LL, file);
        } while (strchr("%# \n", s[0]) != 0x0);

        for (int j = 0; j < *Zbins_dim; j++)
            sscanf(&s[0], "%lg%[^n]s", *Zbins + j, &s[0]);
    }
    else
        (*Zbins)[0] = 0;

    /*    allocate space for mass bins and read/store them */
    do
    {
        fgets(s, LL, file);
    } while (strchr("%# \n", s[0]) != 0x0);
    *Mbins_dim = atoi(s); //  <======== HERE
    *Mbins = (double *)mymalloc("Mbins", *Mbins_dim * sizeof(double));
    memset(*Mbins, 0, *Mbins_dim * sizeof(double));

    if (*Mbins_dim > 1)
    {
        do
        {
            fgets(s, LL, file);
        } while (strchr("%# \n", s[0]) != 0x0);
        for (int j = 0; j < *Mbins_dim; j++)
            sscanf(&s[0], "%lg%[^n]s", *Mbins + j, &s[0]);
    }
    else
        (*Mbins)[0] = 0;

    DataSpace = (double *)mymalloc(DataSpaceName, (LT_NMet * *Zbins_dim * *Mbins_dim) * sizeof(double));
    memset(DataSpace, 0, (LT_NMet * *Zbins_dim * *Mbins_dim) * sizeof(double));

    Yields[0] = &DataSpace[0];
    for (int j = 1; j < LT_NMet; j++)
        Yields[j] = Yields[j - 1] + *Zbins_dim * *Mbins_dim;

    /* actually read yields. they are organized in subsequent blocks, one for each
       metal bin. each block is a table, whose rows refer to a single element and
       columns to the mass array */
    int zbin = 0;
    while (zbin < *Zbins_dim)
    {
        do
        {
            fgets(s, LL, file);
        } while (!feof(file) && (strchr("%# \n", s[0]) != 0x0));

        int Ej_is_present = 0;
        long double *accumulate = (long double *)mymalloc("accumulate", sizeof(long double) * *Mbins_dim);
        memset(accumulate, 0, sizeof(long double) * *Mbins_dim);

        while (!feof(file) && (strchr("%# \n", s[0]) == 0x0))
        {
            // get the element for this row
            char name[6];
            sscanf(s, "%5s %[^\n]s", name, &s[0]);

            // check whether this is an element you want
            int this_is_Ej = 0;
            int el = 0;
            for (el = 0; el < LT_NMet; el++)
                if (strcmp(name, MetNames[el]) == 0)
                {
                    if (strcmp(name, "Ej") == 0)
                    {
                        this_is_Ej = 1;
                        Ej_is_present = 1;
                    }
                    break;
                }

            for (int i = 0; i < *Mbins_dim; i++)
            {
                double temp;
                sscanf(s, "%lg%[^\n]s", &temp, &s[0]);
                if (el < LT_NMet)
                    Yields[el][*Mbins_dim * zbin + i] = temp;
                if (!this_is_Ej)
                    accumulate[i] += temp;
            }

            fgets(s, LL, file);
        }

        if (!Ej_is_present)
        {
            printf("<Ej element not present for zbin %d; "
                   "using the sum of all read elements>\n",
                   zbin);
            fflush(stdout);
            for (int i = 0; i < *Mbins_dim; i++)
                Yields[FillEl][*Mbins_dim * zbin + i] = accumulate[i];
        }

        myfree(accumulate);

        zbin++;
    }
    return nonproc;
}

void read_yields_specie(const char *Specie, int Nset, int **myZbins_dim, double ***myZbins, int **myMbins_dim,
                        double ***myMbins, double ****Yields, double ***Ej, int **NonProcOn, char *datafile)
{
    char buff[300], name[100];

    FILE *file;

    /*
     * read in Sn Ia yields
     */

    sprintf(name, "%s_Zbins_dim", Specie);
    *myZbins_dim = (int *)mymalloc(name, sizeof(int) * Nset);
    memset(*myZbins_dim, 0, sizeof(int) * Nset);

    sprintf(name, "%s_Zbins", Specie);
    *myZbins = (double **)mymalloc(name, sizeof(double *) * Nset);
    memset(*myZbins, 0, sizeof(double *) * Nset);

    sprintf(name, "%s_Mbins_dim", Specie);
    *myMbins_dim = (int *)mymalloc(name, sizeof(int) * Nset);
    memset(*myMbins_dim, 0, sizeof(int) * Nset);

    sprintf(name, "%s_Mbins", Specie);
    *myMbins = (double **)mymalloc(name, sizeof(double *) * Nset);
    memset(*myMbins, 0, sizeof(double *) * Nset);

    sprintf(name, "%s_Yields", Specie);
    *Yields = (double ***)mymalloc(name, sizeof(double **) * Nset);
    memset(*Yields, 0, sizeof(double **) * Nset);

    sprintf(name, "%s_Ej", Specie);
    *Ej = (double **)mymalloc(name, Nset * sizeof(double *));

    for (int set = 0; set < Nset; set++)
    {
        sprintf(buff, "%s_Yields_set_%02d_rep", Specie, set);
        (*Yields)[set] = (double **)mymalloc(buff, LT_NMet * sizeof(double *));
    }

    sprintf(buff, "%s_NonProcOn", Specie);
    *NonProcOn = (int *)mymalloc(buff, Nset * sizeof(int));
    memset(*NonProcOn, 0, Nset * sizeof(int));

    for (int set = 0; set < Nset; set++)
    {
        sprintf(name, "%s_Yields_set_%02d", Specie, set);

        if (ThisTask == 0)
        {
            int check_file_open = 0;
            if (Nset > 1)
                sprintf(buff, "%s.%03d", datafile, set);
            else
                strcpy(buff, datafile);
            if ((file = fopen(buff, "r")) == NULL)
            {
                ENDRUN_MAIN("I can't open %s data input file: <%s>\n", Specie, buff);
            }
            else
            {
                check_file_open = 1;
                if (((*NonProcOn)[set] =
                         read_yields_file(file, &name[0], (*myZbins_dim + set), (*myMbins_dim + set),
                                          &((*myZbins)[set]), &((*myMbins)[set]), &((*Yields)[set][0]))))
                    fprintf(FdSnInit, "%s yields in Set %d need to account for non processed metals\n", Specie, set);
                fclose(file);
            }
            ENDRUN_IF(!check_file_open, "yields input file < %s > could not be open for Specie %s\n", buff, Specie);
        }

        MPI_Bcast(&((*NonProcOn)[set]), 1, MPI_INT, 0, MYMPI_COMM_WORLD);
        MPI_Bcast(&((*myZbins_dim)[set]), 1, MPI_INT, 0, MYMPI_COMM_WORLD);
        MPI_Bcast(&((*myMbins_dim)[set]), 1, MPI_INT, 0, MYMPI_COMM_WORLD);

        if (ThisTask != 0)
        {
            sprintf(buff, "%s_Zbins_set_%02d", Specie, set);
            (*myZbins)[set] = (double *)mymalloc(buff, (*myZbins_dim)[set] * sizeof(double));

            sprintf(buff, "%s_Mbins_set_%02d", Specie, set);
            (*myMbins)[set] = (double *)mymalloc(buff, (*myMbins_dim)[set] * sizeof(double));

            sprintf(buff, "%s_Yields_set_%02d", Specie, set);
            DataSpace =
                (double *)mymalloc(name, (LT_NMet * (*myZbins_dim)[set] * (*myMbins_dim)[set]) * sizeof(double));

            (*Yields)[set][0] = &DataSpace[0];
            for (int i = 1; i < LT_NMet; i++)
                (*Yields)[set][i] = (*Yields)[set][i - 1] + (*myZbins_dim)[set] * (*myMbins_dim)[set];
        }
        MPI_Bcast((*myZbins)[set], (*myZbins_dim)[set] * sizeof(double), MPI_BYTE, 0, MYMPI_COMM_WORLD);
        MPI_Bcast((*myMbins)[set], (*myMbins_dim)[set] * sizeof(double), MPI_BYTE, 0, MYMPI_COMM_WORLD);
        MPI_Bcast(DataSpace, LT_NMet * (*myZbins_dim)[set] * (*myMbins_dim)[set] * sizeof(double), MPI_BYTE, 0,
                  MYMPI_COMM_WORLD);

        /* Ej[set] will contain the total ejected mass in all element present in file
         * for each couple (Zbin,Mbin)
         */

#define TOLERANCE 0.001 // set the tolerance for the Ej to be considered 0
        // let's set this to 1e-7 at the end,
        // I'm keeping 0.01 for backward compatibility

        // allocate space
        sprintf(buff, "%s_Ej_set_%02d", Specie, set);
        (*Ej)[set] = (double *)mymalloc(buff, (*myZbins_dim)[set] * (*myMbins_dim)[set] * sizeof(double));

        // copy the Ej element from the table into Ej
        memcpy((void *)(*Ej)[set], (void *)(*Yields)[set][FillEl],
               (size_t)((*myZbins_dim)[set] * (*myMbins_dim)[set] * sizeof(double)));

        /* In the tables, the element FillEl will contain the difference between the total ejected mass
           and the sum of ejecta from the used elements (which can be less than those present in the file) */
        for (int zbin = 0; zbin < (*myZbins_dim)[set]; zbin++)
            for (int mbin = 0; mbin < (*myMbins_dim)[set]; mbin++)
            {
                // accumulate onto FillEl all the elements that are
                // explicitly tracked
                long double sum_of_tracked_el = 0;
                for (int k = 0; k < LT_NMet; k++)
                    if (k != FillEl)
                        sum_of_tracked_el += (*Yields)[set][k][zbin * (*myMbins_dim)[set] + mbin];

                long double difference =
                    abs((long double)((*Ej)[set][zbin * (*myMbins_dim)[set] + mbin]) - sum_of_tracked_el);

                long double compare = difference;
                if ((*Ej)[set][zbin * (*myMbins_dim)[set] + mbin] > 0)
                    compare = difference / (*Ej)[set][zbin * (*myMbins_dim)[set] + mbin];

                if (compare < TOLERANCE)
                    // the difference (relative difference, in the case in which Ej > 0)
                    // between the sum of all the elements and the total of the tracked elements
                    // is that low that deem it as a round-off;
                    // so, let's put it to zero
                    //
                    (*Yields)[set][FillEl][zbin * (*myMbins_dim)[set] + mbin] = 0;

                else
                {
                    // FillEl is set to the difference between the sum of all the elements
                    // and the total of tracked element
                    //

                    difference = (long double)((*Ej)[set][zbin * (*myMbins_dim)[set] + mbin]) - sum_of_tracked_el;

                    if (difference < 0)
                    {
                        // the difference turns out to be negative
                        // that is un unpleasant situation:
                        // (i)  the first case is that Ej is slightly < FilleEl, but the
                        //      abs difference is larger than TOLERANCE;
                        //      that is some inaccuracy in the table
                        // (ii) the second case is that Ej < FillEl: that should be
                        //      considered as an error in the tables
                        //
                        if (ThisTask == 0)
                        {
                            printf("    --> warning: the fill elements in set %d, zbin %d ( %g ), mass bin %d (%g ) is "
                                   "smaller"
                                   " (%g vs %g) than the sum of the collected elements!\n"
                                   "        better to force it to zero\n",
                                   set, zbin, (*myZbins)[set][zbin], mbin, (*myMbins)[set][mbin],
                                   (*Ej)[set][zbin * (*myMbins_dim)[set] + mbin], (double)sum_of_tracked_el);
                            fflush(stdout);
                        }
                        (*Yields)[set][FillEl][zbin * (*myMbins_dim)[set] + mbin] = 0;
                    }
                    else
                        (*Yields)[set][FillEl][zbin * (*myMbins_dim)[set] + mbin] = difference;
                }
            }
    }

    return;
}

void ReadYields(int readIa, int readII, int readAGB)
{
    if (readII)
    {
        if (ThisTask == 0)
            printf("reading yields for SnII.. \n");
        fflush(stdout);
        read_yields_specie("SnII", All.SnIIDataSetNum, &IIZbins_dim, &IIZbins, &IIMbins_dim, &IIMbins, &SnIIYields,
                           &SnIIEj, &NonProcOn_II, All.SnIIdataFileName);
        if (ThisTask == 0)
            printf("done\n");
        fflush(stdout);
    }

    if (readIa)
    {
        if (ThisTask == 0)
            printf("reading yields for SnIa.. \n");
        fflush(stdout);
        read_yields_specie("SnIa", All.SnIaDataSetNum, &IaZbins_dim, &IaZbins, &IaMbins_dim, &IaMbins, &SnIaYields,
                           &SnIaEj, &NonProcOn_Ia, All.SnIaDataFile);
        if (ThisTask == 0)
            printf("done\n");
        fflush(stdout);
    }

    if (readAGB)
    {
        if (ThisTask == 0)
            printf("reading yields for AGB.. \n");
        fflush(stdout);
        read_yields_specie("AGB", All.AGBDataSetNum, &AGBZbins_dim, &AGBZbins, &AGBMbins_dim, &AGBMbins, &AGBYields,
                           &AGBEj, &NonProcOn_AGB, All.AGBdataFileName);
        if (ThisTask == 0)
            printf("done\n");
        fflush(stdout);
    }
    return;
}

// -==================================================================================

#endif
