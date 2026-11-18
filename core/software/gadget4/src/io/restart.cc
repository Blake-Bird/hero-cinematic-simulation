/*******************************************************************************
 * \copyright   This file is part of the GADGET4 N-body/SPH code developed
 * \copyright   by Volker Springel. Copyright (C) 2014-2020 by Volker Springel
 * \copyright   (vspringel@mpa-garching.mpg.de) and all contributing authors.
 *******************************************************************************/

/*! \file restart.cc
 *
 * \brief handles the reading/writing of restart files
 */

// clang-format off
#include "gadgetconfig.h"
// SIDM clocks/counters are in All and ResearchSourceState, written with native
// restarts. Differential model/table identities are fixed All parameters; the
// retained table is SHA256 checked again on the first resumed source interval.
// Publication/restart boundaries occur only after the complete synchronized
// source interval, never between unfinished conflict-free MPI rounds.
// clang-format on

#include "../io/restart.h"
#include "../io/checkpoint_receipt.h"
#include "../io/checkpoint_epoch.h"
#include "../io/checkpoint_layout.h"
#include "research_restart_fingerprint.h"

#include <gsl/gsl_rng.h>
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <unistd.h>
#include <errno.h>
#include <fcntl.h>
#include <zlib.h>

#include "../data/allvars.h"
#include "../data/dtypes.h"
#include "../data/mymalloc.h"
#include "../domain/domain.h"
#include "../io/io.h"
#include "../lightcone/lightcone.h"
#include "../logs/logs.h"
#include "../logs/timer.h"
#include "../main/simulation.h"
#include "../mpi_utils/mpi_utils.h"
#include "../ngbtree/ngbtree.h"
#include "../system/system.h"
#include "../time_integration/timestep.h"

void restart::write(sim *Sim_ptr)
{
  Sim = Sim_ptr;
  do_restart(MODUS_WRITE);
}

/*! \brief This function loads the last restart file.
 *
 * Some parameters of the parameter file might be changed between restarting.
 * This function ensures that only the allowed parameters change,
 * otherwise the old value from the restart file is taken.
 * If the end time of the simulation changed readjust_timebase() is called in the end.
 */
void restart::load(sim *Sim_ptr)
{
  Sim                           = Sim_ptr;
  global_data_all_processes all = All; /* save global variables. (will be read from restart file) */

  do_restart(MODUS_READ); /* ... read restart file. Note: This also resets
                          all variables in the struct `All'.
                          However, during the run, some variables in the parameter
                          file are allowed to be changed, if desired. These need to
                          copied in the way below.
                        */

  /* now update those parameters that were changed in the parameterfile, and where a change is allowed */

  for(int i = 0; i < All.NParameters; i++)
    {
      if(All.ParametersChangeable[i] == PARAM_CHANGEABLE)
        {
          size_t off = (char *)All.ParametersValue[i] - (char *)&All;

          if(off > sizeof(All))
            Terminate("Invalid parameter pointer: '%s'  i=%d off=%lld\n", All.ParametersTag[i], i, (long long)off);

          switch(All.ParametersType[i])
            {
              case PARAM_DOUBLE:
                {
                  double *old_dbl = (double *)((char *)&All + off);
                  double *new_dbl = (double *)((char *)&all + off);

                  if(*new_dbl != *old_dbl)
                    {
                      mpi_printf("RESTART: %s modified from %g to %g while restarting at Time=%g\n", All.ParametersTag[i], *old_dbl,
                                 *new_dbl, All.Time);
                      *old_dbl = *new_dbl;
                    }
                }
                break;
              case PARAM_STRING:
                {
                  char *old_p = (char *)&All + off;
                  char *new_p = (char *)&all + off;
                  if(strncmp(new_p, old_p, MAXLEN_PARAM_VALUE))
                    {
                      mpi_printf("RESTART: %s modified from '%s' to '%s' while restarting at Time=%g\n", All.ParametersTag[i], old_p,
                                 new_p, All.Time);
                      strncpy(old_p, new_p, MAXLEN_PARAM_VALUE);
                    }
                }
                break;
              case PARAM_INT:
                {
                  int *old_int = (int *)((char *)&All + off);
                  int *new_int = (int *)((char *)&all + off);

                  if(*new_int != *old_int)
                    {
                      mpi_printf("RESTART: %s modified from %d to %d while restarting at Time=%g\n", All.ParametersTag[i], *old_int,
                                 *new_int, All.Time);
                      *old_int = *new_int;
                    }
                }
                break;
            }
        }
    }

  /* change in the output list table is always allowed */
  All.OutputListLength = all.OutputListLength;
  memcpy(All.OutputListTimes, all.OutputListTimes, sizeof(double) * All.OutputListLength);
  memcpy(All.OutputListFlag, all.OutputListFlag, sizeof(char) * All.OutputListLength);

  /* if the final time is changed, we process this with a special function */
  if(All.TimeMax != all.TimeMax)
    readjust_timebase(All.TimeMax, all.TimeMax);
}

void restart::backup_restartfiles(int task)
{
  char buf[MAXLEN_PATH_EXTRA];
  char buf_bak[MAXLEN_PATH_EXTRA];
  FILE *fcheck         = NULL;
  int bak_files_status = 0;

  mpi_printf("RESTART: Backing up restart files...\n");

  snprintf(buf, MAXLEN_PATH_EXTRA, "%s/restartfiles/%s.%d", All.OutputDir, "restart", task);
  snprintf(buf_bak, MAXLEN_PATH_EXTRA, "%s/restartfiles/bak-%s.%d", All.OutputDir, "restart", task);

  if((fcheck = fopen(buf, "r")))
    {
      fclose(fcheck);

      /* Preserve the current rank until its replacement is complete. */
      char temporary[MAXLEN_PATH_EXTRA];
      snprintf(temporary, sizeof(temporary), "%s.tmp.%d", buf_bak, (int)getpid());
      if(link(buf, temporary) != 0 || rename(temporary, buf_bak) != 0)
        Terminate("RESTART: cannot preserve previous rank checkpoint: %s", buf);
      char receipt[MAXLEN_PATH_EXTRA], receipt_bak[MAXLEN_PATH_EXTRA];
      char epoch[MAXLEN_PATH_EXTRA], epoch_bak[MAXLEN_PATH_EXTRA];
      snprintf(epoch, sizeof(epoch), "%s.epoch.json", buf);
      snprintf(epoch_bak, sizeof(epoch_bak), "%s.epoch.json", buf_bak);
      snprintf(temporary, sizeof(temporary), "%s.tmp.%d", epoch_bak, (int)getpid());
      if(link(epoch, temporary) == 0)
        { if(rename(temporary, epoch_bak) != 0) Terminate("RESTART: cannot preserve checkpoint epoch"); }
      else if(errno != ENOENT)
        Terminate("RESTART: cannot preserve checkpoint epoch");
      snprintf(receipt, sizeof(receipt), "%s.checksum.json", buf);
      snprintf(receipt_bak, sizeof(receipt_bak), "%s.checksum.json", buf_bak);
      snprintf(temporary, sizeof(temporary), "%s.tmp.%d", receipt_bak, (int)getpid());
      if(link(receipt, temporary) == 0)
        { if(rename(temporary, receipt_bak) != 0) Terminate("RESTART: cannot preserve checkpoint receipt"); }
      else if(errno != ENOENT)
        Terminate("RESTART: cannot preserve checkpoint receipt");

      char directory[MAXLEN_PATH_EXTRA];
      snprintf(directory, sizeof(directory), "%s/restartfiles", All.OutputDir);
      int descriptor = open(directory, O_RDONLY);
      if(descriptor < 0) Terminate("RESTART: cannot open backup directory");
      if(fsync(descriptor) != 0 && errno != EINVAL && errno != ENOTSUP)
        Terminate("RESTART: backup directory flush failed");
      if(close(descriptor) != 0) Terminate("RESTART: backup directory close failed");
      bak_files_status = 1;
    }

  int bak_files_status_sum;
  MPI_Allreduce(&bak_files_status, &bak_files_status_sum, 1, MPI_INT, MPI_SUM, Communicator);

  if(bak_files_status_sum != NTask && bak_files_status_sum != 0)
    mpi_printf("RESTART: some (%d) restart files were preserved as bak, but some (%d) weren't - something is very possibly wrong!",
               bak_files_status, NTask - bak_files_status);
  else if(bak_files_status_sum == NTask)
    mpi_printf("RESTART: done preserving pre-existing restart files as bak files.\n");
  else if(bak_files_status_sum == 0)
    mpi_printf("RESTART: no pre-existing restart files for renaming were found.\n");
}

/*! \brief This function reads or writes the restart files.
 *
 * Each processor writes its own restart file, with the
 * I/O being done in parallel. To avoid congestion of the disks
 * you can tell the program to restrict the number of files
 * that are simultaneously written to MaxFilesWithConcurrentIO.
 *
 * \param modus if modus>0  the restart()-routine reads,
 * if modus==0 it writes a restart file.
 */
void restart::do_restart(int modus)
{
#ifdef DO_NOT_PRODUCE_BIG_OUTPUT
  if(modus == MODUS_WRITE)
    {
      mpi_printf("RESTART: Omitting writing restart files.\n");
      return;
    }
#endif

  TIMER_START(CPU_RESTART);

  double t0 = Logs.second();
  reset_io_byte_count();

  if(modus == MODUS_READ)
    mpi_printf("RESTART: Loading restart files...\n");
  else if(modus == MODUS_WRITE)
    mpi_printf("RESTART: Writing restart files.\n");

  /* create directory for restartfiles */
  if(ThisTask == 0 && modus == MODUS_WRITE)
    {
      char buf[MAXLEN_PATH_EXTRA];
      snprintf(buf, MAXLEN_PATH_EXTRA, "%s/restartfiles", All.OutputDir);
      mkdir(buf, 02755);
    }
  MPI_Barrier(Communicator);

  if(All.MaxFilesWithConcurrentIO > NTask)
    {
      mpi_printf("NOTICE: MaxFilesWithConcurrentIO has been reduced to the number of processors\n");
      All.MaxFilesWithConcurrentIO = NTask;
    }

  if(All.MaxFilesWithConcurrentIO < 1)
    {
      mpi_printf("NOTICE: MaxFilesWithConcurrentIO has been set to be equal to the number of processors\n");
      All.MaxFilesWithConcurrentIO = NTask;
    }

  files_concurrent = All.MaxFilesWithConcurrentIO;

  files_groups = NTask / All.MaxFilesWithConcurrentIO;

  if(NTask % All.MaxFilesWithConcurrentIO)
    files_groups++;

  if(modus == MODUS_WRITE)
    {
      if(ThisTask == 0)
        {
          struct timeval stamp;
          if(gettimeofday(&stamp, NULL) != 0) Terminate("RESTART: cannot timestamp checkpoint generation");
          checkpoint_generation = (unsigned long long)stamp.tv_sec * 1000000ULL + stamp.tv_usec;
        }
      MPI_Bcast(&checkpoint_generation, 1, MPI_UNSIGNED_LONG_LONG, 0, Communicator);
      backup_restartfiles(ThisTask);
    }

  if(modus == MODUS_READ)
    {
      char payload[MAXLEN_PATH_EXTRA], metadata[MAXLEN_PATH_EXTRA];
      snprintf(payload, sizeof(payload), "%s/restartfiles/restart.%d", All.OutputDir, ThisTask);
      snprintf(metadata, sizeof(metadata), "%s.checksum.json", payload);
      native_checkpoint_receipt receipt;
      if(!native_checkpoint_verify(payload, metadata, ThisTask, NTask, receipt))
        Terminate("RESTART: checkpoint receipt/checksum invalid before particle allocation (rank=%d)", ThisTask);
      if(receipt.comoving != All.ComovingIntegrationOn)
        Terminate("RESTART: native checkpoint clock representation differs from requested integration frame");
      checkpoint_expected_comoving = receipt.comoving;
      unsigned long long minimum, maximum;
      double minimum_a, maximum_a;
      MPI_Allreduce(&receipt.generation, &minimum, 1, MPI_UNSIGNED_LONG_LONG, MPI_MIN, Communicator);
      MPI_Allreduce(&receipt.generation, &maximum, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, Communicator);
      MPI_Allreduce(&receipt.scale_factor, &minimum_a, 1, MPI_DOUBLE, MPI_MIN, Communicator);
      MPI_Allreduce(&receipt.scale_factor, &maximum_a, 1, MPI_DOUBLE, MPI_MAX, Communicator);
      if(minimum != maximum || minimum_a != maximum_a)
        Terminate("RESTART: mixed checkpoint generation/epoch before particle allocation");
      checkpoint_generation = receipt.generation;
      checkpoint_expected_scale_factor = receipt.scale_factor;
      mpi_printf("RESTART: verified native checkpoint checksums/generation before particle allocation.\n");
    }

  /* now work the I/O of the files, controlled by scheduler to achieve optimum I/O bandwidth under the constraint of a maximum number
   * for the concurrent file access */
  work_files(modus);

  /* check whether the restarts are all at the same time */
  if(modus == MODUS_READ) /* read */
    {
      global_data_all_processes all_task0;

      if(ThisTask == 0)
        all_task0 = All;

      MPI_Bcast(&all_task0, sizeof(global_data_all_processes), MPI_BYTE, 0, Communicator);

      if(all_task0.Time != All.Time)
        Terminate("The restart file on task=%d is not consistent with the one on task=0\n", ThisTask);
    }

  long long byte_count = get_io_byte_count(), byte_count_all;
  sumup_longs(1, &byte_count, &byte_count_all, Communicator);

  double t1 = Logs.second();

  mpi_printf("RESTART: done. load/save took %g sec, total size %g MB, corresponds to effective I/O rate of %g MB/sec\n",
             Logs.timediff(t0, t1), byte_count_all / (1024.0 * 1024.0), byte_count_all / (1024.0 * 1024.0) / Logs.timediff(t0, t1));

  TIMER_STOP(CPU_RESTART);
}

void restart::polling(int modus)
{
  if(ThisTask == 0)
    if(files_completed < NTask)
      {
        MPI_Status status;
        int flag;

        /* now check for a completion message  */
        MPI_Iprobe(MPI_ANY_SOURCE, TAG_KEY, Communicator, &flag, &status);

        if(flag)
          {
            int source = status.MPI_SOURCE;

            int dummy;
            MPI_Recv(&dummy, 1, MPI_INT, source, TAG_KEY, Communicator, MPI_STATUS_IGNORE);
            files_completed++;

            if(files_started < NTask)
              {
                if((files_started % files_concurrent) == 0)
                  {
                    if(modus == MODUS_READ)
                      mpi_printf("RESTART: Loading restart files group #%d out of %d...\n", (files_started / files_concurrent) + 1,
                                 files_groups);
                    else if(modus == MODUS_WRITE)
                      mpi_printf("RESTART: Writing restart files group #%d out of %d...\n", (files_started / files_concurrent) + 1,
                                 files_groups);
                  }

                /* send start signal */
                MPI_Ssend(&ThisTask, 1, MPI_INT, seq[files_started++].thistask, TAG_N, Communicator);
              }
          }
      }
}

void restart::work_files(int modus)
{
  if(ThisTask == 0)
    if(!(seq = (seq_data *)malloc(NTask * sizeof(seq_data))))
      Terminate("can't allocate seq_data");

  seq_data seq_loc;
  seq_loc.thistask   = ThisTask;
  seq_loc.rankinnode = RankInThisNode;
  seq_loc.thisnode   = ThisNode;

  MPI_Gather(&seq_loc, sizeof(seq_data), MPI_BYTE, seq, sizeof(seq_data), MPI_BYTE, 0, Communicator);

  if(modus == MODUS_READ)
    MPI_Comm_split(Communicator, Shmem.Island_Smallest_WorldTask, 0, &Sim->NgbTree.TreeSharedMemComm);

  if(ThisTask == 0)
    {
      std::sort(seq, seq + NTask);

      files_started   = 0;
      files_completed = 0;

      if((files_started % files_concurrent) == 0)
        {
          if(modus == MODUS_READ)
            mpi_printf("RESTART: Loading restart files group #%d out of %d...\n", (files_started / files_concurrent) + 1,
                       files_groups);
          else if(modus == MODUS_WRITE)
            mpi_printf("RESTART: Writing restart files group #%d out of %d...\n", (files_started / files_concurrent) + 1,
                       files_groups);
        }

      for(int i = 1; i < All.MaxFilesWithConcurrentIO; i++)
        {
          files_started++;
          MPI_Ssend(&ThisTask, 1, MPI_INT, seq[i].thistask, TAG_N, Communicator);
        }

      files_started++;
      contents_restart_file(modus);
      files_completed++;

      if(files_started < NTask)
        {
          if((files_started % files_concurrent) == 0)
            {
              if(modus == MODUS_READ)
                mpi_printf("RESTART: Loading restart files group #%d out of %d...\n", (files_started / files_concurrent) + 1,
                           files_groups);
              else if(modus == MODUS_WRITE)
                mpi_printf("RESTART: Writing restart files group #%d out of %d...\n", (files_started / files_concurrent) + 1,
                           files_groups);
            }

          /* send start signal */
          MPI_Ssend(&ThisTask, 1, MPI_INT, seq[files_started++].thistask, TAG_N, Communicator);
        }

      while(files_completed < NTask)
        polling(modus);

      free(seq);
    }
  else
    {
      /* wait for start signal */
      int dummy;
      MPI_Recv(&dummy, 1, MPI_INT, 0, TAG_N, Communicator, MPI_STATUS_IGNORE); /* wait until we are told to start */

      contents_restart_file(modus);

      /* send back completion notice */
      MPI_Ssend(&ThisTask, 1, MPI_INT, 0, TAG_KEY, Communicator);
    }

  if(modus == MODUS_READ)
    Sim->NgbTree.treeallocate_share_topnode_addresses();
}

void restart::contents_restart_file(int modus)
{
  char buf[MAXLEN_PATH_EXTRA], staged[MAXLEN_PATH_EXTRA];
  snprintf(buf, MAXLEN_PATH_EXTRA, "%s/restartfiles/%s.%d", All.OutputDir, "restart", ThisTask);
  snprintf(staged, sizeof(staged), "%s.tmp.%d", buf, (int)getpid());

  if(modus == MODUS_READ)
    {
      if(!(fd = fopen(buf, "r")))
        {
          Terminate("RESTART: Restart file '%s' not found.\n", buf);
        }
    }
  else if(modus == MODUS_WRITE)
    {
      if(!(fd = fopen(staged, "w")))
        {
          Terminate("RESTART: Restart file '%s' cannot be opened.\n", buf);
        }
    }
  else
    Terminate("unknown modus\n");

  native_checkpoint_layout expected = native_checkpoint_layout_make(
      All.get_data_size(), sizeof(particle_data), sizeof(sph_particle_data), NTask,
      research_restart_fingerprint);
  native_checkpoint_layout saved = expected;
  byten(&saved, sizeof(saved), modus);
  if(modus == MODUS_READ && !native_checkpoint_layout_equal(saved, expected))
    Terminate("RESTART: incompatible source/configuration/layout before restoring state or allocating particles");

  /* common data  */
  byten(All.get_data_ptr(), All.get_data_size(), modus);
  if(modus == MODUS_READ && (All.Time != checkpoint_expected_scale_factor || All.ComovingIntegrationOn != checkpoint_expected_comoving))
    Terminate("RESTART: native checkpoint time disagrees with checksum receipt before particle allocation");

  /* converter data to integer coordinates*/
  intposconvert *converter = &Sim->Sp;
  byten(converter, sizeof(intposconvert), modus);

  in(&Sim->Sp.MaxPart, modus);
  in(&Sim->Sp.MaxPartSph, modus);
  byten(&Sim->Sp.TotNumPart, sizeof(Sim->Sp.TotNumPart), modus);
  byten(&Sim->Sp.TotNumGas, sizeof(Sim->Sp.TotNumGas), modus);

  if(modus == MODUS_READ) /* read */
    Sim->Sp.allocate_memory();

  in(&Sim->Sp.NumPart, modus);

  /* Particle data  */
  byten(&Sim->Sp.P[0], Sim->Sp.NumPart * sizeof(particle_data), modus);

  in(&Sim->Sp.NumGas, modus);

  if(Sim->Sp.NumGas > 0)
    {
      /* Sph-Particle data  */
      byten(&Sim->Sp.SphP[0], Sim->Sp.NumGas * sizeof(sph_particle_data), modus);
    }

#if defined(MERGERTREE) && defined(SUBFIND)
  byten(&Sim->MergerTree.PrevTotNsubhalos, sizeof(long long), modus);
  byten(&Sim->MergerTree.PrevNsubhalos, sizeof(int), modus);
#endif

  /* lightcone particle data  */
#ifdef LIGHTCONE_PARTICLES
  /* converter data to integer coordinates*/
  intposconvert *converter_lp = &Sim->Lp;
  byten(converter_lp, sizeof(intposconvert), modus);

  in(&Sim->Lp.MaxPart, modus);
  byten(&Sim->Lp.TotNumPart, sizeof(Sim->Lp.TotNumPart), modus);

  if(modus == MODUS_READ) /* read */
    Sim->Lp.allocate_memory();

  in(&Sim->Lp.NumPart, modus);
  byten(&Sim->Lp.P[0], Sim->Lp.NumPart * sizeof(lightcone_particle_data), modus);
#endif

  /* lightcone massmap data  */
#ifdef LIGHTCONE_MASSMAPS
  in(&Sim->Mp.MaxPart, modus);

  if(modus == MODUS_READ)
    Sim->Mp.allocate_memory();

  in(&Sim->Mp.NumPart, modus);
  byten(&Sim->Mp.P[0], Sim->Mp.NumPart * sizeof(lightcone_massmap_data), modus);

  /* allocate and clear local piece of mass map if needed */
  if(modus == MODUS_READ)
    {
      Sim->LightCone.Mp->Npix = nside2npix(All.LightConeMassMapsNside);
      subdivide_evenly(Sim->LightCone.Mp->Npix, NTask, ThisTask, &Sim->LightCone.Mp->FirstPix, &Sim->LightCone.Mp->NpixLoc);

      Sim->LightCone.MassMap =
          (double *)Mem.mymalloc_movable_clear(&Sim->LightCone.MassMap, "MassMap", Sim->LightCone.Mp->NpixLoc * sizeof(double));
    }

  byten(Sim->LightCone.MassMap, Sim->LightCone.Mp->NpixLoc * sizeof(double), modus);
#endif

  /* write state of random number generator */
  byten(gsl_rng_state(random_generator), gsl_rng_size(random_generator), modus);

  byten(Logs.CPU_Step, logs::CPU_LAST * sizeof(double), modus);
  byten(Logs.CPU_Step_Stored, logs::CPU_LAST * sizeof(double), modus);
  byten(Logs.CPU_Sum, logs::CPU_LAST * sizeof(double), modus);

  /* now store variables for time integration bookkeeping */
  byten(Sim->Sp.TimeBinSynchronized, TIMEBINS * sizeof(int), modus);

  in(&Sim->Sp.TimeBinsHydro.NActiveParticles, modus);
  in(&Sim->Sp.TimeBinsGravity.NActiveParticles, modus);
  byten(&Sim->Sp.TimeBinsHydro.GlobalNActiveParticles, sizeof(long long), modus);
  byten(&Sim->Sp.TimeBinsGravity.GlobalNActiveParticles, sizeof(long long), modus);
  byten(Sim->Sp.TimeBinsHydro.ActiveParticleList, Sim->Sp.TimeBinsHydro.NActiveParticles * sizeof(int), modus);
  byten(Sim->Sp.TimeBinsGravity.ActiveParticleList, Sim->Sp.TimeBinsGravity.NActiveParticles * sizeof(int), modus);
  byten(Sim->Sp.TimeBinsHydro.NextInTimeBin, Sim->Sp.NumGas * sizeof(int), modus);
  byten(Sim->Sp.TimeBinsGravity.NextInTimeBin, Sim->Sp.NumPart * sizeof(int), modus);
  byten(Sim->Sp.TimeBinsHydro.PrevInTimeBin, Sim->Sp.NumGas * sizeof(int), modus);
  byten(Sim->Sp.TimeBinsGravity.PrevInTimeBin, Sim->Sp.NumPart * sizeof(int), modus);
  byten(Sim->Sp.TimeBinsHydro.TimeBinCount, TIMEBINS * sizeof(int), modus);
  byten(Sim->Sp.TimeBinsGravity.TimeBinCount, TIMEBINS * sizeof(int), modus);
  byten(Sim->Sp.TimeBinsHydro.FirstInTimeBin, TIMEBINS * sizeof(int), modus);
  byten(Sim->Sp.TimeBinsGravity.FirstInTimeBin, TIMEBINS * sizeof(int), modus);
  byten(Sim->Sp.TimeBinsHydro.LastInTimeBin, TIMEBINS * sizeof(int), modus);
  byten(Sim->Sp.TimeBinsGravity.LastInTimeBin, TIMEBINS * sizeof(int), modus);

#ifdef STARFORMATION
  byten(Sim->Sp.TimeBinSfr, TIMEBINS * sizeof(double), modus);
#endif

  /* now store relevant data for tree */

  in(&Sim->Domain.NTopleaves, modus);
  in(&Sim->Domain.NTopnodes, modus);

  in(&Sim->NgbTree.MaxPart, modus);
  in(&Sim->NgbTree.MaxNodes, modus);
  in(&Sim->NgbTree.NumNodes, modus);
  in(&Sim->NgbTree.NumPartImported, modus);
  in(&Sim->NgbTree.FirstNonTopLevelNode, modus);
  in(&Sim->NgbTree.ImportedNodeOffset, modus);
  in(&Sim->NgbTree.EndOfTreePoints, modus);
  in(&Sim->NgbTree.EndOfForeignNodes, modus);

  if(modus == MODUS_READ)
    {
      Sim->Domain.domain_allocate(Sim->Domain.NTopnodes);

      /* passing a negative number to the allocate call will here prevent that NgbTree.MaxPart and NgbTree.MaxNodes are recomputed */
      Sim->NgbTree.treeallocate(-1, &Sim->Sp, &Sim->Domain);

      if(Sim->NgbTree.MaxPart != 0)
        {
          Sim->NgbTree.Points   = (ngbpoint_data *)Mem.mymalloc_movable(&Sim->NgbTree.Points, "Points",
                                                                        Sim->NgbTree.NumPartImported * sizeof(ngbpoint_data));
          Sim->NgbTree.Nextnode = (int *)Mem.mymalloc_movable(
              &Sim->NgbTree.Nextnode, "Nextnode",
              (Sim->NgbTree.MaxPart + Sim->Domain.NTopleaves + Sim->NgbTree.NumPartImported) * sizeof(int));
          Sim->NgbTree.Father = (int *)Mem.mymalloc_movable(&Sim->NgbTree.Father, "Father",
                                                            (Sim->NgbTree.MaxPart + Sim->NgbTree.NumPartImported) * sizeof(int));
        }
    }

  if(Sim->Sp.TotNumGas > 0)
    {
      byten(Sim->NgbTree.Nodes + Sim->NgbTree.MaxPart + Sim->Domain.NTopnodes,
            (Sim->NgbTree.NumNodes - Sim->Domain.NTopnodes) * sizeof(ngbnode), modus);
      byten(Sim->NgbTree.Nextnode, (Sim->NgbTree.MaxPart + Sim->Domain.NTopleaves) * sizeof(int), modus);
      byten(Sim->NgbTree.Father, Sim->NgbTree.MaxPart * sizeof(int), modus);

      if(Sim->NgbTree.TreeSharedMem_ThisTask == 0)
        {
          byten(Sim->NgbTree.TopNodes + Sim->NgbTree.MaxPart, Sim->Domain.NTopnodes * sizeof(ngbnode), modus);
          byten(Sim->NgbTree.NodeIndex, Sim->Domain.NTopleaves * sizeof(int), modus);
          byten(Sim->NgbTree.NodeSibling, Sim->Domain.NTopleaves * sizeof(int), modus);
          byten(Sim->NgbTree.NodeLevel, Sim->Domain.NTopleaves * sizeof(unsigned char), modus);
        }
    }

  byten(Sim->Domain.TopNodes, Sim->Domain.NTopnodes * Sim->Domain.domain_sizeof_topnode_data(), modus);
  byten(Sim->Domain.TaskOfLeaf, Sim->Domain.NTopleaves * sizeof(int), modus);
  byten(Sim->Domain.ListOfTopleaves, Sim->Domain.NTopleaves * sizeof(int), modus);
  byten(Sim->Domain.FirstTopleafOfTask, NTask * sizeof(int), modus);
  byten(Sim->Domain.NumTopleafOfTask, NTask * sizeof(int), modus);

  if(modus == MODUS_WRITE && (fflush(fd) != 0 || fsync(fileno(fd)) != 0))
    Terminate("RESTART: checkpoint flush failed; previous rank retained");
  if(fclose(fd) != 0)
    Terminate("RESTART: checkpoint close failed");
  if(modus == MODUS_WRITE)
    {
      unsigned char block[65536];
      unsigned long crc = crc32(0L, Z_NULL, 0);
      unsigned long long bytes = 0;
      FILE *verify = fopen(staged, "rb");
      if(!verify) Terminate("RESTART: cannot checksum staged checkpoint");
      size_t count;
      while((count = fread(block, 1, sizeof(block), verify)) != 0)
        { crc = crc32(crc, block, count); bytes += count; }
      if(ferror(verify) || fclose(verify) != 0) Terminate("RESTART: checksum read failed");
      char receipt[MAXLEN_PATH_EXTRA], receipt_stage[MAXLEN_PATH_EXTRA];
      snprintf(receipt, sizeof(receipt), "%s.checksum.json", buf);
      snprintf(receipt_stage, sizeof(receipt_stage), "%s.tmp.%d", receipt, (int)getpid());
      FILE *metadata = fopen(receipt_stage, "w");
      if(!metadata) Terminate("RESTART: cannot stage checkpoint receipt");
      const char *receipt_format = All.ComovingIntegrationOn ? "{\"schema\":\"gadget-rank-checkpoint-v1\",\"rank\":%d,\"ranks\":%d,\"generation\":%llu,\"a\":%.17g,\"size_bytes\":%llu,\"crc32\":\"%08lx\"}\n" : "{\"schema\":\"gadget-rank-checkpoint-v2\",\"rank\":%d,\"ranks\":%d,\"generation\":%llu,\"native_time\":%.17g,\"size_bytes\":%llu,\"crc32\":\"%08lx\"}\n";
      if(fprintf(metadata, receipt_format, ThisTask, NTask, checkpoint_generation, All.Time, bytes, crc) < 0 ||
         fflush(metadata) != 0 || fsync(fileno(metadata)) != 0)
        Terminate("RESTART: checkpoint receipt flush failed");
      if(fclose(metadata) != 0) Terminate("RESTART: checkpoint receipt close failed");
      char epoch_stage[MAXLEN_PATH_EXTRA], epoch[MAXLEN_PATH_EXTRA];
      snprintf(epoch, sizeof(epoch), "%s.epoch.json", buf);
      if(!native_checkpoint_epoch_stage(All.OutputDir, buf, ThisTask, checkpoint_generation,
                                        All.Time, epoch_stage, sizeof(epoch_stage), All.ComovingIntegrationOn != 0))
        Terminate("RESTART: canonical native checkpoint clock invalid");
      if(rename(staged, buf) != 0 || rename(receipt_stage, receipt) != 0 || rename(epoch_stage, epoch) != 0)
        Terminate("RESTART: checkpoint publication failed; backups retained");
      char directory[MAXLEN_PATH_EXTRA];
      snprintf(directory, sizeof(directory), "%s/restartfiles", All.OutputDir);
      int descriptor = open(directory, O_RDONLY);
      if(descriptor < 0) Terminate("RESTART: cannot open checkpoint directory");
      if(fsync(descriptor) != 0 && errno != EINVAL && errno != ENOTSUP)
        Terminate("RESTART: checkpoint directory flush failed");
      if(close(descriptor) != 0) Terminate("RESTART: checkpoint directory close failed");
    }
}

/*! \brief Adjusts the timeline if the TimeMax variable is
 * increased between a restart.
 *
 * The approach taken here is to reduce the resolution of the
 * integer timeline by factors of 2 until the new final time
 * can be reached within TIMEBASE.
 *
 * \param TimeMax_old old final time
 * \param TimeMax_new new final time (must be larger than old one)
 */
void restart::readjust_timebase(double TimeMax_old, double TimeMax_new)
{
  mpi_printf("\nRESTART: All.TimeMax has been changed in the parameterfile from %g to %g. Need to adjust integer timeline.\n",
             TimeMax_old, TimeMax_new);

  if(TimeMax_new < TimeMax_old)
    Terminate("\nIt is not allowed to reduce All.TimeMax\n");

  long long ti_end;

  if(All.ComovingIntegrationOn)
    ti_end = (long long)(log(TimeMax_new / All.TimeBegin) / All.Timebase_interval);
  else
    ti_end = (long long)((TimeMax_new - All.TimeBegin) / All.Timebase_interval);

  while(ti_end > TIMEBASE)
    {
      All.Timebase_interval *= 2.0;

      ti_end /= 2;
      All.Ti_Current /= 2;

#if defined(PMGRID) && !defined(TREEPM_NOTIMESPLIT)
      All.PM_Ti_begstep /= 2;
      All.PM_Ti_endstep /= 2;
#endif

#ifdef FORCE_EQUAL_TIMESTEPS
      All.GlobalTimeStep /= 2;
#endif

      for(int n = 0; n < TIMEBINS; n++)
        All.Ti_begstep[n] /= 2;

      All.Ti_nextoutput /= 2;
      All.Ti_lastoutput /= 2;

      for(int i = 0; i < Sim->Sp.NumPart; i++)
        {
          Sim->Sp.P[i].Ti_Current = Sim->Sp.P[i].Ti_Current / 2;

          if(Sim->Sp.P[i].TimeBinGrav > 0)
            {
              int oldbin = Sim->Sp.P[i].TimeBinGrav;
              int newbin = oldbin - 1;

              if(newbin <= 0)
                Terminate("Error in readjust_timebase(). Minimum Timebin for particle %d reached.\n", i);

              Sim->Sp.TimeBinsGravity.timebin_move_particle(i, oldbin, newbin);
              Sim->Sp.P[i].TimeBinGrav = newbin;
            }

          if(Sim->Sp.P[i].getType() == 0)
            {
              if(Sim->Sp.P[i].getTimeBinHydro() > 0)
                {
                  int oldbin = Sim->Sp.P[i].getTimeBinHydro();
                  int newbin = oldbin - 1;

                  if(newbin <= 0)
                    Terminate("Error in readjust_timebase(). Minimum Timebin (hydro) for sph particle %d reached.\n", i);

                  Sim->Sp.TimeBinsHydro.timebin_move_particle(i, oldbin, newbin);
                  Sim->Sp.P[i].setTimeBinHydro(newbin);
                }
            }
        }
    }

  All.TimeMax = TimeMax_new;
}

void restart::byten(void *x, size_t n, int modus)
{
  char *p = (char *)x;

  while(n > BLKSIZE)
    {
      byten_doit(p, BLKSIZE, modus);
      p += BLKSIZE;
      n -= BLKSIZE;
      polling(modus);
    }

  if(n > 0)
    byten_doit(p, n, modus);
}

/*! \brief reads/writes n bytes to a restart file
 */
void restart::byten_doit(void *x, size_t n, int modus)
{
  if(modus == MODUS_READ)
    my_fread(x, n, 1, fd);
  else
    my_fwrite(x, n, 1, fd);
}

/*! \brief reads/writes one integer to a restart file
 *
 * \param x pointer to the integer
 * \param modus if modus>0  the restart()-routine reads,
 * if modus==0 it writes a restart file.
 */
void restart::in(int *x, int modus) { byten(x, sizeof(int), modus); }
