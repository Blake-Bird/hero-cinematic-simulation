/*******************************************************************************
 * \copyright   This file is part of the GADGET4 N-body/SPH code developed
 * \copyright   by Volker Springel. Copyright (C) 2014-2020 by Volker Springel
 * \copyright   (vspringel@mpa-garching.mpg.de) and all contributing authors.
 *******************************************************************************/

/*! \file starformation.cc
 *
 *  \brief Generic creation routines for creating star particles
 */

#include "gadgetconfig.h"

#ifdef STARFORMATION

#include <assert.h>
#include <limits.h>
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../cooling_sfr/cooling.h"
#include "../data/allvars.h"
#include "../data/dtypes.h"
#include "../data/mymalloc.h"
#include "../logs/logs.h"
#include "../logs/timer.h"
#include "../system/system.h"
#include "../time_integration/timestep.h"
#include "../data/research_stellar_birth.h"
#include "research_stellar_returns.h"
#include "research_restart_fingerprint.h"

/** \brief This routine creates star/wind particles according to their respective rates.
 *
 *  This function loops over all the active gas cells. If in a given cell the SFR is
 *  greater than zero, the probability of forming a star or a wind particle is computed
 *  and the corresponding particle is created stichastically according to the model
 *  in Springel & Hernquist (2003, MNRAS). It also saves information about the formed stellar
 *  mass and the star formation rate in the file FdSfr.
 */
void coolsfr::sfr_create_star_particles(simparticles *Sp)
{
  TIMER_START(CPU_COOLING_SFR);

  double dt, dtime;
  MyDouble mass_of_star;
  double sum_sm, total_sm, rate, sum_mass_stars, total_sum_mass_stars;
  double p = 0, pall = 0, prob, p_decide;
  double rate_in_msunperyear;
  double totsfrrate;
  double w = 0;

  All.set_cosmo_factors_for_current_time();
  uint64_t stellar_seed;
  if(!research_stellar_seed(All.StellarRandomSeed, stellar_seed))
    Terminate("StellarRandomSeed must be an exact decimal uint64");

  stars_spawned = stars_converted = 0;
  std::vector<research_stellar_birth_event> birth_events;

  sum_sm = sum_mass_stars = 0;

  for(int i = 0; i < Sp->TimeBinsHydro.NActiveParticles; i++)
    {
      int target = Sp->TimeBinsHydro.ActiveParticleList[i];
      if(Sp->P[target].getType() == 0)
        {
          if(Sp->P[target].getMass() == 0 && Sp->P[target].ID.get() == 0)
            continue; /* skip cells that have been swallowed or eliminated */

          dt = (Sp->P[target].getTimeBinHydro() ? (((integertime)1) << Sp->P[target].getTimeBinHydro()) : 0) * All.Timebase_interval;
          /*  the actual time-step */

          dtime = All.cf_atime * dt / All.cf_atime_hubble_a;

          mass_of_star = 0;
          prob         = 0;
          p            = 0;

          if(!isfinite(Sp->SphP[target].Sfr) || Sp->SphP[target].Sfr < 0)
            Terminate("Invalid native gas star formation rate");
          if(Sp->SphP[target].Sfr > 0)
            {
              if(!isfinite(Sp->SphP[target].Sfr) || !isfinite(dtime) || dtime < 0 ||
                 !isfinite(Sp->P[target].getMass()) || Sp->P[target].getMass() <= 0)
                Terminate("Nonfinite or nonpositive star-forming gas state");
              if(Sp->P[target].ResearchSourceState.stellar_random_counter == UINT64_MAX)
                Terminate("Stellar random opportunity counter exhausted");
              const uint64_t opportunity = Sp->P[target].ResearchSourceState.stellar_random_counter++;
              p = Sp->SphP[target].Sfr / ((All.UnitMass_in_g / SOLAR_MASS) / (All.UnitTime_in_s / SEC_PER_YEAR)) * dtime /
                  Sp->P[target].getMass();
              pall = p;
              if(!isfinite(p) || p < 0) Terminate("Nonfinite stellar formation hazard");
              sum_sm += Sp->P[target].getMass() * (-expm1(-p));

              w = research_stellar_uniform(stellar_seed, Sp->P[target].ID.get(), opportunity, 0);

              Sp->SphP[target].Metallicity += w * METAL_YIELD * (-expm1(-p));
              Sp->SphP[target].MassMetallicity = Sp->SphP[target].Metallicity * Sp->P[target].getMass();
              Sp->P[target].Metallicity        = Sp->SphP[target].Metallicity;
#ifdef STELLAR_EVOLUTION
              if(!research_stellar_set_composition(Sp->P[target].ResearchSourceState,Sp->P[target].Metallicity))
                Terminate("Prompt SH03 elemental composition update failed");
#endif

              mass_of_star = Sp->P[target].getMass();

              prob = Sp->P[target].getMass() / mass_of_star * (-expm1(-pall));
            }

          if(prob == 0)
            continue;

          if(prob < 0)
            Terminate("prob < 0");

          /* decide what process to consider (currently available: make a star or kick to wind) */
          const double gas_mass_before = Sp->P[target].getMass();
          double gas_velocity_before[3];
          for(int k = 0; k < 3; ++k) gas_velocity_before[k] = Sp->P[target].Vel[k];
          const int spawned_before = stars_spawned;
          p_decide = research_stellar_uniform(stellar_seed, Sp->P[target].ID.get(),
                     Sp->P[target].ResearchSourceState.stellar_random_counter - 1, 1);

          if(p_decide < p / pall) /* ok, a star formation is considered */
            make_star(Sp, target, prob, mass_of_star, &sum_mass_stars);

          if(Sp->SphP[target].Sfr > 0)
            {
              if(Sp->P[target].getType() == 0) /* to protect using a particle that has been turned into a star */
                {
                  Sp->SphP[target].Metallicity += (1 - w) * METAL_YIELD * (-expm1(-p));
                  Sp->SphP[target].MassMetallicity = Sp->SphP[target].Metallicity * Sp->P[target].getMass();
                }
            }
          Sp->P[target].Metallicity = Sp->SphP[target].Metallicity;
#ifdef STELLAR_EVOLUTION
          if(Sp->P[target].getType()==0 && !research_stellar_set_composition(Sp->P[target].ResearchSourceState,Sp->P[target].Metallicity))
            Terminate("Remaining gas prompt elemental composition update failed");
#endif
          int child = -1;
          if(Sp->P[target].getType() == STAR_TYPE) child = target;
          else if(stars_spawned > spawned_before) child = Sp->NumPart + spawned_before;
          if(child >= 0)
            {
              const research_source_state &s = Sp->P[child].ResearchSourceState;
              research_stellar_birth_event e = {};
              e.particle_index = child; e.parent_id = s.stellar_parent_gas_id;
              e.mode = s.stellar_birth_mode; e.opportunity_counter = s.stellar_random_counter;
              e.birth_time = s.stellar_birth_time; e.scale_factor = s.stellar_birth_scale_factor;
              e.metallicity = s.stellar_birth_metallicity; e.gross_mass = s.stellar_initial_mass;
              e.locked_mass = s.stellar_initial_locked_mass; e.prompt_mass = s.stellar_prompt_returned_mass;
              e.gas_mass_before = gas_mass_before;
              e.gas_mass_after = e.mode == 1 ? 0 : Sp->P[target].getMass();
              for(int k = 0; k < 3; ++k)
                {
                  e.gas_velocity_before[k] = gas_velocity_before[k];
                  e.gas_velocity_after[k] = e.mode == 1 ? 0 : Sp->P[target].Vel[k];
                  e.ssp_velocity[k] = Sp->P[child].Vel[k];
                }
              birth_events.push_back(e);
            }
        }
    } /* end of main loop over active gas particles */

  /* The upstream creation allocator uses global int counts. Reject overflow
   * before its MPI_INT reduction; an event journal does not remove that limit. */
  uint64_t local_births = birth_events.size(), total_births;
  MPI_Allreduce(&local_births, &total_births, 1, MPI_UINT64_T, MPI_SUM, Communicator);
  if(total_births > INT_MAX) Terminate("Native per-step stellar creation exceeds the int allocator domain");
  MPI_Allreduce(&stars_spawned, &tot_stars_spawned, 1, MPI_INT, MPI_SUM, Communicator);
  MPI_Allreduce(&stars_converted, &tot_stars_converted, 1, MPI_INT, MPI_SUM, Communicator);

  if(tot_stars_spawned > 0 || tot_stars_converted > 0)
    {
      mpi_printf("SFR: spawned %d stars, converted %d gas particles into stars\n", tot_stars_spawned, tot_stars_converted);
    }

  tot_altogether_spawned = tot_stars_spawned;
  altogether_spawned     = stars_spawned;
  if(tot_altogether_spawned)
    {
      /* need to assign new unique IDs to the spawned stars */

      if(All.MaxID == 0) /* MaxID not calculated yet */
        {
          /* determine maximum ID */
          MyIDType maxid = 0;
          for(int i = 0; i < Sp->NumPart; i++)
            if(Sp->P[i].ID.get() > maxid)
              {
                maxid = Sp->P[i].ID.get();
              }

          MyIDType *tmp = (MyIDType *)Mem.mymalloc("tmp", NTask * sizeof(MyIDType));

          MPI_Allgather(&maxid, sizeof(MyIDType), MPI_BYTE, tmp, sizeof(MyIDType), MPI_BYTE, Communicator);

          for(int i = 0; i < NTask; i++)
            if(tmp[i] > maxid)
              maxid = tmp[i];

          All.MaxID = maxid;

          Mem.myfree(tmp);
        }

      int *list = (int *)Mem.mymalloc("list", NTask * sizeof(int));

      MPI_Allgather(&altogether_spawned, 1, MPI_INT, list, 1, MPI_INT, Communicator);

      if(All.MaxID > ID_MAX || uint64_t(tot_altogether_spawned) > uint64_t(ID_MAX - All.MaxID))
        Terminate("Stellar spawn would exhaust the native particle ID domain");
      MyIDType newid = All.MaxID + 1;

      for(int i = 0; i < ThisTask; i++)
        newid += list[i];

      Mem.myfree(list);

      for(int i = 0; i < altogether_spawned; i++)
        Sp->P[Sp->NumPart + i].ID.set(newid++);

      All.MaxID += tot_altogether_spawned;
    }

  /* Spawn IDs are final only after the allocator above. Publish direct source
   * measurements before snapshots or checkpoints can expose advanced counters. */
  for(auto &e : birth_events) e.ssp_id = Sp->P[e.particle_index].ID.get();
  if(!research_stellar_publish_journal(All.OutputDir, birth_events, All.StellarBirthCommitCount,
      All.StellarBirthEventCount, (long long)All.Ti_Current, All.Time, All.ComovingIntegrationOn,
      All.UnitMass_in_g, All.UnitTime_in_s, All.UnitVelocity_in_cm_per_s, All.UnitLength_in_cm,
      All.HubbleParam, All.FactorSN, stellar_seed, research_restart_fingerprint, Communicator))
    Terminate("Native SSP birth journal publication failed or conflicts with existing history; use an explicit new branch directory");

  /* Note: New tree construction can be avoided because of  `force_add_star_to_tree()' */
  if(tot_stars_spawned > 0 || tot_stars_converted > 0)
    {
      Sp->TotNumPart += tot_stars_spawned;
      Sp->TotNumGas -= tot_stars_converted;
      Sp->NumPart += stars_spawned;
    }

  double sfrrate = 0;
  for(int bin = 0; bin < TIMEBINS; bin++)
    if(Sp->TimeBinsHydro.TimeBinCount[bin])
      sfrrate += Sp->TimeBinSfr[bin];

  MPI_Allreduce(&sfrrate, &totsfrrate, 1, MPI_DOUBLE, MPI_SUM, Communicator);

  MPI_Reduce(&sum_sm, &total_sm, 1, MPI_DOUBLE, MPI_SUM, 0, Communicator);
  MPI_Reduce(&sum_mass_stars, &total_sum_mass_stars, 1, MPI_DOUBLE, MPI_SUM, 0, Communicator);
  if(ThisTask == 0)
    {
      if(All.TimeStep > 0)
        rate = total_sm / (All.TimeStep / All.cf_atime_hubble_a);
      else
        rate = 0;

      /* compute the cumulative mass of stars */
      cum_mass_stars += total_sum_mass_stars;

      /* convert to solar masses per yr */
      rate_in_msunperyear = rate * (All.UnitMass_in_g / SOLAR_MASS) / (All.UnitTime_in_s / SEC_PER_YEAR);

      fprintf(Logs.FdSfr, "%14e %14e %14e %14e %14e %14e\n", All.Time, total_sm, totsfrrate, rate_in_msunperyear, total_sum_mass_stars,
              cum_mass_stars);
      myflush(Logs.FdSfr);
    }

  TIMER_STOP(CPU_COOLING_SFR);
}

/** \brief Convert a SPH particle into a star.
 *
 *  This function convertss an active star-forming gas particle into a star.
 *  The particle information of the gas is copied to the
 *  location istar and the fields necessary for the creation of the star
 *  particle are initialized.
 *
 *  \param i index of the gas particle to be converted
 *  \param birthtime time of birth (in code units) of the stellar particle
 */
void coolsfr::convert_sph_particle_into_star(simparticles *Sp, int i, double birthtime)
{
  if(!research_stellar_record_birth(Sp->P[i].ResearchSourceState, Sp->P[i].ID.get(),
       Sp->P[i].getMass(), birthtime, All.ComovingIntegrationOn ? birthtime : 0,
       Sp->P[i].Metallicity, All.FactorSN, 1))
    Terminate("Invalid or repeated SSP conversion birth state");
  Sp->P[i].setType(STAR_TYPE);
#if NSOFTCLASSES > 1
  Sp->P[i].setSofteningClass(All.SofteningClassOfPartType[Sp->P[i].getType()]);
#endif
#ifdef INDIVIDUAL_GRAVITY_SOFTENING
  if(((1 << Sp->P[i].getType()) & (INDIVIDUAL_GRAVITY_SOFTENING)))
    Sp->P[i].setSofteningClass(Sp->get_softening_type_from_mass(Sp->P[i].getMass()));
#endif

  Sp->TimeBinSfr[Sp->P[i].getTimeBinHydro()] -= Sp->SphP[i].Sfr;

  Sp->P[i].StellarAge = birthtime;

  return;
}

/** \brief Spawn a star particle from a SPH gas particle.
 *
 *  This function spawns a star particle from an active star-forming
 *  SPH gas particle. The particle information of the gas is copied to the
 *  location istar and the fields necessary for the creation of the star
 *  particle are initialized. The total mass of the gas particle is split
 *  between the newly spawned star and the gas particle.
 *  (This function is probably unecessary)
 *
 *  \param igas index of the gas cell from which the star is spawned
 *  \param birthtime time of birth (in code units) of the stellar particle
 *  \param istar index of the spawned stellar particle
 *  \param mass_of_star the mass of the spawned stellar particle
 */
void coolsfr::spawn_star_from_sph_particle(simparticles *Sp, int igas, double birthtime, int istar, MyDouble mass_of_star)
{
  Sp->P[istar] = Sp->P[igas];
  Sp->P[istar].setType(STAR_TYPE);
#if NSOFTCLASSES > 1
  Sp->P[istar].setSofteningClass(All.SofteningClassOfPartType[Sp->P[istar].getType()]);
#endif
#ifdef INDIVIDUAL_GRAVITY_SOFTENING
  if(((1 << Sp->P[istar].getType()) & (INDIVIDUAL_GRAVITY_SOFTENING)))
    Sp->P[istar].setSofteningClass(Sp->get_softening_type_from_mass(Sp->P[istar].getMass()));
#endif

  Sp->TimeBinsGravity.ActiveParticleList[Sp->TimeBinsGravity.NActiveParticles++] = istar;

  Sp->TimeBinsGravity.timebin_add_particle(istar, igas, Sp->P[istar].TimeBinGrav, Sp->TimeBinSynchronized[Sp->P[istar].TimeBinGrav]);

  Sp->P[istar].setMass(mass_of_star);
  if(!research_stellar_record_birth(Sp->P[istar].ResearchSourceState, Sp->P[igas].ID.get(),
       mass_of_star, birthtime, All.ComovingIntegrationOn ? birthtime : 0,
       Sp->P[igas].Metallicity, All.FactorSN, 2))
    Terminate("Invalid or repeated SSP spawned birth state");

  Sp->P[istar].StellarAge = birthtime;

  /* now change the conserved quantities in the cell in proportion */
  double fac = (Sp->P[igas].getMass() - Sp->P[istar].getMass()) / Sp->P[igas].getMass();

  Sp->P[igas].setMass(fac * Sp->P[igas].getMass());

  return;
}

/** \brief Make a star particle from a SPH gas particle.
 *
 *  Given a gas cell where star formation is active and the probability
 *  of forming a star, this function selectes either to convert the gas
 *  particle into a star particle or to spawn a star depending on the
 *  target mass for the star.
 *
 *  \param i index of the gas cell
 *  \param prob probability of making a star
 *  \param mass_of_star desired mass of the star particle
 *  \param sum_mass_stars holds the mass of all the stars created at the current time-step (for the local task)
 */
void coolsfr::make_star(simparticles *Sp, int i, double prob, MyDouble mass_of_star, double *sum_mass_stars)
{
  if(mass_of_star > Sp->P[i].getMass())
    Terminate("mass_of_star > P[i].Mass");

  uint64_t stellar_seed;
  if(!research_stellar_seed(All.StellarRandomSeed, stellar_seed) ||
      Sp->P[i].ResearchSourceState.stellar_random_counter == 0 ||
      !isfinite(prob) || prob < 0 || prob > 1 || mass_of_star <= 0)
    Terminate("Invalid SSP draw/formation probability");
  if(research_stellar_uniform(stellar_seed, Sp->P[i].ID.get(),
       Sp->P[i].ResearchSourceState.stellar_random_counter - 1, 2) < prob)
    {
      if(mass_of_star == Sp->P[i].getMass())
        {
          /* here we turn the gas particle itself into a star particle */
          stars_converted++;

          *sum_mass_stars += Sp->P[i].getMass();

          convert_sph_particle_into_star(Sp, i, All.Time);
        }
      else
        {
          /* in this case we spawn a new star particle, only reducing the mass in the cell by mass_of_star */
          altogether_spawned = stars_spawned;
          if(Sp->NumPart + altogether_spawned >= Sp->MaxPart)
            Terminate("NumPart=%d spwawn %d particles no space left (Sp.MaxPart=%d)\n", Sp->NumPart, altogether_spawned, Sp->MaxPart);

          int j = Sp->NumPart + altogether_spawned; /* index of new star */

          spawn_star_from_sph_particle(Sp, i, All.Time, j, mass_of_star);

          *sum_mass_stars += mass_of_star;
          stars_spawned++;
        }
    }
}

#endif /* closes SFR */
