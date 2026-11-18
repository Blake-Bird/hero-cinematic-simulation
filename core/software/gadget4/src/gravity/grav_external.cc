/*******************************************************************************
 * \copyright   This file is part of the GADGET4 N-body/SPH code developed
 * \copyright   by Volker Springel. Copyright (C) 2014-2020 by Volker Springel
 * \copyright   (vspringel@mpa-garching.mpg.de) and all contributing authors.
 *******************************************************************************/

/*! \file gravity_external.cc
 *
 * \brief can add an optional external gravity field
 */

#include "gadgetconfig.h"

#ifdef EXTERNALGRAVITY

#include <math.h>
#include <mpi.h>
#include <stdlib.h>
#include <string.h>

#include "../data/allvars.h"
#include "../data/dtypes.h"
#include "../data/intposconvert.h"
#include "../data/mymalloc.h"
#include "../domain/domain.h"
#include "../fmm/fmm.h"
#include "../gravtree/gravtree.h"
#include "../logs/logs.h"
#include "../logs/timer.h"
#include "../main/simulation.h"
#include "../mpi_utils/mpi_utils.h"
#include "../pm/pm.h"
#include "../system/system.h"
#include "../time_integration/timestep.h"

void sim::gravity_external(void)
{
#ifdef PERIODIC
  // pick middle of (stretched) box, but could also choose other point
  vector<double> pos_center{0.5 * All.BoxSize / LONG_X, 0.5 * All.BoxSize / LONG_Y, 0.5 * All.BoxSize / LONG_Z};
#else
  // here pick origin
  vector<double> pos_center{0, 0, 0};
#endif
  MyIntPosType intpos_center[3];
  Sp.pos_to_intpos(pos_center.da, intpos_center);

  for(int i = 0; i < Sp.TimeBinsGravity.NActiveParticles; i++)
    {
      int target = Sp.TimeBinsGravity.ActiveParticleList[i];

#if defined(EVALPOTENTIAL) || defined(OUTPUT_POTENTIAL)
      Sp.P[target].ExtPotential = 0;
#endif

#ifdef EXTERNALGRAVITY_CONTROL
      {
        // Dedicated isolated controls: GM is supplied in the existing native
        // length/velocity units. No new production force or cosmology route.
        if(All.ComovingIntegrationOn || All.ControlPotential < 1 || All.ControlPotential > 4 ||
           !(All.ControlGM > 0) || !(All.ControlScale > 0) ||
           !(All.ControlAxisY > 0) || !(All.ControlAxisZ > 0))
          Terminate("Invalid isolated analytic control parameters");
        vector<double> pos;
        Sp.nearest_image_intpos_to_pos(Sp.P[target].IntPos, intpos_center, pos.da);
        double r = sqrt(pos.r2()), b = All.ControlScale, gm = All.ControlGM;
        double factor = 0, phi = 0;
        if(All.ControlPotential == 1) // Plummer
          {
            double d = sqrt(r*r + b*b);
            factor = -gm/(d*d*d); phi = -gm/d;
          }
        else if(All.ControlPotential == 2) // Hernquist, central cusp is excluded by fixtures
          {
            if(r == 0) Terminate("Hernquist control cannot evaluate the cusp");
            factor = -gm/(r*(r+b)*(r+b)); phi = -gm/(r+b);
          }
        else if(All.ControlPotential == 3) // untruncated NFW; GM=4*pi*G*rho_s*r_s^3
          {
            if(r == 0) Terminate("NFW control cannot evaluate the cusp");
            double u = r/b;
            double enclosed = log1p(u)-u/(1+u);
            // Avoid cancellation for small radii.
            if(u < 1.e-4) enclosed = u*u*(0.5-u*(2./3.-u*(0.75-0.8*u)));
            factor = -gm*enclosed/(r*r*r); phi = -gm*log1p(u)/r;
          }
        else // separable anisotropic harmonic: gm/b^3 = omega_x^2
          {
            double w2 = gm/(b*b*b);
            double axes[3] = {1, All.ControlAxisY, All.ControlAxisZ};
            for(int k=0;k<3;k++)
              {
                Sp.P[target].GravAccel[k] -= w2*axes[k]*axes[k]*pos[k];
                phi += 0.5*w2*axes[k]*axes[k]*pos[k]*pos[k];
              }
          }
        if(All.ControlPotential != 4) Sp.P[target].GravAccel += factor*pos;
#if defined(EVALPOTENTIAL) || defined(OUTPUT_POTENTIAL)
        Sp.P[target].ExtPotential += phi;
#endif
      }
#endif

#ifdef EXTERNALGRAVITY_STATICHQ
      {
        vector<double> pos;
        Sp.nearest_image_intpos_to_pos(Sp.P[target].IntPos, intpos_center, pos.da);

        double r = sqrt(pos.r2());

        double m = All.Mass_StaticHQHalo * pow(r / (r + All.A_StaticHQHalo), 2);

        if(r > 0)
          Sp.P[target].GravAccel += (-All.G * m / (r * r * r)) * pos;

#if defined(EVALPOTENTIAL) || defined(OUTPUT_POTENTIAL)
        Sp.P[target].ExtPotential += (-All.G * All.Mass_StaticHQHalo / (r + All.A_StaticHQHalo));
#endif
      }
#endif
    }
}

#endif
