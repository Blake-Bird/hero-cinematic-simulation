/**
 * @file
 *
 * This file is part of the N-body/SPH code OpenGadget3.
 *
 * @brief Header file to define particle data types that are used by the scattering class.
 *
 * @author OpenGadget3-development team, in particular Moritz S. Fischer.
 *
 * @copyright Copyright (c) 2026, OpenGadget3-development team lead by Klaus Dolag and Stefano Borgani
 * (see change logs). All rights reserved.
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License v3 or later.
 * See <https://www.gnu.org/licenses/>.
 */

#ifndef MSIDM_SCATTERING_DATA_HPP
#define MSIDM_SCATTERING_DATA_HPP

#include "../CodeBase/precision.hpp"

//////////////////////////////////////////////////////////////////////////////////////////////////

namespace scatter::pxp_scattering
{
/**
 * @brief Class to define data type for scattering.
 */
class scattering_particle
{
  public:
    scattering_particle(){};
    scattering_particle(MyLongDouble (&Pos_in)[3], MyFloat (&Vel_in)[3], MyFloat &Hsml_in, MyFloat &Mass_in,
                        short int &TimeBin_in, integertime &Ti_begstep_in
#ifdef SCATTERING_VMAX
                        ,
                        MyAtLeastDouble &v_max_in
#endif // SCATTERING_VMAX
#ifdef SCATTERING_HMIN
                        ,
                        MyFloat &h_min_in
#endif // SCATTERING_HMIN
#ifdef SCATTERING_ID
                        ,
                        MyIDType &ID_in
#endif // SCATTERING_ID
#ifdef TYPE_FOR_DATASTRUCT
                        ,
                        short int &Type_in
#endif // TYPE_FOR_DATASTRUCT
#ifdef SCATTERING_COUNT_INDI
                        ,
                        int &scatterCount_in
#endif // SCATTERING_COUNT_INDI
                        )
        : Pos{Pos_in[0], Pos_in[1], Pos_in[2]}, Vel{Vel_in[0], Vel_in[1], Vel_in[2]}, Hsml(Hsml_in), Mass(Mass_in),
          TimeBin(TimeBin_in), Ti_begstep(Ti_begstep_in)
#ifdef SCATTERING_VMAX
          ,
          v_max(v_max_in)
#endif // SCATTERING_VMAX
#ifdef SCATTERING_HMIN
          ,
          h_min(h_min_in)
#endif // SCATTERING_HMIN
#ifdef SCATTERING_ID
          ,
          ID(ID_in)
#endif // SCATTERING_ID
#ifdef TYPE_FOR_DATASTRUCT
          ,
          Type(Type_in)
#endif // TYPE_FOR_DATASTRUCT
#ifdef SCATTERING_COUNT_INDI
          ,
          scatterCount(scatterCount_in)
#endif // SCATTERING_COUNT_INDI
              {};

    MyLongDouble Pos[3];
    MyFloat Vel[3];
    MyFloat Hsml;
    MyFloat Mass;
    short int TimeBin;
    integertime Ti_begstep;

#ifdef SCATTERING_VMAX
    MyAtLeastDouble v_max;
#endif // SCATTERING_VMAX

#ifdef SCATTERING_HMIN
    MyFloat h_min;
#endif // SCATTERING_HMIN

#ifdef SCATTERING_ID
    MyIDType ID;
#endif // SCATTERING_ID

#ifdef TYPE_FOR_DATASTRUCT
    short int Type;
#endif // TYPE_FOR_DATASTRUCT

#ifdef SCATTERING_COUNT_INDI
    int scatterCount;
#endif // SCATTERING_COUNT_INDI
};
} // namespace scatter::pxp_scattering

//////////////////////////////////////////////////////////////////////////////////////////////////

namespace scatter::pxp_scattering
{
/**
 * @brief Class to define reference-based data type for scattering.
 */
class scattering_particle_ref
{
  public:
    scattering_particle_ref(scattering_particle &p_in)
        : Pos(p_in.Pos), Vel(p_in.Vel), Hsml(p_in.Hsml), Mass(p_in.Mass), TimeBin(p_in.TimeBin),
          Ti_begstep(p_in.Ti_begstep)
#ifdef SCATTERING_VMAX
          ,
          v_max(p_in.v_max)
#endif // SCATTERING_VMAX
#ifdef SCATTERING_HMIN
          ,
          h_min(p_in.h_min)
#endif // SCATTERING_HMIN
#ifdef SCATTERING_ID
          ,
          ID(p_in.ID)
#endif // SCATTERING_ID
#ifdef TYPE_FOR_DATASTRUCT
          ,
          Type(p_in.Type)
#endif // TYPE_FOR_DATASTRUCT
#ifdef SCATTERING_COUNT_INDI
          ,
          scatterCount(p_in.scatterCount)
#endif // SCATTERING_COUNT_INDI
              {};

    scattering_particle_ref(MyLongDouble (&Pos_in)[3], MyFloat (&Vel_in)[3], MyFloat &Hsml_in, MyFloat &Mass_in,
                            short int &TimeBin_in, integertime &Ti_begstep_in
#ifdef SCATTERING_VMAX
                            ,
                            MyAtLeastDouble &v_max_in
#endif // SCATTERING_VMAX
#ifdef SCATTERING_HMIN
                            ,
                            MyFloat &h_min_in
#endif // SCATTERING_HMIN
#ifdef SCATTERING_ID
                            ,
                            MyIDType &ID_in
#endif // SCATTERING_ID
#ifdef TYPE_FOR_DATASTRUCT
                            ,
                            short int &Type_in
#endif // TYPE_FOR_DATASTRUCT
#ifdef SCATTERING_COUNT_INDI
                            ,
                            int &scatterCount_in
#endif // SCATTERING_COUNT_INDI
                            )
        : Pos(Pos_in), Vel(Vel_in), Hsml(Hsml_in), Mass(Mass_in), TimeBin(TimeBin_in), Ti_begstep(Ti_begstep_in)
#ifdef SCATTERING_VMAX
          ,
          v_max(v_max_in)
#endif // SCATTERING_VMAX
#ifdef SCATTERING_HMIN
          ,
          h_min(h_min_in)
#endif // SCATTERING_HMIN
#ifdef SCATTERING_ID
          ,
          ID(ID_in)
#endif // SCATTERING_ID
#ifdef TYPE_FOR_DATASTRUCT
          ,
          Type(Type_in)
#endif // TYPE_FOR_DATASTRUCT
#ifdef SCATTERING_COUNT_INDI
          ,
          scatterCount(scatterCount_in)
#endif // SCATTERING_COUNT_INDI
              {};

    MyLongDouble (&Pos)[3];
    MyFloat (&Vel)[3];
    MyFloat &Hsml;
    MyFloat &Mass;
    short int &TimeBin;
    integertime &Ti_begstep;

#ifdef SCATTERING_VMAX
    MyAtLeastDouble &v_max;
#endif // SCATTERING_VMAX

#ifdef SCATTERING_HMIN
    MyFloat &h_min;
#endif // SCATTERING_HMIN

#ifdef SCATTERING_ID
    MyIDType &ID;
#endif // SCATTERING_ID

#ifdef TYPE_FOR_DATASTRUCT
    short int &Type;
#endif // TYPE_FOR_DATASTRUCT

#ifdef SCATTERING_COUNT_INDI
    int &scatterCount;
#endif // SCATTERING_COUNT_INDI
};
} // namespace scatter::pxp_scattering

//////////////////////////////////////////////////////////////////////////////////////////////////

#endif /* MSIDM_SCATTERING_DATA_HPP */
