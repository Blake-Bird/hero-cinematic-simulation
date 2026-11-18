/**
 * @file
 *
 * This file is part of the N-body/SPH code OpenGadget3.
 *
 * @brief Declarations for the dark matter kernel size determination.
 *
 * @author OpenGadget3-development team.
 *
 * @copyright Copyright (c) 2026, Klaus Dolag, and all contributing authors
 * (see change logs). All rights reserved.
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License v3 or later.
 * See <https://www.gnu.org/licenses/>.
 */

#ifndef MSIDM_FIND_HSML_PT1_HPP
#define MSIDM_FIND_HSML_PT1_HPP

//////////////////////////////////////////////////////////////////////////////////////////////////

namespace scatter
{
/**
 * @brief Criterion to find active dark matter particles.
 */
inline int pt1_hsml_isactive(short int &TimeBin, short int &Type)
{
    if ((TimeBin < 0) || (Type != type_dm))
    {
        return 0;
    }
    else
    {
        return 1;
    }
};
} // namespace scatter

//////////////////////////////////////////////////////////////////////////////////////////////////

namespace scatter
{
/**
 * @brief Data structure for kernel size determination.
 */
struct hsmldata_in
{
    MyLongDouble Pos[3];
    MyFloat Hsml;
    int NodeList[NODELISTLENGTH_default];
#ifdef TYPE_FOR_DATASTRUCT
    short int Type;
#endif // TYPE_FOR_DATASTRUCT
};

extern hsmldata_in *HsmlDataIn, *HsmlDataGet;
} // namespace scatter

//////////////////////////////////////////////////////////////////////////////////////////////////

namespace scatter
{
/**
 * @brief Reference data structure for kernel size determination.
 */
class hsmldata_in_ref
{
  public:
    hsmldata_in_ref(struct hsmldata_in &p_in);
    hsmldata_in_ref(struct particle_data &p_in);

    MyLongDouble (&Pos)[3];
    MyFloat &Hsml;
#ifdef TYPE_FOR_DATASTRUCT
    short int &Type;
#endif // TYPE_FOR_DATASTRUCT
};
} // namespace scatter

//////////////////////////////////////////////////////////////////////////////////////////////////

namespace scatter
{
/**
 * @brief Data structure for kernel size determination.
 */
struct hsmldata_out
{
    MyLongDouble Rho;
    int TrueNGB;
};

extern hsmldata_out *HsmlDataResult, *HsmlDataOut;
} // namespace scatter

//////////////////////////////////////////////////////////////////////////////////////////////////

#endif /* MSIDM_FIND_HSML_PT1_HPP */
