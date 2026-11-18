/**
 * @file
 *
 * This file is part of the N-body/SPH code OpenGadget3.
 *
 * @brief Header file for the class template for pairwise interactions.
 *
 * The template specialization for the physics modules is done at the end of
 * this file.
 *
 * @author OpenGadget3-development team, in particular Moritz S. Fischer.
 *
 * @copyright Copyright (c) 2026, OpenGadget3-development team lead by Klaus Dolag and Stefano Borgani
 * (see change logs). All rights reserved.
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License v3 or later.
 * See <https://www.gnu.org/licenses/>.
 */

#ifndef MSIDM_EXECUTE_PXP_HPP
#define MSIDM_EXECUTE_PXP_HPP

#include "../CodeBase/precision.hpp"
#include <stdio.h>

#ifdef mSIDM
#include "msidm_base.hpp"
#endif // MSIDM


#include "parallel.hpp"

#ifdef PXP
#ifndef NODELISTLENGTH_default
#error You need to define NODELISTLENGTH_default
#endif // NODELISTLENGTH_default
#ifndef NODELISTLENGTH_PXP
#error You need to define NODELISTLENGTH_PXP
#endif // NODELISTLENGTH_PXP
#endif // PXP

//////////////////////////////////////////////////////////////////////////////////////////////////

namespace scatter::pxp_execute
{
/**
 * @brief Declaration of main class to execute pair-wise interactions.
 *
 * This class provides the infrastructure to execute pair-wise interactions. The interaction
 * itself can be specified as part of a derived class by providing a definition for the virtual
 * method pxp_physics. This class is built on top of the greentree to find neighbouring
 * particles and contains a base class for the parallelisation. The specific details required
 * for a physics module are expressed in terms of a class, which is given as the template
 * parameter BASE. Such a BASE can be constructed with execute_pxp_base in execute_pxp_base.hpp.
 *
 */
template <typename BASE> class execute_pxp : virtual public BASE, public parallel<BASE>
{
  public:
    execute_pxp();

  private:
    typedef BASE B;
    typedef typename B::tDATA_IN DATA_IN;
    typedef typename B::tDATA_OUT DATA_OUT;
    typedef typename B::tDATA_IN_REF DATA_IN_REF;
    typedef parallel<BASE> PB;

    size_t HighMark_pxp;

  protected:
    /** This method does the actual physics and must be defined in the derived class */
    virtual void pxp_physics(DATA_IN_REF &pi, const int &pj) = 0;

  public:
    /** This method allocates Ngblist and uses greentree to execute evaluate(). It
        also does the MPI communication */
    void all();

  private:
    /** This method is processed in parallel and calls pxp_physics(), via primary_phase and
        secondary_phase, which takes two particles and computes the scattering */
    int evaluate(int target, int mode, const int *ngblist, int extended_numngb_inbox, int *i_startnode, int node,
                 int i_node, void *data) override;

    /** This method is called by evaluate and compute the local interactions */
    int primary_phase(int target, const int *ngblist, int extended_numngb_inbox, int *i_startnode, int node, int i_node,
                      void *data);

    /** This method is called by evaluate and compute the non-local interactions */
    int secondary_phase(int target, const int *ngblist, int extended_numngb_inbox, int *i_startnode, int node,
                        int i_node, void *data);
};
} // namespace scatter::pxp_execute

////////////////////////////////////////////////////////////////////////////////////////////////

#endif /* MSIDM_EXECUTE_PXP_HPP */
