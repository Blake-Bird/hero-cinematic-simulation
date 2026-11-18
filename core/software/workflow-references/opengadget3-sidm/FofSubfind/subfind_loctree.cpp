/**
 * @file
 *
 * This file is part of the N-body/SPH code OpenGadget3.
 *
 * @brief Build local tree structure in Subfind.
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
#include <time.h>

#include "../CodeBase/allvars.hpp"
#include "../CodeBase/proto.hpp"
#include "subfind.hpp"

#ifdef SUBFIND

#ifdef ACC
#include "../OpenACC/gpuallvars_branch.hpp"
#endif // ACC

static int last; /* auxialiary variable used to set-up non-recursive walk */
static double RootLen, RootCenter[3];

/**
 * @brief Determine the spatial extent of a set of particles.
 *
 * This function calculates the spatial extent of a given set of particles by determining the minimum
 * bounding box that encloses all the particles. It iterates over each particle in the set and updates
 * the minimum and maximum coordinates in each dimension to define the bounding box. The function then
 * computes the length of the bounding box along each dimension and sets the center of the bounding box
 * as the average of the minimum and maximum coordinates. Finally, it updates the global variables
 * `RootCenter` and `RootLen` to represent the center and size of the bounding box, respectively.
 *
 * @param npart The number of particles in the set.
 * @param mp An array of unbind_data structures representing the particles. If NULL, the function will terminate with an
 * error.
 */
void subfind_loctree_findExtent(int npart, struct unbind_data *mp)
{
    double xmin[3], xmax[3];

    /* determine extension */
    for (int i = 0; i < 3; i++)
    {
        xmin[i] = MAX_REAL_NUMBER;
        xmax[i] = -MAX_REAL_NUMBER;
    }

    int i = 0;
    for (int k = 0; k < npart; k++)
    {
        if (mp)
        {
            i = mp[k].index;
        }
        else
        {
            ENDRUN("mp is NULL");
        }

        for (int j = 0; j < 3; j++)
        {
            if (xmin[j] > P[i].Pos[j])
            {
                xmin[j] = P[i].Pos[j];
            }

            if (xmax[j] < P[i].Pos[j])
            {
                xmax[j] = P[i].Pos[j];
            }
        }
    }

    double len = 0;
    for (int j = 0; j < 3; j++)
    {
        if (xmax[j] - xmin[j] > len)
        {
            len = xmax[j] - xmin[j];
        }
    }

    len *= 1.001;

    for (int j = 0; j < 3; j++)
    {
        RootCenter[j] = 0.5 * (xmin[j] + xmax[j]);
    }

    RootLen = len;
}

/**
 * @brief Copy the spatial extent of the domain to the root of the local tree.
 *
 * This function copies the spatial extent of the domain to the root of the local tree by
 * updating the global variables `RootCenter` and `RootLen`. It sets the center of the root
 * bounding box to be the same as the center of the domain, and it sets the length of the
 * root bounding box to be the same as the length of the domain.
 */
void subfind_loctree_copyExtent(void)
{
    for (int j = 0; j < 3; j++)
    {
        RootCenter[j] = DomainCenter[j];
    }
    RootLen = DomainLen;
}

/**
 * @brief Build a local tree structure to efficiently search for particles.
 *
 * This function constructs a local tree structure to efficiently search for particles
 * in the given particle distribution. It uses a recursive algorithm to insert particles
 * into the tree nodes based on their spatial positions. If the maximum number of tree
 * nodes is reached, the function terminates with an error message.
 *
 * @param npart The number of particles to insert into the tree.
 * @param mp    An array of unbinding data structures containing the indices of the particles.
 *
 * @return The total number of nodes in the constructed tree.
 */
int subfind_loctree_treebuild(int npart, struct unbind_data *mp)
{
    int subnode = 0, parent = -1;
    struct NODE *nfreep;

    /* select first node */
    int nfree = All.MaxPart;
    nfreep = &Nodes[nfree];

    /* create an empty  root node  */
    nfreep->len = RootLen;
    for (int i = 0; i < 3; i++)
    {
        nfreep->center[i] = RootCenter[i];
    }

    for (int i = 0; i < 8; i++)
    {
        nfreep->u.suns[i] = -1;
    }

    int numnodes = 1;
    nfreep++;
    nfree++;

    /* insert all particles */
    int i = 0;
    for (int k = 0; k < npart; k++)
    {
        if (mp)
        {
            i = mp[k].index;
        }
        else
        {
            ENDRUN("mp is NULL");
        }

        int th = All.MaxPart;

        while (1)
        {
            if (th >= All.MaxPart) /* we are dealing with an internal node */
            {
                subnode = 0;
                if (P[i].Pos[0] > Nodes[th].center[0])
                {
                    subnode += 1;
                }
                if (P[i].Pos[1] > Nodes[th].center[1])
                {
                    subnode += 2;
                }
                if (P[i].Pos[2] > Nodes[th].center[2])
                {
                    subnode += 4;
                }

                int nn = Nodes[th].u.suns[subnode];

                if (nn >= 0) /* ok, something is in the daughter slot already, need to continue */
                {
                    parent = th; /* note: subnode can still be used in the next step of the walk */
                    th = nn;
                }
                else
                {
                    /* here we have found an empty slot where we can
                     * attach the new particle as a leaf
                     */
                    Nodes[th].u.suns[subnode] = i;
                    break; /* done for that particle */
                }
            }
            else
            {
                /* we try to insert into a leaf with a single particle
                 * need to generate a new internal node at that point
                 */
                Nodes[parent].u.suns[subnode] = nfree;

                nfreep->len = 0.5 * Nodes[parent].len;
                double lenhalf = 0.25 * Nodes[parent].len;

                if (subnode & 1)
                {
                    nfreep->center[0] = Nodes[parent].center[0] + lenhalf;
                }
                else
                {
                    nfreep->center[0] = Nodes[parent].center[0] - lenhalf;
                }

                if (subnode & 2)
                {
                    nfreep->center[1] = Nodes[parent].center[1] + lenhalf;
                }
                else
                {
                    nfreep->center[1] = Nodes[parent].center[1] - lenhalf;
                }

                if (subnode & 4)
                {
                    nfreep->center[2] = Nodes[parent].center[2] + lenhalf;
                }
                else
                {
                    nfreep->center[2] = Nodes[parent].center[2] - lenhalf;
                }

                nfreep->u.suns[0] = -1;
                nfreep->u.suns[1] = -1;
                nfreep->u.suns[2] = -1;
                nfreep->u.suns[3] = -1;
                nfreep->u.suns[4] = -1;
                nfreep->u.suns[5] = -1;
                nfreep->u.suns[6] = -1;
                nfreep->u.suns[7] = -1;

                subnode = 0;
                if (P[th].Pos[0] > nfreep->center[0])
                {
                    subnode += 1;
                }
                if (P[th].Pos[1] > nfreep->center[1])
                {
                    subnode += 2;
                }
                if (P[th].Pos[2] > nfreep->center[2])
                {
                    subnode += 4;
                }

                if (nfreep->len < 1.0e-3 * All.SofteningTable[P[i].Type])
                {
                    /* seems like we're dealing with particles
                     * at identical locations. randomize
                     * subnode index (well below gravitational softening scale).
                     */
                    subnode = (int)(8.0 * get_random_number(nfree));
                    if (subnode >= 8)
                    {
                        subnode = 7;
                    }
                }

                nfreep->u.suns[subnode] = th;

                th = nfree; /* resume trying to insert the new particle at
                       the newly created internal node */

                numnodes++;
                nfree++;
                nfreep++;

                if ((numnodes) >= MaxNodes)
                {
                    VERBOSE_ALL(0, "maximum number %d of tree-nodes reached.", MaxNodes);
                    VERBOSE_ALL(0, "for particle %d  %g %g %g", i, P[i].Pos[0], P[i].Pos[1], P[i].Pos[2]);
#ifdef KD_EXTRA_TREEALLOC_FACTOR_SUBFIND
                    VERBOSE_ALL(0, "Try to increase KD_EXTRA_TREEALLOC_FACTOR_SUBFIND ...");
#else // KD_EXTRA_TREEALLOC_FACTOR_SUBFIND
                    VERBOSE_ALL(0, "Try to use KD_EXTRA_TREEALLOC_FACTOR_SUBFIND=1.2 ...");
#endif // KD_EXTRA_TREEALLOC_FACTOR_SUBFIND
                    ENDRUN("maximum number of tree-nodes reached - see verbose statements for more info.");
                }
            }
        }
    }

    /* now compute the multipole moments recursively */
    last = -1;
    subfind_loctree_update_node_recursive(All.MaxPart, -1, -1);

    if (last >= All.MaxPart)
    {
        Nodes[last].u.d.nextnode = -1;
    }
    else
    {
        Nextnode[last] = -1;
    }

    /*
       fprintf(Logfile,"Have put %d particles into tree.\n", num);
     */

    return numnodes;
}

/**
 * @brief Recursively update node properties in the local tree structure.
 *
 * This routine computes the multipole moments for a given internal node and
 * all its subnodes using a recursive computation.  Note that the moments of
 * the daughter nodes are already stored in single precision. For very large
 * particle numbers, loss of precision may results for certain particle
 * distributions.
 *
 * @param no     The index of the current node in the tree structure.
 * @param sib    The index of the next sibling node in the same level of the tree.
 * @param father The index of the parent node in the tree.
 */
void subfind_loctree_update_node_recursive(int no, int sib, int father)
{
    int p, pp = 0, nextsib, suns[8];
    double s[3];

    if (no >= All.MaxPart)
    {
        for (int j = 0; j < 8; j++)
        {
            suns[j] = Nodes[no].u.suns[j]; /* that "backup" is necessary because the nextnode entry will
                              overwrite one element (union!) */
        }
        if (last >= 0)
        {
            if (last >= All.MaxPart)
            {
                Nodes[last].u.d.nextnode = no;
            }
            else
            {
                Nextnode[last] = no;
            }
        }

        last = no;

        double mass = 0;
        s[0] = 0;
        s[1] = 0;
        s[2] = 0;

        for (int j = 0; j < 8; j++)
        {
            if ((p = suns[j]) >= 0)
            {
                /* check if we have a sibling on the same level */
                int jj;
                for (jj = j + 1; jj < 8; jj++)
                {
                    if ((pp = suns[jj]) >= 0)
                    {
                        break;
                    }
                }

                if (jj < 8) /* yes, we do */
                {
                    nextsib = pp;
                }
                else
                {
                    nextsib = sib;
                }

                subfind_loctree_update_node_recursive(p, nextsib, no);

                if (p >= All.MaxPart) /* an internal node or pseudo particle */
                {
                    mass += Nodes[p].u.d.mass; /* we assume a fixed particle mass */
                    s[0] += Nodes[p].u.d.mass * Nodes[p].u.d.s[0];
                    s[1] += Nodes[p].u.d.mass * Nodes[p].u.d.s[1];
                    s[2] += Nodes[p].u.d.mass * Nodes[p].u.d.s[2];
                }
                else /* a particle */
                {
                    mass += P[p].Mass;
                    s[0] += P[p].Mass * P[p].Pos[0];
                    s[1] += P[p].Mass * P[p].Pos[1];
                    s[2] += P[p].Mass * P[p].Pos[2];
                }
            }
        }

        if (mass > 0)
        {
            s[0] /= mass;
            s[1] /= mass;
            s[2] /= mass;
        }
        else
        {
            s[0] = Nodes[no].center[0];
            s[1] = Nodes[no].center[1];
            s[2] = Nodes[no].center[2];
        }

        Nodes[no].u.d.s[0] = s[0];
        Nodes[no].u.d.s[1] = s[1];
        Nodes[no].u.d.s[2] = s[2];
        Nodes[no].u.d.mass = mass;

        Nodes[no].u.d.sibling = sib;
        Nodes[no].u.d.father = father;
    }
    else /* single particle or pseudo particle */
    {
        if (last >= 0)
        {
            if (last >= All.MaxPart)
            {
                Nodes[last].u.d.nextnode = no;
            }
            else
            {
                Nextnode[last] = no;
            }
        }

        last = no;
        if (no < All.MaxPart) /* only set it for single particles */
        {
            Father[no] = father;
        }
    }
}

/**
 * @brief Evaluate the gravitational potential at a given target position.
 *
 * This function calculates the gravitational potential at the position of the target particle
 * based on the local tree structure. It traverses the tree, computing the contribution to
 * the potential from nearby particles or nodes. The potential is computed using the gravitational
 * softening lengths of the particles or nodes, and the softening length can be adapted based on
 * the local density.
 *
 * @param target The index of the target particle for which to compute the potential.
 * @return The gravitational potential at the position of the target particle.
 */
double subfind_loctree_treeevaluate_potential(int target)
{
    struct NODE *nop = 0;
    double r2, dx, dy, dz, mass, r, u, wp;

#ifdef ADAPTGRAVSOFT
    double h_p, h_p_inv, u_p, wp_p, h_max;
    int particle;
#endif // ADAPTGRAVSOFT

    double pos_x = P[target].Pos[0];
    double pos_y = P[target].Pos[1];
    double pos_z = P[target].Pos[2];

#ifndef ADAPTGRAVSOFT
    double h = All.ForceSoftening[P[target].Type];
#else // ADAPTGRAVSOFT
    double h = P[target].AGS_Hsml;
#endif // ADAPTGRAVSOFT
    double h_inv = 1.0 / h;
    double pot = 0;
    int no = All.MaxPart;

    while (no >= 0)
    {
        if (no < All.MaxPart) /* single particle */
        {
            dx = P[no].Pos[0] - pos_x;
            dy = P[no].Pos[1] - pos_y;
            dz = P[no].Pos[2] - pos_z;
            mass = P[no].Mass;
        }
        else
        {
            nop = &Nodes[no];
            dx = nop->u.d.s[0] - pos_x;
            dy = nop->u.d.s[1] - pos_y;
            dz = nop->u.d.s[2] - pos_z;
            mass = nop->u.d.mass;
        }

#ifdef PERIODIC
        dx = NEAREST(dx);
        dy = NEAREST(dy);
        dz = NEAREST(dz);
#endif // PERIODIC
        r2 = dx * dx + dy * dy + dz * dz;

        if (no < All.MaxPart)
        {
#ifdef ADAPTGRAVSOFT
            h_p = P[no].AGS_Hsml;
            particle = no;
#endif // ADAPTGRAVSOFT
            no = Nextnode[no];
        }
        else /* we have an internal node. Need to check opening criterion */
        {
            if (nop->len * nop->len > r2 * All.ErrTolThetaSubfind * All.ErrTolThetaSubfind)
            {
                /* open cell */
                if (mass)
                {
                    no = nop->u.d.nextnode;
                    continue;
                }
            }
#ifdef ADAPTGRAVSOFT
            h_p = nop->maxsoft; /* softening of the node */
            h_max = h >= h_p ? h : h_p;

            if (r2 < h_max * h_max)
            {
                no = nop->u.d.nextnode;
                continue; /* discard cases where the particle-node separation is less than the bigger
                     between the target softening and the node softening
                   */
            }

            particle = no;
#endif // ADAPTGRAVSOFT
            no = nop->u.d.sibling; /* node can be used */
        }

        r = sqrt(r2);

#ifdef ADAPTGRAVSOFT
        h_max = h >= h_p ? h : h_p;
        if (r >= h_max)
        {
            pot -= mass / r;
        }
#else // ADAPTGRAVSOFT
        if (r >= h)
        {
            pot -= mass / r;
        }
#endif // ADAPTGRAVSOFT

        else
        {
#ifdef ADAPTGRAVSOFT
            u = r * h_inv;

            if (u > 1.)
            {
                wp = -1. / u;
            }
            else if (u < 0.5)
            {
                wp = -2.8 + u * u * (5.333333333333 + u * u * (6.4 * u - 9.6));
            }
            else
            {
                wp = -3.2 + 0.066666666667 / u +
                     u * u * (10.666666666667 + u * (-16.0 + u * (9.6 - 2.133333333333 * u)));
            }

            h_p_inv = 1. / h_p;
            u_p = r * h_p_inv;

            if (u_p > 1.)
            {
                wp_p = -1. / u_p;
            }
            else if (u_p < 0.5)
            {
                wp_p = -2.8 + u_p * u_p * (5.333333333333 + u_p * u_p * (6.4 * u_p - 9.6));
            }
            else
            {
                wp_p = -3.2 + 0.066666666667 / u_p +
                       u_p * u_p * (10.666666666667 + u_p * (-16.0 + u_p * (9.6 - 2.133333333333 * u_p)));
            }
            pot += (mass * h_inv * wp + mass * h_p_inv * wp_p) / 2.;
#else // ADAPTGRAVSOFT
            u = r * h_inv;

            if (u < 0.5)
            {
                wp = -2.8 + u * u * (5.333333333333 + u * u * (6.4 * u - 9.6));
            }
            else
            {
                wp = -3.2 + 0.066666666667 / u +
                     u * u * (10.666666666667 + u * (-16.0 + u * (9.6 - 2.133333333333 * u)));
            }

            pot += mass * h_inv * wp;
#endif // ADAPTGRAVSOFT
        }
    }
    return pot;
}

/**
 * @brief Compare function for sorting an array of r2data structures based on the r2 field.
 *
 * This function is used as a comparison function for sorting an array of r2data structures
 * based on the r2 field. It is typically used in conjunction with the qsort function to sort
 * the array in ascending order of r2 values.
 *
 * @param a Pointer to the first element to compare.
 * @param b Pointer to the second element to compare.
 * @return Negative value if the r2 value of @p a is less than the r2 value of @p b,
 *         positive value if the r2 value of @p a is greater than the r2 value of @p b,
 *         and zero if the r2 values of @p a and @p b are equal.
 */
int subfind_locngb_compare_key(const void *a, const void *b)
{
    if (((struct r2data *)a)->r2 < (((struct r2data *)b)->r2))
    {
        return -1;
    }

    if (((struct r2data *)a)->r2 > (((struct r2data *)b)->r2))
    {
        return +1;
    }

    return 0;
}

/**
 * @brief Find neighbors within a specified number density using a tree-based method.
 *
 * This function finds neighbors within a specified number density using a tree-based method.
 * It iteratively adjusts the smoothing length (hguess) until the desired number of neighbors
 * (desngb) is reached. The function then sorts the squared distances of the neighbors and
 * returns the total number of neighbors found.
 *
 * @param xyz The coordinates of the particle for which neighbors are being searched.
 * @param desngb The desired number of neighbors.
 * @param hguess The initial guess for the smoothing length. If set to 0, it is calculated
 *               based on the particle density and the desired number of neighbors.
 * @param part_index The index of the particle for which neighbors are being searched.
 * @return The total number of neighbors found.
 */
int subfind_locngb_treefind(MyLongDouble xyz[3], int desngb, double hguess, int part_index)
{
    int numngb, truenumngb;
    double part_dens;
    //  double h2max;

    if (hguess == 0)
    {
        part_dens = All.Omega0 * 3 * All.Hubble * All.Hubble / (8 * M_PI * All.GravityConstantInternal) / P[0].Mass;
        hguess = pow(3 * desngb / (4 * M_PI) / part_dens, 1.0 / 3);
    }

    do
    {
        truenumngb = subfind_locngb_treefind_variable(xyz, hguess, part_index, &numngb);

        if (truenumngb < desngb)
        {
            hguess *= 1.26;
            continue;
        }

        if (truenumngb >= desngb)
        {
            qsort(R2list, numngb, sizeof(struct r2data), subfind_locngb_compare_key);
            //      h2max = R2list[desngb - 1].r2;
            break;
        }

        hguess *= 1.26;
    } while (1);

    return numngb;
    //  return sqrt(h2max);
}

/**
 * @brief Find neighbors within a variable smoothing length using the tree.
 *
 * This function finds neighbors within a variable smoothing length using a tree-based method.
 * It iterates over particles and tree nodes, checking their distances from the search center
 * and accumulating neighbors based on the provided smoothing length (hguess). The number of
 * neighbors found is returned, and the actual number of neighbors satisfying the condition
 * r^2 < hguess^2 is stored in the numngb parameter.
 *
 * @param searchcenter The coordinates of the search center.
 * @param hguess The smoothing length used for neighbor search.
 * @param part_index The index of the particle for which neighbors are being searched.
 * @param numngb Pointer to an integer where the number of neighbors satisfying the condition
 *               r^2 < hguess^2 will be stored.
 * @return The total number of neighbors found, regardless of whether they satisfy the smoothing
 *         length condition.
 */
int subfind_locngb_treefind_variable(MyLongDouble searchcenter[3], double hguess, int part_index, int *numngb)
{
    double dx, dy, dz, r2;
    struct NODE *that;

    double h2 = hguess * hguess;

    *numngb = 0;
    int truenumngb = 0;
    int no = All.MaxPart;

    while (no >= 0)
    {
        if (no < All.MaxPart) /* single particle */
        {
            int p = no;
            no = Nextnode[no];

#ifdef PERIODIC
            dx = NEAREST(P[p].Pos[0] - searchcenter[0]);
            dy = NEAREST(P[p].Pos[1] - searchcenter[1]);
            dz = NEAREST(P[p].Pos[2] - searchcenter[2]);
#else // PERIODIC
            dx = P[p].Pos[0] - searchcenter[0];
            dy = P[p].Pos[1] - searchcenter[1];
            dz = P[p].Pos[2] - searchcenter[2];
#endif // PERIODIC
            if (dx < -hguess)
            {
                continue;
            }
            if (dx > hguess)
            {
                continue;
            }

            if (dy < -hguess)
            {
                continue;
            }
            if (dy > hguess)
            {
                continue;
            }

            if (dz < -hguess)
            {
                continue;
            }
            if (dz > hguess)
            {
                continue;
            }

            r2 = dx * dx + dy * dy + dz * dz;

            if (r2 < h2)
            {
                truenumngb++;
                if (P[p].u.DM_Density <= P[part_index].u.DM_Density)
                {
                    continue;
                }
                R2list[*numngb].r2 = r2;
                R2list[*numngb].index = p;
                (*numngb)++;
            }
        }
        else
        {
            that = &Nodes[no];

            no = Nodes[no].u.d.sibling; /* in case the node can be discarded */

#ifdef PERIODIC
            if ((NEAREST(that->center[0] - searchcenter[0]) + 0.5 * that->len) < -hguess)
            {
                continue;
            }
            if ((NEAREST(that->center[0] - searchcenter[0]) - 0.5 * that->len) > hguess)
            {
                continue;
            }
            if ((NEAREST(that->center[1] - searchcenter[1]) + 0.5 * that->len) < -hguess)
            {
                continue;
            }
            if ((NEAREST(that->center[1] - searchcenter[1]) - 0.5 * that->len) > hguess)
            {
                continue;
            }
            if ((NEAREST(that->center[2] - searchcenter[2]) + 0.5 * that->len) < -hguess)
            {
                continue;
            }
            if ((NEAREST(that->center[2] - searchcenter[2]) - 0.5 * that->len) > hguess)
            {
                continue;
            }
#else // PERIODIC
            if (((that->center[0] - searchcenter[0]) + 0.5 * that->len) < -hguess)
            {
                continue;
            }
            if (((that->center[0] - searchcenter[0]) - 0.5 * that->len) > hguess)
            {
                continue;
            }
            if (((that->center[1] - searchcenter[1]) + 0.5 * that->len) < -hguess)
            {
                continue;
            }
            if (((that->center[1] - searchcenter[1]) - 0.5 * that->len) > hguess)
            {
                continue;
            }
            if (((that->center[2] - searchcenter[2]) + 0.5 * that->len) < -hguess)
            {
                continue;
            }
            if (((that->center[2] - searchcenter[2]) - 0.5 * that->len) > hguess)
            {
                continue;
            }
#endif // PERIODIC

            no = that->u.d.nextnode; /* ok, we need to open the node */
        }
    }

    return truenumngb;
}

/**
 * @brief Allocate memory for the local tree structure.
 *
 * This function allocates memory for the local tree structure used in substructure finding.
 * It allocates memory for nodes, nextnode array, father array, and an array used for storing
 * squared distances (R2list). The function sets global variables such as MaxNodes and initializes
 * pointers to the allocated memory.
 *
 * @param maxnodes The maximum number of tree nodes to allocate.
 * @param maxpart The maximum number of particles.
 * @return The total number of bytes allocated for the local tree structure.
 */
size_t subfind_loctree_treeallocate(int maxnodes, int maxpart) /* usually maxnodes=0.7*maxpart is sufficient */
{
    size_t bytes, allbytes = 0;

    MaxNodes = maxnodes;

    Nodes_base = (struct NODE *)mymalloc("Nodes_base", bytes = (MaxNodes + 1) * sizeof(struct NODE));
    allbytes += bytes;
#ifndef AR_SUBFIND_ALLOC_TOPNODES
    Nextnode = (int *)mymalloc("Nextnode", bytes = (maxpart) * sizeof(int));
#else // AR_SUBFIND_ALLOC_TOPNODES
    Nextnode = (int *)mymalloc("Nextnode", bytes = (maxpart + NTopnodes) * sizeof(int));
#endif // AR_SUBFIND_ALLOC_TOPNODES
    allbytes += bytes;

    Father = (int *)mymalloc("Father", bytes = maxpart * sizeof(int));
    allbytes += bytes;

    R2list = (struct r2data *)mymalloc("R2list", bytes = maxpart * sizeof(struct r2data));
    allbytes += bytes;

    Nodes = Nodes_base - All.MaxPart;

#ifdef ACC
    acc_all.no_tree_flag = 1;
    acc_all.len_nextnode = (maxpart);
#ifdef AR_SUBFIND_ALLOC_TOPNODES
    acc_all.len_nextnode += NTopnodes;
#endif // AR_SUBFIND_ALLOC_TOPNODES
    acc_all.len_nodes_base = (MaxNodes + 1);
#endif // ACC

    return allbytes;
}

/**
 * @brief Free memory allocated for the local tree structure.
 *
 * This function frees the memory previously allocated for the local tree structure used in
 * substructure finding. It deallocates memory for nodes, nextnode array, father array, and
 * an array used for storing squared distances (R2list). It also resets global variables to
 * their initial values.
 */
void subfind_loctree_treefree(void)
{
    myfree(R2list);
    myfree(Father);
    myfree(Nextnode);
    myfree(Nodes_base);
#ifdef ACC_OPENMP
    // Nitya - removing the delete for Nextnode
    ACC_PRAGMA(ACC_EXIT(ACC_DELETE(Nodes_base [0:acc_all.len_nodes_base])));
#endif // ACC_OPENMP
}
#endif // SUBFIND
