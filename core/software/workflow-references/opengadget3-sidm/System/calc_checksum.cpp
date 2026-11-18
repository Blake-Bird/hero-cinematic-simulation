/**
 * @file
 *
 * This file is part of the N-body/SPH code OpenGadget3.
 *
 * @brief
 *
 * @author OpenGadget3-development team, in particular Klaus Dolag and Volker Springel.
 *
 * @copyright Copyright (c) 2026, OpenGadget3-development team lead by Klaus Dolag and Stefano Borgani
 * (see change logs). All rights reserved.
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License v3 or later.
 * See <https://www.gnu.org/licenses/>.
 */

#include "../CodeBase/allvars.hpp"
#include "../CodeBase/proto.hpp"

#ifdef CHECKSUM_DEBUG
#include "Md5.hpp"
/**
 * @brief Calculate the checksum of a memory region using the MD5 algorithm.
 *
 * This function calculates the checksum of a memory region using the MD5 algorithm.
 * It computes the checksum for the specified memory region and then performs a global
 * reduction operation to obtain the checksum across all MPI processes.
 *
 * @param base Pointer to the start of the memory region.
 * @param bytes Size of the memory region in bytes.
 */
void calc_memory_checksum(void *base, size_t bytes)
{
    MD5_CTX sum;
    union {
        unsigned char digest[16];
        int val[4];
    } u, uglob;

    MD5Init(&sum);
    MD5Update(&sum, base, bytes);
    MD5Final(&sum);

    for (int i = 0; i < 16; i++)
    {
        u.digest[i] = sum.digest[i];
    }
    MPI_Allreduce(u.val, uglob.val, 4, MPI_INT, MPI_SUM, MYMPI_COMM_WORLD);
    if (ThisTask == 0)
    {
        VERBOSE(4, "Step=%d  MD5=", All.NumCurrentTiStep);
        for (int i = 0; i < 16; i++)
        {
            VERBOSE(4, "%02x", uglob.digest[i]);
        }
    }
}
#endif // CHECKSUM_DEBUG
