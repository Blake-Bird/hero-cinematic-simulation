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

#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include "swap.hpp"

/**
 * @brief Reverse the byte order of a double precision floating-point number.
 *
 * This function reverses the byte order of the memory representation of the input double number.
 * It is commonly used to convert between little-endian and big-endian representations.
 *
 * @param Val The input double number whose byte order needs to be reversed.
 * @return The double number with its byte order reversed.
 */
double SwapDouble(double Val)
{
    double nVal;
    const char *readFrom = (const char *)&Val;
    char *writeTo = ((char *)&nVal) + sizeof(nVal);

    for (int i = 0; i < sizeof(Val); ++i)
    {
        *(--writeTo) = *(readFrom++);
    }
    return nVal;
}

/**
 * @brief Reverse the byte order of a single precision floating-point number.
 *
 * This function reverses the byte order of the memory representation of the input float number.
 *
 * @param Val The input float number whose byte order needs to be reversed.
 * @return The float number with its byte order reversed.
 */
float SwapFloat(float Val)
{
    float nVal;
    const char *readFrom = (const char *)&Val;
    char *writeTo = ((char *)&nVal) + sizeof(nVal);

    for (int i = 0; i < sizeof(Val); ++i)
    {
        *(--writeTo) = *(readFrom++);
    }
    return nVal;
}

/**
 * @brief Reverse the byte order of an integer number.
 *
 * This function reverses the byte order of the memory representation of the input integer number.
 * It is commonly used to convert between little-endian and big-endian representations.
 *
 * @param Val The input integer number whose byte order needs to be reversed.
 * @return The integer number with its byte order reversed.
 */
int SwapInt(int Val)
{
    int nVal;
    const char *readFrom = (const char *)&Val;
    char *writeTo = ((char *)&nVal) + sizeof(nVal);

    for (int i = 0; i < sizeof(Val); ++i)
    {
        *(--writeTo) = *(readFrom++);
    }
    return nVal;
}

/**
 * @brief Check the validity of a binary file by examining integer values within the file.
 *
 * This function checks if the binary file has been correctly written by examining integer values within the file.
 * It verifies if the file structure matches the expected pattern of alternating integer values representing block
 * sizes. If the file structure is incorrect, it attempts to fix it by swapping byte order of integers and rechecking.
 *
 * @param fname The name of the binary file to be checked.
 * @param swap A pointer to an integer variable to store the number of swaps performed to fix the file structure (if
 * necessary). If the file structure is correct, this value remains zero.
 * @return Returns 0 if the file structure is correct or fixed successfully, and -1 if there was an error opening the
 * file.
 */
int CheckSwap(char *fname, int *swap)
{
    FILE *fd;
    off_t fsize, fpos;
    int blocksize, blockend;

    if (!(fd = fopen(fname, "r")))
    {
        VERBOSE_ALL(0, "can't open file `%s'.\n", fname);
        return -1;
    }

    fseeko(fd, 0, SEEK_END);
    fsize = ftello(fd);

    *swap = 0;
    fpos = 0;
    fseeko(fd, 0, SEEK_SET);
    safe_fread(&blocksize, sizeof(int), 1, fd);
    while (!feof(fd))
    {
        if (fpos + blocksize + 4 > fsize)
        {
            *swap += 1;
            break;
        }
        fpos += 4 + blocksize;
        fseeko(fd, fpos, SEEK_SET);
        safe_fread(&blockend, sizeof(int), 1, fd);
        if (blocksize != blockend)
        {
            *swap += 1;
            break;
        }
        fpos += 4;
        if (!fread(&blocksize, sizeof(int), 1, fd))
        {
            break;
        }
    }

    if (*swap == 0)
    {
        fclose(fd);
        return 0;
    }

    fpos = 0;
    fseeko(fd, 0, SEEK_SET);
    safe_fread(&blocksize, sizeof(int), 1, fd);
    while (!feof(fd))
    {
        blocksize = SwapInt(blocksize);
        if (fpos + blocksize + 4 > fsize)
        {
            *swap += 1;
            break;
        }
        fpos += 4 + blocksize;
        fseeko(fd, fpos, SEEK_SET);
        safe_fread(&blockend, sizeof(int), 1, fd);
        blockend = SwapInt(blockend);
        if (blocksize != blockend)
        {
            *swap += 1;
            break;
        }
        fpos += 4;
        if (!fread(&blocksize, sizeof(int), 1, fd))
        {
            break;
        }
    }

    fclose(fd);
    return 0;
}
