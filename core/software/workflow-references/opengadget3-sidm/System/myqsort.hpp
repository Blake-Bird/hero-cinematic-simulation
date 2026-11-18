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

#ifndef MYQSORT_HPP
#define MYQSORT_HPP

/**
 * @brief Sorts an array of elements in ascending order using the quicksort algorithm.
 *
 * This function sorts an array of elements in ascending order based on a specified key extracted from each element.
 *
 * @param a Pointer to the array of elements to be sorted.
 * @param n The number of elements in the array.
 *
 * @tparam KEY_TYPE The type of elements in the array.
 *
 * @note The following macros must be defined prior to using this function:
 * - QSORT: Name of the quicksort function.
 * - KEY_TYPE: Type of the elements in the array to be sorted.
 * - KEY_COPY(a, b): Macro/function-like macro to copy the key from element 'a' to element 'b'.
 * - GET_KEYVAL(a): Macro/function-like macro to extract the key value from element 'a'.
 */
static void QSORT(KEY_TYPE *a, int n)
{
#pragma alloca
    KEY_TYPE *b = (KEY_TYPE *)alloca(sizeof(KEY_TYPE) * n);

    if (n > 1)
    {
        double avg = 0.0;
        KEY_BASE_TYPE kavg;

        /* compute average key value and use as pivot point */

        for (int i = 0; i < n; i++)
        {
            avg += KEY_GETVAL(&a[i]);
        }

        kavg = (KEY_BASE_TYPE)(avg / (double)n);

        int j1 = 0;
        int j2 = n;

        for (int i = 0; i < n; i++)
        {
            if (KEY_GETVAL(&a[i]) <= kavg)
            {
                KEY_COPY(&a[i], &b[j1]);
                j1++;
            }
            else
            {
                --j2;
                KEY_COPY(&a[i], &b[j2]);
            }
        }

        if (j1 > 0 && j2 < n)
        {

            if (j1 > 0)
            {
                QSORT(b, j1);
            }
            if (j2 < n)
            {
                QSORT(&b[j1], n - j1);
            }
            memcpy(a, b, sizeof(KEY_TYPE) * n);
        }
    }
}
#endif // MYQSORT_HPP
