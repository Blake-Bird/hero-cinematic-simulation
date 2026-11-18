/**
 * @file
 *
 * This file is part of the N-body/SPH code OpenGadget3.
 *
 * @brief
 *
 * @author OpenGadget3-development team, in particular Antonio Ragagnin.
 *
 * @copyright Copyright (c) 2026, OpenGadget3-development team lead by Klaus Dolag and Stefano Borgani
 * (see change logs). All rights reserved.
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License v3 or later.
 * See <https://www.gnu.org/licenses/>.
 */

/*
 compile this file with command
 nvcc -x cu -D__GCC_ATOMIC_TEST_AND_SET_TRUEVAL=1 -ccbin=pgc++ -std=c++14  -dlink -c thrust_device_sort.cpp -o
 thrust_device_sort.o -lcudart test this file with pgc++  -acc -ta=tesla:cc70  -std=c++11  thrust_device_sort.o
 test.cpp -o test.exe -lcudart
*/

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>

#include "thrust_sort.hpp"

void thrust_device_keys_sort(unsigned long long *host_keys, int *host_indexes, int NumPart)
{

    thrust::device_vector<int> device_index(host_indexes, host_indexes + NumPart); // magical HtoD copy
    thrust::device_vector<unsigned long long> device_keys(host_keys, host_keys + NumPart); // magical HtoD copy
    thrust::sort_by_key(device_keys.begin(), device_keys.end(), device_index.begin());
    thrust::copy(device_index.begin(), device_index.end(), host_indexes);
}
