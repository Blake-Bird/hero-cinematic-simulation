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

#ifndef SOA_HPP
#define SOA_HPP

#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <string>
#include <map>
#include <memory>
#include <vector>
#include <exception>
#include <stddef.h>

using namespace std;

/*
  data of a single array, e.g. T=float, {.name='rho', .is_density_in=true}
*/
typedef MyAtLeastDouble MyAtLeastDouble3[3];
typedef MyLongDouble MyLongDouble3[3];
typedef float float3[3];

/**
 * @brief Enumeration of assignment types for structured arrays (SoA).
 *
 * This enum defines different types of assignment operations that can be performed on
 * structured arrays (SoA). SoAAssignType specifies how new values are assigned to existing
 * values in the SoA data structures.
 */
enum SoAAssignType
{
    ASSIGN_ADD,
    ASSIGN_MAX
    // to be continued if you need more..
}

/**
 * @brief Descriptor for structured array (SoA) data elements.
 *
 * This struct defines a descriptor for storing information about structured array (SoA) data elements,
 * including type, dimensions, name, pointer, and additional properties for data output.
 */
struct SoADescriptor
{

    string name;

    size_t size = 0;

    bool is_density_out = false;
    size_t densdata_out_displacement;
    SoAAssignType densdata_out_assign;

    bool is_hydro_out = false;
    size_t hydrodata_out_displacement;
    SoAAssignType hydrodata_out_assign;

    shared_ptr<void> *data = NULL;
    bool allocated = false;
};

/**
 * @brief Exception class for managing errors when accessing data.
 *
 * This struct defines an exception class for managing errors that occur when accessing data.
 * It inherits from std::exception and overrides the what() method to provide a description of the exception.
 */
struct MyException : public std::exception
{
    std::string s;
    MyException(std::string ss) : s(ss)
    {
    }
    ~MyException() throw()
    {
    } // Updated
    const char *what() const throw()
    {
        return s.c_str();
    }
};

/**
 * @brief Struct to handle errors when accessing data.
 *
 * This struct defines a custom exception class for managing errors that occur when accessing data.
 * It inherits from std::exception and overrides the what() method to provide a description of the exception.
 */
class SoA
{
    map<string, SoADescriptor *> m;
    bool allocated = false;

  public:
    /*
      we initialise our SoA with an array of SoADescriptor
    */
    SoA(vector<SoADescriptor> &arrays)
    {
        this->allocated = false;

        for (auto const &v : arrays)
        {
            this->m[v.name] = &v;
        }
    };
    // access array from key
    shared_ptr<void> *get(string &prop)
    {
        if (m.count(prop))
            return m[prop]->data;
        else
            throw MyException("SoA: key not found: " + prop);
    };

    // assign from DensDataOut our equivalent
    template <typename DataOut, typename DataType>
    void assign(string &prop, T *DataOut, size_t displacement, int range_min, int range_max, data_index *DataIndexTable,
                int mode)
    {
        DataType *a = (DataType *)get(prop);
        for (int i = range_min; i < range_max; i++)
        {
            int place = DataIndexTable[i].Index;
            // we extract the value from the DataOut
            DataType v = (DataType)((byte *)(DataOut + i) + displacement);

            if (mode == 0)
            {
                a[place] = v;
            }
            else
            {
                a[place] += v;
            }
        }
    };

    // template <typename DataOut>
    //   void out2particle(DataOut *out, int i, int mode){

    //  };
    /*
       we allocate each mapped arrays with n elements * size bytes;
       f decides which field we allocate
     */
    void alloc(int n, function<void(SoADescriptor *)> f)
    {
        if (this->allocated)
        {
            throw "SoA already allocated.";
        }

        for (auto const &v : this->m)
        {
            SoADescriptor *e = v.second;
            if (f(e))
            {
                if (e->size == 0)
                {
                    throw MyException("SoA: zero size datatype " + e->name);
                }

                size_t bytes = e->size * n;

                e->data = mymalloc(e->name, bytes);
                e->allocated = true;
            }
        }
        this->allocated = true;
    };

    /*
       we allocate all mapped arrays
     */
    void alloc(int n)
    {
        alloc(n, [](SoADescriptor *d) -> void { return true });
    };
    /*
       deallocate all mapped arrays
     */
    void free()
    {
        if (!this->allocated)
        {
            throw "SoA not allocated.";
        }

        // we free in reverse order
        for (auto iter = m.rbegin(); iter != m.rend(); ++iter)
        {
            if (iter->allocated)
            {
                ::myfree(iter->second->data);
            }
        }

        this->allocated = false;
    };
};

/**
 * @brief Initialize the Structure-of-Arrays (SoA) data.
 *
 * This function initializes the SoA data structure with a predefined set of fields.
 * Each field is described by a SoADescriptor object specifying its name, size, and other attributes.
 * The function creates a new SoA instance with the provided fields.
 */
static void soa_init()
{
    vector<SoADescriptor> fields = {
#ifdef AR_FIX_TIME_DEP_ART_COND
        {.name = "RhoOut",
         .size = sizeof(MyAtLeastDouble),
         .is_density_out = true,
         .densdata_out_displacement = offsetof(DensDataOut, Rho),
         .densdata_out_assign = ASSIGN_ADD},
        {.name = "GradAOut",
         .size = sizeof(MyAtLeastDouble3),
         .is_density_out = true,
         .densdata_out_displacement = offsetof(DensDataOut, GradAOut),
         .densdata_out_assign = ASSIGN_ADD},
#endif // AR_FIX_TIME_DEP_ART_COND
#ifdef AR_FIX_WAKEUP
        {.name = "MaxSignalVelOut",
         .size = sizeof(MyFloat),
         .is_hydro_out = true,
         .densdata_out_displacement = offsetof(HydroDataOut, MaxSignalVelOut),
         .densdata_out_assign = ASSIGN_MAX}
#endif // AR_FIX_WAKEUP
    };

    soa = new SoA(fields);
};

/**
 * @brief Empty function
 */
static void soa_end(){};

#endif // SOA_HPP
