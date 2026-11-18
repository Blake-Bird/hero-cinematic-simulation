/**
 * @file
 *
 * This file is part of the N-body/SPH code OpenGadget3.
 *
 * @brief This file contains the io-matrix.
 *
 * @author OpenGadget3-development team, in particular Geray S. Karademir.
 *
 * @copyright Copyright (c) 2026, OpenGadget3-development team lead by Klaus Dolag and Stefano Borgani
 * (see change logs). All rights reserved.
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License v3 or later.
 * See <https://www.gnu.org/licenses/>.
 */

#include "../CodeBase/verbose.hpp"
#include "io.hpp"

#ifndef IO_MATRIX_HPP
#define IO_MATRIX_HPP

namespace io_matrix
{
int get_values_per_blockelement(enum iofields blocknr);
void get_dataset_name(enum iofields blocknr, char *buf);
int blockpresent(enum iofields blocknr);
int get_bytes_per_blockelement(enum iofields blocknr, int mode);
void get_Tab_IO_Label(enum iofields blocknr, char *label);
unsigned int get_particles_in_block(enum iofields blocknr, unsigned int *typelist);
int get_datatype_in_block(enum iofields blocknr);
int check_if_read_or_write_block(enum iofields blocknr, int RestartFlag);
void init_io_fields(
);
} // namespace io_matrix

#endif // IO_MATRIX_HPP
