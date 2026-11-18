// This file is part of MUSIC
// A software package to generate ICs for cosmological simulations
// Copyright (C) 2010-2024 by Oliver Hahn
// 
// monofonIC is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
// 
// monofonIC is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
// 
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

// Native MUSIC gradient expressions shared by ordinary and wider potential fields.
// Extracted without changing coefficients, staggering, traversal or output arithmetic.
#pragma once
#include "mesh.hh"
#include <cmath>
namespace music_finite_difference {
template<class Potential>
inline void gradient_O2(int dir,const Potential& u,grid_hierarchy& Du)
{
	music::ulog.Print("Computing a 2nd order finite difference gradient...");

	for (unsigned ilevel = u.levelmin(); ilevel <= u.levelmax(); ++ilevel)
	{
		real_t h = pow(2.0, ilevel);
		meshvar_bnd *pvar = Du.get_grid(ilevel);

		if (dir == 0)
#pragma omp parallel for
			for (int ix = 0; ix < (int)(*u.get_grid(ilevel)).size(0); ++ix)
				for (int iy = 0; iy < (int)(*u.get_grid(ilevel)).size(1); ++iy)
					for (int iz = 0; iz < (int)(*u.get_grid(ilevel)).size(2); ++iz)
						(*pvar)(ix, iy, iz) = 0.5 * ((*u.get_grid(ilevel))(ix + 1, iy, iz) - (*u.get_grid(ilevel))(ix - 1, iy, iz)) * h;

		else if (dir == 1)
#pragma omp parallel for
			for (int ix = 0; ix < (int)(*u.get_grid(ilevel)).size(0); ++ix)
				for (int iy = 0; iy < (int)(*u.get_grid(ilevel)).size(1); ++iy)
					for (int iz = 0; iz < (int)(*u.get_grid(ilevel)).size(2); ++iz)
						(*pvar)(ix, iy, iz) = 0.5 * ((*u.get_grid(ilevel))(ix, iy + 1, iz) - (*u.get_grid(ilevel))(ix, iy - 1, iz)) * h;

		else if (dir == 2)
#pragma omp parallel for
			for (int ix = 0; ix < (int)(*u.get_grid(ilevel)).size(0); ++ix)
				for (int iy = 0; iy < (int)(*u.get_grid(ilevel)).size(1); ++iy)
					for (int iz = 0; iz < (int)(*u.get_grid(ilevel)).size(2); ++iz)
						(*pvar)(ix, iy, iz) = 0.5 * ((*u.get_grid(ilevel))(ix, iy, iz + 1) - (*u.get_grid(ilevel))(ix, iy, iz - 1)) * h;
	}

	music::ulog.Print("Done computing a 2nd order finite difference gradient.");
}

template<class Potential>
inline void gradient_add_O2(int dir,const Potential& u,grid_hierarchy& Du)
{
	music::ulog.Print("Computing a 2nd order finite difference gradient...");

	for (unsigned ilevel = u.levelmin(); ilevel <= u.levelmax(); ++ilevel)
	{
		real_t h = pow(2.0, ilevel);
		meshvar_bnd *pvar = Du.get_grid(ilevel);

		if (dir == 0)
#pragma omp parallel for
			for (int ix = 0; ix < (int)(*Du.get_grid(ilevel)).size(0); ++ix)
				for (int iy = 0; iy < (int)(*Du.get_grid(ilevel)).size(1); ++iy)
					for (int iz = 0; iz < (int)(*Du.get_grid(ilevel)).size(2); ++iz)
						(*pvar)(ix, iy, iz) += 0.5 * ((*u.get_grid(ilevel))(ix + 1, iy, iz) - (*u.get_grid(ilevel))(ix - 1, iy, iz)) * h;

		else if (dir == 1)
#pragma omp parallel for
			for (int ix = 0; ix < (int)(*Du.get_grid(ilevel)).size(0); ++ix)
				for (int iy = 0; iy < (int)(*Du.get_grid(ilevel)).size(1); ++iy)
					for (int iz = 0; iz < (int)(*Du.get_grid(ilevel)).size(2); ++iz)
						(*pvar)(ix, iy, iz) += 0.5 * ((*u.get_grid(ilevel))(ix, iy + 1, iz) - (*u.get_grid(ilevel))(ix, iy - 1, iz)) * h;

		else if (dir == 2)
#pragma omp parallel for
			for (int ix = 0; ix < (int)(*Du.get_grid(ilevel)).size(0); ++ix)
				for (int iy = 0; iy < (int)(*Du.get_grid(ilevel)).size(1); ++iy)
					for (int iz = 0; iz < (int)(*Du.get_grid(ilevel)).size(2); ++iz)
						(*pvar)(ix, iy, iz) += 0.5 * ((*u.get_grid(ilevel))(ix, iy, iz + 1) - (*u.get_grid(ilevel))(ix, iy, iz - 1)) * h;
	}

	music::ulog.Print("Done computing a 4th order finite difference gradient.");
}

template<class Potential>
inline void gradient_O4(int dir,const Potential& u,grid_hierarchy& Du)
{
	music::ulog.Print("Computing a 4th order finite difference gradient...");

	for (unsigned ilevel = u.levelmin(); ilevel <= u.levelmax(); ++ilevel)
	{
		real_t h = pow(2.0, ilevel);
		meshvar_bnd *pvar = Du.get_grid(ilevel);

		h /= 12.0;

		if (dir == 0)
#pragma omp parallel for
			for (int ix = 0; ix < (int)(*u.get_grid(ilevel)).size(0); ++ix)
				for (int iy = 0; iy < (int)(*u.get_grid(ilevel)).size(1); ++iy)
					for (int iz = 0; iz < (int)(*u.get_grid(ilevel)).size(2); ++iz)
						(*pvar)(ix, iy, iz) = ((*u.get_grid(ilevel))(ix - 2, iy, iz) - 8.0 * (*u.get_grid(ilevel))(ix - 1, iy, iz) + 8.0 * (*u.get_grid(ilevel))(ix + 1, iy, iz) - (*u.get_grid(ilevel))(ix + 2, iy, iz)) * h;

		else if (dir == 1)
#pragma omp parallel for
			for (int ix = 0; ix < (int)(*u.get_grid(ilevel)).size(0); ++ix)
				for (int iy = 0; iy < (int)(*u.get_grid(ilevel)).size(1); ++iy)
					for (int iz = 0; iz < (int)(*u.get_grid(ilevel)).size(2); ++iz)
						(*pvar)(ix, iy, iz) = ((*u.get_grid(ilevel))(ix, iy - 2, iz) - 8.0 * (*u.get_grid(ilevel))(ix, iy - 1, iz) + 8.0 * (*u.get_grid(ilevel))(ix, iy + 1, iz) - (*u.get_grid(ilevel))(ix, iy + 2, iz)) * h;

		else if (dir == 2)
#pragma omp parallel for
			for (int ix = 0; ix < (int)(*u.get_grid(ilevel)).size(0); ++ix)
				for (int iy = 0; iy < (int)(*u.get_grid(ilevel)).size(1); ++iy)
					for (int iz = 0; iz < (int)(*u.get_grid(ilevel)).size(2); ++iz)
						(*pvar)(ix, iy, iz) = ((*u.get_grid(ilevel))(ix, iy, iz - 2) - 8.0 * (*u.get_grid(ilevel))(ix, iy, iz - 1) + 8.0 * (*u.get_grid(ilevel))(ix, iy, iz + 1) - (*u.get_grid(ilevel))(ix, iy, iz + 2)) * h;
	}

	music::ulog.Print("Done computing a 4th order finite difference gradient.");
}

template<class Potential>
inline void gradient_add_O4(int dir,const Potential& u,grid_hierarchy& Du)
{
	music::ulog.Print("Computing a 4th order finite difference gradient...");

	for (unsigned ilevel = u.levelmin(); ilevel <= u.levelmax(); ++ilevel)
	{
		real_t h = pow(2.0, ilevel);
		meshvar_bnd *pvar = Du.get_grid(ilevel);

		h /= 12.0;

		if (dir == 0)
#pragma omp parallel for
			for (int ix = 0; ix < (int)(*u.get_grid(ilevel)).size(0); ++ix)
				for (int iy = 0; iy < (int)(*u.get_grid(ilevel)).size(1); ++iy)
					for (int iz = 0; iz < (int)(*u.get_grid(ilevel)).size(2); ++iz)
						(*pvar)(ix, iy, iz) += ((*u.get_grid(ilevel))(ix - 2, iy, iz) - 8.0 * (*u.get_grid(ilevel))(ix - 1, iy, iz) + 8.0 * (*u.get_grid(ilevel))(ix + 1, iy, iz) - (*u.get_grid(ilevel))(ix + 2, iy, iz)) * h;

		else if (dir == 1)
#pragma omp parallel for
			for (int ix = 0; ix < (int)(*u.get_grid(ilevel)).size(0); ++ix)
				for (int iy = 0; iy < (int)(*u.get_grid(ilevel)).size(1); ++iy)
					for (int iz = 0; iz < (int)(*u.get_grid(ilevel)).size(2); ++iz)
						(*pvar)(ix, iy, iz) += ((*u.get_grid(ilevel))(ix, iy - 2, iz) - 8.0 * (*u.get_grid(ilevel))(ix, iy - 1, iz) + 8.0 * (*u.get_grid(ilevel))(ix, iy + 1, iz) - (*u.get_grid(ilevel))(ix, iy + 2, iz)) * h;

		else if (dir == 2)
#pragma omp parallel for
			for (int ix = 0; ix < (int)(*u.get_grid(ilevel)).size(0); ++ix)
				for (int iy = 0; iy < (int)(*u.get_grid(ilevel)).size(1); ++iy)
					for (int iz = 0; iz < (int)(*u.get_grid(ilevel)).size(2); ++iz)
						(*pvar)(ix, iy, iz) += ((*u.get_grid(ilevel))(ix, iy, iz - 2) - 8.0 * (*u.get_grid(ilevel))(ix, iy, iz - 1) + 8.0 * (*u.get_grid(ilevel))(ix, iy, iz + 1) - (*u.get_grid(ilevel))(ix, iy, iz + 2)) * h;
	}

	music::ulog.Print("Done computing a 4th order finite difference gradient.");
}

template<class Potential>
inline void gradient_O6(int dir,const Potential& u,grid_hierarchy& Du)
{
	music::ulog.Print("Computing a 6th order finite difference gradient...");

	for (unsigned ilevel = u.levelmin(); ilevel <= u.levelmax(); ++ilevel)
	{
		real_t h = pow(2.0, ilevel);
		meshvar_bnd *pvar = Du.get_grid(ilevel);

		h /= 60.;
		if (dir == 0)
#pragma omp parallel for
			for (int ix = 0; ix < (int)(*u.get_grid(ilevel)).size(0); ++ix)
				for (int iy = 0; iy < (int)(*u.get_grid(ilevel)).size(1); ++iy)
					for (int iz = 0; iz < (int)(*u.get_grid(ilevel)).size(2); ++iz)
						(*pvar)(ix, iy, iz) =
								(-(*u.get_grid(ilevel))(ix - 3, iy, iz) + 9.0 * (*u.get_grid(ilevel))(ix - 2, iy, iz) - 45.0 * (*u.get_grid(ilevel))(ix - 1, iy, iz) + 45.0 * (*u.get_grid(ilevel))(ix + 1, iy, iz) - 9.0 * (*u.get_grid(ilevel))(ix + 2, iy, iz) + (*u.get_grid(ilevel))(ix + 3, iy, iz)) * h;

		else if (dir == 1)
#pragma omp parallel for
			for (int ix = 0; ix < (int)(*u.get_grid(ilevel)).size(0); ++ix)
				for (int iy = 0; iy < (int)(*u.get_grid(ilevel)).size(1); ++iy)
					for (int iz = 0; iz < (int)(*u.get_grid(ilevel)).size(2); ++iz)
						(*pvar)(ix, iy, iz) =
								(-(*u.get_grid(ilevel))(ix, iy - 3, iz) + 9.0 * (*u.get_grid(ilevel))(ix, iy - 2, iz) - 45.0 * (*u.get_grid(ilevel))(ix, iy - 1, iz) + 45.0 * (*u.get_grid(ilevel))(ix, iy + 1, iz) - 9.0 * (*u.get_grid(ilevel))(ix, iy + 2, iz) + (*u.get_grid(ilevel))(ix, iy + 3, iz)) * h;

		else if (dir == 2)
#pragma omp parallel for
			for (int ix = 0; ix < (int)(*u.get_grid(ilevel)).size(0); ++ix)
				for (int iy = 0; iy < (int)(*u.get_grid(ilevel)).size(1); ++iy)
					for (int iz = 0; iz < (int)(*u.get_grid(ilevel)).size(2); ++iz)
						(*pvar)(ix, iy, iz) =
								(-(*u.get_grid(ilevel))(ix, iy, iz - 3) + 9.0 * (*u.get_grid(ilevel))(ix, iy, iz - 2) - 45.0 * (*u.get_grid(ilevel))(ix, iy, iz - 1) + 45.0 * (*u.get_grid(ilevel))(ix, iy, iz + 1) - 9.0 * (*u.get_grid(ilevel))(ix, iy, iz + 2) + (*u.get_grid(ilevel))(ix, iy, iz + 3)) * h;
	}

	music::ulog.Print("Done computing a 6th order finite difference gradient.");
}

template<class Potential>
inline void gradient_add_O6(int dir,const Potential& u,grid_hierarchy& Du)
{
	music::ulog.Print("Computing a 6th order finite difference gradient...");

	for (unsigned ilevel = u.levelmin(); ilevel <= u.levelmax(); ++ilevel)
	{
		real_t h = pow(2.0, ilevel);
		meshvar_bnd *pvar = Du.get_grid(ilevel);

		h /= 60.;
		if (dir == 0)
#pragma omp parallel for
			for (int ix = 0; ix < (int)(*u.get_grid(ilevel)).size(0); ++ix)
				for (int iy = 0; iy < (int)(*u.get_grid(ilevel)).size(1); ++iy)
					for (int iz = 0; iz < (int)(*u.get_grid(ilevel)).size(2); ++iz)
						(*pvar)(ix, iy, iz) +=
								(-(*u.get_grid(ilevel))(ix - 3, iy, iz) + 9.0 * (*u.get_grid(ilevel))(ix - 2, iy, iz) - 45.0 * (*u.get_grid(ilevel))(ix - 1, iy, iz) + 45.0 * (*u.get_grid(ilevel))(ix + 1, iy, iz) - 9.0 * (*u.get_grid(ilevel))(ix + 2, iy, iz) + (*u.get_grid(ilevel))(ix + 3, iy, iz)) * h;

		else if (dir == 1)
#pragma omp parallel for
			for (int ix = 0; ix < (int)(*u.get_grid(ilevel)).size(0); ++ix)
				for (int iy = 0; iy < (int)(*u.get_grid(ilevel)).size(1); ++iy)
					for (int iz = 0; iz < (int)(*u.get_grid(ilevel)).size(2); ++iz)
						(*pvar)(ix, iy, iz) +=
								(-(*u.get_grid(ilevel))(ix, iy - 3, iz) + 9.0 * (*u.get_grid(ilevel))(ix, iy - 2, iz) - 45.0 * (*u.get_grid(ilevel))(ix, iy - 1, iz) + 45.0 * (*u.get_grid(ilevel))(ix, iy + 1, iz) - 9.0 * (*u.get_grid(ilevel))(ix, iy + 2, iz) + (*u.get_grid(ilevel))(ix, iy + 3, iz)) * h;

		else if (dir == 2)
#pragma omp parallel for
			for (int ix = 0; ix < (int)(*u.get_grid(ilevel)).size(0); ++ix)
				for (int iy = 0; iy < (int)(*u.get_grid(ilevel)).size(1); ++iy)
					for (int iz = 0; iz < (int)(*u.get_grid(ilevel)).size(2); ++iz)
						(*pvar)(ix, iy, iz) +=
								(-(*u.get_grid(ilevel))(ix, iy, iz - 3) + 9.0 * (*u.get_grid(ilevel))(ix, iy, iz - 2) - 45.0 * (*u.get_grid(ilevel))(ix, iy, iz - 1) + 45.0 * (*u.get_grid(ilevel))(ix, iy, iz + 1) - 9.0 * (*u.get_grid(ilevel))(ix, iy, iz + 2) + (*u.get_grid(ilevel))(ix, iy, iz + 3)) * h;
	}

	music::ulog.Print("Done computing a 6th order finite difference gradient.");
}
}
