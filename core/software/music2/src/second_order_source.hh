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

// Existing native MUSIC second-order source, shared across potential scalar types.
#pragma once
#include "mesh.hh"
#include "mg_operators.hh"
#include <type_traits>
#define ACC(i,j,k) ((*u.get_grid(ilevel))((i),(j),(k)))
#define SQR(x) ((x)*(x))
namespace music_finite_difference {
template<class Potential>
inline void second_order_source(const Potential& u,grid_hierarchy& fnew,unsigned order)
{
	fnew.copy_geometry_from(u);
	fnew.zero();

	for (unsigned ilevel = u.levelmin(); ilevel <= u.levelmax(); ++ilevel)
	{
		double h = pow(2.0, ilevel), h2 = h * h, h2_4 = 0.25 * h2;
		meshvar_bnd *pvar = fnew.get_grid(ilevel);

		if (order == 2)
		{

#pragma omp parallel for
			for (int ix = 0; ix < (int)(*u.get_grid(ilevel)).size(0); ++ix)
				for (int iy = 0; iy < (int)(*u.get_grid(ilevel)).size(1); ++iy)
					for (int iz = 0; iz < (int)(*u.get_grid(ilevel)).size(2); ++iz)
					{
						using stored_scalar=std::remove_cv_t<std::remove_reference_t<decltype((*u.get_grid(ilevel))(0,0,0))>>;
						using potential_scalar=std::conditional_t<(sizeof(stored_scalar)>sizeof(double)),stored_scalar,double>;
						potential_scalar D[3][3];

						D[0][0] = (ACC(ix - 2, iy, iz) - 2.0 * ACC(ix, iy, iz) + ACC(ix + 2, iy, iz)) * h2_4;
						D[1][1] = (ACC(ix, iy - 2, iz) - 2.0 * ACC(ix, iy, iz) + ACC(ix, iy + 2, iz)) * h2_4;
						D[2][2] = (ACC(ix, iy, iz - 2) - 2.0 * ACC(ix, iy, iz) + ACC(ix, iy, iz + 2)) * h2_4;

						D[0][1] = D[1][0] = (ACC(ix - 1, iy - 1, iz) - ACC(ix - 1, iy + 1, iz) - ACC(ix + 1, iy - 1, iz) + ACC(ix + 1, iy + 1, iz)) * h2_4;
						D[0][2] = D[2][0] = (ACC(ix - 1, iy, iz - 1) - ACC(ix - 1, iy, iz + 1) - ACC(ix + 1, iy, iz - 1) + ACC(ix + 1, iy, iz + 1)) * h2_4;
						D[1][2] = D[2][1] = (ACC(ix, iy - 1, iz - 1) - ACC(ix, iy - 1, iz + 1) - ACC(ix, iy + 1, iz - 1) + ACC(ix, iy + 1, iz + 1)) * h2_4;

						(*pvar)(ix, iy, iz) = (D[0][0] * D[1][1] - D[0][1] * D[0][1] + D[0][0] * D[2][2] - D[0][2] * D[0][2] + D[1][1] * D[2][2] - D[1][2] * D[1][2]);
					}
		}
		else if (order == 4 || order == 6)
		{
			double h2_144 = h2 / 144.;
#pragma omp parallel for
			for (int ix = 0; ix < (int)(*u.get_grid(ilevel)).size(0); ++ix)
				for (int iy = 0; iy < (int)(*u.get_grid(ilevel)).size(1); ++iy)
					for (int iz = 0; iz < (int)(*u.get_grid(ilevel)).size(2); ++iz)
					{
						//.. this is actually 8th order accurate

						using stored_scalar=std::remove_cv_t<std::remove_reference_t<decltype((*u.get_grid(ilevel))(0,0,0))>>;
						using potential_scalar=std::conditional_t<(sizeof(stored_scalar)>sizeof(double)),stored_scalar,double>;
						potential_scalar D[3][3];

						D[0][0] = ((ACC(ix - 4, iy, iz) + ACC(ix + 4, iy, iz)) - 16. * (ACC(ix - 3, iy, iz) + ACC(ix + 3, iy, iz)) + 64. * (ACC(ix - 2, iy, iz) + ACC(ix + 2, iy, iz)) + 16. * (ACC(ix - 1, iy, iz) + ACC(ix + 1, iy, iz)) - 130. * ACC(ix, iy, iz)) * h2_144;

						D[1][1] = ((ACC(ix, iy - 4, iz) + ACC(ix, iy + 4, iz)) - 16. * (ACC(ix, iy - 3, iz) + ACC(ix, iy + 3, iz)) + 64. * (ACC(ix, iy - 2, iz) + ACC(ix, iy + 2, iz)) + 16. * (ACC(ix, iy - 1, iz) + ACC(ix, iy + 1, iz)) - 130. * ACC(ix, iy, iz)) * h2_144;

						D[2][2] = ((ACC(ix, iy, iz - 4) + ACC(ix, iy, iz + 4)) - 16. * (ACC(ix, iy, iz - 3) + ACC(ix, iy, iz + 3)) + 64. * (ACC(ix, iy, iz - 2) + ACC(ix, iy, iz + 2)) + 16. * (ACC(ix, iy, iz - 1) + ACC(ix, iy, iz + 1)) - 130. * ACC(ix, iy, iz)) * h2_144;

						D[0][1] = D[1][0] = (64. * (ACC(ix - 1, iy - 1, iz) - ACC(ix - 1, iy + 1, iz) - ACC(ix + 1, iy - 1, iz) + ACC(ix + 1, iy + 1, iz)) - 8. * (ACC(ix - 2, iy - 1, iz) - ACC(ix + 2, iy - 1, iz) - ACC(ix - 2, iy + 1, iz) + ACC(ix + 2, iy + 1, iz) + ACC(ix - 1, iy - 2, iz) - ACC(ix - 1, iy + 2, iz) - ACC(ix + 1, iy - 2, iz) + ACC(ix + 1, iy + 2, iz)) + 1. * (ACC(ix - 2, iy - 2, iz) - ACC(ix - 2, iy + 2, iz) - ACC(ix + 2, iy - 2, iz) + ACC(ix + 2, iy + 2, iz))) * h2_144;
						D[0][2] = D[2][0] = (64. * (ACC(ix - 1, iy, iz - 1) - ACC(ix - 1, iy, iz + 1) - ACC(ix + 1, iy, iz - 1) + ACC(ix + 1, iy, iz + 1)) - 8. * (ACC(ix - 2, iy, iz - 1) - ACC(ix + 2, iy, iz - 1) - ACC(ix - 2, iy, iz + 1) + ACC(ix + 2, iy, iz + 1) + ACC(ix - 1, iy, iz - 2) - ACC(ix - 1, iy, iz + 2) - ACC(ix + 1, iy, iz - 2) + ACC(ix + 1, iy, iz + 2)) + 1. * (ACC(ix - 2, iy, iz - 2) - ACC(ix - 2, iy, iz + 2) - ACC(ix + 2, iy, iz - 2) + ACC(ix + 2, iy, iz + 2))) * h2_144;
						D[1][2] = D[2][1] = (64. * (ACC(ix, iy - 1, iz - 1) - ACC(ix, iy - 1, iz + 1) - ACC(ix, iy + 1, iz - 1) + ACC(ix, iy + 1, iz + 1)) - 8. * (ACC(ix, iy - 2, iz - 1) - ACC(ix, iy + 2, iz - 1) - ACC(ix, iy - 2, iz + 1) + ACC(ix, iy + 2, iz + 1) + ACC(ix, iy - 1, iz - 2) - ACC(ix, iy - 1, iz + 2) - ACC(ix, iy + 1, iz - 2) + ACC(ix, iy + 1, iz + 2)) + 1. * (ACC(ix, iy - 2, iz - 2) - ACC(ix, iy - 2, iz + 2) - ACC(ix, iy + 2, iz - 2) + ACC(ix, iy + 2, iz + 2))) * h2_144;

						(*pvar)(ix, iy, iz) = (D[0][0] * D[1][1] - SQR(D[0][1]) + D[0][0] * D[2][2] - SQR(D[0][2]) + D[1][1] * D[2][2] - SQR(D[1][2]));
					}
		}
		else
			throw std::runtime_error("compute_2LPT_source : invalid operator order specified");
	}

	//.. subtract global mean so the multi-grid poisson solver behaves well

	for (int i = fnew.levelmax(); i > (int)fnew.levelmin(); --i)
		mg_straight().restrict((*fnew.get_grid(i)), (*fnew.get_grid(i - 1)));

	long double sum = 0.0;
	int nx, ny, nz;

	nx = fnew.get_grid(fnew.levelmin())->size(0);
	ny = fnew.get_grid(fnew.levelmin())->size(1);
	nz = fnew.get_grid(fnew.levelmin())->size(2);

	for (int ix = 0; ix < nx; ++ix)
		for (int iy = 0; iy < ny; ++iy)
			for (int iz = 0; iz < nz; ++iz)
				sum += (*fnew.get_grid(fnew.levelmin()))(ix, iy, iz);

	sum /= (double)((size_t)nx * (size_t)ny * (size_t)nz);

	for (unsigned i = fnew.levelmin(); i <= fnew.levelmax(); ++i)
	{
		nx = fnew.get_grid(i)->size(0);
		ny = fnew.get_grid(i)->size(1);
		nz = fnew.get_grid(i)->size(2);

		for (int ix = 0; ix < nx; ++ix)
			for (int iy = 0; iy < ny; ++iy)
				for (int iz = 0; iz < nz; ++iz)
					(*fnew.get_grid(i))(ix, iy, iz) -= sum;
	}
}
}
#undef SQR
#undef ACC
