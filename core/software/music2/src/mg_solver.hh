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

#pragma once

#include <cmath>
#include <iostream>
#include <limits>

#include <mg_operators.hh>
#include <mg_interp.hh>

#include <mesh.hh>

#define BEGIN_MULTIGRID_NAMESPACE \
	namespace multigrid             \
	{
#define END_MULTIGRID_NAMESPACE }

BEGIN_MULTIGRID_NAMESPACE

//! options for multigrid smoothing operation
namespace opt
{
	enum smtype
	{
		sm_jacobi,
		sm_gauss_seidel,
		sm_sor
	};
}

//! actual implementation of FAS adaptive multigrid solver
template <class S, class I, class O, typename scalar_t=real_t>
class solver
{
public:
	unsigned maximum_iterations = 500;
	typedef S scheme;
	typedef O mgop;
	typedef I interp;

protected:
	scheme m_scheme;				//!< finite difference scheme
	mgop m_gridop;					//!< grid prolongation and restriction operator
	unsigned m_npresmooth,	//!< number of pre sweeps
			m_npostsmooth;			//!< number of post sweeps
	opt::smtype m_smoother; //!< smoothing method to be applied
	unsigned m_ilevelmin;		//!< index of the top grid level

	const static bool m_bperiodic = true; //!< flag whether top grid is periodic

	std::vector<double> m_residu_ini; //!< vector of initial residuals for each level
	bool m_is_ini;										//!< bool that is true for first iteration

	GridHierarchy<scalar_t>
			*m_pu,		 //!< pointer to GridHierarchy for solution u
			*m_pf,		 //!< pointer to GridHierarchy for right-hand-side
			*m_pfsave; //!< pointer to saved state of right-hand-side (unused)

	const MeshvarBnd<scalar_t> *m_pubnd;

	//! compute residual for a level
	double compute_error(const MeshvarBnd<scalar_t> &u, const MeshvarBnd<scalar_t> &unew, int ilevel);

	//! compute residuals for entire grid hierarchy
	double compute_error(const GridHierarchy<scalar_t> &uh, const GridHierarchy<scalar_t> &uhnew, bool verbose);

	//! compute residuals for entire grid hierarchy
	double compute_RMS_resid(const GridHierarchy<scalar_t> &uh, const GridHierarchy<scalar_t> &fh, bool verbose);

protected:
	//! Jacobi smoothing
	void Jacobi(scalar_t h, MeshvarBnd<scalar_t> *u, const MeshvarBnd<scalar_t> *f);

	//! Gauss-Seidel smoothing
	void GaussSeidel(scalar_t h, MeshvarBnd<scalar_t> *u, const MeshvarBnd<scalar_t> *f);

	//! Successive-Overrelaxation smoothing
	void SOR(scalar_t h, MeshvarBnd<scalar_t> *u, const MeshvarBnd<scalar_t> *f);

	//! main two-grid (V-cycle) for multi-grid iterations
	void twoGrid(unsigned ilevel);

	//! apply boundary conditions
	void setBC(unsigned ilevel);

	//! make top grid periodic boundary conditions
	void make_periodic(MeshvarBnd<scalar_t> *u);

	// void interp_coarse_fine_cubic( unsigned ilevel, MeshvarBnd<scalar_t>& coarse, MeshvarBnd<scalar_t>& fine );

public:
	//! constructor
	solver(GridHierarchy<scalar_t> &f, opt::smtype smoother, unsigned npresmooth, unsigned npostsmooth);

	//! destructor
	~solver()
	{
	}

	//! solve Poisson's equation
	double solve(GridHierarchy<scalar_t> &u, double accuracy, double h = -1.0, bool verbose = false);

	//! solve Poisson's equation
	double solve(GridHierarchy<scalar_t> &u, double accuracy, bool verbose = false)
	{
		return this->solve(u, accuracy, -1.0, verbose);
	}
};

template <class S, class I, class O, typename scalar_t>
solver<S, I, O, scalar_t>::solver(GridHierarchy<scalar_t> &f, opt::smtype smoother, unsigned npresmooth, unsigned npostsmooth)
		: m_scheme(), m_gridop(), m_npresmooth(npresmooth), m_npostsmooth(npostsmooth),
			m_smoother(smoother), m_ilevelmin(f.levelmin()), m_is_ini(true), m_pf(&f)
{
	m_is_ini = true;
	m_residu_ini.resize(f.levelmax() + 1, 0.0);
}

template <class S, class I, class O, typename scalar_t>
void solver<S, I, O, scalar_t>::Jacobi(scalar_t h, MeshvarBnd<scalar_t> *u, const MeshvarBnd<scalar_t> *f)
{
	int
			nx = u->size(0),
			ny = u->size(1),
			nz = u->size(2);

	double
			c0 = -1.0 / m_scheme.ccoeff(),
			h2 = h * h;

	MeshvarBnd<scalar_t> uold(*u);

	double alpha = 0.95, ialpha = 1.0 - alpha;

#pragma omp parallel for
	for (int ix = 0; ix < nx; ++ix)
		for (int iy = 0; iy < ny; ++iy)
			for (int iz = 0; iz < nz; ++iz)
				(*u)(ix, iy, iz) = ialpha * uold(ix, iy, iz) + alpha * (m_scheme.rhs(uold, ix, iy, iz) + h2 * (*f)(ix, iy, iz)) * c0;
}

template <class S, class I, class O, typename scalar_t>
void solver<S, I, O, scalar_t>::SOR(scalar_t h, MeshvarBnd<scalar_t> *u, const MeshvarBnd<scalar_t> *f)
{
	int
			nx = u->size(0),
			ny = u->size(1),
			nz = u->size(2);

	double
			c0 = -1.0 / m_scheme.ccoeff(),
			h2 = h * h;

	MeshvarBnd<scalar_t> uold(*u);

	double
			alpha = 1.2,
			// alpha = 2 / (1 + 4 * atan(1.0) / double(u->size(0)))-1.0, //.. ideal alpha
			ialpha = 1.0 - alpha;

#pragma omp parallel for
	for (int ix = 0; ix < nx; ++ix)
		for (int iy = 0; iy < ny; ++iy)
			for (int iz = 0; iz < nz; ++iz)
				if ((ix + iy + iz) % 2 == 0)
					(*u)(ix, iy, iz) = ialpha * uold(ix, iy, iz) + alpha * (m_scheme.rhs(uold, ix, iy, iz) + h2 * (*f)(ix, iy, iz)) * c0;

// Higher-order odd cells share neighbors; update this dependent pass serially.
	for (int ix = 0; ix < nx; ++ix)
		for (int iy = 0; iy < ny; ++iy)
			for (int iz = 0; iz < nz; ++iz)
				if ((ix + iy + iz) % 2 != 0)
					(*u)(ix, iy, iz) = ialpha * uold(ix, iy, iz) + alpha * (m_scheme.rhs(*u, ix, iy, iz) + h2 * (*f)(ix, iy, iz)) * c0;
}

template <class S, class I, class O, typename scalar_t>
void solver<S, I, O, scalar_t>::GaussSeidel(scalar_t h, MeshvarBnd<scalar_t> *u, const MeshvarBnd<scalar_t> *f)
{
	int
			nx = u->size(0),
			ny = u->size(1),
			nz = u->size(2);

	scalar_t
			c0 = -1.0 / m_scheme.ccoeff(),
			h2 = h * h;

	// Same-color cells must not share a nonzero stencil coupling.
	// Red/black is safe only for the nearest-neighbor Laplacian.
	for (int color = 0; color < S::smoothing_colors; ++color)
		#pragma omp parallel for collapse(3)
		for (int ix = 0; ix < nx; ++ix)
			for (int iy = 0; iy < ny; ++iy)
				for (int iz = 0; iz < nz; ++iz)
					if ((ix + iy + iz) % S::smoothing_colors == color)
						// Algebraically identical Gauss-Seidel update, using the
						// local residual rather than reconstructing a near-equal
						// potential from a large weighted neighbour sum.
						(*u)(ix, iy, iz) += (m_scheme.apply(*u, ix, iy, iz) + h2 * (*f)(ix, iy, iz)) * c0;
}

template <class S, class I, class O, typename scalar_t>
void solver<S, I, O, scalar_t>::twoGrid(unsigned ilevel)
{
	MeshvarBnd<scalar_t> *uf, *uc, *ff, *fc;

	double
			h = 1.0 / (1 << ilevel),
			c0 = -1.0 / m_scheme.ccoeff(),
			h2 = h * h;

	uf = m_pu->get_grid(ilevel);
	ff = m_pf->get_grid(ilevel);

	uc = m_pu->get_grid(ilevel - 1);
	fc = m_pf->get_grid(ilevel - 1);

	int
			nx = uf->size(0),
			ny = uf->size(1),
			nz = uf->size(2);

	if (m_bperiodic && ilevel <= m_ilevelmin)
		make_periodic(uf);
	else if (!m_bperiodic)
		setBC(ilevel);

	//... do smoothing sweeps with specified solver
	for (unsigned i = 0; i < m_npresmooth; ++i)
	{

		if (ilevel > m_ilevelmin)
			interp().interp_coarse_fine(ilevel, *uc, *uf);

		if (m_smoother == opt::sm_gauss_seidel)
			GaussSeidel(h, uf, ff);

		else if (m_smoother == opt::sm_jacobi)
			Jacobi(h, uf, ff);

		else if (m_smoother == opt::sm_sor)
			SOR(h, uf, ff);

		if (m_bperiodic && ilevel <= m_ilevelmin)
			make_periodic(uf);
	}

	m_gridop.restrict(*uf, *uc);

	//... essential!!
	if (m_bperiodic && ilevel <= m_ilevelmin)
		make_periodic(uc);
	else if (ilevel > m_ilevelmin)
		interp().interp_coarse_fine(ilevel, *uc, *uf);

	//....................................................................
	//... we now use hard-coded restriction+operatore app, see below
	/*MeshvarBnd<scalar_t> Lu(*uf,false);
	Lu.zero();

	#pragma omp parallel for
	for( int ix=0; ix<nx; ++ix )
		for( int iy=0; iy<ny; ++iy )
			for( int iz=0; iz<nz; ++iz )
				Lu(ix,iy,iz) = m_scheme.apply( (*uf), ix, iy, iz )/h2;

	MeshvarBnd<scalar_t> tLu(*uc,false);


	//... restrict Lu
	m_gridop.restrict( Lu, tLu );
	Lu.deallocate();*/
	//....................................................................

	int
			oxp = uf->offset(0),
			oyp = uf->offset(1),
			ozp = uf->offset(2);

	MeshvarBnd<scalar_t> tLu(*uc, false);
	#pragma omp parallel for
	for (int ix = 0; ix < nx / 2; ++ix)
	{
		int iix = 2 * ix;
		for (int iy = 0, iiy = 0; iy < ny / 2; ++iy, iiy += 2)

			for (int iz = 0, iiz = 0; iz < nz / 2; ++iz, iiz += 2)
				tLu(ix + oxp, iy + oyp, iz + ozp) = 0.125 * (m_scheme.apply((*uf), iix, iiy, iiz) + m_scheme.apply((*uf), iix, iiy, iiz + 1) + m_scheme.apply((*uf), iix, iiy + 1, iiz) + m_scheme.apply((*uf), iix, iiy + 1, iiz + 1) + m_scheme.apply((*uf), iix + 1, iiy, iiz) + m_scheme.apply((*uf), iix + 1, iiy, iiz + 1) + m_scheme.apply((*uf), iix + 1, iiy + 1, iiz) + m_scheme.apply((*uf), iix + 1, iiy + 1, iiz + 1)) / h2;
	}

	//... restrict source term
	m_gridop.restrict(*ff, *fc);

	int oi, oj, ok;
	oi = ff->offset(0);
	oj = ff->offset(1);
	ok = ff->offset(2);

#pragma omp parallel for collapse(3)
	for (int ix = oi; ix < oi + (int)ff->size(0) / 2; ++ix)
		for (int iy = oj; iy < oj + (int)ff->size(1) / 2; ++iy)
			for (int iz = ok; iz < ok + (int)ff->size(2) / 2; ++iz)
				(*fc)(ix, iy, iz) += ((tLu(ix, iy, iz) - (m_scheme.apply(*uc, ix, iy, iz) / (4.0 * h2))));

	// Periodic Poisson has no solution for a nonzero source mean. Restriction
	// across a refinement interface can introduce a DC term into the FAS
	// coarse source, even when the original density contrast is normalized.
	// Project that null-space component out on full periodic grids only.
	if (m_bperiodic && ilevel - 1 <= m_ilevelmin)
	{
		long double source_sum = 0.0;
		const int cnx = fc->size(0), cny = fc->size(1), cnz = fc->size(2);
		#pragma omp parallel for reduction(+ : source_sum) collapse(3)
		for (int ix = 0; ix < cnx; ++ix)
			for (int iy = 0; iy < cny; ++iy)
				for (int iz = 0; iz < cnz; ++iz)
					source_sum += (*fc)(ix, iy, iz);
		const double source_mean = source_sum / (static_cast<size_t>(cnx) * cny * cnz);
		#pragma omp parallel for collapse(3)
		for (int ix = 0; ix < cnx; ++ix)
			for (int iy = 0; iy < cny; ++iy)
				for (int iz = 0; iz < cnz; ++iz)
					(*fc)(ix, iy, iz) -= source_mean;
	}

	tLu.deallocate();

	MeshvarBnd<scalar_t> ucsave(*uc, true);

	//... have we reached the end of the recursion or do we need to go up one level?
	if (ilevel == 1)
		if (m_bperiodic)
			(*uc)(0, 0, 0) = 0.0;
		else
			(*uc)(0, 0, 0) = (m_scheme.rhs((*uc), 0, 0, 0) + 4.0 * h2 * (*fc)(0, 0, 0)) * c0;
	else
		twoGrid(ilevel - 1);

	MeshvarBnd<scalar_t> cc(*uc, false);

//... compute correction on coarse grid
#pragma omp parallel for collapse(3)
	for (int ix = 0; ix < (int)cc.size(0); ++ix)
		for (int iy = 0; iy < (int)cc.size(1); ++iy)
			for (int iz = 0; iz < (int)cc.size(2); ++iz)
				cc(ix, iy, iz) = (*uc)(ix, iy, iz) - ucsave(ix, iy, iz);

	ucsave.deallocate();

	if (m_bperiodic && ilevel <= m_ilevelmin)
		make_periodic(&cc);

	m_gridop.prolong_add(cc, *uf);

	//... interpolate and apply coarse-fine boundary conditions on fine level
	if (m_bperiodic && ilevel <= m_ilevelmin)
		make_periodic(uf);
	else if (!m_bperiodic)
		setBC(ilevel);

	//... do smoothing sweeps with specified solver
	for (unsigned i = 0; i < m_npostsmooth; ++i)
	{

		if (ilevel > m_ilevelmin)
			interp().interp_coarse_fine(ilevel, *uc, *uf);

		if (m_smoother == opt::sm_gauss_seidel)
			GaussSeidel(h, uf, ff);

		else if (m_smoother == opt::sm_jacobi)
			Jacobi(h, uf, ff);

		else if (m_smoother == opt::sm_sor)
			SOR(h, uf, ff);

		if (m_bperiodic && ilevel <= m_ilevelmin)
			make_periodic(uf);
	}
}

template <class S, class I, class O, typename scalar_t>
double solver<S, I, O, scalar_t>::compute_error(const MeshvarBnd<scalar_t> &u, const MeshvarBnd<scalar_t> &f, int ilevel)
{
	int
			nx = u.size(0),
			ny = u.size(1),
			nz = u.size(2);

	double err = 0.0; //, err2 = 0.0;
	size_t count = 0;

	double h = 1.0 / (1ul << ilevel), h2 = h * h;

#pragma omp parallel for reduction(+ : err, count) collapse(3)
	for (int ix = 0; ix < nx; ++ix)
		for (int iy = 0; iy < ny; ++iy)
			for (int iz = 0; iz < nz; ++iz)
				if (true) // fabs(unew(ix,iy,iz)) > 0.0 )//&& u(ix,iy,iz) != unew(ix,iy,iz) )
				{
					// err += fabs(1.0 - (double)u(ix,iy,iz)/(double)unew(ix,iy,iz));
					/*err += fabs(((double)m_scheme.apply( u, ix, iy, iz )/h2 + (double)(f(ix,iy,iz)) ));
						err2 += fabs((double)f(ix,iy,iz));*/

					err += fabs((double)m_scheme.apply(u, ix, iy, iz) / h2 / (double)(f(ix, iy, iz)) + 1.0);
					++count;
				}

	if (count != 0)
		err /= count;

	return err;
}

template <class S, class I, class O, typename scalar_t>
double solver<S, I, O, scalar_t>::compute_error(const GridHierarchy<scalar_t> &uh, const GridHierarchy<scalar_t> &fh, bool verbose)
{
	double maxerr = 0.0;

	for (unsigned ilevel = uh.levelmin(); ilevel <= uh.levelmax(); ++ilevel)
	{
		int
				nx = uh.get_grid(ilevel)->size(0),
				ny = uh.get_grid(ilevel)->size(1),
				nz = uh.get_grid(ilevel)->size(2);

		double err = 0.0, mean_res = 0.0;
		size_t count = 0;

		double h = 1.0 / (1ul << ilevel), h2 = h * h;

		#pragma omp parallel for reduction(+ : err, count, mean_res) collapse(3)
		for (int ix = 0; ix < nx; ++ix)
			for (int iy = 0; iy < ny; ++iy)
				for (int iz = 0; iz < nz; ++iz)
				{
					double res = (double)m_scheme.apply(*uh.get_grid(ilevel), ix, iy, iz) + h2 * (double)((*fh.get_grid(ilevel))(ix, iy, iz));
					double val = (*uh.get_grid(ilevel))(ix, iy, iz);

					if (fabs(val) > 0.0)
					{
						err += fabs(res / val);
						mean_res += fabs(res);
						++count;
					}
				}

		if (count != 0)
		{
			err /= count;
			mean_res /= count;
		}
		if (verbose)
			std::cout << "      Level " << std::setw(6) << ilevel << ",   Error = " << err << std::endl;

		music::dlog.Print("[mg]      level %3d,  residual %g,  rel. error %g", ilevel, mean_res, err);

		maxerr = std::max(maxerr, err);
	}
	return maxerr;
}

template <class S, class I, class O, typename scalar_t>
double solver<S, I, O, scalar_t>::compute_RMS_resid(const GridHierarchy<scalar_t> &uh, const GridHierarchy<scalar_t> &fh, bool verbose)
{
	if (m_is_ini)
		m_residu_ini.assign(uh.levelmax() + 1, 0.0);

	double maxerr = 0.0;

	for (unsigned ilevel = uh.levelmin(); ilevel <= uh.levelmax(); ++ilevel)
	{
		int
				nx = uh.get_grid(ilevel)->size(0),
				ny = uh.get_grid(ilevel)->size(1),
				nz = uh.get_grid(ilevel)->size(2);

		double h = 1.0 / (1 << ilevel), h2 = h * h;
		double sum = 0.0, sumd2 = 0.0, rhs_sum = 0.0, residual_sum = 0.0;
		size_t count = 0;

		#pragma omp parallel for reduction(+ : sum, sumd2, count, rhs_sum, residual_sum)
		for (int ix = 0; ix < nx; ++ix)
			for (int iy = 0; iy < ny; ++iy)
				for (int iz = 0; iz < nz; ++iz)
				{
					// Covered coarse cells carry the FAS correction equation,
					// not independent physical source equations. Converge the
					// requested relative tolerance on actual rectangular leaves.
					if(ilevel<uh.levelmax()){
						const auto *child=uh.get_grid(ilevel+1);
						const int ox=child->offset(0),oy=child->offset(1),oz=child->offset(2);
						if(ix>=ox&&ix<ox+int(child->size(0))/2&&
						   iy>=oy&&iy<oy+int(child->size(1))/2&&
						   iz>=oz&&iz<oz+int(child->size(2))/2)continue;
					}
					double d = (double)(*fh.get_grid(ilevel))(ix, iy, iz);
					sumd2 += d * d;

					double r = ((double)m_scheme.apply(*uh.get_grid(ilevel), ix, iy, iz) / h2 + (double)(*fh.get_grid(ilevel))(ix, iy, iz));
					rhs_sum += d;
					residual_sum += r;
					sum += r * r;

					++count;
				}

		if(!count)continue;
		if (m_is_ini)
			m_residu_ini[ilevel] = sqrt(sum) / count;

		double err_abs = sqrt(sum / count);
		double err_rel = sumd2>0?err_abs/sqrt(sumd2/count):(sum==0?0.0:std::numeric_limits<double>::infinity());

		if (verbose)
		{
			std::cout << "      Level " << std::setw(6) << ilevel << ", RMS residual/source = " << err_rel
			          << ", source mean = " << rhs_sum / count << ", residual mean = " << residual_sum / count << std::endl;
			// Failure diagnostics only: distinguish a large arbitrary potential
			// gauge from the physical curvature being resolved. This does not
			// alter the residual, the field or the convergence criterion.
			long double potential_sum=0.,potential_square=0.,spacing_square=0.;
			for(int ix=0;ix<nx;++ix)for(int iy=0;iy<ny;++iy)for(int iz=0;iz<nz;++iz){
				const double value=(*uh.get_grid(ilevel))(ix,iy,iz);
				const long double spacing=std::nextafter(value,std::numeric_limits<double>::infinity())-value;
				potential_sum+=value;potential_square+=(long double)value*value;spacing_square+=spacing*spacing;
			}
			const long double cells=(long double)nx*ny*nz;
			std::cout << "        potential mean=" << (double)(potential_sum/cells)
			          << ", RMS=" << (double)std::sqrt(potential_square/cells)
			          << ", binary64 spacing RMS/h2/sourceRMS="
			          << (sumd2>0?(double)(std::sqrt(spacing_square/cells)/h2/std::sqrt(sumd2/count)):0.) << std::endl;
		}

		music::dlog.Print("[mg]      level %3d,  rms residual %g,  rel. error %g", ilevel, err_abs, err_rel);

		if (err_rel > maxerr)
			maxerr = err_rel;
	}

	if (m_is_ini)
		m_is_ini = false;

	return maxerr;
}

template <class S, class I, class O, typename scalar_t>
double solver<S, I, O, scalar_t>::solve(GridHierarchy<scalar_t> &uh, double acc, double h, bool verbose)
{

	double err, maxerr = 1e30;
	unsigned niter = 0;

	bool fullverbose = false;

	m_pu = &uh;

	// err = compute_RMS_resid( *m_pu, *m_pf, fullverbose );

	//... iterate ...//
	while (true)
	{

		music::ulog.Print("Performing multi-grid V-cycle...");
		twoGrid(uh.levelmax());

		// err = compute_RMS_resid( *m_pu, *m_pf, fullverbose );
		// Preserve the existing potential-normalized criterion and additionally
		// require the actual physical-leaf source equations to meet the same
		// requested tolerance. The RMS diagnostic must participate in stopping.
		err = std::max(compute_error(*m_pu, *m_pf, fullverbose),
		               compute_RMS_resid(*m_pu, *m_pf, fullverbose));
		++niter;

		if (fullverbose)
		{
			music::ulog.Print("  multigrid iteration %3d, maximum RMS residual = %g", niter, err);
			std::cout << "   - Step No. " << std::setw(3) << niter << ", Max Err = " << err << std::endl;
			std::cout << "     ---------------------------------------------------\n";
		}

		if (err < maxerr)
			maxerr = err;

		if ((niter > 1) && ((err < acc) || (niter >= maximum_iterations)))
			break;
	}

	if (err > acc)
	{
		std::cout << "Error : no convergence in Poisson solver" << std::endl;
		music::elog.Print("No convergence in Poisson solver, final error: %g.", err);
		compute_RMS_resid(*m_pu, *m_pf, true);
	}
	else if (verbose)
	{
		std::cout << " - Converged in " << niter << " steps to " << maxerr << std::endl;
		music::ulog.Print("Poisson solver converged to max. error of %g in %d steps.", err, niter);
	}

	//.. make sure that the RHS does not contain the FAS corrections any more
	for (int i = m_pf->levelmax(); i > 0; --i)
		m_gridop.restrict(*m_pf->get_grid(i), *m_pf->get_grid(i - 1));

	return err;
}

// TODO: this only works for 2nd order! (but actually not needed)
template <class S, class I, class O, typename scalar_t>
void solver<S, I, O, scalar_t>::setBC(unsigned ilevel)
{
	//... set only on level before additional refinement starts
	if (ilevel == m_ilevelmin)
	{
		MeshvarBnd<scalar_t> *u = m_pu->get_grid(ilevel);
		int
				nx = u->size(0),
				ny = u->size(1),
				nz = u->size(2);

		for (int iy = 0; iy < ny; ++iy)
			for (int iz = 0; iz < nz; ++iz)
			{
				(*u)(-1, iy, iz) = 2.0 * (*m_pubnd)(-1, iy, iz) - (*u)(0, iy, iz);
				(*u)(nx, iy, iz) = 2.0 * (*m_pubnd)(nx, iy, iz) - (*u)(nx - 1, iy, iz);
				;
			}

		for (int ix = 0; ix < nx; ++ix)
			for (int iz = 0; iz < nz; ++iz)
			{
				(*u)(ix, -1, iz) = 2.0 * (*m_pubnd)(ix, -1, iz) - (*u)(ix, 0, iz);
				(*u)(ix, ny, iz) = 2.0 * (*m_pubnd)(ix, ny, iz) - (*u)(ix, ny - 1, iz);
			}

		for (int ix = 0; ix < nx; ++ix)
			for (int iy = 0; iy < ny; ++iy)
			{
				(*u)(ix, iy, -1) = 2.0 * (*m_pubnd)(ix, iy, -1) - (*u)(ix, iy, 0);
				(*u)(ix, iy, nz) = 2.0 * (*m_pubnd)(ix, iy, nz) - (*u)(ix, iy, nz - 1);
			}
	}
}

//... enforce periodic boundary conditions
template <class S, class I, class O, typename scalar_t>
void solver<S, I, O, scalar_t>::make_periodic(MeshvarBnd<scalar_t> *u)
{

	int
			nx = u->size(0),
			ny = u->size(1),
			nz = u->size(2);
	int nb = u->m_nbnd;

	// if( u->offset(0) == 0 )
	for (int iy = -nb; iy < ny + nb; ++iy)
		for (int iz = -nb; iz < nz + nb; ++iz)
		{
			int iiy((iy + ny) % ny), iiz((iz + nz) % nz);

			for (int i = -nb; i < 0; ++i)
			{
				(*u)(i, iy, iz) = (*u)(nx + i, iiy, iiz);
				(*u)(nx - 1 - i, iy, iz) = (*u)(-1 - i, iiy, iiz);
			}
		}

	// if( u->offset(1) == 0 )
	for (int ix = -nb; ix < nx + nb; ++ix)
		for (int iz = -nb; iz < nz + nb; ++iz)
		{
			int iix((ix + nx) % nx), iiz((iz + nz) % nz);

			for (int i = -nb; i < 0; ++i)
			{
				(*u)(ix, i, iz) = (*u)(iix, ny + i, iiz);
				(*u)(ix, ny - 1 - i, iz) = (*u)(iix, -1 - i, iiz);
			}
		}

	// if( u->offset(2) == 0 )
	for (int ix = -nb; ix < nx + nb; ++ix)
		for (int iy = -nb; iy < ny + nb; ++iy)
		{
			int iix((ix + nx) % nx), iiy((iy + ny) % ny);

			for (int i = -nb; i < 0; ++i)
			{
				(*u)(ix, iy, i) = (*u)(iix, iiy, nz + i);
				(*u)(ix, iy, nz - 1 - i) = (*u)(iix, iiy, -1 - i);
			}
		}
}

END_MULTIGRID_NAMESPACE
