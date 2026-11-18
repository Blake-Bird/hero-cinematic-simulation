// Wider scalar instantiation of the same native stencils and boundary rules.
// No FFT type, random stream or acceptance criterion is replaced here.
#pragma once
#include "general.hh"
#include "mesh.hh"
#include "mg_solver.hh"
static_assert(sizeof(__float128)>sizeof(double));
#undef __MG_OPERATORS_HH
#undef __FD_SCHEMES_HH
#undef __MG_INTERP_HH
namespace wide_native {
using real_t=__float128;
using meshvar_bnd=MeshvarBnd<real_t>;
#define double __float128
#include "mg_operators.hh"
#include "fd_schemes.hh"
#undef double
#include "mg_interp.hh"
}
