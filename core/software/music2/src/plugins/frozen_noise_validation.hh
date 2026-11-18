#pragma once

// Validate only the band which MUSIC embeds without independent fine noise.
// Meyer transition and Nyquist modes are not exact coarse-field constraints.
template<class Noise>
static void validate_frozen_low_modes(Noise &coarse, Noise &fine)
{
  const unsigned n=coarse.res_, nf=fine.res_;
  if(nf!=2*n) throw std::runtime_error("Frozen noise hierarchy must refine by two");
  const size_t cs=size_t(n)*n*(n+2), fs=size_t(nf)*nf*(nf+2);
  std::vector<real_t> c(cs), f(fs);
  for(unsigned i=0;i<n;i++) for(unsigned j=0;j<n;j++) for(unsigned k=0;k<n;k++)
    c[(size_t(i)*n+j)*(n+2)+k]=coarse(i,j,k);
  for(unsigned i=0;i<nf;i++) for(unsigned j=0;j<nf;j++) for(unsigned k=0;k<nf;k++)
    f[(size_t(i)*nf+j)*(nf+2)+k]=fine(i,j,k);
  auto cc=reinterpret_cast<complex_t*>(c.data());
  auto fc=reinterpret_cast<complex_t*>(f.data());
  auto pc=FFTW_API(plan_dft_r2c_3d)(n,n,n,c.data(),cc,FFTW_ESTIMATE);
  auto pf=FFTW_API(plan_dft_r2c_3d)(nf,nf,nf,f.data(),fc,FFTW_ESTIMATE);
  if(!pc || !pf) throw std::runtime_error("Frozen noise FFT planning failed");
  FFTW_API(execute)(pc); FFTW_API(execute)(pf);
  long double error=0, norm=0;
  for(unsigned i=0;i<n;i++) for(unsigned j=0;j<n;j++) for(unsigned k=0;k<=n/2;k++) {
    const int kx=i<=n/2?int(i):int(i)-int(n), ky=j<=n/2?int(j):int(j)-int(n);
    if(6*std::abs(kx)>=int(n) || 6*std::abs(ky)>=int(n) || 6*k>=n) continue;
    const unsigned ii=kx<0?unsigned(int(nf)+kx):i, jj=ky<0?unsigned(int(nf)+ky):j;
    const size_t ci=(size_t(i)*n+j)*(n/2+1)+k, fi=(size_t(ii)*nf+jj)*(nf/2+1)+k;
    std::complex<double> a(RE(cc[ci]),IM(cc[ci])), b(RE(fc[fi]),IM(fc[fi]));
    const double phase=0.5*M_PI*(kx+ky+int(k))/n;
    b*=std::complex<double>(cos(phase),sin(phase))/sqrt(8.0);
    // Normalize roundoff against the complete coarse field, including DC.
    error+=std::norm(b-a); norm+=std::norm(a);
  }
  FFTW_API(destroy_plan)(pc); FFTW_API(destroy_plan)(pf);
  if(!std::isfinite(double(error)) || !std::isfinite(double(norm)) || norm<=0 || error>norm*1e-20L)
    throw std::runtime_error("Frozen fine noise disagrees with embedded coarse low modes");
  music::ilog.Print("Frozen level %u low-band relative residual = %.17g",unsigned(log2(nf)),sqrt(double(error/norm)));
}

