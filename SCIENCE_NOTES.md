# Volume and fitting-function choices

Between 30 and 40 Mpc/h, the supplied choice is **40 Mpc/h**. It has 2.37 times
the volume of 30 Mpc/h. At equal halo number density, a populated mass bin's
Poisson fractional error would be about 0.65 times as large. That is an estimate
of counting noise, not a prediction of the measured error bars.

For 512³ particles and Ω_m=0.300, m_particle=Ω_m ρ_crit L³/N:

| L [comoving Mpc/h] | Volume [(Mpc/h)³] | Particle mass [Msun/h] | 1000 particles [Msun/h] |
|---|---:|---:|---:|
| 30 | 27,000 | 1.675×10⁷ | 1.675×10¹⁰ |
| 40 | 64,000 | 3.970×10⁷ | 3.970×10¹⁰ |
| 80 | 512,000 | 3.176×10⁸ | 3.176×10¹¹ |
| 100 | 1,000,000 | 6.204×10⁸ | 6.204×10¹¹ |

The 40 Mpc/h profile favors low/intermediate-mass halo work. It cannot guarantee
small errors for rare massive halos or at high redshift. For a broad high-mass
HMF comparison, an 80–100 Mpc/h pair or a volume ladder is worth planning, with
the corresponding loss of mass resolution at fixed N. Several independent
matched seeds and a resolution comparison are needed to characterize scatter
and establish a trusted mass interval.

Counting-only guidance is σ/N≈1/√N for populated bins: roughly 10% for 100 halos,
5% for 400, 1% for 10,000. This is the count **in each mass/redshift bin**, not the
particle count. The comparison exports per-bin counts so sparse bins are visible.
Individual HMF bands use central Garwood Poisson intervals. The ratio's displayed
independent-count approximation omits paired covariance and sample variance;
it is not a precision significance estimate. Shared phases help control the
comparison but do not remove finite-volume bias or establish covariance.

The production HMF applies a 1000-particle cut and quarter-dex bins whose lower
edge is rounded upward above that cut. At L=40, the first bin starts at
5.623×10¹⁰ Msun/h. Bins never straddle the cut and silently lose their low-mass
members. These settings are conservative selections, not a proof of convergence.
The 32³ profile uses a 32-particle cut solely to check the pipeline.

The provided six fitting recipes are Press–Schechter, Sheth–Tormen, Reed 2007,
Watson FOF, Jenkins and Warren. They are diagnostic CDM references; EDE results
are extrapolations, with each recipe's coordinate-range mask retained. Model
spread is not an observational/statistical error bar. These recipes require
**FOF b=0.2**; an M200c catalogue must not be substituted without a calibrated
conversion. SO fields and subhalo outputs remain available for other figures.

The two linear templates have identical H0=70, Ω_b=0.049, Ω_cdm=0.251,
A_s=2.1×10⁻⁹, n_s=0.965 and τ=0.054. EDE adds f_EDE=0.1,
log10(z_c)=3.5, θ_i=2.8 and n=3. This isolates EDE at fixed listed background and
primordial inputs. It is not a separately fitted best-fit-EDE versus best-fit-ΛCDM
comparison. Change the templates explicitly for that different question.

Sources: [Watson et al., halo mass function through the cosmic ages](https://arxiv.org/abs/1212.0095),
[Euclid HMF calibration and finite-volume effects](https://doi.org/10.1051/0004-6361/202244674),
[Slurm job submission](https://slurm.schedmd.com/sbatch.html),
[Slurm node inventory](https://slurm.schedmd.com/sinfo.html).
