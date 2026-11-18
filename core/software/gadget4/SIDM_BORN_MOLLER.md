# Born Moller angular model

The public pinned angular model can select `kind: born_moller` with explicit
`velocity_scale_kms`, `regime: nonrelativistic_born`,
`spin: unpolarized_spin_half`, and
`normalization: rutherford_sigma0_including_identical_final_state_factor`.
Select `particle_convention: folded_identical` in the native SIDM configuration.
The native angular kind is 2. Isotropic (0) and Rutherford (1) retain their
existing normalization.

This model implements exchange interference for unpolarized identical spin-half
particles in the nonrelativistic Born approximation. With a=v^2/(2w^2),
D_plus=1+a(1-u), D_minus=1+a(1+u), and u=|cos(theta)|, its folded differential
cross section divided by sigma0 is

    1/2 [D_plus^-2 + D_minus^-2 - (D_plus D_minus)^-1].

The total divided by sigma0 is

    1/(1+2a) - log(1+2a)/(4a(1+a)),

with limit 1/2 at zero speed. The supplied native normalization is sigma0/m,
not the low-speed Moller total; no additional symmetry multiplier is applied.
The actual pair velocity determines both the rate and angular shape. The pair
hazard schedule retains the larger Rutherford sigma*v bound, which also bounds
this positive Moller cross section. Coarse-particle omitted-exposure guards use
that conservative upper bound.

Rare samples invert the analytic folded primitive with bounded bisection and
check the retained cosine against the requested CDF tolerance. Frequent and
hybrid updates use truncated first and second angular moments including the
interference term, with existing transfer-depth and neglected-moment guards.
Native logs and replay checks retain `angular_kind: born_moller`. The independent
Python reference exports an adaptive CDF grid with a curvature interpolation
bound, and the measured angular control accepts this explicit physical model.

These sources support the declared Born approximation; they do not calculate
nonperturbative quantum resonances or establish applicability of a particular
microscopic model. No simulations, builds, or acceptance controls were executed
when adding this implementation.

Physical formula reference: https://arxiv.org/html/2205.03392 (equations 2-3).
Hybrid algorithm reference: https://arxiv.org/html/2410.07175v2.
