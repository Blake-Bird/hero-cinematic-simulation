# Native elastic SIDM source

The source is selected by `RESEARCH_SIDM_REFERENCE` together with
`FORCE_EQUAL_TIMESTEPS`. The historical compile option also selects the
distributed DM production implementation. The isolated all-pair route remains
a bounded, single-rank fine-DM control backend.

`src/sidm/sidm.h`, `sidm.cc` and `sidm_cross_sections.cc` retain differential
tables, verify their actual SHA256 bytes, evaluate rates and provide normalized
conditional CDF sampling and disjoint angular transport moments. Differential
table values have units cm²/g/sr, cover every cosine in [-1,1], and interpolate
linearly in cosine and logarithmic speed. Explicit error, constant or zero
policies govern both velocity boundaries. Python `ElasticSIDMModel` records
total, ordinary transfer, modified identical-particle transfer and viscosity
normalizations separately. Screened Rutherford total normalization is explicitly
the zero-velocity total; other supplied moments specify their per-speed meaning.

`src/sidm/sidm_neighbors.cc` builds a dedicated DM octree over exchanged compact
support ghosts. It inherits GADGET-4 `basenode` and uses its threaded
`nextnode`/`sibling` subtree topology. The census contains native particle types
1 and 2 (fine and coarse DM); gas SPH neighbors are not reused. Spatial exchange
is retained as the MPI ghost producer. Overlap integrates normalized proper
kernels, and rates use actual physical relative velocities and proper elapsed
time. Cosmological time integrates da/[a H(a)], rather than interpreting Δln(a)
as physical seconds.

Rare scattering uses P=Gamma*dt with bounded pre-draw source substeps. Fine
particle scattering weights must be exactly equal. Updates preserve the
equal-weight center of mass and relative-speed norm before native velocity
storage roundoff. The runtime measures that roundoff and enforces the declared
momentum and energy bounds. Frequent updates use the actual truncated first
and second angular moments in the published drag and transverse diffusion
operator. Hybrid steps perform small-angle half kick, disjoint large-angle
sampling, and a second small-angle half kick. A neglected-moment bound limits
the frequent approximation domain; independent physical convergence remains a
required scientific acceptance condition.

`sidm_random.h` uses pinned Random123 Philox4x64-10. Exact uint64 unordered pair,
operator interval and substep occupy counter lanes; run seed, draw purpose and
hybrid half-kick stage occupy key lanes. MPI rank and event sequence are absent.
Pair priorities use this generator independently of velocities. Owners choose
mutual minimum priorities, update at most one pair per endpoint per round, and
commit new endpoint states before the following round. Decomposition-independent
draws do not imply whole-trajectory bitwise equality across MPI counts.

The source kick runs after both gravitational half kicks and before the native
cooling/stellar/BH source operators. Gas, stars and BHs remain in the common
synchronized gravity/hydro step; only the fine/coarse DM subset enters SIDM.
This is a Lie source split, with a deliberately global timestep and its gravity
and source convergence requirement. An asynchronous production route is not
provided.

Fine/coarse pairs provide support and omitted-hazard diagnostics without an
unequal-resolution scattering update. Every fine particle must remain inside
the configured cumulative omitted-scattering and instantaneous coarse-support
budgets before the source advances. Region expansion/regeneration belongs to
the zoom workflow, and scientific apertures require its separate acceptance.

Snapshots publish `SIDMInteractionCount` and `SIDMOperatorSubstep` as uint64,
`SIDMExposure` as dimensionless cumulative microscopic angular-domain optical
depth, and `SIDMScatteringSupportLength` as proper kpc with explicit cgs unit
metadata. Counts refer to rare macro events. Frequent kicks are effective
transport updates and are not microscopic collision counts. Source state and
global operator counters persist in native restarts. Table path/hash and audit
population persist in the global snapshot/restart contract. Checkpoint
publication occurs only at complete synchronized source boundaries.

Immutable HDF5 event chunks can retain all particles, a declared hashed particle
ID population, or aggregate statistics only. Selection includes a pair when
either endpoint belongs to the selected population. Every applied update is
still included in aggregate counts and conservation residuals. Resource records
report actual retained differential-table vector payload rather than declaring
zero table memory for a tabulated model.

Pinned independent references are retained in
`../workflow-references/native-sidm-reference-sources.json`: OpenGadget3
7d90b03df29fa3c416b2b06c3fa6cc8c7821f1f1 (GPL-3.0-or-later), ChaNGa
39fc62a71382a5aa2c7b2b1355ec07d0bbd1ccbd (upstream license retained), and
Random123 9545ff6413f258be2f04c1d319d99aaef7521150 (BSD). The OpenGadget3
`research_differential_table.hpp` adapter exposes the same physical differential
table as dσ/dcos(theta)=2π dσ/dΩ; its rates, integration, transport and sampling
remain independent OpenGadget3 code. `core/sidm_validation.py` authors every
required control and retains matched input/convention bundles.

This coding session executed no simulations, compilers, tests, validators or
control campaigns. These source changes do not establish physical acceptance,
convergence, reference-engine agreement, or a successful build.
