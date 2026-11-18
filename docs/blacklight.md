# Optional Kerr and GRMHD image handoff

Research2 can prepare or invoke the separately built
[Blacklight](https://github.com/c-white/blacklight) Kerr ray-tracing and
radiative-transfer code on an independently supplied GRMHD snapshot, or its
analytic emission formula. The bridge is **new and has only been statically
inspected**. Blacklight, its build, the bridge, generated images and the
scientific handoff have not been executed or validated for this work. It does
not construct a horizon-scale flow from GADGET. The upstream [method
paper](https://doi.org/10.3847/1538-4365/ac77ef) describes the calculation.

The source is pinned to commit `a7c721529d45a767ae51b9b965ebb93483fd1995`,
last committed 2024-12-24 when inspected. It was fetched to the external runtime
for static review (about 1.1 MiB including shallow Git metadata). The upstream
[Unlicense](../configs/blacklight/LICENSE) and [pin](../configs/blacklight/upstream.json)
are retained here. Upstream's Makefile uses C++17 and OpenMP with GNU g++ or
supported Intel compilers. Platform compatibility, numerical behavior and the
current maintenance state must be checked on the actual machine; macOS Apple
Clang does not by itself supply GNU OpenMP. Do not substitute a binary from a
different revision without reviewing and updating the pin.

For a future authorized installation, with Git, GNU g++ with OpenMP, and make
installed, choose a **fresh external** location:

```sh
git clone https://github.com/c-white/blacklight.git /scratch/research2-runtime/blacklight
cd /scratch/research2-runtime/blacklight
git checkout --detach a7c721529d45a767ae51b9b965ebb93483fd1995
make -j 4
shasum -a 256 bin/blacklight
```

On Linux, `sha256sum` is an alternative to `shasum -a 256`. The checkout must
remain clean. The bridge records the exact binary hash and hashes the pinned
source files, but it **cannot prove** that the declared binary was built from
that checkout. Retain build logs, compiler version, flags and platform separately.
No build has been performed for this change.

Start from the appropriate upstream `input/example_simulation.input` or
`input/example_formula.input`, copy it to scratch, and explicitly configure all
physics, camera, integration and plasma settings for the actual problem. The
bridge currently requires one frequency, `output_format = npz`,
`image_light = true`, `adaptive_max_level = 0`, `render_num_images = 0`,
`ray_flat = false`, `fallback_nan = true`, all checkpoints disabled,
`simulation_multiple = false`, and `slow_light_on = false`. It rewrites only
`output_file` and the simulation snapshot path in a saved copy of the input.
Paths and values containing whitespace, `#` or `=` cannot be passed to the bridge. Predeclare scheduler-enforced memory, disk and wall limits; the bridge's
camera workspace estimate is not an RSS guarantee.

For simulation mode provide a separate provenance JSON with the following
shape, replacing all numbers, strings and the snapshot hash with actual values:

```json
{
  "schema": "blacklight-input-provenance-v1",
  "origin": "independent_grmhd",
  "description": "Source, run, snapshot epoch and reproducibility record",
  "metric": "Kerr",
  "spin": 0.5,
  "format": "athena",
  "coordinates": "sks",
  "mass_msun": 4000000.0,
  "density_unit_g_cm3": 1e-16,
  "snapshot_sha256": "replace-with-actual-sha256",
  "emission_assumptions": "Describe electron model, emission, absorption, normalization and their sources"
}
```

These fields must match `simulation_a`, `simulation_format`,
`simulation_coord`, `simulation_m_msun` and `simulation_rho_cgs` in the input.
Use a real compatible Athena++, AthenaK, iharm3D or harm3d snapshot; the file
hash does **not** independently prove its metric, spin, coordinate system,
units, origin or plasma validity. The bridge records those as user declarations.
For formula mode use `"origin": "analytic_illustration"`, `"metric": "Kerr"`,
`"spin"`, `"description"`, and `"mass_gravitational_cm"` matching
`formula_mass` (GM/c² in cm). Its disk/emission is an illustrative prescribed
model, not a measured GADGET or GRMHD prediction.

With the desired input and provenance prepared, and after replacing all paths
and the binary hash, a future operator can prepare a reviewable invocation:

```sh
cd /path/to/research2
python -m core.blacklight_bridge \
  --source /scratch/research2-runtime/blacklight \
  --binary /scratch/research2-runtime/blacklight/bin/blacklight \
  --binary-sha256 REPLACE_WITH_64_HEX_DIGITS \
  --input /scratch/blacklight-science/input.txt \
  --provenance /scratch/blacklight-science/provenance.json \
  --output /scratch/blacklight-science/prepared \
  --memory-gib 8 --wall-seconds 3600 \
  --display-max 1e-4 --display-stretch 5 --palette ember --prepare-only
```

After reviewing `prepared/handoff.json` and `prepared/blacklight.input`, invoke
the same command with a **different fresh output directory** and without
`--prepare-only` only when execution is authorized. The native NPZ intensity,
complete settings, input and source hashes, log, failed-ray finite count and
status are retained. The NPZ is the numerical intensity. `display.png` is an 8-bit illustrative color mapping with fixed user-selected intensity maximum, asinh stretch and palette; transparent pixels flag nonfinite or negative values, and saturation counts are recorded. The maximum above is an example only and must be selected in the source intensity units. Use the same settings for comparisons. A Blender scene or cinematic sequence is not yet automatically built. Failed rays remain nonfinite; completion
and `science_acceptance: NOT_ACCEPTED` are distinct. The declared single-epoch
inputs, radiation model and screen settings must be compared against independent
Kerr geodesic/redshift and radiative-transfer references and convergence controls
before scientific claims. A credible predictive image additionally requires
validated GRMHD provenance, unit normalization and emission physics.


### Portable Blender display

After an authorized successful calculation, a separate machine with Blender can
copy the complete handoff directory and build a packed display scene:

```sh
blender --background --python /path/to/research2/core/blacklight_scene.py -- \
  /scratch/blacklight-science/completed/handoff.json \
  --output /scratch/blacklight-science/blacklight.blend \
  --width 3840 --height 2160 --exposure 0 --image-scale 1
```

The input NPZ and `display.png` must be alongside `handoff.json`; their hashes
are checked before construction. The PNG is packed, and the scene stores the
full handoff, numerical and display hashes, builder hash, camera and display
settings. Finite zero intensity is black; transparent masked pixels reveal a
magenta background and are labeled unsupported. The PNG row order is flipped
from the array to follow upstream's lower-origin plotting convention. The
color ramps are prescribed false colors, not spectral colors. The orthographic
Blender camera views a single previously calculated screen, so camera movement
in Blender does not retrace Kerr rays. This creates a scene only; it does not
produce a dome, IMAX film, or a physically continuous ray-traced sequence.
The full input sources and NPZ must be retained for scientific reproduction.
Neither the scene nor this command has been run.
