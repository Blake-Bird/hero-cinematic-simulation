# Native stellar reference states for stream campaigns

The existing `python commands.py streams REQUEST.json --output DIRECTORY` campaign can derive a Gala or StreaMAX **static-host reference** from a previously selected native stellar cohort. This implementation has only been statically inspected; neither adapter nor the full handoff has been run for this change. This is not automatic host or progenitor selection and does not establish a scientific acceptance result.

First produce a `measured-native-stream-worldlines-v1` product for species 4 from a completed native run and a `native-encounter-frame-history-v1` conversion for that exact history. The conversion needs an explicitly specified origin, velocity, and, for periodic runs, supported recorded periodic images. It supplies proper positions and proper-position derivatives in fixed Cartesian axes. The declared origin must in fact correspond to the host model used later; matching a frame name alone cannot prove that physical condition. Select a bound progenitor cohort using independent catalogue/membership evidence before creating the history. An arbitrary debris cohort does not become a bound progenitor through this conversion.

In a `stream-campaign-v1` request, add this stage (paths are placeholders relative to the request file):

```json
{
  "stage_id": "native-state",
  "kind": "native_reference_state",
  "inputs": {
    "native_history": "path/to/native-history.json",
    "frame_history": "path/to/frame-history.json"
  },
  "parameters": {
    "host_frame": "chosen-host-fixed-axes",
    "epoch_index": 0,
    "minimum_stars": 32
  }
}
```

`host_frame` must equal the conversion's frame name. The result retains the exact native IDs in their original namespace, source bindings, saved epoch, mass-weighted position and velocity, total stellar mass, and 3D stellar half-mass radius. A Plummer scale is derived by matching that radius, not by fitting an equilibrium profile. The threshold is a user-declared minimum, not a convergence certificate. There is no inferred dark matter mass, gas mass, or progenitor binding. The stage refuses missing or mismatched native histories, frames, epochs, identities, insufficient stars, and unresolved radii.

To use it, add `"reference_state": {"stage": "native-state"}` to the `inputs` of an existing `independent_spray` or `streamamax_reference` stage and include `"depends_on": ["native-state"]`. Both stages need `parameters.host_frame` equal to the recorded frame and a distinct `parameters.tracer_namespace`. Continue supplying the normal backend-specific host model, integration controls, total tracer mass, and resource limits. Omit measured state fields from the backend controls:

- Gala: omit `mass_msun`, `scale_kpc`, `position_kpc`, and `velocity_kms`; use a **negative** `step_myr`, so upstream interprets the native state as the final progenitor state.
- StreaMAX: set `type_sat` to `Plummer`; omit `xv_f`, `params_sat`, and `m_f_sat`. The adapter supplies the measured state, Plummer approximation, and constant stellar mass (no inferred mass-loss history). Its underlying float32 and static analytical host limitations remain.

The tracer arrays have newly assigned IDs in a separate namespace and are not native GADGET stars. Their displayed tracer masses are controlled by the existing `total_tracer_mass_msun` parameter, not automatically interpreted as individual native stellar masses. The source native identities remain in the reference-state product. Without the optional host stage below, both reference models need a separately specified host in the same physical frame, and matching the name does not establish a host-potential fit or force residual. No evolving-host equivalence, convergence, or native stream recovery is established by either route. Record and validate host model domain and forces before interpreting comparisons. Run the complete campaign and independent controls only after validation is authorized.

## Optional measured NFW host

A `native_nfw_host` stage can consume an existing `native-halo-structure-v1`
product, its exact native stellar history, and the history's converted frame.
The structure must be native, have a resolved total-density NFW fit, and pass
its existing reconstructed-field controls. The stage checks the same run and
saved epoch, compares the fixed native host centre to the converted frame
origin, translates the density fit to scale mass `4 pi rho_s r_s^3`, and
compares its spherical force with the structure's withheld direct-force samples.
All withheld positions must lie within the declared density-fit domain. It
saves the vectors, errors and predeclared tolerance; a failed comparison remains
`FAIL` and cannot feed the adapter. For example, add this after preparing the
source products and freeze the tolerance from the study's error budget:

```json
{
  "stage_id": "host-nfw",
  "kind": "native_nfw_host",
  "inputs": {
    "host_structure": "path/to/native-halo-structure.json",
    "native_history": "path/to/native-history.json",
    "frame_history": "path/to/frame-history.json"
  },
  "parameters": {
    "epoch_index": 0,
    "force_relative_tolerance": 0.1,
    "force_floor_km2_s2_kpc": 1.0
  }
}
```

Those numeric error controls illustrate the schema only; they are not a
recommended science threshold. Add `"host_reference": {"stage": "host-nfw"}`
to the adapter inputs and include `host-nfw` in `depends_on`, alongside the
native-state stage. Omit `gala_potential` for Gala or `type_host` and
`params_host` in StreaMAX controls. The consumer requires the same frame,
source history and epoch as the progenitor and requires its measured position
to lie in the fit domain. Before generating tracers, it also evaluates the actual
Gala or float32 StreaMAX NFW force at the same withheld locations against the
same frozen direct-force budget. It stores force arrays on success; on failure,
the campaign retains `backend-force-failure.json` and
`backend-force-failure.npz` in the failed temporary stage directory. It exports
masks for output particles, and, for
StreaMAX, saved progenitor points outside the domain. Those masks do **not**
prove that the continuous orbit or all force evaluations were within the fit.

The force reference is the isolated, unsoftened force of the explicitly
selected finite source, not the periodic GADGET force; fitting a spherical
profile cannot generally capture disks, triaxiality, substructure, or time
variation. The backend comparison includes StreaMAX's float32 arithmetic and internal NFW
epsilon at these saved query locations only; independent numerical and orbit
controls remain necessary. A matching
origin does not establish that an accelerating frame is inertial. No GADGET,
Gala, StreaMAX, force, or scientific validation has been run for this bridge.

The native and converted-frame numeric dataset unit contracts, exact identity
namespace, and upstream source bindings are rechecked before reference-state
or host construction. Reuse requires unchanged recorded sources. All of these
checks are implementation descriptions and remain runtime-unverified.
