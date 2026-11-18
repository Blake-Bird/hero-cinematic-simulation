# Molecular chemistry source delivery

Grackle 3.4.1 source is retained at core/software/chemistry/grackle with its archive hash and license. No build, compiler, test, solver or validation command was executed during this delivery.

Network 0 uses table cooling, 1 transports atomic H/He, 2 adds HM/H2I/H2II, 3 adds DI/DII/HDI. Molecular networks require explicit initial species number fractions and residual ionization. H is normalized per protium nucleus, He species per He nucleus, and D species to the declared D/H number ratio. Electron pseudo-density is derived from charge balance.

Persistent molecular fields are mass-density fractions. Element H includes protium and D. Stellar mass deposition transports existing isotope/ion/molecule reservoirs as extensive quantities and adds neutral H/He, with no replenished D in the stellar yield model. Source solves check H/He/D nuclei and charge balance. Imported nonzero species without a recorded state version require explicit IC migration.

The hydro variable remains A_ref=(gamma_ref-1)u/rho_phys^(gamma_ref-1). Pressure uses the actual Grackle caloric gamma; the compression correction is dln(A_ref)=(gamma_pressure-gamma_ref)dln(rho_phys). Sound-speed gamma and temperature are taken from the pinned library. Thermodynamics are refreshed after density/drift/kick/source changes. This profile uses density-entropy SPH; pressure-entropy SPH is rejected rather than silently used with a changed EOS. SH03 effective-EOS gas retains its separate hot/cold closure and is masked from molecular single-phase temperature inference.

Snapshots expose molecular fractions, gamma pressure/sound, native temperature and molecular weight. Typed source state, hydro fields, global D/H and seed parameters are serialized in current restart layouts. The Python native census checks units, finite states, version/network consistency, nucleus balance and charge, and forwards actual EOS variables to primordial cloud diagnostics.

Example input: configs/research/primordial_chemistry.example.json. It is illustrative, not a prescribed primordial abundance model. Build recipe: configs/research/grackle_build.recipe.json. Actual library/model hashes and physical control receipts remain required. Successful compilation, source splitting accuracy, chemical accuracy and first-star resolution acceptance are unverified.
