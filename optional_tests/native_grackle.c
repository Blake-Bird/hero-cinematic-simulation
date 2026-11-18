/* Optional atomic H/He cooling control using the installed Grackle C API.
 * 32 independent constant-density cells; no gravity, UV background, metals,
 * molecules, dust, transport or particle evolution. CSV units are cgs.
 */
#include <grackle.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define N 32

static int write_epoch(FILE *out, int epoch, double elapsed,
                       chemistry_data *chemistry, chemistry_data_storage *rates,
                       code_units *units, grackle_field_data *fields)
{
    gr_float temperature[N], cooling[N];
    if (!local_calculate_temperature(chemistry, rates, units, fields, temperature)
        || !local_calculate_cooling_time(chemistry, rates, units, fields, cooling))
        return 0;
    for (int i = 0; i < N; ++i) {
        fprintf(out, "%d,%d,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g\n",
                epoch, i, elapsed * units->time_units,
                fields->density[i] * units->density_units,
                fields->internal_energy[i] * pow(units->velocity_units, 2),
                (double)temperature[i], (double)cooling[i] * units->time_units,
                fields->HI_density[i] / fields->density[i],
                fields->HII_density[i] / fields->density[i],
                fields->HeI_density[i] / fields->density[i],
                fields->HeII_density[i] / fields->density[i],
                fields->HeIII_density[i] / fields->density[i],
                fields->e_density[i] / fields->density[i]);
    }
    return !ferror(out);
}

int main(int argc, char **argv)
{
    if (argc != 2 || gr_check_consistency() != GR_SUCCESS) return EXIT_FAILURE;
    grackle_verbose = 0;
    code_units units = {0};
    units.comoving_coordinates = 0;
    units.density_units = 1.67e-24;
    units.length_units = 1e18;
    units.time_units = 1e12;
    units.a_units = units.a_value = 1.;
    set_velocity_units(&units);
    chemistry_data chemistry;
    if (!set_default_chemistry_parameters(&chemistry)) return EXIT_FAILURE;
    chemistry.use_grackle = 1;
    chemistry.with_radiative_cooling = 1;
    chemistry.primordial_chemistry = 1;
    chemistry.metal_cooling = 0;
    chemistry.UVbackground = 0;
    chemistry.dust_chemistry = 0;
    chemistry.HydrogenFractionByMass = .76;
    chemistry_data_storage rates;
    if (!local_initialize_chemistry_data(&chemistry, &rates, &units)) return EXIT_FAILURE;
    grackle_field_data fields;
    gr_initialize_field_data(&fields);
    int dimensions[3] = {N, 1, 1}, start[3] = {0, 0, 0}, end[3] = {N-1, 0, 0};
    fields.grid_rank = 3;
    fields.grid_dimension = dimensions;
    fields.grid_start = start;
    fields.grid_end = end;
    fields.grid_dx = 0.;
    gr_float density[N], energy[N], vx[N], vy[N], vz[N];
    gr_float hi[N], hii[N], hei[N], heii[N], heiii[N], electron[N];
    fields.density = density;
    fields.internal_energy = energy;
    fields.x_velocity = vx; fields.y_velocity = vy; fields.z_velocity = vz;
    fields.HI_density = hi; fields.HII_density = hii;
    fields.HeI_density = hei; fields.HeII_density = heii; fields.HeIII_density = heiii;
    fields.e_density = electron;
    for (int i = 0; i < N; ++i) {
        density[i] = pow(10., -1. + 2.*i/(N-1));
        energy[i] = 20.;
        hi[i] = .759 * density[i]; hii[i] = .001 * density[i];
        hei[i] = .24 * density[i]; heii[i] = heiii[i] = 0.;
        electron[i] = hii[i];
        vx[i] = vy[i] = vz[i] = 0.;
    }
    FILE *out = fopen(argv[1], "w");
    if (!out) return EXIT_FAILURE;
    fprintf(out, "epoch,cell,time_s,density_g_cm3,specific_energy_erg_g,temperature_K,cooling_time_s,HI_mass_fraction,HII_mass_fraction,HeI_mass_fraction,HeII_mass_fraction,HeIII_mass_fraction,electron_hydrogen_mass_fraction\n");
    const double dt = .1;
    int success = write_epoch(out, 0, 0., &chemistry, &rates, &units, &fields)
        && local_solve_chemistry(&chemistry, &rates, &units, &fields, dt)
        && write_epoch(out, 1, dt, &chemistry, &rates, &units, &fields);
    if (fclose(out)) success = 0;
    if (!local_free_chemistry_data(&chemistry, &rates)) success = 0;
    return success ? EXIT_SUCCESS : EXIT_FAILURE;
}
