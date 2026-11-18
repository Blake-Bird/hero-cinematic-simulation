/* Research2 delayed stellar sources. Original integration code; NuPyCEE
 * numerical inputs retain their BSD notice, Grackle retains its own license.
 * Runtime tables are compiled by stars.compile_stellar_return_model. */
#ifndef RESEARCH_STELLAR_RETURNS_H
#define RESEARCH_STELLAR_RETURNS_H
#include "gadgetconfig.h"
#include "../data/research_source_state.h"
#ifdef STELLAR_EVOLUTION
bool research_stellar_set_composition(research_source_state &state, double metallicity);
double research_stellar_elapsed_years(double before,double after);
#ifdef RESEARCH_GRACKLE
class simparticles;
struct particle_data;
struct sph_particle_data;
void research_grackle_initialize(void);
void research_grackle_initialize_particle(simparticles *particles,int index);
double research_grackle_cool_particle(simparticles *particles,int index);
void research_grackle_refresh_thermodynamics(simparticles *particles,int index);
void research_grackle_refresh_thermodynamics(particle_data *particle,sph_particle_data *gas);
#endif
#endif
#endif
