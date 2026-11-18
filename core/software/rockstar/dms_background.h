/* DarkMatterSim background adapter for Rockstar/Consistent Trees.
 * Input: a, E(a)=H(a)/H0 (additional GADGET growth columns are ignored).
 * Interpolation is linear in log(a), log(E), as in patched GADGET. Cosmic
 * time differences integrate each power-law segment analytically:
 * H0 dt = da/(a E); for E=E_i (a/a_i)^p, the segment integral is
 * -expm1[-p log(a/a_i)]/(p E_i), with the p=0 logarithmic limit.
 * The zero of time is today. No extrapolation outside supplied epochs.
 * Header-only static storage lets the pinned native build remain unchanged.
 */
#ifndef DMS_BACKGROUND_H
#define DMS_BACKGROUND_H
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern char *BACKGROUND_TABLE;
static size_t dms_bg_count = 0;
static double *dms_bg_a = NULL, *dms_bg_e = NULL, *dms_bg_time = NULL;
static double *dms_bg_distance = NULL;
static int dms_bg_ready = 0;

static int dms_background_enabled(void) {
  return BACKGROUND_TABLE && BACKGROUND_TABLE[0] && strcmp(BACKGROUND_TABLE, "none");
}

static void dms_bg_error(const char *reason) {
  fprintf(stderr, "[Error] DarkMatterSim background %s: %s\n", BACKGROUND_TABLE, reason);
  exit(EXIT_FAILURE);
}

static double dms_bg_integral(double a0, double a1, double e0, double e1, double a) {
  double slope = log(e1/e0)/log(a1/a0), distance = log(a/a0);
  return fabs(slope*distance) < 1.e-10
    ? distance/e0 * (1. - 0.5*slope*distance + slope*slope*distance*distance/6.)
    : -expm1(-slope*distance)/(slope*e0);
}

static void dms_bg_load_impl(void) {
  FILE *file = fopen(BACKGROUND_TABLE, "r");
  if(!file) dms_bg_error("cannot open table");
  char line[2048];
  size_t capacity = 0;
  while(fgets(line, sizeof(line), file)) {
    char *text = line;
    while(*text == ' ' || *text == '\t' || *text == '\r' || *text == '\n') text++;
    if(!*text || *text == '#') continue;
    double a, e;
    if(sscanf(text, "%lf %lf", &a, &e) != 2 || !isfinite(a) || !isfinite(e) || a <= 0 || e <= 0)
      dms_bg_error("rows must contain finite positive a and E(a)");
    if(dms_bg_count && a <= dms_bg_a[dms_bg_count-1]) dms_bg_error("epochs must strictly increase");
    if(dms_bg_count == capacity) {
      capacity = capacity ? capacity*2 : 1024;
      double *new_a = realloc(dms_bg_a, capacity*sizeof(double));
      if(!new_a) dms_bg_error("allocation failed");
      dms_bg_a = new_a;
      double *new_e = realloc(dms_bg_e, capacity*sizeof(double));
      if(!new_e) dms_bg_error("allocation failed");
      dms_bg_e = new_e;
    }
    dms_bg_a[dms_bg_count] = a;
    dms_bg_e[dms_bg_count++] = e;
  }
  if(ferror(file)) dms_bg_error("reading table failed");
  fclose(file);
  if(dms_bg_count < 2 || fabs(dms_bg_a[dms_bg_count-1]-1.) > 64*DBL_EPSILON ||
     fabs(dms_bg_e[dms_bg_count-1]-1.) > 1.e-6)
    dms_bg_error("table must end at a=1, E=1 with at least two rows");
  dms_bg_time = malloc(dms_bg_count*sizeof(double));
  dms_bg_distance = malloc(dms_bg_count*sizeof(double));
  if(!dms_bg_time || !dms_bg_distance) dms_bg_error("allocation failed");
  dms_bg_time[0] = 0.;
  dms_bg_distance[0] = 0.;
  for(size_t i=1; i<dms_bg_count; i++) {
    dms_bg_time[i] = dms_bg_time[i-1] + dms_bg_integral(dms_bg_a[i-1], dms_bg_a[i],
                                                    dms_bg_e[i-1], dms_bg_e[i], dms_bg_a[i]);
    /* chi H0/c = integral da/(a^2 E). Apply the same exact segment integral
       to E_effective=a E, whose power-law slope is increased by one. */
    dms_bg_distance[i] = dms_bg_distance[i-1] + dms_bg_integral(dms_bg_a[i-1], dms_bg_a[i],
                   dms_bg_a[i-1]*dms_bg_e[i-1], dms_bg_a[i]*dms_bg_e[i], dms_bg_a[i]);
  }
}

static void dms_bg_load(void) {
  /* Native tree mass conversions can start inside an OpenMP region. Protect
     lazy loading and publication; no thread may see half-populated arrays. */
#ifdef _OPENMP
#pragma omp critical(dms_background_load)
#endif
  {
    if(!dms_bg_ready) {
      dms_bg_load_impl();
      dms_bg_ready = 1;
    }
  }
}

static size_t dms_bg_index(double *a) {
  dms_bg_load();
  double minimum = dms_bg_a[0], maximum = dms_bg_a[dms_bg_count-1];
  if(!isfinite(*a) || *a < minimum*(1.-64*DBL_EPSILON) || *a > maximum*(1.+64*DBL_EPSILON))
    dms_bg_error("requested epoch lies outside table; extend the physical background input");
  if(*a < minimum) *a = minimum;
  if(*a > maximum) *a = maximum;
  size_t low = 0, high = dms_bg_count-1;
  while(high-low > 1) {
    size_t middle = low+(high-low)/2;
    if(dms_bg_a[middle] > *a) high = middle; else low = middle;
  }
  return low;
}

static double dms_background_E(double a) {
  size_t i = dms_bg_index(&a);
  double fraction = log(a/dms_bg_a[i])/log(dms_bg_a[i+1]/dms_bg_a[i]);
  return exp(log(dms_bg_e[i]) + fraction*log(dms_bg_e[i+1]/dms_bg_e[i]));
}

static double dms_background_time(double a) {
  size_t i = dms_bg_index(&a);
  return dms_bg_time[i] + dms_bg_integral(dms_bg_a[i], dms_bg_a[i+1],
                                        dms_bg_e[i], dms_bg_e[i+1], a)
         - dms_bg_time[dms_bg_count-1];
}

static double dms_background_distance_to_today(double a) {
  size_t i = dms_bg_index(&a);
  return dms_bg_distance[dms_bg_count-1] - dms_bg_distance[i]
         - dms_bg_integral(dms_bg_a[i], dms_bg_a[i+1],
                          dms_bg_a[i]*dms_bg_e[i], dms_bg_a[i+1]*dms_bg_e[i+1], a);
}
#endif
