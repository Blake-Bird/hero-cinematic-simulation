#ifndef _UNIVERSAL_CONSTANTS_H_
#define _UNIVERSAL_CONSTANTS_H_

/* G=6.67430e-8 cgs, adopted Msun=1.98847e33 g,
 * Mpc=3.0856775814913673e24 cm. Match physical_constants.py. */
#define Gc (4.3010473293148021e-9) // G * (Msun / Mpc) in (km/s)^2
#define CRITICAL_DENSITY 2.7752823482167468e11 // 3H100^2/8piG in (Msun / h) / (Mpc / h)^3
#define VMAX_CONST 6.5582370567972016e-5 //sqrt(G*(Msun/h)/(Mpc/h)) in km/s
#define RMAX_TO_RS 2.1626        /* Rvmax / Rs */
#define RS_CONSTANT 0.216216595 /* ln(1+2.1626)/2.1626 - 1/(1+2.1626) */

#define HUBBLE_TIME_CONVERSION 9.777922216807892e9 /* (100 km/s/Mpc)^-1 in Julian years: Mpc=3.0856775814913673e19 km, year=31557600 s */

#endif /* _UNIVERSAL_CONSTANTS_H_ */
