// Define RESEARCH_INTEGRATOR_FLOAT(name,value) before including this catalog.
RESEARCH_INTEGRATOR_FLOAT("Vel[0]",p.Vel[0])
RESEARCH_INTEGRATOR_FLOAT("Vel[1]",p.Vel[1])
RESEARCH_INTEGRATOR_FLOAT("Vel[2]",p.Vel[2])
RESEARCH_INTEGRATOR_FLOAT("GravAccel.da[0]",p.GravAccel.da[0])
RESEARCH_INTEGRATOR_FLOAT("GravAccel.da[1]",p.GravAccel.da[1])
RESEARCH_INTEGRATOR_FLOAT("GravAccel.da[2]",p.GravAccel.da[2])
RESEARCH_INTEGRATOR_FLOAT("OldAcc",p.OldAcc)
RESEARCH_INTEGRATOR_FLOAT("Mass",p.getMass())
#if defined(PMGRID) && defined(PERIODIC) && !defined(TREEPM_NOTIMESPLIT)
RESEARCH_INTEGRATOR_FLOAT("GravPM[0]",p.GravPM[0])
#endif
#if defined(PMGRID) && defined(PERIODIC) && !defined(TREEPM_NOTIMESPLIT)
RESEARCH_INTEGRATOR_FLOAT("GravPM[1]",p.GravPM[1])
#endif
#if defined(PMGRID) && defined(PERIODIC) && !defined(TREEPM_NOTIMESPLIT)
RESEARCH_INTEGRATOR_FLOAT("GravPM[2]",p.GravPM[2])
#endif
#if defined(EVALPOTENTIAL) || defined(OUTPUT_POTENTIAL)
RESEARCH_INTEGRATOR_FLOAT("Potential",p.Potential)
#endif
#if (defined(EVALPOTENTIAL) || defined(OUTPUT_POTENTIAL)) && defined(PMGRID)
RESEARCH_INTEGRATOR_FLOAT("PM_Potential",p.PM_Potential)
#endif
#if (defined(EVALPOTENTIAL) || defined(OUTPUT_POTENTIAL)) && defined(EXTERNALGRAVITY)
RESEARCH_INTEGRATOR_FLOAT("ExtPotential",p.ExtPotential)
#endif
#if defined(STARFORMATION)
RESEARCH_INTEGRATOR_FLOAT("StellarAge",p.StellarAge)
#endif
#if defined(STARFORMATION)
RESEARCH_INTEGRATOR_FLOAT("Metallicity",p.Metallicity)
#endif
#if defined(FORCETEST)
RESEARCH_INTEGRATOR_FLOAT("GravAccelDirect[0]",p.GravAccelDirect[0])
#endif
#if defined(FORCETEST)
RESEARCH_INTEGRATOR_FLOAT("GravAccelDirect[1]",p.GravAccelDirect[1])
#endif
#if defined(FORCETEST)
RESEARCH_INTEGRATOR_FLOAT("GravAccelDirect[2]",p.GravAccelDirect[2])
#endif
#if defined(FORCETEST)
RESEARCH_INTEGRATOR_FLOAT("PotentialDirect",p.PotentialDirect)
#endif
#if defined(FORCETEST)
RESEARCH_INTEGRATOR_FLOAT("DistToID1",p.DistToID1)
#endif
#if defined(FORCETEST) && defined(PMGRID)
RESEARCH_INTEGRATOR_FLOAT("GravAccelShortRange[0]",p.GravAccelShortRange[0])
#endif
#if defined(FORCETEST) && defined(PMGRID)
RESEARCH_INTEGRATOR_FLOAT("GravAccelShortRange[1]",p.GravAccelShortRange[1])
#endif
#if defined(FORCETEST) && defined(PMGRID)
RESEARCH_INTEGRATOR_FLOAT("GravAccelShortRange[2]",p.GravAccelShortRange[2])
#endif
#if defined(FORCETEST) && defined(PMGRID)
RESEARCH_INTEGRATOR_FLOAT("PotentialShortRange",p.PotentialShortRange)
#endif
#if defined(FORCETEST) && defined(PMGRID) && defined(PLACEHIGHRESREGION)
RESEARCH_INTEGRATOR_FLOAT("GravAccelVeryShortRange[0]",p.GravAccelVeryShortRange[0])
#endif
#if defined(FORCETEST) && defined(PMGRID) && defined(PLACEHIGHRESREGION)
RESEARCH_INTEGRATOR_FLOAT("GravAccelVeryShortRange[1]",p.GravAccelVeryShortRange[1])
#endif
#if defined(FORCETEST) && defined(PMGRID) && defined(PLACEHIGHRESREGION)
RESEARCH_INTEGRATOR_FLOAT("GravAccelVeryShortRange[2]",p.GravAccelVeryShortRange[2])
#endif
#if defined(FORCETEST) && defined(PMGRID) && defined(PLACEHIGHRESREGION)
RESEARCH_INTEGRATOR_FLOAT("GravAccelHPM[0]",p.GravAccelHPM[0])
#endif
#if defined(FORCETEST) && defined(PMGRID) && defined(PLACEHIGHRESREGION)
RESEARCH_INTEGRATOR_FLOAT("GravAccelHPM[1]",p.GravAccelHPM[1])
#endif
#if defined(FORCETEST) && defined(PMGRID) && defined(PLACEHIGHRESREGION)
RESEARCH_INTEGRATOR_FLOAT("GravAccelHPM[2]",p.GravAccelHPM[2])
#endif
#if defined(FORCETEST) && defined(PMGRID) && defined(PLACEHIGHRESREGION)
RESEARCH_INTEGRATOR_FLOAT("PotentialVeryShortRange",p.PotentialVeryShortRange)
#endif
#if defined(FORCETEST) && defined(PMGRID) && defined(PLACEHIGHRESREGION)
RESEARCH_INTEGRATOR_FLOAT("PotentialHPM",p.PotentialHPM)
#endif
