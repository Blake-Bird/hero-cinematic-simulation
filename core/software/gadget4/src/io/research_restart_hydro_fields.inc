// Authored named native restart diagnostic fields. Define RESEARCH_STATE_FIELD before including.
// Conditional fields mirror sph_particle_data.h and the BH particle members.
RESEARCH_STATE_FIELD("R000","ResearchGas_Hsml",MEM_MY_FLOAT,1,A_SPHP,&Sp->SphP[0].Hsml,GAS_ONLY)
RESEARCH_STATE_FIELD("R001","ResearchGas_DhsmlDensityFactor",MEM_MY_FLOAT,1,A_SPHP,&Sp->SphP[0].DhsmlDensityFactor,GAS_ONLY)
RESEARCH_STATE_FIELD("R002","ResearchGas_DivVel",MEM_MY_FLOAT,1,A_SPHP,&Sp->SphP[0].DivVel,GAS_ONLY)
RESEARCH_STATE_FIELD("R003","ResearchGas_CurlVel",MEM_MY_FLOAT,1,A_SPHP,&Sp->SphP[0].CurlVel,GAS_ONLY)
RESEARCH_STATE_FIELD("R004","ResearchGas_Csnd",MEM_MY_FLOAT,1,A_SPHP,&Sp->SphP[0].Csnd,GAS_ONLY)
RESEARCH_STATE_FIELD("R005","ResearchGas_Density",MEM_MY_FLOAT,1,A_SPHP,&Sp->SphP[0].Density,GAS_ONLY)
RESEARCH_STATE_FIELD("R006","ResearchGas_Pressure",MEM_MY_FLOAT,1,A_SPHP,&Sp->SphP[0].Pressure,GAS_ONLY)
RESEARCH_STATE_FIELD("R007","ResearchGas_Entropy",MEM_MY_FLOAT,1,A_SPHP,&Sp->SphP[0].Entropy,GAS_ONLY)
RESEARCH_STATE_FIELD("R008","ResearchGas_EntropyPred",MEM_MY_FLOAT,1,A_SPHP,&Sp->SphP[0].EntropyPred,GAS_ONLY)
RESEARCH_STATE_FIELD("R009","ResearchGas_DtEntropy",MEM_MY_FLOAT,1,A_SPHP,&Sp->SphP[0].DtEntropy,GAS_ONLY)
RESEARCH_STATE_FIELD("R010","ResearchGas_DtDensity",MEM_MY_FLOAT,1,A_SPHP,&Sp->SphP[0].DtDensity,GAS_ONLY)
RESEARCH_STATE_FIELD("R011","ResearchGas_DtHsml",MEM_MY_FLOAT,1,A_SPHP,&Sp->SphP[0].DtHsml,GAS_ONLY)
RESEARCH_STATE_FIELD("R012","ResearchGas_NumNgb",MEM_MY_FLOAT,1,A_SPHP,&Sp->SphP[0].NumNgb,GAS_ONLY)
RESEARCH_STATE_FIELD("R013","ResearchGas_MaxSignalVel",MEM_MY_FLOAT,1,A_SPHP,&Sp->SphP[0].MaxSignalVel,GAS_ONLY)
RESEARCH_STATE_FIELD("R014","ResearchGas_CurrentMaxTiStep",MEM_MY_FLOAT,1,A_SPHP,&Sp->SphP[0].CurrentMaxTiStep,GAS_ONLY)
RESEARCH_STATE_FIELD("R015","ResearchGas_VelPred",MEM_MY_FLOAT,3,A_SPHP,&Sp->SphP[0].VelPred[0],GAS_ONLY)
RESEARCH_STATE_FIELD("R016","ResearchGas_HydroAccel",MEM_MY_FLOAT,3,A_SPHP,&Sp->SphP[0].HydroAccel[0],GAS_ONLY)
#ifdef OUTPUT_ACCELERATION
RESEARCH_STATE_FIELD("R043","ResearchGas_HydroForceNativeTime",MEM_DOUBLE,1,A_SPHP,&Sp->SphP[0].ResearchHydroForceNativeTime,GAS_ONLY)
RESEARCH_STATE_FIELD("R044","ResearchGas_HydroForceEpochVersion",MEM_UINT64,1,A_SPHP,&Sp->SphP[0].ResearchHydroForceEpochVersion,GAS_ONLY)
#endif
RESEARCH_STATE_FIELD("R017","ResearchGas_Rot",MEM_MY_FLOAT,3,A_SPHP,&Sp->SphP[0].Rot[0],GAS_ONLY)
#ifdef BLACK_HOLES
RESEARCH_STATE_FIELD("R018","ResearchGas_BHSourceInternalEnergy",MEM_DOUBLE,1,A_SPHP,&Sp->SphP[0].BHSourceInternalEnergy,GAS_ONLY)
#endif
#ifdef HIERARCHICAL_GRAVITY
RESEARCH_STATE_FIELD("R019","ResearchGas_FullGravAccel",MEM_MY_FLOAT,3,A_SPHP,&Sp->SphP[0].FullGravAccel[0],GAS_ONLY)
#endif
#ifdef PRESSURE_ENTROPY_SPH
RESEARCH_STATE_FIELD("R020","ResearchGas_EntropyToInvGammaPred",MEM_MY_FLOAT,1,A_SPHP,&Sp->SphP[0].EntropyToInvGammaPred,GAS_ONLY)
#endif
#ifdef PRESSURE_ENTROPY_SPH
RESEARCH_STATE_FIELD("R021","ResearchGas_DhsmlDerivedDensityFactor",MEM_MY_FLOAT,1,A_SPHP,&Sp->SphP[0].DhsmlDerivedDensityFactor,GAS_ONLY)
#endif
#ifdef PRESSURE_ENTROPY_SPH
RESEARCH_STATE_FIELD("R022","ResearchGas_PressureSphDensity",MEM_MY_FLOAT,1,A_SPHP,&Sp->SphP[0].PressureSphDensity,GAS_ONLY)
#endif
#ifdef PRESSURE_ENTROPY_SPH
RESEARCH_STATE_FIELD("R023","ResearchGas_DtPressureSphDensity",MEM_MY_FLOAT,1,A_SPHP,&Sp->SphP[0].DtPressureSphDensity,GAS_ONLY)
#endif
#ifdef TIMEDEP_ART_VISC
RESEARCH_STATE_FIELD("R024","ResearchGas_Alpha",MEM_MY_FLOAT,1,A_SPHP,&Sp->SphP[0].Alpha,GAS_ONLY)
#endif
#ifdef TIMEDEP_ART_VISC
RESEARCH_STATE_FIELD("R025","ResearchGas_DivVelOld",MEM_MY_FLOAT,1,A_SPHP,&Sp->SphP[0].DivVelOld,GAS_ONLY)
#endif
#ifdef TIMEDEP_ART_VISC
RESEARCH_STATE_FIELD("R026","ResearchGas_decayVel",MEM_MY_FLOAT,1,A_SPHP,&Sp->SphP[0].decayVel,GAS_ONLY)
#endif
#ifdef IMPROVED_VELOCITY_GRADIENTS
RESEARCH_STATE_FIELD("R027","ResearchGas_dpos_dx_dx",MEM_MY_FLOAT,1,A_SPHP,&Sp->SphP[0].dpos.dx_dx,GAS_ONLY)
#endif
#ifdef IMPROVED_VELOCITY_GRADIENTS
RESEARCH_STATE_FIELD("R028","ResearchGas_dpos_dx_dy",MEM_MY_FLOAT,1,A_SPHP,&Sp->SphP[0].dpos.dx_dy,GAS_ONLY)
#endif
#ifdef IMPROVED_VELOCITY_GRADIENTS
RESEARCH_STATE_FIELD("R029","ResearchGas_dpos_dx_dz",MEM_MY_FLOAT,1,A_SPHP,&Sp->SphP[0].dpos.dx_dz,GAS_ONLY)
#endif
#ifdef IMPROVED_VELOCITY_GRADIENTS
RESEARCH_STATE_FIELD("R030","ResearchGas_dpos_dy_dy",MEM_MY_FLOAT,1,A_SPHP,&Sp->SphP[0].dpos.dy_dy,GAS_ONLY)
#endif
#ifdef IMPROVED_VELOCITY_GRADIENTS
RESEARCH_STATE_FIELD("R031","ResearchGas_dpos_dy_dz",MEM_MY_FLOAT,1,A_SPHP,&Sp->SphP[0].dpos.dy_dz,GAS_ONLY)
#endif
#ifdef IMPROVED_VELOCITY_GRADIENTS
RESEARCH_STATE_FIELD("R032","ResearchGas_dpos_dz_dz",MEM_MY_FLOAT,1,A_SPHP,&Sp->SphP[0].dpos.dz_dz,GAS_ONLY)
#endif
#ifdef IMPROVED_VELOCITY_GRADIENTS
RESEARCH_STATE_FIELD("R033","ResearchGas_dvel",MEM_MY_FLOAT,9,A_SPHP,&Sp->SphP[0].dvel[0][0],GAS_ONLY)
#endif
#ifdef STARFORMATION
RESEARCH_STATE_FIELD("R034","ResearchGas_Metallicity",MEM_MY_FLOAT,1,A_SPHP,&Sp->SphP[0].Metallicity,GAS_ONLY)
#endif
#ifdef STARFORMATION
RESEARCH_STATE_FIELD("R035","ResearchGas_MassMetallicity",MEM_MY_FLOAT,1,A_SPHP,&Sp->SphP[0].MassMetallicity,GAS_ONLY)
#endif
#ifdef STARFORMATION
RESEARCH_STATE_FIELD("R036","ResearchGas_Sfr",MEM_MY_FLOAT,1,A_SPHP,&Sp->SphP[0].Sfr,GAS_ONLY)
#endif
#ifdef COOLING
RESEARCH_STATE_FIELD("R037","ResearchGas_Ne",MEM_MY_FLOAT,1,A_SPHP,&Sp->SphP[0].Ne,GAS_ONLY)
#endif
#ifdef OUTPUT_COOLHEAT
RESEARCH_STATE_FIELD("R038","ResearchGas_CoolHeat",MEM_MY_FLOAT,1,A_SPHP,&Sp->SphP[0].CoolHeat,GAS_ONLY)
#endif
#ifdef BLACK_HOLES
RESEARCH_STATE_FIELD("R039","ResearchBH_BHMass",MEM_DOUBLE,1,A_P,&Sp->P[0].BHMass,32)
#endif
#ifdef BLACK_HOLES
RESEARCH_STATE_FIELD("R040","ResearchBH_BHInflowRate",MEM_DOUBLE,1,A_P,&Sp->P[0].BHInflowRate,32)
#endif
#ifdef BLACK_HOLES
RESEARCH_STATE_FIELD("R041","ResearchBH_BHCumulativeRadiatedMass",MEM_DOUBLE,1,A_P,&Sp->P[0].BHCumulativeRadiatedMass,32)
#endif
#ifdef BLACK_HOLES
RESEARCH_STATE_FIELD("R042","ResearchBH_BHCumulativeFeedbackEnergy",MEM_DOUBLE,1,A_P,&Sp->P[0].BHCumulativeFeedbackEnergy,32)
#endif

#ifdef RESEARCH_GRACKLE
RESEARCH_STATE_FIELD("RGGP","ResearchGas_ResearchGammaPressure",MEM_DOUBLE,1,A_SPHP,&Sp->SphP[0].ResearchGammaPressure,GAS_ONLY)
RESEARCH_STATE_FIELD("RGGS","ResearchGas_ResearchGammaSound",MEM_DOUBLE,1,A_SPHP,&Sp->SphP[0].ResearchGammaSound,GAS_ONLY)
#endif
