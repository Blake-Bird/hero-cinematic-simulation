// Define RESEARCH_GLOBAL_FLOAT/INTEGER/TEXT before including.
RESEARCH_GLOBAL_FLOAT("Time",All.Time)
RESEARCH_GLOBAL_FLOAT("TimeBegin",All.TimeBegin)
RESEARCH_GLOBAL_FLOAT("TimeStep",All.TimeStep)
RESEARCH_GLOBAL_FLOAT("TimeMax",All.TimeMax)
RESEARCH_GLOBAL_FLOAT("Timebase_interval",All.Timebase_interval)
RESEARCH_GLOBAL_FLOAT("G",All.G)
RESEARCH_GLOBAL_FLOAT("Hubble",All.Hubble)
RESEARCH_GLOBAL_FLOAT("HubbleParam",All.HubbleParam)
RESEARCH_GLOBAL_FLOAT("Omega0",All.Omega0)
RESEARCH_GLOBAL_FLOAT("OmegaLambda",All.OmegaLambda)
RESEARCH_GLOBAL_FLOAT("OmegaRadiation",All.OmegaRadiation)
RESEARCH_GLOBAL_FLOAT("OmegaBaryon",All.OmegaBaryon)
RESEARCH_GLOBAL_FLOAT("OmegaCurvature",All.OmegaCurvature)
RESEARCH_GLOBAL_FLOAT("BoxSize",All.BoxSize)
RESEARCH_GLOBAL_FLOAT("cf_atime",All.cf_atime)
RESEARCH_GLOBAL_FLOAT("cf_atime2",All.cf_atime2)
RESEARCH_GLOBAL_FLOAT("cf_ainv",All.cf_ainv)
RESEARCH_GLOBAL_FLOAT("cf_a2inv",All.cf_a2inv)
RESEARCH_GLOBAL_FLOAT("cf_a3inv",All.cf_a3inv)
RESEARCH_GLOBAL_FLOAT("cf_afac1",All.cf_afac1)
RESEARCH_GLOBAL_FLOAT("cf_afac2",All.cf_afac2)
RESEARCH_GLOBAL_FLOAT("cf_afac3",All.cf_afac3)
RESEARCH_GLOBAL_FLOAT("cf_hubble_a",All.cf_hubble_a)
RESEARCH_GLOBAL_FLOAT("cf_atime_hubble_a",All.cf_atime_hubble_a)
RESEARCH_GLOBAL_FLOAT("cf_atime2_hubble_a",All.cf_atime2_hubble_a)
RESEARCH_GLOBAL_FLOAT("cf_redshift",All.cf_redshift)
RESEARCH_GLOBAL_FLOAT("cf_H",All.cf_H)
RESEARCH_GLOBAL_FLOAT("ErrTolTheta",All.ErrTolTheta)
RESEARCH_GLOBAL_FLOAT("ErrTolThetaMax",All.ErrTolThetaMax)
RESEARCH_GLOBAL_FLOAT("ErrTolForceAcc",All.ErrTolForceAcc)
RESEARCH_GLOBAL_FLOAT("ErrTolIntAccuracy",All.ErrTolIntAccuracy)
RESEARCH_GLOBAL_FLOAT("MinSizeTimestep",All.MinSizeTimestep)
RESEARCH_GLOBAL_FLOAT("MaxSizeTimestep",All.MaxSizeTimestep)
RESEARCH_GLOBAL_FLOAT("CourantFac",All.CourantFac)
RESEARCH_GLOBAL_FLOAT("MaxNumNgbDeviation",All.MaxNumNgbDeviation)
RESEARCH_GLOBAL_FLOAT("ArtBulkViscConst",All.ArtBulkViscConst)
RESEARCH_GLOBAL_FLOAT("InitGasU",All.InitGasU)
RESEARCH_GLOBAL_FLOAT("MinEgySpec",All.MinEgySpec)
RESEARCH_GLOBAL_INTEGER("Ti_Current",All.Ti_Current)
RESEARCH_GLOBAL_INTEGER("Ti_nextoutput",All.Ti_nextoutput)
RESEARCH_GLOBAL_INTEGER("Ti_lastoutput",All.Ti_lastoutput)
RESEARCH_GLOBAL_INTEGER("NumCurrentTiStep",All.NumCurrentTiStep)
RESEARCH_GLOBAL_INTEGER("LowestActiveTimeBin",All.LowestActiveTimeBin)
RESEARCH_GLOBAL_INTEGER("HighestActiveTimeBin",All.HighestActiveTimeBin)
RESEARCH_GLOBAL_INTEGER("LowestOccupiedTimeBin",All.LowestOccupiedTimeBin)
RESEARCH_GLOBAL_INTEGER("HighestOccupiedTimeBin",All.HighestOccupiedTimeBin)
RESEARCH_GLOBAL_INTEGER("LowestOccupiedGravTimeBin",All.LowestOccupiedGravTimeBin)
RESEARCH_GLOBAL_INTEGER("HighestOccupiedGravTimeBin",All.HighestOccupiedGravTimeBin)
RESEARCH_GLOBAL_INTEGER("HighestSynchronizedTimeBin",All.HighestSynchronizedTimeBin)
RESEARCH_GLOBAL_INTEGER("SmallestTimeBinWithDomainDecomposition",All.SmallestTimeBinWithDomainDecomposition)
RESEARCH_GLOBAL_INTEGER("GlobalNSynchronizedHydro",All.GlobalNSynchronizedHydro)
RESEARCH_GLOBAL_INTEGER("GlobalNSynchronizedGravity",All.GlobalNSynchronizedGravity)
RESEARCH_GLOBAL_INTEGER("ComovingIntegrationOn",All.ComovingIntegrationOn)
RESEARCH_GLOBAL_INTEGER("TypeOfOpeningCriterion",All.TypeOfOpeningCriterion)
RESEARCH_GLOBAL_INTEGER("RelOpeningCriterionInUse",All.RelOpeningCriterionInUse)
RESEARCH_GLOBAL_INTEGER("DumpFlag_nextoutput",All.DumpFlag_nextoutput)
RESEARCH_GLOBAL_INTEGER("SnapshotFileCount",All.SnapshotFileCount)
RESEARCH_GLOBAL_INTEGER("OutputListOn",All.OutputListOn)
RESEARCH_GLOBAL_INTEGER("OutputListLength",All.OutputListLength)
RESEARCH_GLOBAL_INTEGER("DesNumNgb",All.DesNumNgb)
#if defined(PMGRID) && !defined(TREEPM_NOTIMESPLIT)
RESEARCH_GLOBAL_INTEGER("PM_Ti_endstep",All.PM_Ti_endstep)
#endif
#if defined(PMGRID) && !defined(TREEPM_NOTIMESPLIT)
RESEARCH_GLOBAL_INTEGER("PM_Ti_begstep",All.PM_Ti_begstep)
#endif
#if defined(FORCE_EQUAL_TIMESTEPS)
RESEARCH_GLOBAL_INTEGER("GlobalTimeStep",All.GlobalTimeStep)
#endif
#if defined(PMGRID) && !defined(TREEPM_NOTIMESPLIT)
RESEARCH_GLOBAL_FLOAT("DtDisplacement",All.DtDisplacement)
#endif
#if defined(EVALPOTENTIAL) && defined(PMGRID) && defined(PERIODIC)
RESEARCH_GLOBAL_FLOAT("TotalMass",All.TotalMass)
#endif
#if defined(LEAN)
RESEARCH_GLOBAL_FLOAT("PartMass",All.PartMass)
#endif
#if defined(TIMEDEP_ART_VISC)
RESEARCH_GLOBAL_FLOAT("AlphaMin",All.AlphaMin)
#endif
#if defined(RESEARCH_SIDM_REFERENCE)
RESEARCH_GLOBAL_TEXT("SIDMOperatorSeed",All.SIDMOperatorSeed)
#endif
#if defined(RESEARCH_SIDM_REFERENCE) && defined(RESEARCH_SIDM_CANONICAL_CONTROL)
RESEARCH_GLOBAL_TEXT("SIDMCanonicalClockSHA256",All.SIDMCanonicalClockSHA256)
RESEARCH_GLOBAL_TEXT("SIDMCanonicalPhysicalStateSHA256",All.SIDMCanonicalPhysicalStateSHA256)
RESEARCH_GLOBAL_FLOAT("SIDMCanonicalInitialScaleFactor",All.SIDMCanonicalInitialScaleFactor)
RESEARCH_GLOBAL_INTEGER("SIDMCanonicalMaximumAssetBytes",All.SIDMCanonicalMaximumAssetBytes)
RESEARCH_GLOBAL_INTEGER("SIDMCanonicalMaximumRows",All.SIDMCanonicalMaximumRows)
RESEARCH_GLOBAL_INTEGER("SIDMCanonicalMaximumClockWork",All.SIDMCanonicalMaximumClockWork)
#endif
#if defined(RESEARCH_SIDM_REFERENCE)
RESEARCH_GLOBAL_TEXT("SIDMAngularModelHash",All.SIDMAngularModelHash)
#endif
#if defined(RESEARCH_SIDM_REFERENCE)
RESEARCH_GLOBAL_TEXT("SIDMEventIdentityNamespace",All.SIDMEventIdentityNamespace)
#endif
#if defined(RESEARCH_SIDM_REFERENCE)
RESEARCH_GLOBAL_TEXT("SIDMReferenceMaximumPairVisits",All.SIDMReferenceMaximumPairVisits)
#endif
#if defined(RESEARCH_SIDM_REFERENCE)
RESEARCH_GLOBAL_FLOAT("SIDMReferenceKernelRadiusProperKpc",All.SIDMReferenceKernelRadiusProperKpc)
#endif
#if defined(RESEARCH_SIDM_REFERENCE)
RESEARCH_GLOBAL_FLOAT("SIDMReferenceCrossSectionNormalizationCm2PerG",All.SIDMReferenceCrossSectionNormalizationCm2PerG)
#endif
#if defined(RESEARCH_SIDM_REFERENCE)
RESEARCH_GLOBAL_FLOAT("SIDMReferenceScreeningVelocityKmPerS",All.SIDMReferenceScreeningVelocityKmPerS)
#endif
#if defined(RESEARCH_SIDM_REFERENCE)
RESEARCH_GLOBAL_FLOAT("SIDMReferenceMaximumEventProbability",All.SIDMReferenceMaximumEventProbability)
#endif
#if defined(RESEARCH_SIDM_REFERENCE)
RESEARCH_GLOBAL_FLOAT("SIDMReferenceMacroMassTolerance",All.SIDMReferenceMacroMassTolerance)
#endif
#if defined(RESEARCH_SIDM_REFERENCE)
RESEARCH_GLOBAL_FLOAT("SIDMReferenceConservationTolerance",All.SIDMReferenceConservationTolerance)
#endif
#if defined(RESEARCH_SIDM_REFERENCE)
RESEARCH_GLOBAL_FLOAT("SIDMReferenceEnergyFloor",All.SIDMReferenceEnergyFloor)
#endif
#if defined(RESEARCH_SIDM_REFERENCE)
RESEARCH_GLOBAL_FLOAT("SIDMReferenceMomentumFloor",All.SIDMReferenceMomentumFloor)
#endif
#if defined(RESEARCH_SIDM_REFERENCE)
RESEARCH_GLOBAL_FLOAT("SIDMReferenceCDFTolerance",All.SIDMReferenceCDFTolerance)
#endif
#if defined(RESEARCH_SIDM_REFERENCE)
RESEARCH_GLOBAL_INTEGER("SIDMReferenceMaximumParticles",All.SIDMReferenceMaximumParticles)
#endif
#if defined(RESEARCH_SIDM_REFERENCE)
RESEARCH_GLOBAL_INTEGER("SIDMReferenceMaximumEventsPerStep",All.SIDMReferenceMaximumEventsPerStep)
#endif
#if defined(RESEARCH_SIDM_REFERENCE)
RESEARCH_GLOBAL_INTEGER("SIDMReferenceAngularKind",All.SIDMReferenceAngularKind)
#endif
#if defined(RESEARCH_SIDM_REFERENCE)
RESEARCH_GLOBAL_INTEGER("SIDMReferenceAttemptCount",All.SIDMReferenceAttemptCount)
#endif
#if defined(RESEARCH_SIDM_REFERENCE)
RESEARCH_GLOBAL_INTEGER("SIDMReferenceEventCount",All.SIDMReferenceEventCount)
#endif
#if defined(RESEARCH_SIDM_REFERENCE)
RESEARCH_GLOBAL_FLOAT("SIDMReferenceOpticalDepthSum",All.SIDMReferenceOpticalDepthSum)
#endif
#if defined(RESEARCH_SIDM_REFERENCE)
RESEARCH_GLOBAL_INTEGER("SIDMReferenceKernelKind",All.SIDMReferenceKernelKind)
#endif
#if defined(RESEARCH_SIDM_REFERENCE)
RESEARCH_GLOBAL_FLOAT("SIDMReferenceKernelOriginRelativeBound",All.SIDMReferenceKernelOriginRelativeBound)
#endif
#if defined(RESEARCH_SIDM_REFERENCE)
RESEARCH_GLOBAL_FLOAT("SIDMReferenceMaximumParticleProbability",All.SIDMReferenceMaximumParticleProbability)
#endif
#if defined(RESEARCH_SIDM_REFERENCE)
RESEARCH_GLOBAL_INTEGER("SIDMReferenceMaximumSubsteps",All.SIDMReferenceMaximumSubsteps)
#endif
#if defined(RESEARCH_SIDM_REFERENCE)
RESEARCH_GLOBAL_INTEGER("SIDMReferenceSubstepCount",All.SIDMReferenceSubstepCount)
#endif
#if defined(RESEARCH_SIDM_REFERENCE)
RESEARCH_GLOBAL_INTEGER("SIDMReferencePairVisitCount",All.SIDMReferencePairVisitCount)
#endif
#if defined(RESEARCH_SIDM_REFERENCE)
RESEARCH_GLOBAL_FLOAT("SIDMReferenceSmallestSourceStepSeconds",All.SIDMReferenceSmallestSourceStepSeconds)
#endif
#if defined(RESEARCH_SIDM_REFERENCE)
RESEARCH_GLOBAL_INTEGER("SIDMReferenceSchemeKind",All.SIDMReferenceSchemeKind)
#endif
#if defined(RESEARCH_SIDM_REFERENCE)
RESEARCH_GLOBAL_INTEGER("SIDMReferenceParticleConvention",All.SIDMReferenceParticleConvention)
#endif
#if defined(RESEARCH_SIDM_REFERENCE)
RESEARCH_GLOBAL_FLOAT("SIDMReferenceMaximumTransferDepth",All.SIDMReferenceMaximumTransferDepth)
#endif
#if defined(RESEARCH_SIDM_REFERENCE)
RESEARCH_GLOBAL_FLOAT("SIDMReferenceMaximumNeglectedMomentRatio",All.SIDMReferenceMaximumNeglectedMomentRatio)
#endif
#if defined(RESEARCH_SIDM_REFERENCE)
RESEARCH_GLOBAL_INTEGER("SIDMReferenceFrequentUpdateCount",All.SIDMReferenceFrequentUpdateCount)
#endif
#if defined(RESEARCH_SIDM_REFERENCE)
RESEARCH_GLOBAL_FLOAT("SIDMReferenceSplitAngleRadians",All.SIDMReferenceSplitAngleRadians)
#endif
#if defined(RESEARCH_SIDM_REFERENCE)
RESEARCH_GLOBAL_FLOAT("SIDMReferenceClockAbsoluteToleranceSeconds",All.SIDMReferenceClockAbsoluteToleranceSeconds)
#endif
#if defined(RESEARCH_SIDM_REFERENCE)
RESEARCH_GLOBAL_FLOAT("SIDMReferenceClockRelativeTolerance",All.SIDMReferenceClockRelativeTolerance)
#endif
#if defined(RESEARCH_SIDM_REFERENCE)
RESEARCH_GLOBAL_FLOAT("SIDMReferenceMaximumScaleFactorFractionalChange",All.SIDMReferenceMaximumScaleFactorFractionalChange)
#endif
#if defined(RESEARCH_SIDM_REFERENCE)
RESEARCH_GLOBAL_INTEGER("SIDMReferenceMaximumClockPanels",All.SIDMReferenceMaximumClockPanels)
#endif
#if defined(RESEARCH_SIDM_REFERENCE)
RESEARCH_GLOBAL_INTEGER("SIDMReferenceMaximumLocalPairs",All.SIDMReferenceMaximumLocalPairs)
#endif
#if defined(RESEARCH_SIDM_REFERENCE)
RESEARCH_GLOBAL_INTEGER("SIDMReferenceMaximumMatchingRounds",All.SIDMReferenceMaximumMatchingRounds)
#endif
#if defined(RESEARCH_SIDM_REFERENCE)
RESEARCH_GLOBAL_INTEGER("SIDMReferenceBackendKind",All.SIDMReferenceBackendKind)
#endif
#if defined(RESEARCH_SIDM_REFERENCE)
RESEARCH_GLOBAL_TEXT("SIDMReferenceMaximumExchangeRecords",All.SIDMReferenceMaximumExchangeRecords)
#endif
#if defined(RESEARCH_SIDM_REFERENCE)
RESEARCH_GLOBAL_FLOAT("SIDMReferenceExchangeMemoryMiB",All.SIDMReferenceExchangeMemoryMiB)
#endif
#if defined(RESEARCH_SIDM_REFERENCE)
RESEARCH_GLOBAL_INTEGER("SIDMReferenceMatchingRoundCount",All.SIDMReferenceMatchingRoundCount)
#endif
#if defined(RESEARCH_SIDM_REFERENCE)
RESEARCH_GLOBAL_INTEGER("SIDMReferenceRemoteEndpointUpdateCount",All.SIDMReferenceRemoteEndpointUpdateCount)
#endif
#if defined(RESEARCH_SIDM_REFERENCE)
RESEARCH_GLOBAL_INTEGER("SIDMReferenceExchangeSentBytes",All.SIDMReferenceExchangeSentBytes)
#endif
#if defined(RESEARCH_SIDM_REFERENCE)
RESEARCH_GLOBAL_INTEGER("SIDMReferenceExchangeReceivedBytes",All.SIDMReferenceExchangeReceivedBytes)
#endif
#if defined(RESEARCH_SIDM_REFERENCE)
RESEARCH_GLOBAL_FLOAT("SIDMReferenceCoarseKernelRadiusProperKpc",All.SIDMReferenceCoarseKernelRadiusProperKpc)
#endif
#if defined(RESEARCH_SIDM_REFERENCE)
RESEARCH_GLOBAL_FLOAT("SIDMReferenceMaximumKernelCoarseSupportFraction",All.SIDMReferenceMaximumKernelCoarseSupportFraction)
#endif
#if defined(RESEARCH_SIDM_REFERENCE)
RESEARCH_GLOBAL_FLOAT("SIDMReferenceMaximumOmittedScatteringUpperBound",All.SIDMReferenceMaximumOmittedScatteringUpperBound)
#endif
#if defined(STELLAR_EVOLUTION)
RESEARCH_GLOBAL_TEXT("StellarEvolutionTableCRC32",All.StellarEvolutionTableCRC32)
#endif
#if defined(STELLAR_EVOLUTION)
RESEARCH_GLOBAL_FLOAT("StellarSourceRadiusProperKpc",All.StellarSourceRadiusProperKpc)
#endif
#if defined(STELLAR_EVOLUTION)
RESEARCH_GLOBAL_FLOAT("StellarSourceFraction",All.StellarSourceFraction)
#endif
#if defined(STELLAR_EVOLUTION)
RESEARCH_GLOBAL_INTEGER("StellarMaximumCount",All.StellarMaximumCount)
#endif
#if defined(STELLAR_EVOLUTION)
RESEARCH_GLOBAL_INTEGER("StellarMaximumSubcycles",All.StellarMaximumSubcycles)
#endif
#if defined(STELLAR_EVOLUTION)
RESEARCH_GLOBAL_INTEGER("StellarReturnCommitCount",All.StellarReturnCommitCount)
#endif
#if defined(STELLAR_EVOLUTION)
RESEARCH_GLOBAL_INTEGER("StellarReturnEventCount",All.StellarReturnEventCount)
#endif
#if defined(STELLAR_EVOLUTION) && defined(RESEARCH_GRACKLE)
RESEARCH_GLOBAL_TEXT("GrackleDataCRC32",All.GrackleDataCRC32)
#endif
#if defined(STELLAR_EVOLUTION) && defined(RESEARCH_GRACKLE)
RESEARCH_GLOBAL_INTEGER("GracklePrimordialChemistry",All.GracklePrimordialChemistry)
#endif
#if defined(STELLAR_EVOLUTION) && defined(RESEARCH_GRACKLE)
RESEARCH_GLOBAL_INTEGER("GrackleMetalCooling",All.GrackleMetalCooling)
#endif
#if defined(STELLAR_EVOLUTION) && defined(RESEARCH_GRACKLE)
RESEARCH_GLOBAL_INTEGER("GrackleUVBackground",All.GrackleUVBackground)
#endif
#if defined(BLACK_HOLES)
RESEARCH_GLOBAL_FLOAT("BHRadiativeEfficiency",All.BHRadiativeEfficiency)
#endif
#if defined(BLACK_HOLES)
RESEARCH_GLOBAL_FLOAT("BHFeedbackCoupling",All.BHFeedbackCoupling)
#endif
#if defined(BLACK_HOLES)
RESEARCH_GLOBAL_FLOAT("BHBondiBoost",All.BHBondiBoost)
#endif
#if defined(BLACK_HOLES)
RESEARCH_GLOBAL_FLOAT("BHEddingtonFactor",All.BHEddingtonFactor)
#endif
#if defined(BLACK_HOLES)
RESEARCH_GLOBAL_FLOAT("BHKernelRadiusProperKpc",All.BHKernelRadiusProperKpc)
#endif
#if defined(BLACK_HOLES)
RESEARCH_GLOBAL_FLOAT("BHSeedMassMsun",All.BHSeedMassMsun)
#endif
#if defined(BLACK_HOLES)
RESEARCH_GLOBAL_FLOAT("BHSourceFraction",All.BHSourceFraction)
#endif
#if defined(BLACK_HOLES)
RESEARCH_GLOBAL_INTEGER("BHAngularMomentumModel",All.BHAngularMomentumModel)
#endif
#if defined(BLACK_HOLES)
RESEARCH_GLOBAL_INTEGER("BHFeedbackMode",All.BHFeedbackMode)
#endif
#if defined(BLACK_HOLES)
RESEARCH_GLOBAL_FLOAT("BHViscosityParameter",All.BHViscosityParameter)
#endif
#if defined(BLACK_HOLES)
RESEARCH_GLOBAL_INTEGER("BHMaximumCount",All.BHMaximumCount)
#endif
#if defined(BLACK_HOLES)
RESEARCH_GLOBAL_INTEGER("BHDynamicsEnabled",All.BHDynamicsEnabled)
#endif
#if defined(BLACK_HOLES)
RESEARCH_GLOBAL_INTEGER("BHDynamicsMaximumSubcycles",All.BHDynamicsMaximumSubcycles)
#endif
#if defined(BLACK_HOLES)
RESEARCH_GLOBAL_FLOAT("BHBackgroundRadiusProperKpc",All.BHBackgroundRadiusProperKpc)
#endif
#if defined(BLACK_HOLES)
RESEARCH_GLOBAL_FLOAT("BHMinimumImpactProperKpc",All.BHMinimumImpactProperKpc)
#endif
#if defined(BLACK_HOLES)
RESEARCH_GLOBAL_FLOAT("BHStellarHardeningCoefficient",All.BHStellarHardeningCoefficient)
#endif
#if defined(BLACK_HOLES)
RESEARCH_GLOBAL_FLOAT("BHBinaryMaximumEccentricity",All.BHBinaryMaximumEccentricity)
#endif
#if defined(BLACK_HOLES)
RESEARCH_GLOBAL_FLOAT("BHDynamicsFraction",All.BHDynamicsFraction)
#endif
#if defined(BLACK_HOLES)
RESEARCH_GLOBAL_FLOAT("BHSeedSpinX",All.BHSeedSpinX)
#endif
#if defined(BLACK_HOLES)
RESEARCH_GLOBAL_FLOAT("BHSeedSpinY",All.BHSeedSpinY)
#endif
#if defined(BLACK_HOLES)
RESEARCH_GLOBAL_FLOAT("BHSeedSpinZ",All.BHSeedSpinZ)
#endif
#if defined(BLACK_HOLES)
RESEARCH_GLOBAL_TEXT("BHRecoilSeed",All.BHRecoilSeed)
#endif
#if defined(BLACK_HOLES)
RESEARCH_GLOBAL_INTEGER("BHEventCommitCount",All.BHEventCommitCount)
#endif
#if defined(BLACK_HOLES)
RESEARCH_GLOBAL_INTEGER("BHEventCount",All.BHEventCount)
#endif
#if defined(BLACK_HOLES)
RESEARCH_GLOBAL_INTEGER("BHSeedingMode",All.BHSeedingMode)
#endif
#if defined(BLACK_HOLES)
RESEARCH_GLOBAL_FLOAT("BHSeedDensityHydrogenCm3",All.BHSeedDensityHydrogenCm3)
#endif
#if defined(BLACK_HOLES)
RESEARCH_GLOBAL_FLOAT("BHSeedExclusionProperKpc",All.BHSeedExclusionProperKpc)
#endif
#if defined(STARFORMATION)
RESEARCH_GLOBAL_TEXT("StellarRandomSeed",All.StellarRandomSeed)
#endif
#if defined(STARFORMATION)
RESEARCH_GLOBAL_INTEGER("StellarBirthEventCount",All.StellarBirthEventCount)
#endif
#if defined(STARFORMATION)
RESEARCH_GLOBAL_INTEGER("StellarBirthCommitCount",All.StellarBirthCommitCount)
#endif

#if defined(RESEARCH_SIDM_REFERENCE)
RESEARCH_GLOBAL_TEXT("SIDMNativeTablePath",All.SIDMNativeTablePath)
RESEARCH_GLOBAL_TEXT("SIDMNativeTableSHA256",All.SIDMNativeTableSHA256)
#endif

#if defined(RESEARCH_SIDM_REFERENCE)
RESEARCH_GLOBAL_INTEGER("SIDMEventAuditModulus",All.SIDMEventAuditModulus)
RESEARCH_GLOBAL_INTEGER("SIDMEventAuditResidue",All.SIDMEventAuditResidue)
#endif

#ifdef RESEARCH_GRACKLE
RESEARCH_GLOBAL_FLOAT("GrackleDeuteriumToHydrogenRatio",All.GrackleDeuteriumToHydrogenRatio)
#endif

#ifdef RESEARCH_GRACKLE
RESEARCH_GLOBAL_FLOAT("GrackleSeedHI",All.GrackleInitialSpecies[0])
RESEARCH_GLOBAL_FLOAT("GrackleSeedHII",All.GrackleInitialSpecies[1])
RESEARCH_GLOBAL_FLOAT("GrackleSeedHeI",All.GrackleInitialSpecies[2])
RESEARCH_GLOBAL_FLOAT("GrackleSeedHeII",All.GrackleInitialSpecies[3])
RESEARCH_GLOBAL_FLOAT("GrackleSeedHeIII",All.GrackleInitialSpecies[4])
RESEARCH_GLOBAL_FLOAT("GrackleSeedElectron",All.GrackleInitialSpecies[5])
RESEARCH_GLOBAL_FLOAT("GrackleSeedHM",All.GrackleInitialSpecies[6])
RESEARCH_GLOBAL_FLOAT("GrackleSeedH2I",All.GrackleInitialSpecies[7])
RESEARCH_GLOBAL_FLOAT("GrackleSeedH2II",All.GrackleInitialSpecies[8])
RESEARCH_GLOBAL_FLOAT("GrackleSeedDI",All.GrackleInitialSpecies[9])
RESEARCH_GLOBAL_FLOAT("GrackleSeedDII",All.GrackleInitialSpecies[10])
RESEARCH_GLOBAL_FLOAT("GrackleSeedHDI",All.GrackleInitialSpecies[11])
#endif
