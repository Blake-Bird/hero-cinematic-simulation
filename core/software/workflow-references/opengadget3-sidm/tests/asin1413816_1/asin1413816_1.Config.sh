#!/bin/bash            # this line only there to enable syntax highlighting in this file
##################################################
#  Enable/Disable compile-time options as needed #
##################################################
DEBUG

KD_BH_USE_TOPHAT_FOR_RADIO_MODE
KD_FAST_SINGLE_GUESS

SUBFIND_ID_OVERBOOK_FACTOR=15

#--------------------------------------- Basic operation mode of code
PERIODIC
COOLING
SFR
KD_BUFFER_MANAGEMENT=0.3
KD_RESTRICT_MAXVEL=5000.0
OUTPUTTIMESTEP

WENDLAND_C4_KERNEL        # Switch to Wendland C4 kernel

DS_SHIFT_BOX
#--------------------------------------- TreePM Options
PMGRID=256
KD_PM_NOSENDPOS
KD_PM_FASTGREENS
ASMTH=1.5
RCUT=6.0
PLACEHIGHRESREGION=55
ENLARGEREGION=1.1
HIGHRES_PMGRID=64

#--------------------------------------- Sort Options
EVALPOTENTIAL           # computes gravitational potential

#--------------------------------------- Multi-Domain and Top-Level Tree options
MULTIPLEDOMAINS=4
TOPNODEFACTOR=8

#--------------------------------------- Things that are always recommended
SYNCRONIZ_OUTPUT               # Writes output only at global time steps

#---------------------------------------- Single/Double Precision
DOUBLEPRECISION=3

#---------------------------------------- On the fly FOF groupfinder 
FOF                                # enable FoF output
SUBFIND                            # enables substructure finder
LT_ADD_GAL_TO_SUB=12                # Adds optical luminosities in 6 bands to subhalos

#--------------------------------------- SFR/feedback model
WINDS=350

#-------------------------------------- AGN stuff
BLACK_HOLES             # enables Black-Holes (master switch)

#------------------------------------ # [options from LT and DF]
LT_MAX_TEMP_FEEDBACK=5.0e8
LT_USE_TRUENGB

#-------------------------------------------- Things for special behaviour
NOTEST_FOR_IDUNIQUENESS

#--------------------------------------- Output/Input options
OUTPUTPOTENTIAL
PHIDOT

#--------------------------------------- Testing and Debugging options
VERBOSE_LEVEL=0

#--------------------------------------- Thermal conduction
CONDUCTION
CONDUCTION_SATURATION

#--------------------------------------- Real Cosmic Rays
JD_VTURB				 # Compute vturb,vrms,vblk 
KD_RESTRICT_NEIGHBOURS

#--------------------------------------- SPH viscosity options
TIME_DEP_ART_VISC=3          # Enables time dependend viscosity

#....................................# STELLAR EVOLUTION options
LT_STELLAREVOLUTION        # enable stellar evolution+
EXPERT_LEVEL=42
