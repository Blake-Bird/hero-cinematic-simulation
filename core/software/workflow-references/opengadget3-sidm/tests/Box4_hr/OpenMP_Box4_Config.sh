#!/bin/bash            # this line only there to enable syntax highlighting in this file

##################################################
#  enable/Disable compile-time options as needed #
##################################################

ACC
ACC_OPENMP
ACC_ATOMIC
ACC_ASYNC
ACC_GRAVITY

ACC_HSML
ACC_GRADIENTS
ACC_HYDRA
ACC_CONDUCTION

WAKEUP=0
AR_XMAS_TREE

#-------------------------------------- Some General settings
VERBOSE_LEVEL=2
EXPERT_LEVEL=42

#-------------------------------------- Suggested settings for special behaviour
KD_LIMIT_DIVV=30
SUBFIND_DO_NOT_COUNT_IN_PARALLEL
KD_FAST_SINGLE_GUESS
NOTEST_FOR_IDUNIQUENESS
KD_RESTRICT_NEIGHBOURS
KD_BUFFER_MANAGEMENT=0.9
KD_RESTRICT_MAXVEL=5000.0
WRITE_KEY_FILES=1
SYNCRONIZ_OUTPUT

#--------------------------------------- Special settings for this setup
LONGIDS
PERIODIC
MULTIPLEDOMAINS=16
TOPNODEFACTOR=16
DOUBLEPRECISION=3

#--------------------------------------- Basic operation mode of code
COOLING
SFR

#--------------------------------------- SPH Settings
WENDLAND_C6_KERNEL        # Switch to Wendland C6 kernel
TIME_DEP_ART_VISC=3           # Enables time dependend viscosity
ARTIFICIAL_CONDUCTIVITY    # enables Price-Monaghan artificial conductivity
TIME_DEP_ART_COND=2

#--------------------------------------- TreePM Options
EVALPOTENTIAL           # computes gravitational potential            

#--------------------------------------- SFR/feedback model
LT_STELLAREVOLUTION         # enable LT stellar evolution
WINDS=350

#-------------------------------------- AGN stuff
BLACK_HOLES             # enables Black-Holes (master switch)
LS_BLACK_HOLES_SPIN_EVOLUTION
AD_DYNFRIC

#--------------------------------------- Thermal conduction
CONDUCTION
CONDUCTION_SATURATION

#---------------------------------------- On the fly FOF groupfinder
FOF                                # enable FoF output
SUBFIND                            # enables substructure finder
LT_ADD_GAL_TO_SUB=12               # Adds optical luminosities in 6 bands to subhalos

OLD_KERNEL
AR_FIX_ASSIGN
AR_FIX_TIME_DEP_ART_COND
