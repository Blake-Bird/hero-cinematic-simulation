# General code settings
KD_FAST_SINGLE_GUESS

DIVBCLEANING_TRICCO
EXPERT_LEVEL=42
VERBOSE_LEVEL=1

PERIODIC
READ_HSML                     # reads hsml from IC file   

#MULTIPLEDOMAINS=2
TOPNODEFACTOR=8

WENDLAND_C6_KERNEL
JD_VTURB

# Simulation Setup
DOUBLEPRECISION=1
NOGRAVITY
LONG_X=140
LONG_Y=1
LONG_Z=1

# MHD settings (general)
MU0_UNITY                   # Sets \mu_0 to unity   

# MHD settings (induction equation)
MAGNETIC                    # Turns on B Field (including induction eqn) 

##################################################################################
# Add ons to the MHD module ######################################################
##################################################################################

# MHD settings (artificial magnetic dissipation)
ARTIFICIAL_MAGNETIC_DISSIPATION       # Turns on artificial magnetic dissipation 
ARTIFICIAL_CONDUCTIVITY    # enables Price-Monaghan artificial conductivity

# Viscosity settings
TIME_DEP_ART_VISC=3
KD_BUFFER_MANAGEMENT=0.3
