# might be looking for the compile-time Makefile options of the code...
#
# They have moved to a separate file.
#
# To build the code, do the following:
#
#  (0) Run "make check" for a defaults compilation test.
#      Run "make test"  for a defaults compilation+running test (needs MPI environment..).
#
#  (1) Copy one of the Config files in the tests/ dir to the main code dir as "Config.sh"
#
#        e.g. cp tests/FreefallCollapse/FREEFALL-SPH.Config.sh Config.sh
#
#  (2) Edit "Config.sh" as needed for your application
#
#  (3) Run "make"
#
#  "Config.sh" should *not* be checked in to the repository
#
#  Note: It is possible to override the default name of the
#  Config.sh file, if desired, as well as the name of the
#  executable. For example:
#
#   make  CONFIG=MyNewConf.sh  EXEC=Gadget_new
#
#-----------------------------------------------------------------
#
# You might also be looking for the target system SYSTYPE option
#
# To build the code, do the following:
#
# (A) load build script for avaliable system
#
#        e.g. source Build/LUMI_build.sh LUMI-C
#
# (B) set the SYSTYPE variable in your .bashrc (or similar file):
#
#        e.g. export SYSTYPE=XXX
#
# Set default values for compiler options if no systype options are given or found
CC        = mpiCC                           # sets the C-compiler (default)
CXX       = mpiCC                           # sets the C++-compiler (default)
OPTIMIZE  = -std=c++17 -Wall -g -O2
MPICHLIB  = -lmpich
SWITCHES  =
TOPDIR 	  := $(dir $(lastword $(MAKEFILE_LIST))) # absolute path of main directory, used to find gadgetconfig.hpps

ifdef SYSTYPE
SYSTYPE := $(SYSTYPE)
else
include Build/Makefile.systype
endif

ifeq ($(wildcard Makefile.systype), Makefile.systype)
INCL = Build/Makefile.systype
else
INCL =
endif

CONFIG   =  Config.sh
PERL     =  /usr/bin/perl

GADGETVERSION := $(shell grep 'GADGETVERSION' CodeBase/allvars.hpp | grep -o -E '[0-9.]+')
SHA := $(shell git rev-parse HEAD)
RESULT     := $(shell CONFIG=$(CONFIG) PERL=$(PERL) make -f config-makefile)
CONFIGVARS := $(shell cat gadgetconfig.hpp)


## UM check/print SYSTYPE, CONFIG file and code version before starting...
$(info Compiling on SYSTYPE:		$(SYSTYPE) )
$(info CONFIG file:         		$(CONFIG) )
$(info GADGETVERSION:         		$(GADGETVERSION) )
$(info SHA:         		        $(SHA) )

ifeq (SOFTDOUBLEDOUBLE,$(filter SOFTDOUBLEDOUBLE,$(CONFIGVARS)))
CC       =   mpiCC     # default C++ compiler
OPTIMIZE =   -g
OPT     +=  -DX86FIX   # only needed for 32-bit intel/amd systems
endif

ifeq (HAVE_HDF5,$(filter HAVE_HDF5,$(CONFIGVARS)))
HDF5_LIB_NAMES = -lhdf5
endif

ifeq (LT_STELLAREVOLUTION,$(filter LT_STELLAREVOLUTION,$(CONFIGVARS)))
ifneq (LT_METAL_COOLING,$(filter LT_METAL_COOLING,$(CONFIGVARS)))
HDF5_LIB_NAMES = -lhdf5
endif
endif

-include Build/Makefile.local

#----------------------------------------------------------------------------------------------

include Build/Makefile.$(SYSTYPE)

#----------------------------------------------------------------------------------------------

ifndef MKLFFTW # MKL fftw for Intel works differently
  ifeq (PMGRID,$(filter PMGRID,$(CONFIGVARS)))                              # need fftw ?
    ifeq (SINGLEPRECISION_FFTW,$(filter SINGLEPRECISION_FFTW,$(CONFIGVARS))) # determine fftw precision
      $(info using single-precision FFTW)
      FFTW_NAME=fftw3f
    else
      $(info using double-precision FFTW) 
      FFTW_NAME=fftw3
    endif

    FFTW_LIBNAMES =  -l$(FFTW_NAME)_mpi -l$(FFTW_NAME)
    $(info $$OPTIMIZE is [${OPTIMIZE}])
    ifeq (openmp,$(findstring openmp,$(OPTIMIZE)))                                # need OpenMP fftw ?
      FFTW_LIBNAMES +=  -l$(FFTW_NAME)_omp
    else
      ifeq (multicore,$(findstring multicore,$(OPTIMIZE)))                        # need OpenMP fftw ?
        FFTW_LIBNAMES +=  -l$(FFTW_NAME)_omp
      endif
    endif
  endif
endif # MKLFFTW

# forcetree.o appears in two switches!
ifeq (ACC,$(filter ACC,$(CONFIGVARS))) # OpenACC
	HYDRO_OBJS += OpenACC/gpuallvars_branch.o OpenACC/ngb.o
	INCL    += OpenACC/conduction.hpp OpenACC/find_hsml.hpp OpenACC/gpuallvars_branch.hpp OpenACC/gradients.hpp OpenACC/gravtree.hpp OpenACC/hydra.hpp OpenACC/thrust_sort.hpp
ifeq (THRUST_SORT,$(filter THRUST_SORT,$(CONFIGVARS))) # OpenACC
	HYDRO_OBJS += OpenACC/thrust_sort.o
endif
else
	HYDRO_OBJS += Gravity/ngb.o
endif

OPTIONS = $(OPTIMIZE) $(OPT)

FOPTIONS = $(OPTIMIZE) $(FOPT)

EXEC  = OpenGadget3

OBJS  = CodeBase/allvars.o CodeBase/units.o CodeBase/peano.o CodeBase/endrun.o CodeBase/global.o CodeBase/begrun.o \
        CodeBase/run.o CodeBase/init.o CodeBase/domain.o CodeBase/main.o \
        Gravity/gravtree.o Gravity/forcetree.o Gravity/darkenergy.o Gravity/accel.o Gravity/forcetree_update.o \
        Gravity/potential.o \
        Integrator/driftfac.o Integrator/predict.o Integrator/timestep.o Integrator/kicks.o \
	 	IO/read_ic.o IO/read_ic_cluster.o IO/second_order.o IO/restart.o IO/io.o IO/io_matrix.o \
	 	System/system.o System/allocate.o System/sizelimited_sendrecv.o System/checksummed_sendrecv.o \
        System/parallel_sort_special.o System/mymalloc.o System/calc_checksum.o \
        System/Md5.o System/pinning.o System/rng.o \
		PM/pm_util.o PM/pm_powerspec.o PM/pm_density.o PM/parallel_sort_l3.o \
		PM/pm_init.o PM/pm_readout.o PM/pm.o PM/bounds_init.o \
	 	compile_time_info.o

INCL += Makefile \
		CodeBase/allvars.hpp CodeBase/endrun.cpp CodeBase/proto.hpp CodeBase/precision.hpp \
		CodeBase/domain.hpp CodeBase/units.cpp CodeBase/units.hpp System/dd.hpp System/myqsort.hpp \
		CodeBase/switches.hpp CodeBase/switches_check.hpp CodeBase/constants.hpp CodeBase/verbose.hpp \
		SPACETimers/SPACE_timers.hpp SPACETimers/timer_CPU.hpp IO/io_matrix.hpp IO/io.hpp \
		Hydro/kernel.hpp Hydro/compute_gradients.hpp \
		Hydro/hydra_extra.hpp Hydro/hydra.hpp Hydro/hydra_structs.hpp Hydro/viscosity.hpp Hydro/shockfinder.hpp \
		PM/pm.hpp PM/bounds_init.hpp \
		GreenTree_AR/ar_density.cpp GreenTree_AR/ar_hydra.cpp GreenTree_AR/ar_conduction.cpp GreenTree_AR/recompose_domain.cpp \
        System/tags.hpp System/vector.hpp System/communication.hpp System/assert.hpp \
		Gravity/forcetree.hpp \
	 	Mfm/Matrix.hpp Mfm/Vector.hpp

ifeq (ACC,$(filter ACC,$(CONFIGVARS))) # OpenACC
	OBJS += OpenACC/gpuallvars_branch.o 
ifeq (THRUST_SORT,$(filter THRUST_SORT,$(CONFIGVARS))) # OpenACC
	OBJS += OpenACC/thrust_sort.o
endif
endif

ifneq ($(filter ACC_NGB ACC_HYDRA ACC_HSML ACC_GRADIENTS ACC_CONDUCTION ACC_SPECTRAL_CRs_DIFFUSION,$(CONFIGVARS)),) # OpenACC
    OBJS += OpenACC/ngb.o
else
	OBJS += Gravity/ngb.o
endif

ifeq (MAGNETIC,$(filter MAGNETIC,$(CONFIGVARS)))
INCL    += Hydro/mhd.hpp Hydro/mhd_seeding.hpp
endif

ifeq (DO_NOT_USE_NEW_HSML_AND_GRADIENTS,$(filter DO_NOT_USE_NEW_HSML_AND_GRADIENTS,$(CONFIGVARS)))
OBJS    +=  
else
OBJS    += Hydro/compute_gradients.o Hydro/find_hsml.o
endif

ifneq ($(filter fSIDM rSIDM fIDM rIDM,$(CONFIGVARS)),)
OBJS += mSIDM/tools.o mSIDM/comm.o mSIDM/parallel.o mSIDM/kernel_integration.o \
        mSIDM/scattering.o mSIDM/find_hsml_pt1.o mSIDM/execute_pxp.o mSIDM/dcs_class.o\
        mSIDM/CrossSectionTable.o mSIDM/CrossSectionIntegrator.o
INCL += mSIDM/kernel.hpp mSIDM/scattering_options.hpp mSIDM/scattering_types.hpp \
        mSIDM/scattering_data.hpp mSIDM/scattering_modules.hpp mSIDM/scattering_modules.cpp\
        mSIDM/DcsFile.hpp
CFLAGS += -Ilibs/nlohmann/include  $(GSL_INCL)
LIBS += -lgsl -lgslcblas  $(GSL_LIBS)
endif

ifneq ($(filter fSIDM rSIDM,$(CONFIGVARS)),)
OBJS += mSIDM/msidm.o
INCL += mSIDM/msidm.hpp
endif

ifneq ($(filter fIDM rIDM,$(CONFIGVARS)),)
OBJS += mSIDM/midm.o
INCL += mSIDM/midm.hpp
endif

ifeq (COOLING,$(filter COOLING,$(CONFIGVARS)))
OBJS	+= CoolingSfr/cooling.o
INCL    += CoolingSfr/cooling.hpp
endif

ifneq (,$(filter SFR FG_MAKE_OUTSIDE_HIGHRES_DM FG_MAKE_MAXVEL_DM,$(CONFIGVARS)))
    OBJS += CoolingSfr/rearrange_particles.o
endif

ifeq (SFR,$(filter SFR,$(CONFIGVARS)))
OBJS	+= CoolingSfr/sfr_eff.o
#INCL    += CoolingSfr/sfr_eff.h
endif

ifeq (BLACK_HOLES,$(filter BLACK_HOLES,$(CONFIGVARS)))
OBJS	+= Blackholes/blackhole.o Blackholes/blackhole_feedback_loop.o Blackholes/blackhole_swallow_loop.o
INCL    += Blackholes/blackhole.hpp Blackholes/blackhole_begrun.hpp
endif

ifeq (SUBFIND,$(filter SUBFIND,$(CONFIGVARS)))
OBJS	+= FofSubfind/subfind.o FofSubfind/subfind_vars.o FofSubfind/subfind_collective.o FofSubfind/subfind_serial.o FofSubfind/subfind_so.o \
           FofSubfind/subfind_cont.o FofSubfind/subfind_distribute.o FofSubfind/subfind_findlinkngb.o FofSubfind/subfind_nearesttwo.o FofSubfind/subfind_loctree.o \
           FofSubfind/subfind_alternative_collective.o FofSubfind/subfind_reshuffle.o FofSubfind/subfind_potential.o FofSubfind/subfind_density.o
INCL    += FofSubfind/subfind.hpp
endif

ifeq (FOF,$(filter FOF,$(CONFIGVARS)))
OBJS	+= FofSubfind/fof.o
INCL    += FofSubfind/fof.hpp
endif

ifeq (PMGRID,$(filter PMGRID,$(CONFIGVARS)))
OBJS	+= $(PM_OBJS) Gravity/longrange.o
endif

ifeq (PHIDOT,$(filter PHIDOT,$(CONFIGVARS)))
OBJS	+= Gravity/phidot.o
endif

ifeq (FORCETEST,$(filter FORCETEST,$(CONFIGVARS)))
OBJS	+= Gravity/gravtree_forcetest.o
endif

ifeq (CONDUCTION,$(filter CONDUCTION,$(CONFIGVARS)))
OBJS	+= Hydro/conduction.o
endif

ifeq (GT_DUST,$(filter GT_DUST,$(CONFIGVARS)))
OBJS    += Hydro/dust.o
INCL    += Hydro/dust.hpp
endif
#This is slowing down, so not recomendet to use
#ifeq (AR_GREEN_TREE_FOF,$(filter AR_GREEN_TREE_FOF,$(CONFIGVARS)))
#OBJS  += GreenTree_AR/ar_ngb.o
#endif

#ifeq (RECOMPOSE_DOMAIN,$(filter RECOMPOSE_DOMAIN,$(CONFIGVARS)))
#OBJS    += GreenTree_AR/recompose_domain.o
#endif

ifeq (USE_MESHLESS_FINITE_MASS,$(filter USE_MESHLESS_FINITE_MASS,$(CONFIGVARS)))
OBJS += Mfm/mfm_fluxes.o Mfm/mfm_limiters.o
OBJS += Mfm/ExactRiemannSolver.o Mfm/FluidVector.o Mfm/FluxSolver.o
OBJS += Mfm/RoeRiemannSolver.o Mfm/HlleRiemannSolver.o  Mfm/HllcRiemannSolver.o Mfm/HllRiemannSolver.o Mfm/RiemannSolver.o
ifeq (MAGNETIC,$(filter MAGNETIC,$(CONFIGVARS)))
OBJS += Mfm/HlldRiemannSolver.o
endif
ifeq (GT_DUST,$(filter GT_DUST,$(CONFIGVARS)))
OBJS += Mfm/DustRiemannSolver.o
endif
# Under MFM hydro, SHOCKFINDER selects the ray-traced finder (SHOCKFINDER_MFM,
# auto-derived in CodeBase/switches.hpp); link its object here.
ifeq (SHOCKFINDER,$(filter SHOCKFINDER,$(CONFIGVARS)))
OBJS += Mfm/mfm_shockfinder.o
INCL += Mfm/mfm_shockfinder.hpp
endif
INCL += Mfm/Eos.hpp Mfm/FluidVector.hpp Mfm/FluxSolver.hpp
INCL += Mfm/RiemannSolver.hpp
INCL += Mfm/mfm.hpp Mfm/mfm_fluxes.hpp Mfm/mfm_limiters.hpp Mfm/SlopeLimiter.hpp Mfm/gradients.hpp
else
OBJS += Hydro/hydra.o
endif

ifeq (LT_STELLAREVOLUTION,$(filter LT_STELLAREVOLUTION,$(CONFIGVARS)))
OBJS  += CoolingSfr/Sfr_LT/lt_io.o CoolingSfr/Sfr_LT/lt_utils.o CoolingSfr/ImfLib/libimf.o CoolingSfr/ImfLib/libimf_vars.o
INCL  += CoolingSfr/Sfr_LT/lt_sn.hpp CoolingSfr/Sfr_LT/lt.hpp CoolingSfr/Sfr_LT/lt_error_codes.hpp

ifeq (LT_SEv_INFO,$(filter LT_SEv_INFO,$(CONFIGVARS)))
OBJS  += CoolingSfr/Sfr_LT/lt_sfr.o
else
ifeq (LT_ZAGE,$(filter LT_ZAGE,$(CONFIGVARS)))
OBJS  += CoolingSfr/Sfr_LT/lt_sfr.o
else
OBJS  += CoolingSfr/Sfr_LT/lt_sfr_light.o
endif
endif

ifeq (LT_EJECTA_IN_HOTPHASE,$(filter LT_EJECTA_IN_HOTPHASE,$(CONFIGVARS)))
OBJS  += CoolingSfr/Sfr_LT/lt_sn.o
else
OBJS  += CoolingSfr/Sfr_LT/lt_sn_light.o
endif

endif

ifeq (LT_STELLAREVOLUTION,$(filter LT_STELLAREVOLUTION,$(CONFIGVARS)))
ifneq (LT_METAL_COOLING,$(filter LT_METAL_COOLING,$(CONFIGVARS)))
OBJS  += CoolingSfr/Sfr_LT/lt_wal_cooling.o 
INCL  += CoolingSfr/Sfr_LT/lt_wal_cooling.hpp
endif
endif

ifeq (GM_MUPPI,$(filter GM_MUPPI,$(CONFIGVARS))) # add MUPPI
OBJS += CoolingSfr/Muppi/sfr_muppi.o CoolingSfr/Muppi/muppi_communications.o CoolingSfr/Muppi/muppi_normalize.o
CFLAGS += $(GSL_INCL) 
LIBS += -lgsl -lgslcblas  $(GSL_LIBS)
endif

ifeq (GM_STARDENSITY,$(filter GM_STARDENSITY,$(CONFIGVARS)))
OBJS    += Blackholes/density_stars.o
endif

ifeq (GL_CR_DUST,$(filter GL_CR_DUST,$(CONFIGVARS)))
OBJS  += CoolingSfr/Sfr_LT/gl_cr_dust.o 
endif

ifeq (LMB_SPECTRAL_CRs,$(filter LMB_SPECTRAL_CRs,$(CONFIGVARS))) # add spectral cr part
OBJS += CosmicRays/lmb_cr_lookup_integrals.o \
		CosmicRays/lmb_cr_init.o \
		CosmicRays/lmb_cr_adiabatic_changes.o \
		CosmicRays/lmb_cr_injection.o \
		CosmicRays/lmb_cosmic_rays.o \
		CosmicRays/lmb_cr_conduction.o \
		CosmicRays/lmb_cr_diffusion.o \
		CosmicRays/lmb_cr_tests.o \
		CosmicRays/lmb_cr_debugging_tools.o

INCL += CosmicRays/lmb_cr_e_n_integrals.hpp \
		CosmicRays/lmb_cr_debugging_tools.hpp \
		CosmicRays/lmb_cr_conduction.hpp \
		CosmicRays/lmb_cr_slope_solver.hpp \
		CosmicRays/lmb_cr_update_functions.hpp \
		CosmicRays/lmb_cr_init.hpp \
		CosmicRays/lmb_cr_adiabatic_changes.hpp \
		CosmicRays/lmb_cr_radiative_changes.hpp \
		CosmicRays/lmb_cr_injection.hpp \
		CosmicRays/lmb_cr_Dpp.hpp \
		CosmicRays/lmb_cr_timestep.hpp \
		CosmicRays/lmb_cr_tests.hpp \
		CosmicRays/lmb_cr_lookup_integrals.hpp
endif

ifeq (LMB_CR_OUTPUT_SYNCHROTRON,$(filter LMB_CR_OUTPUT_SYNCHROTRON,$(CONFIGVARS))) # add synchrotron emission calculation
OBJS += CosmicRays/lmb_cr_synchrotron.o
INCL += CosmicRays/lmb_cr_synchrotron.hpp
endif

ifeq (LMB_CR_OUTPUT_GAMMA,$(filter LMB_CR_OUTPUT_GAMMA,$(CONFIGVARS))) # add synchrotron emission calculation
OBJS += CosmicRays/lmb_cr_gamma.o
INCL += CosmicRays/lmb_cr_gamma.hpp
endif

ifeq (SEN_CR_SECONDARIES,$(filter SEN_CR_SECONDARIES,$(CONFIGVARS))) # add contribution of secondary CRs
INCL += CosmicRays/lmb_cr_gamma.hpp \
		CosmicRays/sen_cr_secondaries.h
CFLAGS += $(GSL_INCL) 
LIBS += -lgsl -lgslcblas  $(GSL_LIBS)
endif

ifeq (JD_DPP,$(filter JD_DPP,$(CONFIGVARS))) # add JD cr part
OBJS += CosmicRays/cr_electrons.o
endif

ifeq (CS_DECOUPLING,$(filter CS_DECOUPLING,$(CONFIGVARS)))
	OBJS  += CoolingSfr/cs_model/cs_decoupling.o 
INCL  += CoolingSfr/cs_model/cs_model.hpp 
endif

ifeq (UM_CHEMISTRY,$(filter UM_CHEMISTRY,$(CONFIGVARS)))
INCL += Chemistry/chemistry.hpp
OBJS += Chemistry/chemistry_noneq.o
endif

ifeq (UM_METAL_COOLING,$(filter UM_METAL_COOLING,$(CONFIGVARS)))
OBJS += Chemistry/metal_cooling.o
endif

ifeq (ADAPTGRAVSOFT,$(filter ADAPTGRAVSOFT,$(CONFIGVARS)))
OBJS += Gravity/AdaptGravSoft/ads_ngb.o Gravity/AdaptGravSoft/ads_density.o
endif

ifeq (ANALYTIC_POTENTIALS,$(filter ANALYTIC_POTENTIALS,$(CONFIGVARS)))
OBJS += Gravity/analytic_potentials.o
endif

CFLAGS += $(OPTIONS) $(HDF5_INCL) $(FFTW_INCL) $(GMP_INCL) $(MPICH_INCL) -Ilibs/boost_1_91_0

LIBS   += -lm $(HDF5_LIBS) $(HDF5_LIB_NAMES) $(FFTW_LIBS) $(FFTW_LIBNAMES) -g $(MPICHLIB)

ifeq (TIMER_NVTX,$(filter TIMER_NVTX,$(CONFIGVARS)))
NVTXLIB = -L$(CUDA_ROOT)/lib64 -lnvToolsExt
LIBS += $(NVTXLIB)
endif

ifeq (TIMER_VTUNE,$(filter TIMER_VTUNE, $(CONFIGVARS)))
VTUNE_LIBS = -L$(VTUNE_PROFILER_DIR)/lib64 -littnotify
LIBS    += $(VTUNE_LIBS)
endif

CXXFLAGS = $(CFLAGS)

$(EXEC): $(OBJS)
	$(CC) $(SWITCHES) $(OPTIMIZE) $(OBJS) $(LIBS) $(RLIBS) -o $(EXEC)  
	@echo "// OpenGadget3 Version:" $(GADGETVERSION) 'SHA:' $(SHA) >> gadgetconfig.hpp

# special compilation to pass SHA number to code
CodeBase/begrun.o: CodeBase/begrun.cpp
	$(CC) $(SWITCHES) $(OPTIMIZE) $(CFLAGS) -DGIT_SHA=\"$(SHA)\" -c CodeBase/begrun.cpp -o CodeBase/begrun.o

# do flag check before compilation
flagcheck.out: $(CONFIG)
	@if python3 -c 'import sys; exit(0 if sys.version_info >= (3, 0) else 1)' 2>/dev/null; then \
		python3 Python/check_flags.py $(CONFIG) $(OBJS) $(INCL) $(PM_OBJS); \
	elif Python2.7 -c 'import sys; exit(0 if sys.version_info >= (2, 7) else 1)' 2>/dev/null; then \
		echo "Python3 is not available - using Python2.7"; \
		python2.7 Python/check_flags.py $(CONFIG) $(OBJS) $(INCL) $(PM_OBJS); \
	else \
		echo "Neighter Python3 nor 2.7 are available - skipping flag check."; \
	fi

OpenACC/thrust_sort.o:
	$(NVCC_EXEC)  -x cu  -ccbin=pgc++  -c OpenACC/thrust_sort.cpp -o OpenACC/thrust_sort.o -lcudart

$(OBJS):  $(INCL)  $(CONFIG)  compile_time_info.cpp flagcheck.out


compile_time_info.cpp: $(CONFIG)
	$(PERL) prepare-config.perl $(CONFIG)

.PHONY: clean
clean:
	rm -f $(OBJS) $(EXEC) gadgetconfig.hpp flagcheck.out

.PHONY: check
check:
	OPENGADGET3_TEST_PREAMBLE_COMMAND="set -e" ./tests/run_all_config_tests.sh

.PHONY: test
test:
	OPENGADGET3_TEST_PREAMBLE_COMMAND="set -e" ./run_all_tests.sh

.PHONY: tests
