#Some Makefile for CLASS.
#Julien Lesgourgues, 28.11.2011
#Nils Schöneberg, Matteo Lucca, 27.02.2019

MDIR := $(shell pwd)
WRKDIR = $(MDIR)/build

.base:
	if ! [ -e $(WRKDIR) ]; then mkdir $(WRKDIR) ; mkdir $(WRKDIR)/lib; fi;
	touch build/.base

vpath %.c source:tools:main
vpath %.o build
vpath %.opp build
vpath .base build

########################################################
###### LINES TO ADAPT TO YOUR PLATEFORM ################
########################################################

# your C compiler:
CC       = gcc
#CC       = icc
#CC       = pgcc
CPP      = g++ --std=c++11 -fpermissive -Wno-write-strings

# your tool for creating static libraries:
AR        = ar rv

# Your python interpreter.
# In order to use Python 3, you can manually
# substitute python3 to python in the line below, or you can simply
# add a compilation option on the terminal command line:
# "PYTHON=python3 make all" (Thanks to Marius Millea for python3 compatibility)
PYTHON ?= python

# your optimization flag
OPTFLAG = -O3
#OPTFLAG = -Ofast -ffast-math #-march=native
#OPTFLAG = -fast

# your openmp flag (comment for compiling without openmp)
OMPFLAG   = -pthread #-fopenmp
#OMPFLAG   = -mp -mp=nonuma -mp=allcores -g
#OMPFLAG   = -openmp

# all other compilation flags
CCFLAG = -g -fPIC
LDFLAG = -g -fPIC

#GSL FLAGS
GSL_LIB = /usr/local/lib/

#GSLFLAG += -lgsl -lgslcblas -lm
#GSLFLAG += -lgsl -lgslcblas -lm
GSLFLAG += -lm
# leave blank to compile without HyRec, or put path to HyRec directory
# (with no slash at the end: e.g. "external/RecfastCLASS")
HYREC = external/HyRec2020
RECFAST = external/RecfastCLASS
HEATING = external/heating
HALOFIT = external/Halofit
HMCODE = external/HMcode

########################################################
###### IN PRINCIPLE THE REST SHOULD BE LEFT UNCHANGED ##
########################################################

# pass current working directory to the code
CLASSDIR ?= $(MDIR)
CCFLAG += -D__CLASSDIR__='"$(CLASSDIR)"'

# where to find include files *.h
INCLUDES = -I../include
HEADERFILES = $(wildcard ./include/*.h)

# automatically add external programs if needed. First, initialize to blank.
EXTERNAL =

vpath %.c $(RECFAST)
#CCFLAG += -DRECFAST
INCLUDES += -I../$(RECFAST)
EXTERNAL += wrap_recfast.o
HEADERFILES += $(wildcard ./$(RECFAST)/*.h)

vpath %.c $(HEATING)
#CCFLAG += -DHEATING
INCLUDES += -I../$(HEATING)
EXTERNAL += injection.o noninjection.o
HEADERFILES += $(wildcard ./$(HEATING)/*.h)

# update flags for including HyRec
ifneq ($(HYREC),)
vpath %.c $(HYREC)

CCFLAG += -DHYREC
#LDFLAGS += -DHYREC
INCLUDES += -I../$(HYREC)
EXTERNAL += hyrectools.o helium.o hydrogen.o history.o wrap_hyrec.o energy_injection.o
HEADERFILES += $(wildcard ./$(HYREC)/*.h)
endif

vpath %.c $(HALOFIT)
INCLUDES += -I../$(HALOFIT)
EXTERNAL += halofit.o
HEADERFILES += $(wildcard ./$(HALOFIT)/*.h)

vpath %.c $(HMCODE)
INCLUDES += -I../$(HMCODE)
EXTERNAL += hmcode.opp
HEADERFILES += $(wildcard ./$(HMCODE)/*.h)

%.o:  %.c .base $(HEADERFILES)
	cd $(WRKDIR);$(CC) $(OPTFLAG) $(OMPFLAG) $(CCFLAG) $(INCLUDES) -c ../$< -o $*.o

%.opp:  %.c .base $(HEADERFILES)
	cd $(WRKDIR);$(CPP) $(OPTFLAG) $(OMPFLAG) $(CCFLAG) $(INCLUDES) -c ../$< -o $*.opp

TOOLS = growTable.o dei_rkck.o sparse.o evolver_rkck.o  evolver_ndf15.o arrays.opp parser.o quadrature.o hyperspherical.opp common.o trigonometric_integrals.o

SOURCE = input.o background.o thermodynamics.o perturbations.opp primordial.opp fourier.o transfer.opp harmonic.opp lensing.opp distortions.o

INPUT = input.o

PRECISION = precision.o

BACKGROUND = background.o

THERMO = thermodynamics.o

PERTURBATIONS = perturbations.o

TRANSFER = transfer.o

PRIMORDIAL = primordial.o

HARMONIC = harmonic.o

FOURIER = fourier.o

LENSING = lensing.o

DISTORTIONS = distortions.o

OUTPUT = output.o

CLASS = class.o

# Portable pipeline distribution: native executable only; no regression fixtures or Python wrappers.

all: class libclass.a

libclass.a: $(TOOLS) $(SOURCE) $(EXTERNAL)
	$(AR)  $@ $(addprefix build/, $(TOOLS) $(SOURCE) $(EXTERNAL))

class: $(TOOLS) $(SOURCE) $(EXTERNAL) $(OUTPUT) $(CLASS)
#	$(CC) $(OPTFLAG) $(OMPFLAG) $(LDFLAG) $(GSLFLAG) -o class $(addprefix build/,$(notdir $^)) -lgsl -lgslcblas -lm
	$(CPP) $(OPTFLAG) $(OMPFLAG) $(LDFLAG) -o class $(addprefix build/,$(notdir $^)) -lm
clean: .base
	rm -rf $(WRKDIR);
	rm -f libclass.a
