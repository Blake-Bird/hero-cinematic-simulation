#!/bin/bash 
# get ICs
wget -q https://wwwmpa.mpa-garching.mpg.de/HydroSims/Tests/CosmologicalPerformanceTests/box4_hr/box4_hr.tar.gz
tar -xf box4_hr.tar.gz
rm box4_hr.tar.gz*

# get Coolingtables
wget -q https://wwwmpa.mpa-garching.mpg.de/HydroSims/Tests/CoolingTablesWSS08.tar.gz
tar -xf CoolingTablesWSS08.tar.gz
rm CoolingTablesWSS08.tar.gz*

# The SED files for the galaxy properties computed by SubFind:
wget -q https://homepages.usm.uni-muenchen.de/OpenGadget/SEDs/CB07_new.tar.gz
tar -xf CB07_new.tar.gz
rm CB07_new.tar.gz*

# The yield tables
wget -q https://wwwmpa.mpa-garching.mpg.de/HydroSims/Tests/CosmologicalPerformanceTests/AGB_Karakas.yields
wget -q https://wwwmpa.mpa-garching.mpg.de/HydroSims/Tests/CosmologicalPerformanceTests/Ia_thielemann.yields
wget -q https://wwwmpa.mpa-garching.mpg.de/HydroSims/Tests/CosmologicalPerformanceTests/Massive_Kobayashi.yields

# The IMF, SFR, metals, filter files
wget -q https://wwwmpa.mpa-garching.mpg.de/HydroSims/Tests/CosmologicalPerformanceTests/SFs
wget -q https://wwwmpa.mpa-garching.mpg.de/HydroSims/Tests/CosmologicalPerformanceTests/Chabrier_pow_new.IMF
wget -q https://wwwmpa.mpa-garching.mpg.de/HydroSims/Tests/CosmologicalPerformanceTests/metals.dat
wget -q https://wwwmpa.mpa-garching.mpg.de/HydroSims/Tests/CosmologicalPerformanceTests/filters.dat
