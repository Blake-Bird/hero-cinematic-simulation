#!/bin/bash
eval ${OPENGADGET3_TEST_PREAMBLE_COMMAND}

# Verify that SYSTYPE variable is defined before continuing
if [ -z "$SYSTYPE" ]; then
    echo "SYSTYPE environment variable not defined. Defaulting."
    export SYSTYPE="Default"
    export OPENGADGET3_SERIAL_COMMAND="mpiexec -n 1 ./OpenGadget3 "
    export OPENGADGET3_EXE_COMMAND="mpiexec -n 2 ./OpenGadget3 "
    export OPENGADGET3_MID_COMMAND="mpiexec -n 4 ./OpenGadget3 "
    export OPENGADGET3_LARGE_COMMAND="mpiexec -n 6 ./OpenGadget3 "
    export OMP_NUM_THREADS=2
fi

# Set up the commands for each system
if [[ "$SYSTYPE" == Cast* ]]; then
    export OPENGADGET3_SERIAL_COMMAND="mpiexec -n 1 ./OpenGadget3 "
    export OPENGADGET3_EXE_COMMAND="mpiexec -n 4 ./OpenGadget3 "
    export OPENGADGET3_MID_COMMAND="mpiexec -n 6 ./OpenGadget3 "
    export OPENGADGET3_LARGE_COMMAND="mpiexec -n 8 ./OpenGadget3 "
    export OMP_NUM_THREADS=4
fi

echo SYSTYPE                    : $SYSTYPE
echo OPENGADGET3_SERIAL_COMMAND : $OPENGADGET3_SERIAL_COMMAND
echo OPENGADGET3_EXE_COMMAND    : $OPENGADGET3_EXE_COMMAND
echo OPENGADGET3_MID_COMMAND    : $OPENGADGET3_MID_COMMAND
echo OPENGADGET3_LARGE_COMMAND  : $OPENGADGET3_LARGE_COMMAND
