#!/bin/bash
${OPENGADGET3_TEST_PREAMBLE_COMMAND}
source tests/configure_systype.sh
rm -rf snap_???
make clean CONFIG=tests/GalaxyMerger/Config.sh
make CONFIG=tests/GalaxyMerger/Config.sh -j
$OPENGADGET3_SERIAL_COMMAND  OpenGadget3 tests/GalaxyMerger/merger.par
python3 tests/GalaxyMerger/show_merger.py

# # create movie from frames
# ffmpeg -framerate 3 -i 0%02d.png -an -c:v libx264 merger.mp4
