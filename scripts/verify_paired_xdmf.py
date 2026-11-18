"""Run in ParaView's pvpython to load exact saved frames through the XDMF reader."""
import json
from pathlib import Path
import sys

from paraview.simple import XDMFReader

path, report = map(Path, sys.argv[1:])
reader = XDMFReader(FileNames=[str(path.resolve())])
reader.UpdatePipelineInformation()
times = list(reader.TimestepValues)
if len(times) != 86:
    raise RuntimeError(f'Expected 86 saved scale factors, found {len(times)}')
sample = []
for index in range(len(times)):
    reader.UpdatePipeline(times[index])
    points = reader.GetDataInformation().GetNumberOfPoints()
    if points != 2_097_152:
        raise RuntimeError(f'Snapshot {index} has {points} points')
    sample.append({'snapshot': index, 'scale_factor': times[index], 'loaded_points': points})
report.write_text(json.dumps({'timeline': str(path.resolve()), 'time_steps': times,
                              'checked_epochs': sample, 'status': 'PASS'}, indent=2) + '\n')
