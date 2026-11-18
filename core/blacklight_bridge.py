"""Bounded Blacklight Kerr/GRMHD handoff, without simulated plasma fabrication.

The executable is independently built by the operator. A matching source checkout
and a pinned executable hash are recorded, but the bridge cannot prove that the
binary was built from that checkout. Read docs/blacklight.md before running.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import zipfile

from core.foundation import atomic_json, atomic_text, file_sha256

ROOT = Path(__file__).resolve().parents[1]
PIN = ROOT / 'configs' / 'blacklight' / 'upstream.json'


def _number(settings, key, *, lower=None, upper=None, strict=False):
    try:
        value = float(settings[key])
    except (KeyError, ValueError) as exc:
        raise ValueError('A finite explicit Blacklight '+key+' is required') from exc
    if not math.isfinite(value) or (lower is not None and (value <= lower if strict else value < lower)) or (upper is not None and value > upper):
        raise ValueError('Blacklight '+key+' is outside the supported range')
    return value


def _parse(path):
    settings = {}
    for line in Path(path).read_text().splitlines():
        text = line.split('#', 1)[0].strip()
        if not text:
            continue
        if text.count('=') != 1:
            raise ValueError('Blacklight inputs require one assignment per line')
        key, value = (part.strip() for part in text.split('=', 1))
        if not key or not value or key in settings or any(c.isspace() for c in key):
            raise ValueError('Blacklight inputs need distinct keys and explicit values')
        settings[key] = value
    return settings


def _source(source):
    pin = json.loads(PIN.read_text())
    source = Path(source).resolve()
    def git(*args):
        return subprocess.run(['git', '-C', str(source), *args], check=True, text=True,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout.strip()
    if git('rev-parse', 'HEAD') != pin['revision']:
        raise ValueError('Blacklight checkout is not at the pinned revision')
    if git('status', '--porcelain', '--untracked-files=all'):
        raise ValueError('Blacklight checkout must have no tracked or untracked changes')
    tracked = set(git('ls-files').splitlines())
    selected = sorted(name for name in tracked if name in {'Makefile', 'LICENSE'} or name.startswith('src/'))
    actual = {str(path.relative_to(source)) for path in (source / 'src').rglob('*')
              if path.is_file() and path.suffix in {'.cpp', '.hpp'}}
    if not {'Makefile', 'LICENSE'} <= set(selected) or actual - set(selected):
        raise ValueError('Unexpected Blacklight source files or missing license/build description')
    files = {name: file_sha256(source / name) for name in selected}
    if file_sha256(source / 'LICENSE') != file_sha256(ROOT / pin['license_file']):
        raise ValueError('Upstream license differs from retained license')
    return pin, files


def _inputs(settings, provenance, base):
    mode = settings.get('model_type')
    if mode not in {'formula', 'simulation'}:
        raise ValueError('Use an explicit formula or simulation Blacklight model')
    if provenance.get('schema') != 'blacklight-input-provenance-v1' or provenance.get('metric') != 'Kerr':
        raise ValueError('Declare blacklight-input-provenance-v1 and Kerr metric')
    for key in ('origin', 'description'):
        if not isinstance(provenance.get(key), str) or not provenance[key].strip():
            raise ValueError('Declare physical input origin and description')
    for key, expected in {'output_format':'npz', 'image_light':'true', 'image_num_frequencies':'1',
                          'adaptive_max_level':'0', 'render_num_images':'0', 'ray_flat':'false'}.items():
        if settings.get(key) != expected:
            raise ValueError('Bridge requires '+key+' = '+expected)
    for key in ('checkpoint_geodesic_save','checkpoint_geodesic_load','checkpoint_sample_save','checkpoint_sample_load','simulation_multiple','slow_light_on'):
        if settings.get(key, 'false') != 'false':
            raise ValueError('Single-snapshot bridge requires '+key+' = false')
    if settings.get('fallback_nan') != 'true':
        raise ValueError('Require fallback_nan = true to preserve failed-ray support')
    for key in ('num_threads','camera_resolution','ray_max_steps'):
        value = _number(settings,key,lower=1)
        if not settings[key].isdigit() or value != int(value):
            raise ValueError(key+' requires a decimal integer')
    if int(settings['camera_resolution']) > 8192 or int(settings['num_threads']) > 1024:
        raise ValueError('Declared image or thread count exceeds bridge bounds')
    for key in ('camera_r','camera_width','image_frequency','ray_step'):
        _number(settings,key,lower=0,strict=True)
    _number(settings,'camera_th',lower=0,upper=180)
    for key in ('camera_ph','camera_rotation','camera_urn','camera_uthn','camera_uphn','camera_k_r','camera_k_th','camera_k_ph'):
        _number(settings,key)
    if settings.get('camera_type') not in {'plane','pinhole'}:
        raise ValueError('Declare a supported Blacklight camera')
    if settings.get('image_normalization') not in {'camera','infinity'}:
        raise ValueError('Declare the frequency normalization frame')
    spin = _number(settings, 'formula_spin' if mode == 'formula' else 'simulation_a',lower=-1,upper=1)
    if _number(settings,'camera_r',lower=0,strict=True) <= 1 + math.sqrt(max(0., 1 - spin**2)):
        raise ValueError('Camera Kerr-Schild radius must be outside the outer horizon')
    if isinstance(provenance.get('spin'),bool) or not isinstance(provenance.get('spin'),(int,float)) or float(provenance['spin']) != spin:
        raise ValueError('Declared spin must equal the Blacklight input spin')
    inputs = {}
    if mode == 'formula':
        if provenance.get('origin') != 'analytic_illustration':
            raise ValueError('Formula mode must be explicitly labeled analytic_illustration')
        mass = _number(settings,'formula_mass',lower=0,strict=True)
        if provenance.get('mass_gravitational_cm') != mass:
            raise ValueError('Formula provenance must match the declared GM/c^2 in cm')
        for key in ('formula_r0','formula_nup','formula_cn0'):
            _number(settings,key,lower=0,strict=True)
        for key in ('formula_h','formula_l0','formula_q','formula_alpha','formula_a','formula_beta'):
            _number(settings,key)
    else:
        if provenance.get('origin') != 'independent_grmhd':
            raise ValueError('Simulation mode requires a separately supplied GRMHD snapshot')
        fmt, coord = settings.get('simulation_format'), settings.get('simulation_coord')
        if fmt not in {'athena','athenak','iharm3d','harm3d'} or coord not in {'sks','cks','mks','fmks'}:
            raise ValueError('Declare a supported simulation format and coordinate system')
        mass = _number(settings,'simulation_m_msun',lower=0,strict=True)
        density = _number(settings,'simulation_rho_cgs',lower=0,strict=True)
        for key, value in {'format':fmt,'coordinates':coord,'mass_msun':mass,'density_unit_g_cm3':density}.items():
            if provenance.get(key) != value:
                raise ValueError('GRMHD provenance does not match '+key)
        path = Path(settings['simulation_file']).expanduser()
        path = (base / path).resolve() if not path.is_absolute() else path.resolve()
        if not path.is_file() or file_sha256(path) != provenance.get('snapshot_sha256'):
            raise ValueError('GRMHD snapshot is missing or its declared hash differs')
        settings['simulation_file'] = str(path)
        inputs[str(path)] = provenance['snapshot_sha256']
        if settings.get('plasma_model') not in {'ti_te_beta','code_kappa'}:
            raise ValueError('Declare an upstream-supported plasma temperature model')
        for key in ('plasma_mu','plasma_ne_ni','plasma_rat_low','plasma_rat_high'):
            _number(settings,key,lower=0,strict=True)
        for key in ('plasma_power_frac','plasma_kappa_frac'):
            _number(settings,key,lower=0,upper=1)
        if not isinstance(provenance.get('emission_assumptions'),str) or not provenance['emission_assumptions'].strip():
            raise ValueError('Describe the assumed plasma and emission model')
    # Validate explicitly supplied numerical integration and cut controls for
    # finiteness, while upstream remains authoritative on detailed semantics.
    for key,value in settings.items():
        if key.startswith('cut_') and key not in {'cut_omit_near','cut_omit_far','cut_plane'}:
            _number(settings,key)
        if key in {'ray_factor','ray_tol_abs','ray_tol_rel'}:
            _number(settings,key,lower=0,strict=True)
    return mode, inputs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--binary',type=Path,required=True)
    parser.add_argument('--binary-sha256',required=True)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--provenance',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--memory-gib',type=float,required=True)
    parser.add_argument('--wall-seconds',type=int,required=True)
    parser.add_argument('--display-max',type=float,required=True,help='Fixed maximum source intensity for display')
    parser.add_argument('--display-stretch',type=float,required=True,help='Positive asinh stretch; save the same setting for comparable shots')
    parser.add_argument('--palette',choices=['ember','ice'],default='ember')
    parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args(argv)
    if (not math.isfinite(args.display_max) or args.display_max <= 0 or
            not math.isfinite(args.display_stretch) or args.display_stretch <= 0):
        raise ValueError('Positive finite fixed display maximum and stretch required')
    if not math.isfinite(args.memory_gib) or args.memory_gib <= 0 or args.wall_seconds < 1:
        raise ValueError('Positive memory and wall-time limits are required')
    source=args.source.resolve(); binary=args.binary.resolve(); output=args.output.resolve()
    if output == ROOT or ROOT in output.parents or source == output or source in output.parents:
        raise ValueError('Output must be outside the source trees')
    if output.exists():
        raise FileExistsError('Preserve prior output and select a fresh directory')
    pin, source_hashes = _source(source)
    if not binary.is_file() or not os.access(binary,os.X_OK) or file_sha256(binary)!=args.binary_sha256:
        raise ValueError('Executable must exist, be executable, and match its declared SHA-256')
    input_path=args.input.resolve(); provenance_path=args.provenance.resolve()
    settings=_parse(input_path); provenance=json.loads(provenance_path.read_text())
    mode, input_hashes = _inputs(settings, provenance, input_path.parent)
    resolution=int(settings['camera_resolution'])
    if resolution**2 * 1024 > args.memory_gib * 1024**3:
        raise MemoryError('Conservative camera workspace estimate exceeds memory allocation')
    inputs={str(input_path):file_sha256(input_path),str(provenance_path):file_sha256(provenance_path),
            str(binary):args.binary_sha256,str(PIN):file_sha256(PIN),str(Path(__file__).resolve()):file_sha256(__file__),**input_hashes}
    settings['output_file']=str(output/'image.npz')
    # Blacklight strips whitespace; reject unsafe paths instead of changing them.
    if any(any(ch.isspace() or ch in '#=' for ch in value) for value in settings.values()):
        raise ValueError('Blacklight values and paths cannot contain whitespace, # or =')
    output.mkdir(parents=True)
    text=''.join(key+' = '+value+'\n' for key,value in settings.items())
    atomic_text(output/'blacklight.input',text)
    inputs[str(output/'blacklight.input')]=file_sha256(output/'blacklight.input')
    command=[str(binary),str(output/'blacklight.input')]
    record={'schema':'blacklight-handoff-v1','status':'PREPARED','science_acceptance':'NOT_ACCEPTED',
            'upstream':pin,'source_path':str(source),'source_sha256':source_hashes,'binary_sha256':args.binary_sha256,
            'binary_build_binding':'UNVERIFIED: operator-supplied executable; source revision and binary hash are independent records',
            'model_type':mode,'input_provenance':provenance,'physical_input_evidence':'USER_DECLARED_NOT_INDEPENDENTLY_VALIDATED',
            'settings':settings,'input_hashes':inputs,'command':command,'memory_gib':args.memory_gib,'wall_seconds':args.wall_seconds,
            'display':{'palette':args.palette,'intensity_min':0.,'intensity_max':args.display_max,'asinh_stretch':args.display_stretch,
                       'scope':'Display mapping only; source I_nu NPZ remains the quantitative data'},
            'limitations':['No geodesic, emission, plasma, numerical-convergence or scientific acceptance has been established.',
                           'The input snapshot is independent GRMHD or a declared analytic illustration; GADGET does not resolve the horizon-scale plasma.',
                           'Memory estimate is not a process RSS limit; use scheduler limits and inspect logs.']}
    atomic_json(output/'handoff.json',record)
    if args.prepare_only:
        print(json.dumps({'status':'PREPARED','record':str(output/'handoff.json')},indent=2));return
    try:
        record['status']='RUNNING';atomic_json(output/'handoff.json',record)
        with (output/'blacklight.log').open('xb') as log:
            subprocess.run(command,cwd=output,stdout=log,stderr=subprocess.STDOUT,timeout=args.wall_seconds,check=True)
        for path,checksum in inputs.items():
            if file_sha256(path)!=checksum:raise ValueError('Input changed during calculation: '+path)
        if _source(source)[1] != source_hashes:raise ValueError('Pinned source changed during calculation')
        archive=output/'image.npz'
        with zipfile.ZipFile(archive) as z:
            if sum(item.file_size for item in z.infolist())*3 > args.memory_gib*1024**3:
                raise MemoryError('Image archive exceeds declared load workspace')
        import numpy as np
        with np.load(archive,allow_pickle=False) as z:
            intensity=np.asarray(z['I_nu'])
            if intensity.shape not in {(resolution,resolution),(1,resolution,resolution)}:
                raise ValueError('Intensity shape does not match the declared single-frequency camera')
            intensity=intensity.reshape(resolution,resolution)
            mask=np.isfinite(intensity) & (intensity>=0)
            record['image']={'path':str(archive),'sha256':file_sha256(archive),'quantity':'I_nu',
                             'finite_nonnegative_pixels':int(mask.sum()),'total_pixels':int(mask.size),
                             'unit':'erg s^-1 cm^-2 sr^-1 Hz^-1' }
            # A recorded, fixed display mapping preserves missing/negative rays
            # with transparency. The original floating-point array is untouched.
            scaled=np.zeros_like(intensity,dtype=float)
            scaled[mask]=np.clip(intensity[mask]/args.display_max,0,1)
            scaled=np.arcsinh(args.display_stretch*scaled)/np.arcsinh(args.display_stretch)
            anchors={'ember':[(0,0,0),(.18,.025,.06),(.72,.17,.045),(1,.73,.28),(1,1,.93)],
                     'ice':[(0,0,0),(.02,.04,.17),(.04,.38,.58),(.45,.83,.95),(.95,1,1)]}[args.palette]
            stops=np.linspace(0,1,len(anchors))
            rgb=np.stack([np.interp(scaled,stops,[row[c] for row in anchors]) for c in range(3)],axis=-1)
            rgba=np.concatenate((rgb,mask[...,None].astype(float)),axis=-1)
            from PIL import Image
            display_path=output/'display.png'
            Image.fromarray(np.flipud(np.rint(rgba*255).astype(np.uint8)),'RGBA').save(display_path)
            record['display'].update(path=str(display_path),sha256=file_sha256(display_path),
                masked_pixels=int(mask.size-mask.sum()),encoding='8-bit RGBA; transparent unsupported or negative intensities; row flip preserves upstream lower-origin view',
                saturated_pixels=int(np.count_nonzero(mask & (intensity>args.display_max))))
        record['status']='COMPLETE';atomic_json(output/'handoff.json',record)
    except BaseException as exc:
        record['status']='FAILED';record['failure']={'type':type(exc).__name__,'message':str(exc)}
        atomic_json(output/'handoff.json',record);raise
    print(json.dumps({'status':'COMPLETE','record':str(output/'handoff.json'),'science_acceptance':'NOT_ACCEPTED'},indent=2))


if __name__ == '__main__':
    main()
