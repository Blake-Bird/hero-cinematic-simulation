"""Package the optional teaching folder separately from the scientific source release.

Includes notebook, interface, source and archived figure/array examples. Excludes
movies, generated lab runs, native binaries and machine environments. Requires
matching Research2 source at extraction; the optional folder is not standalone.
"""
from pathlib import Path
import argparse
import hashlib
import json
import zipfile

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
ALLOWED={'.py','.c','.md','.css','.js','.html','.ipynb','.png','.npz','.json'}


def package(destination):
    """Write and verify a new external ZIP; never overwrite an existing archive."""
    destination=Path(destination).expanduser().resolve()
    if destination.exists():raise FileExistsError('Choose a new archive destination')
    if destination.is_relative_to(ROOT):raise ValueError('Keep the optional ZIP outside the source tree')
    files=[]
    for p in HERE.rglob('*'):
        if '__pycache__' in p.parts or p.name.startswith('.') or p.suffix not in ALLOWED:continue
        if p.is_symlink():raise ValueError('Optional archive does not follow symlinks')
        if p.is_file():files.append(p)
    if sum(p.stat().st_size for p in files)>256*1024**2:raise ValueError('Optional reference pack exceeds the 256 MiB budget')
    catalogue=HERE/'reference_gallery/catalogue.json'
    if catalogue.exists():
        for row in json.loads(catalogue.read_text())['figures']:
            for key,checksum in [('image','image_sha256'),('arrays','arrays_sha256')]:
                path=(HERE/'reference_gallery'/row[key]).resolve()
                if not path.is_relative_to(HERE/'reference_gallery') or hashlib.sha256(path.read_bytes()).hexdigest()!=row[checksum]:
                    raise ValueError('Archived reference payload changed')
    records={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}
    destination.parent.mkdir(parents=True,exist_ok=True)
    temporary=destination.with_name(destination.name+'.partial')
    if temporary.exists():raise FileExistsError('Keep or remove an earlier partial archive explicitly')
    try:
        with zipfile.ZipFile(temporary,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
            for p in sorted(files):archive.write(p,p.relative_to(ROOT).as_posix())
            archive.writestr('optional-lab-manifest.json',json.dumps({'schema':'research2-optional-lab-archive-v1',
                'requires':'Matching Research2 source tree and installed environment','files_sha256':records,
                'scope':'Teaching files and read-only examples; no videos or native run inputs'},indent=2))
        with zipfile.ZipFile(temporary) as archive:
            for name,expected in records.items():
                if hashlib.sha256(archive.read(name)).hexdigest()!=expected:raise ValueError('ZIP payload mismatch')
        temporary.replace(destination)
    except BaseException:
        # A failed partial stays visible for inspection; never replace a valid archive.
        raise
    return {'path':str(destination),'files':len(files),'bytes':destination.stat().st_size,
            'sha256':hashlib.sha256(destination.read_bytes()).hexdigest()}

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True,type=Path)
    print(json.dumps(package(parser.parse_args().output),indent=2))
