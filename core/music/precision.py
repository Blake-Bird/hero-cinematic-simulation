"""Exact binary128 storage and its explicit native arithmetic adapter.

Retained fields stay as IEEE754 integer word pairs. NumPy longdouble is not
assumed to represent binary128, and there is no implicit float conversion or
precision fallback. The native reference builds only when its caller asks for
wide-field arithmetic; its derivative observables round to binary64 at output.
"""
import ctypes
import json
import os
import platform
import shutil
import subprocess
from fractions import Fraction
from pathlib import Path
import numpy as np
from core.foundation import file_sha256, canonical_digest, atomic_json, atomic_text
from core.native import run_native_command, native_run_lock
from core.resources import runtime_paths

_CORE_ROOT = Path(__file__).resolve().parent.parent


class Binary128Array:
    scalar_bytes=16
    encoding='IEEE754_binary128'

    def __init__(self,words,*,byte_order):
        words=np.asarray(words)
        if (byte_order not in ('<','>') or words.ndim!=4 or words.shape[-1]!=2
                or words.dtype.kind!='u' or words.dtype.itemsize!=8):
            raise ValueError('Three-dimensional binary128 word-pair grid required')
        self._words=np.array(words,dtype=np.uint64,copy=True,order='C')
        self._words.setflags(write=False)
        self.byte_order=byte_order
        self.shape=self._words.shape[:-1]
        self.size=int(np.prod(self.shape))
        high=self._words[...,1 if byte_order=='<' else 0].reshape(-1)
        for start in range(0,len(high),131072):
            if np.any(((high[start:start+131072]>>48)&0x7fff)==0x7fff):
                raise ValueError('Finite IEEE binary128 samples required')

    def __array__(self,*args,**kwargs):
        raise TypeError('Binary128 potential requires an explicit precision-preserving arithmetic adapter')

    def words(self):
        """Return a read-only view in original word order, without conversion."""
        result=self._words.view()
        result.setflags(write=False)
        return result

    def bits(self,index):
        row=self._words[index]
        if row.shape!=(2,):
            raise ValueError('One exact binary128 cell index required')
        low,high=(int(row[0]),int(row[1])) if self.byte_order=='<' else (int(row[1]),int(row[0]))
        return (high<<64)|low

    def fraction(self,index):
        """Decode sign * significand * 2**power as an exact rational.

        Normal values restore the implicit 113th significand bit. Subnormals
        use exponent 1-bias without that bit; the original sign/zero bits
        remain in words(). No field-sized Python object array is allocated.
        """
        bits=self.bits(index)
        exponent=(bits>>112)&0x7fff
        significand=bits&((1<<112)-1)
        if exponent:
            significand|=1<<112
        power=(exponent or 1)-16383-112
        value=Fraction(significand<<power) if power>=0 else Fraction(significand,1<<-power)
        return -value if bits>>127 else value


def prepare_binary128_reference(*,compiler=None):
    """Build/reuse the bounded reference in external runtime state.

    Called only when actual wider-field arithmetic is requested. A compiler
    supporting IEEE binary128 is required; there is no precision fallback.
    """
    selected=compiler or os.environ.get('CXX') or next((name for name in ('g++-16','g++-15','g++-14','g++') if shutil.which(name)),None)
    executable=shutil.which(selected) if selected else None
    if not executable:
        raise FileNotFoundError('IEEE binary128 C++ compiler required for wider potential reference')
    executable=Path(executable).resolve()
    source=(_CORE_ROOT / 'music_binary128_reference.cc')
    sources={str(p.resolve()):file_sha256(p) for p in (source,Path(__file__),executable)}
    identity=canonical_digest({'sources':sources,'flags':['-std=c++17','-O2','-shared','-fPIC']})
    directory=runtime_paths().build_root/'native/music-binary128-reference'/identity
    receipt=directory/'receipt.json'
    binary=directory/('libmusic_binary128.dylib' if platform.system()=='Darwin' else 'libmusic_binary128.so')
    with native_run_lock(directory,operation='prepare wider potential arithmetic reference'):
        if receipt.exists():
            record=json.loads(receipt.read_text())
            if (receipt.with_name(receipt.name+'.sha256').read_text().strip()!=file_sha256(receipt)
                    or record.get('schema')!='music-binary128-reference-build-v1' or record.get('status')!='BUILT'
                    or record.get('source_sha256')!=sources or file_sha256(binary)!=record.get('binary',{}).get('sha256')):
                raise ValueError('Wider arithmetic reference build binding changed')
            return record
        directory.mkdir(parents=True,exist_ok=True)
        if binary.exists():
            raise FileExistsError('Keep incomplete wider reference build evidence; choose fresh build inputs')
        command=[str(executable),'-std=c++17','-O2','-shared','-fPIC',str(source),'-o',str(binary)]
        with (directory/'build.log').open('x') as log:
            result=run_native_command(command,cwd=directory,inputs=[source,Path(__file__),executable],receipt_path=directory/'invocation.json',stdout=log,stderr=subprocess.STDOUT,timeout=60,check=False)
        if result.returncode:
            raise RuntimeError('Wider arithmetic reference compile failed; inspect '+str(directory/'build.log'))
        for name,digest in sources.items():
            if file_sha256(name)!=digest:
                raise ValueError('Wider reference source changed during compilation')
        library=ctypes.CDLL(str(binary))
        library.music_binary128_reference_abi.restype=ctypes.c_uint64
        if library.music_binary128_reference_abi()!=1:
            raise ValueError('Wider reference ABI mismatch')
        record={'schema':'music-binary128-reference-build-v1','status':'BUILT','abi_version':1,'scalar_bytes':16,
            'binary':{'path':str(binary),'sha256':file_sha256(binary)},'source_sha256':sources,'command':command,
            'scope':'Independent rational finite-difference and weighted sum reference; rounds observables only after wider arithmetic',
            'science_acceptance':'NOT_ACCEPTED'}
        atomic_json(receipt,record)
        atomic_text(receipt.with_name(receipt.name+'.sha256'),file_sha256(receipt)+'\n')
        return record


def _library():
    record=prepare_binary128_reference()
    library=ctypes.CDLL(record['binary']['path'])
    u64=ctypes.c_uint64
    p64=ctypes.POINTER(u64)
    dp=ctypes.POINTER(ctypes.c_double)
    library.music_binary128_reference.argtypes=[p64,u64,ctypes.c_int,p64,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,dp,u64,ctypes.c_void_p,u64]
    library.music_binary128_reference.restype=ctypes.c_int
    library.music_binary128_dot.argtypes=[p64,u64,ctypes.c_int,dp,u64,dp,ctypes.c_void_p,u64]
    library.music_binary128_dot.restype=ctypes.c_int
    return library,record


def reference_binding_sources():
    """Current verified compiler/source/binary pins for numerical receipts."""
    record=prepare_binary128_reference()
    return {**record['source_sha256'],record['binary']['path']:record['binary']['sha256'],
        **{str(p):file_sha256(p) for p in ((_CORE_ROOT / 'music/replay.py'),(_CORE_ROOT / 'foundation.py'))}}


def wide_derivative(values,*,operation,level,order=6,halo=4,memory_gib=1.):
    from core.music.replay import _budget
    if (not isinstance(values,Binary128Array) or operation not in ('poisson','gradient','quadratic')
            or type(level) is not int or not 1<=level<=20 or type(order) is not int or order not in (2,4,6)
            or type(halo) is not int or not 0<=halo<=16):
        raise ValueError('Explicit wider field, supported derivative, order, clock geometry and workspace required')
    radius=(2 if order==2 else 4) if operation=='quadratic' else order//2
    if halo<radius or any(n<=2*halo for n in values.shape):
        raise ValueError('Wider derivative requires actual potential ghosts')
    shape=tuple(n-2*halo for n in values.shape)
    cells=int(np.prod(shape))
    components=3 if operation=='gradient' else 1
    _budget(memory_gib,64*1024**2+16*values.size+8*components*cells)
    words=values.words()
    output=np.empty(shape+(3,) if components==3 else shape,dtype=np.float64)
    library,_=_library()
    u64=ctypes.c_uint64
    p64=ctypes.POINTER(u64)
    dp=ctypes.POINTER(ctypes.c_double)
    error=ctypes.create_string_buffer(2048)
    dimensions=(u64*3)(*values.shape)
    status=library.music_binary128_reference(words.ctypes.data_as(p64),words.size,int(values.byte_order=='<'),dimensions,halo,level,order,
        {'poisson':0,'gradient':1,'quadratic':2}[operation],output.ctypes.data_as(dp),output.size,error,len(error))
    if status:
        raise ValueError(error.value.decode())
    if not np.isfinite(output).all():
        raise ValueError('Wider derivative observable is nonfinite')
    return output


def wide_dot(values,weights,*,memory_gib=1.):
    from core.music.replay import _budget
    if not isinstance(values,Binary128Array):
        raise ValueError('Original binary128 field required')
    weights=np.asarray(weights)
    if weights.shape!=values.shape or weights.dtype.kind not in 'fiu':
        raise ValueError('Aligned real wider-field response required')
    _budget(memory_gib,64*1024**2+32*values.size)
    weights=np.ascontiguousarray(weights,dtype=np.float64)
    words=values.words()
    output=ctypes.c_double()
    error=ctypes.create_string_buffer(2048)
    library,_=_library()
    u64=ctypes.c_uint64
    p64=ctypes.POINTER(u64)
    dp=ctypes.POINTER(ctypes.c_double)
    status=library.music_binary128_dot(words.ctypes.data_as(p64),words.size,int(values.byte_order=='<'),weights.ctypes.data_as(dp),weights.size,ctypes.byref(output),error,len(error))
    if status:
        raise ValueError(error.value.decode())
    return output.value
