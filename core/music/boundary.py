"""Build and use MUSIC's native flux-matched boundary map and transpose.

The build is explicit and source-bound. The operator maps complete padded
coarse/fine potentials to the stencil's supported ghost faces. A successful
native call or adjoint comparison alone does not certify physical boundary
accuracy. Importing this module does not compile or load a native library.
"""
import ctypes
import json
import math
import os
import platform
import shlex
import shutil
import subprocess
from pathlib import Path
import numpy as np
from core.foundation import file_sha256, atomic_json, atomic_text
from core.music.replay import _budget

_CORE_ROOT = Path(__file__).resolve().parent.parent


def verify_boundary_adapter_build(receipt):
    path=Path(receipt).resolve()
    record=json.loads(path.read_text())
    if (path.with_name(path.name+'.sha256').read_text().strip()!=file_sha256(path)
            or record.get('schema')!='native-music-boundary-adapter-build-v1' or record.get('status')!='BUILT'
            or record.get('abi_version')!=1 or record.get('scalar_bytes')!=8):
        raise ValueError('Sealed current native boundary adapter build required')
    for source,expected in record['source_sha256'].items():
        if file_sha256(source)!=expected:
            raise ValueError('Native boundary adapter build source changed: '+source)
    binary=Path(record['binary']['path']).resolve()
    if file_sha256(binary)!=record['binary']['sha256']:
        raise ValueError('Native boundary adapter binary changed')
    return binary,record


class NativeBoundaryOperator:
    """Actual native flux-matched ghost map and its real Euclidean transpose.

    Inputs retain both complete coarse/fine potential interiors and ghosts.
    Output indices are only the axis-aligned ghost faces touched by the
    declared Laplacian stencil. They do not include an inferred boundary
    accuracy or treat source/potential inputs as independent random variables.
    """
    def __init__(self,build_receipt,*,order,coarse_shape,fine_shape,parent_offset,memory_gib=1.):
        self.handle=None
        self.memory_gib=memory_gib
        if (type(order) is not int or order not in (2,4,6)
                or any(len(s)!=3 or any(type(n) is not int or not 4<=n<=2**20 or n%2 for n in s) for s in (coarse_shape,fine_shape))
                or len(parent_offset)!=3 or any(type(n) is not int or n<0 for n in parent_offset)):
            raise ValueError('Actual even parent/fine geometry and native order2/4/6 required')
        if any(o+n//2>p for o,n,p in zip(parent_offset,fine_shape,coarse_shape)):
            raise ValueError('Native fine footprint escapes parent interior')
        self.coarse_shape=tuple(n+8 for n in coarse_shape)
        self.fine_shape=tuple(n+8 for n in fine_shape)
        inputs=math.prod(self.coarse_shape)+math.prod(self.fine_shape)
        band=math.prod(self.fine_shape)-math.prod(max(0,n-8) for n in fine_shape)
        _budget(memory_gib,64*1024**2+128*band+32*inputs+1024*96)
        binary,self.build=verify_boundary_adapter_build(build_receipt)
        self.binary=binary
        self.receipt=Path(build_receipt).resolve()
        from core.music.resources import native_boundary_count_profile,boundary_workspace
        self.allocation_plan=boundary_workspace(coarse_shape,fine_shape,parent_offset,native_boundary_count_profile(build_receipt,order))
        _budget(memory_gib,self.allocation_plan['minimum_workspace_bytes'])
        lib=self.library=ctypes.CDLL(str(binary))
        pointer=ctypes.c_void_p
        u64=ctypes.c_uint64
        ip=ctypes.POINTER(ctypes.c_int)
        dp=ctypes.POINTER(ctypes.c_double)
        lib.music_boundary_abi_version.argtypes=[]
        lib.music_boundary_abi_version.restype=u64
        if lib.music_boundary_abi_version()!=1:
            raise ValueError('Native boundary adapter ABI mismatch')
        lib.music_boundary_create.argtypes=[ctypes.c_int,ip,ip,ip,u64,pointer,u64]
        lib.music_boundary_create.restype=pointer
        lib.music_boundary_destroy.argtypes=[pointer]
        lib.music_boundary_destroy.restype=None
        for name in ('inputs','outputs','nodes'):
            method=getattr(lib,'music_boundary_'+name)
            method.argtypes=[pointer]
            method.restype=u64
        lib.music_boundary_output_indices.argtypes=[pointer,ctypes.POINTER(u64)]
        lib.music_boundary_output_indices.restype=None
        for name in ('apply','direct','transpose'):
            method=getattr(lib,'music_boundary_'+name)
            method.argtypes=[pointer,dp,dp,pointer,u64]
            method.restype=ctypes.c_int
        c=(ctypes.c_int*3)(*coarse_shape)
        f=(ctypes.c_int*3)(*fine_shape)
        offset=(ctypes.c_int*3)(*parent_offset)
        error=ctypes.create_string_buffer(2048)
        self.handle=lib.music_boundary_create(order,c,f,offset,int(memory_gib*1024**3),error,len(error))
        if not self.handle:
            message=error.value.decode()
            if 'memory' in message or 'bad_alloc' in message:
                raise MemoryError(message)
            raise ValueError(message)
        self.input_count=int(lib.music_boundary_inputs(self.handle))
        self.output_count=int(lib.music_boundary_outputs(self.handle))
        self.nodes=int(lib.music_boundary_nodes(self.handle))
        if self.input_count!=inputs:
            self.close()
            raise ValueError('Native boundary adapter input geometry mismatch')
        if self.nodes!=self.allocation_plan['linear_tape_nodes'] or self.output_count!=self.allocation_plan['axis_stencil_ghost_rows']:
            self.close()
            raise ValueError('Native boundary tape differs from its source-derived operation count')
        self.output_indices=np.empty(self.output_count,dtype=np.uint64)
        lib.music_boundary_output_indices(self.handle,self.output_indices.ctypes.data_as(ctypes.POINTER(u64)))
        self.source_binding={'build_receipt':{'path':str(self.receipt),'sha256':file_sha256(self.receipt)},'binary':self.build['binary'],
            'order':order,'coarse_interior_shape':list(coarse_shape),'fine_interior_shape':list(fine_shape),
            'fine_parent_offset':list(parent_offset),'potential_halo':4,'linear_tape_nodes':self.nodes,
            'axis_stencil_ghost_rows':self.output_count,'memory_budget_gib':memory_gib}
        self.source_binding['allocation_plan']=self.allocation_plan

    def close(self):
        if self.handle:
            self.library.music_boundary_destroy(self.handle)
            self.handle=None
    def __enter__(self):
        return self
    def __exit__(self,*args):
        self.close()
    def __del__(self):
        if getattr(self,'handle',None):
            self.close()

    def _call(self,name,inputs,size):
        if not self.handle:
            raise ValueError('Native boundary operator is closed')
        _budget(self.memory_gib,64*1024**2+96*self.nodes+40*self.input_count+24*self.output_count)
        inputs=np.ascontiguousarray(inputs,dtype=np.float64)
        output=np.empty(size,dtype=np.float64)
        error=ctypes.create_string_buffer(2048)
        method=getattr(self.library,'music_boundary_'+name)
        dp=ctypes.POINTER(ctypes.c_double)
        if method(self.handle,inputs.ctypes.data_as(dp),output.ctypes.data_as(dp),error,len(error)):
            raise ValueError(error.value.decode())
        if not np.isfinite(output).all():
            raise ValueError('Native boundary response overflow')
        return output

    def apply(self,coarse,fine,*,direct=False):
        c=np.asarray(coarse)
        f=np.asarray(fine)
        if (type(direct) is not bool or c.shape!=self.coarse_shape or f.shape!=self.fine_shape
                or any(v.dtype.kind not in 'fiu' or not np.isfinite(v).all() for v in (c,f))):
            raise ValueError('Actual finite aligned binary64 potential/ghost inputs required')
        return self._call('direct' if direct else 'apply',np.concatenate((c.ravel(),f.ravel())),self.output_count)

    def transpose(self,weights):
        w=np.asarray(weights)
        if w.shape!=(self.output_count,) or w.dtype.kind not in 'fiu' or not np.isfinite(w).all():
            raise ValueError('Finite aligned native stencil-ghost weights required')
        result=self._call('transpose',w,self.input_count)
        n=math.prod(self.coarse_shape)
        return result[:n].reshape(self.coarse_shape),result[n:].reshape(self.fine_shape)


def prepare_boundary_adapter(destination,*,compiler=None,include_directories=()):
    """Compile the existing serial native operator; never regenerate ICs.

    GCC and Clang builds use standard C++17 shared-library options. FFTW and
    OpenMP headers must be available, even though this adapter does not run
    threaded loops or FFTs. Extra include paths can be supplied explicitly.
    Existing completed builds are reused only when their bindings verify.
    """
    target=Path(destination).expanduser().resolve()
    receipt=target/'receipt.json'
    if target.exists():
        binary,record=verify_boundary_adapter_build(receipt)
        if compiler is not None and Path(shutil.which(compiler) or compiler).resolve()!=Path(record['command'][0]).resolve():
            raise ValueError('Existing adapter was built with a different requested compiler')
        if any(str(Path(p).expanduser().resolve()) not in record['include_directories'] for p in include_directories):
            raise ValueError('Existing adapter does not bind the requested include directories')
        return record
    selected=compiler or os.environ.get('CXX') or next((p for name in ('g++-16','g++-15','g++-14','g++','clang++') if (p:=shutil.which(name))),None)
    if not isinstance(selected,str) or not selected or len(shlex.split(selected))!=1:
        raise ValueError('One explicit C++ compiler executable required')
    executable=shutil.which(selected)
    if executable is None:
        raise FileNotFoundError('Native C++ compiler not found: '+selected)
    executable=Path(executable).resolve()
    if platform.system() not in ('Darwin','Linux'):
        raise ValueError('Use a Unix C++ toolchain (Linux/WSL or macOS) for the native adapter')
    root=_CORE_ROOT
    source=root/'native_boundary_operator.cc'
    headers=root/'software/music2/src'
    staged_headers=target/'headers'
    includes=[*[Path(p).expanduser().resolve() for p in include_directories]]
    pkg=shutil.which('pkg-config')
    if pkg:
        probe=subprocess.run([pkg,'--cflags-only-I','fftw3'],capture_output=True,text=True)
        if probe.returncode==0:
            tokens=shlex.split(probe.stdout)
            i=0
            while i<len(tokens):
                token=tokens[i]
                if token=='-I':
                    i+=1
                    includes.append(Path(tokens[i]).resolve())
                elif token.startswith('-I'):
                    includes.append(Path(token[2:]).resolve())
                i+=1
    includes=list(dict.fromkeys(includes))
    if any(not p.is_dir() for p in includes):
        raise FileNotFoundError('Native adapter include directory is missing')
    configuration_template=headers/'cmake_config.hh.in'
    header_names=('general.hh','fd_schemes.hh','mg_operators.hh','mg_interp.hh','logger.hh','config_file.hh')
    paths=[source,configuration_template,executable,Path(__file__).resolve(),*[headers/name for name in header_names]]
    before={str(p):file_sha256(p) for p in paths}
    version=subprocess.run([str(executable),'--version'],check=True,capture_output=True,text=True).stdout
    target.mkdir(parents=True,exist_ok=False)
    staged_headers.mkdir()
    for name in header_names:
        shutil.copyfile(headers/name,staged_headers/name)
    configuration=staged_headers/'cmake_config.hh'
    text=configuration_template.read_text().replace('${CODE_PRECISION}','DOUBLE').replace('${CMAKE_BUILD_TYPE}','Release')
    if '${' in text:
        raise ValueError('Unresolved native precision configuration variable')
    configuration.write_text(text)
    before[str(configuration)]=file_sha256(configuration)
    includes.insert(0,staged_headers)
    binary=target/('libmusic_boundary.dylib' if platform.system()=='Darwin' else 'libmusic_boundary.so')
    dependencies=target/'dependencies.d'
    command=[str(executable),'-std=c++17','-O2','-shared','-fPIC','-MMD','-MF',str(dependencies),
        *['-I'+str(p) for p in includes],str(source),'-o',str(binary)]
    try:
        with (target/'build.log').open('x') as log:
            subprocess.run(command,check=True,stdout=log,stderr=subprocess.STDOUT)
        # Retain actual non-system transitive includes, with makefile escapes
        # decoded for source trees or include directories containing spaces.
        contents=dependencies.read_text().replace('\\\n','')
        tokens=shlex.split(contents.split(':',1)[1])
        for name in tokens:
            p=Path(name).resolve()
            if p.is_file():
                before.setdefault(str(p),file_sha256(p))
        for p,digest in before.items():
            if file_sha256(p)!=digest:
                raise ValueError('Native adapter source changed during compilation: '+p)
        library=ctypes.CDLL(str(binary))
        library.music_boundary_abi_version.argtypes=[]
        library.music_boundary_abi_version.restype=ctypes.c_uint64
        if library.music_boundary_abi_version()!=1:
            raise ValueError('Compiled native boundary adapter ABI mismatch')
        result={'schema':'native-music-boundary-adapter-build-v1','status':'BUILT','abi_version':1,'scalar_bytes':8,
            'binary':{'path':str(binary),'sha256':file_sha256(binary)},'source_sha256':before,'command':command,
            'compiler_version':version,'platform':platform.platform(),'include_directories':list(map(str,includes)),
            'scope':'Compiled standalone native boundary forward/transpose adapter; ABI checked, no IC generation, evolution or science acceptance',
            'science_acceptance':'NOT_ACCEPTED'}
        atomic_json(receipt,result)
        atomic_text(receipt.with_name(receipt.name+'.sha256'),file_sha256(receipt)+'\n')
        verify_boundary_adapter_build(receipt)
        return result
    except BaseException as error:
        atomic_json(target/'failure.json',{'status':'FAIL','command':command,'error':str(error),'build_log':str(target/'build.log')})
        raise
