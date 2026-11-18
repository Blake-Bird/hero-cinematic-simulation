"""Bounded NumPy storage: streamed NPZ contracts and disk-backed publication.

Numeric identity arrays keep their exact dtype. Members are consumed in C order
under an explicit workspace limit; ZIP member CRCs and canonical payload hashes
are verified by their callers without materializing the whole archive.
"""
from collections.abc import MutableMapping, Mapping
from pathlib import Path
import hashlib
import math
import shutil
import zipfile
import numpy as np

def c_order_slices(shape, itemsize, memory_bytes):
    """Yield C-order slabs, including fields whose full leading row is too large.

    Slice payloads fit the declared read workspace. HDF5 chunk-cache overhead
    remains separate; this does not claim a total process RSS bound.
    """
    if type(memory_bytes) is not int or memory_bytes <= 0:
        raise ValueError('Numeric read workspace must be positive integer bytes')
    if type(itemsize) is not int or itemsize <= 0:
        raise ValueError('Numeric item size must be a positive integer')
    if itemsize > memory_bytes:
        raise MemoryError('Numeric scalar exceeds read workspace')
    if not shape:
        yield ()
        return
    if any(length == 0 for length in shape):
        return
    axis = 0
    while math.prod(shape[axis + 1:]) * itemsize > memory_bytes:
        axis += 1
    slab_bytes = math.prod(shape[axis + 1:]) * itemsize
    rows = max(1, memory_bytes // slab_bytes)
    for prefix in np.ndindex(tuple(shape[:axis])):
        for start in range(0, shape[axis], rows):
            yield prefix + (slice(start, min(start + rows, shape[axis])),) + (slice(None),) * (len(shape) - axis - 1)


class _ArrayDataset:
    """Array adapter for the shared bounded numeric dataset checksum/semantics."""

    def __init__(self, array):
        self.array = array
        self.attrs = {}
        self.shape = array.shape
        self.dtype = array.dtype
        self.ndim = array.ndim
        self.size = array.size
        self.fletcher32 = False

    def __len__(self):
        return len(self.array)

    def __getitem__(self, key):
        return self.array[key]

class ZipNumericDataset:
    """Sequential C-order slices used by the canonical payload hasher."""

    def __init__(self, stream, shape, dtype, offset, memory_bytes):
        self.stream = stream
        self.shape = shape
        self.dtype = dtype
        self.offset = offset
        self.memory_bytes = memory_bytes
        self.ndim = len(shape)
        self.size = math.prod(shape)
        self.attrs = {}
        self.fletcher32 = False

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, selection):
        if self.ndim == 0:
            position = self.offset
            count = 1
            shape = ()
        else:
            keys = selection if isinstance(selection, tuple) else (selection,)
            keys = keys + (slice(None),) * (self.ndim - len(keys))
            coordinates = []
            shape = []
            sliced = False
            for axis, (key, length) in enumerate(zip(keys, self.shape)):
                if isinstance(key, (int, np.integer)) and (not sliced):
                    if not 0 <= int(key) < length:
                        raise IndexError('NPY checksum coordinate outside array')
                    coordinates.append(int(key))
                elif isinstance(key, slice):
                    start, stop, step = key.indices(length)
                    if step != 1 or (sliced and (start != 0 or stop != length)):
                        raise ValueError('NPY checksum read must be contiguous C order')
                    coordinates.append(start)
                    shape.append(max(0, stop - start))
                    sliced = True
                else:
                    raise ValueError('NPY checksum selection must be a C-order contiguous slab')
            flat = sum((i * math.prod(self.shape[axis + 1:]) for axis, i in enumerate(coordinates)))
            position = self.offset + flat * self.dtype.itemsize
            shape = tuple(shape)
            count = math.prod(shape)
        length = count * self.dtype.itemsize
        if length > self.memory_bytes:
            raise MemoryError('NPY payload read exceeds checksum workspace')
        # Sequential reads avoid restarting decompression for arbitrary seeks.
        if self.stream.tell() != position:
            raise ValueError('NPY checksum reader requires sequential C-order traversal')
        payload = self.stream.read(length)
        if len(payload) != length:
            raise ValueError('Truncated NPY numeric payload')
        return np.frombuffer(payload, dtype=self.dtype, count=count).reshape(shape)

def numpy_member_datasets(zipped, *, memory_bytes, require_c_order=False):
    """Yield one member at a time, never materializing a large C-order array.

    Small Fortran-order members keep the former bounded in-memory path. A
    large Fortran member still requires explicit canonicalization by its
    producer; it cannot be silently assigned a C-order byte checksum.
    """
    for info in zipped.infolist():
        if not info.filename.endswith('.npy'):
            raise ValueError('NPZ contract member must be a pickle-free NPY array')
        with zipped.open(info) as stream:
            version = np.lib.format.read_magic(stream)
            if version == (1, 0):
                shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream, max_header_size=10000)
            elif version == (2, 0):
                shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream, max_header_size=10000)
            elif version == (3, 0):
                import ast, struct
                raw = stream.read(4)
                if len(raw) != 4:
                    raise ValueError('Truncated NPY header length')
                length = struct.unpack('<I', raw)[0]
                if length > 10000:
                    raise ValueError('NPY header exceeds supported parsing workspace')
                raw = stream.read(length)
                if len(raw) != length:
                    raise ValueError('Truncated NPY header')
                header = ast.literal_eval(raw.decode('utf8'))
                if not isinstance(header, dict) or set(header) != {'shape', 'fortran_order', 'descr'} or type(header['fortran_order']) is not bool or (not isinstance(header['shape'], tuple)) or any((type(n) is not int or n < 0 for n in header['shape'])):
                    raise ValueError('Invalid NPY version3 header')
                shape = header['shape']
                fortran = header['fortran_order']
                dtype = np.lib.format.descr_to_dtype(header['descr'])
            else:
                raise ValueError('Unsupported NPY format version')
            if dtype.hasobject:
                raise ValueError('Object NPY payloads are not supported')
            payload_offset = stream.tell()
            payload_bytes = math.prod(shape) * dtype.itemsize
            if payload_offset + payload_bytes != info.file_size:
                raise ValueError('NPY header shape/dtype differs from ZIP member size')
            if fortran:
                if require_c_order:
                    raise ValueError('Row-streamed NPY input requires C-order publication')
                if info.file_size > memory_bytes:
                    raise MemoryError('Large Fortran NPY member requires producer C-order canonicalization')
                stream.seek(0)
                array = np.lib.format.read_array(stream, allow_pickle=False)
                dataset = _ArrayDataset(array)
            else:
                dataset = ZipNumericDataset(stream, shape, dtype, payload_offset, memory_bytes)
            yield (info.filename[:-4], dataset)
            # Drain metadata and empty members too, so ZIP CRC validation finishes.
            while stream.read(min(memory_bytes, 1024 * 1024)):
                pass

def validate_response_workspace(settings, *, combined=False):
    keys = {'source_disk_gib', 'noise_disk_gib'} if combined else {'disk_gib'}
    if not isinstance(settings, dict) or set(settings) != keys or any((isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, float)) or (not math.isfinite(v)) or (v <= 0) for v in settings.values())):
        raise ValueError('Explicit positive response publication disk allocation required')

def response_disk_bound(payload_bytes, array_count):
    if type(payload_bytes) is not int or payload_bytes < 0 or type(array_count) is not int or (array_count <= 0):
        raise ValueError('Nonnegative response payload and positive array count required')
    npy = payload_bytes + 65536 * array_count
    archive = npy + 8 * (npy // 16384 + array_count) + 65536 * array_count
    return npy + archive + 1024 ** 2

def preflight_response_disk(destination, disk_gib, *, payload_bytes, array_count):
    validate_response_workspace({'disk_gib': disk_gib})
    required = response_disk_bound(payload_bytes, array_count)
    if required > disk_gib * 1024 ** 3:
        raise OSError('Complete response spool/publication exceeds declared disk allocation before output')
    parent = Path(destination).resolve().parent
    while not parent.exists():
        parent = parent.parent
    if shutil.disk_usage(parent).free < required:
        raise OSError('Insufficient free space for complete response spool/publication')
    return {'payload_upper_bound_bytes': payload_bytes, 'array_count_upper_bound': array_count, 'disk_upper_bound_bytes': required, 'disk_allocation_gib': disk_gib, 'scope': 'Retained NPY workspace plus conservative ZIP publication and headers'}

class NumericArraySpool(MutableMapping):
    """Persist each finished array/row; publish without loading them together.

    The disk allocation includes retained NPY work files and the conservative
    ZIP publication bound. Incomplete workspaces are never reopened as valid
    results. Mathematical/numerical validation remains the producer's job.
    """

    def __init__(self, directory, *, disk_gib, memory_gib=1.0):
        validate_response_workspace({'disk_gib': disk_gib})
        if isinstance(memory_gib, (bool, np.bool_)) or not math.isfinite(memory_gib) or memory_gib <= 0:
            raise ValueError('Positive spool read workspace required')
        self.directory = Path(directory).resolve()
        self.budget = int(disk_gib * 1024 ** 3)
        self.memory_bytes = int(memory_gib * 1024 ** 3)
        if self.directory.exists():
            raise FileExistsError('Preserve previous numeric response workspace')
        self.directory.mkdir(parents=True, exist_ok=False)
        self.entries = {}
        self.total = 0
        self.published = False

    def __iter__(self):
        return iter(self.entries)

    def __len__(self):
        return len(self.entries)

    def __contains__(self, key):
        return key in self.entries

    def __delitem__(self, key):
        raise TypeError('Numeric spool records are immutable')

    @staticmethod
    def _zip_bound(size):
        return size + 8 * (size // 16384 + 1) + 65536

    def _reserve(self, key, size):
        if self.published or not isinstance(key, str) or (not key) or ('\x00' in key) or (key in self.entries):
            raise ValueError('New distinct spool array name required before publication')
        final_bound = sum((self._zip_bound(e['file_bytes']) for e in self.entries.values())) + self._zip_bound(size)
        proposed = self.total + size + final_bound + 1024 ** 2
        if proposed > self.budget:
            raise OSError('Response spool plus publication exceeds declared disk allocation')
        if shutil.disk_usage(self.directory).free < size + final_bound + 1024 ** 2:
            raise OSError('Insufficient free space for response spool and publication')
        return self.directory / (hashlib.sha256(key.encode()).hexdigest() + '.npy')

    def __setitem__(self, key, value):
        array = np.asarray(value)
        if array.dtype.hasobject:
            raise ValueError('Pickle/object response arrays are forbidden')
        if array.ndim and (not array.flags.c_contiguous):
            array = np.ascontiguousarray(array)
        # Reserve both the retained NPY and final archive before creating files.
        size = array.nbytes + 65536
        path = self._reserve(key, size)
        with path.open('xb') as stream:
            np.lib.format.write_array(stream, array, allow_pickle=False)
        actual = path.stat().st_size
        self.entries[key] = {'path': path, 'shape': array.shape, 'dtype': array.dtype, 'file_bytes': actual, 'complete': True}
        self.total += actual

    def create_rows(self, key, shape, *, dtype='<f8'):
        shape = tuple(shape)
        dtype = np.dtype(dtype)
        if not shape or any((type(n) is not int or n <= 0 for n in shape)) or dtype.hasobject:
            raise ValueError('Positive row matrix dimensions and pickle-free dtype required')
        size = math.prod(shape) * dtype.itemsize
        path = self._reserve(key, size + 65536)
        with path.open('xb') as stream:
            np.lib.format.write_array_header_1_0(stream, {'descr': np.lib.format.dtype_to_descr(dtype), 'fortran_order': False, 'shape': shape})
            offset = stream.tell()
            stream.truncate(offset + size)
        actual = path.stat().st_size
        self.total += actual
        self.entries[key] = {'path': path, 'shape': shape, 'dtype': dtype, 'file_bytes': actual, 'complete': False, 'offset': offset, 'next_row': 0}

    def write_row(self, key, index, value):
        entry = self.entries[key]
        array = np.asarray(value)
        if entry['complete'] or type(index) is not int or index != entry['next_row'] or (array.shape != entry['shape'][1:]) or (array.dtype != entry['dtype']):
            raise ValueError('Exact next response row and declared dtype required')
        if not array.flags.c_contiguous:
            array = np.ascontiguousarray(array)
        with entry['path'].open('r+b') as stream:
            stream.seek(entry['offset'] + index * array.nbytes)
            array.tofile(stream)
        entry['next_row'] += 1
        entry['complete'] = entry['next_row'] == entry['shape'][0]

    def __getitem__(self, key):
        entry = self.entries[key]
        if not entry['complete']:
            raise ValueError('Incomplete response spool array')
        if entry['file_bytes'] * 3 > self.memory_bytes:
            raise MemoryError('Requested full spool array exceeds read workspace')
        return np.load(entry['path'], allow_pickle=False)

    def copy_npy_member(self, key, archive):
        """Copy a verified source member without materializing identities/masks."""
        info = archive.getinfo(key + '.npy')
        path = self._reserve(key, info.file_size)
        with archive.open(info) as source, path.open('xb') as target:
            shutil.copyfileobj(source, target, length=min(self.memory_bytes, 1024 ** 2))
        # The caller has already verified the source product and archive bytes.
        with path.open('rb') as stream:
            version = np.lib.format.read_magic(stream)
            if version == (1, 0):
                shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
            elif version == (2, 0):
                shape, _, dtype = np.lib.format.read_array_header_2_0(stream)
            else:
                raise ValueError('Copied numeric response metadata require NPY1/2')
        if dtype.hasobject:
            raise ValueError('Object response source member forbidden')
        self.entries[key] = {'path': path, 'shape': shape, 'dtype': dtype, 'file_bytes': info.file_size, 'complete': True}
        self.total += info.file_size

    def write_npz(self, path):
        if self.published or any((not e['complete'] for e in self.entries.values())):
            raise ValueError('Complete unpublished spool required')
        if shutil.disk_usage(Path(path).resolve().parent).free < sum((self._zip_bound(e['file_bytes']) for e in self.entries.values())):
            raise OSError('Insufficient publication-filesystem capacity for numeric spool')
        with zipfile.ZipFile(path, 'x', compression=zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
            for key, entry in self.entries.items():
                archive.write(entry['path'], arcname=key + '.npy')
        self.published = True

    def metadata(self):
        return {
            'directory': str(self.directory),
            'retained_npy_bytes': self.total,
            'allocated_disk_bytes': self.budget,
            'publication_disk_bound_bytes': sum(self._zip_bound(entry['file_bytes']) for entry in self.entries.values()),
            'arrays': len(self.entries),
            'all_arrays_complete': all(entry['complete'] for entry in self.entries.values()),
            'strategy': 'Disk-backed NPY arrays and sequential ZIP publication; no all-observable materialization',
        }

class LazyNpzArrays(Mapping):
    """Read individual source coefficients from a previously verified product."""

    def __init__(self, product, report, *, memory_gib):
        self.path = Path(product).with_name(Path(product).name + '.data') / 'arrays.npz'
        self.records = report['numeric_data']['array_contract']['datasets']
        self.memory_bytes = int(memory_gib * 1024 ** 3)

    def __iter__(self):
        return iter(self.records)

    def __len__(self):
        return len(self.records)

    def __contains__(self, key):
        return key in self.records

    def nbytes(self, key):
        import json
        row = self.records[key]
        return math.prod(json.loads(row['shape_json'])) * np.dtype(row['dtype']).itemsize

    def __getitem__(self, key):
        if self.nbytes(key) * 3 > self.memory_bytes:
            raise MemoryError('One source coefficient array exceeds declared intake workspace')
        with np.load(self.path, allow_pickle=False) as source:
            return source[key]

    def maximum_channel_bytes(self):
        totals = {}
        for key in self.records:
            if key.startswith('channel'):
                channel = key.split('_', 1)[0]
                totals[channel] = totals.get(channel, 0) + self.nbytes(key)
        return max(totals.values(), default=0)

    def copy_member(self, key, spool):
        with zipfile.ZipFile(self.path) as archive:
            spool.copy_npy_member(key, archive)
