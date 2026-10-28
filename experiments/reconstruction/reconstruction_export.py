"""Bounded array readers for streaming portable reconstruction checkpoints.

Sources retain their original arrays. Callers must not mutate them until the
save completes, and must consume each chunk before requesting the next one.
Importing this module never imports Dr.Jit or initializes a GPU runtime.
"""

from __future__ import annotations

import math
import operator

import numpy as np


CHUNK_BYTES = 4 * 1024**2


def _index(value, name):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a nonnegative integer")
    try:
        result = operator.index(value)
    except TypeError as error:
        raise ValueError(f"{name} must be a nonnegative integer") from error
    if result < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return result


def _shape(value):
    try:
        return tuple(_index(item, "shape dimension") for item in value)
    except TypeError as error:
        raise ValueError("shape must be a sequence of nonnegative integers") from error


def _dtype(value):
    result = np.dtype(value)
    if result.kind not in "biuf" or result.hasobject:
        raise ValueError("Checkpoint arrays require a boolean, integer, or floating dtype")
    return result


def _safe_dtype(source, requested):
    target = source if requested is None else _dtype(requested)
    safe = np.can_cast(source, target, casting="safe")
    # NumPy considers int64 -> float64 safe, although integers above 2**53
    # lose bits. Check mantissa capacity explicitly for checkpoint conversions.
    if source.kind in "iu" and target.kind == "f":
        integer_bits = np.iinfo(source).bits - (source.kind == "i")
        safe &= np.finfo(target).nmant + 1 >= integer_bits
    if not safe:
        raise ValueError(f"Unsafe checkpoint dtype conversion: {source} -> {target}")
    return target


class ArraySource:
    """An explicitly chunked array; no implicit full-array NumPy conversion.

    ``read_chunk(start, count)`` addresses flattened C-order elements. Its
    callback must return exactly that many elements with the declared dtype.
    ``max_chunk_elements`` includes room for UInt32 GPU gather indices, even
    for byte-valued sources, and may be used by the streaming writer.
    """

    def __init__(self, shape, dtype, read_chunk, *, max_chunk_elements=None):
        self.shape = _shape(shape)
        self.dtype = _dtype(dtype)
        self._size = math.prod(self.shape)
        if not callable(read_chunk):
            raise TypeError("read_chunk must be callable")
        self._reader = read_chunk
        limit = CHUNK_BYTES // max(self.dtype.itemsize, 4)
        if max_chunk_elements is not None:
            supplied = _index(max_chunk_elements, "max_chunk_elements")
            if supplied == 0:
                raise ValueError("max_chunk_elements must be positive")
            limit = min(limit, supplied)
        self.max_chunk_elements = limit

    @property
    def size(self):
        return self._size

    def read_chunk(self, start, count):
        start = _index(start, "start")
        count = _index(count, "count")
        if start > self.size or count > self.size - start:
            raise ValueError("Checkpoint chunk lies outside the declared array")
        if count > self.max_chunk_elements:
            raise ValueError("Checkpoint chunk exceeds max_chunk_elements")
        if count == 0:
            return np.empty(0, dtype=self.dtype)
        result = self._reader(start, count)
        if (not isinstance(result, np.ndarray) or result.ndim != 1
                or result.shape != (count,) or result.dtype != self.dtype
                or not result.flags.c_contiguous):
            raise ValueError("read_chunk must return an exact 1D C-contiguous NumPy chunk with the declared dtype")
        return result


def _device_dtype(value, dr):
    names = {"Bool": "?", "Int8": "i1", "UInt8": "u1", "Int16": "i2", "UInt16": "u2",
             "Int32": "i4", "UInt32": "u4", "Int64": "i8", "UInt64": "u8",
             "Float16": "f2", "Float32": "f4", "Float64": "f8"}
    kind = dr.type_v(type(value))
    for name, dtype in names.items():
        if kind == getattr(dr.VarType, name, None):
            return np.dtype(dtype)
    raise ValueError("Unsupported Dr.Jit checkpoint element type")


def device_array_source(value, *, dtype=None, shape=None, dr=None):
    """Read flat device arrays through small, dtype-preserving GPU gathers.

    C-contiguous NumPy arrays are also accepted for CPU callers and tests.
    Optional dtype conversions happen only on the host chunk and must be
    lossless schema conversions, such as int32 -> int64 or uint8 -> uint32.
    """
    if isinstance(value, np.ndarray):
        if not value.flags.c_contiguous:
            raise ValueError("NumPy checkpoint sources must be C-contiguous")
        source_dtype = _dtype(value.dtype)
        target_dtype = _safe_dtype(source_dtype, dtype)
        target_shape = value.shape if shape is None else _shape(shape)
        if math.prod(target_shape) != value.size:
            raise ValueError("Checkpoint source shape does not match its element count")
        flat = value.reshape(-1)

        def read(start, count):
            return flat[start:start+count].astype(target_dtype, copy=False)

        return ArraySource(target_shape, target_dtype, read,
            max_chunk_elements=CHUNK_BYTES // max(source_dtype.itemsize, target_dtype.itemsize, 4))

    if dr is None:
        import drjit as dr
    if (not dr.is_array_v(type(value)) or dr.is_tensor_v(type(value))
            or not dr.is_dynamic_v(type(value)) or dr.depth_v(type(value)) != 1):
        raise ValueError("Device checkpoint sources must be flat dynamic Dr.Jit arrays")
    # Detach without evaluating or copying the full source. Readback must not
    # create an AD tape even if a caller supplies an AD-enabled array.
    source = dr.detach(value, preserve_type=False)
    source_dtype = _device_dtype(source, dr)
    target_dtype = _safe_dtype(source_dtype, dtype)
    size = len(source)
    if size > 2**32 - 1:
        raise ValueError("Device checkpoint source exceeds UInt32 gather indexing")
    target_shape = (size,) if shape is None else _shape(shape)
    if math.prod(target_shape) != size:
        raise ValueError("Checkpoint source shape does not match its element count")
    index_type = dr.uint32_array_t(type(source))

    def read(start, count):
        # An opaque offset reuses the same gather kernel across full chunks.
        # Do not add zero to float data to force a copy: that can change -0.
        index = dr.arange(index_type, count) + dr.opaque(index_type, start)
        block = dr.gather(type(source), source, index)
        dr.eval(block)
        # Only this small gathered allocation migrates into pinned host memory.
        # NumPy's view owns that host buffer until the writer consumes it.
        host = np.asarray(block)
        del block, index
        return host.astype(target_dtype, copy=False)

    return ArraySource(target_shape, target_dtype, read,
        max_chunk_elements=CHUNK_BYTES // max(source_dtype.itemsize, target_dtype.itemsize, 4))


def zero_array_source(shape, dtype):
    """Represent compatibility-only zero fields without allocating full tables."""
    dtype = _dtype(dtype)
    return ArraySource(shape, dtype, lambda start, count: np.zeros(count, dtype=dtype))


def checkpoint_array_sources(value, *, dr=None):
    """Wrap device leaves in nested checkpoint containers without copying arrays."""
    if isinstance(value, (ArraySource, np.ndarray)):
        return value
    if isinstance(value, dict):
        return {key: checkpoint_array_sources(item, dr=dr) for key, item in value.items()}
    if isinstance(value, list):
        return [checkpoint_array_sources(item, dr=dr) for item in value]
    if isinstance(value, tuple):
        return tuple(checkpoint_array_sources(item, dr=dr) for item in value)
    if value is None or isinstance(value, (str, bool, int, float, np.generic)):
        return value
    if dr is None:
        import drjit as dr
    return device_array_source(value, dr=dr) if dr.is_array_v(type(value)) else value
