"""Exact topology remaps without keeping two full GPU fields resident.

Only a topology boundary may use these destructive transfers. The caller must
retire all other source owners first; failed transfers must resume a complete
checkpoint, never serialize a partially migrated model/optimizer.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
import shutil
import tempfile

from reconstruction_memory import TrainingMemoryError, query_cuda_memory, round_cuda_allocation

MIB = 1024 ** 2
GIB = 1024 ** 3


def available_host_memory():
    """Read available physical RAM without importing a GPU or optional package."""
    if os.name == "nt":
        class Memory(ctypes.Structure):
            _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
                (key, ctypes.c_ulonglong) for key in ("total", "available", "page_total",
                    "page_available", "virtual_total", "virtual_available", "reserved")]
        result = Memory()
        result.length = ctypes.sizeof(result)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(result)):
            return int(result.available)
        return None
    try:
        # MemAvailable accounts for reclaimable cache on Linux.
        with open("/proc/meminfo", encoding="ascii") as stream:
            for line in stream:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
        return int(os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
    except (OSError, ValueError, AttributeError):
        return None


class _HostField:
    """One ordinary RAM array, or a sequential disk file when RAM is scarce."""

    def __init__(self, length, dtype, scratch_directory=None):
        import numpy as np
        self.length, self.dtype = length, np.dtype(dtype)
        self.array = self.file = None
        required = length * self.dtype.itemsize
        available = available_host_memory()
        if available is not None and available >= required + 2 * GIB:
            try:
                self.array = np.empty(length, self.dtype)
            except MemoryError:
                pass
        self.storage = "ram" if self.array is not None else "disk"
        if self.array is None:
            directory = Path(scratch_directory or tempfile.gettempdir())
            directory.mkdir(parents=True, exist_ok=True)
            free = shutil.disk_usage(directory).free
            if free < required + GIB:
                raise TrainingMemoryError({"stage": "topology_host_staging", "supported": False,
                    "reason_code": "host_staging_exceeds_budget", "required_bytes": required,
                    "available_disk_bytes": free,
                    "reason": "Insufficient RAM and temporary disk space to stage one topology field."})
            self.file = tempfile.TemporaryFile(prefix="topology-field-", dir=directory)

    def write(self, offset, values):
        if self.array is not None:
            self.array[offset:offset + len(values)] = values
        else:
            self.file.write(memoryview(values).cast("B"))

    def read(self, offset, size):
        import numpy as np
        if self.array is not None:
            return self.array[offset:offset + size]
        self.file.seek(offset * self.dtype.itemsize)
        data = self.file.read(size * self.dtype.itemsize)
        if len(data) != size * self.dtype.itemsize:
            raise IOError("Incomplete topology staging field")
        return np.frombuffer(data, self.dtype)

    def close(self):
        self.array = None
        if self.file is not None:
            self.file.close()
            self.file = None


class DeferredRemapField:
    """A row gather plus optional append, evaluated after retiring old owners.

    ``append`` has exactly ``(new_count - len(survivors)) * width`` elements;
    omitted appended values are zero (Adam's newborn state). No conversion or
    arithmetic is applied to the payload, including UInt32 optimizer clocks.
    """

    def __init__(self, old_value, survivors, width, new_count, *, append=None, dr,
                 scratch_directory=None, chunk_bytes=4 * MIB, allocation_guard=None,
                 allocator_policy="cuda_2m"):
        if width < 1 or new_count < 0 or chunk_bytes < 4:
            raise ValueError("Invalid topology transfer dimensions")
        self.dr, self.array_type = dr, type(old_value)
        if dr.depth_v(self.array_type) != 1 or dr.type_v(self.array_type) not in (
                dr.VarType.Float32, dr.VarType.UInt32, dr.VarType.Int32):
            raise TypeError("Topology transfer fields must be flat Float32/UInt32/Int32 arrays")
        self._old, self._survivors, self._append = old_value, survivors, append
        self.width, self.new_count = int(width), int(new_count)
        if self.new_count * self.width > 2 ** 32 - 1 or int(dr.width(old_value)) > 2 ** 32 - 1:
            raise ValueError("Topology transfer indices must fit UInt32")
        self.kept = int(dr.width(survivors))
        if self.kept > self.new_count:
            raise ValueError("More surviving rows than destination rows")
        if int(dr.width(old_value)) % self.width:
            raise ValueError("Source field has incomplete rows")
        if append is not None and int(dr.width(append)) != (self.new_count - self.kept) * self.width:
            raise ValueError("Appended field does not match newborn row count")
        self.scratch_directory, self.chunk_bytes = scratch_directory, int(chunk_bytes)
        self.allocation_guard = allocation_guard
        self.allocator_policy = allocator_policy
        self.statistics = {"bytes": self.new_count * self.width * 4, "chunk_bytes": self.chunk_bytes}
        self._consumed = False

    def materialize(self, allocation_guard=None):
        import numpy as np
        if self._consumed:
            raise RuntimeError("A topology field transfer can only be materialized once")
        dr, array_type = self.dr, self.array_type
        dtype = np.float32 if dr.is_float_v(array_type) else (
            np.uint32 if dr.is_unsigned_v(array_type) else np.int32)
        length = self.new_count * self.width
        if length == 0:
            self._old = self._append = self._survivors = None
            self._consumed = True
            return dr.empty(array_type, 0)
        block = max(1, self.chunk_bytes // 4)
        host = _HostField(length, dtype, self.scratch_directory)
        self.statistics["storage"] = host.storage
        try:
            # Complete appended expressions while their source remains alive.
            # A one-field transfer never copies the entire source via NumPy:
            # that path would create a huge pinned host allocation.
            dr.eval(self._old, self._survivors)
            if self._append is not None:
                dr.eval(self._append)
            UInt = dr.uint32_array_t(array_type)
            for offset in range(0, length, block):
                count = min(block, length - offset)
                lane = dr.arange(UInt, count) + dr.opaque(UInt, offset)
                retained = lane < self.kept * self.width
                parent = dr.gather(UInt, self._survivors, lane // self.width, retained)
                value = dr.gather(array_type, self._old,
                                  parent * self.width + lane % self.width, retained)
                if self._append is not None:
                    added = dr.gather(array_type, self._append,
                                      lane - self.kept * self.width, ~retained)
                    value = dr.select(retained, value, added)
                    del added
                dr.eval(value)
                values = np.asarray(value)
                host.write(offset, values)
                # NumPy owns a temporary device/host mapping. Drop it before
                # freeing the source and before allocating the replacement.
                del values, value, parent, retained, lane
            dr.sync_thread()
            self._old = self._append = self._survivors = None
            self._consumed = True
            dr.sync_thread()
            dr.flush_malloc_cache()
            # Cache flushing can enqueue cuMemFreeAsync after its own initial
            # synchronization. Wait for those frees before querying physical
            # residency or rejecting the replacement on a stale free count.
            dr.sync_thread()
            required = round_cuda_allocation(length * 4, self.allocator_policy) + max(64 * MIB, 8 * self.chunk_bytes)
            guard = allocation_guard or self.allocation_guard
            if guard is not None:
                try:
                    guard(required)
                except TrainingMemoryError as error:
                    error.plan.update(staged_field_bytes=length * 4, staged_field_width=self.width,
                                      staged_field_rows=self.new_count, staging_storage=host.storage)
                    raise
            else:
                memory = query_cuda_memory()
                if memory is None or memory["free_bytes"] < required:
                    raise TrainingMemoryError({"stage": "topology_staged_allocation", "supported": False,
                        "reason_code": "staged_field_exceeds_free_memory", "required_bytes": required,
                        "live_memory": memory,
                        "reason": "Retired topology field did not free enough GPU memory for its replacement; "
                                  "another owner may still retain the original array."})
            result = dr.empty(array_type, length)
            # Scatter into this sole owner; no old field or second result owner
            # exists here that could trigger a full-field copy-on-write.
            for offset in range(0, length, block):
                count = min(block, length - offset)
                upload = array_type(host.read(offset, count))
                lane = dr.arange(UInt, count) + dr.opaque(UInt, offset)
                dr.scatter(result, upload, lane)
                dr.eval(result)
                dr.sync_thread()
                del upload, lane
            dr.sync_thread()
            dr.flush_malloc_cache()
            dr.sync_thread()
            return result
        finally:
            host.close()
