"""Two-rank NCCL reductions on the CUDA stream owned by Dr.Jit.

The caller starts one process per allocated device, selects its logical device
before constructing arrays, and supervises both processes. No GPU identifier or
Slurm environment variable is changed here. This module never stages gradients
through NumPy, and cannot be used inside ``dr.freeze``.
"""

from __future__ import annotations

from contextlib import contextmanager
import ctypes
import ctypes.util
import os
from pathlib import Path
import sys
import sysconfig
import time


GRADIENT_KEYS = ("radiance_sh", "opacity_sh", "centers", "log_scales", "rotation_tangent")
SUM_COUNTER_KEYS = ("appearance_valid", "geometry_valid", "background_carve_valid")
MAX_COUNTER_KEYS = ("foreground_evidence", "background_evidence")

_NCCL_FLOAT32, _NCCL_UINT32 = 7, 3
_NCCL_SUM, _NCCL_MAX, _NCCL_MIN = 0, 2, 3


class _UniqueId(ctypes.Structure):
    _fields_ = [("internal", ctypes.c_char * 128)]


def _load_nccl(library=None):
    if sys.platform != "linux":
        raise RuntimeError("Two-GPU reconstruction currently requires Linux and NCCL; use one GPU on Windows")
    explicit = library or os.environ.get("MITSUBA_NCCL_LIBRARY")
    if explicit:
        candidates = [os.fspath(explicit)]
    else:
        candidates = []
        found = ctypes.util.find_library("nccl")
        if found:
            candidates.append(found)
        # NVIDIA's small NCCL-only wheel is independent of PyTorch. Prefer an
        # already installed copy and do not install anything automatically.
        for key in ("purelib", "platlib"):
            base = sysconfig.get_path(key)
            if base:
                candidates.append(str(Path(base) / "nvidia" / "nccl" / "lib" / "libnccl.so.2"))
        candidates.extend((str(Path(sys.prefix) / "lib" / "libnccl.so.2"), "libnccl.so.2"))
    errors = []
    for candidate in dict.fromkeys(candidates):
        try:
            return ctypes.CDLL(candidate), candidate
        except OSError as exc:
            errors.append(f"{candidate}: {exc}")
    raise RuntimeError(
        "NCCL could not be loaded. Load your HPC NCCL module, or set "
        "MITSUBA_NCCL_LIBRARY to its libnccl.so.2. Alternatively install a "
        "driver-compatible NVIDIA NCCL package inside this project's environment. "
        "PyTorch and a CUDA Toolkit are not required. Tried: " + "; ".join(errors))


def _bind_nccl(lib):
    signatures = {
        "ncclGetVersion": [ctypes.POINTER(ctypes.c_int)],
        "ncclGetUniqueId": [ctypes.POINTER(_UniqueId)],
        "ncclCommInitRank": [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int, _UniqueId, ctypes.c_int],
        "ncclAllReduce": [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
                          ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p],
        "ncclGroupStart": [],
        "ncclGroupEnd": [],
        "ncclCommDestroy": [ctypes.c_void_p],
        "ncclCommAbort": [ctypes.c_void_p],
    }
    for name, args in signatures.items():
        try:
            function = getattr(lib, name)
        except AttributeError as exc:
            raise RuntimeError(f"Loaded NCCL library is missing required function {name}") from exc
        function.argtypes, function.restype = args, ctypes.c_int
    lib.ncclGetErrorString.argtypes = [ctypes.c_int]
    lib.ncclGetErrorString.restype = ctypes.c_char_p


class NcclCollective:
    """Synchronous view boundary, asynchronous operations on one CUDA stream.

    Each ``reduce_view`` call returns fresh device buffers after one final stream
    synchronization. SUM applies to gradient/validity counts; MAX unions deletion
    evidence. Optional image arrays must be full-size arrays with each rank's
    disjoint pixel shard filled and every other pixel exactly zero.
    """

    def __init__(self, rank, world_size, connection, mi, dr, np, library=None):
        if world_size != 2 or rank not in (0, 1):
            raise ValueError("NCCL reconstruction currently supports exactly two local ranks")
        self.rank, self.world_size, self.connection = int(rank), int(world_size), connection
        self.mi, self.dr, self.np = mi, dr, np
        self._comm = ctypes.c_void_p()
        self._closed = False
        self._failed = False
        self._sequence = 0
        self._stats = {"views": 0, "calls": 0, "payload_bytes": 0, "seconds": 0.}
        required = ("_reconstruction_cuda_context", "_reconstruction_cuda_push_context",
                    "_reconstruction_cuda_pop_context", "_reconstruction_float_buffer_info",
                    "_reconstruction_uint_buffer_info")
        if any(not hasattr(mi, name) for name in required):
            raise RuntimeError("NCCL source bindings are unavailable; rebuild this Mitsuba checkout")
        self.device = dict(mi._reconstruction_cuda_context())
        if self.device["device_count"] < world_size or self.device["device"] != rank:
            raise RuntimeError("Each NCCL worker must first select its own allocated logical CUDA device")
        self._lib, self.library = _load_nccl(library)
        _bind_nccl(self._lib)
        version = ctypes.c_int()
        self._check(self._lib.ncclGetVersion(ctypes.byref(version)), "ncclGetVersion")
        self.version = version.value
        uid = _UniqueId()
        if rank == 0:
            self._check(self._lib.ncclGetUniqueId(ctypes.byref(uid)), "ncclGetUniqueId")
            self.connection.send(("nccl_unique_id", ctypes.string_at(ctypes.byref(uid), ctypes.sizeof(uid))))
        else:
            tag, encoded = self.connection.recv()
            if tag != "nccl_unique_id" or not isinstance(encoded, bytes) or len(encoded) != ctypes.sizeof(uid):
                raise RuntimeError("Invalid NCCL initialization message from rank zero")
            ctypes.memmove(ctypes.byref(uid), encoded, len(encoded))
        try:
            with self._context():
                self._check(self._lib.ncclCommInitRank(ctypes.byref(self._comm), world_size, uid, rank),
                            "ncclCommInitRank")
        except BaseException:
            self._failed = True
            if self._comm:
                self.close(abort=True)
            raise

    @contextmanager
    def _context(self):
        self.mi._reconstruction_cuda_push_context(int(self.device["context"]))
        try:
            yield
        finally:
            self.mi._reconstruction_cuda_pop_context()

    def _check(self, result, operation):
        if result != 0:
            self._failed = True
            message = self._lib.ncclGetErrorString(result)
            detail = message.decode("utf-8", "replace") if message else "unknown NCCL error"
            raise RuntimeError(f"{operation} failed on rank {self.rank}: {detail} (code {result})")

    def broadcast(self, value=None):
        """Broadcast a small CPU control object; never use for gradient arrays."""
        sequence = self._sequence
        self._sequence += 1
        if self.rank == 0:
            self.connection.send(("nccl_control_broadcast", sequence, value))
            return value
        tag, received_sequence, received = self.connection.recv()
        if tag != "nccl_control_broadcast" or sequence != received_sequence:
            raise RuntimeError("NCCL control broadcast order differs between ranks")
        return received

    def barrier(self):
        sequence = self._sequence
        self._sequence += 1
        token = ("nccl_control_barrier", sequence)
        if self.rank == 0:
            self.connection.send(token)
            received = self.connection.recv()
        else:
            received = self.connection.recv()
            self.connection.send(token)
        if received != token:
            raise RuntimeError("NCCL control barrier order differs between ranks")

    def _pack(self, arrays, array_type):
        values = [array_type(value) for value in arrays]
        widths = [int(self.dr.width(value)) for value in values]
        # Dr.Jit concat builds indices even for empty inputs, triggering a
        # zero-width + scalar operation. Preserve group widths for unpacking,
        # but concatenate only populated groups (an empty topology is valid).
        populated = [value for value, width in zip(values, widths) if width]
        packed = self.dr.concat(populated) if populated else self.dr.empty(array_type, 0)
        return packed, widths

    def _unpack(self, packed, names, widths, array_type):
        result, begin = {}, 0
        for name, width in zip(names, widths):
            if width == 0:
                result[name] = self.dr.empty(array_type, 0)
            else:
                index = self.dr.arange(self.mi.UInt32, width) + self.dr.opaque(self.mi.UInt32, begin)
                result[name] = self.dr.gather(array_type, packed, index)
            begin += width
        return result

    def reduce_view(self, accumulator, loss_sum, finite, raw_alpha=None, alpha=None):
        """Collectively SUM a full view's disjoint ray contributions on the GPU."""
        if self._closed or self._failed:
            raise RuntimeError("NCCL collective is closed or has failed")
        started = time.perf_counter()
        dr, mi = self.dr, self.mi
        if set(accumulator["gradients"]) != set(GRADIENT_KEYS):
            raise ValueError("Expected exactly the five reconstruction gradient groups")
        gradients, gradient_widths = self._pack(
            [accumulator["gradients"][name] for name in GRADIENT_KEYS], mi.Float)
        validity, validity_widths = self._pack([accumulator[name] for name in SUM_COUNTER_KEYS], mi.UInt32)
        evidence, evidence_widths = self._pack([accumulator[name] for name in MAX_COUNTER_KEYS], mi.UInt32)
        if len(set(validity_widths + evidence_widths)) != 1:
            raise ValueError("All validity and evidence arrays must have one value per voxel")
        loss_sum, finite = mi.Float(loss_sum), mi.UInt32(finite)
        if dr.width(loss_sum) != 1 or dr.width(finite) != 1:
            raise ValueError("Loss and finite-status reductions require scalar device arrays")
        operations = []

        def prepare(value, dtype, operation):
            array_type = mi.Float if dtype == _NCCL_FLOAT32 else mi.UInt32
            # A fresh undefined allocation is essential: a literal zero array
            # would still be folded to zero by JIT after an external write.
            destination = dr.empty(array_type, dr.width(value))
            operations.append((value, destination, dtype, operation))
            return destination

        global_gradients = prepare(gradients, _NCCL_FLOAT32, _NCCL_SUM)
        global_validity = prepare(validity, _NCCL_UINT32, _NCCL_SUM)
        global_evidence = prepare(evidence, _NCCL_UINT32, _NCCL_MAX)
        global_loss = prepare(loss_sum, _NCCL_FLOAT32, _NCCL_SUM)
        global_finite = prepare(finite, _NCCL_UINT32, _NCCL_MIN)
        global_raw = prepare(mi.Float(raw_alpha), _NCCL_FLOAT32, _NCCL_SUM) if raw_alpha is not None else None
        global_alpha = prepare(mi.Float(alpha), _NCCL_FLOAT32, _NCCL_SUM) if alpha is not None else None
        # Keep sources, destinations, and packed owners alive through the final
        # stream wait. Both JIT work and NCCL enqueue into the same stream.
        dr.eval(*[array for source, destination, _, _ in operations for array in (source, destination)])
        descriptors = []
        for source, destination, dtype, operation in operations:
            width = int(dr.width(source))
            if width == 0:
                continue
            info = (mi._reconstruction_float_buffer_info if dtype == _NCCL_FLOAT32
                    else mi._reconstruction_uint_buffer_info)
            source_info, destination_info = info(source), info(destination)
            if int(source_info["size"]) != width or int(destination_info["size"]) != width:
                raise RuntimeError("NCCL buffer pointer metadata does not match the allocated array")
            descriptors.append((int(source_info["pointer"]), int(destination_info["pointer"]),
                                width, dtype, operation))
        try:
            with self._context():
                self._check(self._lib.ncclGroupStart(), "ncclGroupStart")
                try:
                    for source, destination, width, dtype, operation in descriptors:
                        self._check(self._lib.ncclAllReduce(source, destination, width, dtype, operation,
                                                          self._comm, self.device["stream"]), "ncclAllReduce")
                finally:
                    self._check(self._lib.ncclGroupEnd(), "ncclGroupEnd")
            # This is the only explicit stream synchronization of a reduction.
            # It also makes destination ownership independent of NCCL lifetimes.
            dr.sync_thread()
            result = {"gradients": self._unpack(global_gradients, GRADIENT_KEYS, gradient_widths, mi.Float)}
            result.update(self._unpack(global_validity, SUM_COUNTER_KEYS, validity_widths, mi.UInt32))
            result.update(self._unpack(global_evidence, MAX_COUNTER_KEYS, evidence_widths, mi.UInt32))
            self._stats["views"] += 1
            self._stats["calls"] += len(descriptors)
            self._stats["payload_bytes"] += sum(width * 4 for _, _, width, _, _ in descriptors)
            self._stats["seconds"] += time.perf_counter() - started
            return result, global_loss, mi.Bool(global_finite != 0), global_raw, global_alpha
        except BaseException:
            self._failed = True
            raise

    def statistics(self):
        return {"backend": "nccl", "version": self.version, "library": self.library,
                "rank": self.rank, "world_size": self.world_size,
                "device": dict(self.device), "payload_scope": "per-rank reduced input bytes, not wire traffic",
                **self._stats}

    def close(self, abort=False):
        if self._closed:
            return
        self._closed = True
        if self._comm:
            with self._context():
                function = self._lib.ncclCommAbort if abort or self._failed else self._lib.ncclCommDestroy
                self._check(function(self._comm), function.__name__)
            self._comm = ctypes.c_void_p()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close(abort=exc_type is not None)

