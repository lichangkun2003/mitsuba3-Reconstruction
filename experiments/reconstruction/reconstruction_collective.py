"""N-rank NCCL reductions on the CUDA stream owned by Dr.Jit.

The caller starts one process per allocated device, selects its logical device
before constructing arrays, and supervises every process. No GPU identifier or
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
LOSS_COMPONENT_KEYS = ("rgb_loss_sum", "distortion_loss_sum")

_NCCL_FLOAT32, _NCCL_UINT32 = 7, 3
_NCCL_SUM, _NCCL_MAX, _NCCL_MIN = 0, 2, 3


class _UniqueId(ctypes.Structure):
    _fields_ = [("internal", ctypes.c_char * 128)]


def _load_nccl(library=None):
    if sys.platform != "linux":
        raise RuntimeError("Multi-GPU reconstruction currently requires Linux and NCCL; use one GPU on Windows")
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
        "ncclBroadcast": [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
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
    """GPU reductions on one CUDA stream, with explicit external ownership.

    Each ``reduce_view`` call returns fresh device buffers after one final stream
    synchronization. SUM applies to gradients, counts and raw RGB topology
    evidence; MAX unions legacy alpha deletion flags. Optional images must have
    full-size arrays with each rank's
    disjoint pixel shard filled and every other pixel exactly zero.
    The RGB batch path keeps the final NCCL writes asynchronous using completion
    events; sparse union compression still reads its scalar output length.
    """

    def __init__(self, rank, world_size, connection, mi, dr, np, library=None):
        if (type(world_size) is not int or world_size < 1
                or type(rank) is not int or not 0 <= rank < world_size):
            raise ValueError("NCCL rank must fit within a positive world size")
        self.rank, self.world_size, self.connection = int(rank), int(world_size), connection
        self.mi, self.dr, self.np = mi, dr, np
        self._comm = ctypes.c_void_p()
        self._closed = False
        self._failed = False
        self._sequence = 0
        self._owners = None
        self._launch_owners = None
        self._stats = {"views": 0, "calls": 0, "payload_bytes": 0, "seconds": 0.}
        from train_multigpu import WorkerContext
        self._control = (connection if isinstance(connection, WorkerContext)
                         else WorkerContext(rank, world_size, connection, 1))
        required = ("_reconstruction_cuda_context", "_reconstruction_cuda_push_context",
                    "_reconstruction_cuda_pop_context", "_reconstruction_float_buffer_info",
                    "_reconstruction_uint_buffer_info")
        if any(not hasattr(mi, name) for name in required):
            raise RuntimeError("NCCL source bindings are unavailable; rebuild this Mitsuba checkout")
        self.device = dict(mi._reconstruction_cuda_context())
        if self.device["device_count"] < world_size or self.device["device"] != rank:
            raise RuntimeError("Each NCCL worker must first select its own allocated logical CUDA device")
        if world_size == 1:
            self._lib, self.library, self.version = None, None, None
            return
        self._lib, self.library = _load_nccl(library)
        _bind_nccl(self._lib)
        version = ctypes.c_int()
        self._check(self._lib.ncclGetVersion(ctypes.byref(version)), "ncclGetVersion")
        self.version = version.value
        uid = _UniqueId()
        if rank == 0:
            self._check(self._lib.ncclGetUniqueId(ctypes.byref(uid)), "ncclGetUniqueId")
        encoded = self._control.broadcast(ctypes.string_at(ctypes.byref(uid), ctypes.sizeof(uid))
                                          if rank == 0 else None)
        if not isinstance(encoded, bytes) or len(encoded) != ctypes.sizeof(uid):
            raise RuntimeError("Invalid NCCL initialization message from rank zero")
        ctypes.memmove(ctypes.byref(uid), encoded, len(encoded))
        try:
            with self._context():
                self._check(self._lib.ncclCommInitRank(ctypes.byref(self._comm), world_size, uid, rank),
                            "ncclCommInitRank")
            from reconstruction_cuda_owners import CudaOwnerQueue
            self._owners = CudaOwnerQueue(dr)
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
        return self._control.broadcast(value)

    def barrier(self):
        self._control.barrier()

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
        """SUM gradient/count contributions on GPU; caller supplies normalization.

        Kept for legacy pixel-sharded callers. Whole-view batches should use
        ``reduce_batch`` and keep each camera's RGB topology vote separate.
        """
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
        component_presence = [name in accumulator for name in LOSS_COMPONENT_KEYS]
        if any(component_presence) and not all(component_presence):
            raise ValueError("Distortion reductions require both RGB and distortion loss sums")
        component_names = LOSS_COMPONENT_KEYS if all(component_presence) else ()
        if any(dr.width(accumulator[name]) != 1 for name in component_names):
            raise ValueError("Loss component reductions require scalar device arrays")
        # Pack these scalars into the existing loss collective, so reporting
        # components adds no NCCL call or host synchronization per view.
        packed_losses = (dr.concat([loss_sum, *[mi.Float(accumulator[name]) for name in component_names]])
                         if component_names else loss_sum)
        operations = []

        def prepare(value, dtype, operation):
            array_type = mi.Float if dtype == _NCCL_FLOAT32 else mi.UInt32
            if self.world_size == 1:
                return array_type(value)
            # A fresh undefined allocation is essential: a literal zero array
            # would still be folded to zero by JIT after an external write.
            destination = dr.empty(array_type, dr.width(value))
            operations.append((value, destination, dtype, operation))
            return destination

        global_gradients = prepare(gradients, _NCCL_FLOAT32, _NCCL_SUM)
        global_validity = prepare(validity, _NCCL_UINT32, _NCCL_SUM)
        global_evidence = prepare(evidence, _NCCL_UINT32, _NCCL_MAX)
        global_losses = prepare(packed_losses, _NCCL_FLOAT32, _NCCL_SUM)
        global_finite = prepare(finite, _NCCL_UINT32, _NCCL_MIN)
        global_raw = prepare(mi.Float(raw_alpha), _NCCL_FLOAT32, _NCCL_SUM) if raw_alpha is not None else None
        global_alpha = prepare(mi.Float(alpha), _NCCL_FLOAT32, _NCCL_SUM) if alpha is not None else None
        # Candidate IDs are broadcast once per growth window. Combine every
        # pixel shard before the primary rank classifies a full-view RGB vote.
        global_rgb_evidence = {}
        for kind in ("growth", "deletion"):
            keys = tuple(f"{kind}_{suffix}" for suffix in ("gain", "observed", "unknown"))
            present = [key in accumulator for key in keys]
            if any(present) and not all(present):
                raise ValueError(f"{kind.title()} reductions require gain, observed and unknown arrays together")
            if all(present):
                widths = {int(dr.width(accumulator[key])) for key in keys}
                if len(widths) != 1:
                    raise ValueError(f"{kind.title()} evidence arrays must share IDs and lengths")
                if kind == "deletion" and widths != {validity_widths[0]}:
                    raise ValueError("Deletion evidence must have one value per active voxel")
                for key in keys:
                    dtype = _NCCL_FLOAT32 if key.endswith("_gain") else _NCCL_UINT32
                    global_rgb_evidence[key] = prepare(accumulator[key], dtype, _NCCL_SUM)
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
            if self.world_size > 1:
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
            result.update(global_rgb_evidence)
            if component_names:
                losses = self._unpack(global_losses, ("loss_sum", *component_names),
                                      [1] * (1 + len(component_names)), mi.Float)
                global_loss = losses.pop("loss_sum")
                result.update(losses)
            else:
                global_loss = global_losses
            self._stats["views"] += 1
            self._stats["calls"] += len(descriptors)
            self._stats["payload_bytes"] += sum(width * 4 for _, _, width, _, _ in descriptors)
            self._stats["seconds"] += time.perf_counter() - started
            return result, global_loss, mi.Bool(global_finite != 0), global_raw, global_alpha
        except BaseException:
            self._failed = True
            raise

    def _allreduce_fields(self, fields, *, wait):
        """Reduce independent contiguous buffers, without concat/unpack copies.

        ``wait=False`` is enabled only with the CUDA event owner queue. Sources
        must outlive the external NCCL read, including temporary compact gathers.
        Every operation uses the Dr.Jit stream, so later consumers are ordered
        after NCCL even when the host continues submitting the next update.
        """
        dr, mi = self.dr, self.mi
        result, operations = {}, []
        for name, (source, dtype, operation) in fields.items():
            array_type = mi.Float if dtype == _NCCL_FLOAT32 else mi.UInt32
            source = array_type(source)
            if self.world_size == 1 or not dr.width(source):
                result[name] = source
                continue
            destination = dr.empty(array_type, dr.width(source))
            operations.append((source, destination, dtype, operation))
            result[name] = destination
        if not operations:
            return result
        queue = getattr(self, "_owners", None)
        if queue is not None:
            queue.reap()
        dr.eval(*[array for source, destination, _, _ in operations
                  for array in (source, destination)])
        descriptors = []
        for source, destination, dtype, operation in operations:
            width = int(dr.width(source))
            info = (mi._reconstruction_float_buffer_info if dtype == _NCCL_FLOAT32
                    else mi._reconstruction_uint_buffer_info)
            source_info, destination_info = info(source), info(destination)
            if int(source_info["size"]) != width or int(destination_info["size"]) != width:
                raise RuntimeError("NCCL buffer pointer metadata does not match the allocated array")
            descriptors.append((int(source_info["pointer"]), int(destination_info["pointer"]),
                                width, dtype, operation))
        self._launch_owners = operations
        try:
            with self._context():
                self._check(self._lib.ncclGroupStart(), "ncclGroupStart")
                try:
                    for source, destination, width, dtype, operation in descriptors:
                        self._check(self._lib.ncclAllReduce(source, destination, width, dtype, operation,
                                                          self._comm, self.device["stream"]), "ncclAllReduce")
                finally:
                    self._check(self._lib.ncclGroupEnd(), "ncclGroupEnd")
            nbytes = sum(width * 4 for _, _, width, _, _ in descriptors)
            if wait or queue is None:
                dr.sync_thread()
                if queue is not None:
                    queue.reap()
                self._stats["rgb_explicit_syncs"] = self._stats.get("rgb_explicit_syncs", 0) + 1
            else:
                # Account for both source and destination owners in the bound.
                queue.retain(operations, nbytes * 2)
            self._launch_owners = None
            self._stats["calls"] += len(descriptors)
            self._stats["payload_bytes"] += nbytes
            return result
        except BaseException:
            self._failed = True
            raise

    def _reduce_rgb_batch(self, accumulator, loss_sum, finite, mode):
        """Exact RGB-only reduction, optionally using a common sparse row set.

        Counts and nonzero-gradient support are unioned before compression.
        Thus valid zero gradients keep Adam's moment/clock updates, while even
        unexpected gradients in invalid rows are retained rather than dropped.
        Only the compressed list length reaches the host; IDs and all values
        stay on the GPU. Dense fallback is selected from the common union, so
        every rank performs exactly the same sequence of NCCL operations.
        """
        dr, mi = self.dr, self.mi
        if mode not in ("auto", "dense"):
            raise ValueError("RGB batch reduction mode must be 'auto' or 'dense'")
        if set(accumulator["gradients"]) != set(GRADIENT_KEYS):
            raise ValueError("Expected exactly the five reconstruction gradient groups")
        counts = {name: mi.UInt32(accumulator[name]) for name in SUM_COUNTER_KEYS + MAX_COUNTER_KEYS}
        count = int(dr.width(counts["appearance_valid"]))
        if any(dr.width(value) != count for value in counts.values()):
            raise ValueError("All validity and evidence arrays must have one value per voxel")
        gradients = {name: mi.Float(accumulator["gradients"][name]) for name in GRADIENT_KEYS}
        widths = {name: int(dr.width(value)) for name, value in gradients.items()}
        if any((width % count if count else width) for width in widths.values()):
            raise ValueError("Every gradient group must have an integral number of values per voxel")
        dimensions = {name: width // count if count else 0 for name, width in widths.items()}
        if count and any(not dimension for dimension in dimensions.values()):
            raise ValueError("Nonempty topologies require nonempty gradient groups")
        loss_sum, finite = mi.Float(loss_sum), mi.Bool(finite)
        if dr.width(loss_sum) != 1 or dr.width(finite) != 1:
            raise ValueError("Loss and finite-status reductions require scalar device arrays")
        present = [name in accumulator for name in LOSS_COMPONENT_KEYS]
        if any(present) and not all(present):
            raise ValueError("Distortion reductions require both RGB and distortion loss sums")
        component_names = LOSS_COMPONENT_KEYS if all(present) else ()
        if any(dr.width(accumulator[name]) != 1 for name in component_names):
            raise ValueError("Loss component reductions require scalar device arrays")
        # Native RGB-only tracing guarantees these arrays are zero. Guard that
        # contract on the device; a mistakenly opted-in alpha caller must fail
        # the existing whole-batch finite/allowed guard, not lose its evidence.
        for name in ("background_carve_valid", *MAX_COUNTER_KEYS):
            if count:
                finite &= dr.all(counts[name] == 0)
        packed_losses = (dr.concat([loss_sum, *[mi.Float(accumulator[name]) for name in component_names]])
                         if component_names else loss_sum)
        fields = {name: (counts[name], _NCCL_UINT32, _NCCL_SUM)
                  for name in ("appearance_valid", "geometry_valid")}
        fields.update(losses=(packed_losses, _NCCL_FLOAT32, _NCCL_SUM),
                      finite=(mi.UInt32(finite), _NCCL_UINT32, _NCCL_MIN))
        dense_bytes = (sum(widths.values()) + count * 5 + dr.width(packed_losses) + 1) * 4
        selected, ids, union_count = "dense", None, count
        if mode == "auto" and self.world_size > 1 and count:
            support = (counts["appearance_valid"] > 0) | (counts["geometry_valid"] > 0)
            for name, value in gradients.items():
                support |= dr.block_reduce(dr.ReduceOp.Or, mi.UInt32(value != 0), dimensions[name]) != 0
            fields["support"] = (mi.UInt32(support), _NCCL_UINT32, _NCCL_MAX)
            reduced = self._allreduce_fields(fields, wait=False)
            # compress synchronizes only to obtain its output length. No IDs,
            # gradients, validity tables or images are copied to CPU memory.
            ids = dr.compress(reduced.pop("support") != 0)
            union_count = int(dr.width(ids))
            queue = getattr(self, "_owners", None)
            if queue is not None:
                queue.reap()
            self._stats["rgb_count_readbacks"] = self._stats.get("rgb_count_readbacks", 0) + 1
            if union_count * 4 <= count * 3:
                selected = "compact"
                compact = {}
                for name, value in gradients.items():
                    dimension = dimensions[name]
                    if union_count:
                        lane = dr.arange(mi.UInt32, union_count * dimension)
                        index = dr.gather(mi.UInt32, ids, lane // dimension) * dimension + lane % dimension
                        compact[name] = dr.gather(mi.Float, value, index)
                    else:
                        compact[name] = dr.empty(mi.Float, 0)
                communicated = self._allreduce_fields(
                    {name: (value, _NCCL_FLOAT32, _NCCL_SUM) for name, value in compact.items()}, wait=False)
                global_gradients = {}
                for name, value in communicated.items():
                    output = dr.zeros(mi.Float, widths[name])
                    if union_count:
                        dimension = dimensions[name]
                        lane = dr.arange(mi.UInt32, union_count * dimension)
                        index = dr.gather(mi.UInt32, ids, lane // dimension) * dimension + lane % dimension
                        dr.scatter(output, value, index)
                    global_gradients[name] = output
            else:
                global_gradients = self._allreduce_fields(
                    {name: (value, _NCCL_FLOAT32, _NCCL_SUM) for name, value in gradients.items()}, wait=False)
        else:
            fields.update({name: (value, _NCCL_FLOAT32, _NCCL_SUM) for name, value in gradients.items()})
            reduced = self._allreduce_fields(fields, wait=False)
            global_gradients = {name: reduced.pop(name) for name in GRADIENT_KEYS}
        result = {"gradients": global_gradients,
                  "appearance_valid": reduced["appearance_valid"], "geometry_valid": reduced["geometry_valid"]}
        result.update({name: dr.zeros(mi.UInt32, count)
                       for name in ("background_carve_valid", *MAX_COUNTER_KEYS)})
        losses = self._unpack(reduced["losses"], ("loss_sum", *component_names),
                              [1] * (1 + len(component_names)), mi.Float)
        global_loss = losses.pop("loss_sum")
        result.update(losses)
        self._stats["rgb_batches"] = self._stats.get("rgb_batches", 0) + 1
        key = f"rgb_{selected}_batches"
        self._stats[key] = self._stats.get(key, 0) + 1
        self._stats["rgb_dense_equivalent_bytes"] = self._stats.get("rgb_dense_equivalent_bytes", 0) + dense_bytes
        if mode == "auto" and self.world_size > 1:
            self._stats["rgb_union_rows_total"] = self._stats.get("rgb_union_rows_total", 0) + union_count
            self._stats["rgb_rows_total"] = self._stats.get("rgb_rows_total", 0) + count
        self._stats["rgb_last_mode"] = selected
        self._stats["rgb_last_union_voxels"] = union_count if mode == "auto" and self.world_size > 1 else None
        self._stats["rgb_last_voxels"] = count
        self._stats["rgb_last_union_density"] = union_count / count if count and mode == "auto" and self.world_size > 1 else None
        self._stats["views"] += 1
        return result, global_loss, mi.Bool(reduced["finite"] != 0)

    def reduce_batch(self, accumulator, loss_sum, finite, *, rgb_only=False, mode="auto"):
        """Reduce one full-view-per-rank batch without mixing camera votes.

        Gradients and validity counts are SUMs, not averages. The training
        driver normalizes by the number of actual views (including short final
        batches) when making its one Adam update. RGB topology evidence must be
        consumed per view through ``broadcast_arrays``; mixing its raw gains
        here would change the voting algorithm.
        """
        invalid = [key for key in accumulator
                   if any(key.startswith(f"{kind}_") for kind in ("growth", "deletion"))]
        if invalid:
            raise ValueError(f"Whole-view batches require separate per-camera topology evidence: {invalid}")
        if rgb_only:
            if self._closed or self._failed:
                raise RuntimeError("NCCL collective is closed or has failed")
            started = time.perf_counter()
            before_bytes = self._stats["payload_bytes"]
            result, loss, finite = self._reduce_rgb_batch(accumulator, loss_sum, finite, mode)
            self._stats["rgb_payload_bytes"] = self._stats.get("rgb_payload_bytes", 0) + self._stats["payload_bytes"] - before_bytes
            self._stats["seconds"] += time.perf_counter() - started
        else:
            result, loss, finite, _, _ = self.reduce_view(accumulator, loss_sum, finite)
        self._stats["batches"] = self._stats.get("batches", 0) + 1
        return result, loss, finite

    def broadcast_arrays(self, fields, *, root, uint_keys=()):
        """Broadcast one camera's GPU arrays, preserving its separate vote.

        All ranks pass the same names, array lengths and ``uint_keys``. Only
        ``root`` supplies meaningful contents; other ranks may reuse local
        evidence buffers or zero placeholders. Integer counters must be listed
        in ``uint_keys``; remaining fields are float32. No image/evidence data
        is transferred through CPU memory, and no raw evidence is summed across
        cameras. This operation cannot be recorded inside ``dr.freeze``.
        """
        if self._closed or self._failed:
            raise RuntimeError("NCCL collective is closed or has failed")
        if type(root) is not int or not 0 <= root < self.world_size:
            raise ValueError("Broadcast source rank must fit within the world size")
        uint_keys = set(uint_keys)
        if not uint_keys.issubset(fields) or any(not isinstance(name, str) for name in fields):
            raise ValueError("Broadcast fields must be named arrays, including every uint_key")
        if not fields:
            return {}
        started = time.perf_counter()
        dr, mi = self.dr, self.mi
        result, operations = {}, []
        for name in sorted(fields):
            dtype = _NCCL_UINT32 if name in uint_keys else _NCCL_FLOAT32
            array_type = mi.UInt32 if dtype == _NCCL_UINT32 else mi.Float
            source = array_type(fields[name])
            if self.world_size == 1:
                result[name] = source
            else:
                # NCCL supports an in-place source-rank broadcast. Its bytes
                # already equal the result, so preserve source ownership and
                # avoid a second complete model/Adam allocation on rank zero.
                destination = source if self.rank == root else dr.empty(array_type, dr.width(source))
                operations.append((source, destination, dtype))
                result[name] = destination
        if operations:
            dr.eval(*[array for source, destination, _ in operations for array in (source, destination)])
        descriptors = []
        for source, destination, dtype in operations:
            width = int(dr.width(source))
            if not width:
                continue
            info = (mi._reconstruction_float_buffer_info if dtype == _NCCL_FLOAT32
                    else mi._reconstruction_uint_buffer_info)
            source_info, destination_info = info(source), info(destination)
            if int(source_info["size"]) != width or int(destination_info["size"]) != width:
                raise RuntimeError("NCCL broadcast pointer metadata does not match the allocated array")
            descriptors.append((int(source_info["pointer"]), int(destination_info["pointer"]), width, dtype))
        try:
            if descriptors:
                with self._context():
                    self._check(self._lib.ncclGroupStart(), "ncclGroupStart")
                    try:
                        for source, destination, width, dtype in descriptors:
                            self._check(self._lib.ncclBroadcast(source, destination, width, dtype, root,
                                                               self._comm, self.device["stream"]),
                                        "ncclBroadcast")
                    finally:
                        self._check(self._lib.ncclGroupEnd(), "ncclGroupEnd")
                dr.sync_thread()
            self._stats["broadcasts"] = self._stats.get("broadcasts", 0) + 1
            self._stats["calls"] += len(descriptors)
            self._stats["payload_bytes"] += sum(width * 4 for _, _, width, _ in descriptors)
            self._stats["seconds"] += time.perf_counter() - started
            return result
        except BaseException:
            self._failed = True
            raise

    def statistics(self):
        owners = getattr(self, "_owners", None)
        return {"backend": "nccl" if self.world_size > 1 else "single-rank",
                "version": self.version, "library": self.library,
                "rank": self.rank, "world_size": self.world_size,
                "device": dict(self.device), "payload_scope": "per-rank reduced input bytes, not wire traffic",
                **self._stats, "buffer_owners": owners.statistics() if owners is not None else None}

    def close(self, abort=False):
        if self._closed:
            return
        owners = getattr(self, "_owners", None)
        try:
            if self._comm:
                abort = abort or self._failed
                if not abort and owners is not None:
                    try:
                        owners.finish()
                    except BaseException:
                        # Failed event/wait cleanup must still abort transport.
                        # Keep ownership and allow a later close() retry.
                        self._failed = True
                        with self._context():
                            self._check(self._lib.ncclCommAbort(self._comm), "ncclCommAbort")
                        self._comm = ctypes.c_void_p()
                        raise
                with self._context():
                    function = self._lib.ncclCommAbort if abort else self._lib.ncclCommDestroy
                    self._check(function(self._comm), function.__name__)
                self._comm = ctypes.c_void_p()
            if owners is not None:
                owners.finish()
            # Cover a partially enqueued group that failed before its event was
            # recorded. Never release these owners if the stream wait fails.
            if getattr(self, "_launch_owners", None) is not None:
                self.dr.sync_thread()
        except BaseException:
            self._failed = True
            raise
        self._launch_owners = None
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close(abort=exc_type is not None)

