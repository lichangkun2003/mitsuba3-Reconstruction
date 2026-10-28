"""CPU-only tests of NCCL packing, reductions, pointer ownership, and errors.

These use a fake NCCL library over NumPy buffers; they do not validate real
CUDA/NCCL interoperability, which requires the separate two-GPU smoke test.
"""

from __future__ import annotations

import ctypes
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from reconstruction_collective import (
    GRADIENT_KEYS, MAX_COUNTER_KEYS, SUM_COUNTER_KEYS, NcclCollective,
    _NCCL_FLOAT32, _NCCL_MAX, _NCCL_MIN, _NCCL_SUM,
)


def _float(value):
    return np.atleast_1d(np.asarray(value, dtype=np.float32))


def _uint(value):
    return np.atleast_1d(np.asarray(value, dtype=np.uint32))


class _Dr:
    def __init__(self):
        self.syncs = 0

    @staticmethod
    def width(value):
        return int(value.size)

    @staticmethod
    def concat(values):
        # NumPy permits empty concatenation, while the current Dr.Jit CUDA
        # implementation rejects zero-width index arithmetic internally.
        assert values and all(value.size for value in values), "Empty groups must bypass concat"
        return np.concatenate(values)

    @staticmethod
    def empty(array_type, size):
        return np.empty(size, dtype=np.float32 if array_type is _float else np.uint32)

    @staticmethod
    def eval(*args):
        pass

    @staticmethod
    def arange(array_type, size):
        assert size > 0, "Empty groups must bypass offset indexing"
        return np.arange(size, dtype=np.uint32)

    @staticmethod
    def opaque(array_type, value):
        return array_type(value)

    @staticmethod
    def gather(array_type, value, index):
        return array_type(value[index])

    def sync_thread(self):
        self.syncs += 1


class _Lib:
    def __init__(self, world_size=2):
        self.world_size = world_size
        self.calls = []
        self.broadcast_calls = []
        self.grouped = False

    def ncclGroupStart(self):
        assert not self.grouped
        self.grouped = True
        return 0

    def ncclGroupEnd(self):
        assert self.grouped
        self.grouped = False
        return 0

    def ncclAllReduce(self, source, destination, count, dtype, operation, comm, stream):
        assert self.grouped
        assert source != destination, "NCCL must write into a fresh receiving allocation"
        assert stream == 77
        ctype = ctypes.c_float if dtype == _NCCL_FLOAT32 else ctypes.c_uint32
        source_array = np.ctypeslib.as_array((ctype * count).from_address(source))
        destination_array = np.ctypeslib.as_array((ctype * count).from_address(destination))
        # Deliberately different peer values catch SUM versus MAX mistakes.
        if operation == _NCCL_SUM:
            destination_array[:] = source_array * self.world_size + self.world_size * (self.world_size - 1) // 2
        elif operation == _NCCL_MAX:
            destination_array[:] = source_array + self.world_size - 1
        elif operation == _NCCL_MIN:
            destination_array[:] = 0  # The other rank detected a nonfinite value.
        else:
            raise AssertionError(operation)
        self.calls.append((count, dtype, operation))
        return 0

    def ncclBroadcast(self, source, destination, count, dtype, root, comm, stream):
        assert self.grouped and stream == 77
        assert (source == destination) == (root == 0), "Source-rank broadcasts should retain their existing buffer"
        ctype = ctypes.c_float if dtype == _NCCL_FLOAT32 else ctypes.c_uint32
        source_array = np.ctypeslib.as_array((ctype * count).from_address(source))
        destination_array = np.ctypeslib.as_array((ctype * count).from_address(destination))
        # Fake root-specific values ensure evidence from different cameras is
        # broadcast separately rather than added together or treated as rank 0.
        destination_array[:] = source_array + root * 10
        self.broadcast_calls.append((count, dtype, root))
        return 0

    @staticmethod
    def ncclGetErrorString(code):
        return b"mock error"


def _fixture(world_size=2):
    def info(value):
        return {"pointer": value.ctypes.data, "size": value.size}

    collective = NcclCollective.__new__(NcclCollective)
    collective.mi = SimpleNamespace(
        Float=_float, UInt32=_uint, Bool=lambda value: np.asarray(value, dtype=np.bool_),
        _reconstruction_float_buffer_info=info, _reconstruction_uint_buffer_info=info,
        _reconstruction_cuda_push_context=lambda context: None,
        _reconstruction_cuda_pop_context=lambda: None,
    )
    collective.dr = _Dr()
    collective.np = np
    collective.rank, collective.world_size = 0, world_size
    collective._closed = collective._failed = False
    collective._comm = ctypes.c_void_p(33)
    collective._lib = _Lib(world_size)
    collective.device = {"context": 88, "stream": 77, "device": 0, "raw_device": 0, "device_count": 2}
    collective.version, collective.library = 21800, "cpu-mock"
    collective._stats = {"views": 0, "calls": 0, "payload_bytes": 0, "seconds": 0.}
    return collective


def _accumulator(count):
    sizes = (27, 9, 3, 3, 3)
    result = {"gradients": {
        key: np.arange(count * width, dtype=np.float32) + i * .25
        for i, (key, width) in enumerate(zip(GRADIENT_KEYS, sizes))}}
    for i, key in enumerate(SUM_COUNTER_KEYS + MAX_COUNTER_KEYS):
        result[key] = np.arange(count, dtype=np.uint32) + i
    return result


def main():
    collective = _fixture()
    source = _accumulator(3)
    raw, alpha = np.array([0., .2, .9, 0.], dtype=np.float32), np.array([.1, 0., 0., .8], dtype=np.float32)
    result, loss, finite, out_raw, out_alpha = collective.reduce_view(source, 7., True, raw, alpha)
    for key in GRADIENT_KEYS:
        np.testing.assert_array_equal(result["gradients"][key], source["gradients"][key] * 2 + 1)
    for key in SUM_COUNTER_KEYS:
        np.testing.assert_array_equal(result[key], source[key] * 2 + 1)
    for key in MAX_COUNTER_KEYS:
        np.testing.assert_array_equal(result[key], source[key] + 1)
    np.testing.assert_array_equal(loss, [15.])
    np.testing.assert_array_equal(finite, [False])
    np.testing.assert_array_equal(out_raw, raw * 2 + 1)
    np.testing.assert_array_equal(out_alpha, alpha * 2 + 1)
    assert collective.dr.syncs == 1
    assert [operation for _, _, operation in collective._lib.calls] == [
        _NCCL_SUM, _NCCL_SUM, _NCCL_MAX, _NCCL_SUM, _NCCL_MIN, _NCCL_SUM, _NCCL_SUM]
    stats = collective.statistics()
    assert stats["calls"] == 7 and stats["views"] == 1
    assert stats["payload_bytes"] == (3 * 50 + 2 + 8) * 4

    # Distortion components share the existing loss collective. Unlike
    # mean-of-rank-means, this raw SUM is correct for unequal pixel shards.
    components = _fixture()
    source = _accumulator(2)
    source.update(rgb_loss_sum=_float(2.5), distortion_loss_sum=_float(.125))
    result, loss, finite, _, _ = components.reduce_view(source, 3., True)
    np.testing.assert_array_equal(loss, [7.])
    np.testing.assert_array_equal(result["rgb_loss_sum"], [6.])
    np.testing.assert_array_equal(result["distortion_loss_sum"], [1.25])
    assert components._lib.calls[3] == (3, _NCCL_FLOAT32, _NCCL_SUM)
    assert components.statistics()["calls"] == 5
    assert components.dr.syncs == 1

    # D-SSIM adds a scalar to the same transfer, not another NCCL launch.
    dssim_components = _fixture()
    source["dssim_loss_sum"] = _float(.375)
    result, _, _, _, _ = dssim_components.reduce_view(source, 3., True)
    np.testing.assert_array_equal(result["dssim_loss_sum"], [1.75])
    assert dssim_components._lib.calls[3] == (4, _NCCL_FLOAT32, _NCCL_SUM)
    assert dssim_components.statistics()["calls"] == 5
    assert dssim_components.dr.syncs == 1

    # Empty topology still participates in the loss/finite scalar collectives.
    empty = _fixture()
    result, _, _, out_raw, out_alpha = empty.reduce_view(_accumulator(0), 0., True)
    assert all(value.size == 0 for value in result["gradients"].values())
    assert out_raw is None and out_alpha is None
    assert empty.statistics()["calls"] == 2
    assert empty.dr.syncs == 1

    batch = _fixture(4)
    source = _accumulator(2)
    result, loss, finite = batch.reduce_batch(source, 3., True)
    assert batch.statistics()["batches"] == 1
    assert len(batch._lib.calls) == 5
    np.testing.assert_array_equal(result["appearance_valid"], source["appearance_valid"] * 4 + 6)
    before = len(batch._lib.calls)
    for kind in ("growth", "deletion"):
        mixed = dict(source, **{f"{kind}_gain": _float([1.])})
        try:
            batch.reduce_batch(mixed, 0., True)
        except ValueError:
            pass
        else:
            raise AssertionError("Distinct-camera raw topology votes must never be reduced together")
    assert len(batch._lib.calls) == before
    fields = {"gain": _float([.1, .2]), "observed": _uint([3, 4]), "empty": _float([])}
    for root in range(4):
        result = batch.broadcast_arrays(fields, root=root, uint_keys=("observed",))
        for name in fields:
            np.testing.assert_array_equal(result[name], fields[name] + root * 10)
        assert result["observed"].dtype == np.uint32
        if root == 0:
            assert result["observed"].ctypes.data == fields["observed"].ctypes.data
    assert batch.statistics()["broadcasts"] == 4
    assert batch.dr.syncs == 5
    assert len(batch._lib.broadcast_calls) == 8
    prior_syncs = batch.dr.syncs
    assert batch.broadcast_arrays({}, root=0) == {}
    assert batch.broadcast_arrays({"empty": _float([])}, root=0)["empty"].size == 0
    assert batch.dr.syncs == prior_syncs

    single = _fixture()
    single.world_size, single._lib = 1, None
    result, loss, finite = single.reduce_batch(_accumulator(2), 5., True)
    np.testing.assert_array_equal(result["gradients"][GRADIENT_KEYS[0]], _accumulator(2)["gradients"][GRADIENT_KEYS[0]])
    np.testing.assert_array_equal(loss, [5.])
    np.testing.assert_array_equal(finite, [True])
    copies = single.broadcast_arrays(fields, root=0, uint_keys=("observed",))
    for name in fields:
        np.testing.assert_array_equal(copies[name], fields[name])
    assert single.statistics()["calls"] == 0 and single.dr.syncs == 0

    for malformed in ("groups", "counts", "scalar", "missing_component", "component_shape"):
        broken, value = _fixture(), _accumulator(2)
        loss_value = 0.
        if malformed == "groups":
            del value["gradients"][GRADIENT_KEYS[0]]
        elif malformed == "counts":
            value[SUM_COUNTER_KEYS[0]] = np.zeros(3, dtype=np.uint32)
        elif malformed == "scalar":
            loss_value = [0., 1.]
        elif malformed == "missing_component":
            value["rgb_loss_sum"] = _float(1.)
        else:
            value["rgb_loss_sum"] = _float([1., 2.])
            value["distortion_loss_sum"] = _float(0.)
        try:
            broken.reduce_view(value, loss_value, True)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Malformed {malformed} were accepted")
        assert not broken._lib.calls
    print({"status": "passed", "scope": "CPU mock only", "checks": [
        "sum_gradients_and_validity", "max_evidence", "finite_min", "loss_sum",
        "fresh_receive_buffers", "optional_images", "single_sync", "empty_topology",
        "packed_distortion_component_sums", "reject_invalid_shapes", "statistics"]})


if __name__ == "__main__":
    main()
