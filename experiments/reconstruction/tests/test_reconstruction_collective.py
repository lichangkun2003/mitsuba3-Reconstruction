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
    def __init__(self):
        self.calls = []
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
        peer = source_array + 1
        if operation == _NCCL_SUM:
            destination_array[:] = source_array + peer
        elif operation == _NCCL_MAX:
            destination_array[:] = np.maximum(source_array, peer)
        elif operation == _NCCL_MIN:
            destination_array[:] = 0  # The other rank detected a nonfinite value.
        else:
            raise AssertionError(operation)
        self.calls.append((count, dtype, operation))
        return 0

    @staticmethod
    def ncclGetErrorString(code):
        return b"mock error"


def _fixture():
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
    collective.rank, collective.world_size = 0, 2
    collective._closed = collective._failed = False
    collective._comm = ctypes.c_void_p(33)
    collective._lib = _Lib()
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
    np.testing.assert_array_equal(out_raw, raw + (raw + 1))
    np.testing.assert_array_equal(out_alpha, alpha + (alpha + 1))
    assert collective.dr.syncs == 1
    assert [operation for _, _, operation in collective._lib.calls] == [
        _NCCL_SUM, _NCCL_SUM, _NCCL_MAX, _NCCL_SUM, _NCCL_MIN, _NCCL_SUM, _NCCL_SUM]
    stats = collective.statistics()
    assert stats["calls"] == 7 and stats["views"] == 1
    assert stats["payload_bytes"] == (3 * 50 + 2 + 8) * 4

    # Empty topology still participates in the loss/finite scalar collectives.
    empty = _fixture()
    result, _, _, out_raw, out_alpha = empty.reduce_view(_accumulator(0), 0., True)
    assert all(value.size == 0 for value in result["gradients"].values())
    assert out_raw is None and out_alpha is None
    assert empty.statistics()["calls"] == 2
    assert empty.dr.syncs == 1

    for malformed in ("groups", "counts", "scalar"):
        broken, value = _fixture(), _accumulator(2)
        loss_value = 0.
        if malformed == "groups":
            del value["gradients"][GRADIENT_KEYS[0]]
        elif malformed == "counts":
            value[SUM_COUNTER_KEYS[0]] = np.zeros(3, dtype=np.uint32)
        else:
            loss_value = [0., 1.]
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
        "reject_invalid_shapes", "statistics"]})


if __name__ == "__main__":
    main()
