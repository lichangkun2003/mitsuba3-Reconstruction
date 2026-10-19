"""Verify external CUDA writes update the original Dr.Jit buffer and JIT reads.

Uses only the source build and installed CUDA driver. No NCCL, Toolkit, dataset,
or experiment output is required; runtime metadata uses a temporary directory.
"""

from __future__ import annotations

import argparse
import ctypes
import json
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parent))

from reconstruction_common import initialize_runtime
from reconstruction_preview import RuntimeConfig


def check_collective_packing(mi, dr, np):
    """Exercise the real CUDA array rules without needing a second GPU/NCCL."""
    from reconstruction_collective import NcclCollective

    collective = NcclCollective.__new__(NcclCollective)
    collective.mi, collective.dr = mi, dr
    cases = (([], [], [], [], []), ([], [2, 4], [], [6], []), ([1, 3], [5, 7]))
    checked = 0
    for array_type, dtype in ((mi.Float, np.float32), (mi.UInt32, np.uint32)):
        for parts in cases:
            arrays = [array_type(np.asarray(part, dtype=dtype)) for part in parts]
            packed, widths = collective._pack(arrays, array_type)
            assert widths == [len(part) for part in parts]
            names = [f"group_{index}" for index in range(len(parts))]
            restored = collective._unpack(packed, names, widths, array_type)
            dr.eval(packed, *restored.values())
            dr.sync_thread()
            np.testing.assert_array_equal(np.asarray(packed), np.concatenate(parts).astype(dtype))
            for name, part in zip(names, parts):
                assert type(restored[name]) is array_type
                np.testing.assert_array_equal(np.asarray(restored[name]), np.asarray(part, dtype=dtype))
            checked += 1
    return {"status": "passed", "cases": checked,
            "scope": "real CUDA Float/UInt pack/unpack; no NCCL communication"}


def check_interop(mi, dr, np):
    driver = ctypes.WinDLL("nvcuda.dll") if sys.platform == "win32" else ctypes.CDLL("libcuda.so.1")
    memset = getattr(driver, "cuMemsetD32Async_v2", None) or driver.cuMemsetD32Async
    memset.argtypes = [ctypes.c_uint64, ctypes.c_uint32, ctypes.c_size_t, ctypes.c_void_p]
    memset.restype = ctypes.c_int
    context = mi._reconstruction_cuda_context()
    checks = []

    def write_and_check(value, accessor, bits, expected, name):
        dr.eval(value)
        info = accessor(value)
        assert int(info["size"]) == 7
        assert int(info["pointer"]) != 0
        mi._reconstruction_cuda_push_context(int(context["context"]))
        try:
            result = memset(int(info["pointer"]), bits, int(info["size"]), int(context["stream"]))
            if result:
                raise RuntimeError(f"CUDA external write failed in {name}: driver status {result}")
        finally:
            mi._reconstruction_cuda_pop_context()
        # Schedule a consumer before the host wait: this verifies ordering on
        # Dr.Jit's stream as well as the original Python array's ownership.
        computed = value + 1
        dr.eval(computed)
        dr.sync_thread()
        np.testing.assert_array_equal(np.asarray(value), np.full(7, expected))
        np.testing.assert_array_equal(np.asarray(computed), np.full(7, expected + 1))
        assert accessor(value)["pointer"] == info["pointer"]
        checks.append(name)

    write_and_check(dr.empty(mi.UInt32, 7), mi._reconstruction_uint_buffer_info,
                    123, 123, "uint32_external_write")
    write_and_check(dr.empty(mi.Float, 7), mi._reconstruction_float_buffer_info,
                    0x3fa00000, 1.25, "float32_external_write")
    # The pointer helper must update the actual owning Python object when
    # materializing a literal; mutating a temporary cast would leave it at zero.
    write_and_check(dr.zeros(mi.UInt32, 7), mi._reconstruction_uint_buffer_info,
                    17, 17, "uint32_literal_materialization")
    write_and_check(dr.zeros(mi.Float, 7), mi._reconstruction_float_buffer_info,
                    0x3f800000, 1., "float32_literal_materialization")
    rejected = 0
    for accessor, value in (
        (mi._reconstruction_float_buffer_info, mi.UInt32([1, 2])),
        (mi._reconstruction_uint_buffer_info, mi.Float([1, 2])),
        (mi._reconstruction_float_buffer_info, np.array([1, 2], dtype=np.float32)),
    ):
        try:
            accessor(value)
        except TypeError:
            rejected += 1
        else:
            raise AssertionError("Pointer helper accepted an implicit conversion to a temporary array")
    checks.append("wrong_type_conversions_rejected")
    return {"status": "passed", "checks": checks, "rejected_conversions": rejected,
            "device": context["device"], "raw_device": context["raw_device"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", default=RuntimeConfig().build_dir)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="mitsuba-cuda-interop-") as directory:
        mi, dr, np, handles = initialize_runtime(
            RuntimeConfig(build_dir=args.build_dir, gpu=args.gpu), Path(directory), {})
        result = check_interop(mi, dr, np)
        result["collective_packing"] = check_collective_packing(mi, dr, np)
        print(json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
