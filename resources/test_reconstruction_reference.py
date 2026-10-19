"""Generated CUDA checks for bounded GPU reference caching and exact targets.

No project dataset or experiment is opened. Runtime artifacts live in a
temporary directory and are removed after the source-build test finishes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parent))

from reconstruction_common import initialize_runtime
from reconstruction_preview import RuntimeConfig
from reconstruction_reference import GPUReferenceCache


def expect_value_error(action):
    try:
        action()
    except ValueError:
        return
    raise AssertionError("Expected ValueError")


def check_cache(mi, dr, np):
    width, height = 128, 256
    # Non-contiguous input also exercises independent, contiguous plane uploads.
    data = np.arange(width * height * 4, dtype=np.float32).reshape(height, width, 4)
    rgba = (data / np.float32(data.size))[..., ::-1]
    rgba.setflags(write=False)
    before = rgba.copy()
    cache = GPUReferenceCache(width, height, mi, dr, np, capacity_mib=1)
    first = cache.get("reference-a.png", rgba)
    assert cache.get("unused-subdirectory/../reference-a.png", rgba) is first
    assert cache.statistics() == {
        "capacity_bytes": 1048576, "resident_bytes": 524288, "images": 1,
        "hits": 1, "misses": 1, "uploads": 1, "uploaded_bytes": 524288, "evictions": 0,
    }
    count = width * height
    full_rgb, full_alpha = first.batch(0, count)
    assert full_rgb is first.rgb and full_alpha is first.alpha
    for begin, end in ((0, count), (0, 37), (37, 74), (count - 13, count)):
        rgb, alpha = first.batch(begin, end)
        dr.eval(rgb, alpha)
        expected = rgba.reshape(-1, 4)[begin:end]
        np.testing.assert_array_equal(np.asarray(rgb).T, expected[:, :3])
        np.testing.assert_array_equal(np.asarray(alpha), expected[:, 3])
    np.testing.assert_array_equal(rgba, before)
    for begin, end in ((-1, 1), (0, 0), (0, count + 1), (True, 3), (0, 1.5)):
        expect_value_error(lambda: first.batch(begin, end))

    second = cache.get("reference-b.png", rgba)
    assert cache.get("reference-a.png", rgba) is first  # Make A most recent.
    cache.get("reference-c.png", rgba)  # Must evict B, not A.
    assert cache.get("reference-a.png", rgba) is first
    assert cache.statistics()["evictions"] == 1
    assert cache.get("reference-b.png", rgba) is not second
    stats = cache.statistics()
    assert stats["images"] == 2 and stats["resident_bytes"] == stats["capacity_bytes"]
    assert stats["uploads"] == 4 and stats["evictions"] == 2 and stats["hits"] == 3
    cache.clear()
    cleared = cache.statistics()
    assert cleared["resident_bytes"] == 0 and cleared["images"] == 0
    assert cleared["uploads"] == stats["uploads"] and cleared["evictions"] == stats["evictions"]
    # A caller retaining an evicted reference can finish its current GPU work.
    np.testing.assert_array_equal(np.asarray(first.batch(count - 1, count)[1]),
                                  rgba.reshape(-1, 4)[-1:, 3])

    disabled = GPUReferenceCache(width, height, mi, dr, np, capacity_mib=0)
    assert disabled.get("reference-a.png", rgba) is None
    assert disabled.statistics()["uploads"] == disabled.statistics()["resident_bytes"] == 0
    oversized_rgba = np.zeros((height, 257, 4), dtype=np.float32)
    oversized = GPUReferenceCache(257, height, mi, dr, np, capacity_mib=1)
    assert oversized.get("oversized.png", oversized_rgba) is None
    assert oversized.statistics()["uploads"] == oversized.statistics()["resident_bytes"] == 0

    for capacity in (-1, True, 1.5):
        expect_value_error(lambda: GPUReferenceCache(width, height, mi, dr, np, capacity))
    for bad_width, bad_height in ((0, 1), (1, -1), (True, 1), (1.5, 1), (65536, 65536)):
        expect_value_error(lambda: GPUReferenceCache(bad_width, bad_height, mi, dr, np))
    for invalid in (rgba.astype(np.float64), rgba[:, :, :3], rgba.tolist()):
        expect_value_error(lambda: cache.get("invalid.png", invalid))
    for bad_value in (float("nan"), float("inf"), -0.1, 1.1):
        invalid = rgba.copy()
        invalid[0, 0, 3] = bad_value
        expect_value_error(lambda: cache.get("invalid.png", invalid))
    assert cache.statistics()["uploads"] == 4
    return {"exact_rgba_and_chunk_tail": "passed", "canonical_key_reuse": "passed",
            "bounded_lru_and_clear": "passed", "disabled_and_oversized": "passed",
            "invalid_input_rejected": "passed", "cache_statistics_before_clear": stats}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", default=RuntimeConfig().build_dir)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="mitsuba-reference-cache-regression-") as directory:
        report = {}
        mi, dr, np, handles = initialize_runtime(RuntimeConfig(args.build_dir, args.gpu),
                                               Path(directory), report)
        checks = check_cache(mi, dr, np)
        dr.sync_thread()
        print(json.dumps({"status": "passed", "checks": checks}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
