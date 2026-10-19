"""Generated RGB-only target and sampling checks, without a project dataset.

The default checks use only CPU bitmap decoding. --cuda additionally checks
the real three-plane device cache. All image files live in a temporary folder.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from reconstruction_common import (
    ContourSampler, ReferenceImageCache, UniformSampler, read_reference,
)
from preview import RuntimeConfig, workspace_path
from reconstruction_reference import GPUReferenceCache


def expect_value_error(action):
    try:
        action()
    except ValueError:
        return
    raise AssertionError("Expected ValueError")


def check_rgb_references(directory, mi, np):
    rgba = np.array([[[0, .25, .5, 0], [.25, .5, .75, .5]],
                     [[.5, .25, 1, 1], [1, .75, .5, .25]]], dtype=np.float32)
    expected = rgba[..., :3] * rgba[..., 3:4] + (1 - rgba[..., 3:4])
    rgba_path = directory / "rgba.exr"
    rgb_path = directory / "rgb.exr"
    mi.Bitmap(rgba).write(str(rgba_path))
    mi.Bitmap(expected).write(str(rgb_path))
    np.testing.assert_array_equal(read_reference(rgba_path, 2, 2, mi, np), rgba)
    rgb_from_rgba = read_reference(rgba_path, 2, 2, mi, np, rgb_only=True)
    rgb_from_rgb = read_reference(rgb_path, 2, 2, mi, np, rgb_only=True)
    np.testing.assert_array_equal(rgb_from_rgba, expected)
    np.testing.assert_array_equal(rgb_from_rgb, expected)
    assert rgb_from_rgba.shape == (2, 2, 3) and rgb_from_rgba.flags.c_contiguous
    # The retained image owns three channels; its base is not the decoded RGBA.
    assert rgb_from_rgba.base is None and rgb_from_rgba.nbytes == 2 * 2 * 3 * 4
    np.testing.assert_array_equal(
        read_reference(rgba_path, 2, 2, mi, np, rgb_only=True, background=(0, 0, 0)),
        rgba[..., :3] * rgba[..., 3:4])
    expect_value_error(lambda: read_reference(rgba_path, 3, 2, mi, np, rgb_only=True))
    expect_value_error(lambda: read_reference(rgba_path, 2, 2, mi, np,
                                              rgb_only=True, background=(0, float("nan"), 0)))

    cache = ReferenceImageCache(2, 2, mi, np, capacity_mib=1, rgb_only=True)
    cached = cache.get(rgba_path)
    assert cache.get(rgba_path) is cached and not cached.flags.writeable
    assert cache.statistics()["resident_bytes"] == cached.nbytes == 48
    assert cache.statistics()["hits"] == cache.statistics()["misses"] == 1
    legacy = ReferenceImageCache(2, 2, mi, np, capacity_mib=1)
    assert legacy.get(rgba_path).shape == (2, 2, 4)
    assert legacy.statistics()["resident_bytes"] == 64
    disabled = ReferenceImageCache(2, 2, mi, np, capacity_mib=0, rgb_only=True)
    np.testing.assert_array_equal(disabled.get(rgba_path), expected)
    assert disabled.statistics()["resident_bytes"] == 0
    return {"rgb_files_and_rgba_composition": "passed", "three_channel_host_cache": "passed"}


def check_uniform_sampler(np):
    class ShapeOnly:
        shape = (3, 5, 3)

        def __getitem__(self, key):
            raise AssertionError("Uniform sampling must not inspect target values")

    sampler = UniformSampler(np)
    offsets, inverse_pdf, token = sampler.sample("view", ShapeOnly(), np.random.default_rng(91))
    old_offsets, old_pdf, old_token = ContourSampler(np, enabled=False).sample(
        "view", np.ones((3, 5), dtype=np.float32), np.random.default_rng(91))
    np.testing.assert_array_equal(offsets, old_offsets)
    np.testing.assert_array_equal(inverse_pdf, old_pdf)
    assert token is old_token is None
    assert sampler.state_dict() == {}
    sampler.load_state_dict({})
    expect_value_error(lambda: sampler.load_state_dict({"view": {"mean_alpha": []}}))
    expect_value_error(lambda: sampler.sample("view", np.ones((3, 5, 4)), np.random.default_rng(91)))
    return {"uniform_sampling_without_alpha_or_target_reads": "passed"}


def check_rgb_gpu_cache(mi, dr, np):
    rgb = np.arange(8 * 6 * 3, dtype=np.float32).reshape(6, 8, 3) / np.float32(128)
    cache = GPUReferenceCache(8, 6, mi, dr, np, capacity_mib=1, rgb_only=True)
    device = cache.get("rgb-generated.exr", rgb)
    assert cache.get("rgb-generated.exr", rgb) is device
    assert cache.statistics()["resident_bytes"] == cache.statistics()["uploaded_bytes"] == rgb.nbytes
    assert dr.width(device.alpha) == 1
    for begin, end in ((0, 48), (0, 7), (13, 29), (47, 48)):
        color, compatibility_alpha = device.batch(begin, end)
        dr.eval(color, compatibility_alpha)
        np.testing.assert_array_equal(np.asarray(color).T, rgb.reshape(-1, 3)[begin:end])
        np.testing.assert_array_equal(np.asarray(compatibility_alpha), [1])
        assert dr.width(compatibility_alpha) == 1
    expect_value_error(lambda: cache.get("wrong-channels.exr", np.zeros((6, 8, 4), dtype=np.float32)))
    invalid = rgb.copy()
    invalid[0, 0, 0] = np.nan
    expect_value_error(lambda: cache.get("nonfinite.exr", invalid))
    return {"three_plane_cuda_cache_and_scalar_compatibility": "passed"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", default=RuntimeConfig().build_dir)
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args()
    build = workspace_path(args.build_dir, Path(__file__).resolve().parents[4])
    handles = []
    if os.name == "nt":
        handles.append(os.add_dll_directory(str(build)))
    sys.path.insert(0, str(build / "python"))
    import mitsuba as mi
    import numpy as np
    if not Path(mi.__file__).resolve().is_relative_to(build / "python"):
        raise RuntimeError("These checks require the local source build")
    with tempfile.TemporaryDirectory(prefix="mitsuba-rgb-reference-") as directory:
        checks = check_rgb_references(Path(directory), mi, np)
        checks.update(check_uniform_sampler(np))
        if args.cuda:
            import drjit as dr
            mi.set_variant("cuda_ad_rgb")
            checks.update(check_rgb_gpu_cache(mi, dr, np))
            dr.sync_thread()
        print(json.dumps({"status": "passed", "checks": checks}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
