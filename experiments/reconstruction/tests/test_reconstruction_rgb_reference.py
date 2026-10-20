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
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from reconstruction_common import (
    ContourSampler, ReferenceImageCache, UniformSampler, read_reference,
)
from preview import RuntimeConfig, workspace_path
from reconstruction_reference import GPUReferenceCache


def srgb_decode(values, np):
    """Independent float64 oracle for the PNG's encoded RGB values."""
    values = np.asarray(values, dtype=np.float64)
    return np.where(values <= .04045, values / 12.92, ((values + .055) / 1.055) ** 2.4)


def srgb_encode(values, np):
    values = np.asarray(values, dtype=np.float64)
    return np.where(values <= .0031308, 12.92 * values, 1.055 * values ** (1 / 2.4) - .055)


def write_encoded_png(path, pixels, mi):
    bitmap = mi.Bitmap(pixels)
    bitmap.set_srgb_gamma(True)
    bitmap.write(str(path))
    assert mi.Bitmap(str(path)).srgb_gamma(), "The test must exercise a tagged sRGB PNG"


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
    # The new compositing convention changes encoded images, not linear HDR.
    np.testing.assert_array_equal(
        read_reference(rgba_path, 2, 2, mi, np, rgb_only=True, reference_compositing="srgb"),
        expected)
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


def check_srgb_png_references(directory, mi, np):
    pixels = np.array([
        [[0, 0, 0, 128], [30, 100, 220, 128], [71, 143, 211, 255]],
        [[240, 60, 20, 0], [70, 150, 230, 1], [90, 170, 250, 254]],
    ], dtype=np.uint8)
    rgba_path = directory / "encoded-rgba.png"
    rgb_path = directory / "encoded-rgb.png"
    write_encoded_png(rgba_path, pixels, mi)
    write_encoded_png(rgb_path, pixels[..., :3].copy(), mi)
    encoded = pixels[..., :3].astype(np.float64) / 255
    alpha = pixels[..., 3:4].astype(np.float64) / 255
    legacy = read_reference(rgba_path, 3, 2, mi, np)
    np.testing.assert_allclose(legacy[..., 3:4], alpha, rtol=0, atol=7e-8)
    np.testing.assert_allclose(legacy[..., :3], srgb_decode(encoded, np), rtol=2e-6, atol=2e-7)

    for background in ((1, 1, 1), (.125, .4, .8), (0, 0, 0)):
        expected_encoded = encoded * alpha + srgb_encode(background, np) * (1 - alpha)
        expected_linear = srgb_decode(expected_encoded, np)
        actual = read_reference(rgba_path, 3, 2, mi, np, rgb_only=True,
                                background=background, reference_compositing="srgb")
        np.testing.assert_allclose(actual, expected_linear, rtol=2e-6, atol=2e-7)
        assert actual.shape == (2, 3, 3) and actual.flags.c_contiguous
        assert actual.base is None and actual.nbytes == 2 * 3 * 3 * 4
        # Transparent input colors disappear; alpha is a linear coverage value.
        np.testing.assert_allclose(actual[1, 0], background, rtol=2e-6, atol=2e-7)
        np.testing.assert_allclose(actual[0, 2], legacy[0, 2, :3], rtol=2e-6, atol=2e-7)
        # The final displayed reference is exactly the encoded-space composite,
        # without quantizing the composite back to 8 bits before decoding.
        np.testing.assert_allclose(srgb_encode(actual, np), expected_encoded, rtol=2e-6, atol=2e-7)

    new_white = read_reference(rgba_path, 3, 2, mi, np, rgb_only=True,
                               reference_compositing="srgb")
    old_white = read_reference(rgba_path, 3, 2, mi, np, rgb_only=True,
                               reference_compositing="linear")
    np.testing.assert_allclose(new_white[0, 0], srgb_decode(127 / 255, np), rtol=0, atol=2e-7)
    np.testing.assert_allclose(old_white[0, 0], 127 / 255, rtol=0, atol=7e-8)
    assert float(old_white[0, 0, 0] - new_white[0, 0, 0]) > .28

    # An ordinary RGB PNG is decoded exactly once, regardless of protocol.
    rgb_legacy = read_reference(rgb_path, 3, 2, mi, np, rgb_only=True)
    rgb_new = read_reference(rgb_path, 3, 2, mi, np, rgb_only=True,
                            reference_compositing="srgb")
    np.testing.assert_array_equal(rgb_new, rgb_legacy)
    for capacity in (0, 1):
        cache = ReferenceImageCache(3, 2, mi, np, capacity_mib=capacity, rgb_only=True,
                                    reference_compositing="srgb")
        target = cache.get(rgba_path)
        np.testing.assert_array_equal(target, new_white)
        np.testing.assert_array_equal(cache.get(rgba_path), new_white)
        if capacity:
            assert cache.statistics()["resident_bytes"] == new_white.nbytes == 72
            assert not target.flags.writeable
        else:
            assert cache.statistics()["resident_bytes"] == 0
    expect_value_error(lambda: read_reference(rgba_path, 3, 2, mi, np, rgb_only=True,
                                              reference_compositing="invalid"))
    return {"srgb_png_composite_before_decode": "passed",
            "alpha_coverage_nonwhite_background_and_rgb_noop": "passed",
            "srgb_three_channel_host_cache": "passed"}


def check_reference_pipeline_consistency(directory, mi, np):
    import evaluate
    import preview

    path = directory / "pipeline-reference.png"
    write_encoded_png(path, np.array([[[0, 0, 0, 128], [32, 100, 200, 192]]], np.uint8), mi)
    target = ReferenceImageCache(2, 1, mi, np, rgb_only=True,
                                 reference_compositing="srgb").get(path)
    # Preview decodes an existing Bitmap; training and evaluation load by path.
    preview_target = preview.reference_from_bitmap(
        mi.Bitmap(str(path)), mi, np, rgb_only=True, reference_compositing="srgb")
    np.testing.assert_array_equal(preview_target, target)
    common = SimpleNamespace(
        render_view=lambda *args: (target.copy(), np.ones((1, 2), np.float32)),
        read_reference=read_reference,
    )
    render = preview.RenderConfig(width=2, height=1, background=(1, 1, 1))
    view = {"index": 0, "camera": {}, "reference_path": path}
    config = evaluate.EvaluationConfig(spp=1)
    timings = dict.fromkeys(("render_seconds", "reference_seconds", "metrics_seconds", "write_seconds"), 0.)
    with patch.object(preview, "write_png"):
        record = evaluate.render_evaluation_record(
            config, common, None, render, view, "test", directory, mi, None, np,
            timings.copy(), timings.copy(), reference_compositing="srgb")
        legacy_record = evaluate.render_evaluation_record(
            config, common, None, render, view, "test", directory, mi, None, np,
            timings.copy(), timings.copy(), reference_compositing="linear")
    assert record["metrics"]["linear"]["perfect_match"]
    assert record["metrics"]["srgb"]["perfect_match"]
    assert legacy_record["metrics"]["linear"]["mse"] > .01
    return {"training_preview_evaluation_target_consistency": "passed"}


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
        checks.update(check_srgb_png_references(Path(directory), mi, np))
        checks.update(check_reference_pipeline_consistency(Path(directory), mi, np))
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
