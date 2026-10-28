"""Small CUDA regressions for linear-RGB image D-SSIM and native chaining.

The NumPy oracle uses a direct 2D Gaussian convolution and double-precision
finite differences, independent of the native separable-filter adjoint.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from preview import RuntimeConfig
from reconstruction_common import initialize_runtime, restore_model
from reconstruction_replay import NativeLossReplay
from test_reconstruction_distortion import fixture, oracle as ray_oracle, ray, Y00


def dssim_oracle(image, reference, np):
    image, reference = np.asarray(image, dtype=np.float64), np.asarray(reference, dtype=np.float64)
    height, width, channels = image.shape
    assert reference.shape == image.shape and channels == 3
    offsets = np.arange(-5, 6, dtype=np.float64)
    kernel = np.exp(-offsets**2 / (2 * 1.5**2))
    kernel /= kernel.sum()
    kernel = kernel[:, None] * kernel[None, :]

    def blur(value):
        padded = np.pad(value, ((5, 5), (5, 5), (0, 0)))
        output = np.zeros_like(value)
        for y in range(11):
            for x in range(11):
                output += kernel[y, x] * padded[y:y+height, x:x+width]
        return output

    mu_x, mu_y = blur(image), blur(reference)
    var_x = np.maximum(blur(image**2)-mu_x**2, 0.)
    var_y = np.maximum(blur(reference**2)-mu_y**2, 0.)
    covariance = blur(image*reference)-mu_x*mu_y
    ssim = ((2*mu_x*mu_y+.01**2) * (2*covariance+.03**2) /
            ((mu_x**2+mu_y**2+.01**2) * (var_x+var_y+.03**2)))
    return 1 - np.clip(ssim, -1., 1.).mean(axis=2)


def color(image, mi):
    flat = image.reshape(-1, 3)
    return mi.Color3f(*(mi.Float(flat[:, channel].copy()) for channel in range(3)))


def image_value_gradient(image, reference, mi, dr, np):
    height, width, _ = image.shape
    result = mi.VoxelReconstruction.dssim_image_loss_and_grad(
        color(image, mi), color(reference, mi), width, height)
    dr.eval(result)
    return (np.asarray(result["loss"], dtype=float).reshape(height, width),
            np.asarray(result["dcolor"], dtype=float).T.reshape(image.shape))


def check_numpy():
    import numpy as np
    for shape in ((1, 1, 3), (3, 5, 3), (13, 12, 3)):
        for value in (0., .5, 1.4):
            image = np.full(shape, value)
            np.testing.assert_allclose(dssim_oracle(image, image, np), 0., atol=1e-12)
    return {"double_precision_identity_and_zero_padding_oracle": "passed"}


def check_configuration():
    from unittest.mock import patch
    import train
    assert train.GradientSettings().dssim_loss_weight == 0.
    config = copy.deepcopy(train.CONFIG)
    config.gradient.dssim_loss_weight = 0.
    native = train.settings(config)
    assert "dssim_loss_weight" not in native
    assert "image_width" not in native and "image_height" not in native
    assert (train.normalized_algorithm_group("gradient", {}) ==
            train.normalized_algorithm_group("gradient", {"dssim_loss_weight": 0.}))
    with patch.object(sys, "argv", ["train.py", "--dssim-weight", "0.25"]):
        args = train.parse_arguments()
    train.apply_arguments(config, args, json.loads('{"gradient":{"dssim_loss_weight":0.1}}'))
    assert config.gradient.dssim_loss_weight == .25
    for chunk in (0, config.render.width*config.render.height, config.render.width*config.render.height+1):
        config.training.chunk_size = chunk
        train.validate(config)
    config.training.chunk_size = 0
    for value in (-.1, True, float("nan"), float("inf")):
        changed = copy.deepcopy(config)
        changed.gradient.dssim_loss_weight = value
        try:
            train.validate(changed)
        except ValueError as error:
            assert "dssim_loss_weight" in str(error), str(error)
        else:
            raise AssertionError(f"Invalid weight accepted: {value}")
    for chunk in (1, config.render.width*config.render.height-1):
        changed = copy.deepcopy(config)
        changed.training.chunk_size = chunk
        try:
            train.validate(changed)
        except ValueError as error:
            assert "D-SSIM requires a complete image" in str(error), str(error)
        else:
            raise AssertionError(f"Partial image chunk accepted: {chunk}")
    legacy = train.TrainConfig()
    legacy.gradient.dssim_loss_weight = .1
    try:
        train.validate(legacy)
    except ValueError as error:
        assert "D-SSIM requires training.rgb_only=true" in str(error), str(error)
    else:
        raise AssertionError("Legacy alpha-supervised SSIM accepted")
    return {"JSON_CLI_weight_validation_full_image_and_legacy_default": "passed"}


def check_image_adjoint(mi, dr, np):
    rng = np.random.default_rng(526)
    cases = []
    for height, width in ((1, 1), (2, 4), (5, 7), (13, 12)):
        cases.append((rng.uniform(.02, 1.4, (height, width, 3)),
                      rng.uniform(.02, .95, (height, width, 3))))
    cases.extend((np.full((5, 7, 3), x), np.full((5, 7, 3), y))
                 for x, y in ((0., 0.), (.5, .5), (1.4, .8), (.2, .8)))
    largest_error = 0.
    checks = 0
    for image, reference in cases:
        # Quantize the input once, then compare with double precision at exactly
        # those values. There is no image clamp or sRGB encoding in this loss.
        image, reference = image.astype(np.float32).astype(float), reference.astype(np.float32).astype(float)
        actual, gradient = image_value_gradient(image, reference, mi, dr, np)
        np.testing.assert_allclose(actual, dssim_oracle(image, reference, np), atol=3e-5, rtol=3e-5)
        indices = sorted({0, 1, image.size-1, image.size//2, 3*(image.shape[1]+1) % image.size})
        for index in indices:
            positive, negative = image.copy(), image.copy()
            eps = 2e-5
            positive.flat[index] += eps
            negative.flat[index] -= eps
            expected = (dssim_oracle(positive, reference, np).sum() -
                        dssim_oracle(negative, reference, np).sum()) / (2*eps)
            value = gradient.flat[index]
            np.testing.assert_allclose(value, expected, atol=3e-4, rtol=3e-3,
                err_msg=f"shape={image.shape}, component={index}")
            largest_error = max(largest_error, abs(value-expected))
            checks += 1
    return {"DSSIM_direct_2D_values_and_image_finite_differences": checks,
            "maximum_image_gradient_error": largest_error}


def view_data(np):
    width, height = 7, 5
    x, y = np.meshgrid(np.linspace(1.13, 1.82, width), np.linspace(1.18, 1.78, height))
    origins = np.column_stack((x.ravel(), y.ravel(), np.full(width*height, -1.)))
    reference = np.random.default_rng(402).uniform(.15, .85, (height, width, 3)).astype(np.float32)
    return origins, reference


def evaluate(model, mi, dr, np, *, coefficient=.4, replay=None, options=None, reference=None,
             omit_weight=False, inverse_pdf=1.):
    origins, default = view_data(np)
    reference = default if reference is None else reference
    native = {"rgb_only": 1., "max_hits": 8., "max_candidates": 4.,
              "transmittance_threshold": .00001, "distortion_loss_weight": .1}
    if not omit_weight:
        native.update(dssim_loss_weight=coefficient, image_width=7., image_height=5.)
    if options:
        native.update(options)
    arguments = (ray(mi, np, origins), color(reference, mi), mi.Float(1.), mi.Color3f(1.), mi.Float(inverse_pdf), native)
    result = model.loss_and_grad(*arguments) if replay is None else replay(model, *arguments)
    dr.eval(result)
    dr.sync_thread()
    return result


def compare_results(a, b, np, *, exact=False):
    for key in ("rgb", "raw_rgb", "alpha", "raw_alpha", "hit_count", "loss", "rgb_loss", "distortion_loss"):
        if exact:
            np.testing.assert_array_equal(np.asarray(a[key]), np.asarray(b[key]), err_msg=key)
        else:
            np.testing.assert_allclose(np.asarray(a[key]), np.asarray(b[key]), atol=3e-6, rtol=2e-5, err_msg=key)
    if "dssim_loss" in a and "dssim_loss" in b:
        np.testing.assert_allclose(np.asarray(a["dssim_loss"]), np.asarray(b["dssim_loss"]), atol=3e-6, rtol=2e-5)
    for key in a["gradients"]:
        if exact:
            np.testing.assert_array_equal(np.asarray(a["gradients"][key]), np.asarray(b["gradients"][key]), err_msg=key)
        else:
            np.testing.assert_allclose(np.asarray(a["gradients"][key]), np.asarray(b["gradients"][key]),
                                       atol=2e-5, rtol=2e-4, err_msg=key)


def fixture_image(checkpoint, np):
    origins, reference = view_data(np)
    image = []
    radiance = checkpoint["model_state"]["radiance_sh"].reshape(-1, 3)*Y00
    for origin in origins:
        result = ray_oracle(checkpoint, np, origin=origin)
        value, transmittance = np.zeros(3), 1.
        for hit in result["hits"]:
            _, row, opacity, _, _ = hit
            value += transmittance*opacity*radiance[row]
            transmittance *= 1-opacity
        image.append(value+transmittance)
    return np.asarray(image).reshape(reference.shape)


def check_native_chain(mi, dr, np):
    checkpoint = fixture(np)
    model = restore_model(checkpoint, mi)
    reference = view_data(np)[1]
    off, on = evaluate(model, mi, dr, np, coefficient=0.), evaluate(model, mi, dr, np)
    default = evaluate(model, mi, dr, np, omit_weight=True)
    compare_results(off, default, np, exact=True)
    rendered = np.asarray(on["rgb"]).T.reshape(reference.shape)
    np.testing.assert_allclose(rendered, fixture_image(checkpoint, np), atol=2e-6, rtol=2e-6)
    expected = dssim_oracle(rendered, reference, np).ravel()
    np.testing.assert_allclose(np.asarray(on["dssim_loss"]), expected, atol=1e-5, rtol=3e-5)
    np.testing.assert_allclose(np.asarray(on["loss"]), np.asarray(off["loss"])+.4*expected, atol=1e-5)
    for key in ("rgb", "alpha", "hit_count", "rgb_loss", "distortion_loss"):
        np.testing.assert_array_equal(np.asarray(off[key]), np.asarray(on[key]))

    # Appearance has exact continuous derivatives while membership is fixed.
    # Use independent image composition and scalar SSIM finite differences;
    # do not compare the existing hard-visibility geometry surrogate to FD.
    checks = 0
    for name in ("radiance_sh", "opacity_sh"):
        actual = (np.asarray(on["gradients"][name], dtype=float)-np.asarray(off["gradients"][name], dtype=float))/.4
        for index in range(len(actual)):
            values = []
            eps = 2e-5
            for sign in (1, -1):
                changed = copy.deepcopy(checkpoint)
                changed["model_state"][name][index] += sign*eps
                values.append(dssim_oracle(fixture_image(changed, np), reference, np).sum())
            expected = (values[0]-values[1])/(2*eps)
            np.testing.assert_allclose(actual[index], expected, atol=3e-4, rtol=3e-3, err_msg=f"{name}[{index}]")
            checks += 1
    assert any(np.max(np.abs(np.asarray(on["gradients"][name])-np.asarray(off["gradients"][name]))) > 1e-5
               for name in ("centers", "log_scales", "rotation_tangent")), "SSIM did not reach geometry surrogate"
    empty = restore_model(fixture(np, 0), mi)
    result = evaluate(empty, mi, dr, np)
    np.testing.assert_allclose(np.asarray(result["dssim_loss"]), dssim_oracle(np.ones_like(reference), reference, np).ravel(), atol=1e-5)
    return {"zero_weight_exact_baseline_empty_model_and_RGB_components": "passed",
            "appearance_gradient_finite_differences": checks,
            "geometry_surrogate_receives_image_gradient": "passed"}


def check_execution_and_evidence(mi, dr, np):
    checkpoint = fixture(np)
    model = restore_model(checkpoint, mi)
    proposal = fixture(np, 1)["model_state"]
    proposal["cell_indices"][:] = 53
    model.set_growth_candidates({name: (mi.UInt32(value) if name == "cell_indices" else mi.Float(value))
                                 for name, value in proposal.items()})
    options = {"collect_growth_evidence": 1., "collect_rgb_deletion_evidence": 1., "max_growth_hits": 4.}
    off, on = evaluate(model, mi, dr, np, coefficient=0., options=options), evaluate(model, mi, dr, np, options=options)
    assert np.asarray(on["growth_observed"]).sum() > 0
    assert np.asarray(on["deletion_observed"]).sum() > 0
    for name in ("growth_gain", "growth_observed", "growth_unknown", "deletion_gain", "deletion_observed", "deletion_unknown"):
        np.testing.assert_array_equal(np.asarray(on[name]), np.asarray(off[name]), err_msg=name)

    direct = evaluate(model, mi, dr, np)
    compact = evaluate(model, mi, dr, np, options={"compact_gradient_capacity": 3.})
    count = int(np.asarray(compact["voxel_count"])[0])
    ids = np.asarray(compact["voxel_ids"])[:count]
    for name, value in direct["gradients"].items():
        expected = np.asarray(value).reshape(3, -1)[ids]
        actual = np.asarray(compact["gradients"][name]).reshape(3, -1)[:count]
        np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-4, err_msg=name)
    compare_results(direct, evaluate(model, mi, dr, np, options={"use_octree": 0.}), np)
    profiled = evaluate(model, mi, dr, np, options={"profile_stages": 1.})
    compare_results(direct, profiled, np)
    assert "image_loss" in profiled["stage_profile"]

    replay = NativeLossReplay(dr)
    for reference in (view_data(np)[1], view_data(np)[1]*.7):
        expected = evaluate(model, mi, dr, np, reference=reference)
        for _ in range(2):
            compare_results(expected, evaluate(model, mi, dr, np, reference=reference, replay=replay), np)
    changed = checkpoint["model_state"]["radiance_sh"].copy()*.9
    model.set_parameters({"radiance_sh": mi.Float(changed)})
    compare_results(evaluate(model, mi, dr, np), evaluate(model, mi, dr, np, replay=replay), np)
    assert replay.statistics()["replays"] >= 3
    for options in ({"dssim_loss_weight": -.1}, {"image_width": 8.}, {"image_height": 0.}, {"rgb_only": 0.}):
        try:
            evaluate(model, mi, dr, np, options=options)
        except (RuntimeError, ValueError):
            pass
        else:
            raise AssertionError(f"Invalid image loss settings accepted: {options}")
    for inverse_pdf in (0., 2., float("nan")):
        # Device-side invalidation preserves asynchronous recording; the
        # training guard must reject it before Adam, without a host read here.
        invalid = evaluate(model, mi, dr, np, inverse_pdf=inverse_pdf)
        assert not np.isfinite(np.asarray(invalid["loss"])).all()
    return {"RGB_topology_evidence_unchanged": "passed", "dense_compact_DDA_octree_profile": "passed",
            "frozen_replay_reads_live_reference_and_parameters": "passed", "native_image_contract_validation": "passed"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", default=RuntimeConfig().build_dir)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--cpu-only", action="store_true")
    args = parser.parse_args()
    checks = [check_numpy(), check_configuration()]
    for check in checks:
        print(json.dumps(check), flush=True)
    if not args.cpu_only:
        with tempfile.TemporaryDirectory(prefix="mitsuba-dssim-regression-") as temporary:
            mi, dr, np, handles = initialize_runtime(RuntimeConfig(args.build_dir, args.gpu), Path(temporary), {})
            assert int(getattr(mi.VoxelReconstruction, "dssim_version", 0)) >= 1, "Rebuild this checkout before running D-SSIM regression"
            for check in (check_image_adjoint, check_native_chain, check_execution_and_evidence):
                checks.append(check(mi, dr, np))
                print(json.dumps(checks[-1]), flush=True)
    print(json.dumps({"status": "passed", "checks": checks}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
