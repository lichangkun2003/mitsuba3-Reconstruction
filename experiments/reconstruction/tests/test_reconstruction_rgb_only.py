"""Generated CUDA numerical regressions for the RGB-only reconstruction mode.

Uses a single ellipsoid, explicit hit/miss rays and 8x8 generated references.
Appearance gradients are checked against finite differences of RGB L2;
geometry checks exercise the surrogate without claiming it differentiates
hard visibility. No dataset or existing experiment is opened.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from reconstruction_common import (
    UniformSampler, initialize_runtime, read_reference, restore_model,
)
from reconstruction_optimizer import FalcorOptimizer
from preview import RuntimeConfig
from reconstruction_reference import GPUReferenceCache
from reconstruction_topology import TopologyManager
from train import CONFIG, finalize_view, train_view, validate


Y00 = 1 / (2 * math.sqrt(math.pi))


def fixture(np):
    config = copy.deepcopy(CONFIG)
    # Keep this regression as the fixed-topology RGB-only baseline even when
    # the experiment defaults enable online growth and deletion.
    config.topology.enable_growth = False
    config.topology.enable_deletion = False
    config.topology.use_rgb_deletion = False
    config.topology.use_growth_evidence = False
    config.topology.use_rgb_growth = False
    config.schedule.enable_coarse_growth = False
    config.render.width = config.render.height = 8
    config.training.num_epochs = config.schedule.total_epochs = 2
    config.training.chunk_size = 0
    config.model.sh_count = config.model.opacity_sh_count = 1
    config.schedule.start_resolution = 1
    config.schedule.target_resolution = 2
    config.optimizer.opacity_warmup_epochs = 0
    config.optimizer.opacity_ramp_epochs = 0
    payload = {
        "model_metadata": {"resolution": 1, "grid_min": [0., 0., 0.],
                           "grid_max": [1., 1., 1.], "sh_count": 1,
                           "opacity_sh_count": 1},
        "model_state": {
            "cell_indices": np.array([0], np.uint32),
            "centers": np.array([.5, .5, .5], np.float32),
            "log_scales": np.log(np.array([.28, .32, .3], np.float32)),
            "rotations": np.array([1., 0., 0., 0.], np.float32),
            "radiance_sh": np.array([.2, .35, .5], np.float32) / Y00,
            "opacity_sh": np.array([math.log(.6 / .4) / Y00], np.float32),
        },
    }
    view = {"index": 0, "camera": {
        "position": [.5, .5, -1.5], "forward": [0., 0., 1.],
        "right": [1., 0., 0.], "up": [0., 1., 0.], "camera_angle_x": .55,
    }}
    return config, payload, view


def ray_at(mi, x=.7, y=.56):
    ray = mi.Ray3f(mi.Point3f(x, y, -1.), mi.Vector3f(0., 0., 1.))
    ray.maxt = 4.
    return ray


def evaluate(model, ray, target, alpha, mi, dr, settings=None):
    # Deliberately leave legacy supervision/gating enabled here: rgb_only
    # must override it, rather than depending on callers zeroing every knob.
    options = {"rgb_only": 1., "alpha_loss_weight": 17.,
               "alpha_geometry_weight": 19., "background_carve_weight": 23.,
               "collect_deletion_evidence": 1., "geometry_transition_only": 1.,
               "geometry_two_sided_boundary": 1., "geometry_boundary_band_tau": .001}
    options.update(settings or {})
    result = model.loss_and_grad(ray, mi.Color3f(target), mi.Float(alpha),
                                 mi.Color3f(1.), mi.Float(1.), options)
    dr.eval(result)
    dr.sync_thread()
    return result


def compare(first, second, np):
    for name in first["gradients"]:
        np.testing.assert_allclose(np.asarray(first["gradients"][name]),
                                   np.asarray(second["gradients"][name]),
                                   atol=3e-6, rtol=3e-5, err_msg=name)
    for name in ("appearance_valid", "geometry_valid", "background_carve_valid",
                 "foreground_evidence", "background_evidence"):
        np.testing.assert_array_equal(np.asarray(first[name]), np.asarray(second[name]),
                                      err_msg=name)


def check_reference_alpha(payload, mi, dr, np):
    model = restore_model(payload, mi)
    for x, expected_hits in ((.7, 1), (.81, 0)):
        results = [evaluate(model, ray_at(mi, x), [.3, .45, .6], a, mi, dr)
                   for a in (0., .5, 1.)]
        for result in results:
            np.testing.assert_array_equal(np.asarray(result["hit_count"]), expected_hits)
            assert np.any(np.asarray(result["geometry_valid"]) > 0)
            assert np.any(np.abs(np.asarray(result["gradients"]["centers"])) > 1e-7)
            for name in ("background_carve_valid", "foreground_evidence", "background_evidence"):
                np.testing.assert_array_equal(np.asarray(result[name]), 0, err_msg=name)
            for value in result["gradients"].values():
                assert np.isfinite(np.asarray(value)).all()
            expected_loss = np.sum((np.asarray(result["rgb"]).reshape(3) - [.3, .45, .6]) ** 2)
            np.testing.assert_allclose(np.asarray(result["loss"]), expected_loss, atol=2e-7)
            compare(results[0], result, np)
            for name in ("loss", "rgb", "raw_rgb", "alpha", "raw_alpha"):
                np.testing.assert_array_equal(np.asarray(results[0][name]), np.asarray(result[name]),
                                              err_msg=f"reference_alpha changed {name}")
    return {"reference_alpha_0_half_1_loss_and_all_gradients": "passed",
            "miss_candidates_ignore_reference_alpha": "passed",
            "alpha_carving_and_deletion_evidence_disabled": "passed"}


def check_appearance_finite_difference(payload, mi, dr, np):
    model = restore_model(payload, mi)
    target = np.array([.15, .7, .25], dtype=np.float64)
    result = evaluate(model, ray_at(mi), target.tolist(), 0., mi, dr)
    state = payload["model_state"]
    parameters = np.concatenate((state["radiance_sh"], state["opacity_sh"])).astype(np.float64)

    def rgb_l2(values):
        color = Y00 * values[:3]
        opacity = 1 / (1 + math.exp(-Y00 * values[3]))
        rgb = opacity * color + (1 - opacity)
        return float(np.sum((rgb - target) ** 2))

    eps = 1e-5
    expected = []
    for index in range(len(parameters)):
        delta = np.zeros_like(parameters)
        delta[index] = eps
        expected.append((rgb_l2(parameters + delta) - rgb_l2(parameters - delta)) / (2 * eps))
    actual = np.concatenate([np.asarray(result["gradients"][name]).ravel()
                             for name in ("radiance_sh", "opacity_sh")])
    np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=3e-5)
    np.testing.assert_allclose(np.asarray(result["loss"]), rgb_l2(parameters), atol=2e-7)
    # Also perturb the native model and measure its rendered RGB, independently
    # of the native loss field. Geometry/hit topology is unchanged by these edits.
    rendered_fd = []
    for index in range(len(parameters)):
        name, offset = ("radiance_sh", index) if index < 3 else ("opacity_sh", 0)
        losses, coordinates = [], []
        for sign in (-1., 1.):
            perturbed = copy.deepcopy(payload)
            perturbed["model_state"][name][offset] += sign * .002
            coordinates.append(float(perturbed["model_state"][name][offset]))
            output = evaluate(restore_model(perturbed, mi), ray_at(mi), target.tolist(), .5, mi, dr)
            rgb = np.asarray(output["rgb"]).reshape(3).astype(np.float64)
            losses.append(float(np.sum((rgb - target) ** 2)))
        rendered_fd.append((losses[1] - losses[0]) / (coordinates[1] - coordinates[0]))
    np.testing.assert_allclose(actual, rendered_fd, atol=3e-5, rtol=2e-3)
    return {"rgb_l2_appearance_finite_difference": "passed",
            "native_render_finite_difference": "passed",
            "max_appearance_gradient_error": float(np.max(np.abs(actual - expected)))}


def check_geometry_signs(payload, mi, dr, np):
    model = restore_model(payload, mi)
    dark = evaluate(model, ray_at(mi), [0., 0., 0.], 1., mi, dr)
    white = evaluate(model, ray_at(mi), [1., 1., 1.], 0., mi, dr)
    dark_scale = np.asarray(dark["gradients"]["log_scales"])[0]
    white_scale = np.asarray(white["gradients"]["log_scales"])[0]
    assert dark_scale < -1e-6 and white_scale > 1e-6, (dark_scale, white_scale)
    for name in ("centers", "log_scales", "radiance_sh", "opacity_sh"):
        assert np.any(np.abs(np.asarray(white["gradients"][name])) > 1e-7), name
    # Compare both signs before/after an active symmetric clamp. This tests
    # the update behavior without finite-differencing discontinuous hard hits.
    for target, unclamped in (([0.] * 3, dark_scale), ([1.] * 3, white_scale)):
        clamped = evaluate(model, ray_at(mi), target, 0., mi, dr,
                           {"geometry_grad_clamp": 1e-4})
        value = np.asarray(clamped["gradients"]["log_scales"])[0]
        assert value * unclamped > 0 and abs(value) < abs(unclamped), (value, unclamped)
    return {"white_target_keeps_rgb_and_geometry_gradients": "passed",
            "geometry_expansion_and_shrinkage_with_symmetric_clamp": "passed"}


def check_dc_opacity(config, payload, mi, dr, np):
    model = restore_model(payload, mi)
    optimizer = FalcorOptimizer(model, config.optimizer)

    def directional_results():
        values = []
        for direction in ([1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [-1., 2., 3.]):
            direction = np.asarray(direction, np.float64)
            direction /= np.linalg.norm(direction)
            ray = mi.Ray3f(mi.Point3f((.5 - 2 * direction).tolist()),
                           mi.Vector3f(direction.tolist()))
            ray.maxt = 4.
            result = evaluate(model, ray, [.1, .1, .1], 0., mi, dr)
            np.testing.assert_array_equal(np.asarray(result["hit_count"]), 1)
            values.append(float(np.asarray(result["raw_alpha"])[0]))
        np.testing.assert_allclose(values, values[0], atol=2e-7, rtol=2e-6)
        return result, values[0]

    result, before = directional_results()
    optimizer.step(model, result, epoch=0, coarse_to_fine=True)
    _, after = directional_results()
    assert before != after, "Opacity regression did not exercise an optimizer update"
    assert int(model.opacity_sh_count) == optimizer.dimensions["opacity_sh"] == 1
    assert np.asarray(model.snapshot()["opacity_sh"]).size == int(model.active_voxel_count)
    assert optimizer.state_dict()["groups"]["opacity_sh"]["mean"].size == 1
    return {"dc_opacity_direction_independent_before_and_after_adam": "passed"}


def check_rgb_training_and_refinement(config, payload, view, temporary, mi, dr, np):
    rgba = np.empty((8, 8, 4), np.float32)
    rgba[..., :3] = [.25, .5, .75]
    rgba[..., 3] = np.resize(np.array([0., .5, 1.], np.float32), (8, 8))
    reference_path = Path(temporary) / "reference-rgba.exr"
    mi.Bitmap(rgba).write(str(reference_path))
    rgb = read_reference(reference_path, 8, 8, mi, np, rgb_only=True)
    assert rgb.shape == (8, 8, 3)
    np.testing.assert_allclose(rgb, rgba[..., :3] * rgba[..., 3:4] + 1 - rgba[..., 3:4], atol=2e-6)
    rgb_path = Path(temporary) / "reference-rgb.exr"
    mi.Bitmap(rgb).write(str(rgb_path))
    np.testing.assert_allclose(read_reference(rgb_path, 8, 8, mi, np, rgb_only=True), rgb, atol=2e-6)
    cache = GPUReferenceCache(8, 8, mi, dr, np, capacity_mib=1, rgb_only=True)
    cached = cache.get(reference_path, rgb)
    assert cache.resident_bytes == rgb.nbytes
    outputs = []
    for chunk, reference in ((0, None), (13, cached)):
        current = copy.deepcopy(config)
        current.training.chunk_size = chunk
        model = restore_model(payload, mi)
        sampler = UniformSampler(np)
        output = train_view(model, view, rgb, current, sampler, np.random.default_rng(29),
                            mi, dr, np, gpu_reference=reference)
        assert math.isfinite(output[1]) and output[1] > 0
        assert output[2] is None and sampler.state_dict() == {}
        outputs.append(output)
    compare(outputs[0][0], outputs[1][0], np)
    np.testing.assert_allclose(outputs[0][1], outputs[1][1], atol=2e-6, rtol=3e-5)
    assert not config.topology.enable_growth and not config.topology.enable_deletion
    assert not config.topology.use_growth_evidence and not config.schedule.enable_coarse_growth
    optimizer = FalcorOptimizer(model, config.optimizer)
    optimizer.step(model, outputs[1][0], epoch=0, coarse_to_fine=True)
    manager = TopologyManager(model, 1, config.topology, config.schedule)
    assert not manager.collect_deletion(0) and not manager.needs_rendered_alpha(0)
    replacement, event = manager.finish_epoch(model, optimizer, 1)
    assert event["refined"] and event["grown"] == event["deleted"] == 0
    assert replacement["model_metadata"]["resolution"] == 2
    refined = restore_model(replacement, mi)
    assert 1 < int(refined.active_voxel_count) <= 8
    assert int(refined.opacity_sh_count) == 1
    assert np.asarray(refined.snapshot()["opacity_sh"]).size == int(refined.active_voxel_count)
    following, event = manager.finish_epoch(refined, optimizer, 2)
    assert following is None and event["finished"]
    assert event["grown"] == event["deleted"] == 0
    return {"rgba_composited_once_and_rgb_file_supported": "passed",
            "rgb_full_frame_vs_chunked_device_cache": "passed",
            "no_alpha_history_or_image_readback": "passed",
            "growth_deletion_off_but_resolution_1_to_2_refines": "passed"}


def check_negative_loss(config, view, mi, dr, np):
    # Exercise the post-reduction guard directly with a finite corrupt SUM.
    partial = {"gradients": {}}, mi.Float(-1.), mi.Bool(True), None, None
    try:
        finalize_view(partial, view, config, UniformSampler(np), None, mi, dr, np)
    except RuntimeError as error:
        assert "Negative squared-error loss" in str(error), str(error)
    else:
        raise AssertionError("A finite negative RGB squared-error loss was accepted")
    return {"finite_negative_reduced_loss_rejected": "passed"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", default=RuntimeConfig().build_dir)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    checks = []
    with tempfile.TemporaryDirectory(prefix="mitsuba-rgb-only-regression-") as temporary:
        mi, dr, np, handles = initialize_runtime(RuntimeConfig(args.build_dir, args.gpu),
                                                  Path(temporary), {})
        assert int(getattr(mi.VoxelReconstruction, "rgb_only_version", 0)) >= 1, \
            "Rebuild this checkout before running the RGB-only regression"
        config, payload, view = fixture(np)
        validate(config)
        for check, arguments in (
            (check_reference_alpha, (payload, mi, dr, np)),
            (check_appearance_finite_difference, (payload, mi, dr, np)),
            (check_geometry_signs, (payload, mi, dr, np)),
            (check_dc_opacity, (config, payload, mi, dr, np)),
            (check_rgb_training_and_refinement, (config, payload, view, temporary, mi, dr, np)),
            (check_negative_loss, (config, view, mi, dr, np)),
        ):
            item = check(*arguments)
            checks.append(item)
            print(json.dumps(item), flush=True)
    print(json.dumps({"status": "passed", "checks": checks}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
