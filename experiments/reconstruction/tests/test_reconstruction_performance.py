"""Generated CUDA regression for full-frame training, transfers and JIT reuse.

Uses generated single/multiple-voxel scenes and 16x16 rays. No dataset or experiment is opened;
temporary runtime artifacts are removed automatically. Source build required.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from reconstruction_common import (
    ContourSampler, generate_ray_batch, initialize_runtime, prepare_camera,
    render_view, restore_model,
)
from reconstruction_optimizer import FalcorOptimizer
from preview import RuntimeConfig
from train import TrainConfig, kernel_profile, train_view, validate


class TransferRecorder:
    """Record explicit device-array transfers made through NumPy conversion."""

    def __init__(self, np, dr):
        self.np, self.dr = np, dr
        self.widths = []

    def __getattr__(self, name):
        return getattr(self.np, name)

    def _record(self, value):
        if self.dr.is_array_v(type(value)):
            self.widths.append(self.dr.width(value))

    def asarray(self, value, *args, **kwargs):
        self._record(value)
        return self.np.asarray(value, *args, **kwargs)

    def array(self, value, *args, **kwargs):
        self._record(value)
        return self.np.array(value, *args, **kwargs)


def fixture(np):
    config = TrainConfig()
    config.render.width = config.render.height = 16
    config.training.chunk_size = 0
    config.optimizer.opacity_warmup_epochs = 0
    config.optimizer.opacity_ramp_epochs = 0
    y00 = 1 / (2 * math.sqrt(math.pi))
    state = {
        "cell_indices": np.array([0], np.uint32),
        "centers": np.array([.48, .52, .50], np.float32),
        "log_scales": np.log(np.array([.36, .28, .31], np.float32)),
        "rotations": np.array([1., 0., 0., 0.], np.float32),
        "radiance_sh": np.linspace(-.015, .015, 27, dtype=np.float32),
        "opacity_sh": np.linspace(-.01, .01, 9, dtype=np.float32),
    }
    state["radiance_sh"][:3] = np.array([.2, .3, .4], np.float32) / y00
    state["opacity_sh"][0] = .55 / y00
    payload = {
        "model_state": state,
        "model_metadata": {"resolution": 1, "grid_min": [0., 0., 0.],
                           "grid_max": [1., 1., 1.], "sh_count": 9,
                           "opacity_sh_count": 9},
    }
    camera = {"position": [.5, .5, -1.5], "forward": [0., 0., 1.],
              "right": [1., 0., 0.], "up": [0., 1., 0.],
              "camera_angle_x": .55}
    rgba = np.zeros((16, 16, 4), dtype=np.float32)
    rgba[..., :3] = [.6, .25, .1]
    rgba[3:13, 3:13, 3] = 1.
    rgba[3:13, 3, 3] = .4
    rgba[3:13, 12, 3] = .7
    return config, payload, {"index": 0, "camera": camera}, rgba


def compare_accumulators(first, second, np):
    assert first.keys() == second.keys()
    maximum = 0.
    for group in first:
        pairs = (first[group].items() if group == "gradients"
                 else [(group, first[group])])
        for name, value in pairs:
            other = second[group][name] if group == "gradients" else second[group]
            a, b = np.asarray(value), np.asarray(other)
            maximum = max(maximum, float(np.max(np.abs(a.astype(float) - b))))
            if group == "gradients":
                # GPU atomic reductions need not sum rays in the same order.
                np.testing.assert_allclose(a, b, atol=2e-6, rtol=2e-5,
                                           err_msg=name)
            else:
                np.testing.assert_array_equal(a, b, err_msg=name)
    return maximum


def check_full_frame(config, payload, view, rgba, mi, dr, np):
    results, models, samplers, recorders = [], [], [], []
    for chunk in (0, 37):
        settings = copy.deepcopy(config)
        settings.training.chunk_size = chunk
        model = restore_model(payload, mi)
        sampler = ContourSampler(np, enabled=True)
        sampler.sample(view["index"], rgba[..., 3], np.random.default_rng(9))
        # Exercise a genuinely nonuniform proposal, not just its uniform first view.
        sampler.views["0"]["mean_alpha"][:] = np.linspace(.1, .9, 16)
        recorder = TransferRecorder(np, dr)
        results.append(train_view(model, view, rgba, settings, sampler,
                                  np.random.default_rng(42), mi, dr, recorder,
                                  need_rendered_alpha=True))
        models.append(model)
        samplers.append(sampler)
        recorders.append(recorder)
    maximum = compare_accumulators(results[0][0], results[1][0], np)
    np.testing.assert_allclose(results[0][1], results[1][1], atol=2e-7, rtol=2e-6)
    np.testing.assert_allclose(results[0][2], results[1][2], atol=2e-7, rtol=2e-6)
    np.testing.assert_allclose(samplers[0].views["0"]["mean_alpha"],
                               samplers[1].views["0"]["mean_alpha"], atol=2e-7)
    assert recorders[0].widths.count(256) == 2, recorders[0].widths
    assert np.any(np.asarray(results[0][0]["appearance_valid"]) > 0)
    assert np.any(np.asarray(results[0][0]["geometry_valid"]) > 0)
    optimizers = [FalcorOptimizer(model, copy.deepcopy(config.optimizer))
                  for model in models]
    for optimizer, model, result in zip(optimizers, models, results):
        optimizer.step(model, result[0])
    for name, value in models[0].snapshot().items():
        np.testing.assert_allclose(np.asarray(value), np.asarray(models[1].snapshot()[name]),
                                   atol=2e-6, rtol=2e-6, err_msg=name)
    for group, state in optimizers[0].state_dict()["groups"].items():
        for name, value in state.items():
            np.testing.assert_allclose(value, optimizers[1].state_dict()["groups"][group][name],
                                       atol=2e-6, rtol=3e-5,
                                       err_msg=f"Adam {group}.{name}")
    return {"full_frame_vs_37_ray_chunks": "passed", "max_gradient_absolute_error": maximum,
            "contour_history_and_one_adam_step": "passed"}


def check_transfers(config, payload, view, rgba, mi, dr, np):
    model = restore_model(payload, mi)
    results, widths = [], []
    for need_alpha in (False, True):
        recorder = TransferRecorder(np, dr)
        result = train_view(model, view, rgba, config, ContourSampler(np),
                            np.random.default_rng(71), mi, dr, recorder,
                            need_rendered_alpha=need_alpha)
        results.append(result)
        widths.append(recorder.widths)
    assert results[0][2] is None
    assert results[1][2].shape == rgba.shape[:2]
    assert 256 not in widths[0], widths[0]
    assert widths[1].count(256) == 1, widths[1]
    compare_accumulators(results[0][0], results[1][0], np)
    np.testing.assert_allclose(results[0][1], results[1][1], atol=2e-7, rtol=2e-6)
    device_recorder = TransferRecorder(np, dr)
    device_result = train_view(model, view, rgba, config, ContourSampler(np),
        np.random.default_rng(71), mi, dr, device_recorder,
        need_rendered_alpha=True, alpha_on_device=True)
    assert device_recorder.widths == [2], device_recorder.widths
    assert dr.is_array_v(type(device_result[2]))
    np.testing.assert_array_equal(np.asarray(device_result[2]).reshape(rgba.shape[:2]), results[1][2])
    compare_accumulators(device_result[0], results[1][0], np)
    recorder = TransferRecorder(np, dr)
    sampler = ContourSampler(np, enabled=True)
    result = train_view(model, view, rgba, config, sampler,
                        np.random.default_rng(71), mi, dr, recorder)
    assert result[2] is None
    assert recorder.widths.count(256) == 1, recorder.widths
    return {"image_readbacks_contour_off": widths,
            "device_topology_readbacks": device_recorder.widths,
            "contour_only_image_readbacks": recorder.widths}


def check_unused_deletion(config, payload, view, rgba, mi, dr, np):
    model = restore_model(payload, mi)
    results = [train_view(model, view, rgba, config, ContourSampler(np),
                          np.random.default_rng(71), mi, dr, np,
                          need_rendered_alpha=True, collect_deletion_evidence=enabled)
               for enabled in (True, False)]
    evidence = ("foreground_evidence", "background_evidence")
    assert any(np.any(np.asarray(results[0][0][key])) for key in evidence)
    assert all(not np.any(np.asarray(results[1][0][key])) for key in evidence)
    compare_accumulators(*[{key: value for key, value in result[0].items() if key not in evidence}
                           for result in results], np)
    np.testing.assert_allclose(results[0][1], results[1][1], atol=2e-7, rtol=2e-6)
    np.testing.assert_array_equal(results[0][2], results[1][2])
    models = [restore_model(payload, mi) for _ in results]
    for updated, result in zip(models, results):
        FalcorOptimizer(updated, copy.deepcopy(config.optimizer)).step(updated, result[0])
    for name, value in models[0].snapshot().items():
        np.testing.assert_allclose(np.asarray(value), np.asarray(models[1].snapshot()[name]),
                                   atol=2e-6, rtol=2e-6, err_msg=name)
    return {"unused_deletion_evidence_skipped_without_changing_gradients_or_adam": "passed"}


def check_native_stage_profile(config, payload, view, rgba, mi, dr, np):
    phases = {"forward", "hit_backward", "candidate_backward"}
    fields = {"wall_seconds", "device_ms", "codegen_ms", "backend_compile_ms",
              "operations", "jit_kernels", "cache_hits", "cache_misses"}
    model = restore_model(payload, mi)
    reports = []
    for chunk in (0, 37):
        current = copy.deepcopy(config)
        current.training.chunk_size = chunk
        expected = train_view(model, view, rgba, current, ContourSampler(np),
                              np.random.default_rng(118), mi, dr, np)
        measurements = {}
        recorder = TransferRecorder(np, dr)
        started = time.perf_counter()
        with dr.kernel_history() as history:
            actual = train_view(model, view, rgba, current, ContourSampler(np),
                                np.random.default_rng(118), mi, dr, recorder,
                                stage_profile=measurements)
        elapsed = time.perf_counter() - started
        compare_accumulators(actual[0], expected[0], np)
        np.testing.assert_allclose(actual[1], expected[1], atol=2e-7, rtol=2e-6)
        assert actual[2] is None and recorder.widths == [2], recorder.widths
        assert set(measurements) == phases, measurements
        for phase, measured in measurements.items():
            assert set(measured) == fields, (phase, measured)
            assert all(math.isfinite(value) and value >= 0 for value in measured.values())
            assert measured["cache_hits"] + measured["cache_misses"] == measured["jit_kernels"]
        assert sum(stage["wall_seconds"] for stage in measurements.values()) <= elapsed + .001
        outer = kernel_profile(history)
        # Native nested captures must leave the Python view-level capture intact.
        for name in ("operations", "jit_kernels", "device_ms", "codegen_ms", "backend_compile_ms"):
            assert sum(stage[name] for stage in measurements.values()) <= outer[name] + 1e-3, (name, measurements, outer)
        assert measurements["forward"]["jit_kernels"] > 0
        reports.append({"chunk_size": chunk, "native_stages": measurements})
    empty = copy.deepcopy(payload)
    empty["model_state"] = {key: value[:0] for key, value in empty["model_state"].items()}
    result = restore_model(empty, mi).loss_and_grad(
        mi.Ray3f(mi.Point3f(.5, .5, -1), mi.Vector3f(0, 0, 1)),
        mi.Color3f(.6), mi.Float(.8), mi.Color3f(1), mi.Float(1), {"profile_stages": 1})
    assert set(result["stage_profile"]) == phases
    for name in ("hit_backward", "candidate_backward"):
        assert all(value == 0 for value in result["stage_profile"][name].values())
    return {"native_stage_profile_preserves_gradients_counts_loss_and_transfers": "passed",
            "nested_kernel_history_preserved": True, "empty_model_profile": "passed",
            "profiled_chunk_modes": [report["chunk_size"] for report in reports]}


def check_multiple_hit_layout(config, payload, view, rgba, mi, dr, np):
    payload = copy.deepcopy(payload)
    payload["model_metadata"]["resolution"] = 8
    count = 8 ** 3
    state = payload["model_state"]
    for key, value in state.items():
        state[key] = (np.arange(count, dtype=np.uint32) if key == "cell_indices"
                      else np.tile(value, count))
    state["log_scales"] = np.tile(np.log(np.array([.060, .048, .066], np.float32)), count)
    state["opacity_sh"].reshape(count, -1)[:, 0] = math.log(.15 / .85) * (2 * math.sqrt(math.pi))
    model = restore_model(payload, mi)
    ray = generate_ray_batch(config.render, view["camera"], np.arange(256, dtype=np.uint32),
                             np.full((256, 2), .5, np.float32), mi, dr)
    rgb, alpha, hits = model.trace(ray, mi.Color3f(1))
    assert int(dr.max(hits)[0]) > 1, "Fixture must exercise suffix reads behind multiple hits"
    cases = 0
    maximum = 0.
    reverse_kernel_counts = {}
    for max_hits in (1, 5, 16):
        results = []
        for chunk_size in (0, 37):
            current = copy.deepcopy(config)
            current.render.max_hits = max_hits
            current.training.chunk_size = chunk_size
            results.append(train_view(model, view, rgba, current, ContourSampler(np),
                                      np.random.default_rng(194), mi, dr, np,
                                      need_rendered_alpha=True))
        maximum = max(maximum, compare_accumulators(results[0][0], results[1][0], np))
        np.testing.assert_allclose(results[0][1], results[1][1], atol=2e-7, rtol=2e-6)
        np.testing.assert_array_equal(results[0][2], results[1][2])
        # Record the isolated reverse stage, not the candidate or ray-generation
        # kernels. The symbolic reverse loop must not launch one kernel per
        # configured hit slot. Warm each capacity's separately compiled graph.
        options = {"max_hits": float(max_hits), "profile_stages": 1.}
        for _ in range(2):
            measured = model.loss_and_grad(ray, mi.Color3f(.3, .6, .1), mi.Float(.8),
                                           mi.Color3f(1), mi.Float(1), options)
            dr.eval(measured)
            dr.sync_thread()
        count = measured["stage_profile"]["hit_backward"]["jit_kernels"]
        assert count > 0, measured["stage_profile"]
        reverse_kernel_counts[max_hits] = int(count)
        cases += 1
    # The symbolic reverse stage measured one JIT kernel for all three limits.
    # Permit one extra fixed setup kernel, while rejecting the old host-unrolled
    # implementation that dispatched a separate kernel for each hit slot.
    assert all(count <= 2 for count in reverse_kernel_counts.values()), reverse_kernel_counts
    return {"multiple_hit_suffix_layout_full_frame_vs_chunks": cases,
            "maximum_gradient_absolute_error": maximum,
            "hit_backward_jit_kernel_counts_by_max_hits": reverse_kernel_counts}


def reference_render_view(model, render, camera, spp, seed, chunk_size, mi, dr, np):
    """Previous CPU Float64 accumulation, retained as a numerical oracle."""
    count = render.width * render.height
    rgb_sum = np.zeros((count, 3), dtype=np.float64)
    alpha_sum = np.zeros(count, dtype=np.float64)
    rng = np.random.default_rng(seed)
    prepared = prepare_camera(render, camera, mi, dr)
    for _ in range(spp):
        offsets = rng.random((count, 2)).astype(np.float32)
        np.minimum(offsets, np.nextafter(np.float32(1), np.float32(0)), out=offsets)
        for begin in range(0, count, chunk_size):
            end = min(begin + chunk_size, count)
            ray = generate_ray_batch(render, camera, np.arange(begin, end, dtype=np.uint32),
                                     offsets[begin:end], mi, dr, prepared_camera=prepared)
            rgb, alpha, hits = model.trace(ray, mi.Color3f(render.background),
                render.transmittance_threshold, render.max_hits, render.entry_offset_fraction)
            dr.eval(rgb, alpha, hits)
            dr.sync_thread()
            rgb_sum[begin:end] += np.stack([np.asarray(rgb.x), np.asarray(rgb.y), np.asarray(rgb.z)], axis=-1)
            alpha_sum[begin:end] += np.asarray(alpha)
    return ((rgb_sum / spp).astype(np.float32).reshape(render.height, render.width, 3),
            (alpha_sum / spp).astype(np.float32).reshape(render.height, render.width))


def check_render_accumulation(config, payload, view, mi, dr, np):
    model = restore_model(payload, mi)
    cases = 0
    for spp, seed in ((1, 7), (3, 42), (64, 7)):
        previous = None
        for chunk_size in (37, 256):
            recorder = TransferRecorder(np, dr)
            actual = render_view(model, config.render, view["camera"], spp, seed, chunk_size, mi, dr, recorder)
            expected = reference_render_view(model, config.render, view["camera"], spp, seed, chunk_size, mi, dr, np)
            for a, b in zip(actual, expected):
                np.testing.assert_array_equal(a.view(np.uint32), b.view(np.uint32))
            if previous is not None:
                for a, b in zip(actual, previous):
                    np.testing.assert_array_equal(a, b)
            previous = actual
            assert recorder.widths == [256] * 4, recorder.widths
            cases += 1
    return {"gpu_float64_vs_cpu_accumulation_bitwise_equal": cases,
            "final_image_channel_readbacks": 4}


def literal_ray(render, camera, indices, offsets, mi, dr):
    """The previous camera expression, retained only as a numerical oracle."""
    index = mi.UInt32(indices)
    u = (mi.Float(index % render.width) + mi.Float(offsets[:, 0])) / render.width
    v = (mi.Float(index // render.width) + mi.Float(offsets[:, 1])) / render.height
    tan_x = math.tan(camera["camera_angle_x"] / 2)
    tan_y = tan_x * render.height / render.width
    direction = (mi.Vector3f(camera["forward"])
                 + ((2 * u - 1) * tan_x) * mi.Vector3f(camera["right"])
                 + ((1 - 2 * v) * tan_y) * mi.Vector3f(camera["up"]))
    ray = mi.Ray3f(mi.Point3f(camera["position"]) + render.near_clip * direction,
                  dr.normalize(direction))
    ray.maxt = (render.far_clip - render.near_clip) * dr.norm(direction)
    return ray


def check_camera_cache(config, payload, view, mi, dr, np):
    camera_a = view["camera"]
    angle = .07
    camera_b = {"position": [.42, .57, -1.45],
                "forward": [math.sin(angle), 0., math.cos(angle)],
                "right": [math.cos(angle), 0., -math.sin(angle)],
                "up": [0., 1., 0.], "camera_angle_x": .59}
    indices = np.arange(256, dtype=np.uint32)
    offsets = np.random.default_rng(314).random((256, 2)).astype(np.float32)
    maximum = 0.
    for camera in (camera_a, camera_b):
        actual = generate_ray_batch(config.render, camera, indices, offsets, mi, dr)
        expected = literal_ray(config.render, camera, indices, offsets, mi, dr)
        for name in ("o", "d", "maxt"):
            a, b = np.asarray(getattr(actual, name)), np.asarray(getattr(expected, name))
            maximum = max(maximum, float(np.max(np.abs(a - b))))
            np.testing.assert_allclose(a, b, atol=3e-7, rtol=3e-7,
                                       err_msg=f"camera ray {name}")
    model = restore_model(payload, mi)
    histories, profiles, alphas = [], [], []
    for camera in (camera_a, camera_b):
        prepared = prepare_camera(config.render, camera, mi, dr)
        with dr.kernel_history() as history:
            ray = generate_ray_batch(config.render, camera, indices, offsets, mi, dr,
                                     prepared_camera=prepared)
            result = model.loss_and_grad(ray, mi.Color3f(.6, .3, .1), mi.Float(.8),
                                         mi.Color3f(1), mi.Float(1), {})
            dr.eval(result)
            dr.sync_thread()
        jit = [entry for entry in history if entry.type == dr.KernelType.JIT]
        assert jit, "Native loss_and_grad did not launch a recorded JIT kernel"
        histories.append(jit)
        profile = kernel_profile(history)
        assert profile["jit_kernels"] == len(jit)
        assert profile["cache_hits"] + profile["cache_misses"] == len(jit)
        for name in ("codegen_ms", "backend_compile_ms", "device_ms"):
            assert math.isfinite(profile[name]) and profile[name] >= 0, profile
        profiles.append(profile)
        alphas.append(np.asarray(result["raw_alpha"]).copy())
        del result, ray
    first_hashes, second_hashes = ({entry.hash for entry in entries}
                                   for entries in histories)
    assert first_hashes == second_hashes, (first_hashes, second_hashes)
    assert all(entry.cache_hit for entry in histories[1]), "Changed camera recompiled a JIT kernel"
    assert profiles[1]["cache_misses"] == 0
    assert profiles[1]["backend_compile_ms"] == 0
    assert not np.array_equal(alphas[0], alphas[1]), "Changed camera did not change rendered alpha"
    return {"opaque_camera_vs_literal_max_absolute_error": maximum,
            "shared_native_kernel_hash_count": len(first_hashes),
            "changed_camera_all_cache_hits": True,
            "kernel_profile": "passed"}


def check_nonfinite(config, view, rgba, mi, dr, np):
    rejected = []

    class NonfiniteModel:
        # This fake result tests the driver's finite checks. Match the loaded
        # native traversal capability so the independent build guard is honest.
        octree_version = getattr(mi.VoxelReconstruction, "octree_version", 0)

        def __init__(self, field):
            self.field = field

        def loss_and_grad(self, *args):
            result = {"loss": mi.Float([1.] * 256),
                      "raw_alpha": mi.Float([.5] * 256),
                      "alpha": mi.Float([.5] * 256),
                      "gradients": {"radiance_sh": mi.Float([.1] * 27)}}
            for key in ("appearance_valid", "geometry_valid", "background_carve_valid",
                        "foreground_evidence", "background_evidence"):
                result[key] = mi.UInt32([1])
            if self.field == "gradient":
                result["gradients"]["radiance_sh"] = mi.Float([float("nan")] * 27)
            else:
                result[self.field] = mi.Float([float("nan")] * 256)
            return result

    for field in ("loss", "raw_alpha", "alpha", "gradient"):
        try:
            train_view(NonfiniteModel(field), view, rgba, config, ContourSampler(np),
                       np.random.default_rng(1), mi, dr, np)
        except RuntimeError as exc:
            assert "Nonfinite" in str(exc), str(exc)
            rejected.append(field)
        else:
            raise AssertionError(f"Nonfinite {field} was silently accepted")
    return {"nonfinite_rejected_without_image_readback": rejected}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", default=RuntimeConfig().build_dir)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    checks = []
    with tempfile.TemporaryDirectory(prefix="mitsuba-performance-regression-") as temporary:
        runtime_report = {}
        mi, dr, np, handles = initialize_runtime(RuntimeConfig(args.build_dir, args.gpu),
                                                  Path(temporary), runtime_report)
        config, payload, view, rgba = fixture(np)
        assert TrainConfig().training.chunk_size == 0
        validate(config)
        invalid = copy.deepcopy(config)
        invalid.training.chunk_size = -1
        try:
            validate(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("Negative training chunk size was accepted")
        for check, arguments in (
            (check_full_frame, (config, payload, view, rgba, mi, dr, np)),
            (check_transfers, (config, payload, view, rgba, mi, dr, np)),
            (check_unused_deletion, (config, payload, view, rgba, mi, dr, np)),
            (check_native_stage_profile, (config, payload, view, rgba, mi, dr, np)),
            (check_multiple_hit_layout, (config, payload, view, rgba, mi, dr, np)),
            (check_render_accumulation, (config, payload, view, mi, dr, np)),
            (check_camera_cache, (config, payload, view, mi, dr, np)),
            (check_nonfinite, (config, view, rgba, mi, dr, np)),
        ):
            item = check(*arguments)
            checks.append(item)
            print(json.dumps(item), flush=True)
        print(json.dumps({"status": "passed", "checks": checks}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
