"""Generated CUDA regression for full-frame training, transfers and JIT reuse.

Uses one synthetic voxel and 16x16 rays. No dataset or experiment is opened;
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

sys.path.insert(0, str(Path(__file__).resolve().parent))

from reconstruction_common import (
    ContourSampler, generate_ray_batch, initialize_runtime, prepare_camera,
    restore_model,
)
from reconstruction_optimizer import FalcorOptimizer
from reconstruction_preview import RuntimeConfig
from reconstruction_train import TrainConfig, kernel_profile, train_view, validate


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
    recorder = TransferRecorder(np, dr)
    sampler = ContourSampler(np, enabled=True)
    result = train_view(model, view, rgba, config, sampler,
                        np.random.default_rng(71), mi, dr, recorder)
    assert result[2] is None
    assert recorder.widths.count(256) == 1, recorder.widths
    return {"image_readbacks_contour_off": widths,
            "contour_only_image_readbacks": recorder.widths}


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
