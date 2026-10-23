"""Generated CUDA equivalence and failure-transaction tests; no dataset required."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from reconstruction_checks import DeferredViewChecks
from reconstruction_optimizer import FalcorOptimizer
from reconstruction_common import ContourSampler, initialize_runtime, restore_model
from reconstruction_reference import GPUReferenceCache
from reconstruction_replay import NativeLossReplay
from preview import RuntimeConfig
from train import train_view
from test_reconstruction_performance import fixture, TransferRecorder
from reconstruction_checkpoint import load_checkpoint


def arrays(model, optimizer, np):
    result = {name: np.asarray(value).copy() for name, value in model.snapshot().items()}
    result.update({f"{group}/{key}": np.asarray(value).copy()
                   for group, state in optimizer.state.items() for key, value in state.items()})
    return result


def assert_same(first, second, np, *, atol=1e-7, rtol=1e-6):
    assert first.keys() == second.keys()
    for key in first:
        np.testing.assert_allclose(first[key], second[key], atol=atol, rtol=rtol, err_msg=key)


def check_equivalence(config, payload, view, rgba, mi, dr, np):
    for replay_enabled in (False, True):
        for distortion_weight in (0., .1):
            current = copy.deepcopy(config)
            current.gradient.distortion_loss_weight = distortion_weight
            # Exercise both the legacy RGBA path and the production RGB-only
            # path with both native forward and optimizer replay enabled.
            current.training.rgb_only = replay_enabled
            reference = (np.ascontiguousarray(rgba[..., :3] * rgba[..., 3:4] +
                                              (1. - rgba[..., 3:4]))
                         if current.training.rgb_only else rgba)
            models = [restore_model(payload, mi) for _ in range(2)]
            optimizers = [FalcorOptimizer(model, copy.deepcopy(current.optimizer),
                                         sparse_updates=index == 1,
                                         replay_updates=replay_enabled)
                          for index, model in enumerate(models)]
            rngs = [np.random.default_rng(45) for _ in models]
            cache = GPUReferenceCache(current.render.width, current.render.height, mi, dr, np,
                                      capacity_mib=1, rgb_only=current.training.rgb_only)
            gpu_reference = cache.get("generated-deferred-reference.png", reference)
            recorder = TransferRecorder(np, dr)
            checks = DeferredViewChecks(mi, dr, recorder)
            expected = []
            # Complete the dense reference sequence first. Interleaving it
            # with device steps would let its readbacks flush pending work and
            # hide dirty parameter handles in the actual asynchronous chain.
            for index in range(4):
                current_view = {**view, "index": index}
                parts = {}
                out, loss, _ = train_view(models[0], current_view, reference, current, ContourSampler(np),
                    rngs[0], mi, dr, np, loss_components=parts, gpu_reference=gpu_reference)
                optimizers[0].step(models[0], out)
                expected.append(dict(loss=loss, **parts))
            native_replay = NativeLossReplay(dr, enabled=replay_enabled)
            for index in range(4):
                current_view = {**view, "index": index}
                out, _, _ = train_view(models[1], current_view, reference, current, ContourSampler(np),
                    rngs[1], mi, dr, recorder, deferred_checks=checks,
                    gpu_reference=gpu_reference, native_replay=native_replay)
                update = optimizers[1].step(models[1], out, update_allowed=out["_update_allowed"])
                checks.updated(update["accepted"])
                # No snapshot, scalar readback, explicit sync, or reference
                # sequence work between step() and the next view's forward.
            assert recorder.widths == [], recorder.widths
            actual = checks.read(rgb_only=current.training.rgb_only, distortion_weight=distortion_weight)
            assert recorder.widths == [20], recorder.widths
            for got, want in zip(actual, expected):
                for key in want:
                    if want[key] is None:
                        assert got[key] is None
                    else:
                        np.testing.assert_allclose(got[key], want[key], atol=3e-7, rtol=3e-5)
            assert_same(arrays(models[0], optimizers[0], np), arrays(models[1], optimizers[1], np),
                        np, atol=3e-7, rtol=3e-5)
            assert rngs[0].bit_generator.state == rngs[1].bit_generator.state
            if replay_enabled:
                assert native_replay.statistics()["recordings"] > 0
                assert native_replay.statistics()["replays"] > 0
                assert optimizers[1].statistics()["recordings"] > 0
                assert optimizers[1].statistics()["replays"] > 0


def check_failure(config, payload, view, rgba, mi, dr, np):
    config.gradient.distortion_loss_weight = 0.
    for field in ("loss", "raw_alpha", "alpha", "gradient", "negative_loss", "injected_rejection", "moment"):
        model = restore_model(payload, mi)
        # This Python fault-injection proxy is not a traversable native model.
        # Frozen transaction/latch behavior is covered by the sparse Adam suite.
        optimizer = FalcorOptimizer(model, copy.deepcopy(config.optimizer), replay_updates=False)
        if field == "moment":
            optimizer.config.gradient_clip = 1e30
        original = arrays(model, optimizer, np)
        recorder = TransferRecorder(np, dr)
        checks = DeferredViewChecks(mi, dr, recorder)

        class BrokenModel:
            def __getattr__(self, name):
                return getattr(model, name)

            def loss_and_grad(self, *args):
                result = model.loss_and_grad(*args)
                if field == "gradient":
                    result["gradients"]["radiance_sh"] *= mi.Float(float("nan"))
                elif field == "moment":
                    # Finite input passes DeferredViewChecks but overflows the
                    # Adam second moment, so the update must reject atomically.
                    result["gradients"]["radiance_sh"] = dr.full(
                        mi.Float, 1e30, dr.width(result["gradients"]["radiance_sh"]))
                elif field == "negative_loss":
                    result["loss"] = -dr.abs(result["loss"])-1.
                elif field in ("loss", "raw_alpha", "alpha"):
                    result[field] *= mi.Float(float("nan"))
                return result

            def optimizer_step_device(self, gradients, appearance, geometry, carving,
                                      means, squares, steps, settings, allowed, check_moments=True):
                # Inject a rejected commit, not a claimed internal proposal NaN.
                # Separate native transaction tests exercise invalid proposals
                # through set_parameters_guarded/set_parameters_sparse directly.
                if field == "injected_rejection":
                    allowed &= mi.Bool(False)
                return model.optimizer_step_device(gradients, appearance, geometry, carving,
                    means, squares, steps, settings, allowed, check_moments)

        wrapper = BrokenModel()
        out, _, _ = train_view(wrapper, view, rgba, config, ContourSampler(np),
            np.random.default_rng(91), mi, dr, recorder, deferred_checks=checks)
        result = optimizer.step(wrapper, out, update_allowed=out["_update_allowed"])
        checks.updated(result["accepted"])
        # Even after subsequent clean gradients, the device error latch prevents
        # updating any model parameter, moment, or per-voxel Adam step count.
        out, _, _ = train_view(model, {**view, "index": 1}, rgba, config, ContourSampler(np),
            np.random.default_rng(92), mi, dr, recorder, deferred_checks=checks)
        result = optimizer.step(model, out, update_allowed=out["_update_allowed"])
        checks.updated(result["accepted"])
        assert recorder.widths == [], (field, recorder.widths)
        assert_same(original, arrays(model, optimizer, np), np)
        try:
            checks.read(rgb_only=False, distortion_weight=0.)
        except RuntimeError as error:
            assert "view 0" in str(error), (field, error)
        else:
            raise AssertionError(f"Deferred {field} failure was not reported")


def check_driver(directory, build_dir, gpu, mi, np):
    """The actual driver must report floats/save identical final states."""
    directory = Path(directory)
    dataset = directory / "dataset"
    dataset.mkdir()
    (dataset / "init_points.ply").write_text(
        "ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n0 0 0 100 140 180\n",
        encoding="ascii")
    pixels = np.zeros((16, 16, 4), dtype=np.float32)
    pixels[4:12, 4:12] = [.6, .2, .1, 1.]
    frames = []
    for index in range(2):
        path = dataset / f"view-{index}.png"
        mi.Bitmap(pixels).convert(mi.Bitmap.PixelFormat.RGBA, mi.Struct.Type.UInt8, True).write(str(path))
        frames.append(dict(file_path=f"view-{index}.png", transform_matrix=
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 3], [0, 0, 0, 1]]))
    (dataset / "transforms_train.json").write_text(json.dumps(dict(camera_angle_x=.7, frames=frames)), encoding="utf-8")
    config = dict(data=dict(dataset=str(dataset)), model=dict(grid_resolution=4,
        bounds_min=[-1.]*3, bounds_max=[1.]*3), render=dict(width=16, height=16),
        runtime=dict(build_dir=build_dir, gpu=gpu), training=dict(num_epochs=2,
            save_count=1, profile_views=1, profile_stages=False, profile_interval_epochs=0),
        schedule=dict(enabled=False), topology=dict(enable_growth=False, enable_deletion=False),
        evaluation=dict(enabled=False))
    config_path = directory / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    final = []
    for sync in (True, False):
        output = directory / ("sync" if sync else "deferred")
        command = [sys.executable, "-I", str(Path(__file__).resolve().parents[1] / "train.py"),
                   "--config", str(config_path), "--output-dir", str(output),
                   "--sync-view-checks" if sync else "--no-sync-view-checks"]
        run = subprocess.run(command, text=True, capture_output=True)
        if run.returncode:
            raise RuntimeError(run.stdout + run.stderr)
        history = [json.loads(line) for line in (output / "training.jsonl").read_text().splitlines()]
        assert len(history) == 2 and history[-1]["global_step"] == 4
        assert all(row["epoch_validation_seconds"] >= 0 for row in history)
        report = json.loads((output / "training-report.json").read_text())
        assert report["training_status"] == "passed"
        assert report["training_protocol"]["sync_view_checks"] == sync
        final.append(load_checkpoint(output / "model.npz"))
    for group in ("model_state",):
        assert_same(final[0][group], final[1][group], np)
    for group in final[0]["optimizer_state"]["groups"]:
        assert_same(final[0]["optimizer_state"]["groups"][group], final[1]["optimizer_state"]["groups"][group], np)
    assert final[0]["training_state"]["rng_state"] == final[1]["training_state"]["rng_state"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", default=RuntimeConfig().build_dir)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="mitsuba-deferred-checks-") as temporary:
        mi, dr, np, handles = initialize_runtime(RuntimeConfig(args.build_dir, args.gpu), Path(temporary), {})
        assert callable(getattr(mi.VoxelReconstruction, "optimizer_step_device", None)), "Rebuild native source first"
        config, payload, view, rgba = fixture(np)
        check_equivalence(config, payload, view, rgba, mi, dr, np)
        check_failure(config, payload, view, rgba, mi, dr, np)
        check_driver(temporary, args.build_dir, args.gpu, mi, np)
        print("passed: deferred loss/RNG/Adam equivalence; zero per-view NumPy loss/image readbacks; "
              "device transaction/latch failures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
