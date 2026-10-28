"""Small CUDA regressions for native fixed-capacity gradient storage.

Dense native gradients are the oracle. The fixtures exercise hit and near-miss
rows, zero gradients, projective SH9, topology evidence, overflow, and frozen
replays with a changing number of touched voxels. No real dataset is loaded.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from preview import RuntimeConfig
from reconstruction_common import initialize_runtime, restore_model, generate_ray_batch
from reconstruction_optimizer import FalcorOptimizer, OptimizerConfig
from reconstruction_replay import NativeLossReplay
from reconstruction_gradient_storage import native_gradient_result
from test_reconstruction_replay import fixture as replay_fixture
from test_reconstruction_projective_native import fixture as projective_fixture
import test_reconstruction_compact_optimizer as optimizer_tests
import train


VALIDITY = ("appearance_valid", "geometry_valid", "background_carve_valid")


def capture(out):
    dr.eval(*out["gradients"].values(),
            *(value for name, value in out.items() if name != "gradients"))
    return {name: ({group: np.array(value) for group, value in values.items()}
                   if name == "gradients" else np.array(values))
            for name, values in out.items()}


def bounded_fixture():
    config, payload, camera, rgba = replay_fixture(np)
    # Add corner rows so camera changes exercise different touched-voxel sets.
    state = payload["model_state"]
    old_count = len(state["cell_indices"])
    extra = np.array([0, 3, 12, 15, 48, 51, 60, 63], np.uint32)
    for name, value in list(state.items()):
        if name == "cell_indices":
            state[name] = np.concatenate((value, extra))
        else:
            rows = value.reshape(old_count, -1)
            state[name] = np.concatenate((rows, np.repeat(rows[:1], len(extra), axis=0))).ravel()
    offsets = np.random.default_rng(194).random((256, 2)).astype(np.float32)

    def inputs(shift=0.):
        current = copy.deepcopy(camera)
        current["position"][0] += shift
        ray = generate_ray_batch(config.render, current,
            np.arange(256, dtype=np.uint32), offsets, mi, dr)
        ref = rgba.reshape(-1, 4)
        return (ray, mi.Color3f(*(mi.Float(ref[:, j]) for j in range(3))),
                mi.Float(ref[:, 3]), mi.Color3f(1.), dr.opaque(mi.Float, 1.))

    return payload, inputs


def small_fixture(*, empty=False):
    count = 0 if empty else 2
    y00 = 1 / (2 * np.sqrt(np.pi))
    state = {
        "cell_indices": np.array([384, 63][:count], np.uint32),
        "centers": np.tile([.5, .5, .5], count).astype(np.float32),
        "log_scales": np.full(count * 3, np.log(.25), np.float32),
        "rotations": np.tile([1., 0., 0., 0.], count).astype(np.float32),
        "radiance_sh": np.zeros(count * 27, np.float32),
        "opacity_sh": np.zeros(count, np.float32),
    }
    state["radiance_sh"].reshape(count, 27)[:, :3] = np.array([.2, .3, .4]) / y00
    payload = dict(model_state=state, model_metadata=dict(resolution=8,
        grid_min=[0., 0., 0.], grid_max=[8., 8., 8.], sh_count=9, opacity_sh_count=1))
    return restore_model(payload, mi)


def small_inputs(x=.78):
    ray = mi.Ray3f(mi.Point3f(x, .5, -1.), mi.Vector3f(0., 0., 1.))
    ray.maxt = 20.
    return ray, mi.Color3f(.1), mi.Float(1.), mi.Color3f(1.), mi.Float(1.)


class CompactGradientTests(unittest.TestCase):
    def compare(self, dense, compact, count):
        dense, compact = capture(dense), capture(compact)
        required = int(compact["voxel_count"][0])
        ids = compact["voxel_ids"][:required]
        self.assertFalse(bool(compact["compact_gradient_overflow"][0]))
        self.assertTrue(np.all(ids < count))
        self.assertTrue(np.all(ids[1:] > ids[:-1]))
        np.testing.assert_array_equal(compact["voxel_ids"][required:], np.uint32(0xffffffff))
        for name, expected in dense.items():
            if name == "gradients":
                for group, value in expected.items():
                    width = len(value) // count if count else 0
                    actual = np.zeros_like(value).reshape(count, width)
                    if count:
                        actual[ids] = compact[name][group].reshape(-1, width)[:required]
                    np.testing.assert_allclose(actual.ravel(), value, atol=3e-6, rtol=4e-5,
                                               err_msg=group)
            elif name in VALIDITY:
                actual = np.zeros(count, np.uint32)
                actual[ids] = compact[name][:required]
                np.testing.assert_array_equal(actual, expected, err_msg=name)
            elif expected.dtype.kind in "iub":
                np.testing.assert_array_equal(compact[name], expected, err_msg=name)
            else:
                np.testing.assert_allclose(compact[name], expected, atol=3e-6, rtol=4e-5,
                                           err_msg=name)
        return compact

    def test_bounded_sh9_distortion_and_both_topology_evidence_modes(self):
        payload, inputs = bounded_fixture()
        model = restore_model(payload, mi)
        count = int(model.active_voxel_count)
        growth = {name: mi.Float(value.reshape(count, -1)[0].copy())
                  for name, value in payload["model_state"].items() if name != "cell_indices"}
        model.set_growth_candidates(dict(growth, cell_indices=mi.UInt32([17])))
        for rgb_only in (False, True):
            with self.subTest(rgb_only=rgb_only):
                options = dict(rgb_only=float(rgb_only), distortion_loss_weight=.1,
                    max_hits=8., max_candidates=8., collect_deletion_evidence=float(not rgb_only),
                    collect_rgb_deletion_evidence=float(rgb_only), collect_growth_evidence=float(rgb_only))
                dense = model.loss_and_grad(*inputs(), options)
                compact = model.loss_and_grad(*inputs(), options | {"compact_gradient_capacity": count})
                actual = self.compare(dense, compact, count)
                self.assertGreater(int(actual["voxel_count"][0]), 1)
                self.assertLessEqual(int(actual["voxel_count"][0]), count)
                self.assertEqual(len(actual["foreground_evidence"]), count)
                if rgb_only:
                    self.assertEqual(len(actual["deletion_gain"]), count)
                    self.assertEqual(len(actual["growth_gain"]), 1)

    def test_near_miss_only_and_valid_zero_gradient_rows_are_retained(self):
        model = small_fixture()
        options = dict(rgb_only=1., max_candidates=1., compact_gradient_capacity=2.)
        dense = model.loss_and_grad(*small_inputs(), {k: v for k, v in options.items()
                                                    if k != "compact_gradient_capacity"})
        result = model.loss_and_grad(*small_inputs(), options)
        out = self.compare(dense, result, 2)
        self.assertEqual(int(out["voxel_count"][0]), 1)
        self.assertEqual(int(out["appearance_valid"][0]), 0)
        self.assertGreater(int(out["geometry_valid"][0]), 0)
        self.assertGreater(float(np.linalg.norm(out["gradients"]["centers"])), 0.)

        arguments = list(small_inputs(.62))
        baseline = model.loss_and_grad(*arguments, {"rgb_only": 1.})
        arguments[1] = baseline["rgb"]
        result = model.loss_and_grad(*arguments, options)
        self.assertEqual(int(np.array(result["voxel_count"])[0]), 1)
        self.assertGreater(int(np.array(result["appearance_valid"])[0]), 0)
        for value in result["gradients"].values():
            np.testing.assert_array_equal(np.array(value), 0.)

    def test_projective_full_sh9_core_and_background_match_dense(self):
        payload = projective_fixture(np)
        count = len(payload["model_state"]["cell_indices"])
        old_dc = payload["model_state"]["radiance_sh"].reshape(count, 3)
        sh = np.random.default_rng(43).uniform(-.03, .03, (count, 27)).astype(np.float32)
        sh[:, :3] = old_dc
        payload["model_metadata"]["sh_count"] = 9
        payload["model_state"]["radiance_sh"] = sh.ravel()
        model = restore_model(payload, mi)
        origins = np.array([[.17, .12, -3.], [2., .3, .1], [.1, .2, .3],
                            [0., 0., 0.], [-4., .1, .2]], np.float32)
        directions = np.array([[0., 0., 1.], [-1., .02, -.01], [.3, .4, .5],
                               [1., 0., 0.], [1., .03, -.02]], np.float32)
        ray = mi.Ray3f(mi.Point3f(*(mi.Float(origins[:, j]) for j in range(3))),
                      dr.normalize(mi.Vector3f(*(mi.Float(directions[:, j]) for j in range(3)))))
        arguments = ray, mi.Color3f(.4), mi.Float(1.), mi.Color3f(1.), mi.Float(1.)
        options = dict(rgb_only=1., distortion_loss_weight=.1, max_hits=16.,
                       max_candidates=8., collect_rgb_deletion_evidence=1.)
        actual = self.compare(model.loss_and_grad(*arguments, options),
            model.loss_and_grad(*arguments, options | {"compact_gradient_capacity": count}), count)
        self.assertGreater(int(actual["voxel_count"][0]), 1)

    def test_frozen_replay_handles_changing_touched_rows_without_sync_or_compress(self):
        payload, inputs = bounded_fixture()
        model = restore_model(payload, mi)
        count = int(model.active_voxel_count)
        replay = NativeLossReplay(dr)
        options = dict(rgb_only=1., distortion_loss_weight=.1, max_hits=8., max_candidates=8.,
                       compact_gradient_capacity=float(count))
        counts = []
        for shift in (0., .08, 1.7, .15, 0.):
            arguments = inputs(shift)
            dense_options = {key: value for key, value in options.items() if key != "compact_gradient_capacity"}
            dense = model.loss_and_grad(*arguments, dense_options)
            with patch.object(dr, "compress", side_effect=AssertionError("Host compaction used")):
                with dr.scoped_set_flag(dr.JitFlag.ForbidSynchronization, True):
                    actual = replay(model, *arguments, options)
            out = self.compare(dense, actual, count)
            counts.append(int(out["voxel_count"][0]))
        self.assertGreater(len(set(counts)), 1)
        self.assertEqual(replay.statistics()["recordings"], 1)
        self.assertEqual(replay.statistics()["replays"], 4)

    def test_overflow_is_bounded_and_native_adam_rejects_without_changes(self):
        payload, inputs = bounded_fixture()
        model = restore_model(payload, mi)
        result = model.loss_and_grad(*inputs(), dict(rgb_only=1., max_candidates=8.,
                                                    compact_gradient_capacity=1.))
        self.assertGreater(int(np.array(result["voxel_count"])[0]), 1)
        self.assertTrue(bool(np.array(result["compact_gradient_overflow"])[0]))
        self.assertEqual(dr.width(result["voxel_ids"]), 1)
        optimizer = FalcorOptimizer(model, OptimizerConfig(), sparse_updates=True)
        before = optimizer_tests.capture(model, optimizer)
        updated = optimizer.step(model, result, update_allowed=mi.Bool(True))
        self.assertFalse(bool(np.array(updated["accepted"])[0]))
        optimizer_tests.assert_equal(self, before, optimizer_tests.capture(model, optimizer), exact=True)

    def test_adam_ignores_invalid_ids_and_nonfinite_gradients_in_unused_tail(self):
        models, optimizers = optimizer_tests.paired(29)
        for iteration in range(3):
            dense, compact = optimizer_tests.outputs(29, optimizers[0].dimensions, [1, 3], iteration,
                                                     appearance=[1, 1], geometry=[1, 1])
            compact["voxel_ids"] = mi.UInt32([1, 3, 0xffffffff, 0])
            compact["voxel_count"] = dr.opaque(mi.UInt32, 2)
            for name in VALIDITY:
                compact[name] = mi.UInt32(np.concatenate((np.array(compact[name]), [91, 0])))
            for name, width in optimizers[0].dimensions.items():
                compact["gradients"][name] = mi.Float(np.concatenate((np.array(compact["gradients"][name]),
                                                            np.full(2 * width, np.nan, np.float32))))
            for model, optimizer, out in zip(models, optimizers, (dense, compact)):
                updated = optimizer.step(model, out, update_allowed=mi.Bool(True))
                self.assertTrue(bool(np.array(updated["accepted"])[0]))
            optimizer_tests.assert_equal(self, *(optimizer_tests.capture(m, o)
                                                for m, o in zip(models, optimizers)))

    def test_overflow_retry_helper_and_device_count_adam_match_dense(self):
        payload, inputs = bounded_fixture()
        models = [restore_model(copy.deepcopy(payload), mi) for _ in range(2)]
        optimizers = [FalcorOptimizer(model, OptimizerConfig(), sparse_updates=True)
                      for model in models]
        replay = NativeLossReplay(dr)
        options = dict(rgb_only=1., distortion_loss_weight=.1, max_hits=8., max_candidates=8.)
        count = int(models[0].active_voxel_count)
        for iteration, shift in enumerate((0., .08)):
            arguments = (*inputs(shift), options)
            dense = models[0].loss_and_grad(*arguments)
            compact = native_gradient_result(models[1], arguments, replay, 1, np)
            actual = self.compare(dense, compact, count)
            required = int(actual["voxel_count"][0])
            self.assertGreater(required, 1)
            self.assertLessEqual(required, len(actual["voxel_ids"]))
            self.assertEqual(dr.width(compact["voxel_count"]), 1)
            self.assertEqual(replay.statistics()["gradient_overflow_retries"], 1)
            self.assertEqual(replay.statistics()["gradient_count_readbacks"], 2 + iteration)
            for model, optimizer, out in zip(models, optimizers, (dense, compact)):
                updated = optimizer.step(model, out, sample_count=1, epoch=iteration,
                                         update_allowed=mi.Bool(True))
                self.assertTrue(bool(np.array(updated["accepted"])[0]))
            optimizer_tests.assert_equal(self, *(optimizer_tests.capture(m, o)
                                                for m, o in zip(models, optimizers)))

    def test_compute_view_shard_compact_layout_loss_and_adam_match_dense(self):
        payload, inputs = bounded_fixture()
        models = [restore_model(copy.deepcopy(payload), mi) for _ in range(2)]
        config, _, _, _ = replay_fixture(np)
        config.training.rgb_only = True
        config.training.chunk_size = 0
        config.training.compact_gradients = True
        config.training.compact_gradient_rows = 1
        config.gradient.distortion_loss_weight = .1
        optimizers = [FalcorOptimizer(model, copy.deepcopy(config.optimizer), sparse_updates=True)
                      for model in models]
        replay = NativeLossReplay(dr)
        arguments = inputs()
        reference = np.array(arguments[1]).T.reshape(16, 16, 3).copy()
        offsets = np.zeros((256, 2), np.float32)
        inverse_pdf = np.ones(256, np.float32)
        outputs = []
        with patch.object(train, "prepare_camera", return_value={}) as prepare, \
             patch.object(train, "generate_ray_batch", return_value=arguments[0]) as rays:
            for index, model in enumerate(models):
                outputs.append(train.compute_view_shard(model, {"camera": {}}, reference,
                    config, offsets, inverse_pdf, None, mi, dr,
                    collect_deletion_evidence=False, compact_gradients=bool(index),
                    native_replay=replay if index else None))
        self.assertEqual(prepare.call_count, 2)
        self.assertEqual(rays.call_count, 2)
        dense, compact = outputs[0][0], outputs[1][0]
        count = int(models[0].active_voxel_count)
        actual = self.compare(dense, dict(compact, compact_gradient_overflow=mi.Bool(False)), count)
        capacity = len(actual["voxel_ids"])
        required = int(actual["voxel_count"][0])
        self.assertGreater(required, 1)
        self.assertLessEqual(required, capacity)
        self.assertEqual(dr.width(compact["voxel_count"]), 1)
        for name in VALIDITY:
            self.assertEqual(dr.width(compact[name]), capacity)
        for name, width in optimizers[1].dimensions.items():
            self.assertEqual(dr.width(compact["gradients"][name]), capacity * width)
        for name in ("foreground_evidence", "background_evidence"):
            self.assertEqual(dr.width(compact[name]), count)
        np.testing.assert_allclose(np.array(outputs[0][1]), np.array(outputs[1][1]),
                                   atol=3e-6, rtol=4e-5)
        for model, optimizer, (out, loss, finite, raw_alpha, estimated_alpha) in zip(
                models, optimizers, outputs):
            self.assertTrue(bool(np.array(finite)[0]))
            self.assertIsNone(raw_alpha)
            self.assertIsNone(estimated_alpha)
            updated = optimizer.step(model, out, sample_count=1, update_allowed=finite)
            self.assertTrue(bool(np.array(updated["accepted"])[0]))
        optimizer_tests.assert_equal(self, *(optimizer_tests.capture(m, o)
                                            for m, o in zip(models, optimizers)))
        self.assertEqual(replay.statistics()["gradient_overflow_retries"], 1)
        self.assertEqual(replay.statistics()["gradient_count_readbacks"], 2)

    def test_empty_model_and_zero_touched_rows(self):
        for empty in (False, True):
            model = small_fixture(empty=empty)
            inputs = small_inputs(19.)
            dense = model.loss_and_grad(*inputs, {"rgb_only": 1.})
            compact = model.loss_and_grad(*inputs, {"rgb_only": 1., "compact_gradient_capacity": 7.})
            actual = self.compare(dense, compact, int(model.active_voxel_count))
            self.assertEqual(int(actual["voxel_count"][0]), 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", default=RuntimeConfig().build_dir)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    global mi, dr, np
    with tempfile.TemporaryDirectory(prefix="mitsuba-compact-gradients-") as directory:
        mi, dr, np, handles = initialize_runtime(RuntimeConfig(args.build_dir, args.gpu), Path(directory), {})
        if getattr(mi.VoxelReconstruction, "compact_gradient_version", 0) < 1:
            raise RuntimeError("Rebuild this source checkout for native compact gradients")
        optimizer_tests.mi, optimizer_tests.dr, optimizer_tests.np = mi, dr, np
        optimizer_tests.original_tests.mi = mi
        optimizer_tests.original_tests.dr = dr
        optimizer_tests.original_tests.np = np
        result = unittest.TextTestRunner(verbosity=2).run(
            unittest.defaultTestLoader.loadTestsFromTestCase(CompactGradientTests))
        dr.sync_thread()
        return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
