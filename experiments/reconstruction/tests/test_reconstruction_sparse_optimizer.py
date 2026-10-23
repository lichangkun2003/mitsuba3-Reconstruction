"""CUDA sparse Adam parity with the retained dense implementation.

No images or dataset are rendered. The optional benchmark includes device-side
selection, validation, parameter/moment commits and GPU completion in every step.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
from pathlib import Path
import statistics
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from preview import RuntimeConfig
from reconstruction_common import initialize_runtime, restore_model
from reconstruction_optimizer import FalcorOptimizer, OptimizerConfig
import test_reconstruction_optimization as original_tests


def fixture(count=7, *, sh=4):
    return original_tests.seed_checkpoint(cells=tuple(range(count)), sh=sh)


def capture(model, optimizer, *, expose_state=True):
    result = {name: np.array(value, copy=True) for name, value in model.snapshot().items()}
    groups = optimizer.state if expose_state else optimizer.state_dict()["groups"]
    result.update({f"{name}/{field}": np.array(value, copy=True)
                   for name, group in groups.items() for field, value in group.items()})
    return result


def assert_equal(test, first, second, *, exact=False):
    test.assertEqual(first.keys(), second.keys())
    for key in first:
        if exact or first[key].dtype.kind in "iub":
            np.testing.assert_array_equal(first[key], second[key], err_msg=key)
        else:
            np.testing.assert_allclose(first[key], second[key], rtol=3e-5, atol=3e-7,
                                       equal_nan=True, err_msg=key)


def paired(payload=None, config=None):
    payload = fixture() if payload is None else payload
    models = [restore_model(copy.deepcopy(payload), mi) for _ in range(2)]
    optimizers = [FalcorOptimizer(model, copy.deepcopy(config or OptimizerConfig()),
                                 sparse_updates=sparse)
                  for model, sparse in zip(models, (False, True))]
    return models, optimizers


def gradients(model, iteration=0, *, appearance=None, geometry=None, zero=False):
    out = original_tests.result_for(model)
    count = int(model.active_voxel_count)
    ids = np.arange(count)
    appearance = ((ids + iteration) % 3 != 0) if appearance is None else appearance
    geometry = ((ids + 2 * iteration) % 4 < 2) if geometry is None else geometry
    out["appearance_valid"] = mi.UInt32(np.asarray(appearance, np.uint32))
    out["geometry_valid"] = mi.UInt32(np.asarray(geometry, np.uint32) * 2)
    out["background_carve_valid"] = mi.UInt32((ids % 2).astype(np.uint32))
    for group, value in out["gradients"].items():
        data = np.linspace(-13., 13., dr.width(value), dtype=np.float32)
        data *= -0.003 if iteration % 2 else 1.
        if len(data):
            data[::3] = 0.  # Valid zero gradients must still age/decay Adam.
        if zero:
            data[:] = 0.
        out["gradients"][group] = mi.Float(data)
    return out


class SparseOptimizerTests(unittest.TestCase):
    def assert_pair(self, models, optimizers):
        assert_equal(self, capture(models[0], optimizers[0]), capture(models[1], optimizers[1]))

    def test_unexposed_state_fastpath_multistep_matches_dense(self):
        models, optimizers = paired(fixture(29))
        # No capture(), state_dict(), or mutable-state access inside this loop.
        # Changing validity masks imitates successive views without rendering.
        for iteration in range(12):
            for model, optimizer in zip(models, optimizers):
                optimizer.step(model, gradients(model, iteration, zero=iteration == 5),
                               sample_count=2, epoch=iteration, update_allowed=mi.Bool(True))
                self.assertFalse(optimizer._state_exposed)
        self.assert_pair(models, optimizers)

    def test_record_and_replay_have_no_host_compaction_or_synchronization(self):
        count = 29
        models, optimizers = paired(fixture(count))
        ids = np.arange(count)
        masks = [
            (np.zeros(count), np.zeros(count)),
            (ids % 3 != 0, ids % 4 < 2),
            (np.ones(count), np.ones(count)),
            (np.zeros(count), np.ones(count)),
            (np.ones(count), np.zeros(count)),
            (np.zeros(count), np.zeros(count)),
        ]
        # Upload references before the forbidden-sync region, as a renderer
        # would supply gradients already on the device. Start with an empty
        # selection: recording must not specialize its length or later masks.
        batches = [[gradients(model, iteration, appearance=app, geometry=geom,
                              zero=iteration == 2)
                    for iteration, (app, geom) in enumerate(masks)] for model in models]
        allowed = mi.Bool(True)
        dr.eval(batches, allowed)
        dr.sync_thread()
        for iteration, out in enumerate(batches[0]):
            optimizers[0].step(models[0], out, epoch=iteration, update_allowed=allowed)
        with patch.object(dr, "compress", side_effect=AssertionError("Host-sized compaction used")):
            with dr.scoped_set_flag(dr.JitFlag.ForbidSynchronization, True):
                for iteration, out in enumerate(batches[1]):
                    result = optimizers[1].step(models[1], out, epoch=iteration,
                                                update_allowed=allowed)
                    dr.eval(result["accepted"], models[1].snapshot())
                    self.assertFalse(optimizers[1]._state_exposed)
        self.assertTrue(bool(np.asarray(result["accepted"])[0]))
        statistics = optimizers[1].statistics()
        self.assertLessEqual(statistics["recordings"], 2)
        self.assertGreaterEqual(statistics["replays"], len(masks) - 2)
        self.assert_pair(models, optimizers)

    def test_multiple_device_batches_and_partial_tail_match_dense(self):
        count = 65553
        payload = original_tests.seed_checkpoint(cells=np.arange(count, dtype=np.uint32),
                                                 resolution=64, sh=1)
        models, optimizers = paired(payload)
        tail = np.zeros(count, np.uint32)
        tail[[0, 32767, 32768, 65535, 65536, count - 1]] = 1
        for iteration, (appearance, geometry) in enumerate((
                (np.ones(count), np.ones(count)), (tail, 1 - tail),
                (np.zeros(count), np.zeros(count)))):
            for model, optimizer in zip(models, optimizers):
                optimizer.step(model, gradients(model, iteration,
                    appearance=appearance, geometry=geometry), update_allowed=mi.Bool(True))
            # Do not expose mutable state: this also exercises the ordinary
            # cached execution path through full, partial, then zero batches.
            assert_equal(self, *(capture(model, optimizer, expose_state=False)
                                for model, optimizer in zip(models, optimizers)))

    def test_single_voxel_scalar_buffers_replay_without_synchronization(self):
        for sh in (1, 9):
            with self.subTest(opacity_sh_count=sh):
                models, optimizers = paired(fixture(1, sh=sh))
                masks = ((True, True), (False, False), (True, True),
                         (False, True), (True, False), (True, True))
                batches = []
                for model in models:
                    sequence = []
                    for iteration, (app, geom) in enumerate(masks):
                        out = gradients(model, iteration, appearance=np.array([app]),
                                        geometry=np.array([geom]), zero=iteration == 2)
                        # Unlike a one-element linspace whose lone element is
                        # deliberately zeroed by gradients(), this also makes
                        # the scalar opacity parameter and moments change.
                        out["gradients"]["opacity_sh"] = mi.Float(np.full(
                            sh, 0. if iteration == 2 else .25 * (iteration + 1), np.float32))
                        sequence.append(out)
                    batches.append(sequence)
                allowed = mi.Bool(True)
                dr.eval(batches, allowed)
                dr.sync_thread()
                for out in batches[0]:
                    optimizers[0].step(models[0], out, update_allowed=allowed)
                # A one-element gather can become a direct scalar reference;
                # a subsequent in-place scatter must not leave that input dirty.
                # Do not materialize snapshots or read values between updates.
                with patch.object(dr, "compress", side_effect=AssertionError("Host-sized compaction used")):
                    with dr.scoped_set_flag(dr.JitFlag.ForbidSynchronization, True):
                        for out in batches[1]:
                            result = optimizers[1].step(models[1], out, update_allowed=allowed)
                            dr.eval(result["accepted"])
                self.assertTrue(bool(np.asarray(result["accepted"])[0]))
                self.assertGreater(optimizers[1].statistics()["replays"], 0)
                self.assert_pair(models, optimizers)

    def test_multistep_independent_masks_zero_gradients_and_group_steps(self):
        for guarded in (False, True):
            with self.subTest(guarded=guarded):
                models, optimizers = paired()
                for iteration in range(9):
                    for model, optimizer in zip(models, optimizers):
                        out = gradients(model, iteration, zero=iteration == 4)
                        before = capture(model, optimizer)
                        result = optimizer.step(model, out, sample_count=2, epoch=iteration,
                            update_allowed=mi.Bool(True) if guarded else None)
                        if guarded:
                            self.assertTrue(bool(np.asarray(result["accepted"])[0]))
                        after = capture(model, optimizer)
                        appearance = np.asarray(out["appearance_valid"]) > 0
                        geometry = np.asarray(out["geometry_valid"]) > 0
                        for name, width in optimizer.dimensions.items():
                            active = geometry if name in ("centers", "log_scales", "rotation_tangent") else appearance
                            np.testing.assert_array_equal(after[f"{name}/steps"],
                                before[f"{name}/steps"] + active.astype(np.uint32))
                            for field in ("mean", "squared_mean"):
                                key = f"{name}/{field}"
                                np.testing.assert_array_equal(after[key].reshape(-1, width)[~active],
                                                              before[key].reshape(-1, width)[~active])
                    self.assert_pair(models, optimizers)

    def test_disabled_groups_and_opacity_warmup(self):
        config = OptimizerConfig(lr_rotation=0., lr_radiance=0.)
        models, optimizers = paired(config=config)
        for epoch in (0, 19, 20, 21, 35, 50):
            for model, optimizer in zip(models, optimizers):
                optimizer.step(model, gradients(model, epoch), epoch=epoch, coarse_to_fine=False,
                               update_allowed=mi.Bool(True))
            self.assert_pair(models, optimizers)
        for optimizer in optimizers:
            for group in ("rotation_tangent", "radiance_sh"):
                np.testing.assert_array_equal(np.asarray(optimizer.state[group]["steps"]), 0)
        disabled = OptimizerConfig(lr_rotation=0., lr_radiance=0., lr_opacity=0.,
                                   lr_center=0., lr_log_scale=0.)
        models, optimizers = paired(config=disabled)
        for model, optimizer in zip(models, optimizers):
            before = capture(model, optimizer)
            optimizer.step(model, gradients(model), update_allowed=mi.Bool(True))
            assert_equal(self, capture(model, optimizer), before, exact=True)

    def test_no_valid_entries_appearance_only_geometry_only_and_empty_model(self):
        for count in (0, 7):
            for app, geom in ((False, False), (True, False), (False, True), (True, True)):
                with self.subTest(count=count, appearance=app, geometry=geom):
                    models, optimizers = paired(fixture(count))
                    for model, optimizer in zip(models, optimizers):
                        out = gradients(model, appearance=np.full(count, app), geometry=np.full(count, geom))
                        optimizer.step(model, out, update_allowed=mi.Bool(True))
                    self.assert_pair(models, optimizers)

    def test_direct_nonfinite_gradients_keep_sanitizing_contract(self):
        models, optimizers = paired()
        for model, optimizer in zip(models, optimizers):
            out = gradients(model, appearance=np.ones(7), geometry=np.ones(7))
            for name, values in out["gradients"].items():
                data = np.array(values, copy=True)
                data[0], data[-1] = np.nan, np.inf
                out["gradients"][name] = mi.Float(data)
            optimizer.step(model, out)
        self.assert_pair(models, optimizers)

    def test_late_batch_moment_overflow_rejects_entire_transaction_and_latches(self):
        count = 65553
        payload = original_tests.seed_checkpoint(cells=np.arange(count, dtype=np.uint32),
                                                 resolution=64, sh=1)
        for sparse in (False, True):
            with self.subTest(sparse=sparse):
                model = restore_model(copy.deepcopy(payload), mi)
                optimizer = FalcorOptimizer(model, OptimizerConfig(gradient_clip=1e30),
                                            sparse_updates=sparse)
                before = capture(model, optimizer, expose_state=False)
                out = gradients(model, appearance=np.ones(count), geometry=np.ones(count))
                radiance = np.array(out["gradients"]["radiance_sh"], copy=True)
                # The input itself is finite. Its square overflows only in the
                # last batch, after earlier groups/batches could have committed
                # if validation and updates were incorrectly interleaved.
                radiance[-1] = np.float32(1e30)
                out["gradients"]["radiance_sh"] = mi.Float(radiance)
                result = optimizer.step(model, out, update_allowed=mi.Bool(True))
                self.assertFalse(bool(np.asarray(result["accepted"])[0]))
                assert_equal(self, capture(model, optimizer, expose_state=False), before, exact=True)
                result = optimizer.step(model, gradients(model, 1), update_allowed=mi.Bool(True))
                self.assertFalse(bool(np.asarray(result["accepted"])[0]))
                assert_equal(self, capture(model, optimizer, expose_state=False), before, exact=True)

    def test_false_guard_is_atomic_and_latched(self):
        models, optimizers = paired()
        for model, optimizer in zip(models, optimizers):
            before = capture(model, optimizer)
            for allowed in (False, True):
                result = optimizer.step(model, gradients(model), update_allowed=mi.Bool(allowed))
                self.assertFalse(bool(np.asarray(result["accepted"])[0]))
                assert_equal(self, capture(model, optimizer), before, exact=True)

    def test_external_state_replacement_and_inactive_nonfinite_rejection(self):
        models, optimizers = paired()
        for model, optimizer in zip(models, optimizers):
            optimizer.step(model, gradients(model), update_allowed=mi.Bool(True))
            state = optimizer.state["centers"]
            state["mean"] = mi.Float(np.linspace(-.1, .1, 21, dtype=np.float32))
            state["squared_mean"] = mi.Float(np.full(21, .05, np.float32))
            state["steps"] = mi.UInt32(np.arange(7, dtype=np.uint32) + 17)
            optimizer.step(model, gradients(model, 1), update_allowed=mi.Bool(True))
        self.assert_pair(models, optimizers)
        for model, optimizer in zip(models, optimizers):
            # Slot 0 is inactive. A full-state invariant must notice replacement.
            moment = np.array(optimizer.state["centers"]["mean"], copy=True)
            moment[0] = np.nan
            optimizer.state["centers"]["mean"] = mi.Float(moment)
            before = capture(model, optimizer)
            out = gradients(model, appearance=np.array([0, 1, 0, 0, 0, 0, 0]),
                            geometry=np.array([0, 1, 0, 0, 0, 0, 0]))
            result = optimizer.step(model, out, update_allowed=mi.Bool(True))
            self.assertFalse(bool(np.asarray(result["accepted"])[0]))
            assert_equal(self, capture(model, optimizer), before, exact=True)

    def test_held_model_snapshot_preserves_old_values(self):
        models, optimizers = paired()
        for model, optimizer in zip(models, optimizers):
            held = model.snapshot()
            expected = {name: np.array(value, copy=True) for name, value in held.items()}
            optimizer.step(model, gradients(model), update_allowed=mi.Bool(True))
            actual = {name: np.array(value, copy=True) for name, value in held.items()}
            assert_equal(self, actual, expected, exact=True)
        self.assert_pair(models, optimizers)

    def test_snapshot_and_moment_aliases_survive_already_recorded_updates(self):
        models, optimizers = paired()
        for model, optimizer in zip(models, optimizers):
            for iteration in range(2):
                optimizer.step(model, gradients(model, iteration), update_allowed=mi.Bool(True))
            # Hold independent array handles after recording, when replay must
            # perform copy-on-write for newly introduced external aliases too.
            held = dict(model.snapshot())
            held.update({f"{name}/{field}": type(value)(value)
                         for name, state in optimizer.state.items() for field, value in state.items()})
            expected = {name: np.array(value, copy=True) for name, value in held.items()}
            for iteration in range(2, 5):
                optimizer.step(model, gradients(model, iteration), update_allowed=mi.Bool(True))
            actual = {name: np.array(value, copy=True) for name, value in held.items()}
            assert_equal(self, actual, expected, exact=True)
        self.assert_pair(models, optimizers)

    def test_inactive_inplace_moment_corruption_rejects_transaction(self):
        models, optimizers = paired()
        for model, optimizer in zip(models, optimizers):
            optimizer.step(model, gradients(model), update_allowed=mi.Bool(True))
            dr.scatter(optimizer.state["centers"]["mean"], mi.Float(float("nan")), mi.UInt32(0))
            dr.eval(optimizer.state["centers"]["mean"])
            before = capture(model, optimizer)
            mask = np.array([0, 1, 0, 0, 0, 0, 0])
            result = optimizer.step(model, gradients(model, appearance=mask, geometry=mask),
                                    update_allowed=mi.Bool(True))
            self.assertFalse(bool(np.asarray(result["accepted"])[0]))
            assert_equal(self, capture(model, optimizer), before, exact=True)

    def test_same_mutable_array_in_two_moment_slots_is_rejected_without_update(self):
        model = restore_model(fixture(), mi)
        optimizer = FalcorOptimizer(model)
        optimizer.step(model, gradients(model), update_allowed=mi.Bool(True))
        state = optimizer.state
        # Sharing storage through distinct array handles has normal COW
        # semantics (tested above). One Python mutable object in two slots
        # cannot represent independent Adam moments and must be rejected.
        state["log_scales"]["mean"] = state["centers"]["mean"]
        before = capture(model, optimizer)
        with self.assertRaisesRegex(ValueError, "distinct array objects"):
            optimizer.step(model, gradients(model, 1), update_allowed=mi.Bool(True))
        assert_equal(self, capture(model, optimizer), before, exact=True)

    def test_remap_load_and_model_parameter_replacement(self):
        models, optimizers = paired()
        for model, optimizer in zip(models, optimizers):
            optimizer.step(model, gradients(model), update_allowed=mi.Bool(True))
            optimizer.remap(np.array([5, 1, 3], np.uint32), 7, reset_radiance=True)
            state = optimizer.state_dict()
            optimizer.load_state_dict(copy.deepcopy(state))
            model.set_parameters({"centers": mi.Float(np.tile([.4, .6, .5], 7))})
            for iteration in range(3):
                optimizer.step(model, gradients(model, iteration), update_allowed=mi.Bool(True))
        self.assert_pair(models, optimizers)
        saved = optimizers[0].state_dict()
        optimizers[1].load_state_dict(copy.deepcopy(saved))
        self.assert_pair(models, optimizers)

    def test_replay_tracks_replaced_model_reset_and_remapped_storage(self):
        models, optimizers = paired()
        phases = ("initial", "replace_model", "remap_same_shape", "load",
                  "reset_resize", "remap_grow", "remap_empty", "remap_from_empty")
        for phase_index, phase in enumerate(phases):
            with self.subTest(phase=phase):
                for index, optimizer in enumerate(optimizers):
                    if phase == "replace_model":
                        payload = fixture()
                        payload["model_state"]["centers"][:] = np.tile([.4, .6, .5], 7)
                        payload["model_state"]["opacity_sh"][:] = .2
                        models[index] = restore_model(payload, mi)
                    elif phase == "remap_same_shape":
                        optimizer.remap(np.array([5, 1, 3], np.uint32), 7, reset_radiance=True)
                    elif phase == "load":
                        optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
                    elif phase == "reset_resize":
                        models[index] = restore_model(fixture(37), mi)
                        optimizer.reset(models[index])
                    elif phase == "remap_grow":
                        optimizer.remap(np.array([2, 0, 4], np.uint32), 41)
                        models[index] = restore_model(fixture(41), mi)
                    elif phase in ("remap_empty", "remap_from_empty"):
                        count = 0 if phase == "remap_empty" else 13
                        optimizer.remap(np.zeros(0, np.uint32), count)
                        models[index] = restore_model(fixture(count), mi)
                    for iteration in range(2):
                        optimizer.step(models[index], gradients(models[index], phase_index + iteration),
                                       update_allowed=mi.Bool(True))
                assert_equal(self, *(capture(model, optimizer, expose_state=False)
                                    for model, optimizer in zip(models, optimizers)))

    def test_loading_invalid_moments_does_not_partially_replace_state(self):
        models, optimizers = paired()
        for model, optimizer in zip(models, optimizers):
            optimizer.step(model, gradients(model), update_allowed=mi.Bool(True))
            before = capture(model, optimizer)
            broken = optimizer.state_dict()
            broken["groups"]["radiance_sh"]["mean"][:] = .25
            broken["groups"]["rotation_tangent"]["squared_mean"][0] = np.nan
            with self.assertRaises(ValueError):
                optimizer.load_state_dict(broken)
            assert_equal(self, capture(model, optimizer), before, exact=True)


def benchmark(voxels, iterations):
    """Fixed gradients isolate Adam; this is not a full training speed claim."""
    resolution = max(2, int(np.ceil(voxels ** (1 / 3))))
    while resolution ** 3 < voxels:
        resolution += 1
    payload = original_tests.seed_checkpoint(cells=np.arange(voxels, dtype=np.uint32),
                                             resolution=resolution, sh=9)
    payload["model_metadata"]["opacity_sh_count"] = 1
    payload["model_state"]["opacity_sh"] = np.zeros(voxels, np.float32)
    labels = ("dense_reference", "host_compact_reference", "device_sparse")
    models = [restore_model(copy.deepcopy(payload), mi) for _ in labels]
    optimizers = [FalcorOptimizer(model, sparse_updates=index != 0,
                                 replay_updates=index == 2)
                  for index, model in enumerate(models)]
    del payload
    print(json.dumps({"benchmark": "optimizer only; GPU completion included", "voxels": voxels,
                      "iterations": iterations, "device_completion_included": True,
                      "implementations": list(labels)}))
    for fraction in (.01, .1, .5, 1.):
        count = max(1, int(voxels * fraction))
        mask = np.zeros(voxels, np.uint32)
        mask[np.linspace(0, voxels - 1, count, dtype=np.int64)] = 1
        outs = [gradients(model, appearance=mask, geometry=mask) for model in models]
        dr.eval(*[value for out in outs for value in out["gradients"].values()],
                *[out[name] for out in outs for name in
                  ("appearance_valid", "geometry_valid", "background_carve_valid")])
        allowed = mi.Bool(True)

        def update(index):
            optimizer, model, out = optimizers[index], models[index], outs[index]
            if index == 1:
                # Benchmark the previous implementation, including both
                # dr.compress() length synchronizations and its commits.
                return optimizer._step_compact_reference(model, out, sample_count=1,
                    epoch=0, coarse_to_fine=True, update_allowed=allowed)
            return optimizer.step(model, out, update_allowed=allowed)

        # Warm initial counters/guards and JIT caches, then alternate order to
        # reduce drift bias. A first-step-only warmup misses later kernels.
        for _ in range(5):
            for index, model in enumerate(models):
                update(index)
                dr.eval(*model.snapshot().values())
                dr.sync_thread()
        durations = [[] for _ in labels]
        for iteration in range(iterations):
            order = [(iteration + offset) % len(labels) for offset in range(len(labels))]
            if iteration % 2:
                order.reverse()
            for index in order:
                dr.sync_thread()
                started = time.perf_counter()
                update(index)
                # step() materializes its moments. Complete the native model
                # scatters too, without exposing the optimizer's mutable state.
                dr.eval(*models[index].snapshot().values())
                dr.sync_thread()
                durations[index].append(time.perf_counter() - started)
        medians = [statistics.median(values) for values in durations]
        print(json.dumps({"active_fraction": fraction, "active_voxels": count,
                          "seconds": dict(zip(labels, medians)),
                          "dense_over_device": medians[0] / medians[2],
                          "host_compact_over_device": medians[1] / medians[2],
                          "samples": dict(zip(labels, durations)),
                          "device_replay": optimizers[2].statistics()}))
        del outs
        gc.collect()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", default=RuntimeConfig().build_dir)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--benchmark-voxels", type=int, default=524288)
    parser.add_argument("--benchmark-iterations", type=int, default=9)
    args = parser.parse_args()
    if args.benchmark_voxels < 1 or args.benchmark_iterations < 1:
        parser.error("Benchmark sizes must be positive")
    global mi, dr, np
    with tempfile.TemporaryDirectory(prefix="mitsuba-sparse-optimizer-") as temporary:
        mi, dr, np, handles = initialize_runtime(RuntimeConfig(args.build_dir, args.gpu),
                                                Path(temporary), {})
        original_tests.mi, original_tests.dr, original_tests.np = mi, dr, np
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(SparseOptimizerTests)
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        dr.sync_thread()
        if not result.wasSuccessful():
            return 1
        if args.benchmark:
            gc.collect()
            benchmark(args.benchmark_voxels, args.benchmark_iterations)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
