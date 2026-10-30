"""CPU checks for exact overflow retry and gradient-layout routing."""

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
import weakref

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reconstruction_gradient_storage import gradient_capacity, native_gradient_result
from reconstruction_replay import NativeLossReplay


class Result(dict):
    pass


class GradientStorageTests(unittest.TestCase):
    def test_training_driver_budgets_capacity_growth_before_resizing_cache(self):
        from train import TrainConfig, configure_training_memory
        config = TrainConfig()
        config.render.width, config.render.height = 1237, 822
        config.render.max_hits = config.gradient.max_candidates = 16
        config.gradient.distortion_loss_weight = .1
        config.training.rgb_only = True
        config.training.chunk_size = 0
        config.training.gpu_reference_cache_mib = 4096
        model = SimpleNamespace(active_voxel_count=54_146_397, sh_count=9,
            opacity_sh_count=1, compact_gradient_version=1, core_resolution=1024, resolution=1040)
        resized = []
        cache = SimpleNamespace(capacity_bytes=4096 * 1024**2)
        def resize(capacity):
            resized.append(capacity)
            cache.capacity_bytes = capacity * 1024**2
        cache.resize = resize
        report = {"gpu": {"memory_mib": 32607}, "cuda_allocator": {"policy": "cuda_2m"}}
        plan = configure_training_memory(config, model, cache, report, 169, emit=lambda *a, **kw: None)
        self.assertEqual(plan["gradient_rows"], 2_033_628)
        self.assertEqual(resized, [256])
        with self.assertRaisesRegex(RuntimeError, "memory preflight"):
            configure_training_memory(config, model, cache, report, 169,
                compact_capacity_override=16_269_024, emit=lambda *a, **kw: None)
        self.assertEqual(resized, [256])
        self.assertFalse(report["memory_plans"]["1024"]["supported"])

    def test_small_models_keep_dense_path_and_large_models_require_new_native(self):
        model = SimpleNamespace(active_voxel_count=100)
        self.assertEqual(gradient_capacity(model, 100), 0)
        model.active_voxel_count = 54_146_397
        with self.assertRaisesRegex(RuntimeError, "rebuild"):
            gradient_capacity(model, 1_016_814)
        model.compact_gradient_version = 1
        self.assertEqual(gradient_capacity(model, 1_016_814), 2_033_628)
        self.assertEqual(gradient_capacity(model, 1_016_814, enabled=False), 0)

    def test_overflow_is_discarded_before_retry_and_capacity_hint_is_reused(self):
        case = self
        refs, capacities = [], []

        class Model:
            active_voxel_count = 100
            topology_revision = 1

            def loss_and_grad(self, *arguments):
                case.assertTrue(all(ref() is None for ref in refs))
                capacity = arguments[-1]["compact_gradient_capacity"]
                capacities.append(capacity)
                result = Result(voxel_count=np.array([9], np.uint32),
                                voxel_ids=np.arange(capacity, dtype=np.uint32))
                refs.append(weakref.ref(result))
                return result

        class Replay:
            _gradient_capacity_revision = None
            _gradient_capacity_hint = 0
            _gradient_count_readbacks = 0
            _gradient_overflow_retries = 0
            clears = 0

            def __call__(self, model, *arguments):
                return model.loss_and_grad(*arguments)

            def clear(self):
                self.clears += 1
                self._gradient_capacity_revision = None
                self._gradient_capacity_hint = 0

        model, replay, options = Model(), Replay(), {"rgb_only": True}
        result = native_gradient_result(model, (options,), replay, 4, np)
        self.assertEqual(capacities, [4, 100])
        self.assertEqual(replay._gradient_overflow_retries, 1)
        self.assertEqual(replay._gradient_count_readbacks, 2)
        self.assertEqual(options, {"rgb_only": True})
        del result
        result = native_gradient_result(model, (options,), replay, 4, np)
        self.assertEqual(capacities[-1], 100)
        del result
        model.topology_revision += 1
        result = native_gradient_result(model, (options,), replay, 4, np)
        self.assertEqual(capacities[-2:], [4, 100])

    def test_invalid_required_count_is_rejected(self):
        model = SimpleNamespace(active_voxel_count=20, topology_revision=0,
            loss_and_grad=lambda *args: {"voxel_count": np.array([21]), "voxel_ids": np.arange(4)})
        with self.assertRaisesRegex(RuntimeError, "exceeds"):
            native_gradient_result(model, ({},), None, 4, np)

    def test_capacity_guard_runs_after_old_result_and_recording_are_released(self):
        for reject in (False, True):
            with self.subTest(reject=reject):
                refs, capacities, guarded = [], [], []
                case = self

                class Model:
                    active_voxel_count = 100
                    topology_revision = 1

                    def loss_and_grad(self, *arguments):
                        case.assertTrue(all(ref() is None for ref in refs))
                        capacity = arguments[-1]["compact_gradient_capacity"]
                        capacities.append(capacity)
                        result = Result(voxel_count=np.array([9], np.uint32),
                                        voxel_ids=np.arange(capacity, dtype=np.uint32))
                        refs.append(weakref.ref(result))
                        return result

                class Replay:
                    _gradient_capacity_revision = None
                    _gradient_capacity_hint = 0
                    _gradient_count_readbacks = 0
                    _gradient_overflow_retries = 0
                    clears = 0
                    recording = None

                    def __call__(self, model, *arguments):
                        # Mimic a recording retaining its output until clear().
                        self.recording = model.loss_and_grad(*arguments)
                        return self.recording

                    def clear(self):
                        self.recording = None
                        self.clears += 1
                        self._gradient_capacity_revision = None
                        self._gradient_capacity_hint = 0

                model, replay = Model(), Replay()

                def guard(actual_model, next_capacity):
                    self.assertIs(actual_model, model)
                    self.assertEqual(capacities, [4])
                    self.assertEqual(replay.clears, 1)
                    self.assertIsNone(replay.recording)
                    self.assertTrue(all(ref() is None for ref in refs))
                    guarded.append(next_capacity)
                    if reject:
                        raise RuntimeError("expansion exceeds memory budget")

                replay.gradient_capacity_guard = guard
                if reject:
                    with self.assertRaisesRegex(RuntimeError, "memory budget"):
                        native_gradient_result(model, ({},), replay, 4, np)
                    self.assertEqual(capacities, [4])
                    self.assertTrue(all(ref() is None for ref in refs))
                else:
                    result = native_gradient_result(model, ({},), replay, 4, np)
                    self.assertEqual(capacities, [4, 100])
                    self.assertEqual(int(result["voxel_count"][0]), 9)
                self.assertEqual(guarded, [100])

    def test_replay_clear_preserves_capacity_guard(self):
        replay = NativeLossReplay(SimpleNamespace(freeze=lambda *args, **kwargs: None))
        self.assertIsNone(replay.gradient_capacity_guard)
        guard = lambda model, capacity: None
        replay.gradient_capacity_guard = guard
        replay.clear()
        self.assertIs(replay.gradient_capacity_guard, guard)

    def test_zero_touched_rows_and_dense_calls_are_preserved(self):
        marker = Result(voxel_count=np.array([0]), voxel_ids=np.array([], np.uint32))
        model = SimpleNamespace(active_voxel_count=20, topology_revision=0,
                                loss_and_grad=lambda *args: marker)
        self.assertIs(native_gradient_result(model, ({},), None, 4, np), marker)
        self.assertIs(native_gradient_result(model, ({},), None, 0, np), marker)


if __name__ == "__main__":
    unittest.main()
