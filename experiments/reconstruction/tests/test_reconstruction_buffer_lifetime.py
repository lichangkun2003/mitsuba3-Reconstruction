"""CPU boundary checks for chunk temporaries and refinement cache release."""

from __future__ import annotations

import copy
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import weakref

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import train


class TrackedResult(dict):
    pass


class NumPyDr:
    """Eager CPU arrays expose Python ownership independently of GPU timing."""

    arange = staticmethod(lambda dtype, count: np.arange(count, dtype=dtype))
    opaque = staticmethod(lambda dtype, value: dtype(value))
    zeros = staticmethod(lambda dtype, count: np.zeros(count, dtype=dtype))
    empty = staticmethod(lambda dtype, count: np.empty(count, dtype=dtype))
    maximum = staticmethod(np.maximum)
    sum = staticmethod(np.sum)
    all = staticmethod(np.all)
    isfinite = staticmethod(np.isfinite)
    width = staticmethod(np.size)

    @staticmethod
    def scatter(target, values, index):
        target[index] = values

    @staticmethod
    def eval(*arrays):
        pass


class BufferLifetimeTests(unittest.TestCase):
    def test_chunks_release_native_outputs_before_next_call(self):
        config = copy.deepcopy(train.CONFIG)
        config.render.width, config.render.height = 6, 1
        config.training.chunk_size = 2
        reference = np.zeros((1, 6, 3), np.float32)
        mi = SimpleNamespace(Float=np.float32, Bool=np.bool_, UInt32=np.uint32,
                             Color3f=lambda *values: np.asarray(values, np.float32))
        result_refs, gradient_refs = [], []
        case = self

        class Model:
            calls = 0

            def loss_and_grad(self, ray, rgb, alpha, background, pdf, settings):
                # Checking inside the next producer catches assignment's RHS
                # retaining the previous result, even if it dies on return.
                case.assertTrue(all(value() is None for value in result_refs))
                case.assertTrue(all(value() is None for value in gradient_refs))
                self.calls += 1
                count, scale = len(ray), self.calls
                gradients = {name: np.full(width * 3, scale, np.float32)
                             for name, width in (("radiance_sh", 27), ("opacity_sh", 1),
                                                 ("centers", 3), ("log_scales", 3),
                                                 ("rotation_tangent", 3))}
                result = TrackedResult(
                    gradients=gradients,
                    loss=np.full(count, scale, np.float32),
                    rgb_loss=np.full(count, scale * .75, np.float32),
                    distortion_loss=np.full(count, scale * .25, np.float32),
                    raw_alpha=np.full(count, scale / 4, np.float32),
                    alpha=np.full(count, scale / 4, np.float32))
                for name in ("appearance_valid", "geometry_valid", "background_carve_valid",
                             "foreground_evidence", "background_evidence",
                             "growth_observed", "growth_unknown", "deletion_observed", "deletion_unknown"):
                    result[name] = np.full(3, scale, np.uint32)
                for name in ("growth_gain", "deletion_gain"):
                    result[name] = np.full(3, scale, np.float32)
                result_refs.append(weakref.ref(result))
                gradient_refs.extend(weakref.ref(value) for value in gradients.values())
                return result

        model = Model()
        with (patch.object(train, "prepare_camera", return_value={}),
              patch.object(train, "generate_ray_batch", side_effect=lambda render, camera, indices, *a, **k: indices),
              patch.object(train, "native_traversal_settings", side_effect=lambda model, options: options)):
            accumulator, loss, finite, raw, alpha = train.compute_view_shard(
                model, {"camera": {}}, reference, config, np.zeros((6, 2), np.float32),
                np.ones(6, np.float32), None, mi, NumPyDr,
                need_rendered_alpha=True, collect_growth_evidence=True,
                collect_deletion_evidence=True)
        self.assertEqual(model.calls, 3)
        self.assertTrue(all(value() is None for value in result_refs + gradient_refs))
        self.assertEqual(float(loss), 12.)
        self.assertEqual(float(accumulator["rgb_loss_sum"]), 9.)
        self.assertEqual(float(accumulator["distortion_loss_sum"]), 3.)
        self.assertTrue(finite)
        self.assertIsNone(raw)
        np.testing.assert_array_equal(alpha, [.25, .25, .5, .5, .75, .75])
        for value in accumulator["gradients"].values():
            np.testing.assert_array_equal(value, np.full(value.size, 6, np.float32))
        for name in ("appearance_valid", "geometry_valid", "background_carve_valid",
                     "growth_gain", "growth_observed", "growth_unknown",
                     "deletion_gain", "deletion_observed", "deletion_unknown"):
            np.testing.assert_array_equal(accumulator[name], [6, 6, 6])
        for name in ("foreground_evidence", "background_evidence"):
            np.testing.assert_array_equal(accumulator[name], [3, 3, 3])

    def test_refinement_flush_waits_for_new_model_and_drops_snapshot_alias(self):
        events, refs = [], []

        class Model:
            def snapshot(self):
                result = TrackedResult(parameter=np.ones(3, np.float32))
                refs.append(weakref.ref(result))
                refs.append(weakref.ref(result["parameter"]))
                return result

        class Dr:
            def eval(self, *values):
                self.pending = values
                events.append("evaluated_model")

            def sync_thread(self):
                events.append("finished_model")
                self.pending = None

            def flush_malloc_cache(self):
                self.assert_released()
                events.append("released_cache")

        dr = Dr()
        dr.assert_released = lambda: self.assertTrue(all(value() is None for value in refs))
        with patch.object(train.time, "perf_counter", side_effect=(10., 10.25)):
            elapsed = train.release_topology_cache(Model(), dr)
        self.assertEqual(events, ["evaluated_model", "finished_model", "released_cache"])
        self.assertEqual(elapsed, .25)


if __name__ == "__main__":
    unittest.main()
