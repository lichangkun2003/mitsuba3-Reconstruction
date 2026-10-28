"""CPU checks for topology/Adam values and lazy source-buffer ownership."""

from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
import weakref

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reconstruction_optimizer import FalcorOptimizer
from reconstruction_growth_rgb import RGB_STAT_FIELDS
from reconstruction_topology_gpu import GPUTopologyLifecycle


class DeferredArray:
    """A gather owns its source until evaluation, as a device expression does."""

    def __init__(self, values=None, *, source=None, indices=None, active=None,
                 increment=None, on_add=None):
        if isinstance(values, DeferredArray):
            if values.on_add is not None:
                values.on_add(values)
            on_add = values.on_add
            # The CPU fixture models independent writable wrapper contents;
            # CUDA tests exercise actual shared storage and copy-on-write.
            values = values.evaluate().copy()
        self.values = values
        self.source, self.indices, self.active = source, indices, active
        self.increment, self.on_add = increment, on_add
        self.index = id(self)

    def __add__(self, increment):
        if self.on_add is not None:
            self.on_add(self)
        return DeferredArray(source=self, increment=increment, on_add=self.on_add)

    def evaluate(self):
        if self.source is not None:
            source = self.source.evaluate()
            if self.increment is not None:
                self.values = source + self.increment
            else:
                self.values = np.zeros(self.indices.size, dtype=source.dtype)
                self.values[self.active] = source[self.indices[self.active]]
            self.source = self.indices = self.active = None
            self.increment = None
        return self.values


class DeferredDr:
    arange = staticmethod(lambda dtype, count: np.arange(count, dtype=dtype))
    width = staticmethod(np.size)
    isfinite = staticmethod(np.isfinite)
    maximum = staticmethod(np.maximum)
    select = staticmethod(np.where)
    opaque = staticmethod(lambda dtype, value: dtype(value))
    sync_thread = staticmethod(lambda: None)
    flush_malloc_cache = staticmethod(lambda: None)
    ReduceOp = SimpleNamespace(Add="add")
    ReduceMode = SimpleNamespace(Direct="direct", NoConflicts="no_conflicts")

    def __init__(self, case, *, check_release=True):
        self.case, self.check_release = case, check_release
        self.previous_source = None
        self.sources = []
        self.eval_sizes = []
        self.scatter_modes = []

    def track_source(self, source):
        if self.check_release and self.previous_source is not None:
            self.case.assertIsNone(self.previous_source(),
                "Previous source is still alive at the next replacement")
        self.previous_source = weakref.ref(source)
        self.sources.append(self.previous_source)

    @staticmethod
    def zeros(dtype, count):
        return DeferredArray(np.zeros(count, dtype=dtype))

    def gather(self, dtype, source, indices, active):
        indices, active = np.asarray(indices), np.asarray(active)
        if isinstance(source, DeferredArray):
            self.track_source(source)
            return DeferredArray(source=source, indices=indices, active=active)
        values = np.zeros(indices.size, dtype=dtype)
        values[active] = source[indices[active]]
        return values

    def eval(self, *values):
        self.eval_sizes.append(len(values))
        for value in values:
            if isinstance(value, DeferredArray):
                value.evaluate()

    def scatter_reduce(self, operation, target, values, indices, *, mode):
        self.case.assertEqual(operation, self.ReduceOp.Add)
        data = target.evaluate()
        expected_mode = self.ReduceMode.Direct if data.dtype.kind == "f" else self.ReduceMode.NoConflicts
        self.case.assertEqual(mode, expected_mode)
        self.scatter_modes.append(mode)
        np.add.at(data, indices, values)


class RemapLifetimeTests(unittest.TestCase):
    def make_optimizer(self, *, check_release=True):
        optimizer = FalcorOptimizer.__new__(FalcorOptimizer)
        optimizer.dr = DeferredDr(self, check_release=check_release)
        optimizer.mi = SimpleNamespace(Float=np.float32, UInt32=np.uint32)
        optimizer.count = 4
        optimizer.dimensions = dict(radiance_sh=27, opacity_sh=1, centers=3,
                                    log_scales=3, rotation_tangent=3)
        expected = {
            name: {field: np.arange(4 * width if field != "steps" else 4,
                                   dtype=np.float32 if field != "steps" else np.uint32) + offset
                   for field, offset in (("mean", 10), ("squared_mean", 20), ("steps", 30))}
            for name, width in optimizer.dimensions.items()}
        optimizer._state = {name: {field: DeferredArray(values.copy())
                                  for field, values in group.items()}
                            for name, group in expected.items()}
        # A replay may own the same old buffers. remap must clear it first.
        optimizer._update_cache = {0: optimizer._state["radiance_sh"]["mean"]}
        optimizer._compact_update_cache = {0: optimizer._state["centers"]["mean"]}
        optimizer._update_model = object()
        optimizer._update_key = 123
        optimizer._update_clears = 0
        return optimizer, expected

    def assert_remapped(self, optimizer, before, survivors, count, *, reset_radiance=False):
        for name, group in optimizer._state.items():
            for field, result in group.items():
                width = 1 if field == "steps" else optimizer.dimensions[name]
                expected = np.zeros((count, width), dtype=before[name][field].dtype)
                if not (reset_radiance and name == "radiance_sh"):
                    expected[:len(survivors)] = before[name][field].reshape(-1, width)[survivors]
                np.testing.assert_array_equal(result.evaluate(), expected.reshape(-1),
                                              err_msg=f"{name}.{field}")
        self.assertEqual(optimizer.count, count)
        self.assertEqual(optimizer._update_clears, 1)
        self.assertFalse(optimizer._update_cache)
        self.assertFalse(optimizer._compact_update_cache)
        self.assertIsNone(optimizer._update_model)
        self.assertIsNone(optimizer._update_key)
        self.assertTrue(all(source() is None for source in optimizer.dr.sources))

    def test_reorder_and_append_release_each_old_moment_before_next_gather(self):
        optimizer, before = self.make_optimizer()
        optimizer.remap([3, 1], 5)
        self.assert_remapped(optimizer, before, [3, 1], 5)
        self.assertEqual(len(optimizer.dr.sources), 15)

    def test_radiance_reset_keeps_other_groups_and_step_counts(self):
        optimizer, before = self.make_optimizer()
        optimizer.remap([2, 0], 3, reset_radiance=True)
        self.assert_remapped(optimizer, before, [2, 0], 3, reset_radiance=True)
        self.assertEqual(len(optimizer.dr.sources), 12)

    def test_external_alias_remains_unchanged(self):
        optimizer, before = self.make_optimizer(check_release=False)
        alias = optimizer._state["radiance_sh"]["mean"]
        optimizer.remap([3, 0], 3)
        np.testing.assert_array_equal(alias.evaluate(), before["radiance_sh"]["mean"])
        self.assertIsNot(alias, optimizer._state["radiance_sh"]["mean"])
        del alias
        self.assert_remapped(optimizer, before, [3, 0], 3)

    def test_deletion_observe_releases_old_fields_and_preserves_evidence(self):
        self.check_observe(deletion=True)

    def test_growth_observe_keeps_fused_evaluation(self):
        self.check_observe(deletion=False)

    def check_observe(self, *, deletion):
        lifecycle = GPUTopologyLifecycle.__new__(GPUTopologyLifecycle)
        lifecycle.dr = DeferredDr(self, check_release=deletion)
        lifecycle.Float, lifecycle.UInt = np.float32, np.uint32
        lifecycle.Byte = np.uint8
        pending = dict(cells=np.arange(6, dtype=np.uint32), boundary=5,
                       resolution=1040, seen_views=set())
        # Seed every field so both ordinary accumulation and initialization
        # cannot accidentally pass as one another.
        for key in RGB_STAT_FIELDS:
            dtype = np.float32 if key.endswith("gain_sum") else np.uint32
            pending[key] = DeferredArray(np.ones(6, dtype=dtype),
                                         on_add=lifecycle.dr.track_source)
        prefix = "deletion" if deletion else "growth"
        manager = SimpleNamespace(
            count=6, resolution=1040, view_count=2,
            settings=SimpleNamespace(deletion_rgb_gain_epsilon=.5, growth_rgb_gain_epsilon=.5),
            pending_rgb_deletion=pending if deletion else None,
            pending_growth=None if deletion else pending,
            _rgb_deletion_due=lambda boundary: boundary == 5,
            _growth_due=lambda boundary: boundary == 5,
            _rgb_normalization=lambda value: "observed",
            _device_statistics={f"rgb_{prefix}_device_views": 0})
        lifecycle.manager = manager
        model = SimpleNamespace(active_voxel_count=6)
        out = {f"{prefix}_gain": np.array([4, -6, np.nan, 2, np.inf, -1], np.float32),
               f"{prefix}_observed": np.array([2, 3, 1, 0, 2, 1], np.uint32),
               f"{prefix}_unknown": np.array([0, 1, 0, 2, 0, 3], np.uint32)}
        with self.assertRaises(ValueError):
            lifecycle.observe(model, out, 0.5, 4, 100, deletion=deletion)
        self.assertEqual(pending["seen_views"], set())
        for view_id in (0, 0, 1):
            lifecycle.observe(model, out, view_id, 4, 100, deletion=deletion)
        expected_increment = {
            "raw_gain_sum": [2, -2, 0, 2, 0, -1],
            "gain_sum": [2, -2, 0, 0, 0, -1],
            "support_views": [1, 0, 0, 0, 0, 0],
            "conflict_views": [0, 1, 0, 0, 0, 1],
            "observed_views": [1, 1, 0, 0, 0, 1],
            "unknown_views": [0, 1, 0, 1, 0, 1],
            "invalid_views": [0, 0, 1, 0, 1, 0],
            "observed_rays": [2, 3, 1, 0, 2, 1],
            "unknown_rays": [0, 1, 0, 2, 0, 3]}
        for key, increment in expected_increment.items():
            np.testing.assert_array_equal(pending[key].evaluate(),
                                          1 + 2 * np.asarray(increment), err_msg=key)
        self.assertEqual(pending["seen_views"], {0, 1})
        self.assertEqual(manager._device_statistics[f"rgb_{prefix}_device_views"], 2)
        self.assertEqual(lifecycle.dr.eval_sizes,
                         [1] * (2 * len(RGB_STAT_FIELDS)) if deletion else [len(RGB_STAT_FIELDS)] * 2)
        self.assertEqual(len(lifecycle.dr.scatter_modes), 2 * len(RGB_STAT_FIELDS) if deletion else 0)
        self.assertTrue(all(source() is None for source in lifecycle.dr.sources))


if __name__ == "__main__":
    unittest.main()
