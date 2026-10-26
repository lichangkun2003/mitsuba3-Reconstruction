"""CPU coverage for signed epoch storage bounds and checkpoint compatibility."""

from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reconstruction_topology import CoarseToFineConfig, TopologySettings
from reconstruction_topology_gpu import (
    DELETION_VIEW_FIELDS, EPOCH_FIELDS, GPUTopologyLifecycle, compact_epoch_fields_safe,
)


class NumPyDr:
    """Array casts only; constructing the real lifecycle needs no GPU runtime."""

    cuda = SimpleNamespace(Float32=np.float32, UInt32=np.uint32, UInt8=np.uint8)
    int64_array_t = staticmethod(lambda dtype: np.int64)
    int32_array_t = staticmethod(lambda dtype: np.int32)
    float64_array_t = staticmethod(lambda dtype: np.float64)
    mask_t = staticmethod(lambda dtype: np.bool_)
    eval = staticmethod(lambda *values: None)
    zeros = staticmethod(lambda dtype, count: np.zeros(count, dtype=dtype))
    arange = staticmethod(lambda dtype, count: np.arange(count, dtype=dtype))
    empty = staticmethod(lambda dtype, count: np.empty(count, dtype=dtype))
    is_array_v = staticmethod(lambda dtype: issubclass(dtype, np.ndarray))
    isfinite = staticmethod(np.isfinite)
    maximum = staticmethod(np.maximum)
    minimum = staticmethod(np.minimum)
    max = staticmethod(np.max)
    select = staticmethod(np.where)


class SharedInt32(np.ndarray):
    scalar = np.int32

    def __new__(cls, values):
        return np.asarray(values, dtype=cls.scalar).view(cls)

    @property
    def index(self):
        # Represent shared device storage independently of Python wrappers.
        return self.__array_interface__["data"][0]


class SharedUInt32(SharedInt32):
    scalar = np.uint32


class SharedBool(SharedInt32):
    scalar = np.bool_


class RemapDr:
    arange = staticmethod(lambda dtype, count: dtype(np.arange(count)))
    eval = staticmethod(lambda *values: None)

    @staticmethod
    def gather(dtype, value, indices, active):
        result = dtype(np.zeros(len(indices)))
        result[active] = value[indices[active]]
        return result

    @staticmethod
    def select(active, first, second):
        return type(first)(np.where(active, first, second))


def manager():
    return SimpleNamespace(
        np=np, schedule=CoarseToFineConfig(total_epochs=1500), settings=TopologySettings(),
        level_start_epoch=1050, opacity_warmup=20, opacity_ramp=30, count=3, view_count=2,
        growth_birth=np.array([0, 1061, 1051], np.int64),
        growth_wait_start=np.array([0, 1061, 1051], np.int64),
        deletion_eligible=np.array([0, 1065, 1055], np.int64),
        rgb_deletion_streak=np.array([0, 1, 2], np.uint32),
        low_opacity_deletion_streak=np.array([2, 1, 0], np.uint32),
        candidates=np.zeros(3, bool), cooldown={2: (1 << 31) + 50},
        pending_growth=None, pending_rgb_deletion=None)


def lifecycle(state):
    with patch.dict(sys.modules, drjit=NumPyDr, mitsuba=SimpleNamespace()):
        return GPUTopologyLifecycle(state)


class EpochStorageTests(unittest.TestCase):
    def test_ordinary_run_uses_signed_int32_and_preserves_protection(self):
        state = manager()
        device = lifecycle(state)
        for name in EPOCH_FIELDS:
            self.assertEqual(device.fields[name].dtype, np.dtype(np.int32))
        self.assertEqual(device.cooldown_expiry.dtype, np.dtype(np.int64))
        np.testing.assert_array_equal(device.cooldown_expiry, [(1 << 31) + 50])
        # A birth later than the current boundary must remain a negative age,
        # rather than underflowing to a large unsigned and becoming mature.
        np.testing.assert_array_equal(device.mature(1060), [True, False, False])
        np.testing.assert_array_equal(device.eligible(1060), [True, False, True])

    def test_exact_schedule_limit_reserves_arithmetic_headroom(self):
        state = manager()
        state.level_start_epoch = 0
        offsets = state.opacity_warmup + state.opacity_ramp + state.schedule.coarse_growth_interval
        offsets += sum(value for name, value in vars(state.settings).items()
                       if name.endswith("_epochs") or name.endswith("_interval"))
        state.schedule.total_epochs = (1 << 31) - 2 - offsets
        self.assertTrue(compact_epoch_fields_safe(state))
        state.schedule.total_epochs += 1
        self.assertFalse(compact_epoch_fields_safe(state))
        self.assertEqual(lifecycle(state).fields["growth_birth"].dtype, np.dtype(np.int64))

    def test_saved_epoch_limits_and_unusual_offsets_fall_back_without_truncation(self):
        for name in EPOCH_FIELDS:
            with self.subTest(field=name):
                state = manager()
                getattr(state, name)[0] = (1 << 31) - 1
                self.assertTrue(compact_epoch_fields_safe(state))
                getattr(state, name)[0] += 1
                device = lifecycle(state)
                self.assertEqual(device.fields[name].dtype, np.dtype(np.int64))
                self.assertEqual(int(device.fields[name][0]), 1 << 31)
                getattr(state, name)[0] = -1
                self.assertFalse(compact_epoch_fields_safe(state))
        for name in ("growth_protection_epochs", "deletion_cooldown_epochs", "deletion_protection_epochs"):
            with self.subTest(offset=name):
                state = manager()
                setattr(state.settings, name, 1 << 31)
                self.assertFalse(compact_epoch_fields_safe(state))
        state = manager()
        state.settings.growth_wait_epochs = .5
        self.assertFalse(compact_epoch_fields_safe(state))

    def test_materialize_restores_int64_schema_and_reload_keeps_values(self):
        state = manager()
        expected = {name: getattr(state, name).copy() for name in EPOCH_FIELDS}
        device = lifecycle(state)
        device.materialize()
        for name in EPOCH_FIELDS:
            self.assertEqual(getattr(state, name).dtype, np.dtype(np.int64))
            np.testing.assert_array_equal(getattr(state, name), expected[name])
        self.assertEqual(state.rgb_deletion_streak.dtype, np.dtype(np.uint32))
        self.assertEqual(state.cooldown, {2: (1 << 31) + 50})
        restored = lifecycle(state)
        for name in EPOCH_FIELDS:
            self.assertEqual(restored.fields[name].dtype, np.dtype(np.int32))
            np.testing.assert_array_equal(restored.fields[name], expected[name])

    def test_remap_preserves_only_identical_source_and_newborn_storage(self):
        state = manager()
        device = GPUTopologyLifecycle.__new__(GPUTopologyLifecycle)
        device.manager, device.dr = state, RemapDr
        device.UInt, device.Bool = SharedUInt32, SharedBool
        original = SharedInt32([0, 11, 6])
        device.fields = {
            "growth_birth": SharedInt32(original),
            "growth_wait_start": SharedInt32(original),
            "deletion_eligible": SharedInt32(original),
            "rgb_deletion_streak": SharedUInt32([0, 1, 2]),
            "low_opacity_deletion_streak": SharedUInt32([0, 1, 2])}
        device.candidates = SharedBool([False, True, True])
        device.remap(SharedUInt32([2, 0]), 4, 10)
        first, second = device.fields["growth_birth"], device.fields["growth_wait_start"]
        self.assertIsNot(first, second)
        self.assertEqual(first.index, second.index)
        np.testing.assert_array_equal(first, [6, 0, 11, 11])
        # Same source with a different newborn constant must stay independent.
        eligible = device.fields["deletion_eligible"]
        self.assertNotEqual(first.index, eligible.index)
        np.testing.assert_array_equal(eligible, [6, 0, 15, 15])
        # Equal contents are insufficient: no full-array equality is inferred.
        self.assertNotEqual(device.fields["rgb_deletion_streak"].index,
                            device.fields["low_opacity_deletion_streak"].index)
        np.testing.assert_array_equal(original, [0, 11, 6])
        device.remap(SharedUInt32([3, 1]), 3, 20)
        np.testing.assert_array_equal(first, [6, 0, 11, 11])
        np.testing.assert_array_equal(device.fields["growth_birth"], [11, 0, 21])
        self.assertEqual(device.fields["growth_birth"].index,
                         device.fields["growth_wait_start"].index)
        self.assertEqual(state.count, 3)

    def test_deletion_view_storage_boundary_and_growth_remains_uint32(self):
        for view_count in (1, 255, 256, 32767):
            with self.subTest(view_count=view_count):
                state = manager()
                state.view_count = view_count
                device = lifecycle(state)
                for deletion in (False, True):
                    pending = {"cells": np.arange(3, dtype=np.uint32)}
                    device.statistics(pending, deletion=deletion)
                    for name in DELETION_VIEW_FIELDS:
                        expected = np.uint8 if deletion and view_count <= 255 else np.uint32
                        self.assertEqual(pending[name].dtype, np.dtype(expected))
                    self.assertEqual(pending["gain_sum"].dtype, np.dtype(np.float32))
                    self.assertEqual(pending["observed_rays"].dtype, np.dtype(np.uint32))
                    self.assertEqual(pending["unknown_rays"].dtype, np.dtype(np.uint32))

    def test_compact_deletion_checkpoint_widens_counts_and_resumes_uint32(self):
        state = manager()
        state.view_count = 255
        device = lifecycle(state)
        pending = dict(cells=np.arange(3, dtype=np.uint32), seen_views=set(range(255)))
        device.statistics(pending, deletion=True)
        for name in DELETION_VIEW_FIELDS:
            pending[name][:] = [0, 169, 255]
        state.pending_rgb_deletion = pending
        device.materialize()
        for name in DELETION_VIEW_FIELDS:
            self.assertEqual(pending[name].dtype, np.dtype(np.uint32))
            np.testing.assert_array_equal(pending[name], [0, 169, 255])
        restored = lifecycle(state)
        for name in DELETION_VIEW_FIELDS:
            self.assertEqual(restored.manager.pending_rgb_deletion[name].dtype, np.dtype(np.uint32))
            np.testing.assert_array_equal(restored.manager.pending_rgb_deletion[name], [0, 169, 255])

    def test_compact_count_comparisons_accept_thresholds_larger_than_a_byte(self):
        for support_threshold, expected in ((256, [False, False, False]),
                                            (255, [True, True, False])):
            with self.subTest(support_threshold=support_threshold):
                state = manager()
                state.view_count, state.resolution = 255, 1040
                state._rgb_deletion_due = lambda boundary: True
                state.rgb_deletion_last_boundary = None
                state.growth_birth[:] = 0
                state.settings.deletion_rgb_min_support_views = support_threshold
                state.settings.deletion_rgb_max_conflict_views = 256
                state.settings.deletion_rgb_confirmation_windows = 1
                device = lifecycle(state)
                device.sum = lambda value: np.asarray(value).sum().item()
                device.count = lambda value: int(np.count_nonzero(value))
                pending = dict(cells=np.arange(3, dtype=np.uint32), seen_views=set(range(255)),
                               boundary=10, resolution=1040)
                device.statistics(pending, deletion=True)
                pending["observed_views"][:] = 255
                pending["support_views"][:] = [255, 255, 254]
                pending["conflict_views"][:] = 255
                pending["gain_sum"][:] = 255
                state.pending_rgb_deletion = pending
                np.testing.assert_array_equal(device.evaluate_rgb_deletion(10, {}), expected)


if __name__ == "__main__":
    unittest.main()
