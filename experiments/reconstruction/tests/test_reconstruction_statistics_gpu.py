"""Deletion statistics: in-place ownership and exact old-expression parity."""

from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from reconstruction_common import initialize_runtime, restore_model
from reconstruction_deletion_rgb import RGB_DELETION_EVIDENCE_VERSION
from reconstruction_growth_rgb import RGB_STAT_FIELDS
from reconstruction_topology import CoarseToFineConfig, TopologyManager, TopologySettings
from reconstruction_topology_gpu import DELETION_VIEW_FIELDS, get_gpu_lifecycle
from test_reconstruction_growth_lookup import checkpoint
from preview import RuntimeConfig


class Statistics(unittest.TestCase):
    def fixture(self, count, views=3):
        model = restore_model(checkpoint(np.arange(count, dtype=np.uint32)), mi)
        settings = TopologySettings(gpu_topology=True, use_rgb_growth=True, use_rgb_deletion=True,
            growth_warmup_epochs=0, growth_fine_warmup_epochs=0, growth_interval=1,
            growth_rgb_normalization="observed", deletion_interval=1,
            deletion_warmup_epochs=0, deletion_protection_epochs=0,
            deletion_rgb_min_support_views=1, deletion_rgb_confirmation_windows=1)
        manager = TopologyManager(model, views, settings, CoarseToFineConfig(enabled=False, total_epochs=4))
        device = get_gpu_lifecycle(manager, model)
        self.assertIsNotNone(device)
        pending = dict(cells=dr.arange(dr.cuda.UInt32, count), boundary=1,
                       resolution=8, seen_views=set(), evidence_version=RGB_DELETION_EVIDENCE_VERSION)
        device.statistics(pending, deletion=True)
        manager.pending_rgb_deletion = pending
        return model, manager, pending

    def seed(self, pending, values):
        for name in RGB_STAT_FIELDS:
            pending[name] = type(pending[name])(values[name])
        dr.eval(*[pending[name] for name in RGB_STAT_FIELDS])

    def increments(self, out, epsilon):
        """The previous observe() expressions, before assignment was optimized."""
        F, U = dr.cuda.Float32, dr.cuda.UInt32
        gain, observed, unknown = F(out["deletion_gain"]), U(out["deletion_observed"]), U(out["deletion_unknown"])
        finite = dr.isfinite(gain)
        normalized = dr.select(finite, gain, 0) / F(dr.maximum(observed, 1))
        eligible, threshold = finite & (observed > 0), dr.opaque(F, epsilon)
        return {"raw_gain_sum": normalized, "gain_sum": dr.select(eligible, normalized, 0),
            "support_views": U(eligible & (normalized > threshold)),
            "conflict_views": U(eligible & (normalized < -threshold)),
            "observed_views": U(eligible), "unknown_views": U(unknown > 0),
            "invalid_views": U(~finite), "observed_rays": observed, "unknown_rays": unknown}

    def assert_bits_equal(self, actual, expected, name):
        actual, expected = np.array(actual), np.array(expected)
        if name.endswith("gain_sum"):
            actual, expected = actual.view(np.uint32), expected.view(np.uint32)
        np.testing.assert_array_equal(actual, expected, err_msg=name)

    def test_unique_owner_and_external_aliases(self):
        F, U = dr.cuda.Float32, dr.cuda.UInt32
        count = 4
        host = {name: np.arange(count, dtype=np.float32 if name.endswith("gain_sum") else np.uint32) + 3
                for name in RGB_STAT_FIELDS}
        out = dict(deletion_gain=F([1., -2., 3., 4.]), deletion_observed=U([1, 2, 3, 4]),
                   deletion_unknown=U([0, 1, 0, 2]))
        for alias_kind in ("none", "same_wrapper", "distinct_wrapper", "lazy_expression"):
            with self.subTest(alias_kind=alias_kind):
                model, manager, pending = self.fixture(count, views=256)
                self.seed(pending, host)
                before = {name: pending[name].index for name in RGB_STAT_FIELDS}
                if alias_kind == "same_wrapper":
                    aliases = {name: pending[name] for name in RGB_STAT_FIELDS}
                elif alias_kind == "distinct_wrapper":
                    aliases = {name: type(pending[name])(pending[name]) for name in RGB_STAT_FIELDS}
                elif alias_kind == "lazy_expression":
                    aliases = {name: pending[name] + dr.opaque(type(pending[name]), 7) for name in RGB_STAT_FIELDS}
                else:
                    aliases = {}
                increments = self.increments(out, manager.settings.deletion_rgb_gain_epsilon)
                expected = {name: type(pending[name])(host[name]) + increments[name] for name in RGB_STAT_FIELDS}
                dr.eval(*expected.values())
                manager.observe_rgb_deletion(model, out, 0, 0, 16)
                for name in RGB_STAT_FIELDS:
                    self.assert_bits_equal(pending[name], expected[name], name)
                    self.assertEqual(pending[name].index == before[name], alias_kind == "none", name)
                    if aliases:
                        expected_alias = host[name] + 7 if alias_kind == "lazy_expression" else host[name]
                        self.assert_bits_equal(aliases[name], expected_alias, name)

    def test_float_subnormal_cancellation_and_uint_overflow(self):
        F, U = dr.cuda.Float32, dr.cuda.UInt32
        tiny = np.nextafter(np.float32(0), np.float32(1))
        normal = np.finfo(np.float32).tiny
        above_normal = np.nextafter(normal, np.float32(1))
        old = np.array([tiny, -tiny, normal, above_normal, 1., -1., -0.], dtype=np.float32)
        gain = np.array([normal, normal, -above_normal, -normal, 2., -2., -0.], dtype=np.float32)
        count = len(old)
        model, manager, pending = self.fixture(count, views=256)
        host = {name: old.copy() if name.endswith("gain_sum") else np.full(count, 0xffffffff, dtype=np.uint32)
                for name in RGB_STAT_FIELDS}
        self.seed(pending, host)
        out = dict(deletion_gain=F(gain), deletion_observed=dr.full(U, 1, count),
                   deletion_unknown=U(np.arange(count, dtype=np.uint32)))
        increments = self.increments(out, manager.settings.deletion_rgb_gain_epsilon)
        expected = {name: type(pending[name])(host[name]) + increments[name] for name in RGB_STAT_FIELDS}
        dr.eval(*expected.values())
        manager.observe_rgb_deletion(model, out, 0, 0, 16)
        for name in RGB_STAT_FIELDS:
            self.assert_bits_equal(pending[name], expected[name], name)
        # Normal operands can cancel to a subnormal. Original add.ftz flushes
        # these to signed zero; NoConflicts float addition would retain them.
        bits = np.array(pending["raw_gain_sum"]).view(np.uint32)
        np.testing.assert_array_equal(bits[2:4], [0x80000000, 0])
        np.testing.assert_array_equal(np.array(pending["observed_rays"]), np.zeros(count, np.uint32))

    def test_two_views_match_old_expressions_with_masks_and_fast_math(self):
        F, U = dr.cuda.Float32, dr.cuda.UInt32
        rng, count = np.random.default_rng(917420), 257
        for fast_math in (True, False):
            with self.subTest(fast_math=fast_math), dr.scoped_set_flag(dr.JitFlag.FastMath, fast_math):
                model, manager, pending = self.fixture(count)
                for name in DELETION_VIEW_FIELDS:
                    self.assertIs(type(pending[name]), dr.cuda.UInt8)
                host = {name: rng.uniform(-20, 20, count).astype(np.float32) if name.endswith("gain_sum")
                        else rng.integers(0, 100, count, dtype=np.uint32) for name in RGB_STAT_FIELDS}
                self.seed(pending, host)
                reference = {name: (F if name.endswith("gain_sum") else U)(host[name])
                             for name in RGB_STAT_FIELDS}
                for view in range(2):
                    gain = rng.normal(size=count).astype(np.float32) * np.float32(7.3)
                    gain[:3] = [np.nan, np.inf, -np.inf]
                    observed = rng.integers(2, 100, count, dtype=np.uint32)
                    observed[3:8] = 0
                    out = dict(deletion_gain=F(gain), deletion_observed=U(observed),
                        deletion_unknown=U(rng.integers(0, 5, count, dtype=np.uint32)))
                    increments = self.increments(out, manager.settings.deletion_rgb_gain_epsilon)
                    reference = {name: reference[name] + increments[name] for name in RGB_STAT_FIELDS}
                    dr.eval(*reference.values())
                    manager.observe_rgb_deletion(model, out, view, 0, count)
                    for name in RGB_STAT_FIELDS:
                        self.assert_bits_equal(pending[name], reference[name], name)
                # The same view is idempotent, including all counters and gains.
                manager.observe_rgb_deletion(model, out, 1, 0, count)
                self.assertEqual(pending["seen_views"], {0, 1})
                for name in RGB_STAT_FIELDS:
                    self.assert_bits_equal(pending[name], reference[name], name)

    def test_byte_counts_reach_255_and_keep_checkpoint_schema(self):
        F, U = dr.cuda.Float32, dr.cuda.UInt32
        model, manager, pending = self.fixture(3, views=255)
        out = dict(deletion_gain=F([1., -1., np.nan]), deletion_observed=U([1, 1, 1]),
                   deletion_unknown=U([1, 1, 1]))
        for view in range(255):
            manager.observe_rgb_deletion(model, out, view, 0, 3)
        expected = dict(support_views=[255, 0, 0], conflict_views=[0, 255, 0],
                        observed_views=[255, 255, 0], unknown_views=[255, 255, 255],
                        invalid_views=[0, 0, 255])
        manager.observe_rgb_deletion(model, out, 254, 0, 3)
        for name, values in expected.items():
            self.assertIs(type(pending[name]), dr.cuda.UInt8)
            np.testing.assert_array_equal(np.array(pending[name]), values, err_msg=name)
        device = get_gpu_lifecycle(manager, model)
        manager.settings.deletion_rgb_min_support_views = 256
        manager.settings.deletion_rgb_max_conflict_views = 256
        selected = device.evaluate_rgb_deletion(1, {})
        np.testing.assert_array_equal(np.array(selected), [False, False, False])
        manager.pending_rgb_deletion = pending
        manager.settings.deletion_rgb_min_support_views = 255
        selected = device.evaluate_rgb_deletion(1, {})
        np.testing.assert_array_equal(np.array(selected), [True, False, False])
        manager.pending_rgb_deletion = pending
        saved = manager.state_dict()
        for name, values in expected.items():
            restored = saved["pending_rgb_deletion"][name]
            self.assertEqual(restored.dtype, np.dtype(np.uint32))
            np.testing.assert_array_equal(restored, values, err_msg=name)


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="mitsuba-statistics-device-") as directory:
        mi, dr, _, handles = initialize_runtime(RuntimeConfig(), Path(directory), {})
        unittest.main()
