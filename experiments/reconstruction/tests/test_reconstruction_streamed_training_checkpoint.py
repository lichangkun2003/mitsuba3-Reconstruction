"""CPU checks for complete streaming training snapshots and cache retirement."""

from __future__ import annotations

import copy
from dataclasses import asdict
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from reconstruction_checkpoint import load_checkpoint, save_checkpoint
from reconstruction_common import ReferenceImageCache
from reconstruction_deletion_rgb import initialize_pending
from reconstruction_export import ArraySource
from reconstruction_optimizer import OptimizerConfig
from reconstruction_topology import CoarseToFineConfig, TopologyManager, TopologySettings
from reconstruction_topology_gpu import DELETION_VIEW_FIELDS, EPOCH_FIELDS, GPUTopologyLifecycle
from test_reconstruction_epoch_storage import NumPyDr
from test_reconstruction_rgb_growth_lifecycle import Model, seed_checkpoint
from train import release_training_caches


def materialize(value):
    if isinstance(value, ArraySource):
        # Intentionally cross row boundaries in shaped pending arrays/masks.
        chunks = [value.read_chunk(start, min(5, value.size - start))
                  for start in range(0, value.size, 5)]
        return (np.concatenate(chunks) if chunks else np.empty(0, value.dtype)).reshape(value.shape)
    if isinstance(value, dict):
        return {key: materialize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(materialize(item) for item in value)
    return value


class StreamedTrainingCheckpointTests(unittest.TestCase):
    def assertStateEqual(self, actual, expected):
        if isinstance(expected, np.ndarray):
            self.assertEqual(actual.dtype, expected.dtype)
            self.assertEqual(actual.shape, expected.shape)
            np.testing.assert_array_equal(actual, expected)
        elif isinstance(expected, dict):
            self.assertEqual(set(actual), set(expected))
            for key in expected:
                with self.subTest(field=key):
                    self.assertStateEqual(actual[key], expected[key])
        elif isinstance(expected, (list, tuple)):
            self.assertEqual(type(actual), type(expected))
            self.assertEqual(len(actual), len(expected))
            for item, reference in zip(actual, expected):
                self.assertStateEqual(item, reference)
        else:
            self.assertEqual(actual, expected)

    def fixture(self):
        checkpoint = seed_checkpoint()
        model = Model(checkpoint)
        settings = TopologySettings(enable_deletion=True, enable_growth=True,
            use_rgb_growth=True, use_rgb_deletion=True, gpu_topology=False,
            growth_warmup_epochs=0, growth_interval=1)
        schedule = CoarseToFineConfig(enabled=False, total_epochs=5)
        manager = TopologyManager(model, 70, settings, schedule)
        with patch("reconstruction_topology.model_checkpoint", return_value=copy.deepcopy(checkpoint)):
            self.assertTrue(manager.prepare_rgb_growth(model, 0))
        manager.pending_growth["seen_views"] = {0, 65}
        manager.pending_rgb_deletion = initialize_pending(manager.count, 1, manager.resolution, np)
        manager.pending_rgb_deletion["seen_views"] = {0, 3}
        manager.cooldown = {74: 100, 82: (1 << 31) + 100}
        return checkpoint, model, manager

    def test_cpu_export_borrows_arrays_without_mutating_pending_windows(self):
        _, _, manager = self.fixture()
        expected = manager.state_dict()
        growth = manager.pending_growth
        seen = growth["seen_views"]
        with patch.object(manager, "state_dict", side_effect=AssertionError("eager snapshot")):
            source = manager.checkpoint_state()
        self.assertIs(source["growth_birth"], manager.growth_birth)
        self.assertIsNot(source["pending_growth"], growth)
        self.assertIs(manager.pending_growth, growth)
        self.assertIs(growth["seen_views"], seen)
        self.assertStateEqual(materialize(source), expected)

    def test_resident_export_keeps_device_owners_and_exact_portable_schema(self):
        checkpoint, model, manager = self.fixture()
        stale = manager.growth_birth
        with patch.dict(sys.modules, drjit=NumPyDr, mitsuba=SimpleNamespace()):
            device = GPUTopologyLifecycle(manager)
        manager._gpu_lifecycle = device
        device.fields["growth_birth"] = np.array([1, 2], np.int32)
        device.fields["rgb_deletion_streak"] = np.array([1, 0], np.uint32)
        for name in DELETION_VIEW_FIELDS:
            manager.pending_rgb_deletion[name] = np.array([2, 1], np.uint8)
        growth = manager.pending_growth
        with patch.object(device, "materialize", side_effect=AssertionError("full lifecycle readback")):
            source = manager.checkpoint_state()
            exported = materialize(source)
            self.assertIs(manager._gpu_lifecycle, device)
            self.assertIs(manager.growth_birth, stale)
            self.assertIs(manager.pending_growth, growth)
            self.assertIsInstance(growth["seen_views"], set)
            for name in EPOCH_FIELDS:
                self.assertEqual(exported[name].dtype, np.dtype(np.int64))
            for name in DELETION_VIEW_FIELDS:
                self.assertEqual(exported["pending_rgb_deletion"][name].dtype, np.dtype(np.uint32))
                self.assertEqual(manager.pending_rgb_deletion[name].dtype, np.dtype(np.uint8))
            self.assertIsInstance(source["foreground_masks"], ArraySource)
            self.assertEqual(source["foreground_masks"].shape, (2, 3))
            rng = np.random.default_rng(23)
            rng.random(11)
            dimensions = dict(centers=3, log_scales=3, rotations=3, radiance_sh=12, opacity_sh=1)
            optimizer = {"config": asdict(OptimizerConfig()), "count": 2,
                         "dimensions": dimensions, "radiance_degree_steps": True, "groups": {}}
            for name, width in dimensions.items():
                mean = np.arange(2 * width, dtype=np.float32) * .001
                optimizer["groups"][name] = dict(mean=mean, square=mean ** 2,
                    steps=np.array([3, 0, 4, 1] if name == "radiance_sh" else [3, 4], np.uint32))
            with tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / "complete.npz"
                save_checkpoint(path, checkpoint["model_state"], checkpoint["model_metadata"],
                    training_state={"topology_state": source, "rng_state": rng.bit_generator.state,
                                    "completed_epochs": 1, "pending_epoch_transition": True},
                    optimizer_state=optimizer)
                restored = load_checkpoint(path)
            self.assertStateEqual(restored["training_state"]["topology_state"], exported)
            self.assertStateEqual(restored["optimizer_state"], optimizer)
            resumed_rng = np.random.default_rng()
            resumed_rng.bit_generator.state = restored["training_state"]["rng_state"]
            np.testing.assert_array_equal(resumed_rng.random(20), rng.random(20))
            self.assertTrue(restored["training_state"]["pending_epoch_transition"])
            resumed = TopologyManager(model, 70, manager.settings, manager.schedule)
            resumed.load_state_dict(restored["training_state"]["topology_state"])
            self.assertStateEqual(resumed.state_dict(), exported)
        # The previous eager API remains an independent schema oracle.
        self.assertStateEqual(exported, manager.state_dict())

    def test_legacy_devices_export_fresh_evidence_without_materializing(self):
        _, _, manager = self.fixture()
        growth, deletion = manager.pending_growth, manager.pending_rgb_deletion
        growth_counts = np.full(len(growth["cells"]), 9, np.uint32)
        growth_device = SimpleNamespace(dr=NumPyDr, arrays={"support_views": growth_counts},
                                       materialize=lambda: self.fail("eager growth"))
        deletion_device = SimpleNamespace(dr=NumPyDr, arrays={"support_views": np.array([4, 5], np.uint32)},
                                         materialize=lambda: self.fail("eager deletion"))
        manager._rgb_growth_device = (growth, growth_device)
        manager._rgb_deletion_device = (deletion, deletion_device)
        mask_device = SimpleNamespace(count=2, dr=NumPyDr, is_cleared=False,
            foreground=[np.array([1, 2], np.uint32), np.array([3, 4], np.uint32), np.array([5, 6], np.uint32)],
            background=[np.array([7, 8], np.uint32), np.array([9, 10], np.uint32), np.array([11, 12], np.uint32)],
            materialize=lambda: self.fail("eager masks"))
        manager._deletion_device = mask_device
        state = materialize(manager.checkpoint_state())
        np.testing.assert_array_equal(state["foreground_masks"], [[1, 3, 5], [2, 4, 6]])
        np.testing.assert_array_equal(state["background_masks"], [[7, 9, 11], [8, 10, 12]])
        np.testing.assert_array_equal(state["pending_growth"]["support_views"], growth_counts)
        np.testing.assert_array_equal(state["pending_rgb_deletion"]["support_views"], [4, 5])
        self.assertFalse(np.any(growth["support_views"]))
        self.assertFalse(np.any(deletion["support_views"]))
        self.assertIs(manager._deletion_device, mask_device)
        self.assertIs(manager._rgb_growth_device[1], growth_device)
        mask_device.is_cleared = True
        self.assertFalse(np.any(materialize(manager.checkpoint_state())["foreground_masks"]))
        # The older alpha-based proposal device has separate counters instead
        # of an arrays dict; its host pending counters can be equally stale.
        legacy = dict(cells=np.arange(4, dtype=np.uint32), seen_views={2},
                      foreground=np.zeros(4, np.uint32), background=np.zeros(4, np.uint32))
        manager.pending_growth = legacy
        legacy_device = SimpleNamespace(dr=NumPyDr, foreground=np.array([3, 1, 2, 0], np.uint32),
                                        background=np.array([0, 1, 0, 2], np.uint32),
                                        materialize=lambda: self.fail("eager legacy growth"))
        manager._growth_device = (legacy, legacy_device)
        exported = materialize(manager.checkpoint_state())["pending_growth"]
        self.assertEqual(exported["foreground"].shape, (4,))
        np.testing.assert_array_equal(exported["foreground"], [3, 1, 2, 0])
        np.testing.assert_array_equal(exported["background"], [0, 1, 0, 2])
        self.assertFalse(np.any(legacy["foreground"]))
        self.assertIs(manager._growth_device[1], legacy_device)

    def test_failed_stream_write_preserves_live_lifecycle(self):
        checkpoint, _, manager = self.fixture()
        source = manager.checkpoint_state()
        source["growth_birth"] = ArraySource((manager.count,), np.int64,
            lambda *_: (_ for _ in ()).throw(OSError("readback failed")))
        pending = manager.pending_growth
        expected = manager.state_dict()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "model.npz"
            path.write_bytes(b"previous successful checkpoint")
            with self.assertRaisesRegex(OSError, "readback failed"):
                save_checkpoint(path, checkpoint["model_state"], checkpoint["model_metadata"],
                                training_state={"topology_state": source})
            self.assertEqual(path.read_bytes(), b"previous successful checkpoint")
        self.assertIs(manager.pending_growth, pending)
        self.assertStateEqual(manager.state_dict(), expected)

    def test_final_cache_retirement_preserves_budgets_and_counters(self):
        cache = ReferenceImageCache(2, 1, None, np, 64, rgb_only=True)
        cache._images["first"] = np.ones((1, 2, 3), np.float32)
        cache.resident_bytes = cache._images["first"].nbytes
        cache.hits, cache.misses = 13, 4
        order = []

        class Replay:
            def statistics(self):
                return {"reused": 17}

            def clear(self):
                order.append("clear")

        optimizer = SimpleNamespace(clear_replay=lambda: order.append("optimizer"))
        dr = SimpleNamespace(sync_thread=lambda: order.append("sync"),
                             flush_malloc_cache=lambda: order.append("flush"))
        result = release_training_caches(cache, Replay(), Replay(), optimizer, None, dr)
        self.assertEqual(result["reference_cache"]["resident_bytes"], 24)
        self.assertEqual(cache.statistics(), {"capacity_bytes": 64 * 1024 * 1024,
            "resident_bytes": 0, "images": 0, "hits": 13, "misses": 4, "evictions": 0})
        self.assertEqual(order, ["clear", "clear", "optimizer", "sync", "flush"])


if __name__ == "__main__":
    unittest.main()
