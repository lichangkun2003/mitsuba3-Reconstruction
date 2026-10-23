"""CPU lifecycle checks for persistent DC-opacity pruning, without rendering."""

from __future__ import annotations

import copy
import math
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import reconstruction_topology as topology
import train
import test_reconstruction_rgb_deletion_lifecycle as rgb_lifecycle
from test_reconstruction_rgb_growth_lifecycle import Model


class SnapshotModel(Model):
    """Expose the same opacity payload used by the native model's snapshot."""

    def __init__(self, checkpoint):
        super().__init__(checkpoint)
        self.opacity_sh_count = checkpoint["model_metadata"]["opacity_sh_count"]

    def snapshot(self):
        return self.checkpoint["model_state"]


class LowOpacityDeletionTests(unittest.TestCase):
    evidence = rgb_lifecycle.RGBDeletionLifecycleTests.evidence
    observe = rgb_lifecycle.RGBDeletionLifecycleTests.observe
    finish = rgb_lifecycle.RGBDeletionLifecycleTests.finish

    def setUp(self):
        rgb_lifecycle.RGBDeletionLifecycleTests.setUp(self)
        self.model = SnapshotModel(self.checkpoint)
        self.opacity([.001, .01])

    def opacity(self, values):
        alpha = np.asarray(values, np.float64)
        self.checkpoint["model_state"]["opacity_sh"] = (
            np.log(alpha / (1. - alpha)) * (2. * math.sqrt(math.pi))).astype(np.float32)

    def manager(self, **settings):
        return rgb_lifecycle.RGBDeletionLifecycleTests.manager(
            self, **{"use_low_opacity_deletion": True, **settings})

    def test_unknown_rgb_still_prunes_persistent_low_alpha_and_preserves_survivor(self):
        manager = self.manager()
        for boundary in (30, 40):
            self.observe(manager, boundary, gain=0., observed=0, unknown=5)
            replacement, event = self.finish(manager, boundary)
            self.assertEqual(event["deletion_rgb"]["confirmed"], 0)
            self.assertEqual(event["deletion_opacity"]["low_opacity_voxels"], 1)
            self.assertEqual(event["deletion_opacity"]["threshold"], .005)
            if boundary == 30:
                self.assertIsNone(replacement)
                np.testing.assert_array_equal(manager.low_opacity_deletion_streak, [1, 0])
            else:
                self.assertEqual(event["deleted"], 1)
                self.assertEqual(event["deletion_opacity"]["deleted"], 1)
                self.assertEqual(event["deletion_rgb"]["deleted"], 0)
                for name, old in self.checkpoint["model_state"].items():
                    expected = old.reshape(2, -1)[1:].ravel()
                    np.testing.assert_array_equal(replacement["model_state"][name], expected)
                np.testing.assert_array_equal(self.optimizer.remaps[0][0], [1])
                self.assertEqual(self.optimizer.remaps[0][2], {})
                self.assertEqual(manager.cooldown[73], 45)
                np.testing.assert_array_equal(manager.low_opacity_deletion_streak, [0])

    def test_actual_sigmoid_alpha_not_raw_sh_coefficient_controls_deletion(self):
        manager = self.manager(use_rgb_deletion=False)
        # Both raw coefficients are negative, but .01 is above the .005
        # threshold. Zero b0 is alpha=.5, not a transparent primitive.
        self.assertTrue(np.all(self.checkpoint["model_state"]["opacity_sh"] < .005))
        self.checkpoint["model_state"]["opacity_sh"][0] = 0.
        for boundary in (30, 40):
            replacement, event = self.finish(manager, boundary)
            self.assertIsNone(replacement)
            self.assertEqual(event["deletion_opacity"]["low_opacity_voxels"], 0)
            np.testing.assert_array_equal(manager.low_opacity_deletion_streak, [0, 0])

    def test_radial_far_shell_still_allows_low_opacity_deletion(self):
        # In an 8^3 grid with a radius-two core, both fixture cells (73,75)
        # are outside the core. The no-growth rule must not protect them from
        # ordinary persistent-opacity deletion.
        self.checkpoint["model_metadata"].update(
            core_resolution=4, scene_mapping="radial_seed_shell")
        self.model.core_resolution = 4
        self.model.scene_mapping = "radial_seed_shell"
        manager = self.manager(use_rgb_deletion=False)
        self.finish(manager, 30)
        replacement, event = self.finish(manager, 40)
        self.assertEqual(event["deleted"], 1)
        self.assertEqual(replacement["model_metadata"]["scene_mapping"], "radial_seed_shell")
        np.testing.assert_array_equal(replacement["model_state"]["cell_indices"], [75])

    def test_recovery_resets_streak_and_requires_two_new_low_windows(self):
        manager = self.manager()
        self.finish(manager, 30)
        self.opacity([.02, .01])
        _, event = self.finish(manager, 40)
        self.assertEqual(event["deletion_opacity"]["streak_reset"], 1)
        np.testing.assert_array_equal(manager.low_opacity_deletion_streak, [0, 0])
        self.opacity([.001, .01])
        self.assertIsNone(self.finish(manager, 50)[0])
        self.assertEqual(self.finish(manager, 60)[1]["deleted"], 1)

    def test_skipped_check_breaks_consecutive_confirmation(self):
        manager = self.manager()
        self.finish(manager, 30)
        replacement, event = self.finish(manager, 50)
        self.assertIsNone(replacement)
        self.assertFalse(event["deletion_opacity"]["previous_window_consecutive"])
        np.testing.assert_array_equal(manager.low_opacity_deletion_streak, [1, 0])
        self.assertEqual(self.finish(manager, 60)[1]["deleted"], 1)

    def test_newborn_protection_does_not_accumulate_confirmation(self):
        manager = self.manager()
        manager.growth_birth[0] = 31
        for boundary in (40, 50):
            replacement, event = self.finish(manager, boundary)
            self.assertIsNone(replacement)
            self.assertEqual(event["deletion_opacity"]["protected_voxels"], 1)
            np.testing.assert_array_equal(manager.low_opacity_deletion_streak, [0, 0])
        self.assertIsNone(self.finish(manager, 60)[0])
        self.assertEqual(self.finish(manager, 70)[1]["deleted"], 1)

    def test_disabled_warmup_ordinary_epochs_and_final_tail_do_not_read_opacity(self):
        for settings, boundaries in (
                ({"enable_deletion": False}, (30, 40)),
                ({"use_low_opacity_deletion": False}, (30, 40)),
                ({}, (1, 10, 20, 21, 29, 31, 610, 700))):
            with self.subTest(settings=settings):
                manager = self.manager(**settings)
                with patch.object(self.model, "snapshot",
                                  side_effect=AssertionError("unneeded opacity readback")):
                    for boundary in boundaries:
                        replacement, event = self.finish(manager, boundary)
                        self.assertIsNone(replacement)
                        self.assertEqual(event["deleted"], 0)

    def test_rgb_and_opacity_union_deletes_each_voxel_only_once(self):
        manager = self.manager()
        for boundary in (30, 40):
            self.observe(manager, boundary)
            replacement, event = self.finish(manager, boundary)
        self.assertEqual(event["deleted"], 1)
        self.assertEqual(event["deletion_rgb"]["deleted"], 1)
        self.assertEqual(event["deletion_opacity"]["deleted"], 1)
        self.assertEqual(len(self.optimizer.remaps), 1)
        np.testing.assert_array_equal(replacement["model_state"]["cell_indices"], [75])

    def test_refinement_resets_streaks_and_restarts_per_level_warmup(self):
        manager = self.manager(scheduled=True)
        self.finish(manager, 30)
        replacement, event = self.finish(manager, 420)
        self.assertTrue(event["refined"])
        self.assertIsNone(manager.low_opacity_deletion_last_boundary)
        self.assertFalse(np.any(manager.low_opacity_deletion_streak))
        self.assertEqual(len(manager.low_opacity_deletion_streak),
                         len(replacement["model_state"]["cell_indices"]))
        self.model = SnapshotModel(replacement)
        with patch.object(self.model, "snapshot", side_effect=AssertionError("fine warmup readback")):
            self.assertIsNone(self.finish(manager, 440)[0])
        replacement, event = self.finish(manager, 450)
        self.assertIsNone(replacement)
        self.assertEqual(manager.low_opacity_deletion_streak.max(), 1)

    def test_checkpoint_preserves_streak_and_old_checkpoint_keeps_feature_off(self):
        manager = self.manager()
        self.finish(manager, 30)
        saved = manager.state_dict()
        resumed = self.manager()
        resumed.load_state_dict(saved)
        np.testing.assert_array_equal(resumed.low_opacity_deletion_streak, [1, 0])
        self.assertEqual(resumed.low_opacity_deletion_last_boundary, 30)
        self.assertEqual(self.finish(resumed, 40)[1]["deleted"], 1)
        legacy = copy.deepcopy(saved)
        for key in ("use_low_opacity_deletion", "deletion_opacity_threshold",
                    "deletion_opacity_confirmation_windows"):
            legacy["settings"].pop(key)
        for key in ("low_opacity_deletion_streak", "low_opacity_deletion_last_boundary"):
            legacy.pop(key)
        resumed = self.manager()
        resumed.load_state_dict(legacy)
        self.assertFalse(resumed.settings.use_low_opacity_deletion)
        self.assertFalse(np.any(resumed.low_opacity_deletion_streak))
        self.assertIsNone(self.finish(resumed, 40)[0])

    def test_enabled_checkpoint_requires_complete_valid_opacity_lifecycle(self):
        manager = self.manager()
        self.finish(manager, 30)
        saved = manager.state_dict()
        for key in ("low_opacity_deletion_streak", "low_opacity_deletion_last_boundary"):
            bad = copy.deepcopy(saved)
            bad.pop(key)
            with self.subTest(missing=key), self.assertRaises(ValueError):
                self.manager().load_state_dict(bad)
        for streak, previous in (([1], 30), ([-1, 0], 30), ([.5, 0], 30),
                                 ([1, 0], None), ([1, 0], 31), ([3, 0], 30)):
            bad = copy.deepcopy(saved)
            bad["low_opacity_deletion_streak"] = np.asarray(streak)
            bad["low_opacity_deletion_last_boundary"] = previous
            with self.subTest(streak=streak, previous=previous), self.assertRaises(ValueError):
                self.manager().load_state_dict(bad)

    def test_simultaneous_unrelated_removal_and_growth_remap_all_lifecycle_rows(self):
        state = self.checkpoint["model_state"]
        for key, value in list(state.items()):
            rows = value.reshape(2, -1)
            state[key] = np.concatenate((rows[:1], rows)).reshape(-1).copy()
        state["cell_indices"][0] = 511
        self.opacity([.001, .5, .5])
        self.model = SnapshotModel(self.checkpoint)
        manager = self.manager(
            enable_growth=True, use_rgb_growth=True, growth_warmup_epochs=0,
            growth_rgb_normalization="observed")
        self.finish(manager, 30)
        self.assertTrue(manager.prepare_rgb_growth(self.model, 39))
        pending = manager.pending_growth
        selected = pending["cells"] == 74
        self.assertTrue(np.any(selected))
        growth = {"growth_gain": selected.astype(np.float32),
                  "growth_observed": selected.astype(np.uint32),
                  "growth_unknown": np.zeros(len(selected), np.uint32)}
        for view in range(3):
            manager.observe_rgb_growth(self.model, growth, view, 39, 100)
        replacement, event = self.finish(manager, 40)
        self.assertEqual((event["deleted"], event["grown"]), (1, 1))
        np.testing.assert_array_equal(replacement["model_state"]["cell_indices"], [73, 75, 74])
        np.testing.assert_array_equal(self.optimizer.remaps[0][0], [1, 2])
        np.testing.assert_array_equal(self.optimizer.remaps[1][0], [0, 1])
        np.testing.assert_array_equal(manager.low_opacity_deletion_streak, [0, 0, 0])
        self.assertEqual(manager.growth_birth[-1], 41)

    def test_anisotropic_opacity_rejected_when_pruning_enabled(self):
        self.model.opacity_sh_count = self.checkpoint["model_metadata"]["opacity_sh_count"] = 4
        with self.assertRaisesRegex(ValueError, "opacity_sh_count"):
            self.manager()
        self.manager(enable_deletion=False)
        self.manager(use_low_opacity_deletion=False)

    def test_nonfinite_opacity_is_an_error_not_silent_removal(self):
        for value in (np.nan, np.inf, -np.inf):
            with self.subTest(value=value):
                self.checkpoint["model_state"]["opacity_sh"][0] = value
                manager = self.manager()
                with self.assertRaisesRegex(ValueError, "finite"):
                    self.finish(manager, 30)
                self.assertEqual(self.optimizer.remaps, [])

    def test_configuration_validation_and_historical_default(self):
        self.assertFalse(topology.TopologySettings().use_low_opacity_deletion)
        self.assertTrue(train.CONFIG.topology.use_low_opacity_deletion)
        for key, values in (
                ("use_low_opacity_deletion", (1, "true")),
                ("deletion_opacity_threshold", (-.1, 0., 1., True, float("nan"))),
                ("deletion_opacity_confirmation_windows", (0, -1, 1.5, True))):
            for value in values:
                with self.subTest(key=key, value=value):
                    config = copy.deepcopy(train.CONFIG)
                    setattr(config.topology, key, value)
                    with self.assertRaisesRegex(ValueError, key):
                        train.validate(config)
                    with self.assertRaises(ValueError):
                        self.manager(**{key: value})


if __name__ == "__main__":
    unittest.main(verbosity=2)
