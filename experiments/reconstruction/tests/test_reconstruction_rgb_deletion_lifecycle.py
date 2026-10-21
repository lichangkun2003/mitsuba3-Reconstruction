"""CPU reference/lifecycle checks for RGB-only, counterfactual voxel removal."""

from __future__ import annotations

import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import reconstruction_topology as topology
from reconstruction_deletion_rgb import (
    RGB_DELETION_EVIDENCE_VERSION, initialize_pending, observe_numpy)
from test_reconstruction_rgb_growth_lifecycle import Model, Optimizer, seed_checkpoint


class RGBDeletionStatisticsTests(unittest.TestCase):
    def test_signed_local_mean_uses_reliable_zero_negative_and_positive_hits(self):
        for pixels in (100, 640000):
            pending = initialize_pending(4, 30, 8, np)
            out = {
                "deletion_gain": np.asarray([2., -2., 0., 5.], np.float32),
                "deletion_observed": np.asarray([4, 4, 4, 0], np.uint32),
                "deletion_unknown": np.asarray([7, 7, 7, 2], np.uint32)}
            observe_numpy(pending, out, pixels, 1e-7, np)
            np.testing.assert_allclose(pending["gain_sum"], [.5, -.5, 0., 0.])
            np.testing.assert_array_equal(pending["support_views"], [1, 0, 0, 0])
            np.testing.assert_array_equal(pending["conflict_views"], [0, 1, 0, 0])
            np.testing.assert_array_equal(pending["observed_views"], [1, 1, 1, 0])
            np.testing.assert_array_equal(pending["unknown_views"], [1, 1, 1, 1])

    def test_shards_sum_raw_counts_before_one_view_vote(self):
        # Five rays split unevenly into 2+3; treating each shard as a view
        # would change the mean and vote count.
        first = {"deletion_gain": np.asarray([2.], np.float32),
                 "deletion_observed": np.asarray([2], np.uint32),
                 "deletion_unknown": np.asarray([1], np.uint32)}
        second = {"deletion_gain": np.asarray([-1.], np.float32),
                  "deletion_observed": np.asarray([3], np.uint32),
                  "deletion_unknown": np.asarray([0], np.uint32)}
        pending = initialize_pending(1, 30, 8, np)
        observe_numpy(pending, {key: first[key] + second[key] for key in first},
                      640000, 1e-7, np)
        np.testing.assert_allclose(pending["gain_sum"], [.2])
        np.testing.assert_array_equal(pending["observed_views"], [1])

    def test_nonfinite_gain_is_not_positive_evidence(self):
        pending = initialize_pending(3, 30, 8, np)
        observe_numpy(pending, {
            "deletion_gain": np.asarray([np.nan, np.inf, -np.inf], np.float32),
            "deletion_observed": np.ones(3, np.uint32),
            "deletion_unknown": np.zeros(3, np.uint32)}, 100, 1e-7, np)
        np.testing.assert_array_equal(pending["invalid_views"], [1, 1, 1])
        np.testing.assert_array_equal(pending["observed_views"], [0, 0, 0])
        np.testing.assert_array_equal(pending["gain_sum"], [0., 0., 0.])


class RGBDeletionLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.checkpoint = seed_checkpoint()
        self.model = Model(self.checkpoint)
        self.optimizer = Optimizer()
        self.snapshot = patch.object(
            topology, "model_checkpoint",
            side_effect=lambda model, fields=None: copy.deepcopy(model.checkpoint))
        self.snapshot.start()
        self.addCleanup(self.snapshot.stop)

    def manager(self, *, view_count=3, scheduled=False, **settings):
        defaults = dict(
            enable_deletion=True, use_rgb_deletion=True,
            enable_growth=False, use_growth_evidence=False,
            use_rgb_growth=False, deletion_freeze_last_epochs=100)
        defaults.update(settings)
        config = topology.TopologySettings(**defaults)
        schedule = topology.CoarseToFineConfig(
            enabled=scheduled, start_resolution=8, target_resolution=16,
            total_epochs=700, save_each_level=False)
        return topology.TopologyManager(self.model, view_count, config, schedule)

    def evidence(self, manager, *, gain=1., observed=1, unknown=0, target=0):
        out = {
            "deletion_gain": np.zeros(manager.count, np.float32),
            "deletion_observed": np.zeros(manager.count, np.uint32),
            "deletion_unknown": np.zeros(manager.count, np.uint32)}
        for key, value in (("gain", gain), ("observed", observed), ("unknown", unknown)):
            out[f"deletion_{key}"][target] = value
        return out

    def observe(self, manager, boundary, *, views=None, **evidence):
        out = self.evidence(manager, **evidence)
        for view in range(manager.view_count) if views is None else views:
            manager.observe_rgb_deletion(self.model, out, view, boundary - 1, 640000)

    def finish(self, manager, boundary):
        return manager.finish_epoch(self.model, self.optimizer, boundary)

    def test_schedule_has_per_level_warmup_and_global_final_freeze(self):
        manager = self.manager()
        for boundary in (1, 10, 20, 21, 29, 31, 599, 601, 610, 700):
            self.assertFalse(manager.collect_deletion(boundary - 1), boundary)
        for boundary in (30, 40, 600):
            self.assertTrue(manager.collect_deletion(boundary - 1), boundary)
        manager = self.manager(scheduled=True)
        self.assertTrue(manager.collect_deletion(29))  # Coarse evidence is enabled.
        self.assertFalse(manager.collect_deletion(419))  # Split boundary.
        manager.resolution = manager.core_resolution = 16
        manager.level_start_epoch = 420
        self.assertFalse(manager.collect_deletion(439))  # Fine local epoch 20.
        self.assertTrue(manager.collect_deletion(449))   # Fine local epoch 30.
        self.assertFalse(manager.collect_deletion(601))  # Global tail overrides.

    def test_two_complete_windows_remove_and_preserve_survivor_adam(self):
        manager = self.manager()
        self.observe(manager, 30, unknown=5)
        replacement, event = self.finish(manager, 30)
        self.assertIsNone(replacement)
        self.assertEqual(event["deleted"], 0)
        np.testing.assert_array_equal(manager.rgb_deletion_streak, [1, 0])
        self.assertEqual(event["deletion_rgb"]["awaiting_confirmation"], 1)
        self.observe(manager, 40, unknown=5)
        replacement, event = self.finish(manager, 40)
        self.assertEqual(event["deleted"], 1)
        self.assertEqual(event["deletion_rgb"]["confirmed"], 1)
        self.assertAlmostEqual(event["deletion_rgb"]["encountered_known_fraction"], 1 / 6)
        np.testing.assert_array_equal(replacement["model_state"]["cell_indices"], [75])
        np.testing.assert_array_equal(self.optimizer.remaps[0][0], [1])
        self.assertEqual(self.optimizer.remaps[0][2], {})  # No radiance/moment reset.
        self.assertEqual(manager.cooldown[73], 45)
        self.assertFalse(np.any(manager.candidates))
        self.assertFalse(np.any(manager.qualified))
        np.testing.assert_array_equal(manager.rgb_deletion_streak, [0])

    def test_duplicate_view_cannot_change_vote_or_confirmation(self):
        manager = self.manager()
        self.observe(manager, 30)
        self.observe(manager, 30, views=[0], gain=-100.)
        _, event = self.finish(manager, 30)
        self.assertEqual(event["deletion_rgb"]["support_votes"], 3)
        self.assertEqual(event["deletion_rgb"]["conflict_votes"], 0)
        self.assertEqual(event["deletion_rgb"]["views"], 3)

    def test_one_reliably_opposing_view_vetoes_three_supporting_views(self):
        manager = self.manager(view_count=4)
        for boundary in (30, 40):
            self.observe(manager, boundary, views=[0, 1, 2])
            self.observe(manager, boundary, views=[3], gain=-.01)
            replacement, event = self.finish(manager, boundary)
            self.assertIsNone(replacement)
            self.assertEqual(event["deleted"], 0)
            self.assertEqual(event["deletion_rgb"]["rejected_conflict"], 1)
            self.assertEqual(event["deletion_rgb"]["rgb_qualified"], 0)
            np.testing.assert_array_equal(manager.rgb_deletion_streak, [0, 0])

    def test_reliable_neutral_view_is_in_mean_unseen_unknown_only_view_is_not(self):
        for mean_threshold, expected in ((.07, 1), (.08, 0)):
            with self.subTest(mean_threshold=mean_threshold):
                manager = self.manager(view_count=5, deletion_rgb_min_mean_gain=mean_threshold)
                self.observe(manager, 30, views=[0, 1, 2], gain=.1)
                self.observe(manager, 30, views=[3], gain=0.)
                self.observe(manager, 30, views=[4], gain=0., observed=0, unknown=3)
                _, event = self.finish(manager, 30)
                self.assertEqual(event["deletion_rgb"]["rgb_qualified"], expected)
                self.assertAlmostEqual(event["deletion_rgb"]["mean_gain_sum"], .075, places=6)
                self.assertEqual(event["deletion_rgb"]["observed_views"], 4)

    def test_neutral_unknown_unseen_nonfinite_and_conflict_reset_streak(self):
        evidence = (
            {"gain": 0.}, {"gain": 0., "observed": 0, "unknown": 3},
            {"gain": 0., "observed": 0}, {"gain": np.nan}, {"gain": -.1})
        for values in evidence:
            with self.subTest(values=values):
                manager = self.manager()
                self.observe(manager, 30)
                self.finish(manager, 30)
                self.observe(manager, 40, **values)
                _, event = self.finish(manager, 40)
                self.assertEqual(event["deletion_rgb"]["streak_reset"], 1)
                np.testing.assert_array_equal(manager.rgb_deletion_streak, [0, 0])
                self.observe(manager, 50)
                _, event = self.finish(manager, 50)
                self.assertEqual(event["deleted"], 0)
                np.testing.assert_array_equal(manager.rgb_deletion_streak, [1, 0])

    def test_incomplete_or_missing_window_resets_confirmation(self):
        for missing in (False, True):
            with self.subTest(missing=missing):
                manager = self.manager()
                self.observe(manager, 30)
                self.finish(manager, 30)
                if not missing:
                    self.observe(manager, 40, views=[0, 1])
                _, event = self.finish(manager, 40)
                self.assertEqual(event["deletion_rgb"]["status"], "incomplete_views")
                self.assertEqual(event["deletion_rgb"]["streak_reset"], 1)
                self.observe(manager, 50)
                _, event = self.finish(manager, 50)
                self.assertEqual(event["deleted"], 0)
                self.assertEqual(event["deletion_rgb"]["awaiting_confirmation"], 1)

    def test_skipped_scheduled_boundary_cannot_count_as_consecutive(self):
        manager = self.manager()
        self.observe(manager, 30)
        self.finish(manager, 30)
        self.observe(manager, 50)
        _, event = self.finish(manager, 50)
        self.assertFalse(event["deletion_rgb"]["previous_window_consecutive"])
        self.assertEqual(event["deleted"], 0)

    def test_newborn_gets_twenty_full_training_epochs_before_qualification(self):
        manager = self.manager()
        manager.growth_birth[0] = 31  # Born after epoch 30.
        for boundary in (40, 50):
            self.observe(manager, boundary)
            _, event = self.finish(manager, boundary)
            self.assertEqual(event["deletion_rgb"]["protected_voxels"], 1)
            self.assertEqual(event["deletion_rgb"]["rgb_qualified"], 0)
        self.observe(manager, 60)
        _, event = self.finish(manager, 60)
        self.assertEqual(event["deletion_rgb"]["protected_voxels"], 0)
        self.assertEqual(event["deleted"], 0)
        self.observe(manager, 70)
        _, event = self.finish(manager, 70)
        self.assertEqual(event["deleted"], 1)

    def test_final_freeze_skips_evidence_and_removal_even_with_stale_pending(self):
        manager = self.manager()
        self.observe(manager, 600)
        manager.rgb_deletion_streak[0] = 1
        manager.rgb_deletion_last_boundary = 600
        manager.pending_rgb_deletion["boundary"] = 610
        manager.observe_rgb_deletion(self.model, {}, 0, 609, 640000)
        with patch.object(topology, "model_checkpoint",
                          side_effect=AssertionError("frozen deletion snapshot")):
            replacement, event = self.finish(manager, 610)
        self.assertIsNone(replacement)
        self.assertEqual(event["deleted"], 0)
        self.assertTrue(event["deletion_frozen"])
        self.assertFalse(event["deletion_evidence_requested"])
        self.assertIsNone(manager.pending_rgb_deletion)
        self.assertEqual(self.optimizer.remaps, [])

    def test_refinement_discards_pending_evidence_and_confirmation_streaks(self):
        manager = self.manager(scheduled=True)
        self.observe(manager, 30)
        self.finish(manager, 30)
        self.observe(manager, 40, views=[0])
        replacement, event = self.finish(manager, 420)
        self.assertTrue(event["refined"])
        self.assertEqual(event["deleted"], 0)
        self.assertEqual(manager.resolution, 16)
        self.assertEqual(manager.level_start_epoch, 420)
        self.assertIsNone(manager.pending_rgb_deletion)
        self.assertIsNone(manager.rgb_deletion_last_boundary)
        self.assertFalse(np.any(manager.rgb_deletion_streak))
        self.assertEqual(len(manager.rgb_deletion_streak),
                         len(replacement["model_state"]["cell_indices"]))

    def test_checkpoint_mid_window_restores_distinct_views_and_previous_streak(self):
        manager = self.manager()
        self.observe(manager, 30)
        self.finish(manager, 30)
        self.observe(manager, 40, views=[0])
        saved = manager.state_dict()
        self.assertEqual(saved["rgb_deletion_evidence_version"], RGB_DELETION_EVIDENCE_VERSION)
        resumed = self.manager()
        resumed.load_state_dict(saved)
        self.assertEqual(resumed.pending_rgb_deletion["seen_views"], {0})
        np.testing.assert_array_equal(resumed.rgb_deletion_streak, [1, 0])
        self.observe(resumed, 40, views=[0], gain=-100.)
        self.observe(resumed, 40, views=[1, 2])
        replacement, event = self.finish(resumed, 40)
        self.assertEqual(event["deleted"], 1)
        np.testing.assert_array_equal(replacement["model_state"]["cell_indices"], [75])

    def test_rgb_checkpoint_version_and_lifecycle_fields_are_required(self):
        saved = self.manager().state_dict()
        for key in ("rgb_deletion_evidence_version", "pending_rgb_deletion",
                    "rgb_deletion_streak", "rgb_deletion_last_boundary"):
            with self.subTest(key=key):
                bad = copy.deepcopy(saved)
                bad.pop(key)
                with self.assertRaisesRegex(ValueError, "RGB deletion"):
                    self.manager().load_state_dict(bad)
        self.observe((manager := self.manager()), 30, views=[0])
        bad = manager.state_dict()
        bad["pending_rgb_deletion"]["evidence_version"] = -1
        with self.assertRaisesRegex(ValueError, "pending evidence version"):
            self.manager().load_state_dict(bad)

    def test_legacy_checkpoint_still_selects_legacy_deletion(self):
        # This represents an existing RGB-growth-only run, whose new deletion
        # fields must not be filled from the latest training CONFIG.
        manager = self.manager(use_rgb_deletion=False, use_rgb_growth=True,
                               growth_rgb_normalization="observed")
        saved = manager.state_dict()
        for key in tuple(saved["settings"]):
            if key.startswith("deletion_rgb_") or key in (
                    "use_rgb_deletion", "deletion_warmup_epochs",
                    "deletion_protection_epochs", "deletion_freeze_last_epochs"):
                saved["settings"].pop(key)
        for key in ("rgb_deletion_evidence_version", "pending_rgb_deletion",
                    "rgb_deletion_streak", "rgb_deletion_last_boundary"):
            saved.pop(key)
        resumed = self.manager()
        resumed.load_state_dict(saved)
        self.assertFalse(resumed.settings.use_rgb_deletion)
        self.assertTrue(resumed.settings.use_rgb_growth)
        self.assertEqual(resumed.settings.deletion_freeze_last_epochs, 0)
        self.assertIsNone(resumed.pending_rgb_deletion)
        self.assertFalse(np.any(resumed.rgb_deletion_streak))

    def test_deleted_parent_cannot_generate_a_frozen_growth_proposal(self):
        manager = self.manager(
            enable_growth=True, use_rgb_growth=True, growth_warmup_epochs=0,
            growth_rgb_normalization="observed")
        manager.rgb_deletion_streak[0] = 1
        manager.rgb_deletion_last_boundary = 30
        self.assertTrue(manager.prepare_rgb_growth(self.model, 39))
        pending = manager.pending_growth
        selected = pending["cells"] == 74
        growth = {"growth_gain": selected.astype(np.float32),
                  "growth_observed": selected.astype(np.uint32),
                  "growth_unknown": np.zeros(len(selected), np.uint32)}
        for view in range(3):
            manager.observe_rgb_growth(self.model, growth, view, 39, 100)
        self.observe(manager, 40)
        replacement, event = self.finish(manager, 40)
        self.assertEqual(event["deleted"], 1)
        self.assertEqual(event["grown"], 0)
        self.assertEqual(event["growth_rgb"]["rgb_qualified"], 1)
        self.assertEqual(event["growth_rgb"]["geometry_revalidated"], 0)
        np.testing.assert_array_equal(replacement["model_state"]["cell_indices"], [75])

    def test_growth_parent_ids_are_remapped_after_unrelated_deletion(self):
        state = self.checkpoint["model_state"]
        for key, value in list(state.items()):
            rows = value.reshape(2, -1)
            state[key] = np.concatenate((rows[:1], rows)).reshape(-1).copy()
        state["cell_indices"][0] = 511  # Unrelated row before both growth parents.
        self.model = Model(self.checkpoint)
        manager = self.manager(
            enable_growth=True, use_rgb_growth=True, growth_warmup_epochs=0,
            growth_rgb_normalization="observed")
        manager.rgb_deletion_streak[0] = 1
        manager.rgb_deletion_last_boundary = 30
        self.assertTrue(manager.prepare_rgb_growth(self.model, 39))
        pending = manager.pending_growth
        selected = pending["cells"] == 74
        self.assertTrue(np.any(selected))
        growth = {"growth_gain": selected.astype(np.float32),
                  "growth_observed": selected.astype(np.uint32),
                  "growth_unknown": np.zeros(len(selected), np.uint32)}
        for view in range(3):
            manager.observe_rgb_growth(self.model, growth, view, 39, 100)
        self.observe(manager, 40)
        replacement, event = self.finish(manager, 40)
        self.assertEqual(event["deleted"], 1)
        self.assertEqual(event["grown"], 1)
        np.testing.assert_array_equal(replacement["model_state"]["cell_indices"], [73, 75, 74])
        np.testing.assert_array_equal(self.optimizer.remaps[0][0], [1, 2])
        np.testing.assert_array_equal(self.optimizer.remaps[1][0], [0, 1])
        self.assertEqual(self.optimizer.remaps[0][2], {})
        self.assertEqual(self.optimizer.remaps[1][2], {})
        self.assertEqual(manager.growth_birth[-1], 41)
        self.assertEqual(manager.rgb_deletion_streak[-1], 0)

    def test_invalid_rgb_configuration_is_rejected(self):
        for key, value in (
                ("use_rgb_deletion", 1), ("deletion_warmup_epochs", -.1),
                ("deletion_protection_epochs", True), ("deletion_freeze_last_epochs", -1),
                ("deletion_rgb_gain_epsilon", float("nan")),
                ("deletion_rgb_min_mean_gain", float("inf")),
                ("deletion_rgb_confirmation_windows", 0),
                ("deletion_rgb_max_conflict_views", 0), ("deletion_interval", 1.5)):
            with self.subTest(key=key, value=value):
                with self.assertRaises(ValueError):
                    self.manager(**{key: value})


if __name__ == "__main__":
    unittest.main(verbosity=2)
