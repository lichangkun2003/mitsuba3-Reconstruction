"""CPU lifecycle regression for the final, global growth-free training window."""

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
from test_reconstruction_rgb_growth_lifecycle import Model, Optimizer, seed_checkpoint


class GrowthTailFreezeTests(unittest.TestCase):
    def setUp(self):
        self.checkpoint = seed_checkpoint()
        self.model = Model(self.checkpoint)
        self.optimizer = Optimizer()
        self.snapshot = patch.object(
            topology, "model_checkpoint",
            side_effect=lambda model, fields=None: copy.deepcopy(model.checkpoint))
        self.snapshot.start()
        self.addCleanup(self.snapshot.stop)

    def manager(self, *, total=700, freeze=100, rgb=True, scheduled=False):
        settings = topology.TopologySettings(
            enable_deletion=False, enable_growth=True,
            use_growth_evidence=False, use_rgb_growth=rgb,
            growth_warmup_epochs=0, growth_fine_warmup_epochs=0,
            growth_interval=1, growth_freeze_last_epochs=freeze,
            growth_rgb_normalization="observed")
        schedule = topology.CoarseToFineConfig(
            enabled=scheduled, start_resolution=8, target_resolution=16,
            total_epochs=total, coarse_growth_interval=1,
            save_each_level=False)
        return topology.TopologyManager(
            self.model, 3, settings, schedule, opacity_warmup=0, opacity_ramp=0)

    def positive_evidence(self, pending):
        selected = pending["cells"] == 74
        return {
            "growth_gain": selected.astype(np.float32),
            "growth_observed": selected.astype(np.uint32),
            "growth_unknown": np.zeros(len(selected), np.uint32)}

    def test_700_epochs_last_100_start_at_601_in_both_modes(self):
        for rgb in (True, False):
            with self.subTest(rgb=rgb):
                manager = self.manager(rgb=rgb)
                self.assertFalse(manager.growth_frozen(600))
                self.assertTrue(manager._growth_due(600))
                for boundary in (601, 650, 700):
                    self.assertTrue(manager.growth_frozen(boundary))
                    self.assertFalse(manager._growth_due(boundary))

    def test_final_window_does_not_restart_at_coarse_to_fine_boundary(self):
        for rgb in (True, False):
            with self.subTest(rgb=rgb):
                manager = self.manager(rgb=rgb, scheduled=True)
                self.assertEqual(manager.budgets, {8: 302, 16: 398})
                # The end of the coarse level is not the end of training.
                self.assertTrue(manager._growth_due(301))
                self.assertFalse(manager.growth_frozen(302))
                self.assertFalse(manager._growth_due(302))  # Existing refine boundary.
                manager.resolution = 16
                manager.level_start_epoch = 302
                self.assertTrue(manager._growth_due(600))
                self.assertFalse(manager._growth_due(601))
                self.assertTrue(manager.growth_frozen(601))

    def test_zero_disables_freeze_but_keeps_existing_last_epoch_behavior(self):
        for rgb in (True, False):
            with self.subTest(rgb=rgb):
                manager = self.manager(rgb=rgb, freeze=0)
                self.assertFalse(manager.growth_frozen(699))
                self.assertTrue(manager._growth_due(699))
                self.assertFalse(manager.growth_frozen(700))
                self.assertFalse(manager._growth_due(700))

    def test_window_at_least_total_disables_growth_from_first_epoch(self):
        for freeze in (700, 701, 1000):
            for rgb in (True, False):
                with self.subTest(freeze=freeze, rgb=rgb):
                    manager = self.manager(rgb=rgb, freeze=freeze)
                    for boundary in (1, 300, 700):
                        self.assertTrue(manager.growth_frozen(boundary))
                        self.assertFalse(manager._growth_due(boundary))

    def test_invalid_window_values_are_rejected(self):
        for value in (-1, 1.5, "100", True, None):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "growth_freeze_last_epochs"):
                    self.manager(freeze=value)

    def test_last_allowed_boundary_can_commit_reliable_candidates(self):
        manager = self.manager()
        # prepare_rgb_growth receives a zero-based epoch.
        self.assertTrue(manager.prepare_rgb_growth(self.model, 599))
        pending = manager.pending_growth
        self.assertEqual(pending["boundary"], 600)
        for view in range(3):
            manager.observe_rgb_growth(
                self.model, self.positive_evidence(pending), view, 599, 100)
        replacement, event = manager.finish_epoch(self.model, self.optimizer, 600)
        self.assertEqual(event["grown"], 1)
        self.assertFalse(event["growth_frozen"])
        self.assertEqual(event["growth_freeze_last_epochs"], 100)
        self.assertEqual(replacement["model_state"]["cell_indices"][-1], 74)

    def test_first_frozen_epoch_clears_overlay_without_geometry_or_snapshot(self):
        manager = self.manager()
        self.assertTrue(manager.prepare_rgb_growth(self.model, 599))
        self.assertIsNotNone(self.model.overlay)
        with patch.object(manager, "_generate_growth",
                          side_effect=AssertionError("frozen candidate generation")):
            with patch.object(topology, "model_checkpoint",
                              side_effect=AssertionError("frozen snapshot")):
                self.assertFalse(manager.prepare_rgb_growth(self.model, 600))
        self.assertIsNone(manager.pending_growth)
        self.assertIsNone(manager._rgb_growth_device)
        self.assertIsNone(self.model.overlay)
        self.assertFalse(manager.needs_rendered_alpha(600))

    def test_stale_rgb_evidence_cannot_be_observed_or_committed_in_frozen_tail(self):
        manager = self.manager()
        self.assertTrue(manager.prepare_rgb_growth(self.model, 599))
        pending = manager.pending_growth
        for view in range(3):
            manager.observe_rgb_growth(
                self.model, self.positive_evidence(pending), view, 599, 100)
        before = pending["gain_sum"].copy()
        # Even a stale pending payload falsely labelled for the new boundary
        # must not bypass the final-window gate.
        pending["boundary"] = 601
        manager.observe_rgb_growth(self.model, {}, 0, 600, 100)
        np.testing.assert_array_equal(pending["gain_sum"], before)
        with patch.object(manager, "_select_rgb_growth",
                          side_effect=AssertionError("frozen commit")):
            replacement, event = manager.finish_epoch(
                self.model, self.optimizer, 601)
        self.assertIsNone(replacement)
        self.assertEqual(event["grown"], 0)
        self.assertTrue(event["growth_frozen"])
        self.assertFalse(event["growth_evidence_requested"])
        self.assertEqual(self.optimizer.remaps, [])
        self.assertIsNone(manager.pending_growth)
        self.assertIsNone(self.model.overlay)

    def test_legacy_alpha_evidence_and_geometry_are_disabled_in_frozen_tail(self):
        manager = self.manager(rgb=False)
        manager.settings.use_growth_evidence = True
        self.assertTrue(manager.needs_rendered_alpha(599))
        self.assertFalse(manager.needs_rendered_alpha(600))
        for evidence in (True, False):
            with self.subTest(evidence=evidence):
                manager.settings.use_growth_evidence = evidence
                with patch.object(manager, "_generate_growth",
                                  side_effect=AssertionError("frozen legacy growth")):
                    replacement, event = manager.finish_epoch(
                        self.model, self.optimizer, 601)
                self.assertIsNone(replacement)
                self.assertEqual(event["grown"], 0)
                self.assertTrue(event["growth_frozen"])

    def test_refinement_is_unchanged_even_when_entire_training_freezes_growth(self):
        results = []
        for freeze in (0, 700):
            manager = self.manager(freeze=freeze, scheduled=True)
            replacement, event = manager.finish_epoch(
                self.model, Optimizer(), manager.level_budget)
            self.assertTrue(event["refined"])
            self.assertEqual(event["grown"], 0)
            self.assertEqual(manager.level_start_epoch, 302)
            self.assertEqual(manager.resolution, 16)
            self.assertEqual(event["growth_frozen"], freeze > 0)
            results.append(replacement)
        self.assertEqual(results[0]["model_metadata"], results[1]["model_metadata"])
        for name, values in results[0]["model_state"].items():
            np.testing.assert_array_equal(values, results[1]["model_state"][name])

    def test_resume_retains_global_window_and_legacy_missing_field_means_zero(self):
        manager = self.manager(scheduled=True)
        manager.resolution = 16
        manager.level_start_epoch = 302
        state = manager.state_dict()
        for legacy in (False, True):
            with self.subTest(legacy=legacy):
                saved = copy.deepcopy(state)
                if legacy:
                    saved["settings"].pop("growth_freeze_last_epochs")
                # A fresh experiment default must not overwrite saved policy.
                resumed = self.manager(freeze=100, scheduled=True)
                resumed.load_state_dict(saved)
                self.assertEqual(resumed.level_start_epoch, 302)
                self.assertEqual(resumed.schedule.total_epochs, 700)
                self.assertEqual(
                    resumed.settings.growth_freeze_last_epochs, 0 if legacy else 100)
                self.assertTrue(resumed._growth_due(600))
                self.assertEqual(resumed._growth_due(601), legacy)
                self.assertEqual(resumed.growth_frozen(601), not legacy)


if __name__ == "__main__":
    unittest.main(verbosity=2)
