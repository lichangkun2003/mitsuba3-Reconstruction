"""CPU checks of frozen online RGB proposals, resume, vetoes and commit.

The model stub deliberately has no render/trace method: candidate preparation
and evidence decisions must only consume the native training statistics.
"""

from __future__ import annotations

import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import reconstruction_topology as topology


def seed_checkpoint():
    return {"model_metadata": {"resolution": 8, "grid_min": [0., 0., 0.],
            "grid_max": [8., 8., 8.], "sh_count": 4, "opacity_sh_count": 1},
            "model_state": {
                "cell_indices": np.asarray([73, 75], np.uint32),
                "centers": np.tile([.5, .5, .5], 2).astype(np.float32),
                "log_scales": np.tile(np.log([.8, .7, .6]), 2).astype(np.float32),
                "rotations": np.tile([1., 0., 0., 0.], 2).astype(np.float32),
                "radiance_sh": np.arange(24, dtype=np.float32) * .01,
                "opacity_sh": np.zeros(2, np.float32)}}


class Model:
    def __init__(self, checkpoint):
        self.checkpoint = checkpoint
        self.resolution = checkpoint["model_metadata"]["resolution"]
        self.active_voxel_count = len(checkpoint["model_state"]["cell_indices"])
        self.overlay = None

    def set_growth_candidates(self, state):
        self.overlay = copy.deepcopy(state)

    def clear_growth_candidates(self):
        self.overlay = None


class Optimizer:
    def __init__(self):
        self.remaps = []

    def remap(self, survivors, count, **kwargs):
        self.remaps.append((survivors.copy(), count, kwargs))


class RGBGrowthLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.checkpoint = seed_checkpoint()
        self.model = Model(self.checkpoint)
        self.optimizer = Optimizer()
        self.settings = topology.TopologySettings(
            enable_deletion=False, enable_growth=True, use_growth_evidence=False,
            use_rgb_growth=True, growth_warmup_epochs=0, growth_interval=1,
            growth_rgb_min_support_views=3, growth_rgb_normalization="observed")
        self.schedule = topology.CoarseToFineConfig(enabled=False, total_epochs=5)
        self.manager = topology.TopologyManager(self.model, 3, self.settings, self.schedule)
        self.snapshot = patch.object(topology, "model_checkpoint",
                                     side_effect=lambda model, fields=None: copy.deepcopy(model.checkpoint))
        self.snapshot.start()
        self.addCleanup(self.snapshot.stop)

    def prepare(self):
        self.assertTrue(self.manager.prepare_rgb_growth(self.model, 0))
        pending = self.manager.pending_growth
        self.assertIn(74, pending["cells"])
        self.assertEqual(len(pending["cells"]), len(np.unique(pending["cells"])))
        self.assertTrue(np.all(pending["geometric_support"] >= 2))
        # RGB keeps 26-neighbor/evidence penetration rules even though legacy
        # use_growth_evidence is off, without ever requiring reference alpha.
        self.assertFalse(self.manager.needs_rendered_alpha(0))
        return pending

    def evidence(self, pending, *, gain=1., unknown=0):
        selected = pending["cells"] == 74
        return {"growth_gain": np.where(selected, gain, 0).astype(np.float32),
                "growth_observed": selected.astype(np.uint32),
                "growth_unknown": np.where(selected, unknown, 0).astype(np.uint32)}

    def observe_all(self, pending, *, gain=1., unknown=0):
        out = self.evidence(pending, gain=gain, unknown=unknown)
        for view_id in range(3):
            self.manager.observe_rgb_growth(self.model, out, view_id, 0, 100)

    def test_commit_exact_frozen_geometry_and_appearance(self):
        pending = self.prepare()
        target = int(np.flatnonzero(pending["cells"] == 74)[0])
        frozen = {name: pending[name][target].copy() for name in
                  ("centers", "log_scales", "rotations", "radiance_sh", "opacity_sh")}
        self.observe_all(pending)
        # A duplicate (even conflicting) view must not change the evidence.
        self.manager.observe_rgb_growth(self.model, self.evidence(pending, gain=-100.), 0, 0, 100)
        self.checkpoint["model_state"]["radiance_sh"][:] = 99.
        replacement, event = self.manager.finish_epoch(self.model, self.optimizer, 1)
        self.assertEqual(event["grown"], 1)
        self.assertEqual(replacement["model_state"]["cell_indices"][-1], 74)
        for name, expected in frozen.items():
            np.testing.assert_array_equal(replacement["model_state"][name][-expected.size:], expected.reshape(-1))
        np.testing.assert_array_equal(self.optimizer.remaps[0][0], [0, 1])
        self.assertEqual(self.optimizer.remaps[0][1], 3)
        self.assertEqual(self.manager.growth_wait_start[-1], 2)
        self.assertIsNone(self.model.overlay)
        self.assertIsNone(self.manager.pending_growth)

    def test_unknown_or_conflicting_evidence_does_not_grow(self):
        for gain, unknown in ((1., 1), (-1., 0)):
            with self.subTest(gain=gain, unknown=unknown):
                self.manager = topology.TopologyManager(self.model, 3, self.settings, self.schedule)
                pending = self.prepare()
                self.observe_all(pending, gain=gain, unknown=unknown)
                replacement, event = self.manager.finish_epoch(self.model, self.optimizer, 1)
                self.assertIsNone(replacement)
                self.assertEqual(event["grown"], 0)
                self.assertEqual(len(self.optimizer.remaps), 0)
                field = "unknown_votes" if unknown else "conflict_votes"
                self.assertEqual(event["growth_rgb"][field], 3)

    def test_incomplete_view_set_never_commits(self):
        pending = self.prepare()
        self.manager.observe_rgb_growth(self.model, self.evidence(pending), 0, 0, 100)
        replacement, event = self.manager.finish_epoch(self.model, self.optimizer, 1)
        self.assertIsNone(replacement)
        self.assertEqual(event["growth_rgb"]["status"], "incomplete_views")

    def test_current_parent_contact_is_rechecked(self):
        pending = self.prepare()
        self.observe_all(pending)
        self.checkpoint["model_state"]["log_scales"][:] = np.log(.1)
        replacement, event = self.manager.finish_epoch(self.model, self.optimizer, 1)
        self.assertIsNone(replacement)
        self.assertEqual(event["growth_rgb"]["rgb_qualified"], 1)
        self.assertEqual(event["growth_rgb"]["geometry_revalidated"], 0)

    def test_checkpoint_resume_keeps_proposal_and_distinct_views(self):
        pending = self.prepare()
        self.manager.observe_rgb_growth(self.model, self.evidence(pending), 0, 0, 100)
        saved = self.manager.state_dict()
        resumed = topology.TopologyManager(self.model, 3, self.settings, self.schedule)
        resumed.load_state_dict(saved)
        self.assertTrue(resumed.prepare_rgb_growth(self.model, 0))
        self.assertEqual(resumed.pending_growth["seen_views"], {0})
        for name, value in saved["pending_growth"]["candidate_state"].items():
            np.testing.assert_array_equal(resumed.pending_growth["candidate_state"][name], value)
        self.manager = resumed
        self.observe_all(resumed.pending_growth)
        replacement, event = resumed.finish_epoch(self.model, self.optimizer, 1)
        self.assertEqual(event["grown"], 1)
        self.assertEqual(event["growth_rgb"]["support_votes"], 3)
        self.assertEqual(event["growth_rgb"]["normalization"], "observed")
        self.assertEqual(len(replacement["model_state"]["cell_indices"]), 3)

    def test_observed_mean_excludes_unseen_and_unknown_but_includes_neutral(self):
        # Exactly three supporting views plus one reliable neutral view. All
        # other views are invisible/unknown, and must not dilute the mean.
        for threshold, expected_growth in ((.07, 1), (.08, 0)):
            with self.subTest(threshold=threshold):
                self.settings.growth_rgb_min_mean_gain = threshold
                self.manager = topology.TopologyManager(self.model, 100, self.settings, self.schedule)
                pending = self.prepare()
                for view_id in range(100):
                    if view_id < 3:
                        out = self.evidence(pending, gain=.1)
                    elif view_id == 3:
                        out = self.evidence(pending, gain=0.)
                    elif view_id == 4:
                        out = self.evidence(pending, gain=-99., unknown=1)
                    else:
                        out = self.evidence(pending, gain=0.)
                        out["growth_observed"].fill(0)
                    self.manager.observe_rgb_growth(self.model, out, view_id, 0, 640000)
                _, event = self.manager.finish_epoch(self.model, self.optimizer, 1)
                self.assertEqual(event["grown"], expected_growth)
                detail = event["growth_rgb"]
                self.assertEqual(detail["observed_views"], 4)
                self.assertEqual(detail["unknown_votes"], 1)
                self.assertAlmostEqual(detail["mean_gain_sum"], .075, places=6)

    def test_old_pending_checkpoint_keeps_full_image_units(self):
        self.settings.growth_rgb_normalization = "full_image"
        pending = self.prepare()
        self.manager.observe_rgb_growth(self.model, self.evidence(pending), 0, 0, 100)
        saved = self.manager.state_dict()
        # A checkpoint with no normalization key retains full-image units.
        saved["settings"].pop("growth_rgb_normalization")
        saved["pending_growth"].pop("normalization")
        current = copy.deepcopy(self.settings)
        current.growth_rgb_normalization = "observed"
        resumed = topology.TopologyManager(self.model, 3, current, self.schedule)
        resumed.load_state_dict(saved)
        self.assertEqual(resumed.settings.growth_rgb_normalization, "full_image")
        for view_id in (1, 2):
            resumed.observe_rgb_growth(self.model, self.evidence(pending), view_id, 0, 100)
        target = resumed.pending_growth["cells"] == 74
        np.testing.assert_allclose(resumed.pending_growth["gain_sum"][target], [.03])
        _, event = resumed.finish_epoch(self.model, self.optimizer, 1)
        self.assertEqual(event["grown"], 1)
        self.assertAlmostEqual(event["growth_rgb"]["mean_gain_sum"], .01, places=6)

    def test_pending_normalization_cannot_be_changed_mid_window(self):
        self.prepare()
        saved = self.manager.state_dict()
        saved["settings"]["growth_rgb_normalization"] = "full_image"
        with self.assertRaisesRegex(ValueError, "different normalization"):
            self.manager.load_state_dict(saved)

    def make_many_candidates(self, view_count=3):
        # Replicate the actual two-parent hole geometry, rather than mocking
        # proposal generation: the complete legal candidate set exceeds 1024.
        resolution = 64
        offsets = np.asarray([(x, y, z) for z in range(0, 40, 4)
                              for y in range(0, 40, 4)
                              for x in range(0, 40, 4)])
        cells = (offsets[:, None, :] + np.asarray([[1, 1, 1], [3, 1, 1]])).reshape(-1, 3)
        state = self.checkpoint["model_state"]
        state["cell_indices"] = (cells[:, 0] + resolution * (cells[:, 1] + resolution * cells[:, 2])).astype(np.uint32)
        for name in ("centers", "log_scales", "rotations", "radiance_sh", "opacity_sh"):
            state[name] = np.tile(state[name], len(offsets))
        self.checkpoint["model_metadata"].update(resolution=resolution, grid_max=[float(resolution)] * 3)
        self.model = Model(self.checkpoint)
        self.manager = topology.TopologyManager(self.model, view_count, self.settings, self.schedule)
        self.assertTrue(self.manager.prepare_rgb_growth(self.model, 0))
        pending = self.manager.pending_growth
        self.assertGreater(pending["all_candidate_count"], 1024)
        return pending

    def observe_every_candidate(self, pending):
        count = len(pending["cells"])
        out = {"growth_gain": np.ones(count, np.float32),
               "growth_observed": np.ones(count, np.uint32),
               "growth_unknown": np.zeros(count, np.uint32)}
        for view_id in range(3):
            self.manager.observe_rgb_growth(self.model, out, view_id, 0, 640000)

    def test_evaluate_and_commit_every_legal_candidate(self):
        pending = self.make_many_candidates()
        count = len(pending["cells"])
        self.assertEqual(count, pending["all_candidate_count"])
        self.observe_every_candidate(pending)
        initial_count = self.manager.count
        replacement, event = self.manager.finish_epoch(self.model, self.optimizer, 1)
        self.assertEqual(event["grown"], count)
        self.assertEqual(len(replacement["model_state"]["cell_indices"]), initial_count + count)
        detail = event["growth_rgb"]
        for key in ("capacity_rejected", "geometry_rejected"):
            self.assertEqual(detail[key], 0)
        self.assertEqual(detail["selected_candidates"], count)

    def test_capacity_ranking_uses_local_mean_not_sum_or_coverage(self):
        pending = self.make_many_candidates()
        count = len(pending["cells"])
        self.settings.max_active_voxels = self.manager.count + 1
        low, high = np.argsort(pending["cells"])[:2]
        out = {"growth_gain": np.zeros(count, np.float32),
               "growth_observed": np.zeros(count, np.uint32),
               "growth_unknown": np.zeros(count, np.uint32)}
        # Smaller coverage but better local benefit should win one free slot.
        out["growth_gain"][[low, high]] = [100., 2.]
        out["growth_observed"][[low, high]] = [100, 1]
        for view_id in range(3):
            self.manager.observe_rgb_growth(self.model, out, view_id, 0, 640000)
        target_cell = pending["cells"][high]
        replacement, event = self.manager.finish_epoch(self.model, self.optimizer, 1)
        self.assertEqual(event["grown"], 1)
        self.assertEqual(replacement["model_state"]["cell_indices"][-1], target_cell)
        self.assertEqual(event["growth_rgb"]["capacity_rejected"], 1)

    def test_capacity_ranking_uses_mean_over_reliable_views(self):
        pending = self.make_many_candidates(view_count=5)
        count = len(pending["cells"])
        self.settings.max_active_voxels = self.manager.count + 1
        low, high = np.argsort(pending["cells"])[:2]
        out = {"growth_gain": np.zeros(count, np.float32),
               "growth_observed": np.zeros(count, np.uint32),
               "growth_unknown": np.zeros(count, np.uint32)}
        out["growth_gain"][[low, high]] = [1., 1.5]
        out["growth_observed"][[low, high]] = 1
        for view_id in range(3):
            self.manager.observe_rgb_growth(self.model, out, view_id, 0, 640000)
        # Low is invisible in two views; high has reliable zero gains there.
        # Their final means are 1 and 0.9, despite sums of 3 and 4.5.
        out["growth_gain"].fill(0)
        out["growth_observed"][low] = 0
        for view_id in (3, 4):
            self.manager.observe_rgb_growth(self.model, out, view_id, 0, 640000)
        target_cell = pending["cells"][low]
        replacement, event = self.manager.finish_epoch(self.model, self.optimizer, 1)
        self.assertEqual(event["grown"], 1)
        self.assertEqual(replacement["model_state"]["cell_indices"][-1], target_cell)
        self.assertEqual(event["growth_rgb"]["capacity_rejected"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
