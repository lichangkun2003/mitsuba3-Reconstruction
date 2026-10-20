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
from reconstruction_growth_rgb import RGB_GROWTH_EVIDENCE_VERSION


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
        self.assertGreater(len(pending["cells"]), len(np.unique(pending["cells"])))
        self.assertEqual(pending["all_candidate_count"], len(pending["cells"]))
        self.assertEqual(pending["candidate_cell_count"], len(np.unique(pending["cells"])))
        order = np.lexsort((pending["parent_cells"], pending["cells"]))
        np.testing.assert_array_equal(order, np.arange(len(order)))
        self.assertEqual(len(set(zip(pending["cells"], pending["parent_cells"]))),
                         len(pending["cells"]))
        self.assertTrue(np.all(pending["geometric_support"] >= 2))
        # RGB keeps 26-neighbor/evidence penetration rules even though legacy
        # use_growth_evidence is off, without ever requiring reference alpha.
        self.assertFalse(self.manager.needs_rendered_alpha(0))
        return pending

    def evidence(self, pending, *, gain=1., unknown=0):
        # Most lifecycle cases intentionally vote for ONE parent proposal.
        # Siblings in the same target cell must retain independent statistics.
        selected = np.zeros(len(pending["cells"]), dtype=bool)
        selected[np.flatnonzero(pending["cells"] == 74)[0]] = True
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

    def test_unknown_only_conflicting_and_neutral_evidence_do_not_grow(self):
        for gain, known, unknown in ((0., 0, 1), (-1., 1, 0), (-1., 1, 1), (0., 1, 1)):
            with self.subTest(gain=gain, known=known, unknown=unknown):
                self.manager = topology.TopologyManager(self.model, 3, self.settings, self.schedule)
                pending = self.prepare()
                out = self.evidence(pending, gain=gain, unknown=unknown)
                out["growth_observed"] *= known
                for view_id in range(3):
                    self.manager.observe_rgb_growth(self.model, out, view_id, 0, 100)
                replacement, event = self.manager.finish_epoch(self.model, self.optimizer, 1)
                self.assertIsNone(replacement)
                self.assertEqual(event["grown"], 0)
                self.assertEqual(len(self.optimizer.remaps), 0)
                detail = event["growth_rgb"]
                self.assertEqual(detail["views_with_unknown"], 3 if unknown else 0)
                self.assertEqual(detail["conflict_votes"], 3 if gain < 0 else 0)
                self.assertEqual(detail["observed_views"], 3 if known else 0)

    def test_reliable_positive_rays_grow_despite_unknown_rays(self):
        pending = self.prepare()
        self.observe_all(pending, gain=1., unknown=1)
        replacement, event = self.manager.finish_epoch(self.model, self.optimizer, 1)
        self.assertIsNotNone(replacement)
        self.assertEqual(event["grown"], 1)
        detail = event["growth_rgb"]
        self.assertEqual(detail["support_votes"], 3)
        self.assertEqual(detail["views_with_unknown"], 3)
        self.assertEqual(detail["observed_views"], 3)
        self.assertAlmostEqual(detail["encountered_known_fraction"], .5)

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
        self.assertEqual(saved["rgb_growth_evidence_version"], RGB_GROWTH_EVIDENCE_VERSION)
        self.assertEqual(saved["pending_growth"]["evidence_version"], RGB_GROWTH_EVIDENCE_VERSION)
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

    def test_observed_mean_excludes_unseen_and_unknown_only_but_includes_neutral(self):
        # Exactly three supporting views plus one reliable neutral view. All
        # other views are invisible/unknown, and must not dilute the mean.
        for threshold, expected_growth in ((.07, 1), (.08, 0)):
            with self.subTest(threshold=threshold):
                self.settings.growth_rgb_min_mean_gain = threshold
                self.manager = topology.TopologyManager(self.model, 100, self.settings, self.schedule)
                pending = self.prepare()
                for view_id in range(100):
                    if view_id < 3:
                        out = self.evidence(pending, gain=.1, unknown=7)
                    elif view_id == 3:
                        out = self.evidence(pending, gain=0.)
                    elif view_id == 4:
                        out = self.evidence(pending, gain=0., unknown=1)
                        out["growth_observed"].fill(0)
                    else:
                        out = self.evidence(pending, gain=0.)
                        out["growth_observed"].fill(0)
                    self.manager.observe_rgb_growth(self.model, out, view_id, 0, 640000)
                _, event = self.manager.finish_epoch(self.model, self.optimizer, 1)
                self.assertEqual(event["grown"], expected_growth)
                detail = event["growth_rgb"]
                self.assertEqual(detail["observed_views"], 4)
                self.assertEqual(detail["views_with_unknown"], 4)
                self.assertAlmostEqual(detail["mean_gain_sum"], .075, places=6)

    def test_current_version_without_normalization_key_keeps_full_image_units(self):
        self.settings.growth_rgb_normalization = "full_image"
        pending = self.prepare()
        self.manager.observe_rgb_growth(self.model, self.evidence(pending), 0, 0, 100)
        saved = self.manager.state_dict()
        # Evidence version is current; omitting the independent normalization
        # key retains full-image units, not old unknown-veto semantics.
        self.assertEqual(saved["rgb_growth_evidence_version"], RGB_GROWTH_EVIDENCE_VERSION)
        saved["settings"].pop("growth_rgb_normalization")
        saved["pending_growth"].pop("normalization")
        current = copy.deepcopy(self.settings)
        current.growth_rgb_normalization = "observed"
        resumed = topology.TopologyManager(self.model, 3, current, self.schedule)
        resumed.load_state_dict(saved)
        self.assertEqual(resumed.settings.growth_rgb_normalization, "full_image")
        for view_id in (1, 2):
            resumed.observe_rgb_growth(self.model, self.evidence(pending), view_id, 0, 100)
        target = np.flatnonzero(resumed.pending_growth["cells"] == 74)[0]
        np.testing.assert_allclose(resumed.pending_growth["gain_sum"][target], .03)
        siblings = np.flatnonzero(resumed.pending_growth["cells"] == 74)[1:]
        np.testing.assert_array_equal(resumed.pending_growth["gain_sum"][siblings], 0.)
        _, event = resumed.finish_epoch(self.model, self.optimizer, 1)
        self.assertEqual(event["grown"], 1)
        self.assertAlmostEqual(event["growth_rgb"]["mean_gain_sum"], .01, places=6)

    def test_pending_normalization_cannot_be_changed_mid_window(self):
        self.prepare()
        saved = self.manager.state_dict()
        saved["settings"]["growth_rgb_normalization"] = "full_image"
        with self.assertRaisesRegex(ValueError, "different normalization"):
            self.manager.load_state_dict(saved)

    def test_rgb_checkpoint_requires_current_evidence_version_even_without_pending(self):
        self.prepare()
        state = self.manager.state_dict()
        for has_pending in (False, True):
            for version in (None, 1, 2, RGB_GROWTH_EVIDENCE_VERSION + 1):
                with self.subTest(has_pending=has_pending, version=version):
                    saved = copy.deepcopy(state)
                    if not has_pending:
                        saved["pending_growth"] = None
                    if version is None:
                        saved.pop("rgb_growth_evidence_version")
                    else:
                        saved["rgb_growth_evidence_version"] = version
                    resumed = topology.TopologyManager(self.model, 3, self.settings, self.schedule)
                    with self.assertRaisesRegex(ValueError, "evidence version"):
                        resumed.load_state_dict(saved)

    def test_pending_evidence_version_cannot_mix_old_view_statistics(self):
        pending = self.prepare()
        self.manager.observe_rgb_growth(self.model, self.evidence(pending), 0, 0, 100)
        state = self.manager.state_dict()
        for version in (None, 1, 2, RGB_GROWTH_EVIDENCE_VERSION + 1):
            with self.subTest(version=version):
                saved = copy.deepcopy(state)
                if version is None:
                    saved["pending_growth"].pop("evidence_version")
                else:
                    saved["pending_growth"]["evidence_version"] = version
                resumed = topology.TopologyManager(self.model, 3, self.settings, self.schedule)
                with self.assertRaisesRegex(ValueError, "evidence version"):
                    resumed.load_state_dict(saved)

    def test_non_rgb_checkpoint_does_not_require_rgb_evidence_version(self):
        self.settings.use_rgb_growth = False
        self.manager = topology.TopologyManager(self.model, 3, self.settings, self.schedule)
        saved = self.manager.state_dict()
        saved.pop("rgb_growth_evidence_version", None)
        resumed = topology.TopologyManager(self.model, 3, self.settings, self.schedule)
        resumed.load_state_dict(saved)
        self.assertFalse(resumed.settings.use_rgb_growth)

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
        cell_count = len(np.unique(pending["cells"]))
        self.assertEqual(count, pending["all_candidate_count"])
        self.assertGreater(count, cell_count)
        self.observe_every_candidate(pending)
        initial_count = self.manager.count
        replacement, event = self.manager.finish_epoch(self.model, self.optimizer, 1)
        self.assertEqual(event["grown"], cell_count)
        cells = replacement["model_state"]["cell_indices"]
        self.assertEqual(len(cells), initial_count + cell_count)
        self.assertEqual(len(cells), len(np.unique(cells)))
        detail = event["growth_rgb"]
        for key in ("capacity_rejected", "geometry_rejected"):
            self.assertEqual(detail[key], 0)
        self.assertEqual(detail["selected_candidates"], cell_count)

    def test_capacity_ranking_uses_local_mean_not_sum_or_coverage(self):
        pending = self.make_many_candidates()
        count = len(pending["cells"])
        self.settings.max_active_voxels = self.manager.count + 1
        low, high = np.unique(pending["cells"], return_index=True)[1][:2]
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
        low, high = np.unique(pending["cells"], return_index=True)[1][:2]
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

    def test_later_parent_can_grow_when_first_parent_is_harmful(self):
        pending = self.prepare()
        first, second = np.flatnonzero(pending["cells"] == 74)
        self.assertNotEqual(pending["parent_cells"][first], pending["parent_cells"][second])
        out = self.evidence(pending, gain=-10.)
        out["growth_gain"][second] = 2.
        out["growth_observed"][second] = 1
        for view in range(3):
            self.manager.observe_rgb_growth(self.model, out, view, 0, 640000)
        replacement, event = self.manager.finish_epoch(self.model, self.optimizer, 1)
        self.assertEqual(event["grown"], 1)
        self.assertEqual(event["growth_rgb"]["rgb_qualified"], 1)
        self.assertEqual(event["growth_rgb"]["conflict_votes"], 3)
        self.assertEqual(replacement["model_state"]["cell_indices"][-1], 74)
        for name in ("centers", "log_scales", "rotations", "radiance_sh", "opacity_sh"):
            expected = pending[name][second].reshape(-1)
            np.testing.assert_array_equal(replacement["model_state"][name][-len(expected):], expected)

    def test_same_cell_selects_highest_local_mean_then_parent_cell_on_ties(self):
        for gains, expected_parent in (((1., 2.), 75), ((2., 2.), 73)):
            with self.subTest(gains=gains):
                self.manager = topology.TopologyManager(self.model, 3, self.settings, self.schedule)
                pending = self.prepare()
                ids = np.flatnonzero(pending["cells"] == 74)
                out = self.evidence(pending)
                out["growth_gain"][ids] = gains
                out["growth_observed"][ids] = 1
                for view in range(3):
                    self.manager.observe_rgb_growth(self.model, out, view, 0, 100)
                replacement, event = self.manager.finish_epoch(self.model, self.optimizer, 1)
                self.assertEqual(event["grown"], 1)
                self.assertEqual(event["growth_rgb"]["rgb_qualified"], 2)
                self.assertEqual(event["growth_rgb"]["capacity_rejected"], 0)
                winner = ids[np.flatnonzero(pending["parent_cells"][ids] == expected_parent)[0]]
                expected = pending["centers"][winner].reshape(-1)
                np.testing.assert_array_equal(replacement["model_state"]["centers"][-3:], expected)

    def test_siblings_cannot_pool_supporting_views(self):
        pending = self.prepare()
        first, second = np.flatnonzero(pending["cells"] == 74)
        for view in range(3):
            out = self.evidence(pending, gain=0.)
            out["growth_observed"].fill(0)
            chosen = first if view < 2 else second
            out["growth_gain"][chosen] = 1.
            out["growth_observed"][chosen] = 1
            self.manager.observe_rgb_growth(self.model, out, view, 0, 100)
        np.testing.assert_array_equal(pending["support_views"][[first, second]], [2, 1])
        replacement, event = self.manager.finish_epoch(self.model, self.optimizer, 1)
        self.assertIsNone(replacement)
        self.assertEqual(event["grown"], 0)
        self.assertEqual(event["growth_rgb"]["rgb_qualified"], 0)

    def test_invalid_best_parent_does_not_hide_revalidated_second_choice(self):
        # Three valid parents surround target 74. After voting the highest-gain
        # parent retracts, but the other two still satisfy geometric support.
        state = self.checkpoint["model_state"]
        state["cell_indices"] = np.append(state["cell_indices"], np.uint32(66))
        for name, width in (("centers", 3), ("log_scales", 3), ("rotations", 4),
                            ("radiance_sh", 12), ("opacity_sh", 1)):
            state[name] = np.concatenate((state[name], state[name][:width].copy()))
        self.model = Model(self.checkpoint)
        self.manager = topology.TopologyManager(self.model, 3, self.settings, self.schedule)
        pending = self.prepare()
        ids = np.flatnonzero(pending["cells"] == 74)
        by_parent = dict(zip(pending["parent_cells"][ids], ids))
        self.assertEqual(set(by_parent), {66, 73, 75})
        out = self.evidence(pending, gain=0.)
        out["growth_observed"].fill(0)
        for parent, gain in ((73, 3.), (75, 2.)):
            out["growth_gain"][by_parent[parent]] = gain
            out["growth_observed"][by_parent[parent]] = 1
        for view in range(3):
            self.manager.observe_rgb_growth(self.model, out, view, 0, 100)
        state["log_scales"][:3] = np.log(.1)
        replacement, event = self.manager.finish_epoch(self.model, self.optimizer, 1)
        self.assertEqual(event["grown"], 1)
        self.assertEqual(event["growth_rgb"]["rgb_qualified"], 2)
        self.assertEqual(event["growth_rgb"]["geometry_revalidated"], 1)
        self.assertEqual(event["growth_rgb"]["geometry_rejected"], 1)
        np.testing.assert_array_equal(replacement["model_state"]["centers"][-3:],
                                      pending["centers"][by_parent[75]])

    def test_capacity_is_applied_after_same_cell_competition(self):
        pending = self.make_many_candidates()
        self.settings.max_active_voxels = self.manager.count + 2
        cells, firsts = np.unique(pending["cells"], return_index=True)
        out = {name: np.zeros(len(pending["cells"]), dtype=dtype)
               for name, dtype in (("growth_gain", np.float32),
                                   ("growth_observed", np.uint32),
                                   ("growth_unknown", np.uint32))}
        first_siblings = np.flatnonzero(pending["cells"] == cells[0])
        self.assertGreater(len(first_siblings), 1)
        out["growth_gain"][first_siblings] = 10.
        out["growth_observed"][first_siblings] = 1
        out["growth_gain"][firsts[1:3]] = [2., 1.]
        out["growth_observed"][firsts[1:3]] = 1
        for view in range(3):
            self.manager.observe_rgb_growth(self.model, out, view, 0, 100)
        replacement, event = self.manager.finish_epoch(self.model, self.optimizer, 1)
        self.assertEqual(event["grown"], 2)
        np.testing.assert_array_equal(np.sort(replacement["model_state"]["cell_indices"][-2:]),
                                      cells[:2])
        self.assertEqual(event["growth_rgb"]["capacity_rejected"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
