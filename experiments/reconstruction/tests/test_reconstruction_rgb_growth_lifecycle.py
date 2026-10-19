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
            growth_rgb_min_support_views=3)
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
        self.assertEqual(len(replacement["model_state"]["cell_indices"]), 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
