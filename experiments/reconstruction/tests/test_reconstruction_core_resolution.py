"""CPU checks of core scheduling with fixed world-space background leaves."""

from __future__ import annotations

import copy
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import reconstruction_topology as topology


def checkpoint(resolution=256, *, shell_levels=8, empty=True, bounded=False):
    middle = resolution // 2
    cell = middle + resolution * middle + resolution * resolution * middle
    cells = [] if empty else [cell] if bounded else [cell, resolution**3]
    count = len(cells)
    metadata = {"resolution": resolution, "grid_min": [-1.] * 3,
                "grid_max": [1.] * 3, "sh_count": 1, "opacity_sh_count": 1}
    if not bounded:
        metadata.update(core_resolution=resolution, scene_mapping="world_shell",
                        background_shell_levels=shell_levels, background_refinement="fixed")
    return {"model_metadata": metadata, "model_state": {
        "cell_indices": np.asarray(cells, dtype=np.uint32),
        "centers": np.tile([.5, .5, .5], count).astype(np.float32),
        "log_scales": np.full(count * 3, np.log(.8 * 2 / resolution), dtype=np.float32),
        "rotations": np.tile([1., 0., 0., 0.], count).astype(np.float32),
        "radiance_sh": np.ones(count * 3, dtype=np.float32),
        "opacity_sh": np.zeros(count, dtype=np.float32)}}


def model(checkpoint):
    metadata = checkpoint["model_metadata"]
    return SimpleNamespace(
        **dict(metadata, world_shell_version=5, cell_index_bits=64),
        active_voxel_count=len(checkpoint["model_state"]["cell_indices"]),
        snapshot=lambda: checkpoint["model_state"],
        native_background_shell_levels=lambda: metadata.get("background_shell_levels", 0),
        native_background_subdivision_level=lambda: metadata.get("background_subdivision_level", 0),
        native_background_use_ellipsoid=lambda: metadata.get("background_geometry", "occupied_cell") == "ellipsoid")


class Optimizer:
    def __init__(self):
        self.remaps = []

    def remap(self, survivors, count, **kwargs):
        self.remaps.append((survivors.copy(), count, kwargs))


class CoreResolutionTests(unittest.TestCase):
    def manager(self, state, **schedule_overrides):
        config = topology.CoarseToFineConfig(save_each_level=False, **schedule_overrides)
        settings = topology.TopologySettings(
            enable_growth=False, enable_deletion=False,
            growth_warmup_epochs=7, growth_fine_warmup_epochs=13)
        return topology.TopologyManager(model(state), 3, settings, config)

    def test_shell_schedule_uses_core_without_padding_or_coarsening_cells(self):
        state = checkpoint()
        manager = self.manager(state)
        self.assertEqual((manager.resolution, manager.core_resolution), (256, 256))
        self.assertEqual(manager.budgets, {256: 600, 512: 400})
        self.assertEqual(manager.level_budget, 600)
        self.assertEqual(manager.growth_warmup, 7)
        metadata = state["model_metadata"]
        self.assertEqual((metadata["grid_max"][0] - metadata["grid_min"][0]) /
                         manager.resolution, 2 / 256)

    def test_refine_resume_preserves_bounds_budgets_and_final_level(self):
        state = checkpoint()
        manager = self.manager(state)
        optimizer = Optimizer()
        with patch.object(topology, "model_checkpoint", return_value=copy.deepcopy(state)):
            replacement, event = manager.finish_epoch(model(state), optimizer, 600)
        self.assertTrue(event["refined"])
        self.assertEqual(event["resolution"], 512)
        self.assertEqual((manager.resolution, manager.core_resolution), (512, 512))
        self.assertEqual(manager.level_budget, 400)
        self.assertEqual(manager.growth_warmup, 13)
        metadata = replacement["model_metadata"]
        self.assertEqual(metadata["grid_min"], state["model_metadata"]["grid_min"])
        self.assertEqual(metadata["grid_max"], state["model_metadata"]["grid_max"])
        self.assertEqual((metadata["grid_max"][0] - metadata["grid_min"][0]) / 512, 2 / 512)
        self.assertEqual(metadata["background_shell_levels"], 8)
        self.assertEqual(len(optimizer.remaps), 1)
        saved = manager.state_dict()
        self.assertEqual(saved.get("core_resolution", saved["resolution"]), 512)
        resumed = self.manager(replacement)
        resumed.load_state_dict(saved)
        self.assertEqual(resumed.budgets, {256: 600, 512: 400})
        self.assertEqual(resumed.schedule_start_resolution, 256)
        self.assertEqual(resumed.level_start_epoch, 600)
        self.assertEqual(resumed.level_budget, 400)
        self.assertEqual(resumed.growth_warmup, 13)
        final, event = resumed.finish_epoch(model(replacement), optimizer, 1000)
        self.assertIsNone(final)
        self.assertTrue(event["finished"])
        self.assertFalse(event["refined"])

    def test_nonempty_refinement_changes_only_core_addresses_and_rebases_background_ids(self):
        state = checkpoint(8, empty=False)
        refined = topology.refine_state(state, topology.CoarseToFineConfig())
        self.assertEqual(refined["model_metadata"]["resolution"], 16)
        self.assertEqual(refined["model_metadata"]["core_resolution"], 16)
        parent = topology._coordinates(state["model_state"]["cell_indices"][:1], 8, np)[0]
        cells = refined["model_state"]["cell_indices"]
        background = cells >= 16**3
        np.testing.assert_array_equal(cells[background], [16**3])
        children = topology._coordinates(cells[~background], 16, np)
        self.assertGreater(len(children), 0)
        np.testing.assert_array_equal(children // 2, np.tile(parent, (len(children), 1)))
        self.assertEqual(refined["model_metadata"]["grid_max"], [1.] * 3)
        for name, old in state["model_state"].items():
            if name != "cell_indices":
                np.testing.assert_array_equal(refined["model_state"][name].reshape(len(cells), -1)[background],
                                              old.reshape(2, -1)[-1:])

    def test_legacy_synthetic_metadata_and_state_remain_compatible(self):
        state = checkpoint(256, bounded=True)
        manager = self.manager(state)
        self.assertEqual(manager.core_resolution, 256)
        saved = manager.state_dict()
        self.assertNotIn("core_resolution", saved)
        restored = self.manager(state)
        restored.load_state_dict(saved)
        self.assertEqual(restored.core_resolution, 256)
        refined = topology.refine_state(state, topology.CoarseToFineConfig())
        self.assertEqual(refined["model_metadata"]["resolution"], 512)
        self.assertNotIn("core_resolution", refined["model_metadata"])

    def test_identity_core_resume_does_not_need_a_redundant_core_resolution(self):
        manager = self.manager(checkpoint())
        saved = manager.state_dict()
        saved.pop("core_resolution", None)
        restored = self.manager(checkpoint())
        restored.load_state_dict(saved)
        self.assertEqual(restored.core_resolution, 256)
        self.assertEqual(restored.budgets, manager.budgets)

    def test_resume_rejects_same_count_but_different_grid_or_core(self):
        saved = self.manager(checkpoint()).state_dict()
        for resolution in (128, 512):
            with self.subTest(resolution=resolution):
                restored = self.manager(checkpoint(resolution),
                                        start_resolution=resolution, target_resolution=resolution * 2)
                previous = restored.state_dict()
                with self.assertRaisesRegex(ValueError, "resolution"):
                    restored.load_state_dict(saved)
                self.assertEqual(restored.resolution, previous["resolution"])
                self.assertEqual(restored.core_resolution, previous.get("core_resolution", previous["resolution"]))

    def test_target_limit_applies_to_core_not_the_enlarged_world_root(self):
        manager = self.manager(checkpoint(512), start_resolution=512, target_resolution=1024)
        self.assertEqual(manager.schedule.target_resolution, 1024)
        self.assertEqual(manager.schedule.target_resolution << 8, 262144)
        with self.assertRaisesRegex(ValueError, "1024"):
            self.manager(checkpoint(512), start_resolution=512, target_resolution=2048)

    def test_model_snapshots_record_world_shells_and_preserve_bounded_metadata(self):
        dr = SimpleNamespace(eval=lambda *args: None, sync_thread=lambda: None)
        for bounded in (False, True):
            state = checkpoint(256, bounded=bounded)
            with patch.dict(sys.modules, {"drjit": dr}):
                result = topology.model_checkpoint(model(state))
            expected = dict(state["model_metadata"])
            if not bounded:
                expected.update(world_shell_version=5, background_subdivision_level=0,
                                background_geometry="occupied_cell")
            self.assertEqual(result["model_metadata"], expected)


if __name__ == "__main__":
    unittest.main(verbosity=2)
