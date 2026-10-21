"""CPU checks of core scheduling inside a larger, fixed shell grid."""

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


def checkpoint(resolution=272, core_resolution=256, *, empty=True):
    extent = resolution / core_resolution
    middle = resolution // 2
    cell = middle + resolution * middle + resolution * resolution * middle
    count = 0 if empty else 1
    metadata = {"resolution": resolution, "grid_min": [-extent] * 3,
                "grid_max": [extent] * 3, "sh_count": 1, "opacity_sh_count": 1}
    if core_resolution != resolution:
        metadata["core_resolution"] = core_resolution
    return {"model_metadata": metadata, "model_state": {
        "cell_indices": np.asarray([cell] * count, dtype=np.uint32),
        "centers": np.tile([.5, .5, .5], count).astype(np.float32),
        "log_scales": np.full(count * 3, np.log(.8 * 2 / core_resolution), dtype=np.float32),
        "rotations": np.tile([1., 0., 0., 0.], count).astype(np.float32),
        "radiance_sh": np.ones(count * 3, dtype=np.float32),
        "opacity_sh": np.zeros(count, dtype=np.float32)}}


def model(checkpoint):
    metadata = checkpoint["model_metadata"]
    return SimpleNamespace(
        **metadata, active_voxel_count=len(checkpoint["model_state"]["cell_indices"]),
        snapshot=lambda: checkpoint["model_state"])


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

    def test_shell_schedule_uses_core_without_coarsening_cells(self):
        state = checkpoint()
        manager = self.manager(state)
        self.assertEqual((manager.resolution, manager.core_resolution), (272, 256))
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
        self.assertEqual((event["resolution"], event["core_resolution"]), (544, 512))
        self.assertEqual((manager.resolution, manager.core_resolution), (544, 512))
        self.assertEqual(manager.level_budget, 400)
        self.assertEqual(manager.growth_warmup, 13)
        metadata = replacement["model_metadata"]
        self.assertEqual(metadata["grid_min"], state["model_metadata"]["grid_min"])
        self.assertEqual(metadata["grid_max"], state["model_metadata"]["grid_max"])
        self.assertEqual((metadata["grid_max"][0] - metadata["grid_min"][0]) / 544, 2 / 512)
        self.assertEqual(len(optimizer.remaps), 1)
        saved = manager.state_dict()
        self.assertEqual(saved["core_resolution"], 512)
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

    def test_nonempty_refinement_uses_storage_addresses(self):
        state = checkpoint(10, 8, empty=False)
        refined = topology.refine_state(state, topology.CoarseToFineConfig())
        self.assertEqual(refined["model_metadata"]["resolution"], 20)
        self.assertEqual(refined["model_metadata"]["core_resolution"], 16)
        parent = topology._coordinates(state["model_state"]["cell_indices"], 10, np)[0]
        children = topology._coordinates(refined["model_state"]["cell_indices"], 20, np)
        self.assertGreater(len(children), 0)
        np.testing.assert_array_equal(children // 2, np.tile(parent, (len(children), 1)))
        self.assertEqual(refined["model_metadata"]["grid_max"], [1.25] * 3)

    def test_legacy_synthetic_metadata_and_state_remain_compatible(self):
        state = checkpoint(256, 256)
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

    def test_missing_shell_core_metadata_cannot_silently_resume_wrong_schedule(self):
        manager = self.manager(checkpoint())
        saved = manager.state_dict()
        del saved["core_resolution"]
        with self.assertRaisesRegex(ValueError, "resolution"):
            self.manager(checkpoint()).load_state_dict(saved)

    def test_resume_rejects_same_count_but_different_grid_or_core(self):
        saved = self.manager(checkpoint()).state_dict()
        for resolution, core in ((274, 256), (272, 128)):
            with self.subTest(resolution=resolution, core=core):
                restored = self.manager(checkpoint(resolution, core),
                                        start_resolution=core, target_resolution=core * 2)
                previous = restored.state_dict()
                with self.assertRaisesRegex(ValueError, "resolution"):
                    restored.load_state_dict(saved)
                self.assertEqual(restored.resolution, previous["resolution"])
                self.assertEqual(restored.core_resolution, previous["core_resolution"])

    def test_actual_target_grid_must_fit_native_limit(self):
        with self.assertRaisesRegex(ValueError, "enclosing grid"):
            self.manager(checkpoint(544, 512), start_resolution=512, target_resolution=1024)

    def test_model_snapshots_record_only_nonidentity_core_resolution(self):
        dr = SimpleNamespace(eval=lambda *args: None, sync_thread=lambda: None)
        for resolution, core in ((272, 256), (256, 256)):
            state = checkpoint(resolution, core)
            with patch.dict(sys.modules, {"drjit": dr}):
                result = topology.model_checkpoint(model(state))
            self.assertEqual(result["model_metadata"], state["model_metadata"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
