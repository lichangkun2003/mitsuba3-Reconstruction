"""CPU-only ordering checks for exact topology commit memory guards.

Selection/deletion decisions are mocked: these tests verify that their chosen
counts reach the guard before any replacement gather or optimizer migration.
They do not pretend to reproduce real training elections.
"""

from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reconstruction_memory import TrainingMemoryError
from reconstruction_topology_gpu import GPUTopologyLifecycle, prepare_model_remap


class CommitGuardTests(unittest.TestCase):
    def lifecycle(self, *, delete=False, grow=False):
        calls = []
        cfg = SimpleNamespace(growth_freeze_last_epochs=0, growth_rgb_normalization="observed",
            use_low_opacity_deletion=False, deletion_warmup_epochs=0, deletion_freeze_last_epochs=0,
            deletion_cooldown_epochs=5)
        manager = SimpleNamespace(settings=cfg, before_refine_checkpoint=None, resolution=6,
            core_resolution=4, level_budget=100, level_start_epoch=0, growth_warmup=0, count=4,
            schedule=SimpleNamespace(enabled=False, total_epochs=100),
            growth_frozen=lambda _: False, deletion_frozen=lambda _: False,
            _growth_due=lambda _: grow, _rgb_deletion_due=lambda _: delete,
            pending_growth=None, pending_rgb_deletion=None)
        model = SimpleNamespace(resolution=6, core_resolution=4, active_voxel_count=4,
            sh_count=9, opacity_sh_count=1, active_sh_count=1, scene_mapping="projective_shell",
            clear_growth_candidates=lambda: calls.append("clear_growth"))
        optimizer = SimpleNamespace(count=4, clear_replay=Mock(),
            remap=Mock(side_effect=AssertionError("Optimizer must not migrate before guard accepts")))
        lifecycle = GPUTopologyLifecycle.__new__(GPUTopologyLifecycle)
        lifecycle.manager = manager
        lifecycle.dr = SimpleNamespace(eval=lambda *_: calls.append("evaluate_decisions"),
            compress=lambda value: np.flatnonzero(value).astype(np.uint32),
            arange=Mock(side_effect=AssertionError("No append rows before guard accepts")))
        lifecycle.UInt, lifecycle.Bool, lifecycle.Int64 = np.uint32, np.bool_, np.int64
        lifecycle.fields = {}
        lifecycle.cooldown_cells = np.empty(0, np.uint32)
        lifecycle.cooldown_expiry = np.empty(0, np.int64)
        lifecycle.evaluate_rgb_deletion = Mock(return_value=np.array([False, True, False, True]) if delete else None)
        lifecycle.evaluate_opacity = Mock(return_value=None)
        lifecycle.reduce_event = lambda _: None
        lifecycle.select_growth = Mock(return_value={"cells": np.array([100, 101], np.uint32)} if grow else None)
        lifecycle.remap = Mock(side_effect=AssertionError("Lifecycle must not remap before guard accepts"))
        return lifecycle, model, optimizer, calls

    def test_deletion_refuses_before_source_snapshot_cooldown_or_gather(self):
        lifecycle, model, optimizer, calls = self.lifecycle(delete=True)
        observed = []

        def guard(actual_model, actual_optimizer, request):
            self.assertIs(actual_model, model)
            self.assertIs(actual_optimizer, optimizer)
            observed.append(request)
            raise TrainingMemoryError({"reason": "test refusal"})

        lifecycle.manager.topology_memory_guard = guard
        with patch("reconstruction_topology_gpu.device_checkpoint",
                   side_effect=AssertionError("Snapshot must follow guard")), \
             patch("reconstruction_topology_gpu.take_rows",
                   side_effect=AssertionError("Gather must follow guard")):
            with self.assertRaises(TrainingMemoryError):
                lifecycle.finish(model, optimizer, 10)
        self.assertEqual(len(observed), 1)
        request = observed[0]
        self.assertEqual((request["stage"], request["active_voxels"], request["old_voxels"]),
                         ("deletion_commit", 2, 4))
        self.assertEqual(request["deleted_voxels"], 2)
        self.assertEqual(lifecycle.manager.count, 4)
        self.assertEqual(len(lifecycle.cooldown_cells), 0)
        optimizer.remap.assert_not_called()
        lifecycle.remap.assert_not_called()

    def test_growth_refuses_after_selection_but_before_append_rows_or_adam(self):
        lifecycle, model, optimizer, calls = self.lifecycle(grow=True)
        observed = []

        def guard(_model, _optimizer, request):
            lifecycle.select_growth.assert_called_once()
            observed.append(request)
            raise TrainingMemoryError({"reason": "test refusal"})

        lifecycle.manager.topology_memory_guard = guard
        with patch("reconstruction_topology_gpu.device_checkpoint", return_value={"model_state": {}}), \
             patch("reconstruction_topology_gpu.append_model_rows",
                   side_effect=AssertionError("Append must follow guard")):
            with self.assertRaises(TrainingMemoryError):
                lifecycle.finish(model, optimizer, 10)
        request = observed[0]
        self.assertEqual((request["stage"], request["active_voxels"], request["added_voxels"]),
                         ("growth_commit", 6, 2))
        self.assertEqual(request["model_metadata"]["sh_count"], 9)
        self.assertEqual(request["model_metadata"]["active_sh_count"], 1)
        self.assertEqual(lifecycle.manager.count, 4)
        optimizer.remap.assert_not_called()
        lifecycle.remap.assert_not_called()
        lifecycle.dr.arange.assert_not_called()

    def test_combined_transaction_distinguishes_original_and_surviving_rows(self):
        lifecycle, model, optimizer, calls = self.lifecycle()
        lifecycle.manager.count = 2  # Deletion already selected/compacted lifecycle rows.
        guard = Mock()
        lifecycle.manager.topology_memory_guard = guard
        lifecycle.guard_commit_memory(model, optimizer, "growth_commit", 5, added=3, deleted=2)
        request = guard.call_args.args[2]
        self.assertEqual(request["old_voxels"], 4)
        self.assertEqual(request["current_voxels"], 2)
        self.assertEqual(request["active_voxels"], 5)
        self.assertEqual(request["count"], 5)
        self.assertEqual(request["added_voxels"], 3)
        self.assertEqual(request["deleted_voxels"], 2)
        self.assertTrue(request["exact_count"])

    def test_guard_returns_transfer_plan_to_commit(self):
        lifecycle, model, optimizer, _ = self.lifecycle()
        plan = {"migration_mode": "host_staged", "scratch_directory": "scratch"}
        lifecycle.manager.topology_memory_guard = Mock(return_value=plan)
        self.assertIs(lifecycle.guard_commit_memory(model, optimizer, "growth_commit", 6, added=2), plan)

    def test_optional_guard_and_no_change_leave_the_original_protocol(self):
        lifecycle, model, optimizer, calls = self.lifecycle()
        lifecycle.guard_commit_memory(model, optimizer, "growth_commit", 5, added=1)
        guard = Mock()
        lifecycle.manager.topology_memory_guard = guard
        replacement, event = lifecycle.finish(model, optimizer, 10)
        self.assertIsNone(replacement)
        self.assertEqual(event["active_voxel_count"], 4)
        guard.assert_not_called()
        optimizer.remap.assert_not_called()


class DeferredModelRemapTests(unittest.TestCase):
    class Values:
        def __init__(self, values):
            self.values = np.asarray(values.values if isinstance(values, type(self)) else values)

        def __len__(self):
            return len(self.values)

    def make_checkpoint(self, delete):
        widths = dict(cell_indices=1, centers=3, log_scales=3, rotations=4,
                      radiance_sh=27, opacity_sh=1)
        original = {name: self.Values(np.arange(4*width, dtype=np.uint32 if name == "cell_indices" else np.float32))
                    for name, width in widths.items()}
        checkpoint = dict(model_state=dict(original), model_metadata=dict(sh_count=9, opacity_sh_count=1))
        survivors = np.array([0, 2], np.uint32) if delete else np.arange(4, dtype=np.uint32)
        if delete:
            checkpoint["model_state"]["cell_indices"] = self.Values(original["cell_indices"].values[survivors])
            checkpoint["parameter_parent_rows"] = survivors
        return checkpoint, original, widths, survivors

    def test_staged_model_plans_keep_original_parameters_and_compact_cell_ids(self):
        for delete in (False, True):
            for grow in (False, True):
                with self.subTest(delete=delete, grow=grow):
                    checkpoint, original, widths, survivors = self.make_checkpoint(delete)
                    count = len(survivors)
                    additions = ({"cells" if name == "cell_indices" else name: self.Values(np.full(width, 91, np.float32))
                                  for name, width in widths.items()} if grow else None)
                    captured = {}
                    def deferred(old, rows, width, new_count, **options):
                        plan = SimpleNamespace(old=old, rows=rows, width=width, count=new_count, **options)
                        captured[id(plan)] = plan
                        return plan
                    dr = SimpleNamespace(arange=lambda dtype, size: np.arange(size, dtype=dtype),
                                         uint32_array_t=lambda _: np.uint32)
                    guard = Mock()
                    with patch("reconstruction_parameter_transfer.DeferredRemapField", side_effect=deferred), \
                         patch("reconstruction_topology_gpu.take_rows", side_effect=AssertionError("No full parameter gather")), \
                         patch("reconstruction_topology_gpu.append_model_rows", side_effect=AssertionError("No full parameter append")):
                        prepare_model_remap(checkpoint, count+int(grow), dr, selected=additions,
                            host_staged=True, scratch_directory="scratch", allocation_guard=guard)
                    self.assertNotIn("parameter_parent_rows", checkpoint)
                    self.assertEqual(len(captured), len(widths))
                    for name, plan in checkpoint["model_state"].items():
                        np.testing.assert_array_equal(plan.rows,
                            np.arange(count, dtype=np.uint32) if name == "cell_indices" else survivors)
                        if name != "cell_indices":
                            self.assertIs(plan.old, original[name])
                        self.assertEqual(plan.width, widths[name])
                        self.assertEqual(plan.count, count+int(grow))
                        self.assertIs(plan.allocation_guard, guard)
                        self.assertEqual(plan.scratch_directory, "scratch")
                        self.assertEqual(plan.append is None, not grow)


if __name__ == "__main__":
    unittest.main()
