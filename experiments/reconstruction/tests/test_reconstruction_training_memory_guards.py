"""CPU integration checks for budget guards around training allocations."""

from __future__ import annotations

import copy
from dataclasses import asdict
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import train
from reconstruction_gradient_storage import gradient_capacity, native_gradient_result
from reconstruction_memory import MIB, TrainingMemoryError


def configuration():
    config = train.TrainConfig()
    config.training.rgb_only = True
    config.training.sh_degree_progression = True
    config.training.chunk_size = 0
    config.training.gpu_reference_cache_mib = 4096
    config.render.width, config.render.height = 1559, 1039
    config.render.max_hits = config.gradient.max_candidates = 16
    config.gradient.distortion_loss_weight = config.gradient.dssim_loss_weight = .2
    return config


def model(*, compact_records=True):
    value = SimpleNamespace(active_voxel_count=51_910_528, sh_count=9,
        opacity_sh_count=1, compact_gradient_version=1, core_resolution=512,
        resolution=520, statistics=lambda: {
            "octree_allocated_bytes": 17 * MIB,
            "spatial_index_allocated_bytes": 29 * MIB,
            "other_counter": 900_000_000})
    if compact_records:
        value.path_record_layout_version = 1
    return value


def report():
    return {"gpu": {"memory_mib": 32607}, "cuda_allocator": {"policy": "cuda_2m"},
            "checkpoint": "before-refine-0512-epoch-01050.npz"}


def request():
    return {"stage": "projective_model", "active_voxels": 51_910_528,
        "exact_count": True, "allocation_voxels": 51_910_528,
        "old_voxels": 6_488_816, "existing_child_count": 1_578_880,
        "background_child_count": 50_331_648,
        "model_metadata": {"resolution": 1040, "sh_count": 9,
                           "opacity_sh_count": 1}}


class Cache:
    def __init__(self, events):
        self.capacity_bytes = 4096 * MIB
        self.entries = {"image": np.array([.2, .4, .6], np.float32)}
        self.events = events

    def clear(self):
        self.events.append("cache.clear")
        self.entries.clear()

    def resize(self, capacity):
        self.events.append(("cache.resize", capacity))
        self.capacity_bytes = capacity * MIB


class TrainingMemoryGuardTests(unittest.TestCase):
    def test_native_index_counts_are_rounded_per_array_including_empty_tree_sentinels(self):
        current = SimpleNamespace(active_voxel_count=4_194_305, growth_candidate_count=0,
            statistics=lambda: {"octree_normal_parent_nodes": 65,
                "octree_candidate_parent_nodes": 0,
                "octree_allocated_bytes": 1, "spatial_index_allocated_bytes": 2})
        # 16 MiB + 4 bytes rounds to 18 MiB under the CUDA policy. Each of
        # three 260-byte node arrays rounds independently to 512 bytes, and
        # the empty candidate tree retains four 64-byte scalar sentinels.
        self.assertEqual(train.model_index_bytes(current, "cuda_2m"),
                         18 * MIB + 3 * 512 + 4 * 64)
        self.assertEqual(train.model_index_bytes(current, "power_of_two"),
                         32 * MIB + 3 * 512 + 4 * 64)
        current.active_voxel_count = 0
        current.statistics = lambda: {"octree_normal_parent_nodes": 0,
                                      "octree_candidate_parent_nodes": 0}
        self.assertEqual(train.model_index_bytes(current, "cuda_2m"), 8 * 64)

    def test_old_native_uses_original_record_budget_and_counts_both_indices(self):
        config, current, legacy = configuration(), model(), model(compact_records=False)
        new = train.training_memory_plan(config, current, report(), 255)
        old = train.training_memory_plan(config, legacy, report(), 255)
        self.assertEqual(new["path_record_layout_version"], 1)
        self.assertEqual(old["path_record_layout_version"], 0)
        self.assertTrue(new["compact_path_records"])
        self.assertFalse(old["compact_path_records"])
        self.assertGreater(old["known_logical_bytes"], new["known_logical_bytes"])
        self.assertGreaterEqual(old["known_allocated_bytes"], new["known_allocated_bytes"])
        self.assertEqual(new["gradient_rows"], 2 * 1559 * 1039)
        self.assertEqual(train.model_index_bytes(current), 46 * MIB)
        without_index = train.training_memory_plan(config, current, report(), 255,
                                                   known_index_bytes=0)
        self.assertEqual(new["known_allocated_bytes"] - without_index["known_allocated_bytes"],
                         46 * MIB)

    def test_proposed_count_capacity_and_full_reference_shape_reach_budget(self):
        config, current = configuration(), model()
        with patch.object(train, "plan_training_memory", return_value={}) as planner:
            train.training_memory_plan(config, current, report(), 255,
                active_voxels=70_000_000, known_index_bytes=123456,
                compact_capacity_override=7_000_000)
            arguments = planner.call_args.kwargs
            self.assertEqual(arguments["active_voxels"], 70_000_000)
            self.assertEqual(arguments["compact_capacity"], 7_000_000)
            self.assertEqual(arguments["known_index_bytes"], 123456)
            self.assertEqual(arguments["reference_image_pixels"], 1559 * 1039)
            self.assertTrue(arguments["radiance_degree_steps"])
            # Existing multi-GPU and explicitly chunked paths retain dense
            # gradients; their references remain full images in the ledger.
            for world_size, chunk_size in ((2, 0), (1, 1024)):
                config.training.chunk_size = chunk_size
                train.training_memory_plan(config, current, report(), 255, world_size=world_size)
                arguments = planner.call_args.kwargs
                self.assertEqual(arguments["compact_capacity"], 0)
                self.assertEqual(arguments["reference_image_pixels"], 1559 * 1039)
                self.assertEqual(arguments["ray_count"], chunk_size or 1559 * 1039)
        self.assertEqual(current.active_voxel_count, 51_910_528)

    def test_rejected_training_plan_is_reported_before_cache_resize(self):
        events, state, current = [], report(), model()
        cache = Cache(events)
        refusal = {"supported": False, "reason": "test training budget refusal"}
        with patch.object(train, "training_memory_plan", return_value=refusal):
            with self.assertRaises(TrainingMemoryError) as caught:
                train.configure_training_memory(configuration(), current, cache, state, 255)
        self.assertEqual(caught.exception.plan, refusal)
        self.assertIs(state["memory_plans"]["512"], refusal)
        self.assertEqual(state["checkpoint"], report()["checkpoint"])
        self.assertEqual(events, [])
        self.assertIn("image", cache.entries)
        self.assertEqual(cache.capacity_bytes, 4096 * MIB)

    def exercise_refinement(self, *, refused=None):
        config, current, state, proposed = configuration(), model(), report(), request()
        config_before, proposed_before = asdict(config), copy.deepcopy(proposed)
        current.parameters = np.arange(9, dtype=np.float32)
        original_parameters = current.parameters.copy()
        events = []
        cache = Cache(events)
        optimizer = SimpleNamespace(moments=np.arange(12, dtype=np.float32), step=31,
            clear_replay=lambda: events.append("optimizer.clear_replay"))
        original_moments = optimizer.moments.copy()
        replay = SimpleNamespace(clear=lambda: events.append("native.clear"))
        dr = SimpleNamespace(sync_thread=lambda: events.append("sync"),
                             flush_malloc_cache=lambda: events.append("flush"))
        ordinary = {"supported": refused != "training", "reason": "ordinary budget"}
        transition = {"supported": refused != "transition", "reason": "transition budget"}

        def training_plan(*args, **kwargs):
            events.append("training.plan")
            self.assertEqual(kwargs["active_voxels"], proposed["active_voxels"])
            self.assertEqual(kwargs["known_index_bytes"], 321 * MIB)
            self.assertEqual(cache.entries, {})
            return ordinary

        def transition_plan(actual_request, **kwargs):
            events.append("transition.plan")
            self.assertIs(actual_request, proposed)
            self.assertEqual(kwargs["known_index_bytes"], 46 * MIB)
            self.assertTrue(kwargs["radiance_degree_steps"])
            self.assertEqual(kwargs["live_memory"]["free_bytes"], 20 * 1024**3)
            return transition

        with patch.object(train, "training_memory_plan", side_effect=training_plan), \
             patch.object(train, "plan_refinement_memory", side_effect=transition_plan), \
             patch.object(train, "estimate_spatial_index_bytes", return_value=321 * MIB) as index, \
             patch.object(train, "refinement_memory_snapshot", return_value={
                 "free_bytes": 20 * 1024**3, "total_bytes": 32 * 1024**3}):
            if refused:
                with self.assertRaises(TrainingMemoryError) as caught:
                    train.prepare_refinement_memory(config, current, optimizer, cache, replay,
                        state, 255, proposed, dr)
                self.assertEqual(caught.exception.plan, ordinary if refused == "training" else transition)
            else:
                result = train.prepare_refinement_memory(config, current, optimizer, cache, replay,
                    state, 255, proposed, dr)
                self.assertIs(result, state["refinement_memory_plans"][-1])
            index.assert_called_once_with(proposed["active_voxels"], 1040,
                                          allocator_policy="cuda_2m")
        self.assertEqual(events, ["native.clear", "optimizer.clear_replay", "cache.clear",
                                   "sync", "flush", "training.plan", "transition.plan"])
        self.assertEqual(len(state["refinement_memory_plans"]), 1)
        record = state["refinement_memory_plans"][0]
        self.assertIs(record["training"], ordinary)
        self.assertIs(record["transition"], transition)
        self.assertEqual(record["active_voxels"], proposed["active_voxels"])
        self.assertEqual(record["exact_count"], proposed["exact_count"])
        self.assertEqual(state["checkpoint"], report()["checkpoint"])
        self.assertEqual(cache.capacity_bytes, 4096 * MIB)
        np.testing.assert_array_equal(current.parameters, original_parameters)
        np.testing.assert_array_equal(optimizer.moments, original_moments)
        self.assertEqual(optimizer.step, 31)
        self.assertEqual(asdict(config), config_before)
        self.assertEqual(proposed, proposed_before)

    def test_successful_refinement_guard_changes_only_caches_and_diagnostics(self):
        self.exercise_refinement()

    def test_either_refinement_plan_can_refuse_without_mutating_parameters_or_adam(self):
        for refused in ("training", "transition"):
            with self.subTest(refused=refused):
                self.exercise_refinement(refused=refused)

    def exercise_topology_commit(self, *, failure=None, stage="growth_commit"):
        config, current, state = configuration(), model(), report()
        config_before = asdict(config)
        current.parameters = np.array([.2, .3, .4], np.float32)
        original_parameters = current.parameters.copy()
        events = []
        cache = Cache(events)
        optimizer = SimpleNamespace(step=37, moments=np.array([.07, .11], np.float32),
            clear_replay=lambda: events.append("optimizer.clear_replay"))
        original_moments = optimizer.moments.copy()
        replay = SimpleNamespace(clear=lambda: events.append("native.clear"))
        dr = SimpleNamespace(sync_thread=lambda: events.append("sync"),
                             flush_malloc_cache=lambda: events.append("flush"))
        count = 1_000_001
        proposed = {"stage": stage, "active_voxels": count,
            "model_metadata": {"resolution": 1040, "sh_count": 9, "opacity_sh_count": 1},
            "selected_count": 19, "survivor_count": count - 19}
        proposed_before = copy.deepcopy(proposed)
        field_bytes = 104 * MIB
        ordinary = {"supported": failure != "future", "reason": "future model budget"}
        # A huge CUDA virtual-memory free report must not override smaller
        # physical free memory. Two GiB cover scratch and device reserves.
        live = {"total_bytes": 64 * 1024**3, "free_bytes": 40 * 1024**3,
                "dedicated_free_bytes": field_bytes + 2 * 1024**3}
        if failure == "largest_field":
            live["dedicated_free_bytes"] = 2 * 1024**3 - 1
        if failure == "missing_live":
            live = None

        def training_plan(*args, **kwargs):
            events.append("training.plan")
            self.assertEqual(kwargs["active_voxels"], count)
            self.assertEqual(kwargs["known_index_bytes"], 321 * MIB)
            self.assertEqual(cache.entries, {})
            return ordinary

        with patch.object(train, "training_memory_plan", side_effect=training_plan), \
             patch.object(train, "estimate_spatial_index_bytes", return_value=321 * MIB) as index, \
             patch.object(train, "refinement_memory_snapshot", return_value=live):
            if failure:
                with self.assertRaises(TrainingMemoryError) as caught:
                    train.prepare_topology_commit_memory(config, current, optimizer, cache,
                        replay, state, 255, proposed, dr)
            else:
                result = train.prepare_topology_commit_memory(config, current, optimizer, cache,
                    replay, state, 255, proposed, dr)
                self.assertIs(result, state["topology_commit_memory_plans"][-1])
            index.assert_called_once_with(count, 1040, allocator_policy="cuda_2m")
        self.assertEqual(events, ["native.clear", "optimizer.clear_replay", "cache.clear",
                                  "sync", "flush", "sync", "training.plan"])
        self.assertEqual(len(state["topology_commit_memory_plans"]), 1)
        record = state["topology_commit_memory_plans"][0]
        transition = record["transition"]
        self.assertIs(record["training"], ordinary)
        self.assertEqual(record["stage"], stage)
        self.assertEqual(record["active_voxels"], count)
        self.assertEqual(transition["largest_replacement_field_bytes"], field_bytes)
        self.assertEqual(transition["additional_required_bytes"], field_bytes + 1024**3)
        self.assertEqual(transition["supported"], failure not in ("missing_live", "largest_field"))
        if failure == "missing_live":
            self.assertIsNone(transition["available_bytes"])
            self.assertEqual(transition["reason_code"], "gpu_memory_unknown")
        elif failure == "largest_field":
            self.assertEqual(transition["available_bytes"], 1024**3 - 1)
            self.assertEqual(transition["reason_code"], "topology_commit_exceeds_budget")
        else:
            self.assertEqual(transition["available_bytes"], field_bytes + 1024**3)
            self.assertEqual(transition["reason_code"], "topology_commit_fits")
        if failure:
            self.assertEqual(caught.exception.plan, ordinary if failure == "future" else transition)
        np.testing.assert_array_equal(current.parameters, original_parameters)
        np.testing.assert_array_equal(optimizer.moments, original_moments)
        self.assertEqual(optimizer.step, 37)
        self.assertEqual(proposed, proposed_before)
        self.assertEqual(asdict(config), config_before)
        self.assertEqual(state["checkpoint"], report()["checkpoint"])
        self.assertEqual(cache.capacity_bytes, 4096 * MIB)

    def test_topology_commit_guard_accepts_exact_budget_without_changing_selection(self):
        for stage in ("deletion_commit", "growth_commit"):
            with self.subTest(stage=stage):
                self.exercise_topology_commit(stage=stage)

    def test_topology_commit_guard_refuses_missing_live_or_future_or_replacement_budget(self):
        for failure in ("missing_live", "future", "largest_field"):
            with self.subTest(failure=failure):
                self.exercise_topology_commit(failure=failure)

    def test_two_voxel_growth_uses_staging_when_whole_sh_copy_cannot_fit(self):
        config, current, state = configuration(), model(), report()
        current.active_voxel_count = 50_407_562
        proposed = {"stage": "growth_commit", "active_voxels": 50_407_564,
            "old_voxels": 50_407_562, "added": 2, "deleted": 0,
            "model_metadata": {"resolution": 1040, "sh_count": 9, "opacity_sh_count": 1}}
        events = []
        optimizer = SimpleNamespace(clear_replay=lambda: None)
        replay = SimpleNamespace(clear=lambda: None)
        dr = SimpleNamespace(sync_thread=lambda: None, flush_malloc_cache=lambda: None)
        live = {"free_bytes": int(3.68 * 1024**3), "total_bytes": 32 * 1024**3}
        with patch.object(train, "training_memory_plan", return_value={"supported": True}), \
             patch.object(train, "refinement_memory_snapshot", return_value=live):
            result = train.prepare_topology_commit_memory(config, current, optimizer,
                Cache(events), replay, state, 255, proposed, dr)
            self.assertEqual(result["migration_mode"], "host_staged")
            transition = result["transition"]
            self.assertTrue(transition["supported"])
            self.assertGreater(transition["additional_required_bytes"], transition["available_bytes"])
            self.assertLess(transition["staged_additional_required_bytes"], transition["available_bytes"])
            self.assertEqual(transition["reason_code"], "topology_commit_host_staged")
            # A surviving alias or a competing allocation must still cause a
            # refusal before upload; staging is not a bypass of the budget.
            with self.assertRaises(TrainingMemoryError):
                result["allocation_guard"](5 * 1024**3)
            result["allocation_guard"](1024**3)
        json.dumps(state)
        self.assertNotIn("allocation_guard", state["topology_commit_memory_plans"][-1])

    def test_two_rows_per_pixel_remains_a_hint_with_exact_overflow_retry(self):
        calls = []
        touched = 80_001
        expected_ids = np.arange(touched, dtype=np.uint32)
        expected_gradients = expected_ids.astype(np.float32) * .125

        class Model:
            active_voxel_count = 80_017
            compact_gradient_version = 1
            topology_revision = 0

            def loss_and_grad(self, options):
                capacity = options["compact_gradient_capacity"]
                calls.append(capacity)
                return {"voxel_count": np.array([touched], np.uint32),
                    "voxel_ids": expected_ids[:capacity].copy(),
                    "gradients": {"radiance_sh": expected_gradients[:capacity].copy()}}

        current = Model()
        capacity = gradient_capacity(current, 40_000)
        self.assertEqual(capacity, 80_000)
        options = {"rgb_only": 1., "distortion_loss_weight": .2, "dssim_loss_weight": .2}
        before = options.copy()
        result = native_gradient_result(current, (options,), None, capacity, np)
        self.assertEqual(calls, [80_000, 80_017])
        np.testing.assert_array_equal(result["voxel_ids"], expected_ids)
        np.testing.assert_array_equal(result["gradients"]["radiance_sh"], expected_gradients)
        self.assertEqual(options, before)

    def test_training_memory_refusal_publishes_report_without_overwriting_checkpoint(self):
        with tempfile.TemporaryDirectory(prefix="training-memory-guard-") as temporary:
            output = Path(temporary)
            checkpoint = output / "before-refine-0512-epoch-01050.npz"
            original = b"existing complete checkpoint remains untouched"
            checkpoint.write_bytes(original)
            refusal = {"supported": False, "reason": "injected allocation refusal"}

            def runtime(_config, _output, state, **kwargs):
                # Inject at the runtime seam so this exercises the real outer
                # catch/finally/report path without ever importing CUDA.
                state["checkpoint"] = str(checkpoint)
                state["pre_refine_checkpoints"] = {"512": {
                    "path": str(checkpoint), "pending_epoch_transition": True}}
                raise TrainingMemoryError(refusal)

            with patch.object(sys, "argv", ["train.py"]), \
                 patch.object(train, "prepare_training_output", return_value=output), \
                 patch.object(train, "validate"), \
                 patch.object(train, "initialize_runtime", side_effect=runtime), \
                 patch.object(train, "save_checkpoint") as save_checkpoint, \
                 patch("builtins.print"):
                _config, _model, _output, state, _performance = train.run_training()
            save_checkpoint.assert_not_called()
            self.assertEqual(state["status"], "memory_limited")
            self.assertEqual(state["training_status"], "memory_limited")
            self.assertEqual(state["memory_refusal"], refusal)
            self.assertEqual(checkpoint.read_bytes(), original)
            saved = json.loads((output / "training-report.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["checkpoint"], str(checkpoint))
            self.assertTrue(saved["pre_refine_checkpoints"]["512"]["pending_epoch_transition"])
            self.assertEqual(saved["memory_refusal"], refusal)
            summary = json.loads(Path(state["performance_file"]).read_text(encoding="utf-8"))
            self.assertEqual(summary["status"], "memory_limited")


if __name__ == "__main__":
    unittest.main()
