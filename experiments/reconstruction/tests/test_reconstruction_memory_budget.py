"""CPU-only regression for array accounting and reference-cache budgeting."""

from __future__ import annotations

import copy
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reconstruction_memory import (GIB, MIB, estimate_training_memory,
                                   plan_training_memory, probe_cuda_allocator,
                                   round_cuda_allocation)


class MemoryBudgetTests(unittest.TestCase):
    def inputs(self, **overrides):
        values = dict(active_voxels=100, sh_count=9, opacity_sh_count=1,
                      compact_capacity=30, ray_count=20, max_hits=16, max_candidates=16)
        values.update(overrides)
        return values

    def plan(self, **overrides):
        values = dict(self.inputs(), reference_cache_mib=4096, gpu={"memory_mib": "32607"})
        values.update(overrides)
        return plan_training_memory(**values)

    def test_model_quaternion_and_adam_tangent_widths_are_distinct(self):
        report = estimate_training_memory(**self.inputs())
        c = report["components_bytes"]
        self.assertEqual(c["model_parameters"], 100 * 38 * 4)
        self.assertEqual(c["model_cell_indices"], 100 * 4)
        self.assertEqual(c["adam_means"] + c["adam_squared_means"], 100 * 37 * 8)
        self.assertEqual(c["adam_step_counters"], 100 * 5 * 4)
        self.assertEqual(c["gradients"], 30 * 37 * 4)
        self.assertEqual(c["gradient_validity_counters"], 30 * 3 * 4)
        self.assertEqual(c["compact_voxel_ids"], 30 * 4)
        self.assertEqual(c["foreground_background_evidence_counters"], 100 * 2 * 4)
        self.assertEqual(report["known_logical_bytes"], sum(c.values()))

    def test_cuda_allocator_rounds_each_buffer_and_preserves_legacy_comparison(self):
        for size, expected in ((0, 0), (1, 64), (64, 64), (65, 128),
                               (16 * MIB - 1, 16 * MIB),
                               (16 * MIB + 1, 18 * MIB),
                               (18 * MIB + 64, 20 * MIB)):
            with self.subTest(size=size):
                self.assertEqual(round_cuda_allocation(size), expected)
        self.assertEqual(round_cuda_allocation(16 * MIB + 1, "power_of_two"), 32 * MIB)
        report = estimate_training_memory(**self.inputs(active_voxels=54_146_397))
        fields = ("model_parameters", "model_cell_indices", "adam_means",
                  "adam_squared_means", "adam_step_counters")
        self.assertEqual(sum(report["legacy_allocated_components_bytes"][key] for key in fields),
                         35.25 * GIB)
        self.assertLess(sum(report["allocated_components_bytes"][key] for key in fields), 24 * GIB)
        # Aggregating first would incorrectly lose individual bucket rounding.
        self.assertNotEqual(report["allocated_components_bytes"]["model_parameters"],
                            round_cuda_allocation(report["components_bytes"]["model_parameters"]))

    def test_dense_and_compact_modes_keep_identical_model_and_adam(self):
        compact = estimate_training_memory(**self.inputs())
        dense = estimate_training_memory(**self.inputs(compact_capacity=0))
        self.assertEqual(dense["gradient_rows"], 100)
        self.assertFalse(dense["compact_gradients"])
        for name in ("model_parameters", "model_cell_indices", "adam_means", "adam_squared_means", "adam_step_counters"):
            self.assertEqual(compact["components_bytes"][name], dense["components_bytes"][name])
        self.assertEqual(dense["components_bytes"]["compact_slot_prefix"], 0)
        self.assertEqual(dense["components_bytes"]["compact_touched_scratch"], 0)
        clipped = estimate_training_memory(**self.inputs(compact_capacity=1000))
        self.assertEqual(clipped["gradient_rows"], 100)

    def test_native_hit_suffix_candidate_layout_and_distortion(self):
        enabled = estimate_training_memory(**self.inputs())
        disabled = estimate_training_memory(**self.inputs(), distortion=False)
        c = enabled["components_bytes"]
        self.assertEqual(c["hit_record_floats"], 20 * 16 * 11 * 4)
        self.assertEqual(c["hit_record_ids"], 20 * 16 * 4)
        self.assertEqual(c["reverse_suffix"], 20 * 16 * 4 * 4)
        self.assertEqual(c["geometric_candidate_state"], 20 * 16 * (6 + 2) * 4)
        self.assertEqual(enabled["known_logical_bytes"] - disabled["known_logical_bytes"], 20 * 16 * 2 * 4)

    def test_epoch_int64_fallback_and_known_indices_are_explicit(self):
        narrow = estimate_training_memory(**self.inputs())
        wide = estimate_training_memory(**self.inputs(), lifecycle_epoch_bytes=8, known_index_bytes=123)
        self.assertEqual(wide["known_logical_bytes"] - narrow["known_logical_bytes"], 100 * 3 * 4 + 123)
        self.assertEqual(narrow["components_bytes"]["lifecycle_epochs"] +
                         narrow["components_bytes"]["lifecycle_deletion_streaks"] +
                         narrow["components_bytes"]["lifecycle_candidate_mask"], 100 * 21)

    def test_dssim_workspace_is_one_image_independent_of_distortion_and_cache(self):
        baseline = estimate_training_memory(**self.inputs(), distortion=False)
        enabled = estimate_training_memory(**self.inputs(), distortion=False, dssim=True)
        self.assertEqual(enabled["known_logical_bytes"] - baseline["known_logical_bytes"], 20 * 192)
        self.assertEqual(enabled["components_bytes"]["hit_record_floats"],
                         baseline["components_bytes"]["hit_record_floats"])
        small = self.plan(dssim=True, reference_image_count=1)
        large = self.plan(dssim=True, reference_image_count=1000)
        self.assertEqual(small["components_bytes"]["dssim_image_workspace"],
                         large["components_bytes"]["dssim_image_workspace"])
        self.assertGreater(small["known_allocated_bytes"], self.plan()["known_allocated_bytes"])
        with self.assertRaises(ValueError):
            self.plan(dssim=1)

    def test_small_model_keeps_requested_cache_and_does_not_mutate_gpu_report(self):
        gpu = {"memory_mib": "32607", "name": "test"}
        before = copy.deepcopy(gpu)
        plan = self.plan(gpu=gpu)
        self.assertTrue(plan["supported"])
        self.assertEqual(plan["recommended_reference_cache_mib"], 4096)
        self.assertEqual(plan["reason_code"], "requested_cache_fits")
        self.assertEqual(gpu, before)

    def test_dataset_size_limits_residency_not_configured_capacity(self):
        plan = self.plan(reference_image_count=2, ray_count=1024, gpu={"memory_mib": 4096})
        self.assertTrue(plan["supported"])
        self.assertEqual(plan["recommended_reference_cache_mib"], 4096)
        self.assertEqual(plan["requested_reference_resident_bytes"], 2 * 1024 * 3 * 4)

    def test_budget_pressure_uses_256_128_then_zero_without_overcommitting(self):
        options = dict(ray_count=512 * 512)
        estimate = estimate_training_memory(**self.inputs(**options))
        fixed = estimate["known_allocated_bytes"]
        for available_mib, expected in ((512, 256), (200, 128), (64, 0)):
            plan = self.plan(**options, gpu={"memory_mib": (fixed + available_mib * MIB) / MIB},
                             workspace_reserve_mib=0, device_reserve_mib=0)
            self.assertTrue(plan["supported"])
            self.assertEqual(plan["recommended_reference_cache_mib"], expected)
            self.assertLessEqual(plan["estimated_budget_bytes"], plan["budget_bytes"])

    def test_model_that_cannot_fit_even_without_cache_is_rejected(self):
        plan = self.plan(gpu={"memory_mib": 512})
        self.assertFalse(plan["supported"])
        self.assertEqual(plan["reason_code"], "model_exceeds_budget")
        self.assertEqual(plan["recommended_reference_cache_mib"], 0)
        self.assertIn("Disabling the reference cache cannot fit", plan["reason"])

    def test_unknown_gpu_memory_is_an_explicit_non_mutating_refusal(self):
        for gpu in (None, {}, {"memory_mib": "unknown"}, {"memory_mib": True}, {"memory_mib": float("nan")}):
            with self.subTest(gpu=gpu):
                plan = self.plan(gpu=gpu)
                self.assertFalse(plan["supported"])
                self.assertEqual(plan["reason_code"], "gpu_memory_unknown")

    def test_bicycle_54m_compact_plans_compare_capacities_and_dense_refuses(self):
        common = dict(active_voxels=54_146_397, ray_count=1_016_814,
                      reference_image_count=169, known_index_bytes=96_352_464 + 216_585_592)
        plans = [self.plan(**common, compact_capacity=k) for k in (4_000_000, 6_000_000, 8_134_512)]
        self.assertLess(plans[0]["known_logical_bytes"], plans[1]["known_logical_bytes"])
        self.assertLess(plans[1]["known_logical_bytes"], plans[2]["known_logical_bytes"])
        for plan in plans:
            self.assertEqual(plan["components_bytes"]["model_parameters"], 54_146_397 * 38 * 4)
            self.assertTrue(plan["supported"], plan["reason"])
            self.assertEqual(plan["recommended_reference_cache_mib"], 256)
            self.assertLessEqual(plan["estimated_budget_bytes"], plan["budget_bytes"])
        dense = self.plan(**common, compact_capacity=0)
        self.assertFalse(dense["supported"])
        self.assertGreater(dense["known_logical_bytes"], 32 * GIB)
        unpatched = self.plan(**common, compact_capacity=8_134_512, allocator_policy="power_of_two")
        self.assertFalse(unpatched["supported"])
        self.assertEqual(unpatched["reason_code"], "model_exceeds_budget")

    def test_reference_planes_are_rounded_independently_of_logical_cache_limit(self):
        plan = self.plan(ray_count=1_016_814, reference_image_count=169)
        self.assertEqual(plan["reference_image_bytes"], 1_016_814 * 12)
        self.assertEqual(plan["reference_image_allocated_bytes"], 12 * MIB)
        self.assertEqual(plan["requested_reference_allocated_bytes"], 169 * 12 * MIB)
        self.assertGreater(plan["requested_reference_allocated_bytes"],
                           plan["requested_reference_resident_bytes"])

    def test_partial_ray_batches_keep_full_reference_images_and_cache_sizes(self):
        full = self.plan(ray_count=1_016_814, reference_image_count=169)
        batch = self.plan(ray_count=65_536, reference_image_pixels=1_016_814,
                          reference_image_count=169)
        for name in ("reference_image_bytes", "reference_image_allocated_bytes",
                     "requested_reference_resident_bytes", "requested_reference_allocated_bytes"):
            self.assertEqual(full[name], batch[name], name)
        self.assertEqual(full["components_bytes"]["active_reference_planes"],
                         batch["components_bytes"]["active_reference_planes"])
        self.assertEqual(batch["components_bytes"]["hit_record_floats"], 65_536 * 16 * 11 * 4)
        self.assertEqual(batch["components_bytes"]["geometric_candidate_state"], 65_536 * 16 * 8 * 4)
        self.assertLess(batch["known_allocated_bytes"], full["known_allocated_bytes"])
        self.assertEqual(batch["reference_image_pixels"], 1_016_814)
        with self.assertRaises(ValueError):
            self.plan(reference_image_pixels=0)

    def test_cache_padding_cannot_spend_bytes_outside_the_available_budget(self):
        # Tiny, non-power-of-two image planes deliberately amplify allocator
        # padding: a nominal 128 MiB cache would consume about 204.8 MiB.
        fixed = estimate_training_memory(**self.inputs())["known_allocated_bytes"]
        plan = self.plan(gpu={"memory_mib": (fixed + 200 * MIB) / MIB},
                         workspace_reserve_mib=0, device_reserve_mib=0)
        self.assertTrue(plan["supported"])
        self.assertEqual(plan["recommended_reference_cache_mib"], 0)
        self.assertLessEqual(plan["estimated_budget_bytes"], plan["budget_bytes"])

    def test_invalid_counts_and_empty_model(self):
        for key, value in (("active_voxels", -1), ("compact_capacity", True),
                           ("sh_count", 0), ("ray_count", 1.5), ("max_hits", 0),
                           ("max_candidates", -1)):
            with self.subTest(key=key):
                with self.assertRaises(ValueError):
                    estimate_training_memory(**self.inputs(**{key: value}))
        empty = estimate_training_memory(**self.inputs(active_voxels=0))
        self.assertEqual(empty["gradient_rows"], 0)
        self.assertEqual(empty["components_bytes"]["model_parameters"], 0)
        self.assertEqual(empty["components_bytes"]["hit_record_floats"], 0)
        self.assertGreater(empty["components_bytes"]["active_reference_planes"], 0)


class AllocatorProbeTests(unittest.TestCase):
    def runtime(self, delta, fail_eval=False):
        events = []

        class Array:
            def __init__(self, runtime, count, kind):
                self.runtime, self.count, self.kind = runtime, count, kind
                self.allocated = 0

            def __del__(self):
                events.append("release")
                self.runtime.cached += self.allocated

        class Runtime:
            cuda = SimpleNamespace(Float="cuda_float32")
            JitBackend = SimpleNamespace(CUDA="cuda_backend")

            def __init__(self):
                self.allocated = 123456
                self.cached = 0
                self.watermark = 999999
                self.detail = SimpleNamespace(malloc_clear_statistics=self.clear,
                                              malloc_watermark=self.read)

            def sync_thread(self):
                events.append("sync")

            def flush_malloc_cache(self):
                events.append("flush")
                self.allocated -= self.cached
                self.cached = 0

            def clear(self):
                events.append("clear")
                self.watermark = self.allocated

            def read(self, backend):
                events.append(("watermark", backend))
                return self.watermark

            def empty(self, dtype, count):
                events.append(("empty", dtype, count))
                return Array(self, count, "undefined")

            def arange(self, dtype, count):
                events.append(("arange", dtype, count))
                return Array(self, count, "expression")

            def eval(self, value):
                events.append("eval")
                if value.kind == "undefined":
                    return  # Match real Dr.Jit: eval(empty) need not allocate.
                if fail_eval and value.count == 20 * MIB // 4:
                    raise ValueError("evaluation failure")
                value.allocated = delta if value.count == 20 * MIB // 4 else value.count * 4
                self.allocated += value.allocated
                self.watermark = max(self.watermark, self.allocated)

        return Runtime(), events

    def test_detects_loaded_policy_and_releases_probe_before_flushing(self):
        for delta, policy in ((20 * MIB, "cuda_2m"), (32 * MIB, "power_of_two")):
            with self.subTest(policy=policy):
                dr, events = self.runtime(delta)
                self.assertEqual(probe_cuda_allocator(dr), {
                    "policy": policy, "probe_requested_bytes": 20 * MIB,
                    "probe_allocated_bytes": delta})
                self.assertEqual(events[0], "sync")
                self.assertIn(("arange", "cuda_float32", 1024), events)
                self.assertIn(("arange", "cuda_float32", 20 * MIB // 4), events)
                self.assertFalse(any(isinstance(event, tuple) and event[0] == "empty" for event in events))
                self.assertEqual(events[-3:], ["release", "sync", "flush"])

    def test_mock_undefined_array_does_not_falsely_model_an_allocation(self):
        dr, _ = self.runtime(20 * MIB)
        dr.detail.malloc_clear_statistics()
        baseline = dr.detail.malloc_watermark(dr.JitBackend.CUDA)
        value = dr.empty(dr.cuda.Float, 20 * MIB // 4)
        dr.eval(value)
        self.assertEqual(dr.detail.malloc_watermark(dr.JitBackend.CUDA), baseline)

    def test_unknown_allocator_delta_is_never_guessed(self):
        dr, events = self.runtime(21 * MIB)
        with self.assertRaisesRegex(RuntimeError, "Unsupported CUDA allocator probe"):
            probe_cuda_allocator(dr)
        self.assertEqual(events[-3:], ["release", "sync", "flush"])

    def test_evaluation_failure_still_synchronizes_and_flushes(self):
        dr, events = self.runtime(20 * MIB, fail_eval=True)
        with self.assertRaisesRegex(ValueError, "evaluation failure"):
            probe_cuda_allocator(dr)
        # Python's mock exception traceback can retain its value argument until
        # assertRaises exits; cleanup must still synchronize and flush.
        cleanup = [event for event in events if event != "release"]
        self.assertEqual(cleanup[-2:], ["sync", "flush"])


if __name__ == "__main__":
    unittest.main()
