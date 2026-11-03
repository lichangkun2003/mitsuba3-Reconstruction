"""CPU-only regression for array accounting and reference-cache budgeting."""

from __future__ import annotations

import copy
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reconstruction_memory import (GIB, MIB, estimate_training_memory,
                                   estimate_spatial_index_bytes,
                                   plan_training_memory, probe_cuda_allocator,
                                   round_cuda_allocation)


class MemoryBudgetTests(unittest.TestCase):
    def inputs(self, **overrides):
        values = dict(active_voxels=100, sh_count=9, opacity_sh_count=1,
                      compact_capacity=30, ray_count=20, max_hits=16, max_candidates=16)
        values.update(overrides)
        return values

    def test_spatial_index_bound_uses_flattened_three_arrays_and_padded_root(self):
        # Dense 4^3 grid: 8 parents plus the root, held in three arrays.
        expected = round_cuda_allocation(64 * 4) + 3 * round_cuda_allocation(9 * 4) + 4 * 64
        self.assertEqual(estimate_spatial_index_bytes(64, 4), expected)
        self.assertEqual(estimate_spatial_index_bytes(0, 1000), 8 * 64)
        self.assertEqual(estimate_spatial_index_bytes(1, 1), 8 * 64)
        # A non-power-of-two bounded grid has real-domain parent counts,
        # not a fully occupied padded 1024 cube.
        n = 51_910_528
        nodes = sum(min(n, ((1000 + (1 << level) - 1) >> level)**3)
                    for level in range(1, 11))
        expected = round_cuda_allocation(n * 4) + 3 * round_cuda_allocation(nodes * 4) + 4 * 64
        self.assertEqual(estimate_spatial_index_bytes(n, 1000), expected)
        self.assertLess(estimate_spatial_index_bytes(n, 1000),
                        estimate_spatial_index_bytes(n, 1000, allocator_policy="power_of_two"))

    def test_mixed_spatial_bound_covers_leaf_and_internal_edge_references(self):
        # Independently enumerate ancestors of small actual cube shells. Their
        # coarse leaves stay single entries; no world-root grid is allocated.
        import itertools
        core, shells = 4, 3
        root = core << shells
        depth = root.bit_length() - 1
        offset = (root - core) // 2
        leaves = [(offset+x, offset+y, offset+z, 0)
                  for x, y, z in itertools.product(range(core), repeat=3)]
        for shell in range(1, shells + 1):
            outer = core << shell
            side, minimum = outer//4, (root-outer)//2
            for xyz in itertools.product(range(4), repeat=3):
                if all(1 <= value <= 2 for value in xyz):
                    continue
                leaves.append((*[minimum+value*side for value in xyz], side.bit_length()-1))
        for selected in (leaves, leaves[::3], leaves[-1:], []):
            ancestors = {(level, x >> level, y >> level, z >> level)
                         for x, y, z, log_size in selected
                         for level in range(log_size+1, depth+1)}
            nodes = len(ancestors)
            edges = len(selected) + max(0, nodes-1)
            for policy in ("cuda_2m", "power_of_two"):
                rounded = lambda count: round_cuda_allocation(max(1, count)*4, policy)
                actual = rounded(edges) + 3*rounded(nodes) + 4*rounded(1)
                bound = estimate_spatial_index_bytes(len(selected), core,
                    background_shell_levels=shells, allocator_policy=policy)
                self.assertLessEqual(actual, bound)
        self.assertLess(estimate_spatial_index_bytes(448+8, 1024,
            background_shell_levels=8), MIB)

    def test_uniform_background_spatial_bound_includes_subdivided_leaves(self):
        import itertools
        core, shells, subdivision = 8, 2, 2
        root = core << shells
        depth = root.bit_length() - 1
        leaves = []
        for shell in range(1, shells+1):
            outer = core << shell
            coarse_side, minimum = outer//4, (root-outer)//2
            child_side = coarse_side >> subdivision
            for xyz in itertools.product(range(4), repeat=3):
                if all(1 <= value <= 2 for value in xyz):
                    continue
                for child in itertools.product(range(1 << subdivision), repeat=3):
                    leaves.append((*[minimum+xyz[a]*coarse_side+child[a]*child_side for a in range(3)],
                                   child_side.bit_length()-1))
        self.assertEqual(len(leaves), 56*shells*8**subdivision)
        ancestors = {(level, x >> level, y >> level, z >> level)
                     for x, y, z, log_size in leaves for level in range(log_size+1, depth+1)}
        nodes = len(ancestors)
        actual = round_cuda_allocation((len(leaves)+nodes-1)*4) + 3*round_cuda_allocation(nodes*4) + 4*64
        bound = estimate_spatial_index_bytes(len(leaves), core, background_shell_levels=shells,
                                             background_subdivision_level=subdivision)
        self.assertLessEqual(actual, bound)
        with self.assertRaises(ValueError):
            estimate_spatial_index_bytes(len(leaves), core, background_shell_levels=shells)
        for invalid in (-1, 10, True):
            with self.assertRaises(ValueError):
                estimate_spatial_index_bytes(448, 1024, background_shell_levels=8,
                                             background_subdivision_level=invalid)

    def test_spatial_bound_covers_exact_node_counts_for_small_occupancies(self):
        # Enumerate small occupancy subsets independently of Morton ordering.
        import itertools
        for size in (1, 2, 3, 5):
            cells = list(itertools.product(range(size), repeat=3))
            depth = (size - 1).bit_length()
            for chosen in (cells, cells[::2], cells[::5], cells[:1], []):
                nodes = sum(len({tuple(v >> level for v in xyz) for xyz in chosen})
                            for level in range(1, depth + 1))
                actual = (round_cuda_allocation(max(1, len(chosen)) * 4) +
                          3 * round_cuda_allocation(max(1, nodes) * 4) + 4 * 64)
                self.assertLessEqual(actual, estimate_spatial_index_bytes(len(chosen), size))
        for n, size in ((9, 2), (-1, 4), (1, 0), (0, 2048), (True, 4)):
            with self.assertRaises(ValueError):
                estimate_spatial_index_bytes(n, size)

    def test_sparse_high_address_shell_bound_counts_only_occupied_ancestors(self):
        import itertools
        import random
        core, shells, subdivision = 1024, 5, 8
        root, depth = core << shells, 15
        offset = (root-core)//2
        rng = random.Random(218)
        coordinates = [xyz for xyz in itertools.product(range(4), repeat=3)
                       if not all(1 <= axis <= 2 for axis in xyz)]
        leaves = {(*(offset+rng.randrange(core) for _ in range(3)), 0) for _ in range(200)}
        high_addresses = []
        for ordinal in range(1000):
            shell, coarse = rng.randrange(shells), rng.randrange(56)
            xyz = coordinates[coarse]
            child = [rng.randrange(1 << subdivision) for _ in range(3)]
            child_side = (core << shell) >> (subdivision+1)
            coarse_side = child_side << subdivision
            low = (root-(core << (shell+1)))//2
            leaves.add((*(low+xyz[a]*coarse_side+child[a]*child_side for a in range(3)),
                        child_side.bit_length()-1))
            morton = sum(((child[a] >> bit) & 1) << (3*bit+a)
                         for bit in range(subdivision) for a in range(3))
            high_addresses.append(core**3 + (shell*56+coarse)*8**subdivision + morton)
        self.assertGreater(max(high_addresses), 2**32)
        ancestors = {(level, x >> level, y >> level, z >> level)
                     for x, y, z, leaf_level in leaves for level in range(leaf_level+1, depth+1)}
        nodes = len(ancestors)
        actual = round_cuda_allocation((len(leaves)+nodes-1)*4) + 3*round_cuda_allocation(nodes*4) + 4*64
        bound = estimate_spatial_index_bytes(len(leaves), core, background_shell_levels=shells,
                                             background_subdivision_level=subdivision)
        self.assertLessEqual(actual, bound)
        self.assertLess(bound, MIB)
        self.assertEqual(estimate_spatial_index_bytes(0, core, background_shell_levels=shells,
            background_subdivision_level=subdivision), 8*64)
        # Four million sparse leaves must not be budgeted as the 4.7 billion
        # possible background addresses in this layout.
        self.assertLess(estimate_spatial_index_bytes(4_000_000, core, background_shell_levels=shells,
            background_subdivision_level=subdivision), GIB)
        self.assertLess(estimate_spatial_index_bytes(1, core, background_shell_levels=10,
            background_subdivision_level=9), MIB)

    def plan(self, **overrides):
        values = dict(self.inputs(), reference_cache_mib=4096, gpu={"memory_mib": "32607"})
        values.update(overrides)
        return plan_training_memory(**values)

    def test_model_quaternion_and_adam_tangent_widths_are_distinct(self):
        report = estimate_training_memory(**self.inputs())
        c = report["components_bytes"]
        self.assertEqual(c["model_parameters"], 100 * 38 * 4)
        self.assertEqual(c["model_cell_indices"], 100 * 8)
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
                         35.5 * GIB)
        self.assertLess(sum(report["allocated_components_bytes"][key] for key in fields), 25 * GIB)
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

    def test_compact_path_record_accounting_requires_native_capability_and_rgb(self):
        legacy = estimate_training_memory(**self.inputs())
        compact = estimate_training_memory(**self.inputs(), path_record_layout_version=1)
        c = compact["components_bytes"]
        self.assertEqual(c["hit_record_floats"], 20 * 16 * 7 * 4)
        self.assertEqual(c["reverse_suffix"], 20 * 16 * 3 * 4)
        self.assertEqual(c["geometric_candidate_state"], 20 * 16 * 7 * 4)
        rgba = estimate_training_memory(**self.inputs(), path_record_layout_version=1, rgb_only=False)
        for name in ("hit_record_floats", "reverse_suffix", "geometric_candidate_state"):
            self.assertEqual(rgba["components_bytes"][name], legacy["components_bytes"][name])
        self.assertEqual(self.plan(path_record_layout_version=1)["path_record_layout_version"], 1)
        with self.assertRaises(ValueError):
            self.plan(path_record_layout_version=True)

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
