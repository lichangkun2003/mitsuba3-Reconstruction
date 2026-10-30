"""CPU-only pre-allocation guards; fake-device refinement preserves the oracle."""

from __future__ import annotations

import copy
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from reconstruction_memory import (GIB, MIB, TrainingMemoryError, plan_refinement_memory,
                                   query_cuda_memory)
from reconstruction_projective import core_mask_numpy, refine_projective_device
from reconstruction_topology import CoarseToFineConfig, refine_state
from reconstruction_topology_gpu import GPUTopologyLifecycle
from test_reconstruction_projective_streaming import ArrayDevice, expanded_fixture


class RefinementMemoryTests(unittest.TestCase):
    def request(self, **updates):
        values = {"stage": "projective_model", "active_voxels": 51_910_528,
            "exact_count": True, "allocation_voxels": 51_910_528,
            "old_voxels": 6_488_816, "existing_child_count": 1_578_880,
            "background_child_count": 50_331_648,
            "model_metadata": {"sh_count": 9, "opacity_sh_count": 1}}
        values.update(updates)
        return values

    def test_transition_fits_32_gib_without_counting_the_parent_twice(self):
        request = self.request()
        saved = copy.deepcopy(request)
        live = {"free_bytes": 24 * GIB, "total_bytes": 32 * GIB}
        report = plan_refinement_memory(request, gpu={"memory_mib": 32607}, live_memory=live)
        self.assertTrue(report["supported"], report["reason"])
        self.assertLess(report["additional_required_bytes"], 11 * GIB)
        self.assertGreater(report["static_retained_bytes"], 3 * GIB)
        self.assertEqual(request, saved)
        fallback = plan_refinement_memory(request, gpu={"memory_mib": 32607})
        self.assertTrue(fallback["supported"], fallback["reason"])
        self.assertEqual(fallback["memory_source"], "static_parent_and_child_ledger")

    def test_low_live_or_dedicated_memory_refuses_before_allocation(self):
        for live in ({"free_bytes": 2 * GIB, "total_bytes": 32 * GIB},
                     {"free_bytes": 50 * GIB, "total_bytes": 64 * GIB,
                      "dedicated_free_bytes": 2 * GIB}):
            report = plan_refinement_memory(self.request(), gpu={"memory_mib": 32607},
                                            live_memory=live)
            self.assertFalse(report["supported"])
            self.assertEqual(report["reason_code"], "refinement_exceeds_budget")
            error = TrainingMemoryError(report)
            self.assertEqual(error.plan, report)
            self.assertIn("Parent model and optimizer have not been changed", str(error))

    def test_missing_memory_is_explicit_and_cuda_virtual_total_cannot_expand_budget(self):
        self.assertFalse(plan_refinement_memory(self.request(), gpu={})["supported"])
        report = plan_refinement_memory(self.request(), gpu={"memory_mib": 4096},
            live_memory={"free_bytes": 100 * GIB, "total_bytes": 128 * GIB})
        self.assertFalse(report["supported"])
        with self.assertRaises(ValueError):
            plan_refinement_memory(self.request(), gpu={},
                                   live_memory={"free_bytes": 4, "total_bytes": 3})

    def test_core_guard_only_budgets_appearance_and_keeps_all_stored_sh(self):
        core = self.request(stage="projective_core_appearance", allocation_voxels=1_578_880,
                            core_parent_count=197_360)
        report = plan_refinement_memory(core, gpu={"memory_mib": 32607})
        self.assertTrue(report["supported"])
        self.assertLess(report["new_array_allocated_bytes"], 200 * MIB)
        self.assertGreater(report["new_array_allocated_bytes"], core["allocation_voxels"] * 27 * 4)
        forecast = self.request(stage="projective_forecast", exact_count=False,
                                active_voxels=50_331_648, allocation_voxels=50_331_648,
                                existing_child_count=0)
        report = plan_refinement_memory(forecast, gpu={"memory_mib": 32607})
        self.assertFalse(report["exact_count"])
        self.assertEqual(report["active_voxels"], 50_331_648)

    def test_cuda_query_never_initializes_a_context(self):
        calls = []

        def context(pointer):
            calls.append("context")
            pointer._obj.value = 123
            return 0

        def info(free, total):
            calls.append("memory")
            free._obj.value, total._obj.value = 12 * GIB, 32 * GIB
            return 0

        driver = SimpleNamespace(cuCtxGetCurrent=context, cuMemGetInfo_v2=info)
        with patch("ctypes.CDLL", return_value=driver), patch("ctypes.WinDLL", return_value=driver, create=True):
            report = query_cuda_memory()
        self.assertEqual(report["free_bytes"], 12 * GIB)
        self.assertEqual(calls, ["context", "memory"])

        def no_context(pointer):
            pointer._obj.value = None
            return 0

        driver.cuCtxGetCurrent = no_context
        with patch("ctypes.CDLL", return_value=driver), patch("ctypes.WinDLL", return_value=driver, create=True):
            self.assertIsNone(query_cuda_memory())
        self.assertEqual(calls, ["context", "memory"])


class ProjectiveAllocationGuardTests(unittest.TestCase):
    def setup_refinement(self):
        checkpoint = expanded_fixture(((.2, .1, .1), (1.1, .1, .1), (-1.1, .1, .1)))
        metadata, state = checkpoint["model_metadata"], checkpoint["model_state"]
        background = ~core_mask_numpy(state["cell_indices"], metadata["resolution"],
                                      metadata["core_resolution"], np)
        device = ArrayDevice(8 * int(background.sum()))
        config = CoarseToFineConfig()

        def take_rows(value, ids, width=1):
            return np.asarray(value).reshape(-1, width)[ids].ravel().copy()

        def fit_core(source, settings, maximum, _device, *, before_allocate=None):
            result = refine_state(source, settings, maximum)
            count = len(result["model_state"]["cell_indices"])
            if before_allocate is not None:
                before_allocate({"stage": "core_appearance", "active_voxels": count,
                    "exact_count": True, "allocation_voxels": count,
                    "old_voxels": len(source["model_state"]["cell_indices"]),
                    "existing_child_count": count, "background_child_count": 0,
                    "model_metadata": result["model_metadata"]})
            return result

        return checkpoint, config, device, take_rows, fit_core

    def test_guard_receives_lower_bound_then_exact_count_before_final_targets(self):
        checkpoint, config, device, take_rows, fit_core = self.setup_refinement()
        before = copy.deepcopy(checkpoint)
        requests = []

        def guard(request):
            self.assertEqual(device.targets, {})
            requests.append(request)

        with patch("reconstruction_topology_gpu.take_rows", side_effect=take_rows):
            result = refine_projective_device(checkpoint, config, 1000, fit_core, device,
                                               before_allocate=guard)
        oracle = refine_state(checkpoint, config, 1000)
        for key, value in result["model_state"].items():
            np.testing.assert_allclose(value, oracle["model_state"][key], rtol=2e-6, atol=2e-6)
            np.testing.assert_array_equal(checkpoint["model_state"][key], before["model_state"][key])
        self.assertEqual([r["stage"] for r in requests],
                         ["projective_forecast", "projective_core_appearance", "projective_model"])
        self.assertFalse(requests[0]["exact_count"])
        for request in requests[1:]:
            self.assertTrue(request["exact_count"])
            self.assertEqual(request["active_voxels"], len(result["model_state"]["cell_indices"]))
            self.assertEqual(request["model_metadata"]["scene_mapping"], "projective_shell")

    def test_refusal_preserves_source_and_does_not_allocate_final_arrays(self):
        for stage in ("projective_forecast", "projective_core_appearance", "projective_model"):
            with self.subTest(stage=stage):
                checkpoint, config, device, take_rows, fit_core = self.setup_refinement()
                before = copy.deepcopy(checkpoint)

                def refuse(request):
                    if request["stage"] == stage:
                        raise TrainingMemoryError({"reason": "test refusal"})

                with patch("reconstruction_topology_gpu.take_rows", side_effect=take_rows):
                    with self.assertRaises(TrainingMemoryError):
                        refine_projective_device(checkpoint, config, 1000, fit_core, device,
                                                 before_allocate=refuse)
                self.assertEqual(device.targets, {})
                for key, value in checkpoint["model_state"].items():
                    np.testing.assert_array_equal(value, before["model_state"][key])

    def test_lifecycle_guard_runs_before_manager_or_optimizer_mutation(self):
        model, optimizer = object(), SimpleNamespace(count=17)
        manager = SimpleNamespace(schedule=object(), settings=SimpleNamespace(max_active_voxels=100),
                                  count=17, resolution=4, core_resolution=2)
        before = dict(vars(manager))
        calls = []

        def refuse(actual_model, actual_optimizer, request):
            self.assertIs(actual_model, model)
            self.assertIs(actual_optimizer, optimizer)
            calls.append(request)
            raise TrainingMemoryError({"reason": "test refusal"})

        manager.refinement_memory_guard = refuse
        lifecycle = GPUTopologyLifecycle.__new__(GPUTopologyLifecycle)
        lifecycle.manager, lifecycle.dr = manager, SimpleNamespace()

        def refiner(*_, before_allocate):
            before_allocate({"stage": "projective_forecast"})

        with patch("reconstruction_topology_gpu.device_checkpoint", return_value={}), \
             patch("reconstruction_refinement_gpu.refine_state_gpu", side_effect=refiner):
            with self.assertRaises(TrainingMemoryError):
                lifecycle.refine(model, optimizer, 1, {})
        self.assertEqual(len(calls), 1)
        self.assertEqual(optimizer.count, 17)
        for key, value in before.items():
            self.assertIs(getattr(manager, key), value)


if __name__ == "__main__":
    unittest.main()
