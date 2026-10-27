"""Fixed physical shell refinement, with a layer budget at the final level.

Sparse generated patches exercise the actual 256/512/1024 grid addresses without
allocating a dense shell. --gpu compares consecutive CUDA refinement with the
CPU oracle. Image equality is not assumed: ordinary child fitting is unchanged.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from reconstruction_checkpoint import load_checkpoint, save_checkpoint, validate_model_state
from reconstruction_topology import CoarseToFineConfig, refine_state
from test_reconstruction_fixed_shell import canonical_state, fixture, world_centers


def final_shell_fixture(target=1024, *, empty=False):
    layers = 8 * 256 // target
    h, extent = 2 / 256, 1 + 2 * layers / 256
    # The outermost patch is intentional: cropping or radial remapping at an
    # intermediate level would lose it or move its descendants inward.
    positions = [] if empty else [[.2, .1, .1], [extent - .4*h, .001, .002],
                                  [.001, -extent + .4*h, .002]]
    result = fixture(positions, core=256, layers=layers, axes_fraction=.8)
    result["model_metadata"].pop("background_refinement")
    result["model_metadata"].pop("background_layers")
    return result


def coordinates(state, resolution):
    cells = np.asarray(state["cell_indices"], np.int64)
    return np.column_stack((cells % resolution, cells // resolution % resolution,
                            cells // (resolution * resolution)))


def assert_physical_support(test, initial, current, factor):
    """Every descendant stays in its original parent's world-space grid cell."""
    old, new = initial["model_metadata"], current["model_metadata"]
    test.assertEqual(new["resolution"], old["resolution"] * factor)
    test.assertEqual(new["core_resolution"], old["core_resolution"] * factor)
    test.assertEqual(new["grid_min"], old["grid_min"])
    test.assertEqual(new["grid_max"], old["grid_max"])
    test.assertNotEqual(new.get("background_refinement"), "fixed_layers")
    validate_model_state(current["model_state"], new)
    if not len(initial["model_state"]["cell_indices"]):
        test.assertEqual(len(current["model_state"]["cell_indices"]), 0)
        return
    parent_rgb = initial["model_state"]["radiance_sh"].reshape(-1, 4, 3)[:, 0]
    child_rgb = np.asarray(current["model_state"]["radiance_sh"]).reshape(-1, 4, 3)[:, 0]
    differences = np.max(np.abs(child_rgb[:, None] - parent_rgb[None]), axis=2)
    parent_ids = np.argmin(differences, axis=1)
    np.testing.assert_array_equal(np.min(differences, axis=1), 0)
    test.assertEqual(set(parent_ids), set(range(len(parent_rgb))), "A surviving far patch was cropped")
    parent_cells = coordinates(initial["model_state"], old["resolution"])
    child_cells = coordinates(current["model_state"], new["resolution"])
    np.testing.assert_array_equal(child_cells // factor, parent_cells[parent_ids])
    h = 2 / old["core_resolution"]
    lower = np.asarray(old["grid_min"]) + parent_cells[parent_ids] * h
    points = world_centers(current)
    test.assertTrue(np.all(points >= lower - 2e-7))
    test.assertTrue(np.all(points <= lower + h + 2e-7))


class FinalShellTests(unittest.TestCase):
    def test_sparse_shell_refines_without_moving_bounds_or_cropping_outer_patches(self):
        for target, expected_layers in ((512, (4, 8)), (1024, (2, 4, 8))):
            with self.subTest(target=target):
                initial = final_shell_fixture(target)
                current = copy.deepcopy(initial)
                for level, layers in enumerate(expected_layers):
                    if level:
                        current = refine_state(current, CoarseToFineConfig())
                    metadata = current["model_metadata"]
                    self.assertEqual((metadata["resolution"] - metadata["core_resolution"]) // 2, layers)
                    self.assertEqual(metadata["resolution"], (256 << level) + 2 * layers)
                    assert_physical_support(self, initial, current, 1 << level)

    def test_empty_shell_schedule_keeps_bounds(self):
        initial = final_shell_fixture(empty=True)
        current = initial
        for level in (1, 2):
            current = refine_state(current, CoarseToFineConfig())
            assert_physical_support(self, initial, current, 1 << level)

    def test_middle_checkpoint_retains_fixed_width_continuation(self):
        middle = refine_state(final_shell_fixture(), CoarseToFineConfig())
        config = {"data": {"background_final_layers": 8, "initialization": {
            "version": 4, "scene_mode": "unbounded", "background_refinement": "preserve_physical_width",
            "background_final_layers": 8, "initial_background_layers": 2,
            "start_core_resolution": 256, "target_core_resolution": 1024,
            "shell_layers": 8, "shell_width": 1/64}}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "middle.npz"
            save_checkpoint(path, middle["model_state"], middle["model_metadata"], config=config)
            restored = load_checkpoint(path)
        self.assertEqual(restored["config"], config)
        direct = refine_state(middle, CoarseToFineConfig())
        resumed = refine_state(restored, CoarseToFineConfig())
        self.assertEqual(direct["model_metadata"], resumed["model_metadata"])
        for name, value in direct["model_state"].items():
            np.testing.assert_array_equal(value, resumed["model_state"][name], err_msg=name)
        self.assertEqual(resumed["model_metadata"]["resolution"], 1040)


def check_device_equivalence(mi, dr):
    from reconstruction_common import model_metadata, restore_model
    from reconstruction_refinement_gpu import refine_state_gpu

    checked = []
    test = unittest.TestCase()
    for target, empty in ((512, False), (1024, False), (1024, True)):
        initial = final_shell_fixture(target, empty=empty)
        reference = copy.deepcopy(initial)
        device = {"model_metadata": dict(initial["model_metadata"]), "model_state": {
            key: (mi.UInt32(value) if key == "cell_indices" else mi.Float(value))
            for key, value in initial["model_state"].items()}}
        factor = 1
        while reference["model_metadata"]["core_resolution"] < target:
            reference = refine_state(reference, CoarseToFineConfig())
            device = refine_state_gpu(device, CoarseToFineConfig())
            dr.eval(*device["model_state"].values())
            dr.sync_thread()
            factor *= 2
            host = {"model_metadata": dict(device["model_metadata"]), "model_state": {
                key: np.asarray(value) for key, value in device["model_state"].items()}}
            assert_physical_support(test, initial, host, factor)
            assert reference["model_metadata"] == host["model_metadata"]
            for key, value in canonical_state(reference).items():
                actual = canonical_state(host)[key]
                if key == "cell_indices":
                    np.testing.assert_array_equal(actual, value)
                else:
                    np.testing.assert_allclose(actual, value, rtol=1e-5, atol=1e-6, err_msg=key)
        if not empty:
            restored = restore_model(device, mi)
            assert model_metadata(restored) == device["model_metadata"]
            for key, value in restored.snapshot().items():
                np.testing.assert_array_equal(np.asarray(value), np.asarray(device["model_state"][key]))
        checked.append({"target": target, "empty": empty,
                        "final_grid": reference["model_metadata"]["resolution"]})
    return {"status": "passed", "check": "final_shell_cpu_cuda_equivalence", "cases": checked}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(FinalShellTests))
    if not result.wasSuccessful():
        return 1
    if args.gpu:
        from preview import RuntimeConfig
        from reconstruction_common import initialize_runtime
        with tempfile.TemporaryDirectory(prefix="mitsuba-final-shell-") as directory:
            mi, dr, _, handles = initialize_runtime(RuntimeConfig(), Path(directory), {})
            print(json.dumps(check_device_equivalence(mi, dr)), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
