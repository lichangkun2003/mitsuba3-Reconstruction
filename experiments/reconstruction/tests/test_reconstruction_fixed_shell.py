"""Fixed radial-layer refinement invariants on generated, sparse models.

Run CPU checks by default. --gpu additionally compares the device implementation
with the CPU result; no scene data, training, or dense background fill is needed.
The shell transfer is a geometric approximation, so these tests deliberately do
not claim image equality before and after radial compression. Interior fitting
is compared with the established refinement in world coordinates instead.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import replace
import json
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from reconstruction_topology import (CoarseToFineConfig, TopologyManager, TopologySettings,
                                     level_budgets, refine_state)


Y00 = 1. / (2. * math.sqrt(math.pi))


def fixture(positions=(), *, core=16, layers=8, axes_fraction=.7):
    """Parameter rows are recognizable by independent radiance DC labels."""
    resolution = core + 2*layers
    extent, h = resolution/core, 2/core
    positions = np.asarray(positions, np.float64).reshape(-1, 3)
    coordinates = np.floor((positions + extent)/h).astype(np.int64)
    centers = (positions + extent)/h - coordinates
    assert np.all((coordinates >= 0) & (coordinates < resolution))
    count = len(positions)
    radiance = np.zeros((count, 4, 3), np.float32)
    for row in range(count):
        radiance[row, 0] = np.asarray([.11+.12*row, .23+.07*row, .37+.03*row])/Y00
        radiance[row, 1:] = [.01, -.02, .03]
    metadata = {"resolution": resolution, "core_resolution": core,
                "grid_min": [-extent]*3, "grid_max": [extent]*3,
                "sh_count": 4, "opacity_sh_count": 1,
                "scene_mapping": "radial_seed_shell",
                "background_refinement": "fixed_layers", "background_layers": layers}
    return {"model_metadata": metadata, "model_state": {
        "cell_indices": (coordinates[:, 0] + resolution*(coordinates[:, 1] +
                                                        resolution*coordinates[:, 2])).astype(np.uint32),
        "centers": centers.astype(np.float32).ravel(),
        "log_scales": np.full(count*3, math.log(h*axes_fraction), np.float32),
        "rotations": np.tile(np.asarray([1., 0., 0., 0.], np.float32), count),
        "radiance_sh": radiance.ravel(),
        "opacity_sh": np.full(count, math.log(.7/.3)/Y00, np.float32),
    }}


def sequential_fixture():
    """Four sparse patches at real 256-scale cell sizes, not a dense shell."""
    return fixture([[.2, .1, .1], [1.018, .002, .001], [1.046, .002, .001],
                    [.002, 1.033, .001]], core=256, axes_fraction=.8)


def rotated_opacity_fixture():
    checkpoint = fixture([[.2, .1, .1], [1.42, .04, .03], [.04, 1.72, .03]])
    state = checkpoint["model_state"]
    checkpoint["model_metadata"]["opacity_sh_count"] = 4
    h = 2/checkpoint["model_metadata"]["core_resolution"]
    state["log_scales"] = np.log(h*np.asarray([[.9, .55, .7], [.8, .45, .65],
                                               [.5, .95, .7]])).astype(np.float32).ravel()
    quaternion = np.asarray([[.93, .1, -.2, .25], [.8, -.3, .4, .1], [.7, .2, .1, -.5]])
    quaternion /= np.linalg.norm(quaternion, axis=1)[:, None]
    state["rotations"] = quaternion.astype(np.float32).ravel()
    opacity = np.asarray([[math.log(.7/.3)/Y00, .11, -.23, .37],
                          [math.log(.65/.35)/Y00, -.19, .29, .13],
                          [math.log(.8/.2)/Y00, .31, .17, -.41]], np.float32)
    state["opacity_sh"] = opacity.ravel()
    radiance = state["radiance_sh"].reshape(-1, 4, 3)
    radiance[:, 1:] *= np.asarray([1., -2., 3.], np.float32)[:, None, None]
    return checkpoint


def world_centers(checkpoint):
    values, metadata = checkpoint["model_state"], checkpoint["model_metadata"]
    resolution = metadata["resolution"]
    cells = np.asarray(values["cell_indices"], np.int64)
    xyz = np.column_stack((cells % resolution, cells//resolution % resolution,
                           cells//(resolution*resolution)))
    size = (np.asarray(metadata["grid_max"]) - metadata["grid_min"])/resolution
    return metadata["grid_min"] + (xyz + np.asarray(values["centers"]).reshape(-1, 3))*size


def canonical_state(checkpoint):
    state = checkpoint["model_state"]
    count = len(state["cell_indices"])
    order = np.argsort(np.asarray(state["cell_indices"]), kind="stable")
    return {name: np.asarray(value).reshape(count, -1)[order]
            for name, value in state.items()} if count else {
                name: np.asarray(value).copy() for name, value in state.items()}


def assert_valid(test, checkpoint):
    metadata, state = checkpoint["model_metadata"], checkpoint["model_state"]
    core, layers = metadata["core_resolution"], metadata["background_layers"]
    resolution = metadata["resolution"]
    test.assertEqual(metadata["background_refinement"], "fixed_layers")
    test.assertEqual(resolution, core+2*layers)
    extent = resolution/core
    np.testing.assert_allclose(metadata["grid_min"], [-extent]*3, rtol=0, atol=1e-12)
    np.testing.assert_allclose(metadata["grid_max"], [extent]*3, rtol=0, atol=1e-12)
    cells = np.asarray(state["cell_indices"])
    test.assertEqual(len(cells), len(np.unique(cells)))
    test.assertTrue(np.all(cells < resolution**3))
    for name, values in state.items():
        test.assertTrue(np.isfinite(np.asarray(values)).all(), name)
    if len(cells):
        centers = np.asarray(state["centers"]).reshape(-1, 3)
        test.assertTrue(np.all((centers >= 0) & (centers <= 1)))
        test.assertTrue(np.all(np.exp(np.asarray(state["log_scales"])) > 0))
        np.testing.assert_allclose(np.linalg.norm(np.asarray(state["rotations"]).reshape(-1, 4), axis=1),
                                   1., rtol=1e-5, atol=1e-6)


class FixedShellTests(unittest.TestCase):
    def test_empty_model_reaches_1024_with_exactly_eight_layers(self):
        checkpoint = fixture(core=256)
        for core in (512, 1024):
            checkpoint = refine_state(checkpoint, CoarseToFineConfig())
            self.assertEqual(checkpoint["model_metadata"]["core_resolution"], core)
            self.assertEqual(checkpoint["model_metadata"]["background_layers"], 8)
            self.assertEqual(len(checkpoint["model_state"]["cell_indices"]), 0)
            assert_valid(self, checkpoint)

    def test_nonempty_sparse_model_refines_from_256_through_1024(self):
        checkpoint = sequential_fixture()
        for core in (512, 1024):
            checkpoint = refine_state(checkpoint, CoarseToFineConfig())
            assert_valid(self, checkpoint)
            self.assertEqual(checkpoint["model_metadata"]["core_resolution"], core)
            self.assertEqual(checkpoint["model_metadata"]["resolution"], core+16)
            positions = world_centers(checkpoint)
            self.assertGreater(len(positions), 0)
            self.assertLess(len(positions), 512)
            self.assertTrue(np.any(np.linalg.norm(positions, axis=1) < 1))
            self.assertTrue(np.any(np.linalg.norm(positions, axis=1) > 1))

    def test_rotated_anisotropic_shell_preserves_all_radiance_and_opacity_sh(self):
        checkpoint = rotated_opacity_fixture()
        actual = refine_state(checkpoint, CoarseToFineConfig())
        assert_valid(self, actual)
        parent_rgb = checkpoint["model_state"]["radiance_sh"].reshape(-1, 4, 3)
        parent_opacity = checkpoint["model_state"]["opacity_sh"].reshape(-1, 4)
        child_rgb = actual["model_state"]["radiance_sh"].reshape(-1, 4, 3)
        child_opacity = actual["model_state"]["opacity_sh"].reshape(-1, 4)
        differences = np.max(np.abs(child_rgb[:, None, 0] - parent_rgb[None, :, 0]), axis=2)
        parent_ids = np.argmin(differences, axis=1)
        self.assertTrue(np.all(np.min(differences, axis=1) < 1e-6))
        shell = np.linalg.norm(world_centers(actual), axis=1) > 1
        self.assertTrue(shell.any())
        self.assertTrue((~shell).any())
        # Shell radial layers have not doubled. Keep the entire learned SH
        # functions, including opacity DC, rather than attenuation/reset.
        np.testing.assert_array_equal(child_rgb[shell], parent_rgb[parent_ids[shell]])
        np.testing.assert_array_equal(child_opacity[shell], parent_opacity[parent_ids[shell]])
        for parent in (1, 2):
            self.assertTrue(np.any(parent_ids[shell] == parent))
        # Core children still follow the established DC-only optical transfer.
        np.testing.assert_array_equal(child_rgb[~shell, 1:], 0)
        np.testing.assert_array_equal(child_opacity[~shell, 1:], 0)
        self.assertTrue(np.all(child_opacity[~shell, 0] < parent_opacity[parent_ids[~shell], 0]))

    def test_interior_refinement_keeps_world_geometry_and_legacy_transfer(self):
        checkpoint = fixture([[.2, .1, .1], [-.2, -.2, -.1]], axes_fraction=.9)
        legacy = copy.deepcopy(checkpoint)
        legacy["model_metadata"].pop("background_refinement")
        legacy["model_metadata"].pop("background_layers")
        for attenuation in (.65, 1.):
            with self.subTest(attenuation=attenuation):
                config = replace(CoarseToFineConfig(), opacity_optical_depth_scale=attenuation)
                reference, actual = refine_state(legacy, config), refine_state(checkpoint, config)
                assert_valid(self, actual)
                # The grid origin moves inward by eight new cells; the interior
                # ellipsoids must nevertheless remain at identical world points.
                expected_xyz, actual_xyz = world_centers(reference), world_centers(actual)
                self.assertGreater(len(actual_xyz), 0)
                expected_order = np.lexsort(expected_xyz.T[::-1])
                actual_order = np.lexsort(actual_xyz.T[::-1])
                np.testing.assert_allclose(actual_xyz[actual_order], expected_xyz[expected_order],
                                           rtol=4e-6, atol=3e-7)
                for name in ("log_scales", "rotations", "radiance_sh", "opacity_sh"):
                    left = np.asarray(actual["model_state"][name]).reshape(len(actual_xyz), -1)[actual_order]
                    right = np.asarray(reference["model_state"][name]).reshape(len(expected_xyz), -1)[expected_order]
                    np.testing.assert_allclose(left, right, rtol=4e-6, atol=3e-7, err_msg=name)

    def test_sparse_far_patches_keep_radial_layer_and_do_not_refill_deleted_regions(self):
        # Sparse positive-X patches stand in for a background after deletion.
        # Reinitializing the entire new shell would wrongly resurrect all other
        # directions, even if its bounds and layer count were correct.
        checkpoint = fixture([[1.18, .04, .03], [1.47, .04, .03], [1.78, .04, .03]])
        config = CoarseToFineConfig()
        current = refine_state(checkpoint, config)
        assert_valid(self, current)
        positions = world_centers(current)
        self.assertGreater(len(positions), 0)
        self.assertLess(len(positions), 256)
        self.assertTrue(np.all(positions[:, 0] > .9), "Deleted opposite hemisphere was filled again")
        self.assertTrue(np.all(np.abs(positions[:, 1:]) < .4), "Growth outside surviving parent patches")
        radiance = np.asarray(current["model_state"]["radiance_sh"]).reshape(-1, 4, 3)
        parent_radiance = checkpoint["model_state"]["radiance_sh"].reshape(-1, 4, 3)
        differences = np.max(np.abs(radiance[:, None, 0] - parent_radiance[None, :, 0]), axis=2)
        parent_ids = np.argmin(differences, axis=1)
        self.assertTrue(np.all(np.min(differences, axis=1) < 1e-6), "DC radiance was not inherited")
        old_positions = world_centers(checkpoint)
        old_core = checkpoint["model_metadata"]["core_resolution"]
        new_core = current["model_metadata"]["core_resolution"]
        for parent in range(3):
            selected = positions[parent_ids == parent]
            self.assertGreater(len(selected), 0, f"Surviving radial patch {parent} disappeared")
            old_layer = (np.linalg.norm(old_positions[parent])-1)*old_core/2
            new_layers = (np.linalg.norm(selected, axis=1)-1)*new_core/2
            # Re-voxelization moves representative centers within cells. A full
            # old radial layer is a conservative tolerance, not image equality.
            self.assertLess(abs(float(np.mean(new_layers))-old_layer), 1.0)
        alpha = 1/(1+np.exp(-np.asarray(current["model_state"]["opacity_sh"])*Y00))
        self.assertTrue(np.all((alpha > 0) & (alpha <= .7+1e-6)))

    def test_neighboring_parent_collisions_are_unique_and_deterministic(self):
        checkpoint = fixture([[1.45, .04, .03], [1.45, .17, .03], [1.45, .04, .17]],
                             axes_fraction=.95)
        before = copy.deepcopy(checkpoint)
        first = refine_state(checkpoint, CoarseToFineConfig())
        second = refine_state(checkpoint, CoarseToFineConfig())
        assert_valid(self, first)
        self.assertGreater(len(first["model_state"]["cell_indices"]), 0)
        self.assertEqual(first["model_metadata"], second["model_metadata"])
        for name, value in canonical_state(first).items():
            np.testing.assert_array_equal(value, canonical_state(second)[name], err_msg=name)
        self.assertEqual(checkpoint["model_metadata"], before["model_metadata"])
        for name, value in checkpoint["model_state"].items():
            np.testing.assert_array_equal(value, before["model_state"][name], err_msg=name)

    def test_capacity_failure_leaves_source_intact(self):
        checkpoint = fixture([[.1, .1, .1]], axes_fraction=2.)
        before = copy.deepcopy(checkpoint)
        with self.assertRaisesRegex(RuntimeError, "max_active_voxels"):
            refine_state(checkpoint, CoarseToFineConfig(), max_active_voxels=1)
        self.assertEqual(checkpoint["model_metadata"], before["model_metadata"])
        for name, value in checkpoint["model_state"].items():
            np.testing.assert_array_equal(value, before["model_state"][name], err_msg=name)

    def test_two_and_three_level_budgets_keep_requested_total(self):
        for target in (512, 1024):
            weights = (6, 4) if target == 512 else (6, 2, 2)
            for total in (3, 17, 700, 1500):
                with self.subTest(target=target, total=total):
                    budgets = level_budgets(256, target, total, weights)
                    self.assertEqual(list(budgets), [256, 512] if target == 512 else [256, 512, 1024])
                    self.assertEqual(sum(budgets.values()), total)
                    self.assertTrue(all(type(v) is int and v >= 1 for v in budgets.values()))
                    self.assertEqual(budgets, level_budgets(256, target, total, weights))

    def test_manager_can_construct_at_middle_level_before_loading_saved_budgets(self):
        schedule = CoarseToFineConfig(start_resolution=256, target_resolution=1024,
                                      total_epochs=1500, level_weights=(6, 2, 2))
        settings = TopologySettings(gpu_topology=False, enable_growth=False, enable_deletion=False)

        def model(core):
            return SimpleNamespace(resolution=core+16, core_resolution=core, active_voxel_count=2,
                                   opacity_sh_count=1, background_refinement="fixed_layers",
                                   background_layers=8)

        original = TopologyManager(model(256), 3, settings, schedule)
        saved = original.state_dict()
        saved.update(resolution=528, core_resolution=512,
                     level_start_epoch=original.budgets[256], schedule_start_resolution=256)
        # Resume constructs the manager with the checkpoint's already-refined
        # model first. Rejecting three weights against only 512/1024 here would
        # prevent load_state_dict from ever restoring the original schedule.
        resumed = TopologyManager(model(512), 3, settings, schedule)
        resumed.load_state_dict(saved)
        self.assertEqual(resumed.core_resolution, 512)
        self.assertEqual(resumed.resolution, 528)
        self.assertEqual(resumed.budgets, original.budgets)
        self.assertEqual(resumed.schedule_start_resolution, 256)
        self.assertEqual(resumed.level_start_epoch, original.budgets[256])
        self.assertEqual(resumed.level_budget, original.budgets[512])
        self.assertEqual(resumed.background_layers, 8)


def check_device_equivalence(mi, dr):
    from reconstruction_refinement_gpu import refine_state_gpu

    def to_device(data):
        return {"model_metadata": dict(data["model_metadata"]), "model_state": {
            name: (mi.UInt32(value) if name == "cell_indices" else mi.Float(value))
            for name, value in data["model_state"].items()}}

    def compare_result(expected, actual, label):
        dr.eval(*actual["model_state"].values())
        dr.sync_thread()
        assert expected["model_metadata"] == actual["model_metadata"]
        reference, values = canonical_state(expected), canonical_state(actual)
        for name, value in reference.items():
            assert dr.is_array_v(type(actual["model_state"][name])), name
            if name == "cell_indices":
                np.testing.assert_array_equal(value, values[name], err_msg=f"{label}:{name}")
            else:
                np.testing.assert_allclose(value, values[name], rtol=1e-5, atol=1e-6,
                                           err_msg=f"{label}:{name}")

    cases = [fixture(core=256), fixture([[.2, .1, .1], [-.2, -.2, -.1]]),
             fixture([[1.18, .04, .03], [1.47, .04, .03], [1.78, .04, .03]]),
             fixture([[1.45, .04, .03], [1.45, .17, .03], [1.45, .04, .17]], axes_fraction=.95),
             rotated_opacity_fixture()]
    for index, data in enumerate(cases):
        expected = refine_state(data, CoarseToFineConfig())
        actual = refine_state_gpu(to_device(data), CoarseToFineConfig())
        compare_result(expected, actual, f"fixture {index}")
    reference = sequential_fixture()
    device = to_device(reference)
    levels = []
    for core in (512, 1024):
        reference = refine_state(reference, CoarseToFineConfig())
        # Feed the prior device output into the next device refinement directly.
        device = refine_state_gpu(device, CoarseToFineConfig())
        compare_result(reference, device, f"successive core {core}")
        assert device["model_metadata"]["core_resolution"] == core
        assert device["model_metadata"]["resolution"] == core+16
        count = len(device["model_state"]["cell_indices"])
        assert 0 < count < 512
        levels.append({"core_resolution": core, "resolution": core+16, "voxels": count})
    return {"check": "fixed_shell_cpu_cuda_equivalence", "fixtures": len(cases),
            "successive_levels": levels, "status": "passed"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", action="store_true", help="Also test the source-built CUDA implementation")
    arguments = parser.parse_args()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(FixedShellTests))
    if not result.wasSuccessful():
        return 1
    if arguments.gpu:
        from preview import RuntimeConfig
        from reconstruction_common import initialize_runtime
        with tempfile.TemporaryDirectory(prefix="mitsuba-fixed-shell-") as directory:
            mi, dr, _, handles = initialize_runtime(RuntimeConfig(), Path(directory), {})
            print(json.dumps(check_device_equivalence(mi, dr)), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
