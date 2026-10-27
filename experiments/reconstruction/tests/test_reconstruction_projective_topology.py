"""Projective point charts, dense shell transfer and regional topology rules."""

from __future__ import annotations

import copy
import math
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np

import reconstruction_topology as topology
from reconstruction_projective import projective_points, core_mask_numpy, face_cells_device
from test_reconstruction_fixed_shell import fixture, world_centers
from test_reconstruction_rgb_growth_lifecycle import Optimizer


def projective_fixture(positions=((.2, .1, .1), (1.1, .1, .1)), core=4, layers=1):
    result = fixture(positions, core=core, layers=layers, axes_fraction=.03)
    result["model_metadata"].update(scene_mapping="projective_shell",
                                    background_refinement="preserve_physical_width")
    return result


class Model:
    def __init__(self, checkpoint):
        self.checkpoint = checkpoint
        for name, value in checkpoint["model_metadata"].items():
            setattr(self, name, value)
        self.active_voxel_count = len(checkpoint["model_state"]["cell_indices"])

    def snapshot(self):
        return self.checkpoint["model_state"]

    def clear_growth_candidates(self):
        pass


class ProjectiveTopologyTests(unittest.TestCase):
    def test_points_use_cube_and_projective_face_equations(self):
        points = np.asarray([[.9, .9, .9], [2., .5, -.7], [-3., 1., 2.],
                             [.5, 4., 2.], [1., 2., -5.]])
        actual = projective_points(points, .25, np)
        expected = [[.9, .9, .9], [1.125, .25, -.35],
                    [-1.25+1/12, 1/3, 2/3], [.125, 1.1875, .5], [.2, .4, -1.2]]
        np.testing.assert_allclose(actual, expected, rtol=1e-7, atol=1e-7)

    def test_region_switch_maps_to_distinct_faces_without_corner_cells(self):
        t = np.asarray([.5, 1., 2., 3., 3.99])
        points = np.column_stack((.8+.6*t, np.zeros(len(t)), .8*t))
        mapped = projective_points(points, .25, np)
        np.testing.assert_allclose(mapped[:, 2], 64/15*mapped[:, 0]-4, atol=4e-7)
        # Positive-axis seam ownership is X before Z; tangent edge is inset.
        seam = projective_points(np.array([[3.2, 0., 3.2]]), .25, np)[0]
        self.assertGreater(seam[0], 1.)
        self.assertLess(seam[2], 1.)
        self.assertEqual(np.count_nonzero(np.abs(seam) > 1.), 1)

    def test_growth_uses_cube_membership_and_never_background(self):
        metadata = projective_fixture()["model_metadata"]
        coordinates = np.asarray([[1, 1, 1], [4, 4, 4], [0, 3, 3],
                                  [5, 3, 3], [3, 0, 3], [3, 3, 5]])
        np.testing.assert_array_equal(topology._growth_coordinates_allowed(coordinates, metadata, np),
                                      [True, True, False, False, False, False])

    def test_background_refinement_keeps_all_children_geometry_and_high_order_color(self):
        checkpoint = projective_fixture(((1.1, .1, .1), (-1.1, .1, .1)))
        config = topology.CoarseToFineConfig(parent_opacity_threshold=.99)
        old_centers = world_centers(checkpoint)
        current = checkpoint
        for level in (1, 2):
            current = topology.refine_state(current, config)
            count = len(current["model_state"]["cell_indices"])
            self.assertEqual(count, 2 * 8**level)
            metadata = current["model_metadata"]
            self.assertEqual(metadata["background_layers"], 2**level)
            self.assertEqual(metadata["grid_min"], checkpoint["model_metadata"]["grid_min"])
            points = world_centers(current)
            distances = np.linalg.norm(points[:, None] - old_centers[None], axis=2)
            self.assertLess(distances.min(axis=1).max(), 2e-7)
            self.assertFalse(np.any(core_mask_numpy(current["model_state"]["cell_indices"],
                                                     metadata["resolution"], metadata["core_resolution"], np)))
            sh = current["model_state"]["radiance_sh"].reshape(-1, 4, 3)
            np.testing.assert_array_equal(sh[:, 1:], np.broadcast_to([.01, -.02, .03], sh[:, 1:].shape).astype(np.float32))
            np.testing.assert_array_equal(current["model_state"]["log_scales"].reshape(-1, 3),
                                          np.broadcast_to(checkpoint["model_state"]["log_scales"][:3], (count, 3)))
            self.assertEqual(len(np.unique(current["model_state"]["cell_indices"])), count)

    def test_background_child_capacity_is_checked_before_assembly(self):
        checkpoint = projective_fixture(((1.1, .1, .1),))
        with self.assertRaisesRegex(RuntimeError, "max_active_voxels"):
            topology.refine_state(checkpoint, topology.CoarseToFineConfig(), 7)

    def test_deletion_only_low_opacity_on_finest_background(self):
        from reconstruction_deletion_rgb import initialize_pending
        for target, expected in ((8, [True, False]), (4, [True, True])):
            with self.subTest(target=target):
                checkpoint = projective_fixture()
                checkpoint["model_state"]["opacity_sh"][:] = math.log(.01/.99) * (2*math.sqrt(math.pi))
                model = Model(checkpoint)
                settings = topology.TopologySettings(enable_growth=False, use_rgb_growth=False,
                    use_rgb_deletion=True, use_low_opacity_deletion=True, gpu_topology=False,
                    deletion_warmup_epochs=0, deletion_interval=1, deletion_opacity_threshold=.1,
                    deletion_opacity_confirmation_windows=1, deletion_rgb_confirmation_windows=1)
                schedule = topology.CoarseToFineConfig(enabled=True, start_resolution=4,
                                                       target_resolution=target, total_epochs=100)
                manager = topology.TopologyManager(model, 3, settings, schedule)
                opacity = manager._evaluate_low_opacity_deletion(model, 1, {})
                np.testing.assert_array_equal(opacity, expected)
                pending = initialize_pending(2, 1, model.resolution, np)
                pending["seen_views"] = {0, 1, 2}
                pending["gain_sum"][:] = 3.
                pending["observed_views"][:] = 3
                pending["support_views"][:] = 3
                manager.pending_rgb_deletion = pending
                rgb = manager._evaluate_rgb_deletion(1, {}, region_eligible=manager._core_region_numpy(model))
                np.testing.assert_array_equal(rgb, [True, False])


class ProjectiveDeviceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from preview import prepare_runtime, PreviewConfig, WORKSPACE
        cls.handles = prepare_runtime(PreviewConfig(), WORKSPACE, WORKSPACE, {"resolved": {}})
        import drjit as dr
        if not dr.has_backend(dr.JitBackend.CUDA):
            raise unittest.SkipTest("CUDA backend unavailable")
        cls.dr = dr

    def test_face_enumeration_has_exact_six_slab_occupancy(self):
        dr = self.dr
        for core, layers in ((4, 1), (8, 2)):
            resolution = core+2*layers
            cells = np.asarray(face_cells_device(core, layers, dr.cuda.UInt32, dr))
            self.assertEqual(len(cells), 6*core*core*layers)
            self.assertEqual(len(np.unique(cells)), len(cells))
            all_cells = np.arange(resolution**3)
            xyz = np.column_stack((all_cells % resolution, all_cells // resolution % resolution,
                                   all_cells // (resolution*resolution)))
            outside = (xyz < layers) | (xyz >= layers+core)
            expected = all_cells[np.count_nonzero(outside, axis=1) == 1]
            np.testing.assert_array_equal(np.sort(cells), expected)

    def test_native_point_cell_quantization_keeps_chart_boundary_seeds(self):
        import mitsuba as mi
        mi.set_variant("cuda_ad_rgb")
        points = np.asarray([[1., 1., 1.], [-1., -1., -1.], [3.2, 3.2, 3.2],
                             [-3.2, -3.2, -3.2], [0., 3.2, 3.2], [0., -3.2, -3.2],
                             [0., 0., -1.-1e-12]])
        mapped = projective_points(points, 1/64, np)
        for core, layers in ((256, 2), (512, 4), (1024, 8)):
            resolution, extent = core+2*layers, 1+1/64
            native = mi.VoxelReconstruction.from_points(mapped.ravel(), np.full(mapped.size, .4, np.float32),
                resolution, [-extent]*3, [extent]*3, 4, 1, core)
            cells = np.asarray(native.snapshot()["cell_indices"])
            self.assertEqual(len(cells), len(points))
            xyz = np.column_stack((cells % resolution, cells // resolution % resolution,
                                   cells // (resolution*resolution)))
            outside = np.count_nonzero((xyz < layers) | (xyz >= layers+core), axis=1)
            self.assertTrue(np.all(outside <= 1), "Seed assigned to an invalid atlas corner")
            self.assertEqual(np.count_nonzero(outside == 0), 2)

    def test_cpu_gpu_refinement_matches_with_core_and_background(self):
        from reconstruction_refinement_gpu import refine_state_gpu
        # Include a core ellipsoid large enough for the existing core fitter.
        checkpoint = projective_fixture()
        checkpoint["model_state"]["log_scales"][:3] = np.log(.4)
        config = topology.CoarseToFineConfig()
        for _ in range(2):
            expected = topology.refine_state(checkpoint, config)
            actual = refine_state_gpu(checkpoint, config)
            self.assertEqual(actual["model_metadata"], expected["model_metadata"])
            for name, values in actual["model_state"].items():
                np.testing.assert_allclose(np.asarray(values), expected["model_state"][name],
                                           rtol=3e-5, atol=3e-6, err_msg=name)
            checkpoint = expected

    def test_empty_projective_model_refines_on_device(self):
        from reconstruction_refinement_gpu import refine_state_gpu
        checkpoint = projective_fixture(())
        config = topology.CoarseToFineConfig()
        expected = topology.refine_state(checkpoint, config)
        actual = refine_state_gpu(checkpoint, config)
        self.assertEqual(actual["model_metadata"], expected["model_metadata"])
        for name, values in actual["model_state"].items():
            self.assertEqual(len(values), 0, name)

    def test_device_deletion_policy_matches_cpu_at_coarse_and_finest(self):
        import mitsuba as mi
        mi.set_variant("cuda_ad_rgb")
        from reconstruction_topology_gpu import GPUTopologyLifecycle
        from reconstruction_projective import core_mask_device
        dr = self.dr
        for target, expected in ((8, [True, False]), (4, [True, True])):
            checkpoint = projective_fixture()
            checkpoint["model_state"]["opacity_sh"][:] = math.log(.01/.99) * (2*math.sqrt(math.pi))
            model = Model(checkpoint)
            settings = topology.TopologySettings(enable_growth=False, use_rgb_growth=False,
                deletion_warmup_epochs=0, deletion_interval=1, deletion_opacity_threshold=.1,
                use_rgb_deletion=True, use_low_opacity_deletion=True,
                deletion_rgb_confirmation_windows=1, deletion_opacity_confirmation_windows=1)
            schedule = topology.CoarseToFineConfig(start_resolution=4, target_resolution=target,
                                                   total_epochs=100)
            manager = topology.TopologyManager(model, 3, settings, schedule)
            device = GPUTopologyLifecycle(manager)
            np.testing.assert_array_equal(np.asarray(device.evaluate_opacity(model, 1, {})), expected)
            pending = dict(cells=dr.arange(device.UInt, 2), seen_views={0, 1, 2}, boundary=1,
                           resolution=model.resolution)
            device.statistics(pending, deletion=True)
            for name in ("gain_sum", "observed_views", "support_views"):
                pending[name] = dr.full(type(pending[name]), 3, 2)
            manager.pending_rgb_deletion = pending
            region = core_mask_device(device.UInt(model.snapshot()["cell_indices"]), model.resolution,
                                       model.core_resolution, dr)
            actual = device.evaluate_rgb_deletion(1, {}, region_eligible=region)
            np.testing.assert_array_equal(np.asarray(actual), [True, False])

    def test_background_initialization_preserves_core_and_seed_colors(self):
        from reconstruction_background import initialize_background_model
        import mitsuba as mi
        mi.set_variant("cuda_ad_rgb")
        dr = self.dr
        checkpoint = projective_fixture()
        checkpoint["model_state"] = {name: (mi.UInt32 if name == "cell_indices" else mi.Float)(value)
                                     for name, value in checkpoint["model_state"].items()}
        seed = mi.VoxelReconstruction.from_state(6, [-1.5]*3, [1.5]*3,
            checkpoint["model_state"], 4, 1, core_resolution=4)
        result = initialize_background_model(seed, mi, dr, np, refinement="preserve_physical_width",
                                             layers=1, mapping="projective_shell")
        self.assertEqual(result.active_voxel_count, 97)
        self.assertEqual(result.native_scene_mapping(), "projective_shell")
        self.assertEqual(result.background_layers, 1)
        cells = np.asarray(result.snapshot()["cell_indices"])
        seed_cells = np.asarray(seed.snapshot()["cell_indices"])
        color = np.asarray(result.snapshot()["radiance_sh"]).reshape(-1, 12)
        for i, cell in enumerate(seed_cells):
            row = np.flatnonzero(cells == cell)[0]
            np.testing.assert_array_equal(color[row], np.asarray(seed.snapshot()["radiance_sh"]).reshape(-1, 12)[i])
        np.testing.assert_array_equal(color[~np.isin(cells, seed_cells)], 0.)

    def test_background_initialization_with_no_seed_points(self):
        from reconstruction_background import initialize_background_model
        import mitsuba as mi
        mi.set_variant("cuda_ad_rgb")
        state = projective_fixture(())["model_state"]
        state = {name: (mi.UInt32 if name == "cell_indices" else mi.Float)(value)
                 for name, value in state.items()}
        seed = mi.VoxelReconstruction.from_state(6, [-1.5]*3, [1.5]*3, state, 4, 1, core_resolution=4)
        result = initialize_background_model(seed, mi, self.dr, np, refinement="preserve_physical_width",
                                             layers=1, mapping="projective_shell")
        self.assertEqual(result.active_voxel_count, 96)
        self.assertEqual(result.native_scene_mapping(), "projective_shell")
        np.testing.assert_array_equal(np.asarray(result.snapshot()["radiance_sh"]), 0.)
        np.testing.assert_array_equal(np.asarray(result.snapshot()["opacity_sh"]), 0.)


if __name__ == "__main__":
    unittest.main()
