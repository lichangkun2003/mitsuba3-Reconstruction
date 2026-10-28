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


def background_deletion_configuration(target, *, gpu=False):
    # Resident topology requires the RGB algorithm family even when growth is disabled.
    settings = topology.TopologySettings(enable_growth=False, use_rgb_growth=True,
        use_rgb_deletion=True, use_low_opacity_deletion=True, gpu_topology=gpu,
        deletion_warmup_epochs=0, deletion_interval=1, deletion_opacity_threshold=.1,
        deletion_opacity_confirmation_windows=2, deletion_rgb_confirmation_windows=2)
    schedule = topology.CoarseToFineConfig(enabled=True, start_resolution=4,
        target_resolution=target, total_epochs=100, save_each_level=False)
    return settings, schedule


def background_deletion_evidence(*, gain=1., observed=1, unknown=5):
    return {"deletion_gain": np.asarray([0., gain], np.float32),
            "deletion_observed": np.asarray([0, observed], np.uint32),
            "deletion_unknown": np.asarray([0, unknown], np.uint32)}


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

    def test_low_opacity_deletion_still_requires_finest_background(self):
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

    def test_rgb_deletes_translucent_background_after_confirmation_at_every_level(self):
        for target in (8, 4):
            with self.subTest(target=target):
                checkpoint = projective_fixture()
                checkpoint["model_state"]["opacity_sh"][:] = 0.  # alpha=.5, above pruning threshold.
                model, optimizer = Model(checkpoint), Optimizer()
                settings, schedule = background_deletion_configuration(target)
                manager = topology.TopologyManager(model, 3, settings, schedule)
                with patch.object(topology, "model_checkpoint", side_effect=lambda model: copy.deepcopy(model.checkpoint)):
                    for boundary in (1, 2):
                        for view in range(3):
                            manager.observe_rgb_deletion(model, background_deletion_evidence(), view, boundary-1, 16)
                        replacement, event = manager.finish_epoch(model, optimizer, boundary)
                        self.assertEqual(event["deletion_opacity"]["deleted"], 0)
                        self.assertEqual(event["deletion_rgb"]["eligible_voxels"], 2)
                        if boundary == 1:
                            self.assertIsNone(replacement)
                            self.assertEqual(event["deletion_rgb"]["awaiting_confirmation"], 1)
                            np.testing.assert_array_equal(manager.rgb_deletion_streak, [0, 1])
                self.assertEqual(event["deleted"], 1)
                self.assertEqual(event["deletion_rgb"]["confirmed"], 1)
                for name, old in checkpoint["model_state"].items():
                    np.testing.assert_array_equal(replacement["model_state"][name], old.reshape(2, -1)[:1].ravel())
                np.testing.assert_array_equal(optimizer.remaps[0][0], [0])
                self.assertEqual(optimizer.remaps[0][1:], (1, {}))
                np.testing.assert_array_equal(manager.rgb_deletion_streak, [0])
                if target == 8:
                    refined = topology.refine_state(replacement, schedule)
                    metadata = refined["model_metadata"]
                    self.assertTrue(np.all(core_mask_numpy(refined["model_state"]["cell_indices"],
                        metadata["resolution"], metadata["core_resolution"], np)),
                        "Refinement must not recreate a deleted background parent")

    def test_rgb_background_deletion_requires_complete_reliable_unopposed_evidence(self):
        for case in ("missing_view", "unknown_only", "conflict"):
            with self.subTest(case=case):
                checkpoint = projective_fixture()
                checkpoint["model_state"]["opacity_sh"][:] = 0.
                model = Model(checkpoint)
                settings, schedule = background_deletion_configuration(8)
                count = 4 if case == "conflict" else 3
                manager = topology.TopologyManager(model, count, settings, schedule)
                for boundary in (1, 2):
                    views = range(2 if case == "missing_view" else count)
                    for view in views:
                        out = background_deletion_evidence(
                            gain=0. if case == "unknown_only" else -1. if view == 3 else 1.,
                            observed=0 if case == "unknown_only" else 1)
                        manager.observe_rgb_deletion(model, out, view, boundary-1, 16)
                    replacement, event = manager.finish_epoch(model, Optimizer(), boundary)
                    self.assertIsNone(replacement)
                    self.assertEqual(event["deleted"], 0)
                    np.testing.assert_array_equal(manager.rgb_deletion_streak, [0, 0])

    def test_resume_resets_only_old_background_rgb_windows(self):
        checkpoint = projective_fixture()
        checkpoint["model_state"]["opacity_sh"][:] = 0.
        model = Model(checkpoint)
        settings, schedule = background_deletion_configuration(8)
        manager = topology.TopologyManager(model, 3, settings, schedule)
        for view in range(3):
            manager.observe_rgb_deletion(model, background_deletion_evidence(), view, 0, 16)
        manager.finish_epoch(model, Optimizer(), 1)
        manager.observe_rgb_deletion(model, background_deletion_evidence(), 0, 1, 16)
        manager.low_opacity_deletion_streak[:] = [1, 0]
        manager.low_opacity_deletion_last_boundary = 1
        manager.cooldown = {int(checkpoint["model_state"]["cell_indices"][0]): 9}
        state = manager.state_dict()
        self.assertEqual(state["background_rgb_deletion_version"], 1)
        for version in (None, 0, 1):
            with self.subTest(version=version):
                saved = copy.deepcopy(state)
                if version is None:
                    saved.pop("background_rgb_deletion_version")
                else:
                    saved["background_rgb_deletion_version"] = version
                resumed = topology.TopologyManager(model, 3, settings, schedule)
                resumed._gpu_lifecycle = object()
                resumed._rgb_deletion_device = object()
                resumed.load_state_dict(saved)
                self.assertIsNone(resumed._gpu_lifecycle)
                self.assertIsNone(resumed._rgb_deletion_device)
                np.testing.assert_array_equal(resumed.low_opacity_deletion_streak, [1, 0])
                self.assertEqual(resumed.low_opacity_deletion_last_boundary, 1)
                self.assertEqual(resumed.cooldown, manager.cooldown)
                np.testing.assert_array_equal(resumed.growth_birth, manager.growth_birth)
                if version == 1:
                    self.assertEqual(resumed.pending_rgb_deletion["seen_views"], {0})
                    np.testing.assert_array_equal(resumed.pending_rgb_deletion["gain_sum"],
                                                  state["pending_rgb_deletion"]["gain_sum"])
                    np.testing.assert_array_equal(resumed.rgb_deletion_streak, [0, 1])
                    self.assertEqual(resumed.rgb_deletion_last_boundary, 1)
                else:
                    self.assertIsNone(resumed.pending_rgb_deletion)
                    np.testing.assert_array_equal(resumed.rgb_deletion_streak, [0, 0])
                    self.assertIsNone(resumed.rgb_deletion_last_boundary)
                self.assertEqual(resumed.state_dict()["background_rgb_deletion_version"], 1)
        for version in (2, -1, True):
            with self.subTest(invalid_version=version):
                with self.assertRaisesRegex(ValueError, "background RGB deletion version"):
                    manager.load_state_dict(dict(state, background_rgb_deletion_version=version))


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
            actual = device.evaluate_rgb_deletion(1, {})
            np.testing.assert_array_equal(np.asarray(actual), [True, True])

    def test_device_rgb_background_commit_preserves_survivor_parameters_and_adam(self):
        import mitsuba as mi
        mi.set_variant("cuda_ad_rgb")
        from reconstruction_common import restore_model
        from reconstruction_optimizer import FalcorOptimizer
        for target in (8, 4):
            with self.subTest(target=target):
                checkpoint = projective_fixture()
                checkpoint["model_state"]["opacity_sh"][:] = 0.
                model = restore_model(checkpoint, mi)
                settings, schedule = background_deletion_configuration(target, gpu=True)
                manager = topology.TopologyManager(model, 3, settings, schedule)
                optimizer = FalcorOptimizer(model)
                original_adam = optimizer.state_dict()
                for group in original_adam["groups"].values():
                    group["mean"][:] = np.arange(len(group["mean"]), dtype=np.float32) + .25
                    group["squared_mean"][:] = .125
                    group["steps"][:] = [7, 11]
                optimizer.load_state_dict(original_adam)
                for boundary in (1, 2):
                    out = {name: (mi.Float if name == "deletion_gain" else mi.UInt32)(value)
                           for name, value in background_deletion_evidence().items()}
                    for view in range(3):
                        manager.observe_rgb_deletion(model, out, view, boundary-1, 16)
                    with patch.object(topology, "model_checkpoint", side_effect=AssertionError("GPU commit copied model to host")):
                        replacement, event = manager.finish_epoch(model, optimizer, boundary)
                    self.assertEqual(event["deletion_opacity"]["deleted"], 0)
                    self.assertEqual(event["deletion_rgb"]["eligible_voxels"], 2)
                    if boundary == 1:
                        self.assertIsNone(replacement)
                        self.assertEqual(event["deletion_rgb"]["awaiting_confirmation"], 1)
                self.assertEqual(event["deleted"], 1)
                self.assertEqual(event["deletion_rgb"]["confirmed"], 1)
                for name, old in checkpoint["model_state"].items():
                    np.testing.assert_array_equal(np.asarray(replacement["model_state"][name]), old.reshape(2, -1)[:1].ravel())
                actual_adam = optimizer.state_dict()
                for name, group in original_adam["groups"].items():
                    for field, old in group.items():
                        np.testing.assert_array_equal(actual_adam["groups"][name][field], old.reshape(2, -1)[:1].ravel())

    def test_resumed_background_rgb_window_migration_reaches_device_state(self):
        import mitsuba as mi
        mi.set_variant("cuda_ad_rgb")
        from reconstruction_common import restore_model
        from reconstruction_topology_gpu import get_gpu_lifecycle
        checkpoint = projective_fixture()
        checkpoint["model_state"]["opacity_sh"][:] = 0.
        cpu_model = Model(checkpoint)
        settings, schedule = background_deletion_configuration(8)
        original = topology.TopologyManager(cpu_model, 3, settings, schedule)
        for view in range(3):
            original.observe_rgb_deletion(cpu_model, background_deletion_evidence(), view, 0, 16)
        original.finish_epoch(cpu_model, Optimizer(), 1)
        original.observe_rgb_deletion(cpu_model, background_deletion_evidence(), 0, 1, 16)
        original.low_opacity_deletion_streak[:] = [1, 0]
        original.low_opacity_deletion_last_boundary = 1
        state = original.state_dict()
        model = restore_model(checkpoint, mi)
        for version in (0, 1):
            with self.subTest(version=version):
                settings, schedule = background_deletion_configuration(8, gpu=True)
                manager = topology.TopologyManager(model, 3, settings, schedule)
                manager.load_state_dict(dict(state, background_rgb_deletion_version=version))
                device = get_gpu_lifecycle(manager, model)
                np.testing.assert_array_equal(np.asarray(device.fields["low_opacity_deletion_streak"]), [1, 0])
                np.testing.assert_array_equal(np.asarray(device.fields["rgb_deletion_streak"]),
                                              [0, version])
                if version:
                    self.assertEqual(manager.pending_rgb_deletion["seen_views"], {0})
                    np.testing.assert_allclose(np.asarray(manager.pending_rgb_deletion["gain_sum"]), [0., 1.])
                else:
                    self.assertIsNone(manager.pending_rgb_deletion)
                    self.assertIsNone(manager.rgb_deletion_last_boundary)

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
