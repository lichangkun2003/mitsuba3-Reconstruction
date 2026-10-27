"""CPU integration of separate dataset initialization and shared camera rays."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch, sentinel

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from preview import RenderConfig
import reconstruction_common as common
from test_reconstruction_dataset import write_fixture
from train import TrainConfig, is_coarse_training_end


def numpy_runtime():
    """Run the production ray builder with CPU stand-ins for vector operations."""
    vector = lambda value: np.asarray(value, np.float32).reshape(3, 1)
    norm = lambda value: np.sqrt(np.sum(value * value, axis=0))
    mi = SimpleNamespace(UInt32=lambda value: np.asarray(value, np.uint32),
                         Float=lambda value: np.asarray(value, np.float32),
                         Vector3f=vector, Point3f=vector,
                         Ray3f=lambda origin, direction: SimpleNamespace(o=origin, d=direction))
    dr = SimpleNamespace(norm=norm, normalize=lambda value: value / norm(value),
                         opaque=lambda dtype, value: dtype(value))
    return mi, dr


class InitializationIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.scene = Path(self.temporary.name) / "bonsai"
        write_fixture(self.scene)
        self.config = TrainConfig()
        self.config.data.dataset = str(self.scene)

    def native(self):
        factory = Mock(return_value=sentinel.synthetic_model)
        factory.from_points.return_value = sentinel.colmap_model
        return SimpleNamespace(VoxelReconstruction=factory)

    def test_synthetic_keeps_original_constructor_camera_and_resolution(self):
        scene = Path(self.temporary.name) / "synthetic"
        scene.mkdir()
        (scene / "init_points.ply").write_text("original point cloud", encoding="utf-8")
        (scene / "reference.png").write_bytes(b"reference existence is sufficient for camera loading")
        matrix = np.eye(4)
        matrix[:3, 3] = [1., 2., 3.]
        (scene / "transforms_train.json").write_text(json.dumps({
            "camera_angle_x": .7, "frames": [{"file_path": "reference.png",
                                             "transform_matrix": matrix.tolist()}]}), encoding="utf-8")
        config = copy.deepcopy(self.config)
        config.data.dataset = str(scene)
        before = copy.deepcopy(config)
        with patch("reconstruction_dataset.prepare_colmap", side_effect=AssertionError("synthetic used COLMAP")):
            self.assertIsNone(common.prepare_dataset(config, np))
        self.assertEqual(config, before)
        mi = self.native()
        result = common.initialize_model(config, scene, mi, np)
        self.assertIs(result, sentinel.synthetic_model)
        mi.VoxelReconstruction.assert_called_once_with(
            str(scene / "init_points.ply"), 256, [-1.6 * 1.02] * 3, [1.6 * 1.02] * 3,
            config.model.opacity_threshold, config.model.sh_count, config.model.opacity_sh_count)
        mi.VoxelReconstruction.from_points.assert_not_called()
        views = common.load_views(scene, "transforms_train.json", config.render, np)
        self.assertEqual(len(views), 1)
        self.assertEqual(views[0]["camera"]["position"], [1., 3., -2.])
        self.assertEqual(views[0]["camera"]["camera_angle_x"], .7)
        self.assertNotIn("fx", views[0]["camera"])

    def test_colmap_initialization_passes_contracted_arrays_and_core_to_native(self):
        prepared = common.prepare_dataset(self.config, np)
        self.assertEqual((self.config.render.width, self.config.render.height), (16, 12))
        self.assertEqual(self.config.data.initialization, prepared["initialization"])
        self.assertEqual(prepared["initialization"]["shell_layers"], self.config.data.background_final_layers)
        self.assertEqual((prepared["resolution"], prepared["core_resolution"]), (264, 256))
        mi = self.native()
        with patch.dict(sys.modules, {"drjit": SimpleNamespace()}), patch(
                "reconstruction_background.initialize_background_model", return_value=sentinel.colmap_model) as shell:
            result = common.initialize_model(self.config, self.scene, mi, np, prepared=prepared)
        self.assertEqual(shell.call_args.kwargs,
                         {"opacity": .5, "refinement": "preserve_physical_width", "layers": 4})
        self.assertIs(result, sentinel.colmap_model)
        mi.VoxelReconstruction.assert_not_called()
        args = mi.VoxelReconstruction.from_points.call_args.args
        self.assertEqual(args[2:], (264, [-1.03125] * 3, [1.03125] * 3,
                                    self.config.model.sh_count, self.config.model.opacity_sh_count, 256))
        for actual, expected in zip(args[:2], (prepared["positions"], prepared["colors"])):
            self.assertEqual(actual.dtype, np.dtype(np.float32))
            self.assertEqual(actual.ndim, 1)
            self.assertTrue(actual.flags.c_contiguous)
            np.testing.assert_array_equal(actual, expected.reshape(-1))
        np.testing.assert_allclose(args[0].reshape(-1, 3)[-1], [1.028125, 0., 0.])

    def test_final_layer_initialization_is_independent_of_ray_hit_capacity(self):
        self.config.schedule.target_resolution = 1024
        prepared = []
        for max_hits in (1, 16, 32):
            config = copy.deepcopy(self.config)
            config.render.max_hits = max_hits
            prepared.append(common.prepare_dataset(config, np))
        for value in prepared:
            self.assertEqual(value["resolution"], 260)
            self.assertEqual(value["grid_max"], [65/64]*3)
            self.assertEqual(value["initialization"]["initial_background_layers"], 2)
            np.testing.assert_array_equal(value["positions"], prepared[0]["positions"])

    def test_saved_legacy_config_keeps_original_layout(self):
        import preview
        import train
        from reconstruction_dataset import prepare_colmap
        legacy = prepare_colmap(self.scene, shell_layers=16)["initialization"]
        legacy.update(version=3, scene_mode="unbounded", scene_mapping="radial_seed_shell",
                      background_refinement="fixed_layers", background_layers=8,
                      background_opacity=.5, shell_layers=8)
        entries = {"data": {"dataset": str(self.scene), "initialization": legacy}}
        for update in (train.update, preview.update_config):
            with self.subTest(loader=update.__module__):
                config = TrainConfig()
                update(config, entries)
                self.assertIsNone(config.data.background_final_layers)
                restored = common.prepare_dataset(config, np)
                self.assertEqual(restored["resolution"], 272)
                self.assertEqual(restored["grid_max"], [1.0625]*3)
                self.assertEqual(restored["initialization"], legacy)
                config.data.background_final_layers = 8
                with self.assertRaisesRegex(ValueError, "original background_layers"):
                    common.prepare_dataset(config, np)

    def test_fixed_resolution_uses_same_core_spacing_and_full_final_shell(self):
        self.config.schedule.enabled = False
        self.config.model.grid_resolution = 256
        prepared = common.prepare_dataset(self.config, np)
        self.assertEqual((prepared["resolution"], prepared["core_resolution"]), (272, 256))
        self.assertEqual(prepared["grid_max"], [1.0625] * 3)
        self.assertEqual((2 * prepared["grid_max"][0]) / prepared["resolution"], 2 / 256)

    def test_train_test_views_reuse_saved_transform_and_matching_reference_names(self):
        common.prepare_dataset(self.config, np)
        # A valid saved similarity transform is authoritative on every split.
        # Use one different from the automatic estimate to detect recomputation.
        self.config.data.initialization["center"] = [1., 2., 3.]
        self.config.data.initialization["radius"] = 11.
        saved = copy.deepcopy(self.config.data.initialization)
        with patch("reconstruction_dataset.read_colmap_points", side_effect=AssertionError("evaluation read points")):
            training = common.load_views(self.scene, "transforms_train.json", self.config.render,
                                         np, data=self.config.data)
            testing = common.load_views(self.scene, "transforms_test.json", self.config.render,
                                        np, data=self.config.data)
        self.assertEqual([view["index"] for view in training], list(range(1, 8)))
        self.assertEqual([view["index"] for view in testing], [0, 8])
        self.assertEqual(self.config.data.initialization, saved)
        for view in training + testing:
            camera = view["camera"]
            self.assertEqual(Path(view["reference_path"]).name, camera["image_name"])
            relative = (np.asarray([view["index"] - 4., 0., 0.]) - saved["center"]) / saved["radius"]
            expected = relative[[0, 2, 1]] * [1, 1, -1]
            np.testing.assert_allclose(camera["position"], expected)
            np.testing.assert_allclose([camera[key] for key in ("fx", "fy", "cx", "cy")],
                                       [15.5, 13., 6.75, 5.35])
            self.assertEqual(camera["near_clip"], self.config.render.near_clip)

    def test_view_loading_rejects_missing_transform_and_mismatched_render_size(self):
        with self.assertRaisesRegex(ValueError, "saved initialization"):
            common.load_views(self.scene, "transforms_test.json", self.config.render,
                              np, data=self.config.data)
        common.prepare_dataset(self.config, np)
        self.config.render.width += 1
        with self.assertRaisesRegex(ValueError, "image dimensions"):
            common.load_views(self.scene, "transforms_test.json", self.config.render,
                              np, data=self.config.data)

    def test_resume_preparation_does_not_load_or_recontract_points(self):
        common.prepare_dataset(self.config, np)
        saved = copy.deepcopy(self.config.data.initialization)
        with patch("reconstruction_dataset.read_colmap_points", side_effect=AssertionError("resume read points")), \
                patch("reconstruction_dataset.contract_points", side_effect=AssertionError("resume recompressed points")):
            resumed = common.prepare_dataset(self.config, np, load_points=False)
        self.assertIsNone(resumed["positions"])
        self.assertIsNone(resumed["colors"])
        self.assertEqual(resumed["initialization"], saved)

    def test_full_intrinsics_drive_pixel_rays_near_origin_and_far_range(self):
        render = RenderConfig(width=10, height=6, near_clip=.03, far_clip=9.5)
        camera = dict(position=[.3, -.5, .2], forward=[0., 1., 0.],
                      right=[1., 0., 0.], up=[0., 0., 1.],
                      fx=7., fy=11., cx=3.2, cy=1.8)
        indices = np.asarray([0, 5, 26, 59], np.uint32)
        offsets = np.asarray([[.15, .85], [.5, .5], [.7, .2], [.25, .75]], np.float32)
        mi, dr = numpy_runtime()
        uploaded = common.prepare_camera(render, camera, mi, dr)
        self.assertNotIn("tan_x", uploaded)
        rays = common.generate_ray_batch(render, camera, indices, offsets, mi, dr,
                                         prepared_camera=uploaded)
        x = (indices % render.width + offsets[:, 0] - camera["cx"]) / camera["fx"]
        y = (camera["cy"] - indices // render.width - offsets[:, 1]) / camera["fy"]
        direction = np.stack([x, np.ones(len(x)), y])
        length = np.linalg.norm(direction, axis=0)
        position = np.asarray(camera["position"])[:, None]
        np.testing.assert_allclose(rays.d, direction / length, rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(rays.o, position + render.near_clip * direction, atol=1e-7)
        np.testing.assert_allclose(rays.maxt, (render.far_clip - render.near_clip) * length, rtol=1e-6)
        np.testing.assert_allclose(rays.o + rays.d * rays.maxt,
                                   position + render.far_clip * direction, rtol=1e-6, atol=1e-6)

    def test_coarse_snapshot_follows_core_and_ignores_shell_padding(self):
        manager = SimpleNamespace(
            schedule=SimpleNamespace(enabled=True, start_resolution=256, target_resolution=512),
            resolution=272, core_resolution=256, level_start_epoch=0, level_budget=600)
        self.assertFalse(is_coarse_training_end(manager, 599))
        self.assertTrue(is_coarse_training_end(manager, 600))
        manager.resolution, manager.core_resolution = 544, 512
        manager.level_start_epoch, manager.level_budget = 600, 400
        self.assertFalse(is_coarse_training_end(manager, 1000))


if __name__ == "__main__":
    unittest.main(verbosity=2)
