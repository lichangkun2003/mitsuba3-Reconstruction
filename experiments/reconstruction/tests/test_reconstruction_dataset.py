"""CPU regression tests for COLMAP input, projection and shell initialization."""

from __future__ import annotations

import copy
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import reconstruction_dataset as dataset


def write_fixture(root, model_id=1, names=None):
    model = root / "sparse" / "0"
    model.mkdir(parents=True)
    images = root / "images_2"
    images.mkdir()
    parameters = (31., 26., 13.5, 10.7) if model_id == 1 else (31., 13.5, 10.7, .01)
    with (model / "cameras.bin").open("wb") as stream:
        stream.write(struct.pack("<QiiQQ", 1, 7, model_id, 32, 24))
        stream.write(struct.pack("<" + "d" * len(parameters), *parameters))
    names = names or [f"view_{index:02}.png" for index in range(9)]
    with (model / "images.bin").open("wb") as stream:
        stream.write(struct.pack("<Q", len(names)))
        # Deliberately write reverse order; split membership uses filenames.
        for index in reversed(range(len(names))):
            name = names[index]
            stream.write(struct.pack("<idddddddi", 100 + index, 1., 0., 0., 0.,
                                     4. - index, 0., 0., 7))
            stream.write(name.encode("utf-8") + b"\0")
            stream.write(struct.pack("<Qddq", 1, 2., 3., -1))
            if ".." not in name and not Path(name).is_absolute():
                reference = images / name
                reference.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (16, 12), (index, 127, 255)).save(reference)
    with (model / "points3D.bin").open("wb") as stream:
        points = [(1., 0., 2.), (0., 8.8, 0.), (44., 0., 0.)]
        stream.write(struct.pack("<Q", len(points)))
        for index, point in enumerate(points):
            stream.write(struct.pack("<QdddBBBdQii", 1000 + index, *point, 255, 128, 0,
                                     .2, 1, 100, 0))
    return model


class ColmapDatasetTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "bonsai"
        self.model = write_fixture(self.root)

    def prepare(self, **kwargs):
        return dataset.prepare_colmap(self.root, **kwargs)


    def test_binary_readers_keep_ids_rgb_and_sorted_filenames(self):
        cameras = dataset.read_colmap_cameras(self.model / "cameras.bin")
        self.assertEqual(list(cameras), [7])
        self.assertEqual(cameras[7]["parameters"], (31., 26., 13.5, 10.7))
        images = dataset.read_colmap_images(self.model / "images.bin")
        self.assertEqual([image["id"] for image in images], list(range(100, 109)))
        points, colors = dataset.read_colmap_points(self.model / "points3D.bin")
        np.testing.assert_allclose(points, [(1., 0., 2.), (0., 8.8, 0.), (44., 0., 0.)])
        np.testing.assert_allclose(colors, np.tile([1., 128 / 255., 0.], (3, 1)))

    def test_separate_detection_preserves_synthetic_missing_split_error(self):
        self.assertEqual(dataset.detect_dataset(self.root), "colmap")
        (self.root / "transforms_train.json").write_text("{}")
        self.assertEqual(dataset.detect_dataset(self.root, "transforms_test.json"), "synthetic")

    def test_split_intrinsics_and_exact_core_spacing(self):
        train = self.prepare()
        test = self.prepare(split="test", load_points=False)
        self.assertEqual([view["index"] for view in train["views"]], list(range(1, 8)))
        self.assertEqual([view["index"] for view in test["views"]], [0, 8])
        self.assertEqual((train["width"], train["height"]), (16, 12))
        self.assertEqual(train["resolution"], 256)
        self.assertEqual(train["core_resolution"], 256)
        self.assertEqual(train["grid_max"], [1.] * 3)
        self.assertAlmostEqual((train["grid_max"][0] - train["grid_min"][0]) / train["resolution"], 2 / 256)
        camera = train["views"][0]["camera"]
        np.testing.assert_allclose([camera[key] for key in ("fx", "fy", "cx", "cy")], [15.5, 13., 6.75, 5.35])
        self.assertEqual(camera["image_name"], "view_01.png")
        self.assertIsInstance(train["views"][0]["reference_path"], str)
        self.assertIsNone(test["positions"])

    def test_uniform_normalization_preserves_near_point_projection(self):
        prepared = self.prepare(split="all")
        point = np.array([1., 0., 2.])
        for index, view in enumerate(prepared["views"]):
            camera = view["camera"]
            original_relative = point - np.array([index - 4., 0., 0.])
            expected = np.array([31. * original_relative[0] / original_relative[2] + 13.5,
                                 26. * original_relative[1] / original_relative[2] + 10.7]) / 2.
            relative = prepared["positions"][0] - np.array(camera["position"])
            z = np.dot(relative, camera["forward"])
            actual = [camera["fx"] * np.dot(relative, camera["right"]) / z + camera["cx"],
                      -camera["fy"] * np.dot(relative, camera["up"]) / z + camera["cy"]]
            np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
            self.assertLess(np.linalg.norm(camera["position"]), 1.)

    def test_nonidentity_colmap_rotation_uses_world_to_camera_convention(self):
        path = self.model / "images.bin"
        data = bytearray(path.read_bytes())
        # Second record is view_07, centered at world (3,0,0), looking along -X.
        record_size = 64 + len("view_08.png") + 1 + 8 + 24
        offset = 8 + record_size
        quaternion = (np.sqrt(.5), 0., np.sqrt(.5), 0.)
        rotation = dataset.quaternion_rotation(quaternion)
        translation = -rotation @ np.array([3., 0., 0.])
        struct.pack_into("<ddddddd", data, offset + 4, *quaternion, *translation)
        path.write_bytes(data)
        prepared = self.prepare(split="all")
        camera = prepared["views"][7]["camera"]
        relative = prepared["positions"][0] - np.asarray(camera["position"])
        z = np.dot(relative, camera["forward"])
        actual = [camera["fx"] * np.dot(relative, camera["right"]) / z + camera["cx"],
                  -camera["fy"] * np.dot(relative, camera["up"]) / z + camera["cy"]]
        original = rotation @ np.array([1., 0., 2.]) + translation
        expected = [(31 * original[0] / original[2] + 13.5) / 2,
                    (26 * original[1] / original[2] + 10.7) / 2]
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(camera["position"], [3 / 4.4, 0., 0.], atol=1e-12)

    def test_core_filter_and_saved_normalization_are_stable(self):
        prepared = self.prepare()
        metadata = prepared["initialization"]
        np.testing.assert_allclose(metadata["center"], [0., 0., 0.])
        self.assertAlmostEqual(metadata["radius"], 4.4)
        # Both exterior points color background leaves but never add core seeds.
        counts = prepared["background_point_counts"]
        self.assertEqual(counts.shape, (56*5,))
        self.assertEqual(int(counts.sum()), 2)
        self.assertEqual(int(np.count_nonzero(counts)), 2)
        rgb = np.asarray([1., 128/255., 0.])
        linear = np.where(rgb <= .04045, rgb/12.92, ((rgb+.055)/1.055)**2.4)
        np.testing.assert_allclose(prepared["background_colors"][counts > 0],
                                   np.tile(linear, (2, 1)), rtol=2e-7, atol=1e-8)
        np.testing.assert_array_equal(prepared["background_colors"][counts == 0], 0)
        self.assertEqual(metadata["background_color_initialization"]["colored_points"], 2)
        np.testing.assert_allclose(prepared["positions"], [[1. / 4.4, 2. / 4.4, 0.]])
        repeated = self.prepare(initialization=metadata)
        np.testing.assert_array_equal(repeated["positions"], prepared["positions"])
        self.assertEqual(repeated["initialization"], metadata)
        for option in ({"background_shell_levels": 8}, {"start_core_resolution": 128}, {"target_core_resolution": 1024}):
            with self.assertRaises(ValueError):
                self.prepare(initialization=metadata, **option)
        broken = copy.deepcopy(metadata)
        broken["radius"] = float("nan")
        with self.assertRaisesRegex(ValueError, "center or radius"):
            self.prepare(initialization=broken)















    def test_bounded_colmap_fits_points_without_radial_contraction(self):
        prepared = self.prepare(scene_mode="bounded")
        metadata = prepared["initialization"]
        self.assertEqual(metadata["scene_mapping"], "bounded")
        self.assertEqual(metadata["scene_mode"], "bounded")
        self.assertNotIn("background_shell_levels", metadata)
        self.assertEqual(prepared["resolution"], 256)
        self.assertEqual(prepared["grid_min"], [-1.] * 3)
        self.assertEqual(prepared["grid_max"], [1.] * 3)
        self.assertAlmostEqual(metadata["radius"], 44 * 1.1)
        points, _ = dataset.read_colmap_points(self.model / "points3D.bin")
        expected = ((points - metadata["center"]) / metadata["radius"]) @ dataset._AXES.T
        np.testing.assert_array_equal(prepared["positions"], expected.astype(np.float32))
        self.assertLess(np.max(np.abs(prepared["positions"])), 1.)
        for view in prepared["views"]:
            self.assertLess(np.max(np.abs(view["camera"]["position"])), 1.)
        cameras_only = self.prepare(scene_mode="bounded", load_points=False)
        self.assertEqual(cameras_only["views"], prepared["views"])
        self.assertEqual(cameras_only["initialization"], metadata)
        (self.model / "points3D.bin").unlink()
        restored = self.prepare(initialization=metadata, load_points=False)
        self.assertEqual(restored["views"], prepared["views"])


    def test_explicit_scene_settings_reject_invalid_shells_opacity_and_extent(self):
        for value in (0, -1, .5, True, float("nan"), 11):
            with self.subTest(levels=value), self.assertRaises(ValueError):
                self.prepare(scene_mode="unbounded", background_shell_levels=value)
        for value in (0, 1, -.1, float("nan"), float("inf"), True):
            with self.subTest(opacity=value), self.assertRaises(ValueError):
                self.prepare(scene_mode="unbounded", background_opacity=value)
        for options in ({"target_core_resolution": 2048}, {"start_core_resolution": 300},
                        {"background_final_layers": 8}, {"background_layers": 8}):
            with self.assertRaises(ValueError):
                self.prepare(scene_mode="unbounded", **options)
        self.assertEqual(self.prepare(scene_mode="bounded", target_core_resolution=1024)["resolution"], 256)


    def test_ply_preference_does_not_change_camera_detection(self):
        path = self.root / "init_points.ply"
        path.write_text("ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\nproperty float y\n"
                        "property float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\n"
                        "end_header\n0 0 2 32 64 255\n")
        prepared = self.prepare()
        self.assertEqual(dataset.detect_dataset(self.root), "colmap")
        self.assertEqual(prepared["point_source"], str(path))
        np.testing.assert_allclose(prepared["positions"], [[0., 2 / 4.4, 0.]])
        np.testing.assert_allclose(prepared["colors"], [[32 / 255, 64 / 255, 1.]])

    def test_binary_ply_with_additional_scalar_properties(self):
        path = self.root / "init_points.ply"
        header = ("ply\nformat binary_little_endian 1.0\nelement vertex 2\nproperty float x\n"
                  "property float y\nproperty float z\nproperty float nx\nproperty uchar red\n"
                  "property uchar green\nproperty uchar blue\nend_header\n")
        path.write_bytes(header.encode() + struct.pack("<ffffBBB", 1., 2., 3., 0., 1, 2, 255) * 2)
        positions, colors = dataset.read_ply_points(path)
        np.testing.assert_allclose(positions, [[1., 2., 3.]] * 2)
        np.testing.assert_allclose(colors, [[1 / 255., 2 / 255., 1.]] * 2)

    def test_bad_binary_counts_and_truncation_are_rejected(self):
        for name, reader in (("cameras.bin", dataset.read_colmap_cameras),
                             ("images.bin", dataset.read_colmap_images),
                             ("points3D.bin", dataset.read_colmap_points)):
            path = self.model / name
            original = path.read_bytes()
            for invalid in (original[:-1], struct.pack("<Q", 2**60) + original[8:]):
                path.write_bytes(invalid)
                with self.assertRaises(ValueError):
                    reader(path)
            path.write_bytes(original)

    def test_missing_image_and_invalid_point_colors_fail(self):
        (self.root / "images_2" / "view_08.png").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "no matching reference"):
            self.prepare()
        path = self.root / "invalid.ply"
        path.write_text("ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\nproperty float y\n"
                        "property float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\n"
                        "end_header\n0 0 2 -1 64 255\n")
        with self.assertRaisesRegex(ValueError, "colors"):
            dataset.read_ply_points(path)

    def test_mixed_reference_resolutions_fail_early(self):
        Image.new("RGB", (8, 6)).save(self.root / "images_2" / "view_08.png")
        with self.assertRaisesRegex(ValueError, "must share one resolution"):
            self.prepare()

    def test_path_traversal_and_distortion_are_rejected(self):
        other = Path(self.temporary.name) / "distorted"
        write_fixture(other, model_id=2)
        with self.assertRaisesRegex(ValueError, "undistorted PINHOLE"):
            dataset.prepare_colmap(other, images_dir="images_2")
        bad = Path(self.temporary.name) / "unsafe"
        model = write_fixture(bad, names=["../outside.png"])
        with self.assertRaisesRegex(ValueError, "relative path"):
            dataset.read_colmap_images(model / "images.bin")

    def test_rescale_preserves_offcenter_principal_point_and_unequal_focal_lengths(self):
        camera = self.prepare()["views"][0]["camera"]
        scaled = dataset.rescale_camera(camera, 40, 18)
        np.testing.assert_allclose([scaled[key] for key in ("fx", "fy", "cx", "cy")],
                                   [38.75, 19.5, 16.875, 8.025])
        restored = dataset.rescale_camera(scaled, 16, 12)
        for name in ("fx", "fy", "cx", "cy"):
            self.assertAlmostEqual(restored[name], camera[name])

    def test_shell_count_is_independent_of_foreground_schedule(self):
        for start, target in ((128, 512), (256, 1024), (1024, 1024)):
            prepared = self.prepare(start_core_resolution=start, target_core_resolution=target, background_shell_levels=7)
            self.assertEqual(prepared["resolution"], start)
            self.assertEqual(prepared["initialization"]["background_shell_levels"], 7)
            self.assertEqual(prepared["initialization"]["background_voxels"], 392)


if __name__ == "__main__":
    unittest.main()
