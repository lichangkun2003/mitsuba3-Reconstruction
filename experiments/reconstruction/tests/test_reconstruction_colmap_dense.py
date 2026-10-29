"""Independent COLMAP binary/projection fixtures for dense workspace preparation."""

from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reconstruction_colmap_dense import inspect_dense_inputs, prepare_dense_model
from reconstruction_dataset import read_colmap_cameras, read_colmap_images


def write_fixture(root, simple=False, target_size=(25, 17), multiple_features=False):
    source, images = root / "sparse" / "0", root / "images_4"
    source.mkdir(parents=True)
    images.mkdir()
    parameters = (90., 50.5, 33.5) if simple else (90., 88., 50.5, 33.5)
    with (source / "cameras.bin").open("wb") as stream:
        stream.write(struct.pack("<QiiQQ", 1, 7, 0 if simple else 1, 101, 67))
        stream.write(struct.pack("<" + "d" * len(parameters), *parameters))
    points = {
        1000: ((.1, -.2, 4.), (100, 120, 140), [(100, 0), (101, 0), (102, 0)]),
        1001: ((.2, .3, 5.), (130, 150, 170), [(100, 1), (101, 1)]),
        1002: ((.4, -.5, 6.), (160, 180, 200), [(100, 2), (108, 0)]),
        1003: ((.7, .8, 7.), (190, 210, 230), [(101, 2), (102, 1), (108, 1)]),
    }
    if multiple_features:
        points[1000][2].append((102, 2))
        points[1001][2].append((101, 3))
    assigned = {pair: point_id for point_id, (_, _, track) in points.items() for pair in track}
    with (source / "images.bin").open("wb") as stream:
        stream.write(struct.pack("<Q", 9))
        for index in reversed(range(9)):
            image_id = 100 + index
            translation = (.03 * index, .02 * index, 0.)
            stream.write(struct.pack("<idddddddi", image_id, 1., 0., 0., 0., *translation, 7))
            name = f"view-{index:02d}.png"
            stream.write(name.encode() + b"\0")
            observations = 4 if multiple_features else 3
            stream.write(struct.pack("<Q", observations))
            for point_index in range(observations):
                point_id = assigned.get((image_id, point_index), -1)
                xyz = points[point_id][0] if point_id >= 0 else (.2, .1, 4.)
                x, y, z = np.asarray(xyz) + translation
                pixel = (90. * x / z + 50.5, (90. if simple else 88.) * y / z + 33.5)
                stream.write(struct.pack("<ddq", *pixel, point_id))
            Image.new("RGB", target_size, (index, 100, 150)).save(images / name)
    with (source / "points3D.bin").open("wb") as stream:
        stream.write(struct.pack("<Q", len(points)))
        for point_id, (xyz, rgb, track) in points.items():
            stream.write(struct.pack("<QdddBBBdQ", point_id, *xyz, *rgb, .2, len(track)))
            for pair in track:
                stream.write(struct.pack("<ii", *pair))
    return source, images, points


def read_detail(path):
    # Intentionally independent readers verify emitted binary IDs and tracks.
    images, points = {}, {}
    with (path / "images.bin").open("rb") as stream:
        count, = struct.unpack("<Q", stream.read(8))
        for _ in range(count):
            values = struct.unpack("<idddddddi", stream.read(64))
            name = bytearray()
            while (byte := stream.read(1)) != b"\0":
                if not byte:
                    raise ValueError("Truncated image name")
                name.extend(byte)
            count, = struct.unpack("<Q", stream.read(8))
            observations = [struct.unpack("<ddq", stream.read(24)) for _ in range(count)]
            images[values[0]] = (values, name.decode(), observations)
        assert not stream.read(1)
    with (path / "points3D.bin").open("rb") as stream:
        count, = struct.unpack("<Q", stream.read(8))
        for _ in range(count):
            values = struct.unpack("<QdddBBBdQ", stream.read(51))
            track = [struct.unpack("<ii", stream.read(8)) for _ in range(values[-1])]
            points[values[0]] = (values[1:4], values[4:7], values[7], track)
        assert not stream.read(1)
    return images, points


class DenseModelTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source, self.images, self.points = write_fixture(self.root)
        self.output = self.root / "dense" / "input_sparse"

    def test_inspect_is_read_only_and_matches_training_split(self):
        before = {path: path.read_bytes() for path in self.source.iterdir()}
        result = inspect_dense_inputs(self.source, self.images)
        self.assertEqual(result["registered_images"], 9)
        self.assertEqual(result["selected_image_count"], 7)
        self.assertEqual([image["id"] for image in result["selected_images"]], list(range(101, 108)))
        self.assertEqual([image["id"] for image in result["excluded_images"]], [100, 108])
        self.assertFalse(self.output.parent.exists())
        for path, value in before.items():
            self.assertEqual(path.read_bytes(), value)

    def test_train_tracks_ids_projection_and_source_immutability(self):
        before = {path: path.read_bytes() for path in self.source.iterdir()}
        original_images, _ = read_detail(self.source)
        result = prepare_dense_model(self.source, self.images, self.output)
        self.assertEqual((result["source_points"], result["retained_points"], result["removed_points"]), (4, 2, 2))
        self.assertEqual(result["retained_observations"], 4)
        cameras = read_colmap_cameras(self.output / "cameras.bin")
        self.assertEqual(list(cameras), [7])
        self.assertEqual((cameras[7]["width"], cameras[7]["height"]), (25, 17))
        np.testing.assert_allclose(cameras[7]["parameters"],
                                   [90 * 25 / 101, 88 * 17 / 67, 50.5 * 25 / 101, 33.5 * 17 / 67],
                                   rtol=0, atol=1e-14)
        images, points = read_detail(self.output)
        self.assertEqual(set(points), {1000, 1003})
        self.assertEqual(set(images), set(range(101, 108)))
        for point_id, (xyz, rgb, error, tracks) in points.items():
            self.assertEqual((xyz, rgb), self.points[point_id][:2])
            self.assertEqual(error, .2)
            self.assertEqual(tracks, [(image, index) for image, index in self.points[point_id][2]
                                      if image not in (100, 108)])
            for image_id, observation_index in tracks:
                pose, name, observations = images[image_id]
                self.assertEqual((pose, name), original_images[image_id][:2])
                x, y, z = np.asarray(xyz) + pose[5:8]
                fx, fy, cx, cy = cameras[7]["parameters"]
                np.testing.assert_allclose(observations[observation_index][:2], [fx * x / z + cx, fy * y / z + cy])
                self.assertEqual(observations[observation_index][2], point_id)
        self.assertEqual(images[101][2][1][2], -1)
        self.assertEqual(len(images[101][2]), 3)  # Detached indices do not shift.
        for image_id, (_, _, observations) in images.items():
            old = np.asarray(original_images[image_id][2])[:, :2]
            np.testing.assert_allclose(np.asarray(observations)[:, :2], old * [25 / 101, 17 / 67])
        for path, value in before.items():
            self.assertEqual(path.read_bytes(), value)

    def test_all_views_retains_all_valid_points_and_ids(self):
        result = prepare_dense_model(self.source, self.images, self.output, split="all")
        images, points = read_detail(self.output)
        self.assertEqual(result["retained_points"], 4)
        self.assertEqual(result["excluded_images"], [])
        self.assertEqual(set(images), set(range(100, 109)))
        for point_id, (_, _, _, track) in points.items():
            self.assertEqual(track, self.points[point_id][2])
        self.assertEqual([image["name"] for image in read_colmap_images(self.output / "images.bin")],
                         [f"view-{i:02d}.png" for i in range(9)])

    def test_simple_camera_promotes_for_independently_rounded_dimensions(self):
        source, images, _ = write_fixture(self.root / "simple", simple=True)
        prepare_dense_model(source, images, self.output)
        camera = read_colmap_cameras(self.output / "cameras.bin")[7]
        self.assertEqual(camera["model"], "PINHOLE")
        self.assertNotEqual(camera["parameters"][0], camera["parameters"][1])

    def test_multiple_features_in_one_view_do_not_count_as_two_views(self):
        source, images, _ = write_fixture(self.root / "multiple", multiple_features=True)
        result = prepare_dense_model(source, images, self.output)
        _, points = read_detail(self.output)
        self.assertEqual(result["retained_points"], 2)
        self.assertEqual(points[1000][3], [(101, 0), (102, 0), (102, 2)])
        self.assertNotIn(1001, points)  # Two surviving observations, only one view.

    def test_source_overlap_and_existing_output_are_rejected(self):
        for output in (self.source, self.source / "dense", self.source.parent,
                       self.images, self.images / "dense", self.root):
            with self.subTest(output=output):
                with self.assertRaises(ValueError):
                    prepare_dense_model(self.source, self.images, output)
        self.output.mkdir(parents=True)
        with self.assertRaises(FileExistsError):
            prepare_dense_model(self.source, self.images, self.output)

    def test_inconsistent_dimensions_fail_without_partial_model(self):
        Image.new("RGB", (26, 17)).save(self.images / "view-02.png")
        with self.assertRaisesRegex(ValueError, "inconsistent dimensions"):
            prepare_dense_model(self.source, self.images, self.output)
        self.assertFalse(self.output.exists())

    def test_missing_selected_image_is_rejected(self):
        (self.images / "view-01.png").unlink()
        with self.assertRaises(FileNotFoundError):
            inspect_dense_inputs(self.source, self.images)

    def test_heldout_image_not_required_in_training_mode(self):
        (self.images / "view-00.png").unlink()
        self.assertEqual(inspect_dense_inputs(self.source, self.images)["selected_image_count"], 7)
        with self.assertRaises(FileNotFoundError):
            inspect_dense_inputs(self.source, self.images, "all")

    def test_failed_publish_does_not_leave_partial_sparse_model(self):
        before = {path: path.read_bytes() for path in self.source.iterdir()}
        with patch("reconstruction_colmap_dense.Path.rename", side_effect=OSError("publish failed")):
            with self.assertRaisesRegex(OSError, "publish failed"):
                prepare_dense_model(self.source, self.images, self.output)
        self.assertFalse(self.output.exists())
        self.assertEqual(list(self.output.parent.iterdir()), [])
        for path, value in before.items():
            self.assertEqual(path.read_bytes(), value)


if __name__ == "__main__":
    unittest.main(verbosity=2)
