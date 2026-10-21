"""CPU checks for complete test splits and checkpoint camera reuse in evaluation."""

from __future__ import annotations

import copy
from dataclasses import asdict
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import evaluate
import preview
import reconstruction_common as common
import reconstruction_dataset as dataset


def synthetic_fixture(root):
    root.mkdir(parents=True)
    for split, count in (("train", 17), ("test", 23)):
        frames = []
        folder = root / split
        folder.mkdir()
        for index in range(count):
            # Keep JSON order distinct from filename order.
            name = f"{split}/view_{count - index:03d}"
            transform = np.eye(4)
            transform[0, 3] = index / 10
            frames.append({"file_path": name, "transform_matrix": transform.tolist()})
            Image.new("RGBA", (16, 12), (127, 31, 63, 192)).save(root / f"{name}.png")
        (root / f"transforms_{split}.json").write_text(json.dumps({
            "camera_angle_x": .7, "frames": frames}), encoding="utf-8")


def colmap_fixture(root, count=97):
    model = root / "sparse" / "0"
    model.mkdir(parents=True)
    (root / "images_2").mkdir()
    (root / "images_4").mkdir()
    with (model / "cameras.bin").open("wb") as stream:
        stream.write(struct.pack("<QiiQQdddd", 1, 42, 1, 32, 24, 31., 26., 13.5, 10.7))
    with (model / "images.bin").open("wb") as stream:
        stream.write(struct.pack("<Q", count))
        for index in reversed(range(count)):
            name = f"image_{index:03d}.jpg"
            stream.write(struct.pack("<idddddddi", 300 + index, 1., 0., 0., 0.,
                                     -index / 10, 0., 0., 42))
            stream.write(name.encode("utf-8") + b"\0")
            stream.write(struct.pack("<Q", 0))
            Image.new("RGB", (16, 12), (23, 97, 211)).save(root / "images_2" / name)
            Image.new("RGB", (8, 6), (23, 97, 211)).save(root / "images_4" / name)
    # Deliberately omit points3D.bin and init_points.ply: evaluation needs neither.


class EvaluationDatasetTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.render = preview.RenderConfig(width=16, height=12)

    @staticmethod
    def selected(views, config, split):
        limit = evaluate.evaluation_view_limit(config, split)
        return views[:limit] if limit is not None else views

    def test_synthetic_uses_all_test_json_frames_and_first_ten_train_frames(self):
        root = self.root / "synthetic"
        synthetic_fixture(root)
        config = evaluate.EvaluationConfig()
        for split, total, selected in (("train", 17, 10), ("test", 23, 23)):
            views = common.load_views(root, f"transforms_{split}.json", self.render, np,
                                      data=preview.DataConfig(), split=split)
            self.assertEqual(len(views), total)
            self.assertEqual(len(self.selected(views, config, split)), selected)
            self.assertEqual([view["index"] for view in views], list(range(total)))
            self.assertEqual(Path(views[0]["reference_path"]).name, f"view_{total:03d}.png")

    def test_colmap_complete_holdout_is_disjoint_and_keeps_global_filename_indices(self):
        root = self.root / "bonsai"
        colmap_fixture(root)
        prepared = dataset.prepare_colmap(root, load_points=False)
        saved = {"data": asdict(preview.DataConfig(
            dataset=str(root), images="images_2", initialization=prepared["initialization"]))}
        data = evaluate.checkpoint_data_config(saved)
        config = evaluate.EvaluationConfig()
        with patch.object(dataset, "read_colmap_points", side_effect=AssertionError("No point reload")), \
                patch.object(dataset, "read_ply_points", side_effect=AssertionError("No point reload")):
            train = common.load_views(root, "transforms_train.json", self.render, np, data=data, split="train")
            test = common.load_views(root, "transforms_test.json", self.render, np, data=data, split="test")
        train_ids, test_ids = [v["index"] for v in train], [v["index"] for v in test]
        self.assertEqual(test_ids, list(range(0, 97, 8)))
        self.assertEqual(train_ids, [i for i in range(97) if i % 8])
        self.assertFalse(set(train_ids) & set(test_ids))
        self.assertEqual(sorted(train_ids + test_ids), list(range(97)))
        self.assertEqual(len(self.selected(train, config, "train")), 10)
        self.assertEqual(len(self.selected(test, config, "test")), 13)
        self.assertEqual(Path(test[1]["reference_path"]).name, "image_008.jpg")
        self.assertTrue(all(Path(v["reference_path"]).parent.name == "images_2" for v in train + test))

    def test_colmap_evaluation_preserves_saved_similarity_and_intrinsics(self):
        root = self.root / "bonsai"
        colmap_fixture(root, count=9)
        prepared = dataset.prepare_colmap(root, load_points=False)
        initialization = copy.deepcopy(prepared["initialization"])
        initialization.update(center=[1., 2., 3.], radius=9.)
        data = preview.DataConfig(images="images_2", initialization=initialization)
        views = common.load_views(root, "test", self.render, np, data=data, split="test")
        camera = views[0]["camera"]
        np.testing.assert_allclose(camera["position"], [-1/9, -3/9, 2/9])
        np.testing.assert_allclose([camera[k] for k in ("fx", "fy", "cx", "cy")],
                                   [15.5, 13., 6.75, 5.35])
        self.assertEqual((camera["width"], camera["height"]), (16, 12))
        self.assertEqual(data.initialization, initialization)

    def test_colmap_rejects_changed_image_directory_or_render_dimensions(self):
        root = self.root / "bonsai"
        colmap_fixture(root, count=9)
        initialization = dataset.prepare_colmap(root, load_points=False)["initialization"]
        data = preview.DataConfig(images="images_4", initialization=initialization)
        with self.assertRaisesRegex(ValueError, "Image directory disagrees"):
            common.load_views(root, "test", self.render, np, data=data, split="test")
        data.images = "images_2"
        with self.assertRaisesRegex(ValueError, "dimensions differ"):
            common.load_views(root, "test", preview.RenderConfig(width=800, height=800), np,
                              data=data, split="test")


if __name__ == "__main__":
    unittest.main()
