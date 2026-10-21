"""CPU integration tests for PNG scoring and serial/distributed aggregation."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image

import evaluate
from reconstruction_png_metrics import read_png_metrics


def decode_srgb(values):
    values = np.asarray(values, dtype=np.float64)
    return np.where(values <= .04045, values / 12.92, ((values + .055) / 1.055) ** 2.4)


def metric_record(index, mse):
    score = {"mse": mse, "psnr_db": -10 * math.log10(mse) if mse else None,
             "perfect_match": mse == 0}
    return {"index": index, "metrics": {name: dict(score)
                                         for name in ("linear", "srgb", "srgb_png")}}


class PNGPipelineTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.output = Path(directory.name)
        self.config = evaluate.EvaluationConfig(reference_compositing="srgb")
        self.render = SimpleNamespace(width=2, height=1, background=(1., 1., 1.))
        self.view = {"index": 8, "camera": object(), "reference_path": self.output / "source.jpg"}

    def render_record(self, prediction, target, *, split="test"):
        calls = []

        def read_reference(path, width, height, mi, numpy, **kwargs):
            calls.append(dict(path=path, width=width, height=height, **kwargs))
            return target

        common = SimpleNamespace(render_view=lambda *args: (prediction, np.ones((1, 2))),
                                 read_reference=read_reference)
        stage_seconds = dict.fromkeys(("render_seconds", "reference_seconds", "metrics_seconds", "write_seconds"), 0.)
        record = evaluate.render_evaluation_record(
            self.config, common, object(), self.render, self.view, split, self.output,
            None, None, np, dict(stage_seconds), stage_seconds)
        return record, calls, stage_seconds

    def test_real_png_pair_scored_and_aliases_match(self):
        prediction = decode_srgb(np.array([[[100.49, 100.49, 100.49], [0., 255., 128.]]]) / 255)
        target = decode_srgb(np.array([[[100., 100., 100.], [0., 255., 128.]]]) / 255)
        record, calls, timing = self.render_record(prediction, target)
        self.assertEqual(Path(record["prediction"]), self.output / "test" / "0008.png")
        self.assertEqual(Path(record["reference_png"]), self.output / "test" / "reference" / "0008.png")
        disk_score = read_png_metrics(record["prediction"], record["reference_png"], np)
        self.assertEqual(disk_score, record["metrics"]["srgb_png"])
        self.assertTrue(disk_score["perfect_match"])
        self.assertFalse(record["metrics"]["srgb"]["perfect_match"])
        self.assertIsNone(record["psnr_srgb_png"])
        self.assertTrue(record["psnr_srgb_png_is_infinite"])
        self.assertEqual(record["psnr_srgb_float"], record["metrics"]["srgb"]["psnr_db"])
        self.assertEqual(calls[0]["reference_compositing"], "srgb")
        self.assertTrue(calls[0]["rgb_only"])
        self.assertEqual(calls[0]["background"], (1., 1., 1.))
        for name, seconds in timing.items():
            self.assertTrue(math.isfinite(seconds))
            self.assertGreaterEqual(seconds, 0)
            self.assertEqual(record[name], seconds)
        json.dumps(record, allow_nan=False)

    def test_record_readback_uses_saved_files_instead_of_float_metric(self):
        prediction = np.zeros((1, 2, 3))

        def alter_file_then_read(prediction_path, reference_path, numpy):
            # Mimics an external modification between export and metric readback;
            # a float-array-derived PNG metric would fail to observe it.
            Image.fromarray(np.full((1, 2, 3), 255, np.uint8)).save(prediction_path)
            return read_png_metrics(prediction_path, reference_path, numpy)

        with patch.object(evaluate, "read_png_metrics", side_effect=alter_file_then_read):
            record, _, _ = self.render_record(prediction, prediction)
        self.assertEqual(record["metrics"]["srgb_png"]["mse"], 1.)
        self.assertEqual(record["psnr_srgb_png"], 0.)
        self.assertTrue(record["metrics"]["srgb"]["perfect_match"])

    def test_optional_linear_archive_does_not_change_png_scoring(self):
        self.config.save_linear = True
        prediction = np.full((1, 2, 3), .4)
        target = np.full((1, 2, 3), .2)
        record, _, _ = self.render_record(prediction, target, split="train")
        with np.load(self.output / "train" / "0008.npz") as saved:
            np.testing.assert_array_equal(saved["rgb_linear"], prediction)
        self.assertEqual(record["metrics"]["srgb_png"],
                         read_png_metrics(record["prediction"], record["reference_png"], np))


class PNGAggregationTests(unittest.TestCase):
    def test_arithmetic_view_psnr_not_psnr_of_mean_mse(self):
        records = [metric_record(0, .01), metric_record(1, .0001)]
        result = evaluate.aggregate_metrics(records)
        self.assertAlmostEqual(result["srgb_png"]["mean_psnr_db"], 30.)
        self.assertAlmostEqual(result["srgb_png"]["psnr_from_mean_mse_db"], -10 * math.log10(.00505))
        self.assertNotEqual(result["srgb_png"]["mean_psnr_db"], result["srgb_png"]["psnr_from_mean_mse_db"])
        self.assertEqual(result["psnr_srgb_png"], result["srgb_png"]["mean_psnr_db"])
        self.assertEqual(result["psnr_srgb_float"], result["srgb"]["mean_psnr_db"])

    def test_perfect_view_produces_json_safe_infinite_mean(self):
        result = evaluate.aggregate_metrics([metric_record(0, .01), metric_record(1, 0)])
        self.assertIsNone(result["psnr_srgb_png"])
        self.assertTrue(result["psnr_srgb_png_is_infinite"])
        self.assertFalse(result["srgb_png"]["all_views_perfect"])
        self.assertAlmostEqual(result["srgb_png"]["psnr_from_mean_mse_db"], -10 * math.log10(.005))
        json.dumps(result, allow_nan=False)

    def test_mixed_png_and_float_only_records_are_rejected(self):
        records = [metric_record(0, .01), metric_record(1, .02)]
        del records[1]["metrics"]["srgb_png"]
        with self.assertRaisesRegex(ValueError, "mixed PNG"):
            evaluate.aggregate_metrics(records)

    def test_legacy_float_only_records_remain_readable(self):
        records = [metric_record(0, .01)]
        del records[0]["metrics"]["srgb_png"]
        result = evaluate.aggregate_metrics(records)
        self.assertNotIn("psnr_srgb_png", result)
        self.assertAlmostEqual(result["psnr_srgb_float"], 20.)

    def test_two_workers_merge_matches_serial_with_unequal_train_counts(self):
        selected = {"train": [1, 5, 8], "test": [0, 3, 4, 7]}
        serial = {"train": [metric_record(1, .01), metric_record(5, .001), metric_record(8, .1)],
                  "test": [metric_record(0, .02), metric_record(3, .03),
                           metric_record(4, .04), metric_record(7, .05)]}
        # Test ownership begins after three train tasks, so rank 1 owns test/0.
        assignments = [dict(train=[1, 8], test=[3, 7]), dict(train=[5], test=[0, 4])]
        workers = []
        for rank, assignment in enumerate(assignments):
            workers.append({"rank": rank, "status": "passed", "splits": {
                split: {"views": [copy.deepcopy(record) for record in serial[split]
                                   if record["index"] in assignment[split]],
                        "seconds": rank + .5, "stage_seconds": {"metrics_seconds": .01}}
                for split in selected}})
        merged = evaluate.merge_evaluation_workers(list(reversed(workers)), selected, "srgb")
        self.assertEqual(merged["status"], "passed")
        for split in selected:
            self.assertEqual(merged["splits"][split]["aggregate"], evaluate.aggregate_metrics(serial[split]))
            self.assertEqual([record["index"] for record in merged["splits"][split]["views"]], selected[split])
            self.assertEqual(merged["splits"][split]["stage_seconds"]["metrics_seconds"], .02)
        self.assertAlmostEqual(merged["splits"]["train"]["aggregate"]["psnr_srgb_png"], 20.)
        # Mean(rank0:15 dB, rank1:30 dB) would be the incorrect 22.5 dB.
        self.assertNotEqual(merged["splits"]["train"]["aggregate"]["psnr_srgb_png"], 22.5)


if __name__ == "__main__":
    unittest.main()
