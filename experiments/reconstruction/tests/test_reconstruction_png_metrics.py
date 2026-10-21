"""CPU tests for actual disk-PNG metrics; no renderer or project data needed."""

from __future__ import annotations

import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image

import reconstruction_png_metrics as png_metrics


def decode_srgb(srgb):
    srgb = np.asarray(srgb, dtype=np.float64)
    return np.where(srgb <= .04045, srgb / 12.92, ((srgb + .055) / 1.055) ** 2.4)


class PNGMetricsTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.prediction = self.root / "prediction" / "view.png"
        self.reference = self.root / "reference" / "view.png"

    def score(self, prediction, reference):
        return png_metrics.write_srgb_png_pair(
            self.prediction, self.reference, prediction, reference, np)

    def pixels(self, path):
        with Image.open(path) as image:
            self.assertEqual(image.mode, "RGB")
            self.assertEqual(image.format, "PNG")
            return np.array(image)

    def test_black_white_full_image_formula(self):
        score = self.score(np.zeros((2, 3, 3)), np.ones((2, 3, 3)))
        self.assertEqual(score, {"mse": 1., "psnr_db": 0., "perfect_match": False})

    def test_identical_files_report_json_safe_infinity(self):
        colors = np.array([[[0., .1, .8], [.5, 1., .002]]])
        score = self.score(colors, colors)
        self.assertEqual(score, {"mse": 0., "psnr_db": None, "perfect_match": True})

    def test_quantization_around_half_step_not_uint8_truncation(self):
        # The low sRGB branch is exactly linear, including an exact half step.
        encoded = np.array([[[.5 / 255, (.5 - 1e-6) / 255, (.5 + 1e-6) / 255],
                             [100.49 / 255, 100.51 / 255, 200.51 / 255]]])
        self.score(decode_srgb(encoded), np.zeros_like(encoded))
        np.testing.assert_array_equal(self.pixels(self.prediction),
                                      [[[1, 0, 1], [100, 101, 201]]])

    def test_decoded_rgb_reference_is_encoded_exactly_once(self):
        reference_bytes = np.array([[[17, 128, 240], [255, 0, 80]]], dtype=np.uint8)
        reference = decode_srgb(reference_bytes.astype(np.float64) / 255)
        self.score(np.zeros_like(reference), reference)
        np.testing.assert_array_equal(self.pixels(self.reference), reference_bytes)

    def test_srgb_alpha_composited_target_is_not_blended_again(self):
        # Existing target loader composites encoded RGB over white, then decodes.
        rgba = np.array([[[30, 80, 160, 128], [0, 0, 0, 0], [71, 143, 211, 255]]]) / 255
        encoded = rgba[..., :3] * rgba[..., 3:4] + (1 - rgba[..., 3:4])
        target = decode_srgb(encoded)
        self.score(target, target)
        expected = np.floor(encoded * 255 + .5).astype(np.uint8)
        np.testing.assert_array_equal(self.pixels(self.reference), expected)

    def test_metric_uses_srgb_bytes_without_gamma_decoding(self):
        encoded = np.full((2, 2, 3), 128 / 255)
        score = self.score(decode_srgb(encoded), np.zeros_like(encoded))
        expected_mse = (128 / 255) ** 2
        self.assertAlmostEqual(score["mse"], expected_mse, places=15)
        self.assertAlmostEqual(score["psnr_db"], -10 * math.log10(expected_mse), places=14)

    def test_per_pixel_rgb_full_image_mean(self):
        encoded = np.zeros((2, 2, 3))
        encoded[0, 0, 0] = 1
        score = self.score(encoded, np.zeros_like(encoded))
        self.assertAlmostEqual(score["mse"], 1 / 12, places=15)

    def test_export_reopens_files_and_metric_changes_when_disk_pixels_change(self):
        colors = np.zeros((1, 1, 3))
        original = png_metrics.read_png_metrics
        calls = []

        def replace_saved_prediction(prediction_path, reference_path, numpy):
            calls.append((prediction_path, reference_path))
            # If the implementation accidentally scores pre-export arrays, this
            # change to the actual file would incorrectly remain a perfect match.
            Image.fromarray(np.full((1, 1, 3), 255, dtype=np.uint8)).save(prediction_path)
            return original(prediction_path, reference_path, numpy)

        with patch.object(png_metrics, "read_png_metrics", side_effect=replace_saved_prediction):
            score = self.score(colors, colors)
        self.assertEqual(len(calls), 1)
        self.assertEqual(score["mse"], 1.)
        self.assertEqual(original(self.prediction, self.reference, np), score)

    def test_clip_linear_hdr_before_encoding(self):
        self.score(np.array([[[-3., 4., 0.]]]), np.zeros((1, 1, 3)))
        np.testing.assert_array_equal(self.pixels(self.prediction), [[[0, 255, 0]]])

    def test_invalid_image_dimensions_and_values_write_nothing(self):
        valid = np.zeros((2, 2, 3))
        for invalid in (np.zeros((2, 3, 3)), np.zeros((2, 2, 4)), np.zeros((0, 2, 3)),
                        np.full((2, 2, 3), np.nan), np.full((2, 2, 3), np.inf)):
            with self.subTest(shape=invalid.shape, value=invalid.flat[0] if invalid.size else None):
                with self.assertRaises(ValueError):
                    self.score(valid, invalid)
                self.assertFalse(self.prediction.parent.exists())
                self.assertFalse(self.reference.parent.exists())

    def test_same_output_path_rejected(self):
        with self.assertRaisesRegex(ValueError, "paths must differ"):
            png_metrics.write_srgb_png_pair(
                self.prediction, self.prediction, np.zeros((1, 1, 3)), np.zeros((1, 1, 3)), np)
        self.assertFalse(self.prediction.exists())

    def test_read_rejects_mismatched_dimensions_and_non_rgb(self):
        self.score(np.zeros((1, 1, 3)), np.zeros((1, 1, 3)))
        Image.fromarray(np.zeros((2, 1, 3), np.uint8)).save(self.reference)
        with self.assertRaisesRegex(ValueError, "dimensions differ"):
            png_metrics.read_png_metrics(self.prediction, self.reference, np)
        Image.fromarray(np.zeros((1, 1, 4), np.uint8)).save(self.reference)
        with self.assertRaisesRegex(ValueError, "RGB8 PNG"):
            png_metrics.read_png_metrics(self.prediction, self.reference, np)


if __name__ == "__main__":
    unittest.main()
