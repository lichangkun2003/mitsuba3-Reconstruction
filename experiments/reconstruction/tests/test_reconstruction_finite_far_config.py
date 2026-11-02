"""Finite scene boundary configuration and checkpoint compatibility checks."""

from __future__ import annotations

import copy
from dataclasses import asdict
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import evaluate
import preview
import train


class FiniteFarConfigTests(unittest.TestCase):
    def test_defaults_and_valid_boundaries(self):
        config = preview.PreviewConfig()
        self.assertIsNone(config.data.background_far_distance)
        self.assertEqual(config.data.background_far_percentile, 99.9)
        self.assertEqual(config.data.background_far_margin, 1.2)
        preview.validate_config(config)
        config.data.background_far_distance = 1.0001
        config.data.background_far_percentile = 100
        config.data.background_far_margin = 1
        preview.validate_config(config)

    def test_invalid_values_fail_before_initialization(self):
        for name, values in {
            "background_far_distance": [True, False, 0, 1, -1, "20", float("inf"), float("nan")],
            "background_far_percentile": [None, True, 0, -1, 100.1, "99", float("inf"), float("nan")],
            "background_far_margin": [None, True, 0, .99, "1.2", float("inf"), float("nan")],
        }.items():
            for value in values:
                with self.subTest(name=name, value=value):
                    config = preview.PreviewConfig()
                    setattr(config.data, name, value)
                    with self.assertRaisesRegex(ValueError, name):
                        preview.validate_config(config)

    def test_preview_json_and_cli_overrides(self):
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            (workspace / "finite.json").write_text(json.dumps({"data": {
                "background_far_distance": 30, "background_far_percentile": 98,
                "background_far_margin": 1.5}}), encoding="utf-8")
            with patch.object(sys, "argv", ["preview.py", "--config", "finite.json",
                    "--background-far-distance", "20", "--background-far-percentile", "99.5",
                    "--background-far-margin", "1.3"]):
                config = preview.parse_config(workspace)
            self.assertEqual(config.data.background_far_distance, 20)
            self.assertEqual(config.data.background_far_percentile, 99.5)
            self.assertEqual(config.data.background_far_margin, 1.3)
            preview.validate_config(config)

    def test_training_cli_and_json_null(self):
        with patch.object(sys, "argv", ["train.py", "--background-far-distance", "25",
                "--background-far-percentile", "97", "--background-far-margin", "1.4"]):
            args = train.parse_arguments()
        config = copy.deepcopy(train.CONFIG)
        train.apply_arguments(config, args, {"data": {"background_far_distance": None}})
        self.assertEqual(config.data.background_far_distance, 25)
        self.assertEqual(config.data.background_far_percentile, 97)
        self.assertEqual(config.data.background_far_margin, 1.4)
        train.validate(config)
        train.update(config, {"data": {"background_far_distance": None}})
        self.assertIsNone(config.data.background_far_distance)

    def test_v6_evaluation_uses_resolved_boundary_with_auto_request(self):
        saved_data = {"scene_mode": "unbounded", "background_far_distance": None,
            "initialization": {"version": 6, "scene_mode": "unbounded",
                "scene_mapping": "projective_shell", "background_far_distance": 24.,
                "initial_background_layers": 2, "background_final_layers": 8,
                "shell_layers": 8}}
        config = preview.PreviewConfig()
        preview.update_config(config, {"data": saved_data})
        self.assertEqual(preview.resolve_scene_background(config, "colmap"), "unbounded")
        data = evaluate.checkpoint_data_config({"data": saved_data})
        self.assertIsNone(data.background_far_distance)
        self.assertIsNone(data.background_layers)
        self.assertEqual(data.background_final_layers, 8)
        protocol = preview.scene_background_protocol(data, config.render)
        self.assertEqual(protocol["background_far_distance"], 24.)
        self.assertEqual(protocol["ray_extent"], "finite_scene_boundary")
        self.assertEqual(protocol["background_linear_rgb"], [0., 0., 0.])
        normalized = train.normalized_algorithm_group("data", asdict(data))
        self.assertNotIn("background_layers", normalized)
        self.assertEqual(normalized["initialization"], saved_data["initialization"])

    def test_old_saved_initializers_keep_previous_mapping_and_compare_defaults(self):
        for version in (1, 2, 3, 4, 5):
            with self.subTest(version=version):
                initialization = {"version": version, "scene_mode": "unbounded"}
                if version == 5:
                    initialization["scene_mapping"] = "projective_shell"
                old = {"initialization": initialization}
                data = evaluate.checkpoint_data_config({"data": old})
                self.assertIsNone(data.background_far_distance)
                self.assertEqual(data.initialization, initialization)
                protocol = preview.scene_background_protocol(data, preview.RenderConfig())
                self.assertIsNone(protocol["background_far_distance"])
                self.assertEqual(protocol["ray_extent"], "infinity" if version == 5 else "configured far_clip")
                with_defaults = {**old, "background_far_distance": None,
                    "background_far_percentile": 99.9, "background_far_margin": 1.2}
                self.assertEqual(train.normalized_algorithm_group("data", old),
                                 train.normalized_algorithm_group("data", with_defaults))
                changed = {**with_defaults, "background_far_distance": 20}
                self.assertNotEqual(train.normalized_algorithm_group("data", old),
                                    train.normalized_algorithm_group("data", changed))

    def test_v6_saved_layer_defaults_are_restored(self):
        config = preview.PreviewConfig()
        preview.update_config(config, {"data": {"initialization": {
            "version": 6, "background_final_layers": 16, "background_far_distance": 10}}})
        self.assertIsNone(config.data.background_layers)
        self.assertEqual(config.data.background_final_layers, 16)
        data = evaluate.checkpoint_data_config({"data": {"initialization": {
            "version": 6, "background_final_layers": 16, "background_far_distance": 10}}})
        self.assertEqual(data.background_final_layers, 16)


if __name__ == "__main__":
    unittest.main()
