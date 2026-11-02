"""CPU checks for projective background geometry selection and compatibility."""

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

import preview
import train


class BackgroundGeometryConfigTests(unittest.TestCase):
    def test_default_and_bicycle_keep_ellipsoids(self):
        self.assertIs(preview.ModelConfig().background_use_ellipsoid, True)
        config = copy.deepcopy(train.CONFIG)
        path = Path(__file__).resolve().parents[1] / "configs" / "bicycle_1024.json"
        values = json.loads(path.read_text(encoding="utf-8"))
        self.assertIs(values["model"]["background_use_ellipsoid"], True)
        train.update(config, values)
        train.validate(config)

    def test_json_requires_actual_boolean_in_preview_and_training(self):
        for value in (None, 0, 1, 0., 1., "true", "false", [], {}):
            for training in (False, True):
                with self.subTest(value=value, training=training):
                    config = copy.deepcopy(train.CONFIG) if training else preview.PreviewConfig()
                    update = train.update if training else preview.update_config
                    validate = train.validate if training else preview.validate_config
                    update(config, {"model": {"background_use_ellipsoid": value}})
                    with self.assertRaisesRegex(ValueError, "model.background_use_ellipsoid must be a boolean"):
                        validate(config)
        for value in (True, False):
            config = preview.PreviewConfig()
            preview.update_config(config, {"model": {"background_use_ellipsoid": value}})
            preview.validate_config(config)

    def test_preview_cli_overrides_json_and_omission_preserves_it(self):
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            path = workspace / "geometry.json"
            for saved in (True, False):
                path.write_text(json.dumps({"model": {"background_use_ellipsoid": saved}}), encoding="utf-8")
                for options, expected in (([], saved), (["--background-use-ellipsoid"], True),
                                          (["--no-background-use-ellipsoid"], False)):
                    with self.subTest(saved=saved, options=options), patch.object(
                            sys, "argv", ["preview.py", "--config", path.name, *options]):
                        config = preview.parse_config(workspace)
                        self.assertIs(config.model.background_use_ellipsoid, expected)
                        preview.validate_config(config)

    def test_training_cli_overrides_json_and_omission_preserves_it(self):
        for saved in (True, False):
            for options, expected in (([], saved), (["--background-use-ellipsoid"], True),
                                      (["--no-background-use-ellipsoid"], False)):
                with self.subTest(saved=saved, options=options), patch.object(
                        sys, "argv", ["train.py", *options]):
                    args = train.parse_arguments()
                    config = copy.deepcopy(train.CONFIG)
                    train.apply_arguments(config, args, {"model": {"background_use_ellipsoid": saved}})
                    self.assertIs(config.model.background_use_ellipsoid, expected)
                    train.validate(config)

    def test_legacy_model_normalizes_to_true_and_false_remains_algorithm_change(self):
        old = asdict(preview.ModelConfig())
        old.pop("background_use_ellipsoid")
        legacy = train.normalized_algorithm_group("model", old)
        self.assertIs(legacy["background_use_ellipsoid"], True)
        self.assertEqual(legacy, train.normalized_algorithm_group("model", {
            **old, "background_use_ellipsoid": True}))
        self.assertNotEqual(legacy, train.normalized_algorithm_group("model", {
            **old, "background_use_ellipsoid": False}))
        self.assertNotIn("background_use_ellipsoid", old)

    def test_existing_export_accepts_legacy_mode_and_rejects_changed_mode(self):
        old_config = asdict(train.CONFIG)
        old_config["model"].pop("background_use_ellipsoid")
        previous = {"config": old_config, "training_state": {"dataset_signature": "fixture"}}
        existing = copy.deepcopy(previous)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "model.npz"
            path.touch()
            for mode in (True, False):
                existing["config"]["model"]["background_use_ellipsoid"] = mode
                with self.subTest(mode=mode), patch.object(train, "load_checkpoint", return_value=existing):
                    if mode:
                        train.validate_existing_model(path, previous)
                    else:
                        with self.assertRaisesRegex(ValueError, "different model configuration"):
                            train.validate_existing_model(path, previous)

    def test_report_describes_selected_mode_and_keeps_saved_finite_boundary(self):
        data = preview.DataConfig(scene_mode="unbounded", initialization={
            "version": 6, "scene_mode": "unbounded", "scene_mapping": "projective_shell",
            "background_far_distance": 24., "initial_background_layers": 2, "shell_layers": 8})
        saved = copy.deepcopy(data.initialization)
        render = preview.RenderConfig()
        for mode, expected in ((True, "ellipsoid"), (False, "occupied_cell")):
            with self.subTest(mode=mode):
                protocol = preview.scene_background_protocol(
                    data, render, preview.ModelConfig(background_use_ellipsoid=mode))
                self.assertIs(protocol["background_use_ellipsoid"], mode)
                self.assertEqual(protocol["background_geometry"], expected)
                self.assertEqual(protocol["background_far_distance"], 24.)
                self.assertEqual(protocol["ray_extent"], "finite_scene_boundary")
                self.assertEqual(data.initialization, saved)
        legacy_call = preview.scene_background_protocol(data, render)
        self.assertIs(legacy_call["background_use_ellipsoid"], True)
        self.assertEqual(legacy_call["background_geometry"], "ellipsoid")

    def test_other_scene_mappings_report_no_background_geometry_switch(self):
        for mapping in ("bounded", "radial_seed_shell"):
            with self.subTest(mapping=mapping):
                data = preview.DataConfig(initialization={"scene_mapping": mapping})
                protocol = preview.scene_background_protocol(data, preview.RenderConfig(),
                    preview.ModelConfig(background_use_ellipsoid=False))
                self.assertIs(protocol["background_use_ellipsoid"], False)
                self.assertEqual(protocol["background_geometry"], "not_applicable")


if __name__ == "__main__":
    unittest.main()
