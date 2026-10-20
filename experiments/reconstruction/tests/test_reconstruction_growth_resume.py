"""CPU regression checks for RGB growth configuration and legacy resumes."""

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

import train


def legacy_checkpoint():
    config = asdict(train.CONFIG)
    config["topology"].pop("growth_rgb_normalization")
    return {"config": config,
            "training_state": {"dataset_signature": "same-training-images"}}


def explicit_checkpoint(previous):
    checkpoint = copy.deepcopy(previous)
    for name in ("topology", "gradient"):
        checkpoint["config"][name] = train.normalized_algorithm_group(
            name, checkpoint["config"][name])
    return checkpoint


class GrowthResumeTests(unittest.TestCase):
    def test_new_experiment_uses_observations(self):
        settings = train.CONFIG.topology
        self.assertEqual(settings.growth_rgb_normalization, "observed")
        self.assertEqual(settings.growth_freeze_last_epochs, 100)
        self.assertTrue(settings.growth_inherit_opacity)
        self.assertEqual(settings.growth_opacity_scale, .8)
        self.assertEqual(settings.growth_initial_opacity, .3)
        train.validate(copy.deepcopy(train.CONFIG))

    def test_missing_birth_mode_keeps_legacy_fixed_opacity(self):
        previous = copy.deepcopy(asdict(train.CONFIG.topology))
        previous.pop("growth_inherit_opacity")
        previous.pop("growth_opacity_scale")
        previous["growth_initial_opacity"] = .5
        normalized = train.normalized_algorithm_group("topology", previous)
        self.assertFalse(normalized["growth_inherit_opacity"])
        self.assertEqual(normalized["growth_initial_opacity"], .5)
        self.assertEqual(normalized["growth_opacity_scale"], .8)
        self.assertNotIn("growth_inherit_opacity", previous)
        self.assertEqual(normalized, train.normalized_algorithm_group("topology", {
            **previous, "growth_inherit_opacity": False, "growth_opacity_scale": .8}))
        self.assertNotEqual(normalized, train.normalized_algorithm_group("topology", {
            **previous, "growth_inherit_opacity": True, "growth_initial_opacity": .3}))

    def test_birth_mode_change_cannot_overwrite_existing_model_on_resume(self):
        previous = legacy_checkpoint()
        previous["config"]["topology"].pop("growth_inherit_opacity")
        previous["config"]["topology"].pop("growth_opacity_scale")
        previous["config"]["topology"]["growth_initial_opacity"] = .5
        explicit = explicit_checkpoint(previous)
        with tempfile.TemporaryDirectory(prefix="mitsuba-growth-opacity-resume-") as directory:
            path = Path(directory) / "model.npz"
            path.touch()
            with patch.object(train, "load_checkpoint", return_value=explicit):
                train.validate_existing_model(path, previous)
            for name, value in (("growth_inherit_opacity", True),
                                ("growth_initial_opacity", .3),
                                ("growth_opacity_scale", .7)):
                with self.subTest(name=name):
                    changed = copy.deepcopy(explicit)
                    changed["config"]["topology"][name] = value
                    with patch.object(train, "load_checkpoint", return_value=changed):
                        with self.assertRaisesRegex(ValueError, "different topology"):
                            train.validate_existing_model(path, previous)

    def test_invalid_birth_settings_rejected_before_runtime(self):
        cases = {
            "growth_inherit_opacity": (0, 1, "true", None),
            "growth_opacity_scale": (0., -1., 1.01, float("inf"), float("nan"), True, ".8", None),
            "growth_initial_opacity": (0., -1., 1., float("inf"), float("nan"), True, ".3", None),
        }
        for name, values in cases.items():
            for value in values:
                with self.subTest(name=name, value=value):
                    config = copy.deepcopy(train.CONFIG)
                    setattr(config.topology, name, value)
                    with self.assertRaisesRegex(ValueError, name):
                        train.validate(config)
        config = copy.deepcopy(train.CONFIG)
        config.topology.growth_opacity_scale = 1.
        train.validate(config)

    def test_inherited_birth_rejects_directional_opacity_before_runtime(self):
        # Disable the independent RGB-only restriction, so this checks the
        # inheritance requirement itself rather than an unrelated validator.
        config = copy.deepcopy(train.CONFIG)
        config.training.rgb_only = False
        config.data.reference_compositing = "linear"
        config.topology.use_rgb_growth = False
        config.topology.use_rgb_deletion = False
        config.topology.use_low_opacity_deletion = False
        config.model.opacity_sh_count = 4
        with self.assertRaisesRegex(ValueError, "opacity_sh_count"):
            train.validate(config)

    def test_explicit_legacy_training_restores_fixed_birth_defaults(self):
        with patch.object(sys, "argv", ["train.py", "--no-rgb-only"]):
            args = train.parse_arguments()
        config = copy.deepcopy(train.CONFIG)
        train.apply_arguments(config, args, {})
        self.assertFalse(config.topology.growth_inherit_opacity)
        self.assertEqual(config.topology.growth_initial_opacity, .5)
        train.validate(config)

    def test_explicit_birth_settings_survive_legacy_training_selection(self):
        with patch.object(sys, "argv", ["train.py", "--no-rgb-only"]):
            args = train.parse_arguments()
        for settings, expected_inherit in (
                ({"growth_initial_opacity": .25}, False),
                ({"growth_inherit_opacity": True, "growth_initial_opacity": .25}, True)):
            with self.subTest(settings=settings):
                config = copy.deepcopy(train.CONFIG)
                train.apply_arguments(config, args, {"topology": settings})
                self.assertEqual(config.topology.growth_inherit_opacity, expected_inherit)
                self.assertEqual(config.topology.growth_initial_opacity, .25)
                train.validate(config)

    def test_legacy_resume_preserves_saved_birth_mode_and_custom_opacity(self):
        with patch.object(sys, "argv", ["train.py", "--resume", "saved.npz"]):
            args = train.parse_arguments()
        for inherit in (False, True):
            with self.subTest(inherit=inherit):
                config = copy.deepcopy(train.CONFIG)
                config.training.rgb_only = False
                config.topology = train.TopologySettings(
                    growth_initial_opacity=.7, growth_inherit_opacity=inherit)
                train.apply_arguments(config, args, {})
                self.assertEqual(config.topology.growth_inherit_opacity, inherit)
                self.assertEqual(config.topology.growth_initial_opacity, .7)
                train.validate(config)

    def test_missing_tail_window_preserves_legacy_growth_schedule(self):
        previous = copy.deepcopy(asdict(train.CONFIG.topology))
        previous.pop("growth_freeze_last_epochs")
        normalized = train.normalized_algorithm_group("topology", previous)
        self.assertEqual(normalized["growth_freeze_last_epochs"], 0)
        self.assertNotIn("growth_freeze_last_epochs", previous)
        explicit = {**previous, "growth_freeze_last_epochs": 0}
        self.assertEqual(
            train.normalized_algorithm_group("topology", explicit), normalized)
        current = {**previous, "growth_freeze_last_epochs": 100}
        self.assertNotEqual(
            train.normalized_algorithm_group("topology", current), normalized)

    def test_invalid_tail_window_is_rejected_before_runtime(self):
        for value in (-1, 1.5, "100", True, None):
            with self.subTest(value=value):
                config = copy.deepcopy(train.CONFIG)
                config.topology.growth_freeze_last_epochs = value
                with self.assertRaisesRegex(ValueError, "growth_freeze_last_epochs"):
                    train.validate(config)

    def test_missing_mode_preserves_legacy_units(self):
        previous = legacy_checkpoint()["config"]["topology"]
        original = copy.deepcopy(previous)
        normalized = train.normalized_algorithm_group("topology", previous)
        self.assertEqual(normalized["growth_rgb_normalization"], "full_image")
        self.assertEqual(previous, original)

    def test_json_written_after_resume_remains_equivalent(self):
        previous = legacy_checkpoint()["config"]
        # A new save writes the previously implicit defaults explicitly.
        rewritten = json.loads(json.dumps(explicit_checkpoint(
            {"config": previous})["config"]))
        previous_json = json.loads(json.dumps(previous))
        self.assertNotEqual(rewritten["topology"], previous_json["topology"])
        for name in ("topology", "gradient"):
            self.assertEqual(
                train.normalized_algorithm_group(name, rewritten[name]),
                train.normalized_algorithm_group(name, previous_json[name]))

    def test_genuine_mode_or_support_changes_remain_different(self):
        previous = legacy_checkpoint()["config"]["topology"]
        normalized = train.normalized_algorithm_group("topology", previous)
        for name, value in (("growth_rgb_normalization", "observed"),
                            ("growth_rgb_min_support_views", 7)):
            with self.subTest(name=name):
                changed = {**previous, name: value}
                self.assertNotEqual(normalized,
                    train.normalized_algorithm_group("topology", changed))

    def test_missing_group_is_not_interpreted_as_default_group(self):
        self.assertIsNone(train.normalized_algorithm_group("topology", None))
        self.assertNotEqual(train.normalized_algorithm_group("topology", None),
                            train.normalized_algorithm_group("topology", {}))
        render = {"spp": 1}
        self.assertEqual(train.normalized_algorithm_group("render", render), render)

    def test_existing_model_accepts_equivalent_legacy_and_explicit_defaults(self):
        previous = legacy_checkpoint()
        explicit = explicit_checkpoint(previous)
        with tempfile.TemporaryDirectory(prefix="mitsuba-growth-resume-") as directory:
            path = Path(directory) / "model.npz"
            path.touch()
            for existing, resumed in ((explicit, previous), (previous, explicit)):
                with self.subTest(existing_explicit=existing is explicit):
                    with patch.object(train, "load_checkpoint", return_value=existing):
                        train.validate_existing_model(path, resumed)

    def test_existing_model_rejects_changed_mode_or_support(self):
        previous = legacy_checkpoint()
        with tempfile.TemporaryDirectory(prefix="mitsuba-growth-resume-") as directory:
            path = Path(directory) / "model.npz"
            path.touch()
            for name, value in (("growth_rgb_normalization", "observed"),
                                ("growth_rgb_min_support_views", 7)):
                with self.subTest(name=name):
                    existing = explicit_checkpoint(previous)
                    existing["config"]["topology"][name] = value
                    with patch.object(train, "load_checkpoint", return_value=existing):
                        with self.assertRaisesRegex(ValueError, "different topology"):
                            train.validate_existing_model(path, previous)

    def test_parser_accepts_explicit_legacy_normalization(self):
        with patch.object(sys, "argv", ["train.py"]):
            args = train.parse_arguments()
        config = copy.deepcopy(train.CONFIG)
        train.apply_arguments(config, args, {"topology": {
            "growth_rgb_normalization": "full_image"}})
        train.validate(config)
        self.assertEqual(config.topology.growth_rgb_normalization, "full_image")

    def test_removed_cap_fields_are_rejected_in_json_and_checkpoints(self):
        for name in ("growth_max_candidates", "growth_max_new_voxels"):
            for value in (0, 128, 1024):
                with self.subTest(name=name, value=value):
                    self.assertNotIn(name, asdict(train.CONFIG.topology))
                    # Both config JSON and saved checkpoint settings enter
                    # through update(); removed fields are not migrated away.
                    for values in ({"topology": {name: value}},
                                   legacy_checkpoint()["config"]):
                        values["topology"][name] = value
                        with self.assertRaisesRegex(ValueError, "configuration fields in topology"):
                            train.update(copy.deepcopy(train.CONFIG), values)

    def test_invalid_normalization_is_rejected_before_runtime(self):
        config = copy.deepcopy(train.CONFIG)
        config.topology.growth_rgb_normalization = "unrecognized"
        with self.assertRaisesRegex(ValueError, "growth_rgb_normalization"):
            train.validate(config)


if __name__ == "__main__":
    unittest.main(verbosity=2)
