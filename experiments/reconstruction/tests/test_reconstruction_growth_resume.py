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
        train.validate(copy.deepcopy(train.CONFIG))

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
