"""CPU regressions for RGB deletion configuration and distributed reductions."""

from __future__ import annotations

import copy
from dataclasses import asdict
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import train
from test_reconstruction_collective import _fixture, _accumulator


class RGBDeletionDriverTests(unittest.TestCase):
    def test_default_and_legacy_native_flags_are_separate(self):
        config = copy.deepcopy(train.CONFIG)
        train.validate(config)
        options = train.settings(config)
        self.assertTrue(options["collect_rgb_deletion_evidence"])
        self.assertFalse(options["collect_deletion_evidence"])
        self.assertTrue(config.topology.use_low_opacity_deletion)
        config.topology.use_low_opacity_deletion = False
        # Parameter-only pruning adds no new ray evidence or native forward.
        self.assertEqual(train.settings(config), options)
        legacy = train.TrainConfig()
        options = train.settings(legacy)
        self.assertFalse(options["collect_rgb_deletion_evidence"])
        self.assertTrue(options["collect_deletion_evidence"])
        self.assertFalse(legacy.topology.use_low_opacity_deletion)

    def test_legacy_cli_does_not_inherit_rgb_deletion(self):
        config = copy.deepcopy(train.CONFIG)
        with patch.object(sys, "argv", ["train.py", "--no-rgb-only"]):
            args = train.parse_arguments()
        train.apply_arguments(config, args, {})
        self.assertFalse(config.topology.use_rgb_deletion)
        self.assertFalse(config.topology.use_low_opacity_deletion)
        self.assertFalse(train.settings(config)["collect_rgb_deletion_evidence"])
        with self.assertRaisesRegex(ValueError, "requires training.rgb_only"):
            train.apply_arguments(config, args, {"topology": {"use_rgb_deletion": True}})
            train.validate(config)

    def test_legacy_checkpoint_has_disabled_rgb_deletion(self):
        saved = asdict(train.CONFIG.topology)
        for name in list(saved):
            if name in ("use_rgb_deletion", "use_low_opacity_deletion") or name.startswith(
                    ("deletion_rgb_", "deletion_opacity_")) or name in (
                    "deletion_warmup_epochs", "deletion_protection_epochs", "deletion_freeze_last_epochs"):
                saved.pop(name)
        saved["enable_deletion"] = False
        restored = train.normalized_algorithm_group("topology", saved)
        self.assertFalse(restored["use_rgb_deletion"])
        self.assertFalse(restored["use_low_opacity_deletion"])
        self.assertFalse(restored["enable_deletion"])
        self.assertEqual(restored["deletion_freeze_last_epochs"], 0)

    def test_invalid_deletion_config_rejected_before_runtime(self):
        invalid = {
            "deletion_warmup_epochs": (-1, True, .5),
            "deletion_protection_epochs": (-1, "20"),
            "deletion_freeze_last_epochs": (-1, True),
            "deletion_interval": (0, .5),
            "deletion_rgb_min_support_views": (0, True),
            "deletion_rgb_max_conflict_views": (0, True),
            "deletion_rgb_confirmation_windows": (0, True),
            "deletion_rgb_gain_epsilon": (-1, float("nan")),
            "deletion_rgb_min_mean_gain": (-1, float("inf")),
            "deletion_opacity_threshold": (0., 1., -1., True, float("nan"), float("inf")),
            "deletion_opacity_confirmation_windows": (0, -1, .5, True),
        }
        for name, values in invalid.items():
            for value in values:
                with self.subTest(name=name, value=value):
                    config = copy.deepcopy(train.CONFIG)
                    setattr(config.topology, name, value)
                    with self.assertRaisesRegex(ValueError, name):
                        train.validate(config)

    def test_nccl_sums_raw_deletion_evidence_before_view_vote(self):
        collective = _fixture()
        source = _accumulator(3)
        source.update(deletion_gain=np.array([4., -3., 0.], np.float32),
                      deletion_observed=np.array([2, 5, 0], np.uint32),
                      deletion_unknown=np.array([0, 1, 8], np.uint32))
        # Include growth simultaneously; its candidate count need not equal
        # the number of active voxels used by deletion.
        source.update(growth_gain=np.array([2., -1.], np.float32),
                      growth_observed=np.array([1, 2], np.uint32),
                      growth_unknown=np.array([3, 0], np.uint32))
        reduced, *_ = collective.reduce_view(source, 7., True, None, None)
        for key in ("deletion_gain", "deletion_observed", "deletion_unknown",
                    "growth_gain", "growth_observed", "growth_unknown"):
            np.testing.assert_array_equal(reduced[key], 2 * source[key] + 1)
            self.assertIsNot(reduced[key], source[key])
        self.assertEqual(collective.dr.syncs, 1)

    def test_incomplete_or_wrong_length_evidence_rejected(self):
        for missing in (True, False):
            with self.subTest(missing=missing):
                collective = _fixture()
                source = _accumulator(3)
                source["deletion_gain"] = np.zeros(2, np.float32)
                source["deletion_observed"] = np.zeros(2, np.uint32)
                if not missing:
                    source["deletion_unknown"] = np.zeros(2, np.uint32)
                with self.assertRaisesRegex(ValueError, "Deletion"):
                    collective.reduce_view(source, 0., True, None, None)


if __name__ == "__main__":
    unittest.main(verbosity=2)
