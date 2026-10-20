"""CPU regression checks for saved reference compositing and linear training."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import evaluate
import preview
import reconstruction_checkpoint as checkpoint
import train


class ReferenceProtocolTests(unittest.TestCase):
    def test_new_experiments_use_srgb_compositing_but_legacy_dataclasses_do_not(self):
        self.assertEqual(preview.DataConfig().reference_compositing, "linear")
        self.assertEqual(train.TrainConfig().data.reference_compositing, "linear")
        self.assertEqual(train.CONFIG.data.reference_compositing, "srgb")
        self.assertEqual(preview.CONFIG.data.reference_compositing, "srgb")
        self.assertEqual(evaluate.CONFIG.metric_color_space, "srgb")
        self.assertIsNone(evaluate.CONFIG.reference_compositing)

    def test_target_protocol_does_not_change_native_gradient_settings(self):
        config = deepcopy(train.CONFIG)
        before = train.settings(config)
        config.data.reference_compositing = "linear"
        self.assertEqual(train.settings(config), before)
        self.assertNotIn("reference_compositing", before)
        self.assertTrue(before["rgb_only"])
        self.assertEqual(before["alpha_loss_weight"], 0)

    def test_old_rgb_checkpoint_implicitly_resumes_linear_targets(self):
        config = deepcopy(train.CONFIG)
        old = {"config": {"training": {"rgb_only": True}, "data": {"dataset": "datasets/lego"}}}
        train.resolve_resume_reference_compositing(config, old, {})
        self.assertEqual(config.data.reference_compositing, "linear")
        train.validate_resume_supervision(config, old)

    def test_new_checkpoint_resumes_saved_protocol_not_current_config(self):
        config = deepcopy(train.CONFIG)
        config.data.reference_compositing = "linear"
        new = {"config": {"data": {"reference_compositing": "srgb"}}}
        train.resolve_resume_reference_compositing(config, new, {"training": {"num_epochs": 1000}})
        self.assertEqual(config.data.reference_compositing, "srgb")

    def test_resume_accepts_explicit_match_and_rejects_target_change(self):
        for saved in ("linear", "srgb"):
            old = {"config": {"data": {"reference_compositing": saved}}}
            config = deepcopy(train.CONFIG)
            train.resolve_resume_reference_compositing(
                config, old, {"data": {"reference_compositing": saved}})
            self.assertEqual(config.data.reference_compositing, saved)
            changed = "linear" if saved == "srgb" else "srgb"
            with self.assertRaisesRegex(ValueError, "Cannot change data.reference_compositing"):
                train.resolve_resume_reference_compositing(
                    config, old, {"data": {"reference_compositing": changed}})
        with self.assertRaisesRegex(ValueError, "Cannot change data.reference_compositing"):
            train.resolve_resume_reference_compositing(
                deepcopy(train.CONFIG), {"config": {}}, {"data": {"reference_compositing": "srgb"}})

    def test_legacy_comparison_only_fills_missing_protocol(self):
        old_data = {"dataset": "datasets/lego"}
        explicit_linear = {**old_data, "reference_compositing": "linear"}
        explicit_srgb = {**old_data, "reference_compositing": "srgb"}
        self.assertEqual(train.normalized_algorithm_group("data", old_data), explicit_linear)
        self.assertNotEqual(train.normalized_algorithm_group("data", old_data), explicit_srgb)
        self.assertEqual(old_data, {"dataset": "datasets/lego"})

    def test_invalid_protocols_are_rejected_before_loading_images(self):
        for invalid in (None, "gamma", "", 1, []):
            with self.subTest(value=invalid):
                config = deepcopy(train.CONFIG)
                config.data.reference_compositing = invalid
                with self.assertRaises(ValueError):
                    train.validate(config)
                with self.assertRaises(ValueError):
                    train.resolve_resume_reference_compositing(
                        deepcopy(train.CONFIG), {"config": {"data": {"reference_compositing": invalid}}}, {})
        evaluation = evaluate.EvaluationConfig(reference_compositing="gamma")
        with self.assertRaises(ValueError):
            evaluate.validate_config(evaluation)

    def test_legacy_supervision_never_silently_uses_new_target_semantics(self):
        with patch.object(sys, "argv", ["train.py", "--no-rgb-only"]):
            arguments = train.parse_arguments()
        config = deepcopy(train.CONFIG)
        train.apply_arguments(config, arguments, {})
        self.assertEqual(config.data.reference_compositing, "linear")
        train.validate(config)
        config = deepcopy(train.CONFIG)
        train.apply_arguments(config, arguments, {"data": {"reference_compositing": "srgb"}})
        with self.assertRaisesRegex(ValueError, "RGB-only"):
            train.validate(config)

    def test_evaluation_inherits_old_and_new_checkpoints_with_explicit_override(self):
        default = evaluate.EvaluationConfig()
        self.assertEqual(evaluate.resolve_reference_compositing(default, {}),
                         ("linear", "legacy_checkpoint_default"))
        for saved in ("linear", "srgb"):
            document = {"data": {"reference_compositing": saved}}
            self.assertEqual(evaluate.resolve_reference_compositing(default, document),
                             (saved, "checkpoint"))
            override = "linear" if saved == "srgb" else "srgb"
            explicit = evaluate.EvaluationConfig(reference_compositing=override)
            self.assertEqual(evaluate.resolve_reference_compositing(explicit, document),
                             (override, "evaluation_override"))
        with self.assertRaises(ValueError):
            evaluate.resolve_reference_compositing(default, {"data": {"reference_compositing": "bad"}})

    def test_preview_and_evaluation_cli_can_explicitly_reproduce_old_targets(self):
        with patch.object(sys, "argv", ["preview.py", "--reference-compositing", "linear"]):
            result = preview.parse_config(Path.cwd())
        self.assertEqual(result.data.reference_compositing, "linear")
        with patch.object(sys, "argv", [
                "evaluate.py", "--reference-compositing", "linear", "--metric-color-space", "srgb"]):
            result = evaluate.parse_config(Path.cwd())
        self.assertEqual(result.reference_compositing, "linear")
        self.assertEqual(result.metric_color_space, "srgb")

    def test_new_checkpoint_round_trip_preserves_targets_for_training_and_evaluation(self):
        state = {
            "cell_indices": np.array([0], np.uint32),
            "centers": np.zeros(3, np.float32),
            "log_scales": np.zeros(3, np.float32),
            "rotations": np.array([1, 0, 0, 0], np.float32),
            "radiance_sh": np.zeros(3, np.float32),
            "opacity_sh": np.zeros(1, np.float32),
        }
        metadata = dict(resolution=1, grid_min=[-1., -1., -1.], grid_max=[1., 1., 1.],
                        sh_count=1, opacity_sh_count=1)
        with tempfile.TemporaryDirectory(prefix="reference-protocol-") as temporary:
            path = Path(temporary) / "model.npz"
            checkpoint.save_checkpoint(path, state, metadata, config=asdict(train.CONFIG))
            restored = checkpoint.load_checkpoint(path)
        resumed = deepcopy(train.CONFIG)
        resumed.data.reference_compositing = "linear"
        train.resolve_resume_reference_compositing(resumed, restored, {})
        self.assertEqual(resumed.data.reference_compositing, "srgb")
        self.assertEqual(evaluate.resolve_reference_compositing(
            evaluate.EvaluationConfig(), restored["config"]), ("srgb", "checkpoint"))
        self.assertEqual(restored["config"]["render"]["spp"], 1)
        self.assertTrue(restored["config"]["training"]["rgb_only"])

    def test_reporting_distinguishes_reference_compositing_from_linear_loss(self):
        for mode in ("linear", "srgb"):
            protocol = evaluate.reference_metric_protocol(mode, "checkpoint")
            self.assertEqual(protocol["reference_compositing"], mode)
            self.assertEqual(protocol["training_loss_color_space"], "linear")


if __name__ == "__main__":
    unittest.main()
