"""CPU checks for complete train/test evaluation and explicit legacy limits."""

from copy import deepcopy
from dataclasses import asdict
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import evaluate
import train


class TrainingEvaluationLimitsTests(unittest.TestCase):
    def configure(self, arguments=(), values=None, config=None):
        config = deepcopy(train.CONFIG) if config is None else config
        with patch.object(sys, "argv", ["train.py", *arguments]):
            args = train.parse_arguments()
        stderr = io.StringIO()
        with patch.object(sys, "stderr", stderr):
            train.apply_arguments(config, args, values or {})
        return config, stderr.getvalue()

    def evaluation(self, config):
        result = train.final_evaluation_config(config, Path("experiment/model.npz"))
        evaluate.validate_config(result)
        return result

    def test_defaults_keep_complete_training_and_test_splits(self):
        config, warning = self.configure()
        evaluation = self.evaluation(config)
        self.assertEqual(evaluate.evaluation_view_limits(evaluation), {"train": None, "test": None})
        self.assertEqual(evaluation.spp, 64)
        self.assertEqual(config.render.spp, 1)
        self.assertEqual(evaluation.dataset, config.data.dataset)
        self.assertEqual(warning, "")

    def test_cli_train_and_test_limits_are_independent(self):
        config, warning = self.configure(("--eval-train-max-views", "4", "--eval-test-max-views", "23"))
        self.assertEqual(evaluate.evaluation_view_limits(self.evaluation(config)), {"train": 4, "test": 23})
        self.assertEqual(warning, "")

    def test_old_explicit_split_limits_survive_resume_and_can_be_overridden(self):
        resumed = train.TrainConfig()
        train.update(resumed, {"evaluation": {"train_max_views": 10, "test_max_views": None}})
        resumed, _ = self.configure(("--resume", "checkpoint.npz"), config=resumed)
        self.assertEqual(evaluate.evaluation_view_limits(self.evaluation(resumed)), {"train": 10, "test": None})
        resumed, _ = self.configure(("--resume", "checkpoint.npz", "--eval-train-max-views", "0",
                                     "--eval-test-max-views", "10"), config=resumed)
        self.assertEqual(evaluate.evaluation_view_limits(self.evaluation(resumed)), {"train": None, "test": 10})

    def test_zero_cli_limit_selects_all_without_affecting_other_split(self):
        for arguments, expected in (
                (("--eval-train-max-views", "0"), {"train": None, "test": 7}),
                (("--eval-test-max-views", "0"), {"train": 2, "test": None})):
            with self.subTest(arguments=arguments):
                config, _ = self.configure(arguments, {
                    "evaluation": {"train_max_views": 2, "test_max_views": 7}})
                self.assertEqual(evaluate.evaluation_view_limits(self.evaluation(config)), expected)

    def test_json_limits_accept_null_and_cli_overrides_one_split(self):
        config, _ = self.configure(("--eval-test-max-views", "0"), {
            "evaluation": {"train_max_views": None, "test_max_views": 5}})
        self.assertEqual(evaluate.evaluation_view_limits(self.evaluation(config)), {"train": None, "test": None})
        config, _ = self.configure(values={"evaluation": {"train_max_views": 2, "test_max_views": 7}})
        self.assertEqual(evaluate.evaluation_view_limits(self.evaluation(config)), {"train": 2, "test": 7})

    def test_legacy_cli_limit_explicitly_overrides_both_splits_and_warns(self):
        config, warning = self.configure(("--eval-train-max-views", "4", "--eval-test-max-views", "0",
                                          "--eval-max-views", "6"))
        self.assertEqual(evaluate.evaluation_view_limits(self.evaluation(config)), {"train": 6, "test": 6})
        self.assertIn("Deprecated", warning)
        self.assertIn("BOTH", warning)
        self.assertIn("train_max_views=null and test_max_views=null", warning)
        self.assertEqual(config.evaluation.max_views, 6)

    def test_legacy_zero_explicitly_selects_all_splits(self):
        config, _ = self.configure(("--eval-max-views", "0", "--eval-test-max-views", "2"))
        self.assertEqual(evaluate.evaluation_view_limits(self.evaluation(config)), {"train": None, "test": None})

    def test_legacy_json_and_resumed_checkpoint_keep_explicit_limits(self):
        config, warning = self.configure(values={"evaluation": {"max_views": 3}})
        self.assertEqual(evaluate.evaluation_view_limits(self.evaluation(config)), {"train": 3, "test": 3})
        self.assertIn("Deprecated", warning)
        resumed = train.TrainConfig()
        train.update(resumed, asdict(config))
        resumed, warning = self.configure(("--resume", "checkpoint.npz"), config=resumed)
        self.assertEqual(evaluate.evaluation_view_limits(self.evaluation(resumed)), {"train": 3, "test": 3})
        self.assertIn("Deprecated", warning)

    def test_null_json_override_can_clear_a_saved_legacy_limit(self):
        config = deepcopy(train.CONFIG)
        config.evaluation.max_views = 10
        config, warning = self.configure(values={"evaluation": {"max_views": None}}, config=config)
        self.assertEqual(evaluate.evaluation_view_limits(self.evaluation(config)), {"train": None, "test": None})
        self.assertEqual(warning, "")

    def test_disabled_evaluation_does_not_emit_legacy_warning(self):
        config, warning = self.configure(("--skip-evaluation", "--eval-max-views", "10"))
        self.assertFalse(config.evaluation.enabled)
        self.assertEqual(warning, "")

    def test_invalid_limits_are_rejected_before_training(self):
        for name in ("train_max_views", "test_max_views", "max_views"):
            for value in (-1, True, 1.5):
                with self.subTest(name=name, value=value):
                    config, _ = self.configure(values={"evaluation": {name: value}})
                    with self.assertRaises(ValueError):
                        self.evaluation(config)

    def test_limits_do_not_modify_training_or_dataset_configuration(self):
        config = deepcopy(train.CONFIG)
        before = asdict(config)
        changed, _ = self.configure(("--eval-train-max-views", "1", "--eval-test-max-views", "0"),
                                    config=config)
        after = asdict(changed)
        for name in before:
            if name not in ("evaluation", "schedule"):
                self.assertEqual(before[name], after[name], name)

    def test_standalone_evaluation_defaults_match_training_and_cli_overrides(self):
        for arguments, expected in (
                ((), {"train": None, "test": None}),
                (("--train-max-views", "4"), {"train": 4, "test": None}),
                (("--test-max-views", "0"), {"train": None, "test": None}),
                (("--train-max-views", "4", "--test-max-views", "2"), {"train": 4, "test": 2}),
                (("--train-max-views", "4", "--max-views", "3"), {"train": 3, "test": 3}),
                (("--test-max-views", "2", "--max-views", "0"), {"train": None, "test": None})):
            with self.subTest(arguments=arguments), patch.object(sys, "argv", ["evaluate.py", *arguments]):
                config = evaluate.parse_config(Path("."))
                evaluate.validate_config(config)
                self.assertEqual(evaluate.evaluation_view_limits(config), expected)

    def test_standalone_legacy_warning_describes_current_default_selection(self):
        stderr = io.StringIO()
        with patch.object(sys, "stderr", stderr):
            evaluate.warn_legacy_view_limit(evaluate.EvaluationConfig(max_views=3))
        self.assertIn("--train-max-views 0 --test-max-views 0", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
