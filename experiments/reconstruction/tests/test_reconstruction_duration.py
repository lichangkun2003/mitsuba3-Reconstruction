"""CPU-only completion/relocation checks; rendering and training are mocked.

Run with Python -B -I. All experiment files live in temporary directories.
"""

from __future__ import annotations

import builtins
from copy import deepcopy
from dataclasses import asdict
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
_original_import = builtins.__import__


def _stdlib_only(name, *args, **kwargs):
    if name.partition(".")[0] in {"mitsuba", "drjit", "numpy"}:
        raise AssertionError(f"Duration tests must not initialize {name}")
    return _original_import(name, *args, **kwargs)


with mock.patch.object(builtins, "__import__", side_effect=_stdlib_only):
    import evaluate
    import reconstruction_experiment as experiment
    from reconstruction_performance import PerformanceRecorder
    import train


class ResumeTimingTests(unittest.TestCase):
    def test_new_and_legacy_checkpoint_time_completeness(self):
        self.assertEqual(train.resumed_reconstruction_time(None), (0., True))
        self.assertEqual(train.resumed_reconstruction_time(
            {"training_state": {"completed_epochs": 0}}), (0., True))
        self.assertEqual(train.resumed_reconstruction_time(
            {"training_state": {"completed_epochs": 200}}), (0., False))

    def test_saved_active_time_and_partial_legacy_provenance_survive_resume(self):
        for complete in (True, False):
            with self.subTest(complete=complete):
                checkpoint = {"training_state": {
                    "completed_epochs": 200,
                    "reconstruction_seconds": 142.25,
                    "reconstruction_time_complete": complete,
                }}
                self.assertEqual(train.resumed_reconstruction_time(checkpoint),
                                 (142.25, complete))
        self.assertEqual(train.resumed_reconstruction_time(
            {"training_state": {"reconstruction_seconds": 60}}), (60., True))

    def test_invalid_saved_timers_are_rejected(self):
        for seconds in (-1, True, "60", None, float("inf"), float("nan")):
            with self.subTest(seconds=seconds):
                with self.assertRaises(ValueError):
                    train.resumed_reconstruction_time(
                        {"training_state": {"reconstruction_seconds": seconds}})
        for complete in (0, 1, "false", None):
            with self.subTest(complete=complete):
                with self.assertRaises(ValueError):
                    train.resumed_reconstruction_time({"training_state": {
                        "reconstruction_seconds": 60.,
                        "reconstruction_time_complete": complete}})


class DurationFinalizationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="mitsuba-duration-test-")
        self.addCleanup(temporary.cleanup)
        self.workspace = Path(temporary.name).resolve()
        self.repository = self.workspace / "mitsuba"
        self.repository.mkdir()
        self.patch_workspace = mock.patch.object(train, "WORKSPACE", self.workspace)
        self.patch_repository = mock.patch.object(train, "REPOSITORY", self.repository)
        self.patch_workspace.start()
        self.patch_repository.start()
        self.addCleanup(self.patch_workspace.stop)
        self.addCleanup(self.patch_repository.stop)
        self.output = self.workspace / "outputs/lego-10.20-E3/training"
        self.output.mkdir(parents=True)
        self.model_path = self.output.parent / "model.npz"
        self.model_bytes = b"already saved reconstruction; do not rewrite checkpoint"
        self.model_path.write_bytes(self.model_bytes)
        self.epoch_path = self.output / "epoch-00002.npz"
        self.epoch_path.write_bytes(b"existing checkpoint with original output metadata")
        self.config = deepcopy(train.CONFIG)
        self.config.training.output = self.output.relative_to(self.workspace).as_posix()
        self.config.evaluation.enabled = True
        self.saved_config = asdict(self.config)
        self.image_path = self.output.parent / "train/0000-initial.png"
        self.image_path.parent.mkdir()
        self.image_path.write_bytes(b"existing initialization image")
        self.report = {
            "status": "running",
            "training_status": "passed",
            "trained_epochs_this_run": 2,
            "reconstruction_seconds": 1200.0,
            "reconstruction_time_complete": True,
            "completed_epochs": 1000,
            "checkpoint": str(self.model_path),
            "reconstructed_model": {
                "path": str(self.model_path), "sha256": "keep-existing-sha256",
            },
            "initial_render": {"status": "passed", "image": str(self.image_path)},
            "evaluation": {"status": "pending"},
            "performance_file": str(self.output / "performance-summary.json"),
            "reference_cache": {"hits": 300},
        }
        self.performance = PerformanceRecorder(
            self.output / "performance-summary.json",
            {
                "history_file": str(self.output / "training.jsonl"),
                "resumed_from": str(self.epoch_path),
                "source_file": str(self.repository / "src/render/reconstruction.cpp"),
            })
        self.performance.record_checkpoint(self.model_path, 0.1)
        (self.output / "config.json").write_text(
            json.dumps(self.saved_config), encoding="utf-8")
        (self.output / "training-report.json").write_text(
            json.dumps(self.report), encoding="utf-8")
        self.performance.write("running", training_status="passed")
        self.expected_output = self.workspace / "outputs/lego-10.20-E3-20min/training"
        self.expected_model = self.expected_output.parent / "model.npz"

    def finalize(self, distributed=None):
        with mock.patch("sys.stdout", new_callable=io.StringIO):
            return train.finalize_experiment_output(
                self.config, self.model_path, self.output, self.report,
                self.performance, distributed=distributed)

    def read_json(self, path):
        return json.loads(path.read_text(encoding="utf-8"))

    def test_completed_experiment_moves_intact_and_updates_live_metadata(self):
        model, output = self.finalize()
        self.assertEqual((model, output), (self.expected_model, self.expected_output))
        self.assertFalse(self.output.parent.exists())
        self.assertEqual(model.read_bytes(), self.model_bytes)
        self.assertEqual((output / self.epoch_path.name).read_bytes(),
                         b"existing checkpoint with original output metadata")
        self.assertEqual((model.parent / "train/0000-initial.png").read_bytes(),
                         b"existing initialization image")
        self.assertEqual(self.config.training.output,
                         output.relative_to(self.workspace).as_posix())
        saved = self.read_json(output / "config.json")
        self.assertEqual(saved["training"]["output"], self.config.training.output)
        self.assertEqual(saved["topology"], self.saved_config["topology"])
        self.assertEqual(saved["gradient"], self.saved_config["gradient"])
        report = self.read_json(output / "training-report.json")
        self.assertEqual(report["checkpoint"], str(model))
        self.assertEqual(report["reconstructed_model"]["path"], str(model))
        self.assertEqual(report["reconstructed_model"]["sha256"], "keep-existing-sha256")
        self.assertEqual(report["initial_render"]["image"],
                         str(model.parent / "train/0000-initial.png"))
        self.assertEqual(report["experiment_directory"]["status"], "renamed")
        self.assertEqual(report["experiment_directory"]["minutes"], 20)
        self.assertEqual(report["experiment_directory"]["seconds"], 1200.0)
        self.assertEqual(self.performance.path, output / "performance-summary.json")
        performance = self.read_json(self.performance.path)
        self.assertEqual(performance["metadata"]["history_file"], str(output / "training.jsonl"))
        self.assertEqual(performance["metadata"]["resumed_from"], str(output / self.epoch_path.name))
        self.assertEqual(performance["metadata"]["source_file"],
                         str(self.repository / "src/render/reconstruction.cpp"))
        self.assertEqual(performance["checkpoints"][0]["path"], str(model))
        self.assertEqual(performance["details"]["training_report"],
                         str(output / "training-report.json"))
        self.assertEqual(performance["details"]["reference_cache"], {"hits": 300})

    def test_checkpoint_resume_uses_current_location_despite_saved_old_path(self):
        model, output = self.finalize()
        self.assertEqual(experiment.resumed_training_output(model, self.saved_config), output)
        self.assertEqual(experiment.resumed_training_output(
            output / self.epoch_path.name, self.saved_config), output)
        config = deepcopy(train.CONFIG)
        config.training.output = None
        self.assertEqual(train.prepare_training_output(
            config, model, self.saved_config), output)
        self.assertEqual(config.training.output, output.relative_to(self.workspace).as_posix())
        self.assertEqual(model.read_bytes(), self.model_bytes)

    def test_incomplete_and_already_finished_resume_do_not_rename(self):
        for status, trained in (("stopped", 2), ("failed", 2),
                                ("interrupted", 2), ("passed", 0)):
            with self.subTest(status=status, trained=trained):
                self.report.update(training_status=status, trained_epochs_this_run=trained)
                original = deepcopy(self.report)
                with mock.patch.object(train, "finalize_training_output") as rename:
                    self.assertEqual(self.finalize(), (self.model_path, self.output))
                rename.assert_not_called()
                self.assertEqual(self.report, original)
                self.assertTrue(self.output.is_dir())
                self.assertEqual(self.model_path.read_bytes(), self.model_bytes)
                self.assertFalse(self.expected_output.exists())

    def test_rename_failure_preserves_valid_model_and_original_output(self):
        with mock.patch.object(train, "finalize_training_output",
                               side_effect=PermissionError("test: directory is busy")), \
                mock.patch("sys.stderr", new_callable=io.StringIO):
            model, output = self.finalize()
        self.assertEqual((model, output), (self.model_path, self.output))
        self.assertEqual(model.read_bytes(), self.model_bytes)
        self.assertFalse(self.expected_output.exists())
        self.assertEqual(self.report["training_status"], "passed")
        naming = self.report["experiment_directory"]
        self.assertEqual(naming["status"], "failed")
        self.assertIn("directory is busy", naming["error"])
        self.assertEqual(self.config.training.output,
                         self.output.relative_to(self.workspace).as_posix())
        self.assertEqual(self.performance.path, self.output / "performance-summary.json")

    def main_with_mocked_renderer(self, *, enabled=True, evaluation_result=0):
        self.config.evaluation.enabled = enabled
        if not enabled:
            self.report.update(status="passed", evaluation={"status": "skipped"})
        fake_drjit = types.ModuleType("drjit")
        fake_drjit.sync_thread = mock.Mock()
        fake_drjit.flush_malloc_cache = mock.Mock()
        run_result = (self.config, self.model_path, self.output, self.report, self.performance)

        def evaluate_renamed(config, **kwargs):
            self.assertEqual(Path(config.checkpoint), self.expected_model)
            self.assertEqual(Path(config.output_directory), self.expected_model.parent)
            self.assertEqual(Path(config.checkpoint).read_bytes(), self.model_bytes)
            self.assertFalse(self.output.parent.exists())
            # Evaluation can take arbitrarily long without renaming the training
            # directory again or changing its accumulated reconstruction duration.
            self.performance.set_phase_seconds("evaluation", 3600.)
            return evaluation_result

        with mock.patch.object(train, "run_training", return_value=run_result), \
                mock.patch.object(evaluate, "run_evaluation", side_effect=evaluate_renamed) as run_eval, \
                mock.patch.dict(sys.modules, {"drjit": fake_drjit}), \
                mock.patch("sys.stdout", new_callable=io.StringIO), \
                mock.patch("sys.stderr", new_callable=io.StringIO):
            result = train.main()
        return result, run_eval, fake_drjit

    def test_main_evaluates_from_renamed_checkpoint_and_publishes_new_paths(self):
        result, run_eval, drjit = self.main_with_mocked_renderer()
        self.assertEqual(result, 0)
        run_eval.assert_called_once()
        drjit.sync_thread.assert_called_once()
        report = self.read_json(self.expected_output / "training-report.json")
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["evaluation"]["status"], "passed")
        self.assertEqual(report["evaluation"]["output"], str(self.expected_model.parent))
        self.assertEqual(report["evaluation"]["report"],
                         str(self.expected_model.parent / "evaluation-report.json"))
        self.assertEqual(report["reconstruction_seconds"], 1200.)
        performance = self.read_json(self.expected_output / "performance-summary.json")
        self.assertEqual(performance["details"]["evaluation_report"],
                         str(self.expected_model.parent / "evaluation-report.json"))
        self.assertFalse(self.output.parent.exists())

    def test_skip_evaluation_still_finalizes_directory(self):
        result, run_eval, drjit = self.main_with_mocked_renderer(enabled=False)
        self.assertEqual(result, 0)
        run_eval.assert_not_called()
        drjit.sync_thread.assert_not_called()
        self.assertEqual(self.expected_model.read_bytes(), self.model_bytes)
        self.assertFalse(self.output.parent.exists())
        report = self.read_json(self.expected_output / "training-report.json")
        self.assertEqual(report["training_status"], "passed")
        self.assertEqual(report["evaluation"]["status"], "skipped")

    def test_evaluation_failure_keeps_completed_reconstruction_name(self):
        result, run_eval, _ = self.main_with_mocked_renderer(evaluation_result=1)
        self.assertEqual(result, 1)
        run_eval.assert_called_once()
        self.assertEqual(self.expected_model.read_bytes(), self.model_bytes)
        report = self.read_json(self.expected_output / "training-report.json")
        self.assertEqual(report["training_status"], "passed")
        self.assertEqual(report["evaluation"]["status"], "failed")
        self.assertEqual(report["reconstruction_seconds"], 1200.)

    def test_primary_broadcasts_new_paths_and_secondary_does_not_rename_or_write(self):
        secondary_config = deepcopy(self.config)
        secondary_report = deepcopy(self.report)
        secondary_performance = PerformanceRecorder(
            self.performance.path, deepcopy(self.performance.metadata))
        calls = []
        payload = []

        def primary_broadcast(value):
            calls.append("broadcast")
            self.assertTrue(self.expected_model.is_file())
            payload.append(value)
            return value

        primary = types.SimpleNamespace(
            rank=0, world_size=2,
            barrier=lambda: calls.append("barrier"),
            broadcast=primary_broadcast)
        self.assertEqual(self.finalize(primary), (self.expected_model, self.expected_output))
        self.assertIn("barrier", calls)
        self.assertLess(calls.index("barrier"), calls.index("broadcast"))
        self.assertEqual(len(payload), 1)
        report_before_secondary = (self.expected_output / "training-report.json").read_bytes()

        secondary = types.SimpleNamespace(
            rank=1, world_size=2, barrier=mock.Mock(),
            broadcast=mock.Mock(return_value=payload[0]))
        with mock.patch.object(train, "finalize_training_output") as rename, \
                mock.patch.object(secondary_performance, "write") as write, \
                mock.patch("sys.stdout", new_callable=io.StringIO):
            result = train.finalize_experiment_output(
                secondary_config, self.model_path, self.output,
                secondary_report, secondary_performance, distributed=secondary)
        rename.assert_not_called()
        write.assert_not_called()
        secondary.broadcast.assert_called_once_with(None)
        self.assertEqual(result, (self.expected_model, self.expected_output))
        self.assertEqual(secondary_config.training.output, self.config.training.output)
        self.assertEqual(secondary_report["experiment_directory"],
                         self.report["experiment_directory"])
        self.assertEqual(secondary_performance.path, self.performance.path)
        self.assertEqual((self.expected_output / "training-report.json").read_bytes(),
                         report_before_secondary)


if __name__ == "__main__":
    unittest.main(verbosity=2)
