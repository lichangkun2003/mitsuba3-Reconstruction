"""CPU-only checks for experiment naming, configuration and resume locations.

Run with Python -I -B. All outputs and lock files are confined to temporary
directories; no renderer, training dataset or installed dependencies are needed.
"""

from __future__ import annotations

import builtins
from copy import deepcopy
from dataclasses import asdict
from datetime import date
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


RESOURCES = Path(__file__).resolve().parent
sys.path.insert(0, str(RESOURCES))

_original_import = builtins.__import__


def _stdlib_only(name, *args, **kwargs):
    if name.partition(".")[0] in {"mitsuba", "drjit", "numpy"}:
        raise AssertionError(f"Experiment tests must not initialize {name}")
    return _original_import(name, *args, **kwargs)


with mock.patch.object(builtins, "__import__", side_effect=_stdlib_only):
    import reconstruction_experiment as experiment
    import reconstruction_train as train


DAY = date(2026, 10, 15)


class ExperimentOutputTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="mitsuba-experiment-test-")
        self.addCleanup(temporary.cleanup)
        self.workspace = Path(temporary.name).resolve()
        self.repository = self.workspace / "mitsuba"
        self.repository.mkdir()
        self.addCleanup(mock.patch.stopall)
        mock.patch.object(train, "WORKSPACE", self.workspace).start()
        mock.patch.object(train, "REPOSITORY", self.repository).start()

    def allocate(self, dataset="datasets/lego"):
        return experiment.allocate_training_output(dataset, self.workspace, day=DAY)

    def test_first_experiment_is_e0_with_training_subdirectory(self):
        output = self.allocate()
        self.assertEqual(output, self.workspace / "outputs/lego-10.15-E0/training")
        self.assertTrue(output.is_dir())

    def test_daily_index_is_global_and_numeric_across_scenes(self):
        root = self.workspace / "outputs"
        for name in ("lego-10.15-E2", "ficus-10.15-E9", "hotdog-10.15-E10"):
            (root / name).mkdir(parents=True)
        sentinel = root / "ficus-10.15-E9/model.npz"
        sentinel.write_bytes(b"existing reconstruction")
        self.assertEqual(self.allocate("datasets/materials"), root / "materials-10.15-E11/training")
        self.assertEqual(sentinel.read_bytes(), b"existing reconstruction")
        self.assertEqual(self.allocate(), root / "lego-10.15-E12/training")

    def test_other_days_invalid_names_and_files_do_not_set_daily_counter(self):
        root = self.workspace / "outputs"
        for name in ("lego-10.14-E100", "lego-10.16-E100", "cache",
                     "lego-10.15-Ebad", "lego-10.15-E5-backup", "-10.15-E20"):
            (root / name).mkdir(parents=True)
        (root / "other-10.15-E999").write_text("ordinary file", encoding="utf-8")
        self.assertEqual(self.allocate(), root / "lego-10.15-E0/training")

    def test_same_name_file_is_preserved_and_skipped(self):
        root = self.workspace / "outputs"
        root.mkdir()
        collision = root / "lego-10.15-E0"
        collision.write_bytes(b"do not overwrite")
        self.assertEqual(self.allocate(), root / "lego-10.15-E1/training")
        self.assertEqual(collision.read_bytes(), b"do not overwrite")

    def test_scene_name_comes_from_configured_dataset_directory(self):
        for dataset, scene in (("datasets/lego/", "lego"),
                               (r"datasets\ficus", "ficus"),
                               ("datasets/materials-detailed", "materials-detailed")):
            with self.subTest(dataset=dataset):
                output = self.allocate(dataset)
                self.assertTrue(output.parent.name.startswith(f"{scene}-10.15-E"))

    def test_dataset_symlink_keeps_its_visible_scene_name(self):
        target = self.workspace / "dataset-storage-with-different-name"
        target.mkdir()
        link = self.workspace / "datasets/lego"
        link.parent.mkdir()
        try:
            link.symlink_to(target, target_is_directory=True)
        except (OSError, NotImplementedError) as error:
            self.skipTest(f"Creating directory symlinks is unavailable: {error}")
        self.assertEqual(self.allocate(str(link)).parent.name, "lego-10.15-E0")

    def test_automatic_path_is_written_back_as_portable_configuration(self):
        config = train.TrainConfig()
        output = train.prepare_training_output(config)
        self.assertTrue(output.is_dir())
        self.assertEqual(output, self.workspace / config.training.output)
        self.assertFalse(Path(config.training.output).is_absolute())
        self.assertNotIn("\\", config.training.output)
        self.assertEqual(train.final_model_path(output), output.parent / "model.npz")

    def test_explicit_output_bypasses_automatic_number_allocation(self):
        config = train.TrainConfig()
        config.training.output = "outputs/custom-experiment/training"
        with mock.patch.object(train, "allocate_training_output", side_effect=AssertionError("unexpected allocation")):
            output = train.prepare_training_output(config)
        self.assertEqual(output, self.workspace / "outputs/custom-experiment/training")
        self.assertFalse((self.workspace / ".cache").exists())

    def test_explicit_existing_results_are_never_overwritten(self):
        for artifact in ("model.npz", "training/training.jsonl", "training/latest.npz", "training/epoch-00001.npz"):
            with self.subTest(artifact=artifact):
                config = train.TrainConfig()
                config.training.output = "outputs/custom-experiment/training"
                protected = self.workspace / "outputs/custom-experiment" / artifact
                protected.parent.mkdir(parents=True, exist_ok=True)
                protected.write_bytes(b"existing result")
                try:
                    with self.assertRaisesRegex(ValueError, "already contains"):
                        train.prepare_training_output(config)
                    self.assertEqual(protected.read_bytes(), b"existing result")
                finally:
                    protected.unlink()

    def test_training_outputs_cannot_be_created_inside_repository(self):
        config = train.TrainConfig()
        config.training.output = "mitsuba/outputs/experiment/training"
        with self.assertRaisesRegex(ValueError, "outside the Git checkout"):
            train.prepare_training_output(config)
        self.assertFalse((self.repository / "outputs").exists())

    def test_resume_uses_checkpoint_location_and_allocates_no_new_experiment(self):
        cases = (
            ("nested/training/epoch-00001.npz", "outputs/nested/training", "nested/training"),
            ("nested/model.npz", "outputs/nested/training", "nested/training"),
            ("copied/model.npz", r"D:\old-machine\outputs\original\training", "copied/training"),
            ("legacy/model.npz", "outputs/legacy", "legacy"),
            ("legacy/latest.npz", "/old-machine/outputs/legacy", "legacy"),
        )
        for relative, previous, expected in cases:
            with self.subTest(checkpoint=relative, previous=previous):
                checkpoint = self.workspace / "outputs" / relative
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                checkpoint.write_bytes(b"checkpoint placeholder; not loaded")
                config = train.TrainConfig()
                with mock.patch.object(train, "allocate_training_output", side_effect=AssertionError("resume allocated a new experiment")):
                    output = train.prepare_training_output(config, checkpoint, {"training": {"output": previous}})
                self.assertEqual(output, self.workspace / "outputs" / expected)
                self.assertEqual(config.training.output, "outputs/" + expected)
                self.assertEqual(checkpoint.read_bytes(), b"checkpoint placeholder; not loaded")
        self.assertFalse((self.workspace / ".cache").exists())

    def test_final_model_with_training_subdirectory_can_resume_old_metadata(self):
        root = self.workspace / "outputs/copied-experiment"
        (root / "training").mkdir(parents=True)
        checkpoint = root / "model.npz"
        checkpoint.write_bytes(b"preserve")
        config = train.TrainConfig()
        output = train.prepare_training_output(config, checkpoint, {})
        self.assertEqual(output, root / "training")
        self.assertEqual(checkpoint.read_bytes(), b"preserve")

    def test_explicit_resume_output_is_honored(self):
        checkpoint = self.workspace / "outputs/old/model.npz"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"preserve")
        config = train.TrainConfig()
        config.training.output = "outputs/continued/training"
        with mock.patch.object(train, "allocate_training_output", side_effect=AssertionError("unexpected allocation")):
            output = train.prepare_training_output(config, checkpoint, {"training": {"output": "outputs/old/training"}})
        self.assertEqual(output, self.workspace / "outputs/continued/training")
        self.assertEqual(checkpoint.read_bytes(), b"preserve")

    def test_config_json_and_cli_override_priority(self):
        config = train.TrainConfig()
        train.update(config, asdict(deepcopy(train.CONFIG)))
        config.data.dataset = "datasets/config-scene"
        config.training.output = None
        config.training.num_epochs = 17
        values = json.loads('{"data": {"dataset": "datasets/json-scene"}, "training": {"num_epochs": 20, "output": "outputs/json-experiment/training"}}')
        with mock.patch.object(sys, "argv", ["reconstruction_train.py", "--dataset", "datasets/cli-scene", "--epochs", "30"]):
            args = train.parse_arguments()
        train.apply_arguments(config, args, values)
        self.assertEqual(config.data.dataset, "datasets/cli-scene")
        self.assertEqual(config.training.num_epochs, 30)
        self.assertEqual(config.schedule.total_epochs, 30)
        self.assertEqual(config.training.output, "outputs/json-experiment/training")
        with mock.patch.object(sys, "argv", ["reconstruction_train.py", "--output-dir", "outputs/cli-experiment/training"]):
            args = train.parse_arguments()
        train.apply_arguments(config, args, {"training": {"output": None}})
        self.assertEqual(config.training.output, "outputs/cli-experiment/training")

    def test_json_null_selects_automatic_naming(self):
        config = train.TrainConfig()
        config.training.output = "outputs/manually-configured/training"
        with mock.patch.object(sys, "argv", ["reconstruction_train.py"]):
            args = train.parse_arguments()
        train.apply_arguments(config, args, json.loads('{"training": {"output": null}}'))
        output = train.prepare_training_output(config)
        self.assertNotEqual(output.parent.name, "manually-configured")
        self.assertTrue(output.parent.name.endswith("-E0"))

    def test_concurrent_processes_reserve_distinct_global_indices(self):
        worker = """
from datetime import date
from pathlib import Path
import sys
sys.path.insert(0, sys.argv[1])
from reconstruction_experiment import allocate_training_output
sys.stdin.readline()
print(allocate_training_output(sys.argv[3], Path(sys.argv[2]), day=date(2026, 10, 15)))
"""
        processes = []
        try:
            for index in range(6):
                processes.append(subprocess.Popen(
                    [sys.executable, "-I", "-B", "-c", worker, str(RESOURCES), str(self.workspace), f"datasets/scene-{index}"],
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    text=True))
            for process in processes:
                process.stdin.write("start\n")
                process.stdin.flush()
            outputs = []
            for process in processes:
                stdout, stderr = process.communicate(timeout=30)
                self.assertEqual(process.returncode, 0, stderr)
                outputs.append(Path(stdout.strip()))
            self.assertEqual(len(set(outputs)), 6)
            self.assertEqual({int(path.parent.name.rsplit("-E", 1)[1]) for path in outputs}, set(range(6)))
            self.assertTrue(all(path.is_dir() for path in outputs))
        finally:
            for process in processes:
                if process.poll() is None:
                    process.kill()
                process.communicate()


if __name__ == "__main__":
    unittest.main(verbosity=2)
