"""CPU checks for coarse-image timing and safe repeated/resumed output."""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train import (TrainConfig, is_coarse_training_end, is_refinement_boundary,
                   save_coarse_training_view, save_refinement_boundary,
                   record_refinement_boundary, refinement_checkpoint_epochs, update)


class CoarseImageTests(unittest.TestCase):
    def topology(self, **overrides):
        values = dict(schedule=SimpleNamespace(enabled=True, start_resolution=256,
                                               target_resolution=512),
                      resolution=256, level_start_epoch=0, level_budget=600)
        values.update(overrides)
        return SimpleNamespace(**values)

    def test_waits_for_all_coarse_epochs_and_excludes_fine_epochs(self):
        self.assertFalse(is_coarse_training_end(self.topology(), 599))
        self.assertTrue(is_coarse_training_end(self.topology(), 600))
        # Pending-transition legacy checkpoints may restore at the boundary.
        self.assertTrue(is_coarse_training_end(self.topology(), 601))
        self.assertFalse(is_coarse_training_end(
            self.topology(resolution=512, level_start_epoch=600, level_budget=400), 1000))

    def test_no_snapshot_when_there_is_no_refinement(self):
        topology = self.topology()
        topology.schedule.enabled = False
        self.assertFalse(is_coarse_training_end(topology, 600))
        topology.schedule.enabled = True
        topology.schedule.target_resolution = 256
        self.assertFalse(is_coarse_training_end(topology, 600))

    def test_disabled_evaluation_does_not_render_or_create_output(self):
        config = TrainConfig()
        config.evaluation.enabled = False
        with tempfile.TemporaryDirectory() as temporary, \
                mock.patch("train.render_view", side_effect=AssertionError("unexpected rendering")):
            model_path = Path(temporary) / "experiment" / "model.npz"
            record = save_coarse_training_view(config, None, [], model_path,
                None, None, None, completed_epochs=600, global_step=60000)
            self.assertEqual(record["status"], "skipped")
            self.assertEqual(record["reason"], "evaluation_disabled")
            self.assertFalse(model_path.parent.exists())

    def test_existing_coarse_output_is_preserved_without_accessing_current_model(self):
        with tempfile.TemporaryDirectory() as temporary, \
                mock.patch("train.render_view", side_effect=AssertionError("unexpected rendering")):
            model_path = Path(temporary) / "model.npz"
            image = model_path.parent / "train" / "0000-coarse.png"
            image.parent.mkdir()
            image.write_bytes(b"historical coarse pixels")
            record = save_coarse_training_view(TrainConfig(), None, [], model_path,
                None, None, None, completed_epochs=1000, global_step=100000)
            self.assertEqual(record["status"], "skipped")
            self.assertEqual(record["reason"], "existing_image")
            self.assertEqual(image.read_bytes(), b"historical coarse pixels")

    def test_each_old_core_level_has_a_boundary_including_fixed_shell_storage(self):
        topology = self.topology(resolution=528, core_resolution=512,
                                 level_start_epoch=600, level_budget=450)
        topology.schedule.target_resolution = 1024
        self.assertFalse(is_refinement_boundary(topology, 1049))
        self.assertTrue(is_refinement_boundary(topology, 1050))
        self.assertFalse(is_coarse_training_end(topology, 1050))
        topology.core_resolution = 1024
        self.assertFalse(is_refinement_boundary(topology, 1500))

    def test_legacy_save_count_does_not_schedule_periodic_snapshots(self):
        config = TrainConfig()
        config.training.num_epochs = 1500
        config.schedule.target_resolution = 1024
        config.schedule.level_weights = (4, 3, 3)
        for count in (1, 3, 100):
            update(config, {"training": {"save_count": count, "save_every_epochs": 2}})
            self.assertEqual(refinement_checkpoint_epochs(config), (600, 1050))
        config.schedule.enabled = False
        self.assertEqual(refinement_checkpoint_epochs(config), ())
        config.schedule.enabled = True
        config.schedule.target_resolution = config.schedule.start_resolution
        self.assertEqual(refinement_checkpoint_epochs(config), ())

    def boundary(self, directory, topology, *, config=None, completed=600, primary=True, save=None):
        model = SimpleNamespace(resolution=topology.resolution)
        return save_refinement_boundary(config or TrainConfig(), topology, model, [],
            directory / "model.npz", directory, None, None, None, save or mock.Mock(),
            completed_epochs=completed, global_step=completed * 169, primary=primary)

    def test_distinct_level_images_render_old_models_after_full_pending_save(self):
        events = []
        def save(path, completed, **kwargs):
            self.assertTrue(kwargs["pending_epoch_transition"])
            events.append(("save", path.name, completed))
            path.write_bytes(b"complete model Adam RNG and lifecycle")
        def render(config, model, views, path, *args, **kwargs):
            events.append(("render", model.resolution, kwargs["completed_epochs"]))
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(str(model.resolution).encode())
            return dict(status="passed", image=str(path), resolution=model.resolution, seconds=.25)
        with tempfile.TemporaryDirectory() as temporary, \
                mock.patch("train._save_training_view_zero", side_effect=render):
            directory = Path(temporary)
            coarse = self.topology(resolution=272, core_resolution=256)
            coarse.schedule.target_resolution = 1024
            first = self.boundary(directory, coarse, save=save)
            fine = self.topology(resolution=528, core_resolution=512,
                                 level_start_epoch=600, level_budget=450)
            fine.schedule.target_resolution = 1024
            second = self.boundary(directory, fine, completed=1050, save=save)
            self.assertEqual(events, [
                ("save", "before-refine-0256-epoch-00600.npz", 600), ("render", 272, 600),
                ("save", "before-refine-0512-epoch-01050.npz", 1050), ("render", 528, 1050)])
            self.assertEqual((directory / "train/0000-coarse.png").read_bytes(), b"272")
            self.assertEqual((directory / "train/0000-level-0256.png").read_bytes(), b"272")
            self.assertEqual((directory / "train/0000-level-0512.png").read_bytes(), b"528")
            report = {}
            record_refinement_boundary(report, first)
            record_refinement_boundary(report, second)
            self.assertEqual(set(report["level_renders"]), {"256", "512"})
            self.assertEqual(report["coarse_render"]["core_resolution"], 256)
            self.assertEqual(set(report["pre_refine_checkpoints"]), {"256", "512"})

    def test_pending_resume_preserves_existing_checkpoint_and_level_image(self):
        with tempfile.TemporaryDirectory() as temporary, \
                mock.patch("train._save_training_view_zero", side_effect=AssertionError("rerender")):
            directory = Path(temporary)
            checkpoint = directory / "before-refine-0256-epoch-00600.npz"
            checkpoint.write_bytes(b"original pending checkpoint")
            image = directory / "train/0000-level-0256.png"
            image.parent.mkdir()
            image.write_bytes(b"original level image")
            save = mock.Mock(side_effect=AssertionError("overwrote source checkpoint"))
            result = self.boundary(directory, self.topology(), save=save)
            self.assertEqual(result["checkpoint"]["reason"], "existing_checkpoint")
            self.assertEqual(result["render"]["reason"], "existing_image")
            self.assertEqual(checkpoint.read_bytes(), b"original pending checkpoint")
            self.assertEqual(image.read_bytes(), b"original level image")

    def test_legacy_coarse_image_becomes_first_level_only_without_rendering(self):
        with tempfile.TemporaryDirectory() as temporary, \
                mock.patch("train._save_training_view_zero", side_effect=AssertionError("rerender")):
            directory = Path(temporary)
            coarse = directory / "train/0000-coarse.png"
            coarse.parent.mkdir()
            coarse.write_bytes(b"legacy first level")
            result = self.boundary(directory, self.topology())
            self.assertEqual(result["render"]["reason"], "existing_coarse_image")
            self.assertEqual((directory / "train/0000-level-0256.png").read_bytes(), coarse.read_bytes())

    def test_disabled_evaluation_still_saves_complete_boundary_checkpoint(self):
        config = TrainConfig()
        config.evaluation.enabled = False
        with tempfile.TemporaryDirectory() as temporary, \
                mock.patch("train._save_training_view_zero", side_effect=AssertionError("render")):
            save = mock.Mock()
            directory = Path(temporary)
            result = self.boundary(directory, self.topology(), config=config, save=save)
            save.assert_called_once_with(directory / "before-refine-0256-epoch-00600.npz", 600,
                                         pending_epoch_transition=True)
            self.assertEqual(result["render"]["reason"], "evaluation_disabled")
            self.assertFalse((directory / "train").exists())

    def test_secondary_rank_and_incomplete_epoch_do_not_access_model_or_files(self):
        save = mock.Mock(side_effect=AssertionError("secondary save"))
        for primary, completed in ((False, 600), (True, 599)):
            with self.subTest(primary=primary, completed=completed), \
                    mock.patch("train._save_training_view_zero", side_effect=AssertionError("render")):
                result = save_refinement_boundary(TrainConfig(), self.topology(), None, [],
                    None, None, None, None, None, save,
                    completed_epochs=completed, global_step=0, primary=primary)
                self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
