"""CPU checks for adaptive checkpoint milestones and completed-state previews."""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train import (TrainConfig, adaptive_checkpoint_epochs,
                   refinement_checkpoint_epochs, save_epoch_checkpoint)


class AdaptiveCheckpointTests(unittest.TestCase):
    def config(self, *, epochs=1500, fraction=.2):
        config = TrainConfig()
        config.schedule.refinement = "adaptive"
        config.training.num_epochs = epochs
        config.topology.freeze_last_fraction = fraction
        return config

    def test_saves_halfway_to_freeze_at_freeze_and_at_completion(self):
        self.assertEqual(adaptive_checkpoint_epochs(self.config()), (600, 1200, 1500))
        self.assertEqual(adaptive_checkpoint_epochs(self.config(epochs=3000)),
                         (1200, 2400, 3000))

    def test_freeze_and_its_midpoint_round_down_to_completed_epochs(self):
        self.assertEqual(adaptive_checkpoint_epochs(self.config(epochs=9)), (3, 7, 9))
        self.assertEqual(adaptive_checkpoint_epochs(self.config(epochs=1501)),
                         (600, 1200, 1501))

    def test_short_runs_have_unique_nonzero_milestones(self):
        for epochs, expected in ((1, (1,)), (2, (1, 2)), (3, (1, 2, 3))):
            with self.subTest(epochs=epochs):
                self.assertEqual(adaptive_checkpoint_epochs(self.config(epochs=epochs)), expected)
        self.assertEqual(adaptive_checkpoint_epochs(self.config(epochs=6, fraction=.99)), (1, 6))
        self.assertEqual(adaptive_checkpoint_epochs(self.config(fraction=0)), (750, 1500))

    def test_legacy_save_count_does_not_move_adaptive_milestones(self):
        config = self.config()
        for count in (1, 3, 100):
            config.training.save_count = count
            self.assertEqual(adaptive_checkpoint_epochs(config), (600, 1200, 1500))

    def test_uniform_mode_keeps_its_existing_pre_refinement_plan(self):
        config = self.config()
        config.schedule.refinement = "uniform"
        config.schedule.start_resolution = 256
        config.schedule.target_resolution = 1024
        config.schedule.level_weights = (4, 3, 3)
        self.assertEqual(adaptive_checkpoint_epochs(config), ())
        self.assertEqual(refinement_checkpoint_epochs(config), (600, 1050))

    def save(self, directory, *, config=None, completed=600, save=None, primary=True):
        config = config or self.config()
        output = directory / "training"
        output.mkdir(exist_ok=True)
        return save_epoch_checkpoint(config, self.model, self.views,
            directory / "model.npz", output, None, None, None, save or mock.Mock(),
            completed_epochs=completed, global_step=completed * 169, primary=primary)

    def setUp(self):
        self.model = object()
        self.views = [{"index": 1, "camera": object()}]

    def test_checkpoint_is_completed_before_render_and_final_uses_model_path(self):
        events = []

        def save(path, completed, **kwargs):
            self.assertFalse(kwargs["pending_epoch_transition"])
            events.append(("save", path.name, completed))
            path.write_bytes(b"completed model optimizer and topology")

        def render(config, model, views, path, *args, **kwargs):
            self.assertIs(model, self.model)
            self.assertIs(views, self.views)
            completed = kwargs["completed_epochs"]
            self.assertEqual(kwargs["global_step"], completed * 169)
            events.append(("render", path.name, completed))
            return dict(status="passed", image=str(path))

        with tempfile.TemporaryDirectory() as temporary, \
                mock.patch("train._save_training_view_zero", side_effect=render):
            directory = Path(temporary)
            for completed in (600, 1200, 1500):
                result = self.save(directory, completed=completed, save=save)
                expected = (directory / "model.npz" if completed == 1500 else
                            directory / "training" / f"epoch-{completed:05d}.npz")
                self.assertEqual(result["checkpoint"], dict(status="passed", path=str(expected),
                    completed_epochs=completed, global_step=completed * 169,
                    pending_epoch_transition=False))
                self.assertEqual(result["render"]["image"],
                    str(directory / "train" / f"0000-epoch-{completed:05d}.png"))
            self.assertEqual(events, [
                ("save", "epoch-00600.npz", 600), ("render", "0000-epoch-00600.png", 600),
                ("save", "epoch-01200.npz", 1200), ("render", "0000-epoch-01200.png", 1200),
                ("save", "model.npz", 1500), ("render", "0000-epoch-01500.png", 1500)])
            self.assertFalse((directory / "training/epoch-01500.npz").exists())

    def test_explicit_stop_can_save_off_schedule_as_completed_state(self):
        config = self.config()
        self.assertNotIn(617, adaptive_checkpoint_epochs(config))
        with tempfile.TemporaryDirectory() as temporary, \
                mock.patch("train._save_training_view_zero", return_value={}) as render:
            directory = Path(temporary)
            save = mock.Mock()
            result = self.save(directory, config=config, completed=617, save=save)
            save.assert_called_once_with(directory / "training/epoch-00617.npz", 617,
                                         pending_epoch_transition=False)
            self.assertFalse(result["checkpoint"]["pending_epoch_transition"])
            self.assertEqual(render.call_args.kwargs["completed_epochs"], 617)

    def test_repeated_epoch_refreshes_checkpoint_and_its_preview(self):
        def save(path, completed, **kwargs):
            path.write_bytes(b"new completed checkpoint")

        def render(config, model, views, path, *args, **kwargs):
            path.write_bytes(b"new completed-state preview")
            return dict(status="passed", image=str(path))

        with tempfile.TemporaryDirectory() as temporary, \
                mock.patch("train._save_training_view_zero", side_effect=render) as rendered:
            directory = Path(temporary)
            checkpoint = directory / "training/epoch-00600.npz"
            image = directory / "train/0000-epoch-00600.png"
            checkpoint.parent.mkdir()
            image.parent.mkdir()
            checkpoint.write_bytes(b"old checkpoint")
            image.write_bytes(b"old preview")
            self.save(directory, save=save)
            self.assertEqual(checkpoint.read_bytes(), b"new completed checkpoint")
            self.assertEqual(image.read_bytes(), b"new completed-state preview")
            rendered.assert_called_once()

    def test_failed_save_does_not_render(self):
        with tempfile.TemporaryDirectory() as temporary, \
                mock.patch("train._save_training_view_zero") as render:
            save = mock.Mock(side_effect=OSError(28, "No space left on device"))
            with self.assertRaises(OSError):
                self.save(Path(temporary), save=save)
            render.assert_not_called()

    def test_render_failure_preserves_the_completed_checkpoint(self):
        def save(path, completed, **kwargs):
            path.write_bytes(b"recoverable completed checkpoint")

        with tempfile.TemporaryDirectory() as temporary, \
                mock.patch("train._save_training_view_zero", side_effect=RuntimeError("render failed")):
            directory = Path(temporary)
            with self.assertRaisesRegex(RuntimeError, "render failed"):
                self.save(directory, save=save)
            self.assertEqual((directory / "training/epoch-00600.npz").read_bytes(),
                             b"recoverable completed checkpoint")

    def test_disabled_evaluation_still_saves_checkpoint(self):
        config = self.config()
        config.evaluation.enabled = False
        with tempfile.TemporaryDirectory() as temporary, \
                mock.patch("train._save_training_view_zero") as render:
            directory = Path(temporary)
            save = mock.Mock()
            result = self.save(directory, config=config, save=save)
            save.assert_called_once_with(directory / "training/epoch-00600.npz", 600,
                                         pending_epoch_transition=False)
            self.assertEqual(result["render"], dict(status="skipped", reason="evaluation_disabled"))
            render.assert_not_called()

    def test_secondary_rank_does_not_access_config_model_or_files(self):
        save = mock.Mock()
        with mock.patch("train._save_training_view_zero") as render:
            result = save_epoch_checkpoint(None, None, None, None, None, None, None, None, save,
                completed_epochs=600, global_step=0, primary=False)
            self.assertIsNone(result)
            save.assert_not_called()
            render.assert_not_called()


if __name__ == "__main__":
    unittest.main()
