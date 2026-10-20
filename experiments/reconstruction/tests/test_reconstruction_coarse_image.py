"""CPU checks for coarse-image timing and safe repeated/resumed output."""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train import TrainConfig, is_coarse_training_end, save_coarse_training_view


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


if __name__ == "__main__":
    unittest.main()
