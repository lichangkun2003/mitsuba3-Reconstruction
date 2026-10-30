"""CPU checks for sequential missing-scene dense generation, without COLMAP."""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import io
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import generate_dense_points_all as batch


def write_ply(path):
    """A complete binary colored point accepted by the real output validator."""
    path.write_bytes(
        b"ply\nformat binary_little_endian 1.0\nelement vertex 1\n"
        b"property float x\nproperty float y\nproperty float z\n"
        b"property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
        + struct.pack("<fffBBB", 1., 2., 3., 101, 102, 103))


class DenseBatchTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="dense-batch-cpu-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def scenes(self, *names):
        paths = [self.root / name for name in names]
        for path in paths:
            path.mkdir()
        return paths

    def run_quietly(self, datasets, options):
        output, errors = io.StringIO(), io.StringIO()
        with redirect_stdout(output), redirect_stderr(errors):
            status = batch.run_batch(datasets, options)
        return status, output.getvalue(), errors.getvalue()

    def test_discovery_includes_installed_360_scenes_only_in_sorted_order(self):
        self.scenes("stump", "lego", "counter", "garden", "chair", "Bonsai")
        (self.root / "room").write_text("a file is not an installed scene")
        self.assertEqual([p.name for p in batch.discover_scenes(self.root)],
                         ["Bonsai", "counter", "garden", "stump"])

    def test_explicit_scene_selection_is_deduplicated_and_missing_is_reported(self):
        self.scenes("counter", "room")
        self.assertEqual([p.name for p in batch.discover_scenes(
            self.root, ["room", "counter", "room"])], ["counter", "room"])
        with self.assertRaisesRegex(FileNotFoundError, "garden"):
            batch.discover_scenes(self.root, ["counter", "garden"])

    def test_valid_existing_output_skips_even_without_images_or_sparse_model(self):
        scene, = self.scenes("bonsai")
        output = scene / "init_points.ply"
        write_ply(output)
        before = output.read_bytes()
        _, options = batch.parse_arguments([])
        with patch.object(batch.single, "generate", side_effect=AssertionError("must skip")):
            status, stdout, stderr = self.run_quietly([scene], options)
        self.assertEqual(status, 0)
        self.assertIn("skipped: 1", stdout)
        self.assertEqual(stderr, "")
        self.assertEqual(output.read_bytes(), before)

    def test_failure_continues_in_order_and_returns_nonzero_after_other_scenes_finish(self):
        datasets = self.scenes("counter", "garden", "kitchen")
        _, options = batch.parse_arguments([])
        calls = []

        def generate(args):
            calls.append(args.dataset.name)
            if args.dataset.name == "garden":
                raise RuntimeError("simulated fusion failure")
            write_ply(args.dataset / "init_points.ply")

        with patch.object(batch.single, "generate", side_effect=generate):
            status, stdout, stderr = self.run_quietly(datasets, options)
        self.assertEqual(calls, ["counter", "garden", "kitchen"])
        self.assertEqual(status, 1)
        self.assertIn("Generated: 2 | skipped: 0 | failed: 1", stdout)
        self.assertIn("garden: simulated fusion failure", stderr)
        self.assertTrue((datasets[0] / "init_points.ply").is_file())
        self.assertFalse((datasets[1] / "init_points.ply").exists())
        self.assertTrue((datasets[2] / "init_points.ply").is_file())
        self.assertIsNone(options.dataset)
        self.assertEqual(options.scene, batch.single.SCENE)

    def test_corrupt_existing_point_cloud_is_preserved_and_not_regenerated(self):
        damaged, following = self.scenes("counter", "garden")
        output = damaged / "init_points.ply"
        write_ply(output)
        original = output.read_bytes()[:-1]
        output.write_bytes(original)
        _, options = batch.parse_arguments([])
        calls = []

        def generate(args):
            calls.append(args.dataset.name)
            write_ply(args.dataset / "init_points.ply")

        with patch.object(batch.single, "generate", side_effect=generate):
            status, _, stderr = self.run_quietly([damaged, following], options)
        self.assertEqual(status, 1)
        self.assertEqual(calls, ["garden"])
        self.assertIn("Truncated fused PLY", stderr)
        self.assertEqual(output.read_bytes(), original)

    def test_keyboard_interrupt_stops_batch_without_launching_remaining_scene(self):
        datasets = self.scenes("counter", "garden", "room")
        _, options = batch.parse_arguments([])
        calls = []

        def generate(args):
            calls.append(args.dataset.name)
            if args.dataset.name == "garden":
                raise KeyboardInterrupt
            write_ply(args.dataset / "init_points.ply")

        with patch.object(batch.single, "generate", side_effect=generate):
            status, _, stderr = self.run_quietly(datasets, options)
        self.assertEqual(status, 130)
        self.assertEqual(calls, ["counter", "garden"])
        self.assertTrue((datasets[0] / "init_points.ply").is_file())
        self.assertFalse((datasets[2] / "init_points.ply").exists())
        self.assertIn("Batch interrupted", stderr)

    def test_dry_run_forwards_settings_and_scene_paths_without_outputs(self):
        datasets = self.scenes("counter", "garden")
        _, options = batch.parse_arguments([
            "--dry-run", "--gpu", "0,1", "--num-threads", "6", "--split", "all",
            "--cache-size", "8", "--iterations", "2", "--source-images", "12",
            "--max-image-size", "800", "--images", "images_4", "--fusion-use-cache"])
        calls = []

        def generate(args):
            self.assertTrue(args.dry_run)
            self.assertEqual(args.scene, args.dataset.name)
            self.assertIsNone(args.output)
            self.assertIsNone(args.workspace)
            self.assertFalse(args.overwrite)
            self.assertEqual((args.gpu, args.num_threads, args.split, args.cache_size,
                args.iterations, args.source_images, args.max_image_size, args.images,
                args.fusion_use_cache), ("0,1", 6, "all", 8., 2, 12, 800, "images_4", True))
            calls.append(args)

        with patch.object(batch.single, "generate", side_effect=generate):
            status, stdout, stderr = self.run_quietly(datasets, options)
        self.assertEqual(status, 0)
        self.assertEqual(len(calls), 2)
        self.assertIsNot(calls[0], calls[1])
        self.assertEqual([args.dataset for args in calls], datasets)
        self.assertTrue(all(not list(scene.iterdir()) for scene in datasets))
        self.assertIn("Validated: 2", stdout)
        self.assertEqual(stderr, "")

    def test_dry_run_abbreviation_cannot_silently_start_generation(self):
        # Either reject an abbreviation or preserve the parsed dry-run flag.
        # The single-scene parser must never accept it and then lose the flag.
        with redirect_stderr(io.StringIO()):
            try:
                _, options = batch.parse_arguments(["--dry"])
            except SystemExit as error:
                self.assertEqual(error.code, 2)
            else:
                self.assertTrue(options.dry_run)

    def test_per_scene_destinations_and_overwrite_are_rejected(self):
        for args in (["--scene", batch.single.SCENE], ["--scene=garden"],
                     ["--dataset", str(self.root)], ["--workspace", str(self.root)],
                     ["--output", str(self.root / "points.ply")], ["--overwrite"]):
            with self.subTest(args=args), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as caught:
                    batch.parse_arguments(args)
                self.assertEqual(caught.exception.code, 2)

    def test_prepare_only_does_not_require_a_published_cloud(self):
        scene, = self.scenes("counter")
        _, options = batch.parse_arguments(["--prepare-only"])
        with patch.object(batch.single, "generate") as generate:
            status, stdout, _ = self.run_quietly([scene], options)
        self.assertEqual(status, 0)
        generate.assert_called_once()
        self.assertTrue(generate.call_args.args[0].prepare_only)
        self.assertIn("Prepared: 1", stdout)

    def test_empty_installation_does_not_start_generator(self):
        _, options = batch.parse_arguments([])
        with patch.object(batch.single, "generate") as generate:
            status, stdout, _ = self.run_quietly([], options)
        self.assertEqual(status, 0)
        generate.assert_not_called()
        self.assertIn("No installed", stdout)


if __name__ == "__main__":
    unittest.main()
