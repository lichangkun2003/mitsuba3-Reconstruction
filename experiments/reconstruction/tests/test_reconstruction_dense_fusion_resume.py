"""CPU-only checks for resuming dense fusion with different execution resources."""

from __future__ import annotations

import copy
from contextlib import redirect_stdout
from datetime import datetime, timedelta
import io
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import generate_dense_points as generator
from test_reconstruction_colmap_dense import write_fixture


def identity():
    return {
        "version": 1,
        "input": {"source_model": "dataset/sparse/0", "image_root": "dataset/images_2",
                  "split": "train", "selected_image_count": 7},
        "colmap": {"path": "colmap.exe", "size": 100, "mtime_ns": 123},
        "sparse": [{"path": "dataset/sparse/0/points3D.bin", "sha256": "original"}],
        "images": [{"path": "dataset/images_2/one.png", "size": 80, "mtime_ns": 456}],
        "commands": [
            ["colmap.exe", "image_undistorter", "--image_path", "dataset/images_2",
             "--input_path", "cache/scaled_sparse", "--output_path", "cache/dense", "--num_threads", "8"],
            ["colmap.exe", "patch_match_stereo", "--workspace_path", "cache/dense",
             "--PatchMatchStereo.geom_consistency", "1", "--PatchMatchStereo.cache_size", "4.0",
             "--PatchMatchStereo.num_threads", "8"],
            ["colmap.exe", "stereo_fusion", "--workspace_path", "cache/dense",
             "--workspace_format", "COLMAP", "--input_type", "geometric",
             "--output_path", "cache/fused.ply", "--StereoFusion.use_cache", "1",
             "--StereoFusion.cache_size", "4.0", "--StereoFusion.max_image_size", "-1",
             "--StereoFusion.num_threads", "8"],
        ],
    }


def change_option(command, option, value):
    command[command.index(option) + 1] = value


def assert_execution_change(test, history, before, after):
    test.assertEqual(len(history), 1)
    change = history[0]
    test.assertEqual(change["from_command"], before)
    test.assertEqual(change["to_command"], after)
    changed = datetime.fromisoformat(change["changed_utc"])
    test.assertEqual(changed.utcoffset(), timedelta(0))


def manifest(completed=False):
    stages = {"undistort": {"seconds": 1.25}, "patch_match": {"seconds": 99.5}}
    if completed:
        stages["fusion"] = {"seconds": 8.75}
    return {"identity": identity(), "stages": stages,
            "status": "complete" if completed else "failed", "current_stage": "fusion",
            "last_run_seconds": 120., "error": "previous fusion interrupted"}


class FusionManifestTests(unittest.TestCase):
    def test_identical_old_and_new_identities_are_noops(self):
        for use_cache in ("0", "1"):
            for completed in (False, True):
                with self.subTest(use_cache=use_cache, completed=completed):
                    previous = manifest(completed)
                    change_option(previous["identity"]["commands"][2], "--StereoFusion.use_cache", use_cache)
                    before = copy.deepcopy(previous)
                    self.assertIsNone(generator.reconcile_manifest_identity(previous, copy.deepcopy(previous["identity"])))
                    self.assertEqual(previous, before)

    def test_only_fusion_resource_changes_migrate_unfinished_stage_and_record_provenance(self):
        changes = (("--StereoFusion.use_cache", "0"), ("--StereoFusion.cache_size", "12.0"),
                   ("--StereoFusion.num_threads", "4"))
        for selected in ((change,) for change in changes):
            self.check_resource_migration(selected)
        self.check_resource_migration(changes)

    def check_resource_migration(self, changes):
        with self.subTest(changes=changes):
            previous = manifest()
            before = copy.deepcopy(previous)
            requested = copy.deepcopy(previous["identity"])
            for option, value in changes:
                change_option(requested["commands"][2], option, value)
            self.assertIsNone(generator.reconcile_manifest_identity(previous, requested))
            self.assertEqual(previous["identity"], requested)
            self.assertEqual(previous["stages"], before["stages"])
            assert_execution_change(self, previous["fusion_execution_changes"],
                                    before["identity"]["commands"][2], requested["commands"][2])
            for field in ("status", "current_stage", "last_run_seconds", "error"):
                self.assertEqual(previous[field], before[field])
            # Repeating the same request must not invent another migration.
            after = copy.deepcopy(previous)
            generator.reconcile_manifest_identity(previous, copy.deepcopy(requested))
            self.assertEqual(previous, after)

    def test_completed_fusion_reuses_existing_identity_and_artifact_provenance(self):
        previous = manifest(completed=True)
        previous["output"] = "dataset/init_points.ply"
        previous["vertex_count"] = 1234
        previous["fusion_execution_changes"] = [{"from_command": ["older"], "to_command": ["old"]}]
        requested = copy.deepcopy(previous["identity"])
        for option, value in (("--StereoFusion.use_cache", "0"), ("--StereoFusion.cache_size", "12.0"),
                              ("--StereoFusion.num_threads", "4")):
            change_option(requested["commands"][2], option, value)
        before = copy.deepcopy(previous)
        self.assertIsNone(generator.reconcile_manifest_identity(previous, requested))
        self.assertEqual(previous, before)

    def test_input_and_previous_stage_changes_are_rejected_without_mutation(self):
        def alter_input(value):
            value["input"]["split"] = "all"

        def alter_sparse(value):
            value["sparse"][0]["sha256"] = "changed"

        def alter_image(value):
            value["images"][0]["mtime_ns"] += 1

        def alter_executable(value):
            value["colmap"]["mtime_ns"] += 1

        cases = {
            "input split": alter_input, "source sparse": alter_sparse, "image file": alter_image,
            "COLMAP executable": alter_executable,
            "undistort resources": lambda value: change_option(value["commands"][0], "--num_threads", "4"),
            "PatchMatch resources": lambda value: change_option(value["commands"][1], "--PatchMatchStereo.cache_size", "8.0"),
            "PatchMatch geometry": lambda value: change_option(value["commands"][1], "--PatchMatchStereo.geom_consistency", "0"),
        }
        for completed in (False, True):
            for name, modify in cases.items():
                with self.subTest(completed=completed, changed=name):
                    previous = manifest(completed)
                    requested = copy.deepcopy(previous["identity"])
                    change_option(requested["commands"][2], "--StereoFusion.use_cache", "0")
                    modify(requested)
                    before = copy.deepcopy(previous)
                    with self.assertRaises(ValueError):
                        generator.reconcile_manifest_identity(previous, requested)
                    self.assertEqual(previous, before)

    def test_fusion_geometry_paths_and_added_options_are_rejected_without_mutation(self):
        changes = (("--StereoFusion.max_image_size", "800"), ("--input_type", "photometric"),
                   ("--workspace_path", "another/dense"), ("--output_path", "another/fused.ply"),
                   ("--workspace_format", "PMVS"))
        for completed in (False, True):
            for option, value in (*changes, (None, None)):
                with self.subTest(completed=completed, option=option):
                    previous = manifest(completed)
                    requested = copy.deepcopy(previous["identity"])
                    change_option(requested["commands"][2], "--StereoFusion.use_cache", "0")
                    if option is None:
                        requested["commands"][2] += ["--StereoFusion.min_num_pixels", "3"]
                    else:
                        change_option(requested["commands"][2], option, value)
                    before = copy.deepcopy(previous)
                    with self.assertRaises(ValueError):
                        generator.reconcile_manifest_identity(previous, requested)
                    self.assertEqual(previous, before)

    def test_argument_default_and_explicit_cache_modes(self):
        for options, expected in (([], False), (["--fusion-use-cache"], True),
                                  (["--no-fusion-use-cache"], False)):
            with self.subTest(options=options), patch.object(sys, "argv", ["generate_dense_points", *options]):
                args = generator.parse_arguments()
                self.assertIs(args.fusion_use_cache, expected)
                command = generator.commands_for(args, Path("colmap.exe"), Path("images"), Path("work"))[2][1]
                self.assertEqual(command[command.index("--StereoFusion.use_cache") + 1], str(int(expected)))


class FusionResumeIntegrationTests(unittest.TestCase):
    def test_failed_fusion_resumes_alone_with_new_cache_mode_and_keeps_sources(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "datasets" / "bonsai"
            write_fixture(dataset)
            (dataset / "images_4").rename(dataset / "images_2")
            work_root = root / "cache"
            executable = root / "colmap.exe"
            executable.write_bytes(b"fake executable is never launched")

            def snapshot():
                return {str(path.relative_to(dataset)): path.read_bytes() if path.is_file() else None
                        for path in dataset.rglob("*")}

            original = snapshot()
            fused = (b"ply\nformat binary_little_endian 1.0\nelement vertex 1\n"
                     b"property float x\nproperty float y\nproperty float z\n"
                     b"property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
                     + struct.pack("<fffBBB", 1., 2., 3., 100, 120, 140))
            calls = []

            def simulate_colmap(command, env, log_path):
                calls.append([str(item) for item in command])
                if command[1] != "stereo_fusion":
                    return
                if command[command.index("--StereoFusion.use_cache") + 1] == "1":
                    raise RuntimeError("simulated slow fusion interrupted")
                Path(command[command.index("--output_path") + 1]).write_bytes(fused)

            with patch.object(generator, "DATASETS_ROOT", dataset.parent), \
                 patch.object(generator, "DENSE_WORK_ROOT", work_root), \
                 patch.object(generator, "run_command", side_effect=simulate_colmap), \
                 redirect_stdout(io.StringIO()):
                with patch.object(sys, "argv", ["generate_dense_points", "--scene", "bonsai",
                                                "--colmap", str(executable), "--fusion-use-cache"]):
                    args = generator.parse_arguments()
                with self.assertRaisesRegex(RuntimeError, "simulated slow fusion"):
                    generator.generate(args)
                self.assertEqual([command[1] for command in calls],
                                 ["image_undistorter", "patch_match_stereo", "stereo_fusion"])
                self.assertEqual(snapshot(), original)
                manifest_path = work_root / "bonsai" / "images_2-train" / "dense-init.json"
                failed = json.loads(manifest_path.read_text(encoding="utf-8"))
                self.assertEqual(failed["status"], "failed")
                self.assertEqual(set(failed["stages"]), {"undistort", "patch_match"})
                scaled = manifest_path.parent / "scaled_sparse"
                scaled_before = {path.name: (path.read_bytes(), path.stat().st_mtime_ns) for path in scaled.iterdir()}
                calls.clear()
                args.fusion_use_cache = False
                with patch.object(generator, "prepare_dense_model", side_effect=AssertionError("reprepared sparse model")):
                    generator.generate(args)
                self.assertEqual([command[1] for command in calls], ["stereo_fusion"])
                self.assertEqual(snapshot(), {**original, "init_points.ply": fused})
                self.assertEqual({path.name: (path.read_bytes(), path.stat().st_mtime_ns) for path in scaled.iterdir()},
                                 scaled_before)
                complete = json.loads(manifest_path.read_text(encoding="utf-8"))
                self.assertEqual(complete["status"], "complete")
                self.assertEqual(set(complete["stages"]), {"undistort", "patch_match", "fusion"})
                for stage in ("undistort", "patch_match"):
                    self.assertEqual(complete["stages"][stage], failed["stages"][stage])
                assert_execution_change(self, complete["fusion_execution_changes"],
                                        failed["identity"]["commands"][2], calls[0])
                # A completed output can be republished without rerunning any
                # stage, even when the requested cache setting changes again.
                calls.clear()
                args.fusion_use_cache = True
                args.overwrite = True
                generator.generate(args)
                self.assertEqual(calls, [])
                reused = json.loads(manifest_path.read_text(encoding="utf-8"))
                self.assertEqual(reused["identity"], complete["identity"])
                self.assertEqual(reused["fusion_execution_changes"], complete["fusion_execution_changes"])
                self.assertEqual(snapshot(), {**original, "init_points.ply": fused})


if __name__ == "__main__":
    unittest.main(verbosity=2)
