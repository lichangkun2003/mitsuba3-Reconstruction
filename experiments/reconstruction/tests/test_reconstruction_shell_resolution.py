"""CPU regression for a common world-space background refinement ceiling."""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reconstruction_checkpoint import load_checkpoint, save_checkpoint, validate_model_state
from reconstruction_export import ArraySource
from reconstruction_world_shell import (
    ADAPTIVE_EXTENDED_TAG, ADAPTIVE_TAG, adaptive_decode_numpy, adaptive_depth_limit,
    adaptive_encode_numpy, adaptive_max_depth, leaf_boxes_numpy, validate_adaptive_cells_numpy,
)


def metadata(resolution=1024, levels=5, maximum=1024, mode="core"):
    result = dict(resolution=resolution, core_resolution=resolution,
        grid_min=[-1.] * 3, grid_max=[1.] * 3, sh_count=1, opacity_sh_count=1,
        scene_mapping="world_shell", world_shell_version=7, background_shell_levels=levels,
        background_refinement="adaptive", background_subdivision_level=0,
        background_geometry="ellipsoid", background_resolution_mode=mode,
        background_max_resolution=maximum)
    result["background_max_depth"] = adaptive_max_depth(result)
    return result


def state_for(cells):
    cells = np.asarray(cells, np.uint64)
    count = len(cells)
    return dict(cell_indices=cells, centers=np.full(count * 3, .5, np.float32),
        log_scales=np.zeros(count * 3, np.float32),
        rotations=np.tile(np.array([1., 0., 0., 0.], np.float32), count),
        radiance_sh=np.ones(count * 3, np.float32), opacity_sh=np.zeros(count, np.float32))


class ShellResolutionTests(unittest.TestCase):
    def test_each_shell_reaches_the_current_foreground_level(self):
        shells = np.arange(5)
        for n, expected in ((256, [7, 8, 9, 10, 11]),
                            (512, [8, 9, 10, 11, 12]),
                            (1024, [9, 10, 11, 12, 13])):
            with self.subTest(core_resolution=n):
                saved = metadata(n)
                depths = adaptive_depth_limit(saved, shells)
                np.testing.assert_array_equal(depths, expected)
                self.assertEqual(adaptive_max_depth(saved), expected[-1])
                cells = adaptive_encode_numpy(shells * 56, depths, np.zeros(5, np.uint64), np)
                lower, upper = leaf_boxes_numpy(cells, n, [-1.] * 3, [1.] * 3, 5, np)
                np.testing.assert_array_equal(upper - lower, np.full((5, 3), 2. / n))
                validate_model_state(state_for(cells), saved)

    def test_core_parent_keeps_the_previous_two_to_one_edge_policy(self):
        shells = np.arange(5)
        for n, expected in ((256, [6, 7, 8, 9, 10]),
                            (512, [7, 8, 9, 10, 11]),
                            (1024, [8, 9, 10, 11, 12])):
            with self.subTest(core_resolution=n):
                saved = metadata(n, maximum=512, mode="core_parent")
                saved["world_shell_version"] = 6
                depths = adaptive_depth_limit(saved, shells)
                np.testing.assert_array_equal(depths, expected)
                cells = adaptive_encode_numpy(shells * 56, depths, np.zeros(5, np.uint64), np)
                lower, upper = leaf_boxes_numpy(cells, n, [-1.] * 3, [1.] * 3, 5, np)
                np.testing.assert_array_equal(upper - lower, np.full((5, 3), 4. / n))
                validate_model_state(state_for(cells), saved)

    def test_coarse_roots_keep_doubling_and_existing_boxes_do_not_move_on_core_refine(self):
        shells = np.arange(5)
        cells = adaptive_encode_numpy(shells * 56, np.zeros(5, np.int32), np.zeros(5, np.uint64), np)
        low, high = leaf_boxes_numpy(cells, 256, [-1.] * 3, [1.] * 3, 5, np)
        np.testing.assert_array_equal(high - low, np.repeat(np.exp2(shells)[:, None], 3, axis=1))
        for n in (512, 1024):
            actual_low, actual_high = leaf_boxes_numpy(cells, n, [-1.] * 3, [1.] * 3, 5, np)
            np.testing.assert_array_equal(actual_low, low)
            np.testing.assert_array_equal(actual_high, high)

    def test_core_refinement_refreshes_policy_metadata_without_moving_background(self):
        from reconstruction_world_refinement import _metadata
        original = metadata(256)
        current = original
        cells = adaptive_encode_numpy([0, 224], [7, 11], [0, 2**33 - 1], np)
        expected_boxes = leaf_boxes_numpy(cells, 256, [-1.] * 3, [1.] * 3, 5, np)
        for n, maximum in ((512, 12), (1024, 13)):
            current = _metadata(current)
            self.assertEqual(current["resolution"], n)
            self.assertEqual(current["core_resolution"], n)
            self.assertEqual(current["background_max_depth"], maximum)
            self.assertEqual(current["background_resolution_mode"], "core")
            self.assertEqual(current["background_max_resolution"], 1024)
            validate_model_state(state_for(cells), current)
            for actual, expected in zip(leaf_boxes_numpy(cells, n, [-1.] * 3, [1.] * 3, 5, np), expected_boxes):
                np.testing.assert_array_equal(actual, expected)
        self.assertEqual(original["resolution"], 256)
        self.assertEqual(original["background_max_depth"], 11)

    def test_explicit_world_ceiling_and_legacy_per_shell_policy_remain_distinct(self):
        saved = metadata(1024, maximum=128)
        np.testing.assert_array_equal(adaptive_depth_limit(saved, np.arange(5)), [6, 7, 8, 9, 10])
        cells = adaptive_encode_numpy(np.arange(5) * 56, adaptive_depth_limit(saved, np.arange(5)),
                                       np.zeros(5, np.uint64), np)
        lo, hi = leaf_boxes_numpy(cells, 1024, [-1.] * 3, [1.] * 3, 5, np)
        np.testing.assert_array_equal(hi - lo, np.full((5, 3), 2. / 128))
        legacy = dict(saved, world_shell_version=5, background_max_depth=8)
        legacy.pop("background_resolution_mode")
        legacy.pop("background_max_resolution")
        np.testing.assert_array_equal(np.broadcast_to(adaptive_depth_limit(legacy, np.arange(5)), (5,)), [8] * 5)
        self.assertEqual(adaptive_max_depth(legacy), 8)

    def test_extended_paths_preserve_legacy_addresses_and_all_morton_bits(self):
        coarse = np.array([0, 55, 112, 224, 504])
        depths = np.array([0, 9, 10, 12, 17])
        morton = np.array([0, 2**27 - 1, 2**30 - 3, 2**36 - 1, 2**51 - 5], np.uint64)
        cells = adaptive_encode_numpy(coarse, depths, morton, np)
        for i in (0, 1):
            self.assertEqual(int(cells[i]), ADAPTIVE_TAG | (int(coarse[i]) << 32)
                             | (1 << (3 * int(depths[i]))) | int(morton[i]))
        self.assertTrue(np.all((cells[2:] & np.uint64(ADAPTIVE_EXTENDED_TAG)) != 0))
        decoded = adaptive_decode_numpy(cells, 10, np)
        for actual, expected in zip(decoded, (coarse, depths, morton)):
            np.testing.assert_array_equal(actual, expected)
        self.assertEqual(len(np.unique(cells)), len(cells))
        validate_adaptive_cells_numpy(cells, 1024, 10, np)
        # A depth-10 path in the old layout remains malformed. An extended
        # depth-9 alias must not become a second address for an existing leaf.
        for invalid in (ADAPTIVE_TAG | (1 << 30),
                        ADAPTIVE_TAG | ADAPTIVE_EXTENDED_TAG | (1 << 27)):
            with self.subTest(address=invalid), self.assertRaises(ValueError):
                adaptive_decode_numpy(np.array([invalid], np.uint64), 10, np)

    def test_ancestor_conflicts_cross_old_and_extended_encoding_boundary(self):
        coarse = 112
        parent = adaptive_encode_numpy([coarse], [9], [2**27 - 2], np)
        child = adaptive_encode_numpy([coarse], [10], [(2**27 - 2) * 8 + 3], np)
        sibling = adaptive_encode_numpy([coarse], [9], [2**27 - 1], np)
        validate_adaptive_cells_numpy(np.r_[parent, sibling], 1024, 5, np)
        with self.assertRaisesRegex(ValueError, "overlap"):
            validate_adaptive_cells_numpy(np.r_[parent, child], 1024, 5, np)

    def test_ten_shells_reach_depth_eighteen_with_distinct_full_paths(self):
        saved = metadata(levels=10)
        np.testing.assert_array_equal(adaptive_depth_limit(saved, np.arange(10)), np.arange(9, 19))
        self.assertEqual(adaptive_max_depth(saved), 18)
        coarse = np.array([504, 504, 505, 559, 559])
        depths = np.full(5, 18)
        morton = np.array([0, 2**54 - 1, 0, 0, 2**54 - 1], np.uint64)
        cells = adaptive_encode_numpy(coarse, depths, morton, np)
        expected = np.uint64(0xF << 60) | ((coarse.astype(np.uint64) - 504) << np.uint64(54)) | morton
        np.testing.assert_array_equal(cells, expected)
        for actual, original in zip(adaptive_decode_numpy(cells, 10, np), (coarse, depths, morton)):
            np.testing.assert_array_equal(actual, original)
        self.assertEqual(len(np.unique(cells)), len(cells))
        lower, upper = leaf_boxes_numpy(cells, 1024, [-1.] * 3, [1.] * 3, 10, np)
        np.testing.assert_array_equal(upper - lower, np.full((5, 3), 2. / 1024))
        state = state_for(cells)
        validate_model_state(state, saved)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "deepest-background.npz"
            streamed = {name: ArraySource(value.shape, value.dtype,
                lambda start, count, value=value: value[start:start + count], max_chunk_elements=1)
                for name, value in state.items()}
            save_checkpoint(path, streamed, saved)
            restored = load_checkpoint(path)
            self.assertEqual(restored["model_metadata"], saved)
            np.testing.assert_array_equal(restored["model_state"]["cell_indices"], cells)
        with self.assertRaises(ValueError):
            adaptive_encode_numpy([503], [18], [0], np)
        parent = adaptive_encode_numpy([559], [17], [2**51 - 1], np)
        with self.assertRaisesRegex(ValueError, "overlap"):
            validate_adaptive_cells_numpy(np.r_[parent, cells[-1:]], 1024, 10, np)

    def test_core_resolution_policy_requires_version_seven(self):
        saved = metadata(256)
        cells = adaptive_encode_numpy([0], [0], [0], np)
        for version in (5, 6):
            with self.subTest(version=version), self.assertRaises(ValueError):
                validate_model_state(state_for(cells), dict(saved, world_shell_version=version))

    def test_checkpoint_and_streamed_save_enforce_each_shell_limit(self):
        saved = metadata(256)
        allowed = adaptive_encode_numpy([0, 224], [7, 11], [0, 2**33 - 1], np)
        state = state_for(allowed)
        config = {"data": {"background_resolution_mode": "core", "background_max_resolution": 1024}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "background.npz"
            save_checkpoint(path, state, saved, config=config)
            restored = load_checkpoint(path)
            np.testing.assert_array_equal(restored["model_state"]["cell_indices"], allowed)
            self.assertEqual(restored["model_metadata"], saved)
            original = path.read_bytes()
            # Depth 8 is below the global maximum 11, but already too fine
            # in the nearest shell while the foreground is still 256.
            invalid = state_for(adaptive_encode_numpy([0, 224], [8, 11], [0, 2**33 - 1], np))
            with self.assertRaises(ValueError):
                validate_model_state(invalid, saved)
            streamed = {name: ArraySource(value.shape, value.dtype,
                lambda start, count, value=value: value[start:start + count], max_chunk_elements=1)
                for name, value in invalid.items()}
            with self.assertRaises(ValueError):
                save_checkpoint(path, streamed, saved, config=config)
            self.assertEqual(path.read_bytes(), original)
            # Refining only the foreground lifts every shell's ceiling by
            # one level; the same previously refused cell can then be saved.
            validate_model_state(invalid, metadata(512))


if __name__ == "__main__":
    unittest.main(verbosity=2)
