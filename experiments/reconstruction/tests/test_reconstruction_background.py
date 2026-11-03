"""Check bounded/unbounded background initialization without running training.

CPU configuration checks run by default. --gpu also checks a small generated
shell with the source-built CUDA module; it never falls back to a CPU renderer.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import json
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import preview
from reconstruction_background import background_point_colors, initialize_background_model, prepare_sparse_background
from reconstruction_color import srgb_to_linear
from reconstruction_world_shell import background_cells_numpy, leaf_boxes_numpy


class BackgroundPointColorTests(unittest.TestCase):
    def colors(self, positions, colors, levels=2, lo=None, hi=None):
        return background_point_colors(np.asarray(positions), np.asarray(colors),
            [-1.] * 3 if lo is None else lo, [1.] * 3 if hi is None else hi, levels, np)

    def test_decodes_each_point_before_mean_and_ignores_core_and_outside(self):
        rgb = np.array([[1., 1., 1.], [1., .5, .02], [.2, .7, .04], [1., 1., 1.]])
        means, counts = self.colors([[0, 0, 0], [-1.8, -1.8, -1.8], [-1.7, -1.7, -1.7], [4, 0, 0]], rgb)
        self.assertEqual(means.shape, (112, 3))
        self.assertEqual(means.dtype, np.dtype(np.float32))
        self.assertEqual(counts.dtype, np.dtype(np.int64))
        self.assertEqual(int(counts.sum()), 2)
        self.assertEqual(counts[0], 2)
        expected = srgb_to_linear(rgb[1:3], np).astype(np.float64).mean(axis=0).astype(np.float32)
        np.testing.assert_array_equal(means[0], expected)
        np.testing.assert_array_equal(means[1:], 0.)
        self.assertGreater(float(np.max(np.abs(expected - srgb_to_linear(rgb[1:3].mean(axis=0), np)))), .01)

    def test_all_leaf_centers_match_address_order_in_translated_cube(self):
        lo, hi = [3., 4., 5.], [7., 8., 9.]
        cells = background_cells_numpy(4, 5, np)
        lower, upper = leaf_boxes_numpy(cells, 4, lo, hi, 5, np)
        colors = np.linspace(0., 1., len(cells) * 3).reshape(-1, 3)
        means, counts = self.colors((lower + upper) * .5, colors, levels=5, lo=lo, hi=hi)
        np.testing.assert_array_equal(counts, 1)
        np.testing.assert_array_equal(means, srgb_to_linear(colors, np))

    def test_half_open_negative_positive_and_near_boundaries(self):
        cells = background_cells_numpy(4, 2, np)
        lower, upper = leaf_boxes_numpy(cells, 4, [-1.] * 3, [1.] * 3, 2, np)
        values = [-1., -2., 1., 2., -4., 4., np.nextafter(-1., -np.inf),
                  np.nextafter(1., -np.inf), np.nextafter(2., -np.inf), np.nextafter(-2., -np.inf)]
        expected_shells = [None, 0, 0, 1, 1, None, 0, None, 0, 1]
        for x, shell in zip(values, expected_shells):
            with self.subTest(x=x):
                point = np.array([[x, 0., 0.]])
                means, counts = self.colors(point, [[.25, .5, .75]])
                expected = np.all((point >= lower) & (point < upper), axis=1)
                np.testing.assert_array_equal(counts, expected.astype(np.int64))
                if shell is None:
                    self.assertEqual(int(counts.sum()), 0)
                    np.testing.assert_array_equal(means, 0.)
                else:
                    row = int(np.flatnonzero(counts)[0])
                    self.assertEqual(row // 56, shell)
                    np.testing.assert_array_equal(means[row], srgb_to_linear([.25, .5, .75], np))

    def test_empty_points_keep_full_black_background_table(self):
        means, counts = self.colors(np.empty((0, 3)), np.empty((0, 3)), levels=8)
        self.assertEqual(means.shape, (448, 3))
        self.assertEqual(counts.shape, (448,))
        np.testing.assert_array_equal(means, 0.)
        np.testing.assert_array_equal(counts, 0)

    def test_invalid_source_shapes_and_values_are_rejected(self):
        for points, colors in (([[0, 0]], [[0, 0, 0]]), ([[0, 0, 0]], [[0, 0]]),
                ([[float("nan"), 0, 0]], [[0, 0, 0]]), ([[0, 0, 0]], [[0, float("inf"), 0]]),
                ([[0, 0, 0]], [[0, -.1, 0]]), ([[0, 0, 0]], [[0, 1.1, 0]]),
                ([[0, 0, 0]], [[False, False, False]])):
            with self.subTest(points=points, colors=colors), self.assertRaises(ValueError):
                self.colors(points, colors)

    def runtime(self):
        state = {"cell_indices": np.array([0], np.uint32),
                 "centers": np.array([.2, .3, .4], np.float32),
                 "log_scales": np.zeros(3, np.float32), "rotations": np.array([1, 0, 0, 0], np.float32),
                 "radiance_sh": np.arange(27, dtype=np.float32), "opacity_sh": np.array([.3], np.float32)}
        def make_model(resolution, lo, hi, snapshot, sh, opacity_sh, **options):
            result = SimpleNamespace(resolution=resolution, core_resolution=resolution, grid_min=lo,
                grid_max=hi, active_voxel_count=len(snapshot["cell_indices"]), sh_count=sh,
                opacity_sh_count=opacity_sh, snapshot=lambda: snapshot, **options)
            result.native_background_shell_levels = lambda: getattr(result, "background_shell_levels", 0)
            result.native_background_subdivision_level = lambda: getattr(result, "background_subdivision_level", 0)
            result.native_background_use_ellipsoid = lambda: getattr(result, "background_use_ellipsoid", True)
            result.native_background_adaptive = lambda: getattr(result, "background_adaptive", False)
            result.set_world_shells = lambda *args: None
            result.set_background_use_ellipsoid = lambda value: setattr(result, "background_use_ellipsoid", value)
            result.cell_index_bits = 64
            result.world_shell_version = 5
            return result
        def from_state(resolution, lo, hi, snapshot, sh, opacity_sh, **options):
            options.pop("core_resolution", None)
            return make_model(resolution, lo, hi, snapshot, sh, opacity_sh, **options)
        native = SimpleNamespace(from_state=from_state, set_world_shells=lambda *args: None,
            native_background_shell_levels=lambda: 0, native_background_subdivision_level=lambda: 0,
            native_background_use_ellipsoid=lambda: True, set_background_use_ellipsoid=lambda value: None,
            world_shell_version=5, cell_index_bits=64)
        mi = SimpleNamespace(VoxelReconstruction=native, UInt32=lambda value: np.asarray(value, np.uint32),
                             UInt64=lambda value: np.asarray(value, np.uint64),
                             Float=lambda value: np.asarray(value, np.float32))
        mi.fit_calls = []
        def from_points(positions, colors, resolution, lo, hi, sh, opacity_sh):
            # CPU stand-in for native fitting. Deliberately nondefault geometry
            # verifies that the initializer forwards the fit without replacing it.
            points, rgb = positions.reshape(-1, 3), colors.reshape(-1, 3)
            edge = (np.asarray(hi, np.float32) - np.asarray(lo, np.float32)) / np.float32(resolution)
            coordinate = (points.astype(np.float64) - np.asarray(lo, np.float32)) / edge
            xyz = np.floor(coordinate).astype(np.uint64)
            flat = xyz[:, 0] + resolution * xyz[:, 1] + resolution ** 2 * xyz[:, 2]
            cells, inverse, counts = np.unique(flat, return_inverse=True, return_counts=True)
            n = len(cells)
            centers = np.stack([np.bincount(inverse, weights=(coordinate - xyz)[:, axis]) / counts
                                for axis in range(3)], axis=1).clip(.01, .99).astype(np.float32)
            radiance = np.zeros((n, sh, 3), np.float32)
            linear = srgb_to_linear(rgb, np)
            for axis in range(3):
                radiance[:, 0, axis] = np.bincount(inverse, weights=linear[:, axis]) / counts / .28209479177387814
            fitted = dict(cell_indices=cells, centers=centers.ravel(),
                log_scales=np.tile(np.log(edge * np.array([.35, .45, .55])), (n, 1)).astype(np.float32).ravel(),
                rotations=np.tile(np.array([.8, .6, 0., 0.], np.float32), n),
                radiance_sh=radiance.ravel(), opacity_sh=np.zeros(n * opacity_sh, np.float32))
            mi.fit_calls.append((resolution, lo, hi, fitted))
            return make_model(resolution, lo, hi, fitted, sh, opacity_sh)
        native.from_points = from_points
        dr = SimpleNamespace(concat=np.concatenate, eval=lambda *args: None, any=np.any)
        return make_model(4, [-1.] * 3, [1.] * 3, state, 9, 1), mi, dr, state

    def test_initialization_colors_only_dc_and_keeps_all_leaves_and_fixed_opacity(self):
        seed, mi, dr, state = self.runtime()
        colors = np.zeros((112, 3), np.float32)
        colors[:2] = [[.25, .5, .75], [.8, .2, .1]]
        counts = np.zeros(112, np.int64)
        counts[:2] = [1, 100]
        model = initialize_background_model(seed, mi, dr, np, opacity=.3, shell_levels=2,
            background_colors=colors, background_point_counts=counts)
        actual = model.snapshot()
        self.assertEqual(model.active_voxel_count, 113)
        np.testing.assert_array_equal(actual["cell_indices"][1:], np.arange(64, 176, dtype=np.uint32))
        for name, values in state.items():
            np.testing.assert_array_equal(actual[name][:len(values)], values)
        radiance = actual["radiance_sh"].reshape(-1, 9, 3)[1:]
        np.testing.assert_allclose(radiance[:, 0] * .28209479177387814, colors, rtol=1e-7)
        np.testing.assert_array_equal(radiance[:, 1:], 0.)
        np.testing.assert_array_equal(actual["opacity_sh"][1:], np.float32(math.log(.3 / .7) / .28209479177387814))
        self.assertEqual(model.background_geometry, "ellipsoid")
        self.assertTrue(model.native_background_use_ellipsoid())
        lower, upper = leaf_boxes_numpy(actual["cell_indices"][1:], 4, [-1.] * 3, [1.] * 3, 2, np)
        scales = np.exp(actual["log_scales"].reshape(-1, 3)[1:].astype(np.float64))
        np.testing.assert_allclose(scales, (upper - lower) * (np.sqrt(3.) * .5), rtol=2e-6)
        # Initial ellipsoids cover every leaf corner, preventing initialization
        # holes between neighboring background cells of different sizes.
        corner_offset = (upper - lower) * .5
        self.assertTrue(np.all(np.sum((corner_offset / scales) ** 2, axis=1) <= 1. + 1e-6))
        report = model.background_initialization
        self.assertEqual(report["background_geometry"], "ellipsoid")
        self.assertEqual(report["color"], "point_cloud_mean_linear_rgb")
        self.assertEqual(report["background_colored_voxels"], 2)
        self.assertEqual(report["background_color_source_points"], 101)
        self.assertFalse(any(isinstance(value, np.ndarray) for value in report.values()))
        black = initialize_background_model(seed, mi, dr, np, shell_levels=2)
        self.assertEqual(black.active_voxel_count, 113)
        np.testing.assert_array_equal(black.snapshot()["radiance_sh"][27:], 0.)
        self.assertEqual(black.background_initialization["color"], "black")

    def test_initialization_rejects_incomplete_or_invalid_aggregates(self):
        seed, mi, dr, _ = self.runtime()
        colors, counts = np.zeros((56, 3), np.float32), np.zeros(56, np.int64)
        for rgb, n in ((colors, None), (None, counts), (colors[:55], counts),
                (colors, counts[:55]), (colors, counts.astype(float)), (colors, counts - 1),
                (colors + np.nan, counts), (colors + 2, counts), (colors + .1, counts)):
            with self.subTest(rgb_shape=getattr(rgb, "shape", None), n_shape=getattr(n, "shape", None)):
                with self.assertRaises(ValueError):
                    initialize_background_model(seed, mi, dr, np, shell_levels=1,
                        background_colors=rgb, background_point_counts=n)

    def test_explicit_legacy_occupied_cell_initialization_remains_available(self):
        seed, mi, dr, _ = self.runtime()
        model = initialize_background_model(seed, mi, dr, np, shell_levels=2,
                                             background_geometry="occupied_cell")
        self.assertEqual(model.background_geometry, "occupied_cell")
        self.assertFalse(model.native_background_use_ellipsoid())
        self.assertEqual(model.background_initialization["background_geometry"], "occupied_cell")
        state = model.snapshot()
        lower, upper = leaf_boxes_numpy(state["cell_indices"][1:], 4, [-1.] * 3, [1.] * 3, 2, np)
        np.testing.assert_allclose(np.exp(state["log_scales"].reshape(-1, 3)[1:]),
                                   (upper - lower) * .5, rtol=2e-6)


class SparseBackgroundTests(unittest.TestCase):
    def prepare(self, points, colors=None, core=256, subdivision=6, levels=5, lo=None, hi=None):
        points = np.asarray(points, np.float32).reshape(-1, 3)
        colors = np.full_like(points, .5) if colors is None else np.asarray(colors, np.float32)
        return prepare_sparse_background(points, colors, core, [-1.] * 3 if lo is None else lo,
                                         [1.] * 3 if hi is None else hi, levels, np, subdivision)

    def test_sparse_s8_addresses_exceed_uint32_without_dense_allocation(self):
        points = [[0., 0., 0.], [1.25, .2, .3], [-2., 0., 0.], [31.98, 31.98, 31.98],
                  [31.981, 31.981, 31.981], [32., 0., 0.]]
        with patch("reconstruction_background.background_cells_numpy", side_effect=AssertionError("Dense shell allocation")):
            prepared = self.prepare(points, core=1024, subdivision=8)
        self.assertEqual(prepared["cell_indices"].dtype, np.dtype(np.uint64))
        self.assertGreater(int(prepared["cell_indices"].max()), 2 ** 32)
        self.assertEqual(len(prepared["cell_indices"]), 3)
        self.assertEqual(prepared["point_counts"].tolist(), [1, 1, 2])
        self.assertEqual(prepared["excluded_core_points"], 1)
        self.assertEqual(prepared["outside_root_points"], 1)
        lower, upper = leaf_boxes_numpy(prepared["point_cell_indices"], 1024, [-1.] * 3,
                                        [1.] * 3, 5, np, 8)
        self.assertTrue(np.all(prepared["positions"] >= lower))
        self.assertTrue(np.all(prepared["positions"] < upper))
        self.assertLess(sum(value.nbytes for value in prepared.values() if isinstance(value, np.ndarray)), 1024)

    def test_per_leaf_color_decodes_before_averaging(self):
        rgb = np.asarray([[.01, .3, .9], [.5, .7, .2], [1., 0., 0.]], np.float32)
        prepared = self.prepare([[1.2, .2, .3], [1.201, .201, .301], [0, 0, 0]], rgb)
        self.assertEqual(len(prepared["cell_indices"]), 1)
        self.assertEqual(prepared["point_counts"].tolist(), [2])
        np.testing.assert_array_equal(prepared["mean_linear_colors"][0],
            srgb_to_linear(rgb[:2], np).astype(np.float64).mean(axis=0).astype(np.float32))

    def test_half_open_faces_and_translated_cube(self):
        values = np.asarray([-4., -2., -1., 1., 2., 4.,
            np.nextafter(np.float32(-1.), np.float32(-np.inf)),
            np.nextafter(np.float32(1.), np.float32(-np.inf)),
            np.nextafter(np.float32(2.), np.float32(-np.inf))], np.float32)
        points = np.stack((values, np.zeros_like(values), np.zeros_like(values)), axis=1)
        prepared = self.prepare(points, levels=2)
        self.assertEqual(prepared["excluded_core_points"], 2)
        self.assertEqual(prepared["outside_root_points"], 1)
        for point, cell, shell in zip(prepared["positions"], prepared["point_cell_indices"], prepared["shell_indices"]):
            lower, upper = leaf_boxes_numpy(np.array([cell], np.uint64), 256, [-1.] * 3, [1.] * 3, 2, np, 6)
            self.assertTrue(np.all((point >= lower[0]) & (point < upper[0])))
            self.assertEqual(int((cell - np.uint64(256 ** 3)) // np.uint64(56 * 8 ** 6)), int(shell))
        translated = self.prepare([[8.3, 6.2, 7.4], [5., 6., 7.], [-58., 6., 7.]],
                                  lo=[3., 4., 5.], hi=[7., 8., 9.])
        lower, upper = leaf_boxes_numpy(translated["point_cell_indices"], 256, [3., 4., 5.],
                                        [7., 8., 9.], 5, np, 6)
        self.assertTrue(np.all((translated["positions"] >= lower) & (translated["positions"] < upper)))

    def test_empty_background_has_no_allocated_leaves(self):
        prepared = self.prepare([[0, 0, 0], [32, 0, 0]])
        self.assertEqual(prepared["positions"].shape, (0, 3))
        self.assertEqual(prepared["cell_indices"].shape, (0,))
        self.assertEqual(prepared["mean_linear_colors"].shape, (0, 3))

    def test_initialization_preserves_foreground_and_native_geometry(self):
        seed, mi, dr, foreground = BackgroundPointColorTests().runtime()
        seed.resolution = seed.core_resolution = 1024
        points = self.prepare([[1.201, .207, .303], [1.202, .208, .304], [-8.21, 3.41, .19],
                               [31.98, 31.98, 31.98]], core=1024, subdivision=8)
        with patch("reconstruction_background.background_cells_numpy", side_effect=AssertionError("Dense shell allocation")):
            model = initialize_background_model(seed, mi, dr, np, opacity=.3, shell_levels=5,
                background_points=points, background_subdivision_level=8)
        actual = model.snapshot()
        self.assertEqual(actual["cell_indices"].dtype, np.dtype(np.uint64))
        np.testing.assert_array_equal(np.sort(actual["cell_indices"][1:]), points["cell_indices"])
        for name, values in foreground.items():
            np.testing.assert_array_equal(actual[name][:len(values)], values)
        for name in ("centers", "log_scales", "rotations", "radiance_sh"):
            np.testing.assert_array_equal(actual[name][len(foreground[name]):],
                                          np.concatenate([call[3][name] for call in mi.fit_calls]))
        np.testing.assert_array_equal(actual["opacity_sh"][1:],
                                      np.float32(math.log(.3/.7) / .28209479177387814))
        self.assertTrue(all(call[0] == 1024 for call in mi.fit_calls))
        self.assertEqual(model.background_subdivision_level, 8)
        self.assertEqual(model.background_initialization["initialization"], "point_cloud_sparse")
        self.assertEqual(model.background_initialization["background_voxels"], len(points["cell_indices"]))
        self.assertEqual(model.background_initialization["background_color_source_points"], 4)

    def test_empty_sparse_initialization_keeps_only_foreground(self):
        seed, mi, dr, foreground = BackgroundPointColorTests().runtime()
        seed.resolution = seed.core_resolution = 256
        points = self.prepare([[0, 0, 0]])
        result = initialize_background_model(seed, mi, dr, np, background_points=points,
                                            background_subdivision_level=6)
        for name, values in foreground.items():
            np.testing.assert_array_equal(result.snapshot()[name], values)
        self.assertEqual(mi.fit_calls, [])
        self.assertEqual(result.background_initialization["background_voxels"], 0)

    def test_native_occupancy_mismatch_is_reported(self):
        seed, mi, dr, _ = BackgroundPointColorTests().runtime()
        seed.resolution = seed.core_resolution = 256
        points = self.prepare([[1.2, .2, .3]])
        points["cell_indices"][0] += np.uint64(1)
        with self.assertRaisesRegex(RuntimeError, "CPU preflight"):
            initialize_background_model(seed, mi, dr, np, background_points=points,
                                        background_subdivision_level=6)

    def test_sparse_initialization_requires_matching_point_data(self):
        seed, mi, dr, _ = BackgroundPointColorTests().runtime()
        seed.resolution = seed.core_resolution = 256
        with self.assertRaisesRegex(ValueError, "sparse background_points"):
            initialize_background_model(seed, mi, dr, np, background_subdivision_level=6)
        points = self.prepare([[1.2, .2, .3]])
        with self.assertRaisesRegex(ValueError, "disagrees"):
            initialize_background_model(seed, mi, dr, np, background_points=points,
                                        background_subdivision_level=5)


class SparseBackgroundDatasetTests(unittest.TestCase):
    def setUp(self):
        from test_reconstruction_dataset import write_fixture
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.scene = Path(self.directory.name) / "bonsai"
        write_fixture(self.scene)

    def config(self, resolution=128):
        from train import TrainConfig
        config = TrainConfig()
        config.data.dataset = str(self.scene)
        config.data.scene_mode = "unbounded"
        config.data.background_initial_resolution = resolution
        return config

    def test_data_config_routes_sparse_points_and_actual_occupied_count(self):
        from reconstruction_common import prepare_dataset
        config = self.config()
        prepared = prepare_dataset(config, np)
        points, initial = prepared["background_points"], prepared["initialization"]
        self.assertEqual(initial["background_initial_resolution"], 128)
        self.assertEqual(initial["background_subdivision_level"], 6)
        self.assertEqual(initial["background_initialization"], "point_cloud_sparse")
        self.assertEqual(initial["background_voxels"], 2)
        self.assertEqual(len(points["cell_indices"]), initial["background_voxels"])
        self.assertEqual(points["point_counts"].sum(), 2)
        self.assertIsNone(prepared["background_colors"])
        self.assertIsNone(prepared["background_point_counts"])
        self.assertEqual(initial["background_color_initialization"]["empty_leaf_rule"], "unallocated")
        coarse = prepare_dataset(self.config(2), np)
        np.testing.assert_array_equal(prepared["positions"], coarse["positions"])
        np.testing.assert_array_equal(prepared["colors"], coarse["colors"])
        self.assertEqual(prepared["views"], coarse["views"])
        self.assertIsNone(coarse["background_points"])
        self.assertEqual(coarse["background_colors"].shape, (280, 3))

    def test_sparse_camera_only_resume_preserves_metadata_without_point_reload(self):
        from reconstruction_common import prepare_dataset
        config = self.config()
        prepared = prepare_dataset(config, np)
        saved = copy.deepcopy(prepared["initialization"])
        with patch("reconstruction_dataset.read_colmap_points", side_effect=AssertionError("Point reload")):
            resumed = prepare_dataset(config, np, load_points=False)
        self.assertEqual(resumed["initialization"], saved)
        self.assertIsNone(resumed["background_points"])
        self.assertEqual(saved["background_voxels"], 2)
        config.data.background_initial_resolution = 2
        with self.assertRaisesRegex(ValueError, "initial background resolution|disagrees"):
            prepare_dataset(config, np, load_points=False)


class BackgroundConfigTests(unittest.TestCase):
    def test_auto_and_explicit_modes_select_residual_color(self):
        for dataset, requested, expected in (
                ("synthetic", "auto", "bounded"), ("colmap", "auto", "unbounded"),
                ("colmap", "bounded", "bounded"), ("synthetic", "unbounded", "unbounded")):
            with self.subTest(dataset=dataset, requested=requested):
                config = preview.PreviewConfig()
                config.data.scene_mode = requested
                self.assertEqual(preview.resolve_scene_background(config, dataset), expected)
                color = (1., 1., 1.) if expected == "bounded" else (0., 0., 0.)
                self.assertEqual(config.render.background, color)
                self.assertEqual(config.data.scene_mode, requested)
                preview.validate_config(config)

    def test_legacy_initialization_is_explicitly_rejected(self):
        config = preview.PreviewConfig()
        config.data.initialization = {"version": 1, "kind": "colmap"}
        config.render.background = (.1, .2, .3)
        for mode in ("auto", "bounded", "unbounded"):
            config.data.scene_mode = mode
            with self.assertRaisesRegex(ValueError, "Legacy compressed"):
                preview.resolve_scene_background(config, "colmap")
        self.assertEqual(config.render.background, (.1, .2, .3))

    def test_saved_scene_mode_is_authoritative_and_conflicts_rejected(self):
        config = preview.PreviewConfig()
        config.data.initialization = {"version": 2, "scene_mode": "bounded", "scene_mapping": "bounded"}
        self.assertEqual(preview.resolve_scene_background(config, "colmap"), "bounded")
        config.data.scene_mode = "unbounded"
        with self.assertRaisesRegex(ValueError, "disagrees"):
            preview.resolve_scene_background(config, "colmap")
        config.data.scene_mode = "auto"
        config.data.initialization = {"version": 7, "scene_mapping": "bounded"}
        with self.assertRaisesRegex(ValueError, "scene_mode"):
            preview.resolve_scene_background(config, "colmap")

    def test_invalid_background_settings_fail(self):
        for name, values in {
                "scene_mode": [None, "outside"],
                "background_shell_levels": [0, -1, True, 1.5, 11],
                "background_opacity": [0., 1., -1., True, float("inf"), float("nan")],
        }.items():
            for value in values:
                with self.subTest(name=name, value=value):
                    config = preview.PreviewConfig()
                    setattr(config.data, name, value)
                    with self.assertRaisesRegex(ValueError, name):
                        preview.validate_config(config)

    def test_shell_layers_independent_of_hit_capacity(self):
        config = preview.PreviewConfig()
        config.render.max_hits = 1
        self.assertEqual(config.data.background_shell_levels, 5)
        preview.validate_config(config)
        config.render.max_hits = 32
        self.assertEqual(config.data.background_shell_levels, 5)
        preview.validate_config(config)

    def test_old_resume_configuration_gets_equivalent_defaults(self):
        import train
        old = {"dataset": "datasets/lego", "reference_compositing": "linear"}
        new = {**old, "scene_mode": "auto", "background_shell_levels": 5, "background_opacity": .5}
        self.assertEqual(train.normalized_algorithm_group("data", old),
                         train.normalized_algorithm_group("data", new))
        changed = {**new, "background_opacity": .25}
        self.assertNotEqual(train.normalized_algorithm_group("data", old),
                            train.normalized_algorithm_group("data", changed))

    def test_training_cli_keeps_requested_initialization_settings(self):
        import train
        with patch.object(sys, "argv", ["train.py", "--scene-mode", "unbounded",
                         "--background-shell-levels", "5", "--background-opacity", "0.5"]):
            args = train.parse_arguments()
        config = train.TrainConfig()
        train.apply_arguments(config, args, asdict(train.CONFIG))
        self.assertEqual(config.data.scene_mode, "unbounded")
        self.assertEqual(config.data.background_shell_levels, 5)
        self.assertEqual(config.data.background_opacity, .5)

    def test_saved_shell_protocol_and_evaluation(self):
        import evaluate
        import train
        for target in (512, 1024):
            with self.subTest(target=target):
                saved = {"data": {"scene_mode": "unbounded", "background_shell_levels": 5,
                         "background_opacity": .5,
                         "initialization": {"version": 7, "scene_mode": "unbounded", "scene_mapping": "world_shell",
                             "background_refinement": "fixed", "background_shell_levels": 5,
                             "background_opacity": .5, "start_core_resolution": 256,
                             "target_core_resolution": target}}}
                data = evaluate.checkpoint_data_config(saved)
                report = preview.scene_background_protocol(data, preview.RenderConfig())
                self.assertEqual(report["background_shell_levels"], 5)
                self.assertEqual(report["background_initial_voxels"], 280)
                self.assertEqual(report["background_refinement"], "fixed")
                normalized = train.normalized_algorithm_group("data", asdict(data))
                self.assertEqual(normalized["background_shell_levels"], 5)
                self.assertNotIn("background_layers", normalized)

    def test_target_resolution_cli_defaults_and_explicit_level_weights(self):
        import train
        for options, expected in (([], (6, 4)),
                (["--target-resolution", "1024"], (4, 3, 3)),
                (["--target-resolution", "1024", "--level-weights", "5", "3", "2"], (5, 3, 2))):
            with self.subTest(options=options), patch.object(sys, "argv", ["train.py", *options]):
                args = train.parse_arguments()
                config = copy.deepcopy(train.CONFIG)
                train.apply_arguments(config, args, {})
                self.assertEqual(tuple(config.schedule.level_weights), expected)
                train.validate(config)
        with patch.object(sys, "argv", ["train.py", "--target-resolution", "1024"]):
            args = train.parse_arguments()
        config = copy.deepcopy(train.CONFIG)
        train.apply_arguments(config, args, {"schedule": {"level_weights": [6, 4]}})
        with self.assertRaises(ValueError):
            train.validate(config)

    def test_resume_does_not_replace_saved_schedule_weights(self):
        import train
        with patch.object(sys, "argv", ["train.py", "--resume", "model.npz"]):
            args = train.parse_arguments()
        config = copy.deepcopy(train.CONFIG)
        config.schedule.target_resolution = 1024
        config.schedule.level_weights = (6, 4)
        train.apply_arguments(config, args, {})
        self.assertEqual(config.schedule.level_weights, (6, 4))

    def test_evaluation_keeps_saved_scene_configuration(self):
        import evaluate
        saved = {"data": {"scene_mode": "unbounded", "background_shell_levels": 8,
                         "background_opacity": .5,
                         "initialization": {"version": 7, "scene_mode": "unbounded", "scene_mapping": "world_shell",
                                            "background_shell_levels": 8, "background_opacity": .5}}}
        data = evaluate.checkpoint_data_config(saved)
        render = preview.RenderConfig(background=(0., 0., 0.))
        report = preview.scene_background_protocol(data, render)
        self.assertEqual(report["scene_mode"], "unbounded")
        self.assertEqual(report["background_linear_rgb"], [0., 0., 0.])
        self.assertEqual(report["background_shell_levels"], 8)
        self.assertEqual(report["initial_background_opacity"], .5)
        self.assertIn("no environment illumination", report["background_role"])


def shell_fixture(mi, np):
    """Two interior seeds with nondefault geometry and directional radiance."""
    y00 = 1. / (2. * math.sqrt(math.pi))
    colors = np.zeros((2, 4, 3), dtype=np.float32)
    colors[:, 0] = np.asarray([[.3, .6, .8], [.7, .4, .2]]) / y00
    colors[:, 1:] = np.asarray([[[.01, .02, .03]] * 3, [[.03, .02, .01]] * 3])
    state = {
        "cell_indices": np.asarray([0, 63], dtype=np.uint32),
        "centers": np.asarray([[.55, .45, .65], [.6, .4, .7]], dtype=np.float32).ravel(),
        "log_scales": np.log(np.asarray([[.31, .32, .33], [.3, .25, .4]], dtype=np.float32)).ravel(),
        "rotations": np.asarray([[1., 0., 0., 0.]] * 2, dtype=np.float32).ravel(),
        "radiance_sh": colors.ravel(),
        "opacity_sh": np.asarray([math.log(.23/.77)/y00, math.log(.8/.2)/y00], dtype=np.float32),
    }
    device = {name: mi.UInt64(value) if name == "cell_indices" else mi.Float(value)
              for name, value in state.items()}
    model = mi.VoxelReconstruction.from_state(4, [-1.] * 3, [1.] * 3, device, 4, 1, core_resolution=4)
    return model, state


def native_background_checks(mi, dr, np):
    from reconstruction_background import initialize_background_model
    from reconstruction_common import model_metadata, restore_model
    from reconstruction_world_shell import leaf_boxes_numpy
    from test_reconstruction import native_trace

    seed, seed_state = shell_fixture(mi, np)
    model = initialize_background_model(seed, mi, dr, np, opacity=.5, shell_levels=5)
    metadata = model_metadata(model)
    assert metadata["scene_mapping"] == "world_shell"
    assert metadata["background_shell_levels"] == 5
    assert metadata["background_geometry"] == "ellipsoid"
    assert metadata["world_shell_version"] == 7
    assert model.native_background_use_ellipsoid()
    state = {name: np.asarray(value).copy() for name, value in model.snapshot().items()}
    indices = state["cell_indices"]
    np.testing.assert_array_equal(indices, np.r_[seed_state["cell_indices"], np.arange(64, 344, dtype=np.uint32)])
    assert len(np.unique(indices)) == len(indices)
    for name, width in (("centers", 3), ("log_scales", 3), ("rotations", 4),
                        ("radiance_sh", 12), ("opacity_sh", 1)):
        np.testing.assert_array_equal(state[name][:2 * width], seed_state[name], err_msg=f"core {name}")
    is_far = indices >= 64
    np.testing.assert_array_equal(state["centers"].reshape(-1, 3)[is_far], .5)
    np.testing.assert_array_equal(state["opacity_sh"][is_far], 0.)
    np.testing.assert_array_equal(state["radiance_sh"].reshape(-1, 12)[is_far], 0.)
    lower, upper = leaf_boxes_numpy(indices[is_far], 4, [-1.] * 3, [1.] * 3, 5, np)
    np.testing.assert_allclose(np.exp(state["log_scales"].reshape(-1, 3)[is_far]),
                              (upper - lower) * (np.sqrt(3.) * .5), rtol=3e-6)
    restored = restore_model({"model_metadata": metadata, "model_state": state}, mi)
    assert restored.scene_mapping == "world_shell"
    assert restored.native_background_use_ellipsoid()
    origins = np.asarray([[.125, .125, .125], [-.6, .125, .125], [.13, -.11, .07], [.2, -.9, .1]], dtype=np.float32)
    directions = np.asarray([[1., 0., 0.], [1., 0., 0.], [.7, .3, -.2], [.1, .9, .2]], dtype=np.float32)
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    maxt = np.full(len(origins), 128., dtype=np.float32)
    actual = native_trace(np, mi, dr, model, origins, directions, maxt, np.zeros(3), .00001, 32)
    compared = native_trace(np, mi, dr, restored, origins, directions, maxt, np.zeros(3), .00001, 32)
    for a, b in zip(actual, compared):
        np.testing.assert_array_equal(a, b)
    assert np.all(actual[2] > 0)
    empty = {name: value[:0] for name, value in state.items()}
    empty_model = restore_model({"model_metadata": metadata, "model_state": empty}, mi)
    for background, expected in ((np.zeros(3), 0.), (np.ones(3), 1.)):
        rgb, alpha, hits = native_trace(np, mi, dr, empty_model, origins, directions, maxt, background, .00001, 32)
        np.testing.assert_array_equal(rgb, expected)
        np.testing.assert_array_equal(alpha, 0.)
        np.testing.assert_array_equal(hits, 0)
    point_rgb = np.asarray([[.8, .3, .1], [.2, .7, .4]], dtype=np.float32)
    means, counts = background_point_colors(np.asarray([[1.25, .25, .25], [1.75, .25, .25]]),
        point_rgb, [-1.] * 3, [1.] * 3, 5, np)
    colored = initialize_background_model(seed, mi, dr, np, opacity=.5, shell_levels=5,
        background_colors=means, background_point_counts=counts)
    colored_state = {name: np.asarray(value).copy() for name, value in colored.snapshot().items()}
    for name in ("cell_indices", "centers", "log_scales", "rotations", "opacity_sh"):
        np.testing.assert_array_equal(colored_state[name], state[name])
    radiance = colored_state["radiance_sh"].reshape(-1, 4, 3)
    np.testing.assert_array_equal(radiance[:2].ravel(), seed_state["radiance_sh"])
    np.testing.assert_allclose(radiance[2:, 0] * .28209479177387814, means, rtol=1e-7)
    np.testing.assert_array_equal(radiance[2:, 1:], 0.)
    rgb, alpha, hits = native_trace(np, mi, dr, colored,
        np.asarray([[1.25, .25, .25]], np.float32), np.asarray([[1., 0., 0.]], np.float32),
        np.asarray([.5], np.float32), np.zeros(3), .00001, 32)
    expected = srgb_to_linear(point_rgb, np).astype(np.float64).mean(axis=0) * .5
    np.testing.assert_allclose(rgb[0], expected, rtol=2e-6, atol=1e-7)
    np.testing.assert_allclose(alpha, .5, rtol=2e-6)
    np.testing.assert_array_equal(hits, 1)
    assert colored.background_initialization["background_color_source_points"] == 2
    assert colored.background_initialization["background_colored_voxels"] == 1
    # Sparse S=8 contains only a few leaves, including an address beyond 32 bits.
    # This exercises the actual native point fit and compares its occupancy with
    # CPU preflight, without ever allocating the theoretical full shell table.
    core = mi.VoxelReconstruction.from_state(1024, [-1.] * 3, [1.] * 3,
        {name: mi.UInt64(value) if name == "cell_indices" else mi.Float(value)
         for name, value in seed_state.items()}, 4, 1, core_resolution=1024)
    sparse_positions = np.asarray([[1.201, .207, .303], [-8.21, 3.41, .19],
                                   [31.98, 31.98, 31.98], [31.981, 31.981, 31.981]], np.float32)
    sparse_colors = np.asarray([[.8, .2, .1], [.3, .4, .7], [.2, .6, .8], [.7, .2, .4]], np.float32)
    points = prepare_sparse_background(sparse_positions, sparse_colors, 1024,
                                       [-1.] * 3, [1.] * 3, 5, np, 8)
    sparse = initialize_background_model(core, mi, dr, np, opacity=.3, shell_levels=5,
                                        background_points=points, background_subdivision_level=8)
    sparse_state = {name: np.asarray(value).copy() for name, value in sparse.snapshot().items()}
    ids = sparse_state["cell_indices"][2:]
    assert ids.dtype == np.dtype(np.uint64) and int(ids.max()) > 2 ** 32
    np.testing.assert_array_equal(np.sort(ids), points["cell_indices"])
    for name, original in seed_state.items():
        np.testing.assert_array_equal(sparse_state[name][:len(original)], original)
    expected_rows = np.searchsorted(points["cell_indices"], ids)
    np.testing.assert_allclose(sparse_state["radiance_sh"].reshape(-1, 4, 3)[2:, 0] * .28209479177387814,
                               points["mean_linear_colors"][expected_rows], rtol=3e-6, atol=1e-7)
    sparse_meta = model_metadata(sparse)
    assert sparse_meta["background_subdivision_level"] == 8
    assert sparse.cell_index_bits == 64 and sparse_state["cell_indices"].dtype.itemsize == 8
    sparse_restored = restore_model({"model_metadata": sparse_meta, "model_state": sparse_state}, mi)
    low, high = leaf_boxes_numpy(points["point_cell_indices"], 1024, [-1.] * 3, [1.] * 3, 5, np, 8)
    edge = high[:, 0] - low[:, 0]
    origins = points["positions"].copy()
    origins[:, 0] -= (.01 * edge).astype(np.float32)
    directions = np.tile(np.asarray([1., 0., 0.], np.float32), (len(origins), 1))
    maxt = (.02 * edge).astype(np.float32)
    actual = native_trace(np, mi, dr, sparse, origins, directions, maxt, np.zeros(3), .00001, 32)
    restored_actual = native_trace(np, mi, dr, sparse_restored, origins, directions, maxt, np.zeros(3), .00001, 32)
    for value, restored_value in zip(actual, restored_actual):
        np.testing.assert_array_equal(value, restored_value)
    rows = np.searchsorted(points["cell_indices"], points["point_cell_indices"])
    np.testing.assert_allclose(actual[0], .3 * points["mean_linear_colors"][rows], rtol=4e-6, atol=1e-7)
    np.testing.assert_allclose(actual[1], .3, rtol=3e-6)
    np.testing.assert_array_equal(actual[2], 1)
    return {"complete_world_shell_cells": 280, "shell_opacity": .5,
            "core_preserved": True, "saved_mapping_restored": True, "empty_scene_residual": True,
            "point_color_forward": True, "color_preserves_shell_occupancy_and_opacity": True,
            "sparse_point_fitted_leaves": len(ids), "sparse_s8_uint64_addresses": True,
            "sparse_preflight_matches_native": True, "sparse_rgb_alpha_and_restore": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--build-dir")
    parser.add_argument("--gpu-index", type=int, default=0)
    args = parser.parse_args()
    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__]))
    if not result.wasSuccessful():
        return 1
    if args.gpu:
        from reconstruction_common import initialize_runtime
        runtime = preview.RuntimeConfig(gpu=args.gpu_index)
        if args.build_dir:
            runtime.build_dir = args.build_dir
        with tempfile.TemporaryDirectory(prefix="mitsuba-background-") as temporary:
            mi, dr, np, handles = initialize_runtime(runtime, Path(temporary), {})
            native = native_background_checks(mi, dr, np)
            print(json.dumps({"status": "passed", "cuda_checks": native}))
    print(json.dumps({"status": "passed", "cpu_tests": result.testsRun}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
