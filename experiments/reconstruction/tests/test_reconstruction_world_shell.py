"""CPU regressions for world-space shell subdivision and portable metadata."""

from __future__ import annotations

import copy
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from preview import PreviewConfig, update_config, validate_config, scene_background_protocol
from reconstruction_checkpoint import load_checkpoint, save_checkpoint, validate_model_state
from reconstruction_dataset import prepare_colmap
from reconstruction_world_shell import (
    background_cells_numpy, configure_world_shell_model, leaf_boxes_numpy, shell_coordinates_numpy,
    validate_shell_levels, validate_world_shell_layout, world_shell_ray_maxt,
    world_shell_metadata_geometry,
)
from test_reconstruction_dataset import write_fixture


def small_model(levels=5, resolution=4, subdivision=0, refinement="fixed"):
    cells = np.r_[np.uint32(0), background_cells_numpy(resolution, levels, np, subdivision)]
    n = len(cells)
    state = {"cell_indices": cells, "centers": np.full(n * 3, .5, dtype=np.float32),
             "log_scales": np.zeros(n * 3, dtype=np.float32),
             "rotations": np.tile(np.array([1., 0., 0., 0.], dtype=np.float32), n),
             "radiance_sh": np.zeros(n * 27, dtype=np.float32),
             "opacity_sh": np.zeros(n * 9, dtype=np.float32)}
    metadata = dict(resolution=resolution, core_resolution=resolution,
                    grid_min=[-1.] * 3, grid_max=[1.] * 3, sh_count=9, opacity_sh_count=9,
                    scene_mapping="world_shell", background_shell_levels=levels,
                    background_refinement=refinement, background_subdivision_level=subdivision,
                    world_shell_version=3, background_geometry="ellipsoid")
    return state, metadata


class WorldShellLayoutTests(unittest.TestCase):
    def test_world_volume_partition_and_order(self):
        coordinates = shell_coordinates_numpy(np)
        self.assertEqual(coordinates.shape, (56, 3))
        np.testing.assert_array_equal(coordinates[:5], [[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0], [0, 1, 0]])
        self.assertFalse(np.any(np.all((coordinates >= 1) & (coordinates <= 2), axis=1)))
        for levels in (1, 5, 10):
            cells = background_cells_numpy(256, levels, np)
            lower, upper = leaf_boxes_numpy(cells, 256, [-1.] * 3, [1.] * 3, levels, np)
            np.testing.assert_array_equal(lower.min(axis=0), [-2 ** levels] * 3)
            np.testing.assert_array_equal(upper.max(axis=0), [2 ** levels] * 3)
            self.assertEqual(float(np.prod(upper - lower, axis=1).sum()) + 8., float((2 ** (levels + 1)) ** 3))
            # Interiors may never overlap, including adjacent shell levels.
            overlap = np.minimum(upper[:, None, :], upper[None, :, :]) - np.maximum(lower[:, None, :], lower[None, :, :])
            intersects = np.all(overlap > 0, axis=-1)
            np.fill_diagonal(intersects, False)
            self.assertFalse(intersects.any())

    def test_background_boxes_do_not_move_when_foreground_refines(self):
        expected = None
        for resolution in (256, 512, 1024):
            cells = background_cells_numpy(resolution, 5, np)
            self.assertEqual(len(cells), 280)
            boxes = leaf_boxes_numpy(cells, resolution, [3., 4., 5.], [7., 8., 9.], 5, np)
            if expected is not None:
                np.testing.assert_array_equal(boxes[0], expected[0])
                np.testing.assert_array_equal(boxes[1], expected[1])
            expected = boxes
        self.assertEqual(validate_world_shell_layout(256, 5), 13)
        self.assertEqual(validate_world_shell_layout(1024, 5), 15)

    def test_uniform_children_partition_each_parent_in_morton_order(self):
        base = background_cells_numpy(256, 5, np)
        lo, hi = leaf_boxes_numpy(base, 256, [-1.] * 3, [1.] * 3, 5, np)
        for subdivision in (1, 2):
            children = 8 ** subdivision
            cells = background_cells_numpy(256, 5, np, subdivision)
            lower, upper = leaf_boxes_numpy(cells, 256, [-1.] * 3, [1.] * 3, 5, np, subdivision)
            self.assertEqual(len(cells), 280 * children)
            grouped_lower, grouped_upper = lower.reshape(-1, children, 3), upper.reshape(-1, children, 3)
            np.testing.assert_array_equal(grouped_lower.min(axis=1), lo)
            np.testing.assert_array_equal(grouped_upper.max(axis=1), hi)
            np.testing.assert_array_equal(np.prod(upper - lower, axis=1).reshape(-1, children).sum(axis=1),
                                          np.prod(hi - lo, axis=1))
            # Consecutive child Morton IDs toggle x, then y, then z.
            edge = 1. / 2 ** subdivision
            np.testing.assert_array_equal(lower[:8], -2. + edge * np.array([
                [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
                [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]]))
            if subdivision == 2:
                np.testing.assert_array_equal(lower[8], [-1.5, -2., -2.])
            overlap = np.minimum(upper[:children, None], upper[None, :children]) - np.maximum(
                lower[:children, None], lower[None, :children])
            intersects = np.all(overlap > 0, axis=-1)
            np.fill_diagonal(intersects, False)
            self.assertFalse(intersects.any())

    def test_subdivision_limits_and_rebased_addresses(self):
        for subdivision in (-1, True, 1.5, 10):
            with self.assertRaisesRegex(ValueError, "background_subdivision_level"):
                validate_world_shell_layout(1024, 10, background_subdivision_level=subdivision)
        with self.assertRaisesRegex(ValueError, "background_subdivision_level"):
            validate_world_shell_layout(4, 2, background_subdivision_level=2)
        validate_world_shell_layout(1024, 10, background_subdivision_level=9)
        cells = np.array([256 ** 3, 256 ** 3 + 56 * 5 * 64 - 1], dtype=np.uint32)
        reference = leaf_boxes_numpy(cells, 256, [-1.] * 3, [1.] * 3, 5, np, 2)
        shifted = cells + (1024 ** 3 - 256 ** 3)
        actual = leaf_boxes_numpy(shifted, 1024, [-1.] * 3, [1.] * 3, 5, np, 2)
        np.testing.assert_array_equal(actual, reference)

    def test_native_configuration_sets_policy_and_subdivision(self):
        changes = []
        geometry_changes = []
        model = SimpleNamespace(core_resolution=16, grid_min=[-1.] * 3, grid_max=[1.] * 3,
            cell_index_bits=64, world_shell_version=4,
            native_background_shell_levels=lambda: 5, native_background_subdivision_level=lambda: 0,
            set_world_shells=lambda *args: changes.append(args))
        model.ellipsoid = True
        model.native_background_use_ellipsoid = lambda: model.ellipsoid
        def set_geometry(enabled):
            geometry_changes.append(enabled)
            model.ellipsoid = enabled
        model.set_background_use_ellipsoid = set_geometry
        configure_world_shell_model(model, 5)
        self.assertEqual(changes, [])
        self.assertEqual(geometry_changes, [])
        self.assertEqual(model.background_refinement, "fixed")
        self.assertEqual(model.background_geometry, "ellipsoid")
        configure_world_shell_model(model, 5, "uniform", 2)
        self.assertEqual(changes, [(5, 2)])
        self.assertEqual((model.background_refinement, model.background_subdivision_level,
                          model.world_shell_version), ("uniform", 2, 5))
        configure_world_shell_model(model, 5, "fixed", 0, "occupied_cell")
        self.assertEqual(model.background_geometry, "occupied_cell")
        self.assertEqual(geometry_changes, [False])
        configure_world_shell_model(model, 5, "fixed", 0, "occupied_cell")
        self.assertEqual(geometry_changes, [False])
        configure_world_shell_model(model, 5)
        self.assertEqual(geometry_changes, [False, True])
        with patch.object(model, "cell_index_bits", 32), self.assertRaisesRegex(RuntimeError, "64-bit"):
            configure_world_shell_model(model, 5)

    def test_sparse_uint64_addresses_at_maximum_supported_subdivision(self):
        resolution, levels, subdivision = 1024, 10, 9
        extent = resolution ** 3 + 56 * levels * 8 ** subdivision
        self.assertGreater(extent, 2 ** 32)
        cells = np.array([resolution ** 3, 2 ** 32 + 17, extent - 1], np.uint64)
        lower, upper = leaf_boxes_numpy(cells, resolution, [-1.] * 3, [1.] * 3,
                                        levels, np, subdivision)
        np.testing.assert_array_equal(lower[0], [-2.] * 3)
        np.testing.assert_array_equal(upper[0] - lower[0], [1. / 512] * 3)
        np.testing.assert_array_equal(lower[-1], [1023.] * 3)
        np.testing.assert_array_equal(upper[-1], [1024.] * 3)
        self.assertTrue(np.isfinite(lower).all() and np.all(upper > lower))
        with self.assertRaisesRegex(ValueError, "cell address"):
            leaf_boxes_numpy(np.array([extent], np.uint64), resolution, [-1.] * 3, [1.] * 3,
                             levels, np, subdivision)

    def test_core_and_shell_address_bounds(self):
        cells = np.array([0, 4 ** 3 - 1, 4 ** 3, 4 ** 3 + 55], dtype=np.uint32)
        lower, upper = leaf_boxes_numpy(cells, 4, [-1.] * 3, [1.] * 3, 1, np)
        np.testing.assert_array_equal(lower, [[-1, -1, -1], [.5, .5, .5], [-2, -2, -2], [1, 1, 1]])
        np.testing.assert_array_equal(upper[-1], [2, 2, 2])
        for levels in (0, -1, 11, True, 1.5):
            with self.assertRaises(ValueError):
                validate_shell_levels(levels)
        for resolution in (1, 3, 2048, True):
            with self.assertRaises(ValueError):
                validate_world_shell_layout(resolution, 5)
        with self.assertRaises(ValueError):
            leaf_boxes_numpy(np.array([120]), 4, [-1.] * 3, [1.] * 3, 1, np)

    def test_parallel_rays_and_root_exit(self):
        dr = SimpleNamespace(select=np.where, minimum=np.minimum, maximum=np.maximum,
                             any=np.any, max=np.max, min=np.min)
        mi = SimpleNamespace(Point3f=lambda value: np.asarray(value, dtype=float))
        camera = dict(world_shell_root_min=[-32.] * 3, world_shell_root_max=[32.] * 3)
        for origin, direction, distance in (([0, 0, 0], [0, 0, 1], 32),
                ([33, 0, 0], [0, 0, 1], 0), ([0, 0, 40], [0, 0, -1], 72),
                ([0, 0, 40], [0, 0, 1], 0), ([32, 0, 0], [0, 1, 0], 32)):
            ray = SimpleNamespace(o=np.array(origin, dtype=float), d=np.array(direction, dtype=float))
            self.assertEqual(float(world_shell_ray_maxt(ray, camera, mi, dr)), distance)


class WorldShellConfigurationTests(unittest.TestCase):
    def test_new_configuration_and_protocol(self):
        config = PreviewConfig()
        update_config(config, {"data": {"scene_mode": "unbounded", "background_shell_levels": 8}})
        validate_config(config)
        config.data.initialization = dict(version=7, scene_mode="unbounded", scene_mapping="world_shell",
                                          background_shell_levels=8, background_opacity=.5)
        report = scene_background_protocol(config.data, config.render)
        self.assertEqual(report["background_initial_voxels"], 448)
        self.assertEqual(report["root_half_extent"], 256)
        self.assertEqual(report["background_refinement"], "fixed")
        self.assertEqual(report["background_geometry"], "occupied_cell")

    def test_new_geometry_protocol_is_explicit(self):
        config = PreviewConfig()
        config.data.initialization = dict(version=8, scene_mode="unbounded", scene_mapping="world_shell",
            background_shell_levels=8, background_opacity=.5, background_refinement="uniform",
            background_subdivision_level=0, world_shell_version=3, background_geometry="ellipsoid")
        report = scene_background_protocol(config.data, config.render)
        self.assertEqual(report["world_shell_version"], 3)
        self.assertEqual(report["background_geometry"], "ellipsoid")
        report = scene_background_protocol(config.data, config.render,
            {**config.data.initialization, "background_geometry": "occupied_cell"})
        self.assertEqual(report["background_geometry"], "occupied_cell")

    def test_old_background_configuration_is_not_reinterpreted(self):
        for group, field, value in (("data", "background_final_layers", 8),
                ("data", "background_far_distance", None),
                ("model", "background_layered_sampling", False),
                ("model", "background_use_ellipsoid", False)):
            with self.assertRaisesRegex(ValueError, "Compressed background"):
                update_config(PreviewConfig(), {group: {field: value}})

    def test_evaluation_protocol_uses_current_model_subdivision(self):
        config = PreviewConfig()
        config.data.initialization = dict(version=8, scene_mode="unbounded", scene_mapping="world_shell",
            background_shell_levels=8, background_opacity=.5, background_refinement="uniform",
            background_subdivision_level=0, world_shell_version=2)
        report = scene_background_protocol(config.data, config.render,
            {**config.data.initialization, "background_subdivision_level": 2})
        self.assertEqual(report["background_initial_voxels"], 448)
        self.assertEqual(report["background_leaf_capacity"], 28672)
        self.assertEqual(report["background_subdivision_level"], 2)
        self.assertEqual(report["background_refinement"], "uniform")


class WorldShellCheckpointTests(unittest.TestCase):
    def test_background_geometry_round_trip_and_version_guards(self):
        state, metadata = small_model()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.npz"
            for geometry in ("ellipsoid", "occupied_cell"):
                saved = {**metadata, "background_geometry": geometry}
                save_checkpoint(path, state, saved)
                restored = load_checkpoint(path)["model_metadata"]
                self.assertEqual(restored, saved)
                self.assertEqual(world_shell_metadata_geometry(restored), geometry)
            for version in (1, 2):
                legacy = {**metadata, "world_shell_version": version}
                legacy.pop("background_geometry")
                save_checkpoint(path, state, legacy)
                restored = load_checkpoint(path)["model_metadata"]
                self.assertEqual(restored, legacy)
                self.assertEqual(world_shell_metadata_geometry(restored), "occupied_cell")
        for geometry in (None, "sphere", True):
            invalid = {**metadata, "background_geometry": geometry}
            if geometry is None:
                invalid.pop("background_geometry")
            with self.subTest(geometry=geometry), self.assertRaisesRegex(ValueError, "background_geometry"):
                validate_model_state(state, invalid)

    def test_shell_addresses_round_trip_and_validate(self):
        state, metadata = small_model()
        validate_model_state(state, metadata)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.npz"
            save_checkpoint(path, state, metadata)
            checkpoint = load_checkpoint(path)
        self.assertEqual(checkpoint["model_metadata"], metadata)
        for name in state:
            np.testing.assert_array_equal(checkpoint["model_state"][name], state[name])

    def test_invalid_address_duplicate_and_legacy_mapping_rejected(self):
        state, metadata = small_model()
        for change in (lambda cells: cells.__setitem__(-1, 4 ** 3 + 280),
                       lambda cells: cells.__setitem__(-1, cells[-2])):
            invalid = copy.deepcopy(state)
            change(invalid["cell_indices"])
            with self.assertRaises(ValueError):
                validate_model_state(invalid, metadata)
        for mapping in ("projective_shell", "radial_seed_shell"):
            with self.assertRaisesRegex(ValueError, "Legacy compressed"):
                validate_model_state(state, {**metadata, "scene_mapping": mapping})

    def test_saved_configuration_must_match_shell_count(self):
        state, metadata = small_model()
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "background_shell_levels"):
                save_checkpoint(Path(directory) / "model.npz", state, metadata,
                                config={"data": {"background_shell_levels": 6}})

    def test_uniform_checkpoint_round_trip_and_initialization_policy(self):
        state, metadata = small_model(levels=2, resolution=16, subdivision=2, refinement="uniform")
        config = {"data": {"background_shell_levels": 2, "initialization": dict(version=8,
            scene_mapping="world_shell", background_shell_levels=2, background_refinement="uniform",
            background_subdivision_level=0, world_shell_version=3, background_geometry="ellipsoid")}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.npz"
            save_checkpoint(path, state, metadata, config=config)
            restored = load_checkpoint(path)
            self.assertEqual(restored["model_metadata"], metadata)
            np.testing.assert_array_equal(restored["model_state"]["cell_indices"], state["cell_indices"])
            config["data"]["initialization"]["background_refinement"] = "fixed"
            with self.assertRaisesRegex(ValueError, "layout differs"):
                save_checkpoint(path, state, metadata, config=config)
        invalid = {**state, "cell_indices": state["cell_indices"].copy()}
        invalid["cell_indices"][-1] += 1
        with self.assertRaisesRegex(ValueError, "address space"):
            validate_model_state(invalid, metadata)
        with self.assertRaisesRegex(ValueError, "world_shell_version 2"):
            validate_model_state(state, {**metadata, "world_shell_version": 1})


class WorldShellDatasetTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name) / "bonsai"
        write_fixture(self.root)

    def test_camera_transform_preserved_points_filtered_without_compression(self):
        prepared = prepare_colmap(self.root)
        initial = prepared["initialization"]
        self.assertEqual(initial["version"], 8)
        self.assertEqual(initial["background_refinement"], "uniform")
        self.assertEqual(initial["background_subdivision_level"], 0)
        self.assertEqual(initial["world_shell_version"], 5)
        self.assertEqual(initial["background_geometry"], "ellipsoid")
        self.assertEqual(initial["background_shell_levels"], 5)
        self.assertEqual(initial["background_voxels"], 280)
        self.assertEqual(prepared["resolution"], 256)
        self.assertEqual(prepared["grid_min"], [-1.] * 3)
        np.testing.assert_allclose(initial["center"], [0., 0., 0.])
        self.assertAlmostEqual(initial["radius"], 4.4)
        np.testing.assert_allclose(prepared["positions"], [[1. / 4.4, 2. / 4.4, 0.]])
        self.assertEqual(initial["point_filter"]["excluded_count"], 2)
        self.assertEqual(prepared["colors"].shape, (1, 3))

    def test_shell_count_changes_neither_core_points_nor_cameras(self):
        a = prepare_colmap(self.root, background_shell_levels=1)
        b = prepare_colmap(self.root, background_shell_levels=8)
        np.testing.assert_array_equal(a["positions"], b["positions"])
        self.assertEqual(a["views"], b["views"])

    def test_saved_camera_only_load_and_mismatched_settings(self):
        initialization = prepare_colmap(self.root, background_shell_levels=7)["initialization"]
        with patch("reconstruction_dataset.read_colmap_points", side_effect=AssertionError("Point reload")):
            prepared = prepare_colmap(self.root, initialization=initialization, load_points=False)
        self.assertIsNone(prepared["positions"])
        self.assertEqual(prepared["initialization"], initialization)
        with self.assertRaisesRegex(ValueError, "background_shell_levels"):
            prepare_colmap(self.root, initialization=initialization, background_shell_levels=5, load_points=False)

    def test_legacy_mapping_rejected_and_bounded_points_retained(self):
        for mapping in ("projective_shell", "radial_seed_shell"):
            with self.assertRaisesRegex(ValueError, "Legacy compressed"):
                prepare_colmap(self.root, initialization={"version": 6, "scene_mode": "unbounded", "scene_mapping": mapping})
        with self.assertRaisesRegex(ValueError, "Compressed background"):
            prepare_colmap(self.root, background_final_layers=8)
        bounded = prepare_colmap(self.root, scene_mode="bounded")
        self.assertEqual(len(bounded["positions"]), 3)
        self.assertEqual(bounded["initialization"]["scene_mapping"], "bounded")
        self.assertTrue(np.all(np.abs(bounded["positions"]) < 1.))

    def test_previous_world_shell_initialization_remains_fixed(self):
        initial = prepare_colmap(self.root)["initialization"]
        initial.update(version=7, background_refinement="fixed")
        initial.pop("background_subdivision_level")
        initial.pop("world_shell_version")
        initial.pop("background_geometry")
        restored = prepare_colmap(self.root, initialization=initial, load_points=False)
        self.assertEqual(restored["initialization"], initial)
        self.assertEqual(restored["initialization"]["background_refinement"], "fixed")
        self.assertNotIn("background_subdivision_level", restored["initialization"])
        self.assertEqual(world_shell_metadata_geometry(restored["initialization"]), "occupied_cell")


if __name__ == "__main__":
    unittest.main()
