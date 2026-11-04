"""World-shell refinement preserves legacy shells or subdivides live parents."""
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from reconstruction_topology import CoarseToFineConfig, NoRepresentableChildren, refine_state
from reconstruction_world_refinement import refine_world_device, refine_world_numpy
from reconstruction_world_shell import leaf_boxes_numpy


def fixture(n=4, shells=5, refinement="fixed", geometry=None):
    # Interleave foreground/background rows to exercise preserved Adam row maps.
    core_cell = (n // 2 - 1) * (1 + n + n * n)
    cells = np.concatenate((np.arange(n**3, n**3 + 91), [core_cell],
                            np.arange(n**3 + 91, n**3 + 56 * shells))).astype(np.uint32)
    count = len(cells)
    rng = np.random.default_rng(183)
    metadata = dict(resolution=n, core_resolution=n, grid_min=[-1.] * 3, grid_max=[1.] * 3,
                    sh_count=9, active_sh_count=9, opacity_sh_count=1,
                    scene_mapping="world_shell", background_shell_levels=shells,
                    background_refinement=refinement)
    if refinement == "uniform":
        metadata.update(world_shell_version=2, background_subdivision_level=0)
    state = dict(cell_indices=cells, centers=np.full(count * 3, .5, np.float32),
                 log_scales=np.full(count * 3, np.log(1.6 / n), np.float32),
                 rotations=np.tile(np.array([1, 0, 0, 0], np.float32), count),
                 radiance_sh=rng.normal(size=count * 27).astype(np.float32),
                 opacity_sh=rng.normal(size=count).astype(np.float32))
    if geometry is not None:
        metadata.update(world_shell_version=3, background_geometry=geometry,
                        background_subdivision_level=0)
    if geometry == "ellipsoid":
        rows = np.flatnonzero(cells >= n ** 3)
        lower, upper = leaf_boxes_numpy(cells[rows], n, [-1.] * 3, [1.] * 3, shells, np)
        state["centers"].reshape(-1, 3)[rows] = [.32, .61, .46]
        state["log_scales"].reshape(-1, 3)[rows] = np.log((upper - lower) * [.42, .25, .31])
        state["rotations"].reshape(-1, 4)[rows] = [np.cos(.37), 0., 0., np.sin(.37)]
    return dict(model_metadata=metadata, model_state=state)


def sparse_ellipsoid_fixture(n=4, shells=3, refinement="uniform"):
    checkpoint = fixture(n, shells, refinement, "ellipsoid")
    state = checkpoint["model_state"]
    count = len(state["cell_indices"])
    cells = state["cell_indices"]
    selected = np.flatnonzero((cells < n ** 3) | np.isin(cells - np.uint32(n ** 3),
                                                      [shell * 56 + 3 for shell in range(shells)]))
    checkpoint["model_state"] = {name: values.reshape(count, -1)[selected].ravel()
                                 for name, values in state.items()}
    return checkpoint


def empty_first_shell_fixture():
    checkpoint = fixture(4, 3, "uniform", "ellipsoid")
    state = checkpoint["model_state"]
    cells = state["cell_indices"]
    first_shell = (cells >= 4 ** 3) & (cells < 4 ** 3 + 56)
    state["centers"].reshape(-1, 3)[first_shell] = .5
    state["log_scales"].reshape(-1, 3)[first_shell] = np.log(1e-4)
    return checkpoint


def high_address_fixture():
    """A few physical leaves in each B=128 shell, never a dense shell table."""
    n, subdivision, shells = 256, 6, 8
    local = []
    for shell in range(shells):
        for ordinal, child in ((3, (1, 3, 5)), (24, (17, 32, 41)), (55, (63, 62, 61))):
            morton = sum(((child[axis] >> bit) & 1) << (3 * bit + axis)
                         for bit in range(subdivision) for axis in range(3))
            local.append((shell * 56 + ordinal) * 8 ** subdivision + morton)
    core_cell = (n // 2 - 1) * (1 + n + n * n)
    cells = np.insert(np.asarray(local, np.uint64) + n ** 3, 5, np.uint64(core_cell))
    count = len(cells)
    lower, upper = leaf_boxes_numpy(cells, n, [-1.] * 3, [1.] * 3, shells, np, subdivision)
    center = np.tile([.23, .31, .42], (count, 1)).astype(np.float32)
    scales = (upper - lower) * [.25, .18, .13]
    center[5], scales[5] = .5, 1.6 / n
    rng = np.random.default_rng(8357)
    metadata = dict(resolution=n, core_resolution=n, grid_min=[-1.] * 3, grid_max=[1.] * 3,
        sh_count=9, active_sh_count=1, opacity_sh_count=1, scene_mapping="world_shell",
        world_shell_version=4, background_shell_levels=shells, background_subdivision_level=subdivision,
        background_geometry="ellipsoid", background_refinement="uniform")
    state = dict(cell_indices=cells, centers=center.ravel(), log_scales=np.log(scales).astype(np.float32).ravel(),
        rotations=np.tile(np.asarray([np.cos(.37), 0., 0., np.sin(.37)], np.float32), count),
        radiance_sh=rng.normal(size=count * 27).astype(np.float32),
        opacity_sh=rng.normal(size=count).astype(np.float32))
    return dict(model_metadata=metadata, model_state=state)


def growth_address_alias_fixture():
    from test_reconstruction_growth_lookup import checkpoint, manager_for
    state = checkpoint([73, 75, 0, 0], resolution=512)
    state["model_state"]["cell_indices"] = np.asarray([73, 75, 2 ** 32 + 74, 2 ** 32 + 73], np.uint64)
    state["model_metadata"].update(core_resolution=512, scene_mapping="world_shell",
        world_shell_version=4, background_shell_levels=8, background_subdivision_level=8,
        background_geometry="ellipsoid", background_refinement="uniform")
    manager = manager_for(state, use_rgb_growth=True, growth_min_supporting_parents=2,
                           growth_evidence_penetration=0.)
    manager.cooldown = {2 ** 32 + 74: 12}
    return state, manager


def bounded_shell_reference(checkpoint, config):
    """Independent physical-box mapping to the established foreground fitter."""
    metadata, state = checkpoint["model_metadata"], checkpoint["model_state"]
    n, subdivision = metadata["resolution"], metadata.get("background_subdivision_level", 0)
    cells, count = state["cell_indices"], len(state["cell_indices"])
    pieces = []
    parents = cells[cells >= n ** 3].astype(np.uint64) - n ** 3
    next_cells = ((parents[:, None] * 8) + np.arange(8, dtype=np.uint64) + (n * 2) ** 3).ravel()
    new_lower, _ = leaf_boxes_numpy(next_cells, n * 2, metadata["grid_min"], metadata["grid_max"],
                                    metadata["background_shell_levels"], np, subdivision + 1)
    address_by_lower = {tuple(box): cell for box, cell in zip(new_lower, next_cells)}
    center = (np.asarray(metadata["grid_min"]) + metadata["grid_max"]) * .5
    for shell in range(metadata["background_shell_levels"]):
        begin = n ** 3 + shell * 56 * 8 ** subdivision
        selected = (cells >= begin) & (cells < begin + 56 * 8 ** subdivision)
        if not np.any(selected):
            continue
        lower, upper = leaf_boxes_numpy(cells[selected], n, metadata["grid_min"], metadata["grid_max"],
                                        metadata["background_shell_levels"], np, subdivision)
        edge = upper[0, 0] - lower[0, 0]
        resolution = 4 * 2 ** subdivision
        lo, hi = center - resolution * edge * .5, center + resolution * edge * .5
        xyz = np.rint((lower - lo) / edge).astype(np.uint32)
        shell_state = {name: value.reshape(count, -1)[selected].ravel() for name, value in state.items()}
        shell_state["cell_indices"] = xyz @ np.array([1, resolution, resolution ** 2], np.uint32)
        bounded = {"model_metadata": dict(resolution=resolution, core_resolution=resolution,
            grid_min=lo.tolist(), grid_max=hi.tolist(), sh_count=metadata["sh_count"],
            active_sh_count=metadata["active_sh_count"], opacity_sh_count=metadata["opacity_sh_count"],
            scene_mapping="bounded"), "model_state": shell_state}
        fitted = refine_state(bounded, config)["model_state"]
        child_cells = fitted["cell_indices"]
        fine = resolution * 2
        xyz = np.stack((child_cells % fine, child_cells // fine % fine, child_cells // (fine ** 2)), axis=1)
        physical_lower = lo + xyz * (edge * .5)
        fitted["cell_indices"] = np.asarray([address_by_lower[tuple(box)] for box in physical_lower], np.uint64)
        pieces.append(fitted)
    return {name: np.concatenate([piece[name] for piece in pieces]) for name in state}


class WorldRefinementTests(unittest.TestCase):
    def test_public_uniform_refinement_rejects_adaptive_core_before_touching_arrays(self):
        metadata = dict(fixture()["model_metadata"], foreground_refinement="adaptive",
                        foreground_max_resolution=1024)
        # No state arrays or GPU runtime are supplied: rejection must happen
        # before any conversion, allocation, fitting or parameter access.
        checkpoint = dict(model_metadata=metadata, model_state=object())
        fit = Mock(side_effect=AssertionError("Uniform fitter must not run"))
        for callback in (
            lambda: refine_world_numpy(checkpoint, CoarseToFineConfig(), 1000, fit, np),
            lambda: refine_world_device(checkpoint, CoarseToFineConfig(), 1000, fit, object()),
            lambda: refine_state(checkpoint, CoarseToFineConfig(), 1000),
        ):
            with self.assertRaisesRegex(ValueError, "adaptive foreground.*selective leaf"):
                callback()
        fit.assert_not_called()

    def test_uint64_background_and_cooldown_do_not_alias_core_growth_cells(self):
        from reconstruction_topology import TopologyManager
        checkpoint, manager = growth_address_alias_fixture()
        proposed = TopologyManager._generate_growth(manager, checkpoint, 10)
        selected = proposed["cells"] == 74
        self.assertEqual(int(selected.sum()), 2)
        np.testing.assert_array_equal(np.sort(proposed["parent_ids"][selected]), [0, 1])
        self.assertTrue(np.all(proposed["parent_ids"] < 2))

    def test_sparse_s6_to_s8_keeps_uint64_world_addresses_and_physical_fits(self):
        current = high_address_fixture()
        config = CoarseToFineConfig()
        for resolution, subdivision in ((512, 7), (1024, 8)):
            expected = bounded_shell_reference(current, config)
            actual = refine_state(current, config)
            self.assertEqual(actual["model_metadata"]["resolution"], resolution)
            self.assertEqual(actual["model_metadata"]["background_subdivision_level"], subdivision)
            cells = actual["model_state"]["cell_indices"]
            self.assertEqual(cells.dtype, np.dtype(np.uint64))
            self.assertEqual(np.unique(cells).size, len(cells))
            self.assertLess(len(cells), 10000)
            for key, values in expected.items():
                np.testing.assert_array_equal(actual["model_state"][key][:len(values)], values, err_msg=key)
            current = actual
        self.assertTrue(np.any(current["model_state"]["cell_indices"] > np.iinfo(np.uint32).max))

    def test_empty_shell_fit_does_not_abort_other_shells_or_core(self):
        checkpoint = empty_first_shell_fixture()
        state = checkpoint["model_state"]
        cells = state["cell_indices"]
        first_shell = (cells >= 4 ** 3) & (cells < 4 ** 3 + 56)
        self.assertEqual(int(first_shell.sum()), 56)
        retained = {name: value.reshape(len(cells), -1)[~first_shell].ravel()
                    for name, value in state.items()}
        expected = refine_state({"model_metadata": checkpoint["model_metadata"], "model_state": retained},
                                CoarseToFineConfig())
        actual = refine_state(checkpoint, CoarseToFineConfig())
        new_cells = actual["model_state"]["cell_indices"]
        self.assertTrue(np.any(new_cells < 8 ** 3))
        self.assertTrue(np.any(new_cells >= 8 ** 3 + 56 * 8))
        self.assertFalse(np.any((new_cells >= 8 ** 3) & (new_cells < 8 ** 3 + 56 * 8)))
        for key, value in expected["model_state"].items():
            np.testing.assert_array_equal(actual["model_state"][key], value)

    def test_shell_fit_does_not_swallow_capacity_or_invalid_data_errors(self):
        checkpoint = empty_first_shell_fixture()
        for error in (RuntimeError("Refinement exceeds max_active_voxels"), ValueError("invalid geometry")):
            fit_core = Mock(side_effect=error)
            with self.subTest(error=error), self.assertRaises(type(error)) as raised:
                refine_world_numpy(checkpoint, CoarseToFineConfig(), 1000000, fit_core, np)
            self.assertIs(raised.exception, error)
            fit_core.assert_called_once()

    def test_empty_core_fit_and_whole_world_still_raise_typed_outcome(self):
        checkpoint = empty_first_shell_fixture()
        state = checkpoint["model_state"]
        cells = state["cell_indices"]
        for selected in ((cells >= 4 ** 3) & (cells < 4 ** 3 + 56), cells < 4 ** 3):
            limited = {key: values.reshape(len(cells), -1)[selected].ravel()
                       for key, values in state.items()}
            limited["centers"].fill(.5)
            limited["log_scales"].fill(np.log(1e-4))
            with self.assertRaises(NoRepresentableChildren):
                refine_state({"model_metadata": checkpoint["model_metadata"], "model_state": limited},
                             CoarseToFineConfig())

    def test_ellipsoid_shells_match_foreground_physical_fit_twice(self):
        current = sparse_ellipsoid_fixture()
        current["model_metadata"]["active_sh_count"] = 1
        config = CoarseToFineConfig(opacity_optical_depth_scale=.37)
        original = {key: value.copy() for key, value in current["model_state"].items()}
        first = current
        for _ in range(2):
            expected = bounded_shell_reference(current, config)
            refined = refine_state(current, config)
            bg_count = len(expected["cell_indices"])
            self.assertGreater(bg_count, 0)
            self.assertEqual(refined["model_metadata"]["background_geometry"], "ellipsoid")
            self.assertEqual(refined["model_metadata"]["active_sh_count"], 1)
            self.assertEqual(len(refined["refinement_background_rows"]), 0)
            for key, values in expected.items():
                np.testing.assert_array_equal(refined["model_state"][key][:len(values)], values, err_msg=key)
            self.assertTrue(np.any(refined["model_state"]["centers"][:bg_count * 3] != .5))
            rotations = refined["model_state"]["rotations"][:bg_count * 4].reshape(-1, 4)
            self.assertTrue(np.all(rotations[:, 3] != 0.))
            current = refined
        for key, values in original.items():
            np.testing.assert_array_equal(first["model_state"][key], values)

    def test_ellipsoid_capacity_uses_actual_fitted_children(self):
        checkpoint = sparse_ellipsoid_fixture()
        state = checkpoint["model_state"]
        selected = state["cell_indices"] >= 4 ** 3
        checkpoint["model_state"] = {key: values.reshape(len(selected), -1)[selected].ravel()
                                     for key, values in state.items()}
        checkpoint["model_state"]["centers"].fill(.2)
        checkpoint["model_state"]["log_scales"] -= np.log(3.)
        expected = refine_state(checkpoint, CoarseToFineConfig())
        count = len(expected["model_state"]["cell_indices"])
        self.assertLess(count, int(selected.sum()) * 8)
        actual = refine_state(checkpoint, CoarseToFineConfig(), max_active_voxels=count)
        for key, values in expected["model_state"].items():
            np.testing.assert_array_equal(actual["model_state"][key], values)
        with self.assertRaisesRegex(RuntimeError, "max_active_voxels"):
            refine_state(checkpoint, CoarseToFineConfig(), max_active_voxels=count - 1)

    def test_fixed_ellipsoid_background_preserves_trained_geometry_and_row_map(self):
        checkpoint = sparse_ellipsoid_fixture(refinement="fixed")
        cells = checkpoint["model_state"]["cell_indices"]
        rows = np.flatnonzero(cells >= 4 ** 3)
        refined = refine_state(checkpoint, CoarseToFineConfig())
        np.testing.assert_array_equal(refined["refinement_background_rows"], rows)
        for key, values in checkpoint["model_state"].items():
            expected = values.reshape(len(cells), -1)[rows].ravel()
            if key == "cell_indices":
                expected = expected + 8 ** 3 - 4 ** 3
            np.testing.assert_array_equal(refined["model_state"][key][:len(expected)], expected)

    def test_uniform_directional_opacity_is_rejected_before_fitting_or_upload(self):
        metadata = fixture(refinement="uniform")["model_metadata"]
        metadata["opacity_sh_count"] = 9
        for implementation, runtime in ((refine_world_numpy, np), (refine_world_device, None)):
            with self.subTest(implementation=implementation.__name__):
                fit_core = Mock(side_effect=AssertionError("Invalid opacity must not reach core fitting"))
                with self.assertRaisesRegex(ValueError, "opacity_sh_count=1"):
                    implementation({"model_metadata": metadata, "model_state": object()},
                        CoarseToFineConfig(), 1000000, fit_core, runtime)
                fit_core.assert_not_called()

    def test_invalid_next_layout_is_rejected_before_fitting_or_upload(self):
        original = fixture(512, 8, "uniform")["model_metadata"]
        for changes, message in (({"background_subdivision_level": 9}, "background_subdivision_level"),
                                 ({"background_subdivision_level": 1.5}, "background_subdivision_level"),
                                 ({"resolution": 3}, "power of two"),
                                 ({"resolution": 1024}, "power of two"),
                                 ({"world_shell_version": 1}, "world_shell_version 2")):
            metadata = dict(original, **changes)
            for implementation, runtime in ((refine_world_numpy, np), (refine_world_device, None)):
                with self.subTest(changes=changes, implementation=implementation.__name__):
                    fit_core = Mock(side_effect=AssertionError("Invalid layout must not reach core fitting"))
                    with self.assertRaisesRegex(ValueError, message):
                        implementation({"model_metadata": metadata, "model_state": object()},
                            CoarseToFineConfig(), 1000000, fit_core, runtime)
                    fit_core.assert_not_called()

    def test_bounded_core_fit_receives_no_world_shell_metadata(self):
        checkpoint = fixture(refinement="uniform")
        fit_core = Mock(side_effect=refine_state)
        refine_world_numpy(checkpoint, CoarseToFineConfig(), 1000000, fit_core, np)
        core_metadata = fit_core.call_args.args[0]["model_metadata"]
        self.assertEqual(core_metadata["scene_mapping"], "bounded")
        self.assertFalse(any(key.startswith("background_") for key in core_metadata))
        self.assertNotIn("world_shell_version", core_metadata)

    def test_uniform_shell_children_tile_parents_and_inherit_inactive_sh_twice(self):
        current = fixture(256, 8, "uniform")
        current["model_metadata"]["active_sh_count"] = 1
        original = {name: values.copy() for name, values in current["model_state"].items()}
        checkpoint = current
        bits = np.asarray([(i & 1, (i >> 1) & 1, (i >> 2) & 1) for i in range(8)])
        for resolution, subdivision, expected_count in ((512, 1, 3584), (1024, 2, 28672)):
            metadata, state = current["model_metadata"], current["model_state"]
            old_count = len(state["cell_indices"])
            background_rows = np.flatnonzero(state["cell_indices"] >= metadata["resolution"] ** 3)
            old_cells = state["cell_indices"][background_rows]
            lower, upper = leaf_boxes_numpy(old_cells, metadata["resolution"], [-1.] * 3,
                [1.] * 3, 8, np, background_subdivision_level=subdivision - 1)
            current = refine_state(current, CoarseToFineConfig())
            self.assertEqual(current["model_metadata"]["resolution"], resolution)
            self.assertEqual(current["model_metadata"]["background_subdivision_level"], subdivision)
            self.assertEqual(current["model_metadata"]["active_sh_count"], 1)
            self.assertEqual(len(current["refinement_background_rows"]), 0)
            actual = current["model_state"]
            self.assertEqual(np.count_nonzero(actual["cell_indices"] >= resolution ** 3), expected_count)
            expected_cells = (old_cells.astype(np.uint64) - metadata["resolution"] ** 3)[:, None] * 8
            expected_cells = expected_cells + np.arange(8) + resolution ** 3
            np.testing.assert_array_equal(actual["cell_indices"][:expected_count], expected_cells.ravel())
            new_lower, new_upper = leaf_boxes_numpy(actual["cell_indices"][:expected_count], resolution,
                [-1.] * 3, [1.] * 3, 8, np, background_subdivision_level=subdivision)
            half_edge = (upper - lower) * .5
            expected_lower = lower[:, None, :] + bits[None, :, :] * half_edge[:, None, :]
            np.testing.assert_array_equal(new_lower, expected_lower.reshape(-1, 3))
            np.testing.assert_array_equal(new_upper, (expected_lower + half_edge[:, None, :]).reshape(-1, 3))
            for key in ("radiance_sh", "rotations", "log_scales"):
                parents = state[key].reshape(old_count, -1)[background_rows]
                expected = np.repeat(parents, 8, axis=0)
                if key == "log_scales":
                    expected -= np.log(2.)
                np.testing.assert_array_equal(actual[key][:expected.size], expected.ravel())
            np.testing.assert_array_equal(actual["centers"][:expected_count * 3], .5)
        for name, values in original.items():
            np.testing.assert_array_equal(checkpoint["model_state"][name], values)

    def test_uniform_opacity_preserves_two_child_axis_transmittance(self):
        checkpoint = fixture(refinement="uniform")
        state = checkpoint["model_state"]
        count = len(state["cell_indices"])
        rows = np.flatnonzero(state["cell_indices"] >= 4 ** 3)[:9]
        checkpoint["model_state"] = {k: v.reshape(count, -1)[rows].ravel()
                                     for k, v in state.items()}
        # Include values for which explicitly forming sigmoid loses all precision.
        logits = np.array([-1e6, -300., -30., -1., 0., 1., 30., 300., 1e6])
        y0 = 1 / (2 * np.sqrt(np.pi))
        checkpoint["model_state"]["opacity_sh"] = (logits / y0).astype(np.float32)
        current = checkpoint
        softplus = lambda x: np.maximum(x, 0.) + np.log1p(np.exp(-np.abs(x)))
        for step in range(2):
            parent_logits = current["model_state"]["opacity_sh"].astype(np.float64) * y0
            current = refine_state(current, CoarseToFineConfig())
            child_logits = current["model_state"]["opacity_sh"].astype(np.float64) * y0
            self.assertTrue(np.isfinite(child_logits).all())
            np.testing.assert_allclose(2 * softplus(child_logits),
                np.repeat(softplus(parent_logits), 8), rtol=1e-6, atol=1e-12)
            parent_alpha = -np.expm1(-softplus(parent_logits))
            child_alpha = -np.expm1(-softplus(child_logits))
            np.testing.assert_allclose(1 - (1 - child_alpha) ** 2,
                np.repeat(parent_alpha, 8), rtol=1e-6, atol=1e-8)

    def test_uniform_only_splits_surviving_background_and_checks_actual_capacity(self):
        checkpoint = fixture(refinement="uniform")
        state = checkpoint["model_state"]
        count = len(state["cell_indices"])
        selected = np.array([1, 35, 95])
        checkpoint["model_state"] = {k: v.reshape(count, -1)[selected].ravel()
                                     for k, v in state.items()}
        with self.assertRaisesRegex(RuntimeError, "max_active_voxels"):
            refine_state(checkpoint, CoarseToFineConfig(), max_active_voxels=23)
        refined = refine_state(checkpoint, CoarseToFineConfig(), max_active_voxels=24)
        self.assertEqual(len(refined["model_state"]["cell_indices"]), 24)
        self.assertEqual(len(refined["refinement_background_rows"]), 0)
        old_local = checkpoint["model_state"]["cell_indices"].astype(np.uint64) - 4 ** 3
        np.testing.assert_array_equal(refined["model_state"]["cell_indices"],
            ((old_local[:, None] * 8) + np.arange(8) + 8 ** 3).ravel())
        # Delete selected first-generation children; a second split must not recreate them.
        retained = np.array([0, 3, 7, 15, 22])
        refined["model_state"] = {k: v.reshape(24, -1)[retained].ravel()
                                  for k, v in refined["model_state"].items()}
        next_refined = refine_state(refined, CoarseToFineConfig(), max_active_voxels=40)
        parents = (next_refined["model_state"]["cell_indices"] - 16 ** 3) // 8
        np.testing.assert_array_equal(parents,
            np.repeat(refined["model_state"]["cell_indices"] - 8 ** 3, 8))

    def test_missing_refinement_policy_keeps_legacy_background_fixed(self):
        checkpoint = fixture()
        checkpoint["model_metadata"].pop("background_refinement")
        refined = refine_state(checkpoint, CoarseToFineConfig())
        self.assertNotIn("background_subdivision_level", refined["model_metadata"])
        self.assertEqual(len(refined["refinement_background_rows"]), 56 * 5)

    def test_two_core_refinements_keep_background_boxes_and_every_parameter(self):
        original = fixture()
        md, state = original["model_metadata"], original["model_state"]
        rows = np.flatnonzero(state["cell_indices"] >= md["resolution"]**3)
        background = {k: v.reshape(len(state["cell_indices"]), -1)[rows].ravel().copy()
                      for k, v in state.items()}
        boxes = leaf_boxes_numpy(background["cell_indices"], 4, [-1.] * 3, [1.] * 3, 5, np)
        current = original
        for resolution in (8, 16):
            current = refine_state(current, CoarseToFineConfig())
            self.assertEqual(current["model_metadata"]["resolution"], resolution)
            new_state = current["model_state"]
            bg_count = len(rows)
            self.assertEqual(np.count_nonzero(new_state["cell_indices"] >= resolution**3), bg_count)
            for key, values in background.items():
                width = len(values) // bg_count
                actual = new_state[key][:bg_count * width]
                if key == "cell_indices":
                    np.testing.assert_array_equal(actual - resolution**3, values - 4**3)
                else:
                    np.testing.assert_array_equal(actual, values)
            new_boxes = leaf_boxes_numpy(new_state["cell_indices"][:bg_count], resolution,
                                         [-1.] * 3, [1.] * 3, 5, np)
            for before, after in zip(boxes, new_boxes):
                np.testing.assert_array_equal(before, after)
        np.testing.assert_array_equal(original["model_state"]["cell_indices"], state["cell_indices"])

    def test_core_children_equal_bounded_fit_and_optimizer_maps_old_background_rows(self):
        checkpoint = fixture()
        state, md = checkpoint["model_state"], checkpoint["model_metadata"]
        core = state["cell_indices"] < md["resolution"]**3
        bounded = {"model_metadata": dict(md, scene_mapping="bounded"), "model_state": {
            k: v.reshape(len(core), -1)[core].ravel() for k, v in state.items()}}
        expected = refine_state(bounded, CoarseToFineConfig())
        actual = refine_state(checkpoint, CoarseToFineConfig())
        rows = np.flatnonzero(~core)
        np.testing.assert_array_equal(actual["refinement_background_rows"], rows)
        for key, values in expected["model_state"].items():
            width = len(state[key]) // len(core)
            np.testing.assert_array_equal(actual["model_state"][key][len(rows) * width:], values)

    def test_empty_core_and_pruned_background_are_not_regrown(self):
        checkpoint = fixture()
        state = checkpoint["model_state"]
        count = len(state["cell_indices"])
        selected = np.array([1, 35, 95])
        checkpoint["model_state"] = {k: v.reshape(count, -1)[selected].ravel()
                                     for k, v in state.items()}
        refined = refine_state(checkpoint, CoarseToFineConfig())
        self.assertEqual(len(refined["model_state"]["cell_indices"]), 3)
        with self.assertRaisesRegex(RuntimeError, "max_active_voxels"):
            refine_state(checkpoint, CoarseToFineConfig(), max_active_voxels=2)


if __name__ == "__main__":
    unittest.main()
