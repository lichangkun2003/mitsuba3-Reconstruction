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
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import preview


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

    def test_legacy_initialization_keeps_saved_background(self):
        config = preview.PreviewConfig()
        config.data.initialization = {"version": 1, "kind": "colmap"}
        config.render.background = (.1, .2, .3)
        self.assertEqual(preview.resolve_scene_background(config, "colmap"), "legacy")
        self.assertEqual(config.render.background, (.1, .2, .3))
        for mode in ("bounded", "unbounded"):
            config.data.scene_mode = mode
            with self.assertRaisesRegex(ValueError, "Legacy version-1"):
                preview.resolve_scene_background(config, "colmap")

    def test_saved_scene_mode_is_authoritative_and_conflicts_rejected(self):
        config = preview.PreviewConfig()
        config.data.initialization = {"version": 2, "scene_mode": "bounded"}
        self.assertEqual(preview.resolve_scene_background(config, "colmap"), "bounded")
        config.data.scene_mode = "unbounded"
        with self.assertRaisesRegex(ValueError, "disagrees"):
            preview.resolve_scene_background(config, "colmap")
        config.data.scene_mode = "auto"
        config.data.initialization = {"version": 2}
        with self.assertRaisesRegex(ValueError, "scene_mode"):
            preview.resolve_scene_background(config, "colmap")

    def test_invalid_background_settings_fail(self):
        for name, values in {
                "scene_mode": [None, "outside"],
                "background_layers": [0, -1, True, 1.5],
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
        self.assertEqual(config.data.background_layers, 4)
        preview.validate_config(config)
        config.render.max_hits = 32
        self.assertEqual(config.data.background_layers, 4)
        preview.validate_config(config)

    def test_old_resume_configuration_gets_equivalent_defaults(self):
        import train
        old = {"dataset": "datasets/lego", "reference_compositing": "linear"}
        new = {**old, "scene_mode": "auto", "background_layers": 8, "background_opacity": .5}
        self.assertEqual(train.normalized_algorithm_group("data", old),
                         train.normalized_algorithm_group("data", new))
        changed = {**new, "background_opacity": .25}
        self.assertNotEqual(train.normalized_algorithm_group("data", old),
                            train.normalized_algorithm_group("data", changed))

    def test_training_cli_keeps_requested_initialization_settings(self):
        import train
        with patch.object(sys, "argv", ["train.py", "--scene-mode", "unbounded",
                         "--background-layers", "8", "--background-opacity", "0.5"]):
            args = train.parse_arguments()
        config = train.TrainConfig()
        train.apply_arguments(config, args, asdict(train.CONFIG))
        self.assertEqual(config.data.scene_mode, "unbounded")
        self.assertEqual(config.data.background_layers, 8)
        self.assertEqual(config.data.background_opacity, .5)

    def test_evaluation_keeps_saved_scene_configuration(self):
        import evaluate
        saved = {"data": {"scene_mode": "unbounded", "background_layers": 8,
                         "background_opacity": .5,
                         "initialization": {"version": 2, "scene_mode": "unbounded"}}}
        data = evaluate.checkpoint_data_config(saved)
        render = preview.RenderConfig(background=(0., 0., 0.))
        report = preview.scene_background_protocol(data, render)
        self.assertEqual(report["scene_mode"], "unbounded")
        self.assertEqual(report["background_linear_rgb"], [0., 0., 0.])
        self.assertEqual(report["initial_background_layers"], 8)
        self.assertEqual(report["initial_background_opacity"], .5)
        self.assertIn("no environment illumination", report["background_role"])


def shell_fixture(mi, np):
    """One unchanged interior seed and one colored far-field seed."""
    y00 = 1. / (2. * math.sqrt(math.pi))
    core_cell, far_cell = 4 + 8*4 + 64*4, 7 + 8*4 + 64*4
    colors = np.zeros((2, 4, 3), dtype=np.float32)
    colors[:, 0] = np.asarray([[.3, .6, .8], [.7, .4, .2]]) / y00
    colors[:, 1:] = np.asarray([[[.01, .02, .03]] * 3, [[.03, .02, .01]] * 3])
    state = {
        "cell_indices": np.asarray([core_cell, far_cell], dtype=np.uint32),
        "centers": np.asarray([[.55, .45, .65], [.6, .4, .7]], dtype=np.float32).ravel(),
        "log_scales": np.log(np.asarray([[.31, .32, .33], [.3, .25, .4]], dtype=np.float32)).ravel(),
        "rotations": np.asarray([[1., 0., 0., 0.]] * 2, dtype=np.float32).ravel(),
        "radiance_sh": colors.ravel(),
        "opacity_sh": np.asarray([math.log(.23/.77)/y00, math.log(.8/.2)/y00], dtype=np.float32),
    }
    device = {name: mi.UInt32(value) if name == "cell_indices" else mi.Float(value)
              for name, value in state.items()}
    model = mi.VoxelReconstruction.from_state(8, [-2.] * 3, [2.] * 3, device, 4, 1,
                                             core_resolution=4)
    return model, state


def native_background_checks(mi, dr, np):
    from reconstruction_background import initialize_background_model
    from reconstruction_common import model_metadata, restore_model
    from test_reconstruction import native_trace, oracle, unpack_snapshot

    seed, seed_state = shell_fixture(mi, np)
    model = initialize_background_model(seed, mi, dr, np, opacity=.5)
    self_metadata = model_metadata(model)
    assert self_metadata["scene_mapping"] == "radial_seed_shell"
    state = {name: np.asarray(value).copy() for name, value in model.snapshot().items()}
    indices = state["cell_indices"]
    all_ids = np.arange(8**3, dtype=np.uint32)
    xyz = np.stack((all_ids % 8, all_ids // 8 % 8, all_ids // 64), axis=1)
    # Brute-force AABB-to-sphere distances; independent of production integer
    # coordinate masks. A leaf is occupied if its volume overlaps 1 < r < 2.
    lower, upper = -2. + xyz*.5, -2. + (xyz+1)*.5
    nearest = np.maximum(np.maximum(lower, -upper), 0.)
    farthest = np.maximum(np.abs(lower), np.abs(upper))
    expected_far = all_ids[(np.sum(nearest**2, axis=1) < 4.) &
                          (np.sum(farthest**2, axis=1) > 1.)]
    expected_ids = np.union1d(expected_far, seed_state["cell_indices"])
    np.testing.assert_array_equal(np.sort(indices), expected_ids)
    assert len(np.unique(indices)) == len(indices)
    rows = {int(cell): row for row, cell in enumerate(indices)}
    is_far = np.isin(indices, expected_far)
    for name, width in (("centers", 3), ("log_scales", 3), ("rotations", 4),
                        ("radiance_sh", 12), ("opacity_sh", 1)):
        np.testing.assert_array_equal(state[name].reshape(-1, width)[rows[int(seed_state["cell_indices"][0])]],
                                      seed_state[name].reshape(-1, width)[0], err_msg=f"core {name}")
    np.testing.assert_array_equal(state["centers"].reshape(-1, 3)[is_far], .5)
    np.testing.assert_array_equal(state["opacity_sh"][is_far], 0.)
    scales = np.exp(state["log_scales"].reshape(-1, 3)[is_far])
    assert np.all(scales >= math.sqrt(3.)*.25), "Far ellipsoids must cover their cell corners"
    np.testing.assert_allclose(scales, math.sqrt(3.)*.25, rtol=3e-6)
    colors = state["radiance_sh"].reshape(-1, 12)
    far_seed_row = rows[int(seed_state["cell_indices"][1])]
    np.testing.assert_array_equal(colors[far_seed_row], seed_state["radiance_sh"].reshape(-1, 12)[1])
    np.testing.assert_array_equal(colors[is_far & (indices != seed_state["cell_indices"][1])], 0.)

    # Marked models and ordinary models reconstructed from identical buffers
    # must take exactly the same straight rays: contraction is seed-only.
    checkpoint = {"model_metadata": self_metadata, "model_state": state}
    restored = restore_model(checkpoint, mi)
    assert restored.scene_mapping == "radial_seed_shell"
    bounded_checkpoint = copy.deepcopy(checkpoint)
    bounded_checkpoint["model_metadata"].pop("scene_mapping")
    bounded = restore_model(bounded_checkpoint, mi)
    origins = np.asarray([[.125, .125, .125], [-.6, .125, .125], [.13, -.11, .07], [.2, -.9, .1]], dtype=np.float32)
    directions = np.asarray([[1., 0., 0.], [1., 0., 0.], [.7, .3, -.2], [.1, .9, .2]], dtype=np.float32)
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    maxt = np.asarray([np.inf, np.inf, 8., np.inf], dtype=np.float32)
    expected_state = unpack_snapshot(np, model, 8, [-2.] * 3, [2.] * 3)
    reference = oracle(np, expected_state, origins, directions, maxt, np.zeros(3), .00001, 32)
    actual = native_trace(np, mi, dr, model, origins, directions, maxt, np.zeros(3), .00001, 32)
    for actual_value, expected_value in zip(actual, reference):
        np.testing.assert_allclose(actual_value, expected_value, atol=3e-6, rtol=3e-6)
    for other in (restored, bounded):
        compared = native_trace(np, mi, dr, other, origins, directions, maxt, np.zeros(3), .00001, 32)
        for a, b in zip(actual, compared):
            np.testing.assert_array_equal(a, b)
    assert np.all(actual[2] > 0)

    # A deleted corridor really exposes a black residual, even when alpha is
    # zero. There is no fixed opaque terminal shell or implicit white skylight.
    keep = ~((indices // 8 % 8 == 4) & (indices // 64 == 4))
    deleted = {name: value.reshape(len(indices), -1)[keep].ravel() for name, value in state.items()}
    empty_ray_model = restore_model({"model_metadata": self_metadata, "model_state": deleted}, mi)
    for background, expected in ((np.zeros(3), 0.), (np.ones(3), 1.)):
        rgb, alpha, hits = native_trace(np, mi, dr, empty_ray_model, origins[:1], directions[:1], maxt[:1],
                                       background, .00001, 32)
        np.testing.assert_array_equal(rgb, expected)
        np.testing.assert_array_equal(alpha, 0.)
        np.testing.assert_array_equal(hits, 0)
    return {"complete_radial_shell_cells": len(expected_far), "shell_opacity": .5,
            "core_preserved": True, "seed_color_inheritance": True,
            "straight_ray_forward_unchanged": True, "saved_mapping_restored": True,
            "deleted_corridor_black_background": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--build-dir")
    parser.add_argument("--gpu-index", type=int, default=0)
    args = parser.parse_args()
    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(BackgroundConfigTests))
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
