"""Small CUDA optimizer and CPU topology checks against Falcor update rules.

Uses the source build, no external data and no training-image render. NumPy
references check parameter constraints, Adam moments, deletion voting, growth
evidence and exact geometric containment across a coarse-to-fine split.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from reconstruction_common import WORKSPACE, initialize_runtime, restore_model, workspace_path
from preview import RuntimeConfig
from reconstruction_optimizer import FalcorOptimizer, OptimizerConfig
from reconstruction_topology import (TopologyManager, TopologySettings, CoarseToFineConfig,
    level_budgets, model_checkpoint, refine_state, _geometry, _coordinates, _contact,
    _matching_growth_pairs)


def seed_checkpoint(cells=(73, 75), resolution=8, axes=(.8, .7, .6), sh=4):
    n = len(cells)
    return {"model_metadata": {"resolution": resolution, "grid_min": [0., 0., 0.],
        "grid_max": [float(resolution)] * 3, "sh_count": sh, "opacity_sh_count": sh},
        "model_state": {
            "cell_indices": np.asarray(cells, dtype=np.uint32),
            "centers": np.tile([.5, .5, .5], n).astype(np.float32),
            "log_scales": np.tile(np.log(axes), n).astype(np.float32),
            "rotations": np.tile([1., 0., 0., 0.], n).astype(np.float32),
            "radiance_sh": np.arange(n * sh * 3, dtype=np.float32) * .01,
            "opacity_sh": np.zeros(n * sh, dtype=np.float32)}}


def result_for(model, value=1.):
    n = int(model.active_voxel_count)
    return {"gradients": {name: mi.Float(np.full(n * size, value, dtype=np.float32))
        for name, size in (("centers", 3), ("log_scales", 3), ("rotation_tangent", 3),
            ("radiance_sh", int(model.sh_count) * 3), ("opacity_sh", int(model.opacity_sh_count)))},
        "appearance_valid": mi.UInt32(np.ones(n, dtype=np.uint32)),
        "geometry_valid": mi.UInt32(np.ones(n, dtype=np.uint32)),
        "background_carve_valid": mi.UInt32(np.zeros(n, dtype=np.uint32)),
        "foreground_evidence": mi.UInt32(np.zeros(n, dtype=np.uint32)),
        "background_evidence": mi.UInt32(np.zeros(n, dtype=np.uint32))}


def oracle_adam(gradient, mean, squared_mean, steps, valid, dimension, lr, cfg, geometry=False, scale=None):
    """Scalar-loop implementation of Shader/UpdatePass.cs.slang."""
    output = np.zeros_like(gradient, dtype=np.float64)
    square = squared_mean
    for voxel in range(len(valid)):
        if not valid[voxel] or lr <= 0:
            continue
        steps[voxel] += 1
        for axis in range(dimension):
            j = voxel * dimension + axis
            g = float(gradient[j]) if math.isfinite(float(gradient[j])) else 0.
            clipped = min(max(g, -cfg.gradient_clip), cfg.gradient_clip)
            mean[j] = cfg.beta1 * mean[j] + (1 - cfg.beta1) * clipped
            square[j] = cfg.beta2 * square[j] + (1 - cfg.beta2) * clipped * clipped
            step = lr * mean[j] / max(1 - cfg.beta1 ** int(steps[voxel]), 1e-8)
            step /= math.sqrt(max(square[j] / max(1 - cfg.beta2 ** int(steps[voxel]), 1e-8), 0.)) + cfg.epsilon
            if geometry and cfg.geometry_direction_guard and (g == 0 or mean[j] * g < 0):
                mean[j], step = 0., 0.
            output[j] = step * (1. if scale is None else scale[voxel])
    return output


class OptimizationTests(unittest.TestCase):
    def test_cuda_adam_matches_scalar_oracle_and_group_counters(self):
        model = restore_model(seed_checkpoint(), mi)
        cfg = OptimizerConfig()
        opt = FalcorOptimizer(model, cfg)
        dimensions = opt.dimensions
        oracle = {name: {"mean": np.zeros(2 * size), "squared_mean": np.zeros(2 * size),
                         "steps": np.zeros(2, dtype=np.uint32)} for name, size in dimensions.items()}
        for iteration in range(3):
            old = model_checkpoint(model)["model_state"]
            out = result_for(model)
            out["appearance_valid"] = mi.UInt32([1, 0 if iteration < 2 else 1])
            out["geometry_valid"] = mi.UInt32([2, 0 if iteration < 2 else 1])
            out["background_carve_valid"] = mi.UInt32([1, 0])
            for name, size in dimensions.items():
                g = np.linspace(-12., 12., 2 * size, dtype=np.float32)
                if iteration == 1:
                    g[:] = -g * .001
                    g[0] = 0.
                if iteration == 2:
                    g[0] = np.nan
                out["gradients"][name] = mi.Float(g)
            opt.step(model, out, sample_count=2, epoch=iteration, coarse_to_fine=True)
            got = model_checkpoint(model)["model_state"]
            state = opt.state_dict()["groups"]
            for name, size in dimensions.items():
                geom = name in ("centers", "log_scales", "rotation_tangent")
                valid = np.array([True, iteration == 2])
                lr = {"centers": cfg.lr_center, "log_scales": cfg.lr_log_scale,
                      "rotation_tangent": cfg.lr_rotation, "radiance_sh": cfg.lr_radiance,
                      "opacity_sh": cfg.lr_opacity}[name]
                step = oracle_adam(np.array(out["gradients"][name]) / 2, **oracle[name],
                    valid=valid, dimension=size, lr=lr, cfg=cfg, geometry=geom,
                    scale=np.array([3., 1.]) if geom else None)
                for field in ("mean", "squared_mean", "steps"):
                    np.testing.assert_allclose(state[name][field], oracle[name][field], rtol=3e-5, atol=2e-7,
                        err_msg=f"{name}.{field}, step {iteration}")
                if name != "rotation_tangent":
                    limit = {"centers": .02, "log_scales": .02, "radiance_sh": .05, "opacity_sh": .01}[name]
                    expected = old[name] - np.clip(step, -limit, limit)
                    if name == "centers":
                        expected = np.clip(expected, .01, .99)
                    if name == "log_scales":
                        expected = np.clip(expected, math.log(1e-4), math.log(4))
                    np.testing.assert_allclose(got[name], expected, rtol=2e-5, atol=2e-7)
            np.testing.assert_allclose(np.linalg.norm(got["rotations"].reshape(-1, 4), axis=1), 1., atol=2e-6)

    def test_constraints_and_local_tangent_quaternion(self):
        checkpoint = seed_checkpoint(cells=(73,), axes=(3.999, .0001001, .8))
        checkpoint["model_state"]["centers"][:] = [.015, .985, .5]
        model = restore_model(checkpoint, mi)
        cfg = OptimizerConfig(lr_center=1, lr_log_scale=1, lr_rotation=1, lr_radiance=1, lr_opacity=1)
        opt = FalcorOptimizer(model, cfg)
        out = result_for(model)
        out["gradients"]["centers"] = mi.Float([1, -1, 0])
        out["gradients"]["log_scales"] = mi.Float([-1, 1, 0])
        old = model_checkpoint(model)["model_state"]
        opt.step(model, out)
        got = model_checkpoint(model)["model_state"]
        np.testing.assert_allclose(got["centers"], [.01, .99, .5], atol=1e-7)
        np.testing.assert_allclose(got["log_scales"][:2], [math.log(4), math.log(1e-4)], atol=1e-6)
        np.testing.assert_allclose(old["radiance_sh"] - got["radiance_sh"], .05, atol=1e-7)
        np.testing.assert_allclose(old["opacity_sh"] - got["opacity_sh"], .01, atol=1e-7)
        expected_q = [math.cos(.01)] + [-math.sin(.01) / math.sqrt(3)] * 3
        np.testing.assert_allclose(got["rotations"], expected_q, atol=1e-7)

    def test_opacity_schedule_no_age_when_disabled(self):
        model = restore_model(seed_checkpoint(), mi)
        opt = FalcorOptimizer(model)
        for epoch, expected in ((0, 0), (19, 0), (20, 0), (35, .00025), (50, .0005)):
            self.assertAlmostEqual(opt.effective_opacity_lr(epoch, False), expected)
        self.assertEqual(opt.effective_opacity_lr(0, True), .0005)
        opt.step(model, result_for(model), epoch=0, coarse_to_fine=False)
        np.testing.assert_array_equal(opt.state_dict()["groups"]["opacity_sh"]["steps"], 0)

    def test_state_remap_and_roundtrip(self):
        model = restore_model(seed_checkpoint(), mi)
        opt = FalcorOptimizer(model)
        opt.step(model, result_for(model))
        old = opt.state_dict()
        opt.remap(np.array([1], dtype=np.uint32), 3, reset_radiance=True)
        after = opt.state_dict()
        for name, width in opt.dimensions.items():
            for field in ("mean", "squared_mean", "steps"):
                size = 1 if field == "steps" else width
                expected = np.zeros(3 * size)
                if name != "radiance_sh":
                    expected[:size] = old["groups"][name][field][size:2 * size]
                np.testing.assert_array_equal(after["groups"][name][field], expected)
        opt.load_state_dict(after)
        opt.remap(np.empty(0, dtype=np.uint32), 5)
        for group in opt.state_dict()["groups"].values():
            for value in group.values():
                np.testing.assert_array_equal(value, 0)

    def test_empty_native_model_update_and_refinement(self):
        checkpoint = seed_checkpoint(cells=())
        model = restore_model(checkpoint, mi)
        self.assertEqual(model.active_voxel_count, 0)
        optimizer = FalcorOptimizer(model)
        ray = mi.Ray3f(mi.Point3f(0., 0., -1.), mi.Vector3f(0., 0., 1.))
        ray.maxt = 100.
        rgb, alpha, hits = model.trace(ray, mi.Color3f(.2, .3, .4))
        dr.eval(rgb, alpha, hits)
        np.testing.assert_allclose(np.array(rgb).ravel(), [.2, .3, .4], atol=1e-7)
        np.testing.assert_array_equal(np.array(alpha), 0)
        np.testing.assert_array_equal(np.array(hits), 0)
        out = model.loss_and_grad(ray, mi.Color3f(0.), mi.Float(0.), mi.Color3f(1.))
        optimizer.step(model, out)
        for values in out["gradients"].values():
            self.assertEqual(dr.width(values), 0)
        manager = TopologyManager(model, 1, schedule=CoarseToFineConfig(start_resolution=8, target_resolution=16, total_epochs=2))
        replacement, event = manager.finish_epoch(model, optimizer, 1)
        self.assertTrue(event["refined"])
        self.assertEqual(replacement["model_metadata"]["resolution"], 16)
        self.assertEqual(restore_model(replacement, mi).active_voxel_count, 0)


class TopologyCacheTests(unittest.TestCase):
    """CPU-only checks; also included in the complete CUDA/topology suite."""

    def manager(self):
        from types import SimpleNamespace
        return TopologyManager(SimpleNamespace(resolution=8, active_voxel_count=2), 8,
            TopologySettings(enable_deletion=False, growth_interval=1),
            CoarseToFineConfig(enabled=False, total_epochs=30), 0, 0)

    def test_frozen_geometry_cache_preserves_view_votes_and_checkpoint_resume(self):
        import copy
        manager, uncached = self.manager(), self.manager()
        checkpoint = seed_checkpoint(axes=(.9, .9, .9))
        metadata = checkpoint["model_metadata"]
        manager.pending_growth = manager._generate_growth(checkpoint, 1)
        uncached.pending_growth = copy.deepcopy(manager.pending_growth)
        self.assertGreater(len(manager.pending_growth["cells"]), 0)
        camera = dict(position=[4., 4., -10.], right=[1., 0., 0.], up=[0., 1., 0.],
                      forward=[0., 0., 1.], camera_angle_x=1.2, near_clip=.1, far_clip=100.)
        rng = np.random.default_rng(37)
        reference = rng.choice([0., .5, 1.], (32, 32))
        points = None
        for view in range(8):
            camera["position"][0] += .2
            rendered = rng.uniform(0., 1., (32, 32))
            if view == 3:
                state = manager.state_dict()
                self.assertNotIn("_growth_world_cache", state)
                self.assertNotIn("world_points", state["pending_growth"])
                manager = self.manager()
                manager.load_state_dict(state)
                self.assertIsNone(manager._growth_world_cache)
                points = None
            uncached._growth_world_cache = None
            for target in (manager, uncached):
                target._observe_growth(camera, reference, rendered, view, metadata)
            current_points = manager._growth_world_points(manager.pending_growth, metadata)
            if points is not None:
                self.assertIs(current_points, points)
            points = current_points
            for actual, expected in zip(points, uncached._growth_world_cache[2]):
                np.testing.assert_array_equal(actual, expected)
                self.assertFalse(actual.flags.writeable)
            for field in ("foreground", "background"):
                np.testing.assert_array_equal(manager.pending_growth[field], uncached.pending_growth[field])
        state = manager.state_dict()
        manager._observe_growth(camera, reference, rendered, 0, metadata)
        for field in ("foreground", "background"):
            np.testing.assert_array_equal(manager.pending_growth[field], state["pending_growth"][field])
        # A new frozen batch and grid bounds must never reuse previous geometry.
        old_points = points
        manager.pending_growth = copy.deepcopy(manager.pending_growth)
        points = manager._growth_world_points(manager.pending_growth, metadata)
        self.assertIsNot(points, old_points)
        shifted = {**metadata, "grid_min": [1., 1., 1.], "grid_max": [9., 9., 9.]}
        new_points = manager._growth_world_points(manager.pending_growth, shifted)
        self.assertIsNot(new_points, points)
        for actual, expected in zip(new_points, points):
            np.testing.assert_allclose(actual, expected + 1., rtol=0., atol=2e-15)

    def test_growth_pair_membership_preserves_order_duplicates_and_empty_batches(self):
        rng = np.random.default_rng(19)
        pairs = rng.integers(0, 2**32, (2000, 2), dtype=np.uint64)
        pairs[:4] = [(0, 2**32 - 1), (2**32 - 1, 0), (2**32 - 1, 2**32 - 1), (0, 0)]
        proposed = np.concatenate((pairs[::3], pairs[::-2], rng.integers(0, 2**32, (1000, 2), dtype=np.uint64)))
        def batch(values):
            return {"parent_cells": values[:, 0], "cells": values[:, 1]}
        for current, pending in ((pairs, proposed), (pairs[:0], proposed), (pairs, proposed[:0])):
            allowed = set(map(tuple, current.tolist()))
            expected = np.asarray([tuple(pair) in allowed for pair in pending.tolist()], dtype=bool)
            np.testing.assert_array_equal(_matching_growth_pairs(batch(current), batch(pending), np), expected)


class TopologyTests(unittest.TestCase):
    def manager(self, model, **overrides):
        cfg = TopologySettings(**overrides)
        return TopologyManager(model, 8, cfg, CoarseToFineConfig(enabled=False, total_epochs=30), 0, 0)

    def test_level_budgets_exact_integer_totals(self):
        self.assertEqual(level_budgets(256, 512, 1000), {256: 600, 512: 400})
        self.assertEqual(level_budgets(256, 512, 1000, (19, 12, 14, 25)), {256: 431, 512: 569})
        for start in (1, 16, 128, 256, 512):
            budgets = level_budgets(start, 512, 1000, (19, 12, 14, 25))
            self.assertEqual(sum(budgets.values()), 1000)
            self.assertGreaterEqual(min(budgets.values()), 1)

    def test_deletion_unique_views_two_windows_and_moment_remap(self):
        model = restore_model(seed_checkpoint(), mi)
        opt = FalcorOptimizer(model)
        opt.step(model, result_for(model))
        manager = self.manager(model, enable_growth=False)
        out = result_for(model)
        out["background_evidence"] = mi.UInt32([1, 0])
        kwargs = dict(camera={}, reference_alpha=None, rendered_alpha=None)
        # Repeated pixels/visits from one camera count only once.
        for epoch in range(5):
            manager.observe_view(model, out, view_id=0, epoch=epoch, **kwargs)
        self.assertEqual(int(manager.background_masks[0, 0]), 1)
        for view in range(1, 5):
            manager.observe_view(model, out, view_id=view, epoch=4, **kwargs)
        replacement, event = manager.finish_epoch(model, opt, 5)
        self.assertIsNone(replacement)
        self.assertTrue(manager.qualified[0])
        self.assertFalse(manager.candidates[0])
        for view in range(5):
            manager.observe_view(model, out, view_id=view, epoch=9, **kwargs)
        replacement, event = manager.finish_epoch(model, opt, 10)
        self.assertEqual(event["deleted"], 1)
        np.testing.assert_array_equal(replacement["model_state"]["cell_indices"], [75])
        self.assertEqual(manager.cooldown[73], 15)
        self.assertEqual(opt.count, 1)
        np.testing.assert_array_equal(opt.state_dict()["groups"]["radiance_sh"]["steps"], [0])
        np.testing.assert_array_equal(opt.state_dict()["groups"]["centers"]["steps"], [1])
        json.dumps({k: v for k, v in manager.state_dict().items() if isinstance(v, dict)})
        restored = TopologyManager(restore_model(replacement, mi), 8)
        restored.load_state_dict(manager.state_dict())
        self.assertEqual(restored.cooldown, manager.cooldown)

    def test_reliable_foreground_resets_confirmed_candidate(self):
        model = restore_model(seed_checkpoint(), mi)
        manager = self.manager(model, enable_growth=False)
        manager.qualified[0] = manager.candidates[0] = True
        manager.foreground_masks[0, 0] = 0b111
        manager.background_masks[0, 0] = 0b1
        manager._evaluate_deletion(5)
        self.assertFalse(manager.qualified[0])
        self.assertFalse(manager.candidates[0])

    def test_all_deleted_remains_empty_and_does_not_regrow(self):
        model = restore_model(seed_checkpoint(), mi)
        optimizer = FalcorOptimizer(model)
        manager = self.manager(model, growth_interval=1)
        manager.candidates[:] = True
        replacement, event = manager.finish_epoch(model, optimizer, 10)
        self.assertEqual(event["deleted"], 2)
        self.assertEqual(event["grown"], 0)
        self.assertEqual(manager.count, 0)
        self.assertEqual(optimizer.count, 0)
        self.assertEqual(restore_model(replacement, mi).active_voxel_count, 0)

    def test_refinement_preserves_parent_volume_and_dc_attenuation(self):
        checkpoint = seed_checkpoint(cells=(73,), axes=(.8, .35, .22))
        angle = .7
        checkpoint["model_state"]["rotations"] = np.array([math.cos(angle / 2), 0, 0, math.sin(angle / 2)], dtype=np.float32)
        checkpoint["model_state"]["opacity_sh"][:] = [2., .4, -.3, .2]
        refined = refine_state(checkpoint, CoarseToFineConfig())
        state, meta = refined["model_state"], refined["model_metadata"]
        self.assertEqual(meta["resolution"], 16)
        n = len(state["cell_indices"])
        self.assertGreater(n, 0)
        self.assertEqual(len(np.unique(state["cell_indices"])), n)
        center, axes, rotation, _, size, metric, extent = _geometry(checkpoint["model_state"], checkpoint["model_metadata"], np)
        new_center, new_axes, new_rotation, _, fine_size, _, _ = _geometry(state, meta, np)
        local = _coordinates(state["cell_indices"], 16, np) % 2
        relative = (local + new_center) * .5 - center
        ratio = new_axes / axes
        np.testing.assert_allclose(ratio[:, 0], ratio[:, 1], rtol=1e-6)
        np.testing.assert_allclose(ratio[:, 0], ratio[:, 2], rtol=1e-6)
        unit_distance = np.sqrt(np.einsum("ni,ij,nj->n", relative, metric[0], relative))
        self.assertTrue(np.all(unit_distance + ratio[:, 0] <= 1. + 1e-6))
        self.assertTrue(np.all((new_center >= .00999) & (new_center <= .99001)))
        new_radiance = state["radiance_sh"].reshape(n, 4, 3)
        np.testing.assert_allclose(new_radiance[:, 0], np.tile(checkpoint["model_state"]["radiance_sh"][:3], (n, 1)))
        np.testing.assert_array_equal(new_radiance[:, 1:], 0.)
        new_opacity = state["opacity_sh"].reshape(n, 4)
        np.testing.assert_array_equal(new_opacity[:, 1:], 0.)
        y0 = 1 / (2 * math.sqrt(math.pi))
        np.testing.assert_allclose(np.logaddexp(0., new_opacity[:, 0] * y0), .65 * np.logaddexp(0., 2 * y0), rtol=1e-6)

    def test_coarse_boundary_snapshots_before_reset(self):
        model = restore_model(seed_checkpoint(), mi)
        opt = FalcorOptimizer(model)
        opt.step(model, result_for(model))
        manager = TopologyManager(model, 8, schedule=CoarseToFineConfig(start_resolution=8, target_resolution=16, total_epochs=2))
        replacement, event = manager.finish_epoch(model, opt, 1)
        self.assertTrue(event["refined"])
        self.assertEqual(manager.before_refine_checkpoint["model_metadata"]["resolution"], 8)
        self.assertEqual(manager.before_refine_checkpoint["topology_state"]["resolution"], 8)
        self.assertEqual(replacement["model_metadata"]["resolution"], 16)
        json.dumps(event)
        for group in opt.state_dict()["groups"].values():
            for values in group.values():
                np.testing.assert_array_equal(values, 0)

    def test_growth_evidence_frozen_batch_votes_and_newborn_states(self):
        model = restore_model(seed_checkpoint(axes=(.9, .9, .9)), mi)
        opt = FalcorOptimizer(model)
        manager = self.manager(model, enable_deletion=False, growth_interval=1)
        out = result_for(model)
        camera = dict(position=[4., 4., -10.], right=[1., 0., 0.], up=[0., 1., 0.],
                      forward=[0., 0., 1.], camera_angle_x=1.2, near_clip=.1, far_clip=100.)
        reference, rendered = np.ones((32, 32)), np.zeros((32, 32))
        for view in range(8):
            manager.observe_view(model, out, view_id=view, epoch=0, camera=camera,
                                 reference_alpha=reference, rendered_alpha=rendered)
        pending = manager.pending_growth
        self.assertGreater(len(pending["cells"]), 0)
        np.testing.assert_array_equal(pending["foreground"], 8)
        manager.observe_view(model, out, view_id=0, epoch=0, camera=camera,
                             reference_alpha=reference, rendered_alpha=rendered)
        np.testing.assert_array_equal(pending["foreground"], 8)
        replacement, event = manager.finish_epoch(model, opt, 1)
        self.assertGreater(event["grown"], 0)
        self.assertEqual(len(np.unique(replacement["model_state"]["cell_indices"])), manager.count)
        np.testing.assert_array_equal(manager.deletion_eligible[2:], 6)
        np.testing.assert_array_equal(manager.growth_wait_start[2:], 2)
        np.testing.assert_array_equal(replacement["model_state"]["opacity_sh"].reshape(-1, 4)[2:], 0)
        for group in opt.state_dict()["groups"].values():
            np.testing.assert_array_equal(group["steps"], 0)

    def test_growth_background_veto_and_empty_frozen_batch(self):
        model = restore_model(seed_checkpoint(axes=(.1, .1, .1)), mi)
        manager = self.manager(model, enable_deletion=False, growth_interval=1)
        out = result_for(model)
        camera = dict(position=[4., 4., -10.], right=[1., 0., 0.], up=[0., 1., 0.],
                      forward=[0., 0., 1.], camera_angle_x=1.2)
        reference = np.ones((32, 32))
        manager.observe_view(model, out, view_id=0, epoch=0, camera=camera,
                             reference_alpha=reference, rendered_alpha=reference * 0)
        self.assertEqual(len(manager.pending_growth["cells"]), 0)
        frozen = manager.pending_growth
        model.set_parameters({"log_scales": mi.Float(np.tile(np.log([.9] * 3), 2))})
        manager.observe_view(model, out, view_id=1, epoch=0, camera=camera,
                             reference_alpha=reference, rendered_alpha=reference * 0)
        self.assertIs(manager.pending_growth, frozen)
        self.assertEqual(len(frozen["cells"]), 0)
        manager.pending_growth = None
        for view in range(8):
            manager.observe_view(model, out, view_id=view, epoch=1, camera=camera,
                reference_alpha=reference * (0 if view < 2 else 1), rendered_alpha=reference * 0)
        pending = manager.pending_growth
        np.testing.assert_array_equal(pending["background"], 2)
        np.testing.assert_array_equal(pending["foreground"], 6)
        replacement, event = manager.finish_epoch(model, FalcorOptimizer(model), 2)
        self.assertIsNone(replacement)
        self.assertEqual(event["grown"], 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", default=RuntimeConfig().build_dir, help="Relative to workspace parent")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output-dir", default="outputs/mitsuba/optimization-tests")
    args = parser.parse_args()
    output = workspace_path(args.output_dir, WORKSPACE)
    output.mkdir(parents=True, exist_ok=True)
    report = {"scope": "CUDA Adam/state tests and CPU topology rules; no image training or real Slurm validation"}
    global mi, dr, np
    mi, dr, np, handles = initialize_runtime(RuntimeConfig(build_dir=args.build_dir, gpu=args.gpu), output, report)
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    dr.sync_thread()
    report.update(tests=result.testsRun, failures=len(result.failures), errors=len(result.errors), success=result.wasSuccessful())
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
