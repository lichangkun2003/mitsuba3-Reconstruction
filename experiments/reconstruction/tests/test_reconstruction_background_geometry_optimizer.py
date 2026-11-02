"""CPU execution of Python Adam fallbacks with occupied-cell background geometry."""

from __future__ import annotations

import copy
from collections import OrderedDict
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reconstruction_optimizer import FalcorOptimizer, OptimizerConfig


class NumPyDr:
    """Only the array operations used by the two Python fallback paths."""

    width = staticmethod(np.size)
    arange = staticmethod(lambda dtype, count: np.arange(count, dtype=dtype))
    zeros = staticmethod(lambda dtype, count: np.zeros(count, dtype=dtype))
    full = staticmethod(lambda dtype, value, count: np.full(count, value, dtype=dtype))
    gather = staticmethod(lambda dtype, source, index: np.asarray(source)[index].copy())
    compress = staticmethod(lambda value: np.flatnonzero(value).astype(np.uint32))
    select = staticmethod(np.where)
    clip = staticmethod(np.clip)
    maximum = staticmethod(np.maximum)
    minimum = staticmethod(np.minimum)
    isfinite = staticmethod(np.isfinite)
    square = staticmethod(np.square)
    power = staticmethod(np.power)
    sqrt = staticmethod(np.sqrt)
    rsqrt = staticmethod(lambda value: 1 / np.sqrt(value))
    log = staticmethod(np.log)
    sin = staticmethod(np.sin)
    cos = staticmethod(np.cos)
    all = staticmethod(np.all)
    norm = staticmethod(lambda value: np.linalg.norm(value, axis=0))
    dot = staticmethod(lambda first, second: np.sum(first * second, axis=0))
    cross = staticmethod(lambda first, second: np.cross(first.T, second.T).T)
    is_array_v = staticmethod(lambda kind: issubclass(kind, np.ndarray))
    eval = staticmethod(lambda *args: None)

    @staticmethod
    def scatter(target, source, index, active=True):
        index, source = np.broadcast_arrays(index, source)
        active = np.broadcast_to(active, index.shape)
        target[index[active]] = source[active]


class Model:
    resolution = 8
    core_resolution = 4
    active_voxel_count = 7
    sh_count = 4
    active_sh_count = 4
    opacity_sh_count = 1
    voxel_size = (1., 1., 1.)

    def __init__(self, use_ellipsoid, mapping="projective_shell"):
        self.scene_mapping = mapping
        self.background_use_ellipsoid = use_ellipsoid
        self.native_use_ellipsoid = use_ellipsoid
        # One core and all six background faces; geometry must use cell identity.
        xyz = np.array([[3, 3, 3], [1, 3, 3], [6, 3, 3], [3, 1, 3],
                        [3, 6, 3], [3, 3, 1], [3, 3, 6]], dtype=np.uint32)
        self.parameters = {
            "cell_indices": xyz[:, 0] + 8 * xyz[:, 1] + 64 * xyz[:, 2],
            "centers": np.tile([.5, .5, .5], 7).astype(np.float32),
            "log_scales": np.full(21, -.5, dtype=np.float32),
            "rotations": np.tile([1., 0., 0., 0.], 7).astype(np.float32),
            "radiance_sh": np.full(84, .3, dtype=np.float32),
            "opacity_sh": np.full(7, .2, dtype=np.float32),
        }

    def native_background_use_ellipsoid(self):
        return self.native_use_ellipsoid

    def snapshot(self):
        return {name: value.copy() for name, value in self.parameters.items()}

    def set_parameters(self, values):
        self.parameters.update({name: value.copy() for name, value in values.items()})

    def set_parameters_guarded(self, values, allowed):
        if np.all(allowed):
            self.set_parameters(values)
        return np.atleast_1d(allowed)

    def set_parameters_sparse(self, appearance, geometry, values, allowed):
        self.last_geometry_ids = geometry.copy()
        if np.all(allowed):
            for name, value in values.items():
                ids = geometry if name in ("centers", "log_scales", "rotations") else appearance
                self.parameters[name].reshape(self.active_voxel_count, -1)[ids] = value.reshape(len(ids), -1)
        return np.atleast_1d(allowed)


def optimizer_and_output(model, gradient):
    """Populate real Adam with prior momentum while avoiding native imports."""
    opt = FalcorOptimizer.__new__(FalcorOptimizer)
    opt.dr = NumPyDr
    opt.mi = SimpleNamespace(Float=np.float32, UInt32=np.uint32, Bool=np.bool_,
                             Vector3f=lambda *components: np.asarray(components, dtype=np.float32))
    opt.config = OptimizerConfig(geometry_direction_guard=False)
    opt.count = model.active_voxel_count
    opt.sh_count = opt.active_sh_count = model.sh_count
    opt.sh_degree_count = 2
    opt.radiance_degree_steps = False
    opt.update_healthy = np.atleast_1d(True)
    opt.dimensions = {"radiance_sh": 12, "opacity_sh": 1, "centers": 3,
                      "log_scales": 3, "rotation_tangent": 3}
    opt._state = {name: {"mean": np.full(opt.count * size, .2, dtype=np.float32),
                         "squared_mean": np.full(opt.count * size, .04, dtype=np.float32),
                         "steps": np.full(opt.count, 7, dtype=np.uint32)}
                  for name, size in opt.dimensions.items()}
    # Runtime signatures concern Dr.Jit ownership, unrelated to the update math.
    opt._remember_state = lambda: None
    opt._check_external_state = lambda: np.atleast_1d(True)
    out = {"gradients": {name: np.full(opt.count * size, gradient, dtype=np.float32)
                         for name, size in opt.dimensions.items()},
           "appearance_valid": np.ones(opt.count, dtype=np.uint32),
           "geometry_valid": np.ones(opt.count, dtype=np.uint32),
           "background_carve_valid": np.zeros(opt.count, dtype=np.uint32)}
    return opt, out


def step(opt, model, out, sparse, guarded=False):
    update = opt._step_compact_reference if sparse else opt._step_dense
    return update(model, out, sample_count=1, epoch=50, coarse_to_fine=True,
                  update_allowed=np.atleast_1d(True) if guarded else None)


class BackgroundGeometryOptimizerTests(unittest.TestCase):
    def test_python_fallbacks_freeze_far_parameters_and_moments_with_nonzero_geometry_valid(self):
        for sparse in (False, True):
            for gradient in (0., 1.):
                for guarded in (False, True):
                    with self.subTest(sparse=sparse, gradient=gradient, guarded=guarded):
                        model = Model(False)
                        opt, out = optimizer_and_output(model, gradient)
                        before = model.snapshot()
                        moments = copy.deepcopy(opt._state)
                        step(opt, model, out, sparse, guarded)
                        for name in ("centers", "log_scales", "rotations"):
                            old, new = before[name].reshape(7, -1), model.parameters[name].reshape(7, -1)
                            np.testing.assert_array_equal(new[1:], old[1:])
                            self.assertFalse(np.array_equal(new[0], old[0]), name)
                        for name in ("centers", "log_scales", "rotation_tangent"):
                            for field, old in moments[name].items():
                                new = opt._state[name][field].reshape(7, -1)
                                np.testing.assert_array_equal(new[1:], old.reshape(7, -1)[1:])
                            self.assertEqual(opt._state[name]["steps"][0], 8)
                        for name in ("radiance_sh", "opacity_sh"):
                            self.assertTrue(np.all(model.parameters[name] != before[name]))
                            np.testing.assert_array_equal(opt._state[name]["steps"], 8)
                        if sparse:
                            np.testing.assert_array_equal(model.last_geometry_ids, [0])

    def test_enabled_and_nonprojective_modes_keep_background_geometry_updates(self):
        for sparse in (False, True):
            for enabled, mapping in ((True, "projective_shell"), (False, "bounded"),
                                     (False, "radial_seed_shell")):
                with self.subTest(sparse=sparse, enabled=enabled, mapping=mapping):
                    model = Model(enabled, mapping)
                    opt, out = optimizer_and_output(model, 0.)
                    before = model.snapshot()
                    step(opt, model, out, sparse)
                    for name in ("centers", "log_scales", "rotations"):
                        old, new = before[name].reshape(7, -1), model.parameters[name].reshape(7, -1)
                        self.assertTrue(np.all(np.any(new != old, axis=1)), name)
                    for name in ("centers", "log_scales", "rotation_tangent"):
                        np.testing.assert_array_equal(opt._state[name]["steps"], 8)

    def test_native_getter_is_authoritative_and_legacy_default_is_ellipsoid(self):
        model = Model(False)
        model.background_use_ellipsoid = True
        opt, out = optimizer_and_output(model, 1.)
        cells = model.snapshot()["cell_indices"]
        np.testing.assert_array_equal(opt._geometry_valid(model, cells, out["geometry_valid"]),
                                      [1, 0, 0, 0, 0, 0, 0])
        legacy = SimpleNamespace(scene_mapping="projective_shell", resolution=8, core_resolution=4)
        np.testing.assert_array_equal(opt._geometry_valid(legacy, cells, out["geometry_valid"]), 1)
        legacy.background_use_ellipsoid = False
        np.testing.assert_array_equal(opt._geometry_valid(legacy, cells, out["geometry_valid"]),
                                      [1, 0, 0, 0, 0, 0, 0])

    def test_native_optimizer_mode_change_clears_both_replay_caches_on_same_model(self):
        for compact in (False, True):
            with self.subTest(compact=compact):
                model = Model(True)
                opt, out = optimizer_and_output(model, 1.)
                model.optimizer_step_device = lambda *args: np.atleast_1d(True)
                model.optimizer_step_compact_device = model.optimizer_step_device
                model.optimizer_update_scope_begin = lambda *args: None
                opt.mi._reconstruction_optimizer_update_scope_end = lambda scope: None
                opt._native_update = opt._native_compact_update = model.optimizer_step_device
                opt._state_exposed = False
                opt.replay_updates = False  # Exercise the real cache key without any GPU/JIT.
                opt._update_model = opt._update_key = None
                opt._update_cache = OrderedDict()
                opt._compact_update_cache = OrderedDict()
                opt._update_calls = opt._update_clears = opt._update_bypasses = 0
                opt._compact_calls = opt._compact_rows = 0
                if compact:
                    out["voxel_ids"] = np.arange(opt.count, dtype=np.uint32)
                    out["voxel_count"] = np.array([opt.count], dtype=np.uint32)
                previous_key = None
                for enabled, clears in ((True, 1), (True, 1), (False, 2), (False, 2), (True, 3)):
                    model.native_use_ellipsoid = enabled
                    opt._update_cache[1.] = "existing dense recording"
                    opt._compact_update_cache[1.] = "existing compact recording"
                    opt._step_sparse(model, out, sample_count=1, epoch=50,
                                     coarse_to_fine=True, update_allowed=None)
                    self.assertEqual(opt._update_clears, clears)
                    changed = previous_key != opt._update_key
                    self.assertEqual(bool(opt._update_cache), not changed)
                    self.assertEqual(bool(opt._compact_update_cache), not changed)
                    previous_key = opt._update_key


if __name__ == "__main__":
    unittest.main()
