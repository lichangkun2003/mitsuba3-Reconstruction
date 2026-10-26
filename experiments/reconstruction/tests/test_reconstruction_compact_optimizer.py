"""CUDA parity for compact communicated gradients consumed directly by Adam.

The oracle is the retained full-table optimizer. No dataset, render, NCCL, or
timing benchmark is involved; actual multi-GPU transport has separate tests.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from preview import RuntimeConfig
from reconstruction_common import initialize_runtime, restore_model
from reconstruction_optimizer import FalcorOptimizer, OptimizerConfig
from reconstruction_batch_epoch import _finite_accumulator
import test_reconstruction_optimization as original_tests


def paired(count=29, sh=9, config=None):
    resolution = 8
    while resolution ** 3 < count:
        resolution *= 2
    payload = original_tests.seed_checkpoint(cells=tuple(range(count)), resolution=resolution, sh=sh)
    payload["model_metadata"]["opacity_sh_count"] = 1
    payload["model_state"]["opacity_sh"] = np.zeros(count, np.float32)
    models = [restore_model(copy.deepcopy(payload), mi) for _ in range(2)]
    optimizers = [FalcorOptimizer(model, copy.deepcopy(config or OptimizerConfig()),
                                 sparse_updates=index == 1)
                  for index, model in enumerate(models)]
    return models, optimizers


def outputs(count, dimensions, ids, iteration=0, *, appearance=None, geometry=None, zero=False):
    """Generate both layouts from independent host reference arrays.

    Rows outside IDs are zero. Some supplied rows deliberately have nonzero
    gradients but zero counts: preserving a transport row must not update it.
    """
    ids = np.asarray(ids, np.uint32)
    k = len(ids)
    appearance = ((ids + iteration) % 3 != 0) if appearance is None else appearance
    geometry = ((ids + 2 * iteration) % 4 < 2) if geometry is None else geometry
    counts = {"appearance_valid": np.asarray(appearance, np.uint32),
              "geometry_valid": np.asarray(geometry, np.uint32) * 2,
              "background_carve_valid": (ids % 2).astype(np.uint32),
              "foreground_evidence": np.zeros(k, np.uint32),
              "background_evidence": np.zeros(k, np.uint32)}
    full, compact = {"gradients": {}}, {"gradients": {}, "voxel_ids": mi.UInt32(ids)}
    for key, value in counts.items():
        dense = np.zeros(count, np.uint32)
        dense[ids] = value
        full[key], compact[key] = mi.UInt32(dense), mi.UInt32(value)
    for group, dimension in dimensions.items():
        values = np.linspace(-13., 13., k * dimension, dtype=np.float32).reshape(k, dimension)
        values *= -0.003 if iteration % 2 else 1.
        values.ravel()[::3] = 0.
        if zero:
            values[:] = 0.
        dense = np.zeros((count, dimension), np.float32)
        dense[ids] = values
        full["gradients"][group] = mi.Float(dense.ravel())
        compact["gradients"][group] = mi.Float(values.ravel())
    return full, compact


def capture(model, optimizer):
    # state_dict() copies values without opting into the mutable-state API.
    result = {name: np.array(value) for name, value in model.snapshot().items()}
    result.update({f"{group}/{field}": value.copy()
                   for group, fields in optimizer.state_dict()["groups"].items()
                   for field, value in fields.items()})
    return result


def assert_equal(test, first, second, *, exact=False):
    test.assertEqual(first.keys(), second.keys())
    for name in first:
        if exact or first[name].dtype.kind in "iub":
            np.testing.assert_array_equal(first[name], second[name], err_msg=name)
        else:
            np.testing.assert_allclose(first[name], second[name], atol=3e-7, rtol=3e-5,
                                       equal_nan=True, err_msg=name)


class CompactOptimizerTests(unittest.TestCase):
    def assert_pair(self, models, optimizers):
        assert_equal(self, *(capture(model, opt) for model, opt in zip(models, optimizers)))

    def test_multistep_noncontiguous_rows_independent_masks_full_and_tail(self):
        models, optimizers = paired()
        selections = ([1, 5, 9, 22], list(range(29)), [14], [], [2, 6, 13, 28],
                      [0, 7, 11, 19, 27], [3, 17], list(range(0, 29, 2)))
        for iteration, ids in enumerate(selections):
            cases = outputs(29, optimizers[0].dimensions, ids, iteration, zero=iteration == 4)
            for model, optimizer, out in zip(models, optimizers, cases):
                updated = optimizer.step(model, out, sample_count=1 if iteration % 3 == 2 else 4,
                                         epoch=iteration, update_allowed=mi.Bool(True))
                self.assertTrue(bool(np.asarray(updated["accepted"])[0]))
            self.assert_pair(models, optimizers)

    def test_valid_zero_gradients_age_adam_but_invalid_nonzero_rows_do_not(self):
        models, optimizers = paired(11)
        ids = [2, 5, 8]
        warm = outputs(11, optimizers[0].dimensions, ids,
                       appearance=np.ones(3), geometry=np.ones(3))
        for model, optimizer, out in zip(models, optimizers, warm):
            optimizer.step(model, out, update_allowed=mi.Bool(True))
        previous = [capture(model, opt) for model, opt in zip(models, optimizers)]
        cases = outputs(11, optimizers[0].dimensions, ids, 1,
                        appearance=[1, 1, 0], geometry=[0, 1, 0])
        for out, compact in zip(cases, (False, True)):
            row = 1 if compact else 5
            for group, dimension in optimizers[0].dimensions.items():
                value = np.array(out["gradients"][group]).reshape(-1, dimension)
                value[row] = 0.
                out["gradients"][group] = mi.Float(value.ravel())
        for model, optimizer, out in zip(models, optimizers, cases):
            optimizer.step(model, out, update_allowed=mi.Bool(True))
        self.assert_pair(models, optimizers)
        after = capture(models[1], optimizers[1])
        for group, dimension in optimizers[1].dimensions.items():
            geometry = group in ("centers", "log_scales", "rotation_tangent")
            np.testing.assert_array_equal(after[f"{group}/steps"][ids],
                                          [1, 2, 1] if geometry else [2, 2, 1])
            for field in ("mean", "squared_mean"):
                np.testing.assert_array_equal(after[f"{group}/{field}"].reshape(11, dimension)[8],
                                              previous[1][f"{group}/{field}"].reshape(11, dimension)[8])
        dimension = optimizers[1].dimensions["radiance_sh"]
        np.testing.assert_allclose(after["radiance_sh/mean"].reshape(11, dimension)[5],
            .9 * previous[1]["radiance_sh/mean"].reshape(11, dimension)[5], atol=3e-7, rtol=3e-5)

    def test_changing_k_does_not_repeatedly_record_or_read_lengths(self):
        models, optimizers = paired(71)
        # Keep scalar K=1 as a separate case: Dr.Jit may specialize a scalar
        # gather differently from a general buffer, independently of K changes.
        sizes = (3, 17, 29, 0, 2, 41, 9, 63) * 3
        cases = [outputs(71, optimizers[0].dimensions, np.arange(k), iteration)
                 for iteration, k in enumerate(sizes)]
        allowed = mi.Bool(True)
        dr.eval(cases, allowed)
        dr.sync_thread()
        for iteration, pair in enumerate(cases):
            optimizers[0].step(models[0], pair[0], sample_count=1 if iteration % 2 else 4,
                               update_allowed=allowed)
        with patch.object(dr, "compress", side_effect=AssertionError("Host-sized compaction used")):
            with dr.scoped_set_flag(dr.JitFlag.ForbidSynchronization, True):
                for iteration, pair in enumerate(cases):
                    optimizers[1].step(models[1], pair[1], sample_count=1 if iteration % 2 else 4,
                                       update_allowed=allowed)
                    if iteration == 7:
                        warmed = optimizers[1].statistics()
                    elif iteration > 7:
                        self.assertEqual(optimizers[1].statistics()["recordings"], warmed["recordings"])
                        self.assertEqual(optimizers[1].statistics()["clears"], warmed["clears"])
        self.assertLessEqual(warmed["recordings"], 4)
        self.assertEqual(optimizers[1].statistics()["host_length_readbacks"], 0)
        self.assert_pair(models, optimizers)

    def test_single_voxel_scalar_compact_buffers(self):
        models, optimizers = paired(1)
        for iteration, (app, geom) in enumerate(((1, 1), (0, 1), (1, 0), (1, 1))):
            cases = outputs(1, optimizers[0].dimensions, [0], iteration,
                            appearance=[app], geometry=[geom])
            for model, optimizer, out in zip(models, optimizers, cases):
                out["gradients"]["opacity_sh"] = mi.Float(.25 * (iteration + 1))
                optimizer.step(model, out, update_allowed=mi.Bool(True))
        self.assert_pair(models, optimizers)

    def test_multiple_device_chunks_and_partial_last_chunk(self):
        count = 65553
        models, optimizers = paired(count, sh=1)
        for iteration, ids in enumerate(([1, 32767, 32768, 65535, 65551],
                                         np.arange(1, count - 1, dtype=np.uint32))):
            cases = outputs(count, optimizers[0].dimensions, ids, iteration)
            for model, optimizer, out in zip(models, optimizers, cases):
                optimizer.step(model, out, sample_count=2, update_allowed=mi.Bool(True))
        self.assert_pair(models, optimizers)

    def test_empty_compact_and_empty_model_are_noops(self):
        for count in (0, 29):
            for finite in (True, False):
                with self.subTest(count=count, finite=finite):
                    models, optimizers = paired(count)
                    cases = outputs(count, optimizers[0].dimensions, [])
                    for model, optimizer, out in zip(models, optimizers, cases):
                        out.update(growth_gain=dr.empty(mi.Float, 0), deletion_gain=dr.empty(mi.Float, 0))
                        before = capture(model, optimizer)
                        allowed = _finite_accumulator(out, mi.Float(0), mi.Bool(finite), dr)
                        self.assertEqual(dr.width(allowed), 1)
                        updated = optimizer.step(model, out, update_allowed=allowed)
                        self.assertEqual(bool(np.asarray(updated["accepted"])[0]), finite)
                        assert_equal(self, capture(model, optimizer), before, exact=True)

    def test_invalid_ids_reject_whole_update_and_latch_failure(self):
        for ids in ([2, 2], [9, 3], [1, 29], [0, 0xffffffff]):
            with self.subTest(ids=ids):
                models, optimizers = paired()
                model, optimizer = models[1], optimizers[1]
                _, out = outputs(29, optimizer.dimensions, [1, 3],
                                 appearance=[1, 1], geometry=[1, 1])
                out["voxel_ids"] = mi.UInt32(ids)
                before = capture(model, optimizer)
                result = optimizer.step(model, out, update_allowed=mi.Bool(True))
                self.assertFalse(bool(np.asarray(result["accepted"])[0]))
                assert_equal(self, capture(model, optimizer), before, exact=True)
                _, healthy = outputs(29, optimizer.dimensions, [1, 3])
                result = optimizer.step(model, healthy, update_allowed=mi.Bool(True))
                self.assertFalse(bool(np.asarray(result["accepted"])[0]))
                assert_equal(self, capture(model, optimizer), before, exact=True)

    def test_bad_compact_lengths_fail_before_update(self):
        for field in ("gradient", "count"):
            with self.subTest(field=field):
                models, optimizers = paired()
                model, optimizer = models[1], optimizers[1]
                _, out = outputs(29, optimizer.dimensions, [2, 7])
                if field == "gradient":
                    out["gradients"]["centers"] = mi.Float([1., 2., 3.])
                else:
                    out["appearance_valid"] = mi.UInt32([1])
                before = capture(model, optimizer)
                with self.assertRaises(ValueError):
                    optimizer.step(model, out, update_allowed=mi.Bool(True))
                assert_equal(self, capture(model, optimizer), before, exact=True)

    def test_nonfinite_batch_guard_and_external_inactive_moment(self):
        for fault in ("gradient", "loss", "inactive_moment"):
            with self.subTest(fault=fault):
                models, optimizers = paired()
                model, optimizer = models[1], optimizers[1]
                _, out = outputs(29, optimizer.dimensions, [2, 7],
                                 appearance=[1, 0], geometry=[1, 0])
                loss = mi.Float(.25)
                if fault == "gradient":
                    # Invalid-row NaNs must still block the whole batch.
                    out["gradients"]["opacity_sh"] = mi.Float([.1, float("nan")])
                elif fault == "loss":
                    loss = mi.Float(float("nan"))
                else:
                    dr.scatter(optimizer.state["rotation_tangent"]["mean"],
                               mi.Float(float("nan")), mi.UInt32(0))
                before = capture(model, optimizer)
                allowed = _finite_accumulator(out, loss, mi.Bool(True), dr)
                result = optimizer.step(model, out, update_allowed=allowed)
                self.assertFalse(bool(np.asarray(result["accepted"])[0]))
                assert_equal(self, capture(model, optimizer), before, exact=True)

    def test_direct_native_count_mismatch_rejects_without_partial_commit(self):
        for ids, supplied_count in (([1, 3], 1), ([1, 3], 3), ([], 1)):
            with self.subTest(ids=ids, supplied_count=supplied_count):
                models, optimizers = paired()
                model, optimizer = models[1], optimizers[1]
                _, out = outputs(29, optimizer.dimensions, ids,
                                 appearance=np.ones(len(ids)), geometry=np.ones(len(ids)))
                before = capture(model, optimizer)
                state = optimizer._state
                means = {key: value["mean"] for key, value in state.items()}
                squares = {key: value["squared_mean"] for key, value in state.items()}
                steps = {key: value["steps"] for key, value in state.items()}
                settings = {name: float(value) for name, value in asdict(optimizer.config).items()}
                settings["gradient_factor"] = 1.
                scope = model.optimizer_update_scope_begin(means, squares, steps)
                try:
                    accepted = model.optimizer_step_compact_device(out["voxel_ids"],
                        dr.opaque(mi.UInt32, supplied_count), out["gradients"],
                        out["appearance_valid"], out["geometry_valid"], out["background_carve_valid"],
                        means, squares, steps, settings, mi.Bool(True), True)
                    dr.eval(accepted, *means.values(), *squares.values(), *steps.values())
                finally:
                    mi._reconstruction_optimizer_update_scope_end(scope)
                self.assertFalse(bool(np.asarray(accepted)[0]))
                assert_equal(self, capture(model, optimizer), before, exact=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", default=RuntimeConfig().build_dir)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    global mi, dr, np
    with tempfile.TemporaryDirectory(prefix="mitsuba-compact-optimizer-") as directory:
        mi, dr, np, handles = initialize_runtime(RuntimeConfig(args.build_dir, args.gpu), Path(directory), {})
        original_tests.mi, original_tests.dr, original_tests.np = mi, dr, np
        if not hasattr(mi.VoxelReconstruction, "optimizer_step_compact_device"):
            raise RuntimeError("Rebuild this source checkout for compact-gradient Adam")
        result = unittest.TextTestRunner(verbosity=2).run(
            unittest.defaultTestLoader.loadTestsFromTestCase(CompactOptimizerTests))
        dr.sync_thread()
        return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
