"""CPU integration checks for one Adam migration per topology commit."""

from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import weakref

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reconstruction_optimizer import FalcorOptimizer
from reconstruction_topology import CoarseToFineConfig, TopologySettings
import reconstruction_topology_gpu as topology_gpu


class Float(np.ndarray):
    scalar = np.float32

    def __new__(cls, values):
        return np.asarray(values, dtype=cls.scalar).view(cls)

    @property
    def index(self):
        return self.__array_interface__["data"][0]


class UInt(Float):
    scalar = np.uint32


class Int64(Float):
    scalar = np.int64


class Bool(Float):
    scalar = np.bool_


class Dr:
    arange = staticmethod(lambda dtype, count: dtype(np.arange(count)))
    zeros = staticmethod(lambda dtype, count: dtype(np.zeros(count)))
    empty = staticmethod(lambda dtype, count: dtype(np.empty(count)))
    full = staticmethod(lambda dtype, value, count: dtype(np.full(count, value)))
    width = staticmethod(len)
    eval = staticmethod(lambda *values: None)
    compress = staticmethod(lambda mask: UInt(np.flatnonzero(mask)))
    select = staticmethod(lambda mask, first, second: type(first)(np.where(mask, first, second)))

    @staticmethod
    def gather(dtype, values, indices, active):
        result = dtype(np.zeros(len(indices)))
        result[active] = values[indices[active]]
        return result


def take_rows(values, indices, width=1):
    return type(values)(np.asarray(values).reshape(-1, width)[indices].reshape(-1))


class ReplayOwner:
    pass


class Optimizer(FalcorOptimizer):
    def __init__(self, count):
        # Deliberately bypass runtime initialization: exercise the real remap
        # on CPU arrays, including its actual cache clearing and step counts.
        self.mi, self.dr = SimpleNamespace(Float=Float, UInt32=UInt), Dr
        self.count = count
        self.dimensions = dict(radiance_sh=27, opacity_sh=1, centers=3,
                               log_scales=3, rotation_tangent=3)
        self._state = {
            name: {field: (UInt if field == "steps" else Float)(
                np.arange(count if field == "steps" else count * width) + offset)
                for field, offset in (("mean", 10), ("squared_mean", 20), ("steps", 30))}
            for name, width in self.dimensions.items()}
        owner = ReplayOwner()
        self.replay_owner = weakref.ref(owner)
        self._update_cache = {0: owner}
        self._compact_update_cache = {0: owner}
        self._update_model = owner
        self._update_key, self._update_clears = None, 0
        self.calls = []

    def remap(self, survivors, new_count, **kwargs):
        self.calls.append((np.asarray(survivors).copy(), new_count))
        return super().remap(survivors, new_count, **kwargs)


class CommitRemapTests(unittest.TestCase):
    def run_commit(self, *, removed=(), added=0, growth_due=True,
                   selected_none=False, count=4, failure=None):
        settings = TopologySettings(growth_protection_epochs=5, deletion_cooldown_epochs=5)
        manager = SimpleNamespace(settings=settings,
            schedule=CoarseToFineConfig(enabled=False, total_epochs=30),
            resolution=8, core_resolution=8, scene_mapping="bounded", level_start_epoch=0,
            level_budget=30, growth_warmup=0, count=count, completed_windows=3,
            growth_frozen=lambda boundary: False, deletion_frozen=lambda boundary: False,
            _growth_due=lambda boundary: growth_due,
            _rgb_deletion_due=lambda boundary: bool(removed),
            _low_opacity_deletion_due=lambda boundary: False,
            pending_growth={}, pending_rgb_deletion=None,
            _growth_world_cache=None, _growth_device=None, _rgb_growth_device=None,
            _rgb_deletion_device=None)
        model_state = dict(cell_indices=UInt(np.arange(count) * 10 + 1),
            centers=Float(np.arange(count * 3)), log_scales=Float(np.arange(count * 3) + 20),
            rotations=Float(np.arange(count * 4) + 30),
            radiance_sh=Float(np.arange(count * 27) + 40), opacity_sh=Float(np.arange(count) + 50))
        model = SimpleNamespace(clear_growth_candidates=lambda: None)
        optimizer = Optimizer(count)
        lifecycle = topology_gpu.GPUTopologyLifecycle.__new__(topology_gpu.GPUTopologyLifecycle)
        lifecycle.manager, lifecycle.dr = manager, Dr
        lifecycle.UInt, lifecycle.Int64, lifecycle.Bool = UInt, Int64, Bool
        lifecycle.fields = {name: Int64(np.arange(count) + offset)
                            for name, offset in (("growth_birth", 1), ("growth_wait_start", 2),
                                                 ("deletion_eligible", 3))}
        lifecycle.candidates = Bool(np.ones(count, dtype=bool))
        lifecycle.cooldown_cells, lifecycle.cooldown_expiry = UInt([]), Int64([])
        lifecycle.count = lambda mask: int(np.count_nonzero(mask))
        lifecycle.reduce_event = lambda event: None
        lifecycle.evaluate_opacity = lambda model, boundary, event: None
        survivors = np.array([i for i in range(count) if i not in removed], dtype=np.uint32)
        phase = {"selected": False, "model_concats": 0}

        def evaluate(boundary, event, *, region_eligible=None):
            self.assertIsNone(region_eligible)
            if not removed:
                return None
            event["deletion_rgb"] = {}
            mask = np.zeros(count, dtype=bool)
            mask[list(removed)] = True
            return Bool(mask)

        def select(checkpoint, boundary, event):
            # Parent row IDs and lifecycle fields must already describe the
            # compact model while Adam still uses its original row layout.
            self.assertEqual(manager.count, len(survivors))
            self.assertEqual(optimizer.count, count)
            self.assertEqual(len(optimizer.calls), 0)
            np.testing.assert_array_equal(checkpoint["model_state"]["cell_indices"],
                                          model_state["cell_indices"][survivors])
            np.testing.assert_array_equal(lifecycle.fields["growth_birth"], survivors + 1)
            if removed:
                self.assertIsNone(optimizer.replay_owner())
                self.assertFalse(optimizer._update_cache)
                self.assertFalse(optimizer._compact_update_cache)
            if failure is not None:
                raise failure
            phase["selected"] = True
            if selected_none:
                return None
            selected = {("cells" if name == "cell_indices" else name):
                        type(value)(np.arange(added * (len(value) // count if count else 1)) + 100)
                        for name, value in model_state.items()}
            selected["parent_ids"] = UInt(np.zeros(added))
            return selected

        def concat(first, second):
            if phase["selected"]:
                self.assertEqual(len(optimizer.calls), 1,
                                 "Model allocation must follow the single Adam migration")
                phase["model_concats"] += 1
            return type(first)(np.concatenate((first, second)))

        lifecycle.evaluate_rgb_deletion, lifecycle.select_growth = evaluate, select
        checkpoint = {"model_state": model_state, "model_metadata": {"sh_count": 9, "opacity_sh_count": 1}}
        with (patch.object(topology_gpu, "device_checkpoint", return_value={
                  "model_state": dict(model_state), "model_metadata": checkpoint["model_metadata"]}),
              patch.object(topology_gpu, "take_rows", side_effect=take_rows),
              patch.object(topology_gpu, "concat_rows", side_effect=concat)):
            if failure is not None:
                with self.assertRaises(type(failure)):
                    lifecycle.finish(model, optimizer, 10)
                self.assertEqual(optimizer.calls, [])
                self.assertEqual(phase["model_concats"], 0)
                reference = Optimizer(count)
                self.assert_optimizer_equal(optimizer, reference)
                return
            replacement, event = lifecycle.finish(model, optimizer, 10)

        effective_added = added if growth_due and not selected_none else 0
        expected_count = len(survivors) + effective_added
        reference = Optimizer(count)
        if removed:
            reference.remap(UInt(survivors), len(survivors))
        if effective_added:
            reference.remap(UInt(np.arange(len(survivors))), expected_count)
        self.assert_optimizer_equal(optimizer, reference)
        self.assertEqual(manager.count, expected_count)
        self.assertEqual(event["deleted"], len(removed))
        self.assertEqual(event["grown"], effective_added)
        self.assertEqual(len(optimizer.calls), int(bool(removed or effective_added)))
        if optimizer.calls:
            np.testing.assert_array_equal(optimizer.calls[0][0], survivors)
            self.assertEqual(optimizer.calls[0][1], expected_count)
            self.assertIsNotNone(replacement)
            self.assertEqual(len(replacement["model_state"]["cell_indices"]), expected_count)
            np.testing.assert_array_equal(replacement["model_state"]["cell_indices"][:len(survivors)],
                                          model_state["cell_indices"][survivors])
        else:
            self.assertIsNone(replacement)
        self.assertEqual(phase["model_concats"], len(model_state) if effective_added else 0)

    def assert_optimizer_equal(self, actual, expected):
        self.assertEqual(actual.count, expected.count)
        for name, group in expected._state.items():
            for field, values in group.items():
                np.testing.assert_array_equal(actual._state[name][field], values,
                                              err_msg=f"{name}.{field}")

    def test_commit_variants_match_sequential_adam_migrations(self):
        cases = (
            dict(removed=(1, 3), added=2),  # Same final count, different row identities.
            dict(removed=(1,), added=2),
            dict(removed=(1, 3), selected_none=True),
            dict(removed=(1, 3), added=0),
            dict(removed=(1, 3), growth_due=False),
            dict(removed=(0, 1, 2, 3), selected_none=True),
            dict(added=2), dict(added=0), dict(growth_due=False),
            dict(count=0, selected_none=True))
        for case in cases:
            with self.subTest(**case):
                self.run_commit(**case)

    def test_failed_revalidation_does_not_migrate_or_append(self):
        for failure in (RuntimeError("revalidation failed"), KeyboardInterrupt()):
            with self.subTest(failure=type(failure).__name__):
                self.run_commit(removed=(1, 3), added=2, failure=failure)


if __name__ == "__main__":
    unittest.main()
