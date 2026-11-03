"""CPU-only nested replica transfers with explicit 32/64-bit wire types."""

from __future__ import annotations

import copy
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reconstruction_replica import broadcast_device_state


class _Array:
    def __init__(self, value):
        self.values = np.asarray(value.values if isinstance(value, _Array) else value,
                                 dtype=self.dtype).reshape(-1).copy()


class _Float(_Array):
    dtype = np.dtype(np.float32)


class _UInt32(_Array):
    dtype = np.dtype(np.uint32)


class _UInt64(_Array):
    dtype = np.dtype(np.uint64)


class _Int64(_Array):
    dtype = np.dtype(np.int64)


class _Double(_Array):
    dtype = np.dtype(np.float64)


class _Dr:
    VarType = SimpleNamespace(Float32="float32", Float64="float64", UInt32="uint32",
                              Int32="int32", UInt64="uint64", Int64="int64")

    @staticmethod
    def is_array_v(cls):
        return issubclass(cls, _Array)

    @staticmethod
    def type_v(cls):
        return {_Float: "float32", _UInt32: "uint32", _UInt64: "uint64",
                _Int64: "int64", _Double: "float64"}[cls]

    @staticmethod
    def width(value):
        return len(value.values)

    @staticmethod
    def zeros(cls, count):
        return cls(np.zeros(count, cls.dtype))


class _Distributed:
    world_size = 2

    def __init__(self, rank, shared):
        self.rank, self.shared = rank, shared

    def broadcast(self, description):
        if self.rank == 0:
            self.shared["description"] = copy.deepcopy(description)
        return copy.deepcopy(self.shared["description"])


class _Collective:
    def __init__(self, rank, shared):
        self.rank, self.shared = rank, shared

    def broadcast_arrays(self, fields, *, root, uint_keys=(), cell_keys=()):
        assert root == 0
        assert not set(uint_keys).intersection(cell_keys)
        self.shared.setdefault("wire_types", []).append((tuple(uint_keys), tuple(cell_keys)))
        for key, value in fields.items():
            expected = _UInt64 if key in cell_keys else _UInt32 if key in uint_keys else _Float
            assert type(value) is expected, (key, type(value), expected)
        if self.rank == 0:
            self.shared["buffers"] = {key: type(value)(value) for key, value in fields.items()}
            return fields
        for key, value in fields.items():
            source = self.shared["buffers"][key]
            assert type(source) is type(value) and len(source.values) == len(value.values)
        return {key: type(value)(self.shared["buffers"][key]) for key, value in fields.items()}


class ReplicaStateTests(unittest.TestCase):
    def setUp(self):
        self.mi = SimpleNamespace(Float=_Float, UInt32=_UInt32, UInt64=_UInt64)
        self.dr = _Dr()

    def transfer(self, state):
        shared = {}
        root = broadcast_device_state(state, _Distributed(0, shared), _Collective(0, shared), self.mi, self.dr)
        peer = broadcast_device_state(None, _Distributed(1, shared), _Collective(1, shared), self.mi, self.dr)
        self.assertEqual(shared["wire_types"][0], shared["wire_types"][1])
        return root, peer, shared

    def test_numpy_nested_state_keeps_high_cells_cooldown_and_32bit_counts(self):
        state = {"model_state": {"cell_indices": np.array([7, 2 ** 32 + 19, 2 ** 40 + 3], np.uint64),
                                 "centers": np.array([.1, .2, .3], np.float32)},
                 "topology": {"cooldown_cells": np.array([2 ** 40 + 3], np.uint64),
                              "observed": np.array([4, 5], np.uint32)},
                 "optimizer": (np.array([9, 11], np.uint32), [np.empty(0, np.uint64)]),
                 "metadata": {"epoch": 3, "label": "state", "enabled": True}}
        root, peer, shared = self.transfer(state)
        for restored in (root, peer):
            np.testing.assert_array_equal(restored["model_state"]["cell_indices"].values,
                                          state["model_state"]["cell_indices"])
            np.testing.assert_array_equal(restored["topology"]["cooldown_cells"].values,
                                          state["topology"]["cooldown_cells"])
            self.assertIs(type(restored["topology"]["observed"]), _UInt32)
            self.assertIs(type(restored["optimizer"][0]), _UInt32)
            self.assertIs(type(restored["optimizer"][1][0]), _UInt64)
            self.assertEqual(restored["optimizer"][1][0].values.size, 0)
            self.assertEqual(restored["metadata"], state["metadata"])
            self.assertIsInstance(restored["optimizer"], tuple)
            self.assertIsInstance(restored["optimizer"][1], list)
        self.assertEqual(shared["wire_types"][0], (("3", "4"), ("0", "2", "5")))
        self.assertIn(("array", "0", 3, "uint64"), shared["description"][1][0][1][1][0])

    def test_native_types_use_trait_without_downloading_or_path_heuristics(self):
        state = {"anonymous_address_buffer": _UInt64([2 ** 45 + 1]),
                 "counts": _UInt32([7]), "radiance": _Float([.1]),
                 "long_epoch": _Int64([2 ** 34 + 3])}
        _, peer, shared = self.transfer(state)
        self.assertIs(type(peer["anonymous_address_buffer"]), _UInt64)
        self.assertIs(type(peer["long_epoch"]), _UInt64)
        np.testing.assert_array_equal(peer["long_epoch"].values, [2 ** 34 + 3])
        self.assertEqual(shared["wire_types"][0], (("1",), ("0", "3")))

    def test_old_core_cell_ids_promote_without_promoting_steps(self):
        _, peer, _ = self.transfer({"cell_indices": np.array([0, 8], np.uint32),
                                    "steps": np.array([1, 2], np.uint32)})
        self.assertIs(type(peer["cell_indices"]), _UInt64)
        self.assertIs(type(peer["steps"]), _UInt32)
        np.testing.assert_array_equal(peer["cell_indices"].values, [0, 8])

    def test_single_rank_preserves_original_objects_and_shapes(self):
        state = {"image": np.zeros((2, 3), np.float64)}
        result = broadcast_device_state(state, SimpleNamespace(world_size=1), None, None, None)
        self.assertIs(result, state)

    def test_empty_state_and_nonzero_root(self):
        shared = {}
        result = broadcast_device_state({}, _Distributed(0, shared), _Collective(0, shared), self.mi, self.dr)
        self.assertEqual(result, {})
        self.assertNotIn("buffers", shared)
        with self.assertRaisesRegex(ValueError, "rank 0"):
            broadcast_device_state({}, _Distributed(0, {}), None, self.mi, self.dr, root=1)

    def test_unsupported_dtype_shape_and_floating_addresses_fail(self):
        for array in (np.zeros((2, 2), np.float32), np.zeros(1, np.float64),
                      np.zeros(1, np.uint8), _Double([1.])):
            with self.subTest(array_type=type(array)), self.assertRaises(ValueError):
                broadcast_device_state({"buffer": array}, _Distributed(0, {}), None, self.mi, self.dr)
        with self.assertRaisesRegex(ValueError, "integer addresses"):
            broadcast_device_state({"cell_indices": np.array([1.], np.float32)},
                                   _Distributed(0, {}), None, self.mi, self.dr)


if __name__ == "__main__":
    unittest.main()
