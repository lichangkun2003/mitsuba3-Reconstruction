"""CPU checks for bounded device assembly of projective background children.

The fake device executes array operations with NumPy. The production CPU
refiner remains the value/order oracle; no Dr.Jit or Mitsuba import is needed.
"""

from __future__ import annotations

import copy
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

import reconstruction_topology as topology
from reconstruction_projective import core_mask_numpy, refine_projective_device
from test_reconstruction_projective_topology import projective_fixture


class ArrayDevice:
    """Reject background-width scratch and track complete, disjoint writes."""

    cuda = SimpleNamespace(UInt32=np.uint32, Float32=np.float32, Float64=np.float64)
    maximum = staticmethod(np.maximum)
    minimum = staticmethod(np.minimum)
    log1p = staticmethod(np.log1p)
    exp = staticmethod(np.exp)
    abs = staticmethod(np.abs)
    log = staticmethod(np.log)
    expm1 = staticmethod(np.expm1)
    select = staticmethod(np.where)

    def __init__(self, background_children):
        self.background_children = background_children
        self.targets = {}
        self.gather_widths = []

    @staticmethod
    def mask_t(_):
        return np.bool_

    def empty(self, dtype, count):
        value = np.empty(count, dtype=dtype)
        self.targets[id(value)] = (value, np.zeros(count, np.uint8))
        return value

    @staticmethod
    def arange(dtype, count):
        return np.arange(count, dtype=dtype)

    @staticmethod
    def compress(mask):
        return np.flatnonzero(mask).astype(np.uint32)

    def gather(self, dtype, source, indices):
        self.gather_widths.append(len(indices))
        if len(indices) > self.background_children:
            raise AssertionError("Background gather must be bounded by rows, not rows times SH width")
        return np.asarray(source)[indices].astype(dtype, copy=True)

    def scatter(self, target, values, indices):
        target[indices] = values
        writes = self.targets[id(target)][1]
        np.add.at(writes, indices, 1)
        if np.any(writes > 1):
            raise AssertionError("Projective output components must have disjoint writes")

    @staticmethod
    def eval(*_):
        pass


def expanded_fixture(positions):
    checkpoint = projective_fixture(positions)
    count = len(checkpoint["model_state"]["cell_indices"])
    checkpoint["model_metadata"].update(sh_count=9, opacity_sh_count=4)
    state = checkpoint["model_state"]
    state["radiance_sh"] = np.linspace(-.7, 1.1, count * 27, dtype=np.float32)
    state["opacity_sh"] = np.linspace(-.2, .4, count * 4, dtype=np.float32)
    state["log_scales"][:] = np.log(.4)
    return checkpoint


class ProjectiveStreamingTests(unittest.TestCase):
    def device_refine(self, checkpoint, config, limit=67108864):
        state, metadata = checkpoint["model_state"], checkpoint["model_metadata"]
        background = ~core_mask_numpy(state["cell_indices"], metadata["resolution"],
                                     metadata["core_resolution"], np)
        device = ArrayDevice(8 * int(background.sum()))

        def take_rows(value, ids, width=1):
            return np.asarray(value).reshape(-1, width)[ids].ravel().copy()

        def fit_core(core_checkpoint, settings, maximum, _device):
            return topology.refine_state(core_checkpoint, settings, maximum)

        with patch("reconstruction_topology_gpu.take_rows", side_effect=take_rows):
            result = refine_projective_device(checkpoint, config, limit, fit_core, device)
        for value, writes in device.targets.values():
            np.testing.assert_array_equal(writes, np.ones(len(value), np.uint8))
        return result, device

    def assert_matches_oracle(self, checkpoint, config):
        before = copy.deepcopy(checkpoint)
        expected = topology.refine_state(checkpoint, config)
        actual, device = self.device_refine(checkpoint, config)
        self.assertEqual(actual["model_metadata"], expected["model_metadata"])
        for name, values in actual["model_state"].items():
            np.testing.assert_allclose(values, expected["model_state"][name],
                                       rtol=3e-6, atol=3e-6, err_msg=name)
            np.testing.assert_array_equal(checkpoint["model_state"][name], before["model_state"][name])
        return actual, expected, device

    def test_mixed_core_and_nine_sh_background_match_for_two_levels(self):
        checkpoint = expanded_fixture(((.2, .1, .1), (1.1, .1, .1), (-1.1, .1, .1)))
        config = topology.CoarseToFineConfig(opacity_optical_depth_scale=.65)
        for _ in range(2):
            actual, expected, device = self.assert_matches_oracle(checkpoint, config)
            # Every field has one final output allocation; no dense shell table.
            self.assertEqual(len(device.targets), len(checkpoint["model_state"]))
            self.assertLessEqual(max(device.gather_widths), device.background_children)
            for name in ("cell_indices", "radiance_sh", "rotations"):
                np.testing.assert_array_equal(actual["model_state"][name], expected["model_state"][name])
            checkpoint = expected

    def test_background_only_preserves_high_order_opacity_and_extreme_dc(self):
        checkpoint = expanded_fixture(((1.1, .1, .1), (-1.1, .1, .1), (.1, 1.1, .1), (.1, -1.1, .1)))
        checkpoint["model_state"]["opacity_sh"].reshape(-1, 4)[:, 0] = [-2000., -1e-7, 20., 2000.]
        for attenuation in (.01, .65, 1.):
            with self.subTest(attenuation=attenuation):
                config = topology.CoarseToFineConfig(opacity_optical_depth_scale=attenuation)
                actual, expected, _ = self.assert_matches_oracle(checkpoint, config)
                np.testing.assert_array_equal(actual["model_state"]["opacity_sh"].reshape(-1, 4)[:, 1:],
                                              expected["model_state"]["opacity_sh"].reshape(-1, 4)[:, 1:])

    def test_core_only_and_empty_models_use_core_result(self):
        for positions in (((.2, .1, .1),), ()):
            with self.subTest(positions=positions):
                self.assert_matches_oracle(expanded_fixture(positions), topology.CoarseToFineConfig())

    def test_capacity_failure_happens_before_core_fit_or_output_allocation(self):
        checkpoint = expanded_fixture(((.2, .1, .1), (1.1, .1, .1)))
        device = ArrayDevice(8)
        fit = unittest.mock.Mock(side_effect=AssertionError("Must reject capacity before fitting"))
        with self.assertRaisesRegex(RuntimeError, "max_active_voxels"):
            refine_projective_device(checkpoint, topology.CoarseToFineConfig(), 7, fit, device)
        fit.assert_not_called()
        self.assertEqual(device.targets, {})


if __name__ == "__main__":
    unittest.main()
