"""CUDA checks for atomic compact parameter commits, without training a scene."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reconstruction_common import initialize_runtime
from preview import RuntimeConfig


DIMENSIONS = {"centers": 3, "log_scales": 3, "rotations": 4,
              "radiance_sh": 12, "opacity_sh": 2}
APPEARANCE = {"radiance_sh", "opacity_sh"}


def fixture(count=4):
    state = {
        "cell_indices": np.asarray([0, 1, 3, 7][:count], dtype=np.uint32),
        "centers": np.tile([.4, .5, .6], count).astype(np.float32),
        "log_scales": np.tile([-1., -.8, -.6], count).astype(np.float32),
        "rotations": np.tile([1., 0., 0., 0.], count).astype(np.float32),
        "radiance_sh": np.arange(count * 12, dtype=np.float32) * .01,
        "opacity_sh": np.arange(count * 2, dtype=np.float32) * .02,
    }
    device = {name: mi.UInt32(value) if name == "cell_indices" else mi.Float(value)
              for name, value in state.items()}
    return mi.VoxelReconstruction.from_state(2, [0.] * 3, [2.] * 3,
                                             device, 4, 2), state


def proposed(appearance, geometry):
    return {name: mi.Float(np.full(len(appearance if name in APPEARANCE else geometry)
                                  * width, .375, dtype=np.float32))
            for name, width in DIMENSIONS.items()}


class SparseCommitTests(unittest.TestCase):
    def assert_state(self, model, expected):
        for name, value in model.snapshot().items():
            np.testing.assert_array_equal(np.asarray(value), expected[name], err_msg=name)

    def commit(self, model, appearance, geometry, parameters, allowed=True):
        accepted = model.set_parameters_sparse(
            appearance_indices=mi.UInt32(appearance),
            geometry_indices=mi.UInt32(geometry), parameters=parameters,
            allowed=mi.Bool(allowed))
        self.assertEqual(dr.width(accepted), 1)
        return bool(accepted[0])

    def test_distinct_group_rows_and_nondefault_sh_dimensions(self):
        model, expected = fixture()
        appearance, geometry = [0, 3], [1, 2]
        revision = model.topology_revision
        parameters = proposed(appearance, geometry)
        self.assertTrue(self.commit(model, appearance, geometry, parameters))
        for name, width in DIMENSIONS.items():
            rows = appearance if name in APPEARANCE else geometry
            expected[name].reshape(-1, width)[rows] = .375
        self.assert_state(model, expected)
        self.assertEqual(model.topology_revision, revision)

    def test_each_parameter_can_be_committed_independently(self):
        for name, width in DIMENSIONS.items():
            with self.subTest(name=name):
                model, expected = fixture()
                appearance = [0, 2] if name in APPEARANCE else []
                geometry = [] if name in APPEARANCE else [1, 3]
                values = np.arange(2 * width, dtype=np.float32) * .125
                self.assertTrue(self.commit(model, appearance, geometry,
                                            {name: mi.Float(values)}))
                expected[name].reshape(-1, width)[appearance or geometry] = values.reshape(2, width)
                self.assert_state(model, expected)

    def test_empty_groups_partial_groups_and_empty_model(self):
        model, expected = fixture()
        self.assertTrue(self.commit(model, [], [], {}))
        self.assertTrue(self.commit(model, [], [], proposed([], [])))
        self.assert_state(model, expected)
        self.assertTrue(self.commit(model, [], [2], proposed([], [2])))
        for name, width in DIMENSIONS.items():
            if name not in APPEARANCE:
                expected[name].reshape(-1, width)[2] = .375
        self.assert_state(model, expected)
        empty, empty_state = fixture(count=0)
        self.assertTrue(self.commit(empty, [], [], proposed([], [])))
        self.assert_state(empty, empty_state)
        self.assertFalse(self.commit(empty, [0], [], proposed([0], [])))
        self.assert_state(empty, empty_state)

    def test_false_guard_rejects_every_group(self):
        model, expected = fixture()
        self.assertFalse(self.commit(model, [0, 3], [1, 2], proposed([0, 3], [1, 2]),
                                     allowed=False))
        self.assert_state(model, expected)

    def test_nonfinite_in_any_group_rejects_the_complete_transaction(self):
        for name in DIMENSIONS:
            for invalid in (float("nan"), float("inf"), -float("inf")):
                with self.subTest(name=name, invalid=invalid):
                    model, expected = fixture()
                    parameters = proposed([0, 2], [1, 3])
                    values = np.full(2 * DIMENSIONS[name], .375, dtype=np.float32)
                    values[-1] = invalid
                    parameters[name] = mi.Float(values)
                    self.assertFalse(self.commit(model, [0, 2], [1, 3], parameters))
                    self.assert_state(model, expected)

    def test_invalid_row_indices_reject_all_groups(self):
        for bad_rows in ([4], [4294967295], [1, 1], [3, 1]):
            for group in ("appearance", "geometry"):
                with self.subTest(rows=bad_rows, group=group):
                    model, expected = fixture()
                    appearance = bad_rows if group == "appearance" else [0, 2]
                    geometry = bad_rows if group == "geometry" else [1, 3]
                    self.assertFalse(self.commit(model, appearance, geometry,
                                                 proposed(appearance, geometry)))
                    self.assert_state(model, expected)

    def test_invalid_unused_indices_also_reject_transaction(self):
        model, expected = fixture()
        # An unused group still belongs to the input contract and cannot hide
        # malformed IDs beside an otherwise valid update.
        self.assertFalse(self.commit(model, [2, 1], [0], {"centers": mi.Float([.2] * 3)}))
        self.assert_state(model, expected)

    def test_shape_key_and_guard_errors_leave_every_parameter_unchanged(self):
        cases = [{"unknown": mi.Float([1.])}, {"centers": mi.Float([1., 2.])},
                 {"radiance_sh": mi.Float([1.] * 27)},
                 {"opacity_sh": mi.Float([1.])}]
        for parameters in cases:
            with self.subTest(keys=list(parameters)):
                model, expected = fixture()
                parameters = {**proposed([0], [1]), **parameters}
                with self.assertRaisesRegex(RuntimeError, "unknown key or wrong compact size"):
                    self.commit(model, [0], [1], parameters)
                self.assert_state(model, expected)
        for allowed in (mi.Bool([]), mi.Bool([True, False])):
            model, expected = fixture()
            with self.assertRaisesRegex(RuntimeError, "scalar mask"):
                model.set_parameters_sparse(mi.UInt32([0]), mi.UInt32([1]),
                                            proposed([0], [1]), allowed)
            self.assert_state(model, expected)

    def test_lazy_cross_group_proposals_and_external_snapshot_keep_old_values(self):
        model, expected = fixture()
        before = {name: value.copy() for name, value in expected.items()}
        snapshot = model.snapshot()
        # Leave proposals unevaluated. Opacity and radiance deliberately read
        # the old centers that the same transaction is about to overwrite.
        parameters = {
            "centers": dr.gather(mi.Float, snapshot["centers"], mi.UInt32([3, 4, 5])) + .125,
            "opacity_sh": dr.gather(mi.Float, snapshot["centers"], mi.UInt32([3, 4])) + 1.,
            "radiance_sh": dr.gather(mi.Float, snapshot["centers"],
                                     dr.arange(mi.UInt32, 12) % 3 + 3) + 2.,
        }
        self.assertTrue(self.commit(model, [1], [1], parameters))
        expected["centers"][3:6] = before["centers"][3:6] + .125
        expected["opacity_sh"][2:4] = before["centers"][3:5] + 1.
        expected["radiance_sh"][12:24] = np.tile(before["centers"][3:6], 4) + 2.
        self.assert_state(model, expected)
        for name, value in snapshot.items():
            np.testing.assert_array_equal(np.asarray(value), before[name], err_msg=name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", default=RuntimeConfig().build_dir)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    global mi, dr, np
    with tempfile.TemporaryDirectory(prefix="mitsuba-sparse-commit-") as temporary:
        mi, dr, np, handles = initialize_runtime(RuntimeConfig(args.build_dir, args.gpu),
                                                Path(temporary), {})
        if not hasattr(mi.VoxelReconstruction, "set_parameters_sparse"):
            raise RuntimeError("Rebuild this checkout: compact parameter commit binding is missing")
        result = unittest.TextTestRunner(verbosity=2).run(
            unittest.defaultTestLoader.loadTestsFromTestCase(SparseCommitTests))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
