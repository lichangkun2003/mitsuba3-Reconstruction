"""CUDA bitwise parity for bounded-memory topology field and Adam migration."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from preview import RuntimeConfig
from reconstruction_common import initialize_runtime, restore_model
from reconstruction_memory import TrainingMemoryError
from reconstruction_optimizer import FalcorOptimizer
from reconstruction_parameter_transfer import DeferredRemapField
import test_reconstruction_optimization as original_tests


def float_values(size):
    # Include negative zero and subnormals: numeric allclose is insufficient.
    bits = np.resize(np.array([0, 0x80000000, 0x3f800000, 0xbf000000,
                                1, 0x007fffff, 0x4b123456], np.uint32), size)
    return bits.view(np.float32)


class ParameterTransferTests(unittest.TestCase):
    def test_deletion_reorder_and_append_match_host_bits_across_chunk_boundaries(self):
        width, old_count = 27, 7
        source = float_values(width * old_count)
        append = float_values(width * 2)[::-1].copy()
        survivors = np.array([6, 2, 0, 5], np.uint32)
        expected = np.concatenate((source.reshape(old_count, width)[survivors].ravel(), append))
        plan = DeferredRemapField(mi.Float(source), mi.UInt32(survivors), width, 6,
                                  append=mi.Float(append), dr=dr, chunk_bytes=4 * 31)
        result = plan.materialize()
        np.testing.assert_array_equal(np.asarray(result).view(np.uint32), expected.view(np.uint32))
        self.assertIsNone(plan._old)
        with self.assertRaisesRegex(RuntimeError, "only be materialized once"):
            plan.materialize()

    def test_unsigned_clocks_preserve_high_bits_and_newborns_are_zero(self):
        clocks = np.resize(np.array([0, 1, 2 ** 24 + 1, 2 ** 32 - 1], np.uint32), 21)
        survivors = np.array([5, 0, 3], np.uint32)
        plan = DeferredRemapField(mi.UInt32(clocks), mi.UInt32(survivors), 3, 5,
                                  dr=dr, chunk_bytes=4 * 5)
        expected = np.concatenate((clocks.reshape(7, 3)[survivors].ravel(), np.zeros(6, np.uint32)))
        np.testing.assert_array_equal(np.asarray(plan.materialize()), expected)

    def test_forced_disk_staging_matches_ram_and_cleans_temporary_field(self):
        with tempfile.TemporaryDirectory(prefix="topology-disk-parity-") as directory:
            source = float_values(51)
            expected = np.concatenate((source.reshape(17, 3)[[16, 5, 1]].ravel(), np.zeros(6, np.float32)))
            with patch("reconstruction_parameter_transfer.available_host_memory", return_value=0):
                plan = DeferredRemapField(mi.Float(source), mi.UInt32([16, 5, 1]), 3, 5,
                                          dr=dr, chunk_bytes=4 * 7, scratch_directory=directory)
                actual = np.asarray(plan.materialize())
            self.assertEqual(plan.statistics["storage"], "disk")
            np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_post_retirement_guard_refuses_before_replacement_allocation(self):
        plan = DeferredRemapField(mi.Float(float_values(21)), mi.UInt32([6, 1]), 3, 3, dr=dr)
        def refuse(required_bytes):
            self.assertIsNone(plan._old)
            self.assertGreater(required_bytes, 9 * 4)
            raise TrainingMemoryError({"reason": "external source owner consumes remaining GPU budget"})
        with self.assertRaises(TrainingMemoryError):
            plan.materialize(allocation_guard=refuse)

    def test_default_guard_refuses_unknown_or_insufficient_memory(self):
        for snapshot in (None, {"free_bytes": 0}):
            with self.subTest(snapshot=snapshot), patch(
                    "reconstruction_parameter_transfer.query_cuda_memory", return_value=snapshot):
                plan = DeferredRemapField(mi.Float([1, 2]), mi.UInt32([1]), 1, 2, dr=dr)
                with self.assertRaises(TrainingMemoryError):
                    plan.materialize()

    def test_empty_transfer_and_unsupported_precision(self):
        plan = DeferredRemapField(mi.Float([1, 2]), mi.UInt32([]), 1, 0, dr=dr)
        self.assertEqual(dr.width(plan.materialize()), 0)
        with self.assertRaises(TypeError):
            DeferredRemapField(dr.cuda.Float64([1]), mi.UInt32([0]), 1, 1, dr=dr)

    def test_optimizer_all_fields_and_degree_clocks_match_gpu_remap_bitwise(self):
        for reset_radiance in (False, True):
            for survivor_ids in ([6, 2, 0, 5], []):
                with self.subTest(reset_radiance=reset_radiance, survivors=survivor_ids):
                    payload = original_tests.seed_checkpoint(cells=tuple(range(7)), resolution=8, sh=9)
                    model = restore_model(payload, mi)
                    optimizers = [FalcorOptimizer(model, radiance_degree_steps=True) for _ in range(2)]
                    for optimizer in optimizers:
                        for name, state in optimizer._state.items():
                            size = 7 * optimizer.dimensions[name]
                            state["mean"] = mi.Float(float_values(size))
                            state["squared_mean"] = mi.Float(float_values(size)[::-1].copy())
                            state["steps"] = mi.UInt32(np.resize(np.array(
                                [0, 1, 2 ** 24 + 1, 2 ** 32 - 1], np.uint32),
                                7 * optimizer._step_size(name)))
                        optimizer._remember_state()
                    for staged, optimizer in zip((False, True), optimizers):
                        optimizer.remap(mi.UInt32(survivor_ids), len(survivor_ids) + 2,
                                        reset_radiance=reset_radiance, host_staged=staged)
                    expected, actual = [optimizer.state_dict()["groups"] for optimizer in optimizers]
                    for group in expected:
                        for field in expected[group]:
                            np.testing.assert_array_equal(actual[group][field].view(np.uint32),
                                                          expected[group][field].view(np.uint32))

    def test_optimizer_exposed_state_is_rejected_for_destructive_transfer(self):
        payload = original_tests.seed_checkpoint(cells=(0, 1), resolution=8, sh=9)
        model = restore_model(payload, mi)
        optimizer = FalcorOptimizer(model)
        borrowed = optimizer.state
        with self.assertRaises(TrainingMemoryError):
            optimizer.remap(mi.UInt32([1]), 2, host_staged=True)
        self.assertIs(optimizer._state, borrowed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", default=RuntimeConfig().build_dir)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    global mi, dr, np
    with tempfile.TemporaryDirectory(prefix="mitsuba-parameter-transfer-") as directory:
        report = {}
        mi, dr, np, handles = initialize_runtime(RuntimeConfig(args.build_dir, args.gpu), Path(directory), report)
        original_tests.mi, original_tests.dr, original_tests.np = mi, dr, np
        from train import refinement_memory_snapshot
        # Dr.Jit can leave no CUDA context current between operations. Match
        # training's real GPU snapshot, including its physical-memory fallback;
        # never replace allocation guards with a fabricated budget/no-op.
        with patch("reconstruction_parameter_transfer.query_cuda_memory",
                   side_effect=lambda: refinement_memory_snapshot(report["gpu"])):
            result = unittest.TextTestRunner(verbosity=2).run(
                unittest.defaultTestLoader.loadTestsFromTestCase(ParameterTransferTests))
        dr.sync_thread()
        return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
