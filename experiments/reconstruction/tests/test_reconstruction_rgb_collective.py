"""Exact sparse RGB collectives: CPU transport oracle and CUDA gather/scatter.

CUDA mode uses simulated peer tensors on one device. Real multi-device NCCL
transport is covered separately by test_reconstruction_nccl.py on the HPC.
"""

from __future__ import annotations

import argparse
import copy
import ctypes
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from reconstruction_collective import (
    GRADIENT_KEYS, SUM_COUNTER_KEYS, MAX_COUNTER_KEYS, NcclCollective,
    _NCCL_FLOAT32, _NCCL_UINT32, _NCCL_SUM, _NCCL_MAX, _NCCL_MIN,
)
from test_reconstruction_collective import _Dr, _fixture, _float, _uint


class ArrayDr(_Dr):
    ReduceOp = SimpleNamespace(Or="or")

    @staticmethod
    def zeros(array_type, size):
        return np.zeros(size, dtype=np.float32 if array_type is _float else np.uint32)

    @staticmethod
    def all(value):
        return np.atleast_1d(np.all(value))

    @staticmethod
    def block_reduce(op, value, block_size):
        assert op == "or"
        return np.any(value.reshape(-1, block_size), axis=1)

    @staticmethod
    def compress(value):
        return np.flatnonzero(value).astype(np.uint32)

    @staticmethod
    def scatter(target, value, index):
        target[index] = value


def inputs(count=32, *, dense=False, invalid=False, nan=False):
    peers = []
    for rank in range(3):
        value = {"gradients": {}, "loss_sum": _float(rank + .25), "finite": _uint(not (nan and rank == 1)),
                 "rgb_loss_sum": _float(rank + .125), "distortion_loss_sum": _float(.125)}
        for key in SUM_COUNTER_KEYS + MAX_COUNTER_KEYS:
            value[key] = np.zeros(count, np.uint32)
        active = np.arange(count) if dense else np.asarray([1, 3] if rank == 0 else [3, 7])
        active = active[active < count] if rank < 2 else np.asarray([], dtype=np.int64)  # Tail padding rank.
        value["appearance_valid"][active] = rank + 1
        value["geometry_valid"][active] = (rank + 1) * 2
        if count > 11 and rank == 0:
            value["appearance_valid"][11] = 1  # Valid zero must advance Adam.
        for group, dimension in zip(GRADIENT_KEYS, (27, 1, 3, 3, 3)):
            rows = np.zeros((count, dimension), np.float32)
            rows[active] = (rank + 1) * np.arange(1, dimension + 1) * .125
            if group == "centers" and count > 13 and rank == 1:
                rows[13] = .75  # Unexpected invalid-row data must not disappear.
            if nan and group == "opacity_sh" and count > 2 and rank == 1:
                rows[2] = np.nan
            value["gradients"][group] = rows.reshape(-1)
        if invalid and count and rank == 1:
            value["background_evidence"][0] = 1
        peers.append(value)
    return peers


def packets(peers, mode):
    count = len(peers[0]["appearance_valid"])
    support = []
    for value in peers:
        mask = (value["appearance_valid"] > 0) | (value["geometry_valid"] > 0)
        if count:
            for rows in value["gradients"].values():
                mask |= np.any(rows.reshape(count, -1) != 0, axis=1)
        support.append(mask.astype(np.uint32))
    ids = np.flatnonzero(np.max(np.stack(support), axis=0)).astype(np.uint32)
    sparse = mode == "auto" and count > 0 and len(ids) * 4 <= count * 3
    stages = []
    for rank, value in enumerate(peers):
        first = {key: value[key] for key in ("appearance_valid", "geometry_valid")}
        first.update(losses=np.concatenate([value[key] for key in
                     ("loss_sum", "rgb_loss_sum", "distortion_loss_sum")]),
                     finite=value["finite"] & _uint(all(np.all(value[key] == 0) for key in
                         ("background_carve_valid", *MAX_COUNTER_KEYS))))
        gradient = {key: rows.reshape(count, -1)[ids].reshape(-1) if sparse else rows
                    for key, rows in value["gradients"].items()}
        if mode == "auto" and count:
            first["support"] = support[rank]
            stages.append((first, gradient))
        else:
            first.update(gradient)
            stages.append((first,))
    return stages, ids, sparse


class PeerLibrary:
    """CPU fake NCCL with distinct complete-view peer inputs at each call."""
    def __init__(self, stages, rank):
        self.rounds, self.rank = stages, rank
        self.stage, self.index, self.calls = -1, 0, []

    def ncclGroupStart(self):
        self.stage += 1
        self.index = 0
        return 0

    def ncclGroupEnd(self):
        expected = sum(bool(value.size) for value in self.rounds[self.rank][self.stage].values())
        assert self.index == expected
        return 0

    def ncclAllReduce(self, source, destination, count, dtype, operation, comm, stream):
        keys = [key for key, value in self.rounds[self.rank][self.stage].items() if value.size]
        name = keys[self.index]
        self.index += 1
        ctype = ctypes.c_float if dtype == _NCCL_FLOAT32 else ctypes.c_uint32
        src = np.ctypeslib.as_array((ctype * count).from_address(source))
        dst = np.ctypeslib.as_array((ctype * count).from_address(destination))
        np.testing.assert_array_equal(src, self.rounds[self.rank][self.stage][name])
        assert source != destination
        arrays = np.stack([value[self.stage][name] for value in self.rounds])
        dst[:] = { _NCCL_SUM: lambda: arrays.sum(axis=0, dtype=arrays.dtype),
                   _NCCL_MAX: lambda: arrays.max(axis=0),
                   _NCCL_MIN: lambda: arrays.min(axis=0)}[operation]()
        self.calls.append((name, count, dtype, operation))
        return 0

    @staticmethod
    def ncclGetErrorString(code):
        return b"mock failure"


def expected(peers):
    result = {"gradients": {key: np.sum([value["gradients"][key] for value in peers], axis=0)
                            for key in GRADIENT_KEYS}}
    for key in ("appearance_valid", "geometry_valid", "rgb_loss_sum", "distortion_loss_sum"):
        result[key] = np.sum([value[key] for value in peers], axis=0)
    for key in ("background_carve_valid", *MAX_COUNTER_KEYS):
        result[key] = np.zeros_like(peers[0][key])
    loss = np.sum([value["loss_sum"] for value in peers], axis=0)
    finite = all(bool(value["finite"][0]) and all(np.all(value[key] == 0) for key in
                  ("background_carve_valid", *MAX_COUNTER_KEYS)) for value in peers)
    return result, loss, finite


def compare(test, output, oracle):
    result, loss, finite = output
    truth, true_loss, true_finite = oracle
    if "voxel_ids" in result:
        # Expansion is an oracle-only operation. Production Adam must consume
        # exactly these K rows and must never reconstruct this full table.
        ids = np.asarray(result["voxel_ids"])
        count = len(truth["appearance_valid"])
        test.assertTrue(np.all(ids < count))
        test.assertTrue(np.all(ids[1:] > ids[:-1]))
        expanded = dict(result)
        expanded["gradients"] = {}
        for key in GRADIENT_KEYS:
            dimension = len(truth["gradients"][key]) // count
            values = np.asarray(result["gradients"][key])
            test.assertEqual(values.size, ids.size * dimension)
            table = np.zeros((count, dimension), np.float32)
            table[ids] = values.reshape(len(ids), dimension)
            expanded["gradients"][key] = table.reshape(-1)
        for key in SUM_COUNTER_KEYS + MAX_COUNTER_KEYS:
            values = np.asarray(result[key])
            test.assertEqual(values.size, ids.size)
            table = np.zeros(count, np.uint32)
            table[ids] = values
            expanded[key] = table
        result = expanded
    for key in GRADIENT_KEYS:
        np.testing.assert_allclose(np.asarray(result["gradients"][key]), truth["gradients"][key],
                                   atol=0., rtol=0., equal_nan=True)
    for key in truth.keys() - {"gradients"}:
        np.testing.assert_array_equal(np.asarray(result[key]), truth[key])
    np.testing.assert_array_equal(np.asarray(loss), true_loss)
    test.assertEqual(bool(np.asarray(finite)[0]), true_finite)


class CpuTests(unittest.TestCase):
    def test_compact_output_keeps_ids_counts_and_gradient_rows(self):
        class NoGradientExpansionDr(ArrayDr):
            @staticmethod
            def zeros(array_type, size):
                assert array_type is not _float or size == 0, "Direct output allocated a zeroed gradient table"
                return ArrayDr.zeros(array_type, size)

        for count, dense, invalid, nan, empty_union in (
                (32, False, False, False, False), (32, True, False, False, False),
                (32, False, True, False, False), (32, False, False, True, False),
                (32, False, False, False, True), (0, False, False, False, False)):
            peers = inputs(count, dense=dense, invalid=invalid, nan=nan)
            if empty_union:
                for peer in peers:
                    for gradient in peer["gradients"].values():
                        gradient.fill(0)
                    for key in SUM_COUNTER_KEYS + MAX_COUNTER_KEYS:
                        peer[key].fill(0)
            for mode in ("auto", "dense"):
                stages, ids, sparse = packets(peers, mode)
                for rank in range(len(peers)):
                    with self.subTest(count=count, dense=dense, invalid=invalid,
                                      nan=nan, empty_union=empty_union, mode=mode, rank=rank):
                        collective = _fixture(len(peers))
                        collective.rank, collective.dr = rank, NoGradientExpansionDr()
                        collective._lib = PeerLibrary(stages, rank)
                        local = peers[rank]
                        result = collective.reduce_batch(local, local["loss_sum"], local["finite"],
                            rgb_only=True, mode=mode, compact_output=True)
                        compare(self, result, expected(peers))
                        stats = collective.statistics()
                        self.assertEqual("voxel_ids" in result[0], sparse)
                        self.assertEqual(stats["rgb_last_output_layout"], "compact" if sparse else "dense")
                        self.assertEqual(stats.get("compact_direct_batches", 0), int(sparse))
                        avoided = sum(value.size for value in local["gradients"].values()) * 4
                        self.assertEqual(stats.get("expanded_gradient_bytes_avoided", 0), avoided if sparse else 0)
                        if sparse:
                            np.testing.assert_array_equal(result[0]["voxel_ids"], ids)
                            if not empty_union:
                                self.assertIn(11, ids)  # Valid zero-gradient row.
                                self.assertIn(13, ids)  # Invalid nonzero row is guarded, not discarded.
                                if nan:
                                    self.assertIn(2, ids)  # NaN must survive support selection.

    def test_dense_completes_external_writes_before_consumption(self):
        peers = inputs()
        stages, _, _ = packets(peers, "dense")
        collective = _fixture(len(peers))
        library = PeerLibrary(stages, 0)
        pending = []

        class DelayedLibrary:
            ncclGroupStart = library.ncclGroupStart
            ncclGroupEnd = staticmethod(lambda: 0)
            ncclGetErrorString = library.ncclGetErrorString

            @staticmethod
            def ncclAllReduce(*args):
                pending.append(args)
                return 0

        class DelayedDr(ArrayDr):
            @staticmethod
            def empty(array_type, size):
                # Make a premature consumer fail deterministically, rather
                # than depending on arbitrary contents of NumPy empty().
                return np.full(size, np.nan if array_type is _float else 0,
                               dtype=np.float32 if array_type is _float else np.uint32)

            def sync_thread(self):
                for args in pending:
                    library.ncclAllReduce(*args)
                library.ncclGroupEnd()
                pending.clear()
                super().sync_thread()

        collective.dr = DelayedDr()
        collective._lib = DelayedLibrary()
        # An available owner queue must not suppress dense's completion wait.
        retained = []
        collective._owners = SimpleNamespace(reap=lambda: None,
            retain=lambda owners, size: retained.append((owners, size)))
        local = peers[0]
        result = collective.reduce_batch(local, local["loss_sum"], local["finite"],
                                         rgb_only=True, mode="dense")
        compare(self, result, expected(peers))
        self.assertEqual(collective.dr.syncs, 1)
        self.assertEqual(collective._stats["rgb_explicit_syncs"], 1)
        self.assertEqual(pending, [])
        self.assertEqual(retained, [])

    def test_cleanup_failure_keeps_owners_and_can_be_retried(self):
        collective = _fixture(3)
        calls = []
        pending = object()
        collective._launch_owners = pending

        class Owners:
            fail = True

            def finish(self):
                calls.append("wait")
                if self.fail:
                    raise RuntimeError("event wait failed")

        owners = Owners()
        collective._owners = owners
        collective._lib.ncclCommAbort = lambda comm: (calls.append("abort") or 0)
        collective._lib.ncclCommDestroy = lambda comm: (calls.append("destroy") or 0)
        with self.assertRaisesRegex(RuntimeError, "event wait failed"):
            collective.close()
        self.assertEqual(calls, ["wait", "abort"])
        self.assertFalse(collective._closed)
        self.assertFalse(collective._comm)
        self.assertIs(collective._launch_owners, pending)
        owners.fail = False
        collective.close(abort=True)
        self.assertTrue(collective._closed)
        self.assertIsNone(collective._launch_owners)
        self.assertEqual(collective.dr.syncs, 1)

    def test_group_failure_retains_buffers_until_abort(self):
        collective = _fixture(3)
        calls = []
        collective._lib = SimpleNamespace(
            ncclGroupStart=lambda: 0, ncclGroupEnd=lambda: 0,
            ncclAllReduce=lambda *args: 7, ncclGetErrorString=lambda code: b"injected launch failure",
            ncclCommAbort=lambda comm: (calls.append("abort") or 0))
        with self.assertRaisesRegex(RuntimeError, "injected launch failure"):
            collective._allreduce_fields({"sample": (_float([1, 2]), _NCCL_FLOAT32, _NCCL_SUM)}, wait=False)
        self.assertTrue(collective._failed)
        self.assertIsNotNone(collective._launch_owners)
        collective.close(abort=True)
        self.assertEqual(calls, ["abort"])
        self.assertIsNone(collective._launch_owners)
        self.assertEqual(collective.dr.syncs, 1)

    def test_dense_compact_tail_zero_invalid_and_empty(self):
        for count, dense, invalid, nan in ((32, False, False, False), (32, True, False, False),
                (32, False, True, False), (32, False, False, True), (0, False, False, False)):
            peers = inputs(count, dense=dense, invalid=invalid, nan=nan)
            for mode in ("auto", "dense"):
                stages, ids, sparse = packets(peers, mode)
                for rank in range(3):
                    with self.subTest(count=count, dense=dense, invalid=invalid, nan=nan, mode=mode, rank=rank):
                        collective = _fixture(3)
                        collective.rank, collective.dr = rank, ArrayDr()
                        collective._lib = PeerLibrary(stages, rank)
                        local = peers[rank]
                        result = collective.reduce_batch(local, local["loss_sum"], local["finite"],
                                                         rgb_only=True, mode=mode)
                        compare(self, result, expected(peers))
                        stats = collective.statistics()
                        self.assertEqual(stats["rgb_last_mode"], "compact" if sparse else "dense")
                        self.assertEqual(stats["rgb_count_readbacks"] if count and mode == "auto" else 0,
                                         int(count > 0 and mode == "auto"))
                        if sparse:
                            self.assertLess(stats["rgb_payload_bytes"], stats["rgb_dense_equivalent_bytes"])
                            self.assertEqual(stats["rgb_last_union_voxels"], len(ids))
                        # No nonzero valid count is discarded, even when every
                        # gradient on that row is zero; unexpected gradients
                        # on row 13 also survive the compact representation.
                        if count and not invalid:
                            self.assertEqual(int(result[0]["appearance_valid"][11]), 1 if not dense else 3)

    def test_bad_mode_and_gradient_shape(self):
        for malformed in ("mode", "shape", "empty_group"):
            collective = _fixture(3)
            collective.dr = ArrayDr()
            local = inputs()[0]
            mode = "unsupported" if malformed == "mode" else "auto"
            if malformed == "shape":
                local["gradients"]["centers"] = _float([1.])
            if malformed == "empty_group":
                local["gradients"]["centers"] = _float([])
            with self.assertRaises(ValueError):
                collective.reduce_batch(local, 0., True, rgb_only=True, mode=mode)
            self.assertEqual(collective._lib.calls, [])

    def test_single_rank_no_communication_or_count_readback(self):
        local = inputs()[0]
        collective = _fixture(1)
        collective.dr = ArrayDr()
        compare(self, collective.reduce_batch(local, local["loss_sum"], local["finite"], rgb_only=True),
                expected([local]))
        self.assertEqual(collective.statistics()["calls"], 0)
        self.assertEqual(collective.dr.syncs, 0)
        self.assertNotIn("rgb_count_readbacks", collective.statistics())


def cuda_reduction(peers, mode, rank=0, *, check_inputs=True, compact_output=False):
    stages, ids, sparse = packets(peers, mode)
    collective = _fixture(len(peers))
    collective.mi, collective.dr, collective.rank = mi, dr, rank
    stage = 0

    def simulated_reduce(fields, *, wait):
        nonlocal stage
        output = {}
        for key, (local, dtype, operation) in fields.items():
            array_type = mi.Float if dtype == _NCCL_FLOAT32 else mi.UInt32
            if check_inputs:
                np.testing.assert_array_equal(np.asarray(local), stages[rank][stage][key])
            values = [local if peer_rank == rank else array_type(peer[stage][key])
                      for peer_rank, peer in enumerate(stages)]
            value = values[0]
            for peer in values[1:]:
                if operation == _NCCL_SUM:
                    value = value + peer
                elif operation == _NCCL_MAX:
                    value = dr.maximum(value, peer)
                else:
                    value = dr.minimum(value, peer)
            dr.eval(value)
            output[key] = value
        stage += 1
        return output

    collective._allreduce_fields = simulated_reduce
    local = {key: (mi.Float(value) if key.endswith("loss_sum") else mi.UInt32(value))
             for key, value in peers[rank].items() if key != "gradients"}
    local["gradients"] = {key: mi.Float(value) for key, value in peers[rank]["gradients"].items()}
    result = collective.reduce_batch(local, local["loss_sum"], mi.Bool(local["finite"] != 0),
                                     rgb_only=True, mode=mode, compact_output=compact_output)
    return result, collective.statistics()


class CudaTests(unittest.TestCase):
    def test_cuda_union_gather_scatter(self):
        for count, dense in ((32, False), (32, True), (0, False)):
            peers = inputs(count, dense=dense)
            for mode in ("auto", "dense"):
                stages, ids, sparse = packets(peers, mode)
                for rank in range(3):
                    result, stats = cuda_reduction(peers, mode, rank)
                    compare(self, result, expected(peers))
                    self.assertEqual(stats["rgb_last_mode"], "compact" if sparse else "dense")

    def test_cuda_direct_compact_output(self):
        for count, dense in ((32, False), (32, True), (0, False)):
            peers = inputs(count, dense=dense)
            for mode in ("auto", "dense"):
                stages, ids, sparse = packets(peers, mode)
                for rank in range(3):
                    result, stats = cuda_reduction(peers, mode, rank, compact_output=True)
                    compare(self, result, expected(peers))
                    self.assertEqual("voxel_ids" in result[0], sparse)
                    if sparse:
                        np.testing.assert_array_equal(np.asarray(result[0]["voxel_ids"]), ids)

    def test_compact_and_dense_consumed_by_adam_match_oracle_full_and_tail(self):
        from reconstruction_common import restore_model
        from reconstruction_optimizer import FalcorOptimizer, OptimizerConfig
        import test_reconstruction_optimization as optimizer_tests

        optimizer_tests.np = np
        payload = optimizer_tests.seed_checkpoint(cells=tuple(range(32)), sh=9)
        payload["model_metadata"]["opacity_sh_count"] = 1
        payload["model_state"]["opacity_sh"] = np.zeros(32, np.float32)
        models = [restore_model(copy.deepcopy(payload), mi) for _ in range(3)]
        config = OptimizerConfig(opacity_warmup_epochs=0, opacity_ramp_epochs=0)
        optimizers = [FalcorOptimizer(model, copy.deepcopy(config), sparse_updates=index < 2)
                      for index, model in enumerate(models)]
        for step, actual_views in enumerate((3, 1, 3, 1)):
            peers = inputs()
            for rank, value in enumerate(peers):
                for key in value["gradients"]:
                    value["gradients"][key] *= 1 if step % 2 == 0 else -.01
                if step == 0 and rank == 0:
                    value["gradients"]["radiance_sh"][11 * 27:12 * 27] = .25
                if rank >= actual_views:
                    for key, array in value.items():
                        if key == "gradients":
                            for gradient in array.values():
                                gradient.fill(0)
                        elif key != "finite":
                            array.fill(0)
            oracle, _, _ = expected(peers)
            reference = {key: mi.UInt32(value) for key, value in oracle.items()
                         if key not in ("gradients", "rgb_loss_sum", "distortion_loss_sum")}
            reference["gradients"] = {key: mi.Float(value) for key, value in oracle["gradients"].items()}
            for index, mode in enumerate(("auto", "dense")):
                (reduced, _, finite), stats = cuda_reduction(peers, mode, check_inputs=False, compact_output=True)
                # Submit Adam directly, before converting any reduced output to
                # host. Auto retains the compact representation through Adam.
                self.assertEqual("voxel_ids" in reduced, mode == "auto")
                optimizers[index].step(models[index], reduced, sample_count=actual_views,
                                       epoch=35, update_allowed=finite)
            optimizers[2].step(models[2], reference, sample_count=actual_views,
                               epoch=35, update_allowed=mi.Bool(True))
            reference_state = optimizers[2].state_dict()["groups"]
            for index in (0, 1):
                for key, value in models[index].snapshot().items():
                    np.testing.assert_allclose(np.asarray(value), np.asarray(models[2].snapshot()[key]),
                                               atol=3e-7, rtol=3e-5, err_msg=key)
                states = optimizers[index].state_dict()["groups"]
                for group in reference_state:
                    for key in reference_state[group]:
                        np.testing.assert_allclose(states[group][key], reference_state[group][key],
                                                   atol=3e-7, rtol=3e-5, err_msg=f"{group}.{key}")
            # A valid zero row must keep its moment decay and clock, while the
            # deliberately nonzero gradient in invalid row 13 must not age it.
            self.assertEqual(int(reference_state["radiance_sh"]["steps"][11]), step + 1)
            self.assertEqual(int(reference_state["centers"]["steps"][13]), 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(CpuTests))
    if not result.wasSuccessful():
        return 1
    if args.cuda:
        global mi, dr
        from preview import RuntimeConfig
        from reconstruction_common import initialize_runtime
        with tempfile.TemporaryDirectory(prefix="mitsuba-rgb-collective-") as directory:
            mi, dr, _, handles = initialize_runtime(RuntimeConfig(), Path(directory), {})
            result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(CudaTests))
            dr.sync_thread()
            if not result.wasSuccessful():
                return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
