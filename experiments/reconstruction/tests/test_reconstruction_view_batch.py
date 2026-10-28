"""View-batch scheduling/resume tests, plus an optional single-GPU Adam oracle.

The CUDA oracle renders different small cameras at the same parameter state,
then compares summed-gradient/B Adam against an independently averaged host
reference. It does not claim to test physical multi-GPU NCCL interoperability.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from reconstruction_collective import GRADIENT_KEYS, MAX_COUNTER_KEYS, SUM_COUNTER_KEYS
from reconstruction_view_batch import (batch_optimizer_steps_per_epoch, batch_view_seeds,
    iter_view_batches, validate_batch_resume, view_batch_protocol, zero_view_partial)
from reconstruction_batch_epoch import _finite_accumulator, run_view_batch_epoch


class SchedulingTests(unittest.TestCase):
    def test_batch_transport_configuration_preserves_update_protocol(self):
        import train
        for mode in ("auto", "dense"):
            with patch.object(sys, "argv", ["train.py", "--batch-reduction", mode]):
                args = train.parse_arguments()
            config = copy.deepcopy(train.CONFIG)
            train.apply_arguments(config, args, {})
            self.assertEqual(config.training.batch_reduction, mode)
            self.assertEqual(config.training.num_epochs, train.CONFIG.training.num_epochs)
            self.assertEqual(config.optimizer, train.CONFIG.optimizer)
            self.assertEqual(view_batch_protocol(4)["views_per_update"], 4)

    def test_order_and_short_tail_without_duplicate_views(self):
        order = np.array([4, 0, 2, 1, 3])
        self.assertEqual(list(iter_view_batches(order, 3)), [(4, 0, 2), (1, 3)])
        self.assertEqual(list(iter_view_batches(order, 8)), [(4, 0, 2, 1, 3)])
        self.assertEqual(list(iter_view_batches(order, 1)), [(4,), (0,), (2,), (1,), (3,)])
        self.assertEqual(list(iter_view_batches([], 2)), [])
        for size in (0, -1, True, 2.5):
            with self.assertRaises(ValueError):
                list(iter_view_batches(order, size))
        for order in ([0, 0], [0, -1], [0, True], [0, 1.5]):
            with self.assertRaises(ValueError):
                list(iter_view_batches(order, 2))

    def test_canonical_seeds_agree_across_ranks_and_resume(self):
        first, second = np.random.default_rng(42), np.random.default_rng(42)
        first_seeds = batch_view_seeds(first, 3)
        second_seeds = batch_view_seeds(second, 3)
        self.assertEqual(first_seeds, second_seeds)
        self.assertEqual(len(set(first_seeds)), 3)
        # Different image work does not consume the canonical stream.
        a = np.random.default_rng(first_seeds[0]).random((16, 2))
        b = np.random.default_rng(second_seeds[1]).random((16, 2))
        self.assertFalse(np.array_equal(a, b))
        self.assertEqual(first.bit_generator.state, second.bit_generator.state)
        restored = np.random.default_rng()
        restored.bit_generator.state = copy.deepcopy(first.bit_generator.state)
        self.assertEqual(batch_view_seeds(first, 2), batch_view_seeds(restored, 2))
        with self.assertRaises(ValueError):
            batch_view_seeds(first, 0)

    def test_seed_sequence_does_not_depend_on_batch_partition(self):
        first, second = np.random.default_rng(53), np.random.default_rng(53)
        partitioned = batch_view_seeds(first, 3) + batch_view_seeds(first, 2)
        self.assertEqual(partitioned, batch_view_seeds(second, 5))
        self.assertEqual(first.bit_generator.state, second.bit_generator.state)

    def test_resume_validates_optimizer_steps_separately_from_views(self):
        for world_size in (1, 2, 3, 8):
            steps = 4 * batch_optimizer_steps_per_epoch(5, world_size)
            state = {"global_step": 20, "optimizer_step": steps,
                     "update_protocol": view_batch_protocol(world_size)}
            self.assertEqual(validate_batch_resume(state, world_size, 5, 4), steps)
            with self.assertRaises(ValueError):
                validate_batch_resume(dict(state, optimizer_step=steps + 1), world_size, 5, 4)
            with self.assertRaises(ValueError):
                validate_batch_resume(dict(state, global_step=19), world_size, 5, 4)
            with self.assertRaises(ValueError):
                validate_batch_resume(state, world_size + 1, 5, 4)

    def test_legacy_resume_is_single_view_only(self):
        self.assertEqual(validate_batch_resume({"global_step": 15}, 1, 5, 3), 15)
        self.assertEqual(validate_batch_resume({"global_step": 0}, 1, 5, 0), 0)
        for size in (2, 3):
            with self.assertRaisesRegex(ValueError, "Legacy"):
                validate_batch_resume({"global_step": 15}, size, 5, 3)
        with self.assertRaisesRegex(ValueError, "missing optimizer_step"):
            validate_batch_resume({"global_step": 15, "update_protocol": view_batch_protocol(2)}, 2, 5, 3)
        with self.assertRaises(ValueError):
            validate_batch_resume({"global_step": 15, "optimizer_step": True}, 1, 5, 3)

    def test_zero_rank_has_native_widths_and_no_evidence(self):
        mi = SimpleNamespace(Float=np.float32, UInt32=np.uint32, Bool=np.bool_)
        dr = SimpleNamespace(zeros=lambda dtype, count: np.zeros(count, dtype=dtype))
        for count in (0, 7):
            model = SimpleNamespace(active_voxel_count=count, sh_count=9, opacity_sh_count=1)
            out, loss, finite, raw_alpha, alpha = zero_view_partial(model, mi, dr)
            self.assertEqual(set(out), {"gradients", *SUM_COUNTER_KEYS, *MAX_COUNTER_KEYS})
            widths = (27, 1, 3, 3, 3)
            for key, width in zip(GRADIENT_KEYS, widths):
                self.assertEqual(out["gradients"][key].shape, (count * width,))
                np.testing.assert_array_equal(out["gradients"][key], 0)
            self.assertEqual(float(loss), 0)
            self.assertTrue(finite)
            self.assertIsNone(raw_alpha)
            self.assertIsNone(alpha)

    def test_finite_guard_rejects_local_negative_and_global_overflow(self):
        dr = SimpleNamespace(all=np.all, isfinite=np.isfinite, width=np.size)
        baseline = {"gradients": {"radiance_sh": np.array([1.], np.float32)}}
        for loss in (-1., np.nan, np.inf):
            self.assertFalse(_finite_accumulator(baseline, np.array([loss]), np.bool_(True), dr))
        invalid = {"gradients": {"radiance_sh": np.array([np.inf], np.float32)}}
        self.assertFalse(_finite_accumulator(invalid, np.array([1.]), np.bool_(True), dr))
        self.assertTrue(_finite_accumulator(baseline, np.array([0.]), np.bool_(True), dr))


def _fake_epoch(losses=(1., 3., 9.), *, gradient_nan=False, sync=False, rank=0,
                dssim_weight=0., distortion_weight=0., dssim_nan=False):
    """Run the actual epoch driver using CPU arrays and simulated peer views."""
    import train

    def array(dtype):
        return lambda value: np.atleast_1d(np.asarray(value, dtype=dtype))

    mi = SimpleNamespace(Float=array(np.float32), UInt32=array(np.uint32), Bool=array(np.bool_))
    dr = SimpleNamespace(all=np.all, isfinite=np.isfinite, width=np.size, concat=np.concatenate,
        eval=lambda *args: None,
        zeros=lambda kind, count: np.zeros(count, dtype=np.float32 if kind is mi.Float else np.uint32))
    model = SimpleNamespace(active_voxel_count=1, sh_count=1, opacity_sh_count=1,
                            growth_candidate_count=1, value=0.)
    config = SimpleNamespace(
        training=SimpleNamespace(rgb_only=True, profile_interval_epochs=0, profile_views=0,
                                 sync_view_checks=sync, num_epochs=1, freeze_native=True),
        render=SimpleNamespace(width=1, height=1, spp=1),
        gradient=SimpleNamespace(distortion_loss_weight=distortion_weight, dssim_loss_weight=dssim_weight),
        schedule=SimpleNamespace(enabled=False))
    traced, votes, updates, broadcasts, submitted = [], [], [], [], []
    gradient_values = (2., 4., 6.)
    evidence_values = (1., -1., 2.)
    dssim_values = (.2, .4, .8)
    distortion_values = (.1, .2, .3)

    def partial(index):
        out, _, finite, _, _ = zero_view_partial(model, mi, dr)
        for key in GRADIENT_KEYS:
            out["gradients"][key][:] = gradient_values[index]
        if gradient_nan and index == 1:
            out["gradients"]["radiance_sh"][0] = np.nan
        for key in SUM_COUNTER_KEYS:
            out[key][:] = 1 if key != "background_carve_valid" else 0
        for kind in ("growth", "deletion"):
            out[f"{kind}_gain"] = mi.Float(evidence_values[index])
            out[f"{kind}_observed"] = mi.UInt32(1)
            out[f"{kind}_unknown"] = mi.UInt32(0)
        loss = losses[index]
        if distortion_weight or dssim_weight:
            out["rgb_loss_sum"] = mi.Float(loss)
            out["distortion_loss_sum"] = mi.Float(distortion_values[index] if distortion_weight else 0.)
            loss += distortion_weight * distortion_values[index]
        if dssim_weight:
            out["dssim_loss_sum"] = mi.Float(np.nan if dssim_nan and index == 1 else dssim_values[index])
            loss += dssim_weight * dssim_values[index]
        traced.append((index, model.value))
        return out, mi.Float(loss), finite, None, None

    def compute(model, view, *args, **kwargs):
        submitted.append(view["index"])
        return partial(view["index"])

    class Collective:
        world_size = 2

        def __init__(self):
            self.rank, self.batch_index, self.sources = rank, 0, {}

        def statistics(self):
            return dict(calls=0, payload_bytes=0, seconds=0.)

        def reduce_batch(self, accumulator, loss, finite, *, rgb_only=False, mode="auto", compact_output=False):
            assert rgb_only and mode in ("auto", "dense") and compact_output
            self.indices = list(range(self.batch_index * 2, min(self.batch_index * 2 + 2, 3)))
            self.sources = {source: partial(index) for source, index in enumerate(self.indices)}
            neutral = zero_view_partial(model, mi, dr)
            component_keys = (("rgb_loss_sum", "distortion_loss_sum") if distortion_weight or dssim_weight else ())
            if dssim_weight:
                component_keys += ("dssim_loss_sum",)
            neutral[0].update({key: mi.Float(0.) for key in component_keys})
            items = []
            for source in range(2):
                if source == rank:
                    items.append((accumulator, loss, finite, None, None))
                else:
                    item = self.sources.get(source, neutral)
                    items.append((item[0], item[1], _finite_accumulator(item[0], item[1], item[2], dr), None, None))
            out = zero_view_partial(model, mi, dr)[0]
            for key in GRADIENT_KEYS:
                out["gradients"][key] = sum(item[0]["gradients"][key] for item in items)
            for key in SUM_COUNTER_KEYS:
                out[key] = sum(item[0][key] for item in items)
            for key in MAX_COUNTER_KEYS:
                out[key] = np.maximum(items[0][0][key], items[1][0][key])
            for key in component_keys:
                out[key] = sum(item[0][key] for item in items)
            self.batch_index += 1
            return out, sum(item[1] for item in items), mi.Bool(all(item[2] for item in items))

        def broadcast_arrays(self, fields, *, root, uint_keys):
            broadcasts.append((self.batch_index, root))
            self.assert_types = set(uint_keys) == {key for key in fields if not key.endswith("gain")}
            return {key: self.sources[root][0][key].copy() for key in fields}

    class Optimizer:
        healthy = True

        def step(self, model, out, *, sample_count, update_allowed, **kwargs):
            self.healthy = self.healthy and bool(update_allowed[0])
            updates.append((sample_count, float(out["gradients"]["opacity_sh"][0]), self.healthy))
            if self.healthy:
                model.value += float(out["gradients"]["opacity_sh"][0]) / sample_count
            return {"accepted": mi.Bool(self.healthy)}

    collective, optimizer = Collective(), Optimizer()
    topology = SimpleNamespace(
        observe_rgb_growth=lambda model, out, *, view_id, **kwargs:
            votes.append(("growth", view_id, float(out["growth_gain"][0]), model.value)),
        observe_rgb_deletion=lambda model, out, *, view_id, **kwargs:
            votes.append(("deletion", view_id, float(out["deletion_gain"][0]), model.value)))
    views = [dict(index=index, reference_path=str(index)) for index in range(3)]
    kwargs = dict(model=model, config=config, views=views, order=np.arange(3), epoch=0,
        optimizer=optimizer, topology=topology,
        sampler=SimpleNamespace(enabled=False, sample=lambda *args: (None, None, None)),
        rng=np.random.default_rng(1), mi=mi, dr=dr, np=np,
        reference_cache=SimpleNamespace(get=lambda path: np.zeros((1, 1, 3), np.float32)),
        gpu_reference_cache=SimpleNamespace(get=lambda *args: None), native_replay=None,
        collective=collective, distributed=None, primary=rank == 0, ray_count=1,
        alpha_growth_requested=False, deletion_evidence_collected=True,
        rgb_growth_requested=True, rgb_deletion_enabled=True, training_resolution=8,
        entering_resolution=False, processed_views=0, global_step=0, optimizer_step=0,
        profile_path=None, performance=None,
        progress_line=SimpleNamespace(update=lambda *args, **kwargs: None))
    with patch.object(train, "compute_view_shard", compute):
        try:
            result = run_view_batch_epoch(**kwargs)
        except RuntimeError as error:
            return SimpleNamespace(error=error, model=model, updates=updates)
    return SimpleNamespace(result=result, model=model, updates=updates, votes=votes,
                           broadcasts=broadcasts, traced=traced, submitted=submitted,
                           uint_keys_correct=collective.assert_types)


class EpochDriverTests(unittest.TestCase):
    def test_dssim_components_weight_actual_tail_and_keep_rgb_separate(self):
        for rank in (0, 1):
            for distortion in (0., .2):
                for sync in (False, True):
                    with self.subTest(rank=rank, distortion=distortion, sync=sync):
                        trial = _fake_epoch(dssim_weight=.1, distortion_weight=distortion, rank=rank, sync=sync)
                        self.assertAlmostEqual(trial.result["epoch_rgb_loss"], 13.)
                        self.assertAlmostEqual(trial.result["epoch_dssim_loss"], 1.4, places=6)
                        self.assertAlmostEqual(trial.result["epoch_distortion_loss"], .6 if distortion else 0., places=6)
                        self.assertAlmostEqual(trial.result["epoch_loss"], 13.14 + distortion * .6, places=5)
                        self.assertEqual(trial.updates, [(2, 6., True), (1, 6., True)])

    def test_nonfinite_dssim_component_blocks_update_even_when_total_is_finite(self):
        trial = _fake_epoch(dssim_weight=.1, dssim_nan=True)
        self.assertIsInstance(trial.error, RuntimeError)
        self.assertEqual(trial.model.value, 0.)
        self.assertTrue(all(not accepted for _, _, accepted in trial.updates))

    def test_per_view_votes_do_not_cancel_and_adam_runs_once_per_batch(self):
        trial = _fake_epoch()
        self.assertTrue(trial.uint_keys_correct)
        self.assertEqual(trial.updates, [(2, 6., True), (1, 6., True)])
        self.assertEqual(trial.model.value, 9.)
        self.assertEqual(trial.broadcasts, [(1, 0), (1, 1), (2, 0)])
        self.assertEqual(trial.votes, [
            ("growth", 0, 1., 0.), ("deletion", 0, 1., 0.),
            ("growth", 1, -1., 0.), ("deletion", 1, -1., 0.),
            ("growth", 2, 2., 3.), ("deletion", 2, 2., 3.)])
        self.assertEqual(trial.result["epoch_loss"], 13.)
        self.assertEqual(trial.result["epoch_rgb_loss"], 13.)
        self.assertEqual(trial.result["global_step"], 3)
        self.assertEqual(trial.result["optimizer_step"], 2)
        self.assertEqual(trial.result["processed_views"], 3)

    def test_inactive_tail_rank_joins_collectives_without_rendering(self):
        trial = _fake_epoch(rank=1)
        self.assertEqual(trial.submitted, [1])
        self.assertEqual(trial.updates, [(2, 6., True), (1, 6., True)])
        self.assertEqual(trial.broadcasts, [(1, 0), (1, 1), (2, 0)])
        self.assertEqual(trial.votes, [])
        self.assertEqual(trial.result["epoch_loss"], 13.)

    def test_negative_local_loss_cannot_be_hidden_by_positive_peer(self):
        for rank in (0, 1):
            with self.subTest(rank=rank):
                trial = _fake_epoch(losses=(-1., 20., 9.), rank=rank)
                self.assertIsInstance(trial.error, RuntimeError)
                self.assertEqual(trial.model.value, 0.)
                self.assertTrue(all(not accepted for _, _, accepted in trial.updates))

    def test_peer_nan_blocks_this_and_later_updates(self):
        trial = _fake_epoch(gradient_nan=True)
        self.assertIsInstance(trial.error, RuntimeError)
        self.assertEqual(trial.model.value, 0.)
        self.assertTrue(all(not accepted for _, _, accepted in trial.updates))

    def test_sync_diagnostic_does_not_change_epoch_weighting(self):
        trial = _fake_epoch(sync=True)
        self.assertEqual(trial.result["epoch_loss"], 13.)
        self.assertEqual(trial.updates, [(2, 6., True), (1, 6., True)])


class CudaBatchTests(unittest.TestCase):
    def test_whole_view_average_matches_one_adam_step_including_tail(self):
        from reconstruction_common import UniformSampler, restore_model
        from reconstruction_optimizer import FalcorOptimizer
        from train import compute_view_shard
        from test_reconstruction_performance import fixture

        config, payload, view, rgba = fixture(np)
        config.training.rgb_only = True
        config.model.opacity_sh_count = 1
        payload["model_metadata"]["opacity_sh_count"] = 1
        payload["model_state"]["opacity_sh"] = payload["model_state"]["opacity_sh"][:1].copy()
        config.gradient.distortion_loss_weight = .1
        models = [restore_model(copy.deepcopy(payload), mi) for _ in range(2)]
        optimizers = [FalcorOptimizer(model, copy.deepcopy(config.optimizer)) for model in models]
        sampler = UniformSampler(np)
        rng = np.random.default_rng(987)
        views = []
        for index in range(5):
            candidate = copy.deepcopy(view)
            candidate["index"] = index
            candidate["camera"]["position"][0] += index * .02
            if index == 4:
                # No hits still constitutes a real view in the tail divisor.
                candidate["camera"]["position"][0] = 100.
            views.append(candidate)
        snapshots = []
        for indices in iter_view_batches(range(5), 3):
            seeds = batch_view_seeds(rng, len(indices))
            before = {key: np.array(value) for key, value in models[0].snapshot().items()}
            partials = []
            for index, seed in zip(indices, seeds):
                reference = rgba[..., :3].copy() + np.float32(index * .01)
                offsets, pdf, token = sampler.sample(index, reference, np.random.default_rng(seed))
                partial = compute_view_shard(models[0], views[index], reference, config,
                    offsets, pdf, token, mi, dr, collect_deletion_evidence=False,
                    collect_growth_evidence=False)
                self.assertTrue(bool(partial[2][0]))
                partials.append(partial)
            # The camera computations must not have changed any parameter.
            for key, value in models[0].snapshot().items():
                np.testing.assert_array_equal(value, before[key])
            while len(partials) < 3:
                zero = zero_view_partial(models[0], mi, dr)
                zero[0].update(rgb_loss_sum=mi.Float(0), distortion_loss_sum=mi.Float(0))
                partials.append(zero)
            reduced = zero_view_partial(models[0], mi, dr)[0]
            for key in GRADIENT_KEYS:
                reduced["gradients"][key] = sum(item[0]["gradients"][key] for item in partials)
            for key in SUM_COUNTER_KEYS:
                reduced[key] = sum(item[0][key] for item in partials)
            for key in MAX_COUNTER_KEYS:
                for item in partials:
                    reduced[key] = dr.maximum(reduced[key], item[0][key])
            # Independent CPU reference: average unnormalized view sums once.
            reference = {"gradients": {key: mi.Float(
                sum(np.asarray(item[0]["gradients"][key], dtype=np.float64) for item in partials)
                / len(indices)) for key in GRADIENT_KEYS}}
            reference.update({key: mi.UInt32(np.asarray(reduced[key]))
                              for key in (*SUM_COUNTER_KEYS, *MAX_COUNTER_KEYS)})
            optimizers[0].step(models[0], reduced, sample_count=len(indices),
                               epoch=35, update_allowed=mi.Bool(True))
            optimizers[1].step(models[1], reference, sample_count=1,
                               epoch=35, update_allowed=mi.Bool(True))
            for key in models[0].snapshot():
                np.testing.assert_allclose(models[0].snapshot()[key], models[1].snapshot()[key],
                                           rtol=4e-5, atol=4e-7, err_msg=key)
            states = [optimizer.state_dict()["groups"] for optimizer in optimizers]
            for group in states[0]:
                for field in states[0][group]:
                    np.testing.assert_allclose(states[0][group][field], states[1][group][field],
                                               rtol=4e-5, atol=4e-7, err_msg=f"{group}.{field}")
            snapshots.append(states[0])
        self.assertEqual(optimizers[0].statistics()["calls"], 2)
        for group in snapshots[-1].values():
            self.assertLessEqual(int(np.max(group["steps"])), 2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda", action="store_true", help="Also test rendered view gradients and CUDA Adam")
    parser.add_argument("--build-dir")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromTestCase(case)
                               for case in (SchedulingTests, EpochDriverTests))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        return 1
    if args.cuda:
        from preview import RuntimeConfig
        from reconstruction_common import initialize_runtime
        global mi, dr
        with tempfile.TemporaryDirectory(prefix="mitsuba-view-batch-") as temporary:
            runtime = RuntimeConfig(args.build_dir or RuntimeConfig().build_dir, args.gpu)
            mi, dr, _, handles = initialize_runtime(runtime, Path(temporary), {})
            result = unittest.TextTestRunner(verbosity=2).run(
                unittest.defaultTestLoader.loadTestsFromTestCase(CudaBatchTests))
            dr.sync_thread()
            if not result.wasSuccessful():
                return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
