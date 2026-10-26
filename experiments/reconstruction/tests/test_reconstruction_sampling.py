"""CPU-only exact-sequence and lifecycle checks for uniform sampling prefetch."""

from __future__ import annotations

import copy
import gc
from pathlib import Path
import sys
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import weakref

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reconstruction_common import UniformSampler
from reconstruction_sampling import PrefetchedUniformSampler


class SamplingTests(unittest.TestCase):
    def setUp(self):
        self.reference = SimpleNamespace(shape=(17, 23, 3))
        self.synchronous = UniformSampler(np)

    def assert_sample(self, sampler, rng, oracle, reference=None, view=0):
        reference = self.reference if reference is None else reference
        expected = self.synchronous.sample(view, reference, oracle)
        actual = sampler.sample(view, reference, rng)
        np.testing.assert_array_equal(actual[0], expected[0])
        np.testing.assert_array_equal(actual[1], expected[1])
        self.assertIsNone(actual[2])
        self.assertEqual(actual[0].dtype, np.float32)
        self.assertEqual(actual[1].dtype, np.float32)
        self.assertTrue(actual[1].flags.writeable)
        np.testing.assert_equal(rng.bit_generator.state, oracle.bit_generator.state)
        return actual

    def test_exact_sequence_and_cached_uint32(self):
        for workers in (1, 2, 4):
            for uint32_draws in (0, 1, 2):
                with self.subTest(workers=workers, uint32_draws=uint32_draws):
                    rng = np.random.default_rng(194)
                    for _ in range(uint32_draws):
                        rng.integers(0, 2**32, dtype=np.uint32)
                    self.assertEqual(rng.bit_generator.state["has_uint32"], uint32_draws % 2)
                    oracle = copy.deepcopy(rng)
                    with PrefetchedUniformSampler(np, workers=workers) as sampler:
                        for view in range(15):
                            self.assert_sample(sampler, rng, oracle, view=view)
                            self.assertLessEqual(sampler.statistics()["pending"], sampler.lookahead)
                        self.assertEqual(sampler.statistics()["consumed"], 15)
                        self.assertEqual(sampler.statistics()["reused"], 14)
                        np.testing.assert_array_equal(rng.permutation(301), oracle.permutation(301))

    def test_shuffle_invalidates_without_changing_sequence(self):
        rng, oracle = np.random.default_rng(45), np.random.default_rng(45)
        with PrefetchedUniformSampler(np, workers=4) as sampler:
            for epoch in range(4):
                np.testing.assert_array_equal(rng.permutation(31), oracle.permutation(31))
                for view in range(3):
                    self.assert_sample(sampler, rng, oracle, view=view)
            self.assertEqual(sampler.statistics()["invalidations"], 3)

    def test_shape_and_rng_identity_invalidate(self):
        rng, oracle = np.random.default_rng(9), np.random.default_rng(9)
        with PrefetchedUniformSampler(np, workers=2, lookahead=3) as sampler:
            self.assert_sample(sampler, rng, oracle)
            # Equal pixel count but changed image shape is also a new request.
            for shape in ((23, 17, 3), (7, 11, 3), (17, 23, 3)):
                self.assert_sample(sampler, rng, oracle, SimpleNamespace(shape=shape))
            replacement = copy.deepcopy(rng)
            self.assert_sample(sampler, replacement, oracle)
            self.assertEqual(sampler.statistics()["invalidations"], 4)

    def test_close_does_not_consume_pending_samples(self):
        rng, oracle = np.random.default_rng(51), np.random.default_rng(51)
        sampler = PrefetchedUniformSampler(np, workers=4)
        self.assert_sample(sampler, rng, oracle)
        before = copy.deepcopy(rng.bit_generator.state)
        self.assertGreater(sampler.statistics()["pending"], 0)
        sampler.close()
        sampler.close()
        self.assertEqual(rng.bit_generator.state, before)
        self.assertEqual(sampler.statistics()["pending"], 0)
        self.assertFalse(any(thread.name.startswith(sampler._thread_prefix)
                             for thread in threading.enumerate()))
        with self.assertRaises(RuntimeError):
            sampler.sample(0, self.reference, rng)

    def test_resume_and_load_discard_pending(self):
        rng, oracle = np.random.default_rng(812), np.random.default_rng(812)
        with PrefetchedUniformSampler(np, workers=2) as sampler:
            for _ in range(3):
                self.assert_sample(sampler, rng, oracle)
            checkpoint = copy.deepcopy(rng.bit_generator.state)
            self.assertEqual(sampler.state_dict(), {})
            self.assertEqual(rng.bit_generator.state, checkpoint)
            for _ in range(2):
                self.assert_sample(sampler, rng, oracle)
            rng.bit_generator.state = checkpoint
            oracle.bit_generator.state = checkpoint
            sampler.load_state_dict({})
            self.assertEqual(sampler.statistics()["pending"], 0)
            for _ in range(5):
                self.assert_sample(sampler, rng, oracle)
        resumed = np.random.default_rng(1)
        resumed.bit_generator.state = checkpoint
        expected = np.random.default_rng(1)
        expected.bit_generator.state = checkpoint
        with PrefetchedUniformSampler(np, workers=4) as sampler:
            sampler.load_state_dict({})
            for _ in range(5):
                self.assert_sample(sampler, resumed, expected)
        self.assertEqual(resumed.bit_generator.state, rng.bit_generator.state)

    def test_zero_workers_and_unsupported_bit_generators(self):
        generators = [np.random.PCG64, np.random.PCG64DXSM, np.random.MT19937,
                      np.random.Philox, np.random.SFC64]
        for workers in (0, 2):
            for generator in generators:
                if workers and generator is np.random.PCG64:
                    continue
                with self.subTest(workers=workers, generator=generator):
                    rng = np.random.Generator(generator(42))
                    oracle = copy.deepcopy(rng)
                    with PrefetchedUniformSampler(np, workers=workers) as sampler:
                        for _ in range(3):
                            self.assert_sample(sampler, rng, oracle)
                        self.assertEqual(sampler.statistics()["synchronous"], 3)
                        self.assertEqual(sampler.statistics()["submitted"], 0)
                        self.assertIsNone(sampler._executor)

    def test_custom_rng_uses_original_sampler(self):
        class CustomRNG:
            def __init__(self):
                self.calls = []

            def random(self, shape):
                self.calls.append(shape)
                return np.full(shape, .25, dtype=np.float64)

        rng = CustomRNG()
        with PrefetchedUniformSampler(np) as sampler:
            offsets, weights, token = sampler.sample(0, self.reference, rng)
            np.testing.assert_array_equal(offsets, np.full((391, 2), .25, np.float32))
            np.testing.assert_array_equal(weights, np.ones(391, np.float32))
            self.assertIsNone(token)
            self.assertEqual(rng.calls, [(391, 2)])
            self.assertEqual(sampler.statistics()["synchronous"], 1)

    def test_fallback_discards_an_existing_queue(self):
        rng, oracle = np.random.default_rng(40), np.random.default_rng(40)
        with PrefetchedUniformSampler(np, workers=2) as sampler:
            self.assert_sample(sampler, rng, oracle)
            original_state = copy.deepcopy(rng.bit_generator.state)
            other = np.random.Generator(np.random.MT19937(2))
            self.assert_sample(sampler, other, copy.deepcopy(other))
            self.assertEqual(sampler.statistics()["pending"], 0)
            self.assertEqual(rng.bit_generator.state, original_state)
            self.assert_sample(sampler, rng, oracle)

    def test_clear_and_parameter_validation(self):
        for workers in (-1, True, 1.5, "2"):
            with self.assertRaises(ValueError):
                PrefetchedUniformSampler(np, workers=workers)
        for lookahead in (-1, 0, True, 1.5, "2"):
            with self.assertRaises(ValueError):
                PrefetchedUniformSampler(np, lookahead=lookahead)
        rng, oracle = np.random.default_rng(6), np.random.default_rng(6)
        with PrefetchedUniformSampler(np, workers=4, lookahead=1) as sampler:
            self.assert_sample(sampler, rng, oracle)
            sampler.clear()
            self.assertEqual(sampler.statistics()["pending"], 0)
            self.assert_sample(sampler, rng, oracle)
            before = copy.deepcopy(rng.bit_generator.state)
            with self.assertRaises(ValueError):
                sampler.sample(0, SimpleNamespace(shape=(7, 11, 4)), rng)
            with self.assertRaises(ValueError):
                sampler.observe(("unexpected",), None)
            with self.assertRaises(ValueError):
                sampler.load_state_dict({"unexpected": 1})
            sampler.observe(None, None)
            self.assertEqual(rng.bit_generator.state, before)

    def test_destructor_joins_workers(self):
        sampler = PrefetchedUniformSampler(np, workers=2)
        sampler.sample(0, self.reference, np.random.default_rng(9))
        prefix, reference = sampler._thread_prefix, weakref.ref(sampler)
        del sampler
        gc.collect()
        self.assertIsNone(reference())
        self.assertFalse(any(thread.name.startswith(prefix) for thread in threading.enumerate()))

    def test_worker_failure_preserves_rng_and_clears_pending(self):
        rng, oracle = np.random.default_rng(16), np.random.default_rng(16)
        before = copy.deepcopy(rng.bit_generator.state)
        with PrefetchedUniformSampler(np, workers=2) as sampler:
            with patch("reconstruction_sampling._generate", side_effect=RuntimeError("sample failed")):
                with self.assertRaisesRegex(RuntimeError, "sample failed"):
                    sampler.sample(0, self.reference, rng)
            self.assertEqual(rng.bit_generator.state, before)
            self.assertEqual(sampler.statistics()["pending"], 0)
            self.assert_sample(sampler, rng, oracle)

    def test_cli_and_config_validation(self):
        from train import CONFIG, apply_arguments, parse_arguments, validate

        self.assertEqual(CONFIG.training.sampling_workers, 2)
        for workers in (0, 1, 2, 4):
            with self.subTest(workers=workers):
                with patch.object(sys, "argv", ["train.py", "--sampling-workers", str(workers)]):
                    args = parse_arguments()
                config = copy.deepcopy(CONFIG)
                apply_arguments(config, args, {})
                validate(config)
                self.assertEqual(config.training.sampling_workers, workers)
        for workers in (-1, True, 1.5):
            config = copy.deepcopy(CONFIG)
            config.training.sampling_workers = workers
            with self.assertRaisesRegex(ValueError, "training.sampling_workers"):
                validate(config)

    def test_training_sampler_selection_and_refinement_reset(self):
        from reconstruction_common import ContourSampler
        from train import CONFIG, TrainConfig, make_sampler, sampling_statistics

        config = copy.deepcopy(CONFIG)
        # Direct diagnostic callers and the multi-GPU path explicitly retain
        # synchronous sampling unless the training driver selects worker count.
        self.assertIs(type(make_sampler(config, np)), UniformSampler)
        self.assertIs(type(make_sampler(config, np, sampling_workers=0)), UniformSampler)
        legacy = make_sampler(TrainConfig(), np, sampling_workers=2)
        self.assertIsInstance(legacy, ContourSampler)
        self.assertEqual(sampling_statistics(legacy), {"workers": 0})

        rng, oracle = np.random.default_rng(72), np.random.default_rng(72)
        sampler = make_sampler(config, np, sampling_workers=2)
        try:
            self.assert_sample(sampler, rng, oracle)
            before = copy.deepcopy(rng.bit_generator.state)
            # RGB has no contour history to reset; reuse its worker pool and
            # counters while discarding speculative samples at refinement.
            reset = make_sampler(config, np, sampling_workers=2, previous=sampler)
            self.assertIs(reset, sampler)
            self.assertEqual(sampling_statistics(reset)["consumed"], 1)
            self.assertEqual(sampling_statistics(reset)["pending"], 0)
            self.assertEqual(rng.bit_generator.state, before)
            self.assert_sample(reset, rng, oracle)
            replacement = make_sampler(config, np, sampling_workers=4, previous=reset)
            self.assertTrue(sampling_statistics(reset)["closed"])
            self.assertFalse(any(thread.name.startswith(reset._thread_prefix)
                                 for thread in threading.enumerate()))
            sampler = replacement
            self.assert_sample(sampler, rng, oracle)
            synchronous = make_sampler(config, np, sampling_workers=0, previous=sampler)
            self.assertTrue(sampling_statistics(sampler)["closed"])
            self.assertIs(type(synchronous), UniformSampler)
            self.assert_sample(synchronous, rng, oracle)
        finally:
            sampler.close()


if __name__ == "__main__":
    unittest.main()
