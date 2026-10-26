"""Bounded CPU prefetch of the existing uniform RGB sampling sequence.

Only independent NumPy generators run in workers. The caller's generator is
advanced when a sample is consumed, so lookahead cannot affect checkpoints or
epoch shuffles. No device arrays or training state cross a worker boundary.
"""

from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor

from reconstruction_common import UniformSampler


def _advance_state(np, state, draws):
    """Advance PCG64's uint64 stream without changing its unused uint32 cache."""
    bit_generator = np.random.PCG64(0)
    bit_generator.state = state
    bit_generator.advance(draws)
    advanced = bit_generator.state
    # Generator.random(dtype=float64) consumes uint64 values and leaves a
    # previously cached uint32 alone. PCG64.advance() instead resets this cache.
    # Preserve it, including the stale uinteger value when has_uint32 is zero:
    # exact checkpoint state matters as well as the generated offsets.
    advanced["has_uint32"] = state["has_uint32"]
    advanced["uinteger"] = state["uinteger"]
    return advanced


def _generate(np, state, count):
    """Run the original sampling operations with a worker-owned generator."""
    bit_generator = np.random.PCG64(0)
    bit_generator.state = state
    rng = np.random.Generator(bit_generator)
    offsets = rng.random((count, 2)).astype(np.float32)
    np.minimum(offsets, np.nextafter(np.float32(1), np.float32(0)), out=offsets)
    return offsets, np.ones(count, dtype=np.float32), rng.bit_generator.state


class PrefetchedUniformSampler:
    """UniformSampler's interface, with optional bounded PCG64 lookahead.

    ``workers=0`` uses the original synchronous sampler. Other generator types
    and custom RNG objects also use that path. ``lookahead`` counts unconsumed
    samples and defaults to twice the worker count. Only a single caller may
    use this sampler or its main generator at a time. Workers never reference
    that generator, image contents, or this sampler.

    Use ``close()`` or a context manager to join workers when replacing the
    sampler or leaving training. ``clear()`` and ``load_state_dict({})`` discard
    pending work without advancing the caller's generator. Sampling after
    close is an error; repeated close calls are harmless.
    """

    def __init__(self, np, workers=2, lookahead=None):
        if isinstance(workers, bool) or not isinstance(workers, int) or workers < 0:
            raise ValueError("Sampling workers must be a nonnegative integer")
        if lookahead is None:
            lookahead = max(1, workers * 2)
        if isinstance(lookahead, bool) or not isinstance(lookahead, int) or lookahead < 1:
            raise ValueError("Sampling lookahead must be a positive integer")
        self.np, self.workers, self.lookahead = np, workers, lookahead
        self.enabled = False
        self._sampler = UniformSampler(np)
        self._executor = None
        self._pending = deque()
        self._rng = self._shape = self._tail_state = None
        self._closed = False
        self._thread_prefix = f"reconstruction-sampling-{id(self):x}"
        self._submitted = self._consumed = self._reused = 0
        self._synchronous = self._discarded = self._invalidations = 0

    def _supported(self, rng):
        # Subclasses can override random(), so use exact types. PCG64DXSM,
        # MT19937 and wrappers deliberately retain their original behavior.
        np = self.np
        return (type(rng) is np.random.Generator
                and type(rng.bit_generator) is np.random.PCG64)

    def _discard(self):
        self._discarded += len(self._pending)
        while self._pending:
            self._pending.popleft()[2].cancel()
        self._rng = self._shape = self._tail_state = None

    def clear(self):
        """Cancel unconsumed work; preserve counters and the main RNG state."""
        self._discard()

    def _fill(self, count):
        if self._executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=self.workers, thread_name_prefix=self._thread_prefix)
        while len(self._pending) < self.lookahead:
            before = self._tail_state
            after = _advance_state(self.np, before, 2 * count)
            future = self._executor.submit(_generate, self.np, before, count)
            self._pending.append((before, after, future))
            self._tail_state = after
            self._submitted += 1

    def sample(self, view_id, reference_rgb, rng):
        if self._closed:
            raise RuntimeError("Cannot sample after closing the uniform prefetcher")
        shape = tuple(reference_rgb.shape)
        if len(shape) != 3 or shape[-1] != 3:
            raise ValueError("UniformSampler requires an RGB reference of shape (height, width, 3)")
        if not self.workers or not self._supported(rng):
            self._discard()
            result = self._sampler.sample(view_id, reference_rgb, rng)
            self._synchronous += 1
            return result

        count = shape[0] * shape[1]
        state = rng.bit_generator.state
        reusable = (self._rng is rng and self._shape == shape and self._pending
                    and self._pending[0][0] == state)
        if not reusable:
            if self._pending:
                self._invalidations += 1
            self._discard()
            self._rng, self._shape, self._tail_state = rng, shape, state
            self._fill(count)
        before, expected_after, future = self._pending[0]
        try:
            offsets, inverse_pdf, after = future.result()
            if after != expected_after:
                raise RuntimeError("PCG64 sampling did not match its predicted stream position")
            if rng.bit_generator.state != before:
                raise RuntimeError("The sampling RNG changed while waiting for a prefetched sample")
        except BaseException:
            self._discard()
            raise
        # Commit only this consumed sample, never the speculative tail. A saved
        # state thus equals the synchronous sampler after every successful call.
        rng.bit_generator.state = after
        self._pending.popleft()
        self._consumed += 1
        self._reused += int(bool(reusable))
        self._fill(count)
        return offsets, inverse_pdf, None

    def observe(self, token, raw_alpha):
        self._sampler.observe(token, raw_alpha)

    def state_dict(self):
        return self._sampler.state_dict()

    def load_state_dict(self, state):
        self._sampler.load_state_dict(state)
        self._discard()

    def statistics(self):
        return {"workers": self.workers, "lookahead": self.lookahead,
                "submitted": self._submitted, "consumed": self._consumed,
                "reused": self._reused, "synchronous": self._synchronous,
                "discarded": self._discarded, "invalidations": self._invalidations,
                "pending": len(self._pending), "closed": self._closed}

    def close(self):
        if getattr(self, "_closed", True):
            return
        self._closed = True
        self._discard()
        executor, self._executor = self._executor, None
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)

    def __enter__(self):
        if self._closed:
            raise RuntimeError("Cannot reopen a closed uniform prefetcher")
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        # The training driver should close explicitly. This also releases
        # threads if a short-lived diagnostic caller forgets the context manager.
        try:
            self.close()
        except Exception:
            pass
