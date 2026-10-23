"""Retain external CUDA operation buffers until their stream event completes.

NCCL uses Dr.Jit's own stream, so later Dr.Jit kernels are ordered after its
writes without a host wait. Dr.Jit cannot see NCCL's pointer ownership, however:
every source and destination must stay alive until communication completes.
This queue provides that ownership and bounded backpressure, without assuming
that the allocator delays reuse of an otherwise unreferenced device buffer.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass


@dataclass
class _Pending:
    owners: tuple
    nbytes: int
    event: object = None


class CudaOwnerQueue:
    """Own buffers after an external operation has been enqueued on one stream.

    Call ``retain`` immediately after a successful NCCL group submission, and
    ``reap`` before subsequent work. ``finish`` waits and releases outstanding
    owners before normal communicator destruction. After a transport failure,
    abort the communicator first, then finish the queue; never clear pending
    owners while the transport may still access their allocations.

    The limits bound accumulated ownership, not one operation's allocation. An
    individual operation larger than the byte limit is allowed to remain in
    flight alone. Event creation/record/query failures retain all affected
    owners, so the caller can abort the communicator safely before cleanup.
    """

    def __init__(self, dr, *, max_pending_bytes=512 * 1024 ** 2,
                 max_pending_operations=4):
        if type(max_pending_bytes) is not int or max_pending_bytes < 1:
            raise ValueError("Pending CUDA owner byte limit must be a positive integer")
        if type(max_pending_operations) is not int or max_pending_operations < 1:
            raise ValueError("Pending CUDA owner operation limit must be a positive integer")
        event_type = getattr(getattr(dr, "cuda", None), "Event", None)
        if event_type is None:
            raise RuntimeError("Asynchronous NCCL ownership requires Dr.Jit CUDA events")
        self._dr, self._event_type = dr, event_type
        self._max_bytes, self._max_operations = max_pending_bytes, max_pending_operations
        self._pending = deque()
        self._bytes = 0
        self._stats = dict(retained=0, completed=0, event_queries=0, event_waits=0,
                           fallback_syncs=0, peak_pending_bytes=0, peak_pending_operations=0)

    def retain(self, owners, nbytes):
        """Record completion after enqueue; retain all array wrappers meanwhile.

        ``nbytes`` should include both source and destination allocations. It is
        an ownership/backpressure estimate, not a measurement of wire traffic.
        """
        if type(nbytes) is not int or nbytes < 0:
            raise ValueError("Pending CUDA owner bytes must be a nonnegative integer")
        pending = _Pending(tuple(owners), nbytes)
        # Store owners before creating/recording the event: either CUDA API may
        # raise even though the external operation has already been submitted.
        self._pending.append(pending)
        self._bytes += nbytes
        self._stats["retained"] += 1
        self._stats["peak_pending_bytes"] = max(self._stats["peak_pending_bytes"], self._bytes)
        self._stats["peak_pending_operations"] = max(
            self._stats["peak_pending_operations"], len(self._pending))
        event = self._event_type(enable_timing=False)
        event.record()
        # An unrecorded event can report completion. Publish it only after
        # record succeeds; otherwise cleanup must synchronize the whole stream.
        pending.event = event
        self.reap()
        while len(self._pending) > 1 and (
                len(self._pending) > self._max_operations or self._bytes > self._max_bytes):
            self._wait_oldest()
            self.reap()

    def _release_oldest(self):
        pending = self._pending.popleft()
        self._bytes -= pending.nbytes
        self._stats["completed"] += 1

    def reap(self):
        """Nonblocking FIFO event queries; release only proven-complete owners."""
        while self._pending:
            pending = self._pending[0]
            if pending.event is None:
                return
            self._stats["event_queries"] += 1
            if not pending.event.query():
                return
            self._release_oldest()

    def _wait_oldest(self):
        pending = self._pending[0]
        if pending.event is None:
            self._stats["fallback_syncs"] += 1
            self._dr.sync_thread()
        else:
            self._stats["event_waits"] += 1
            pending.event.wait()
        self._release_oldest()

    def finish(self):
        """Wait for outstanding operations before releasing their buffer owners."""
        self.reap()
        while self._pending:
            self._wait_oldest()

    def statistics(self):
        return {**self._stats, "pending_bytes": self._bytes,
                "pending_operations": len(self._pending),
                "max_pending_bytes": self._max_bytes,
                "max_pending_operations": self._max_operations}
