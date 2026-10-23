"""CUDA event ownership, bounded backpressure, and external stream writes.

Default tests use controllable events to prove lifetime/error behavior. ``--cuda``
also submits actual CUDA driver copies on Dr.Jit's stream and consumes their
destinations with Dr.Jit, without a per-copy host synchronization or NCCL.
"""

from __future__ import annotations

import argparse
import ctypes
import gc
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import weakref

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reconstruction_cuda_owners import CudaOwnerQueue


class Owner:
    pass


class EventRuntime:
    def __init__(self):
        self.events = []
        self.syncs = 0
        self.fail_record = False
        self.fail_query = False
        self.fail_wait = False
        runtime = self

        class Event:
            def __init__(self, enable_timing):
                assert enable_timing is False
                self.complete, self.recorded, self.waits = False, False, 0
                runtime.events.append(self)

            def record(self):
                if runtime.fail_record:
                    raise RuntimeError("record failed")
                self.recorded = True

            def query(self):
                assert self.recorded
                if runtime.fail_query:
                    raise RuntimeError("query failed")
                return self.complete

            def wait(self):
                assert self.recorded
                if runtime.fail_wait:
                    raise RuntimeError("wait failed")
                self.waits += 1
                self.complete = True

        self.cuda = SimpleNamespace(Event=Event)

    def sync_thread(self):
        self.syncs += 1
        for event in self.events:
            event.complete = True


def expect_error(function, match):
    try:
        function()
    except (ValueError, RuntimeError) as error:
        assert match in str(error), str(error)
    else:
        raise AssertionError(f"Expected {match}")


def retained(queue, size=10):
    owner = Owner()
    reference = weakref.ref(owner)
    queue.retain([owner], size)
    return reference


def test_lifetime():
    runtime = EventRuntime()
    queue = CudaOwnerQueue(runtime)
    first, second = retained(queue), retained(queue)
    gc.collect()
    assert first() is not None and second() is not None
    assert runtime.syncs == 0 and not any(event.waits for event in runtime.events)
    queue.reap()
    assert queue.statistics()["pending_operations"] == 2
    runtime.events[0].complete = True
    queue.reap()
    gc.collect()
    assert first() is None and second() is not None
    queue.finish()
    gc.collect()
    assert second() is None
    assert runtime.syncs == 0 and runtime.events[1].waits == 1
    assert queue.statistics()["pending_bytes"] == 0


def test_limits():
    runtime = EventRuntime()
    queue = CudaOwnerQueue(runtime, max_pending_bytes=15, max_pending_operations=4)
    first, second = retained(queue, 10), retained(queue, 10)
    assert first() is None and second() is not None
    assert runtime.events[0].waits == 1 and runtime.events[1].waits == 0
    assert queue.statistics()["pending_bytes"] == 10
    queue.finish()
    # A single operation can be larger than the limit without forcing a wait.
    huge = retained(queue, 50)
    assert huge() is not None and runtime.events[-1].waits == 0
    retained(queue, 50)
    assert huge() is None and queue.statistics()["pending_operations"] == 1
    queue.finish()
    runtime = EventRuntime()
    queue = CudaOwnerQueue(runtime, max_pending_operations=1)
    first, second = retained(queue), retained(queue)
    assert first() is None and second() is not None
    assert runtime.events[0].waits == 1
    queue.finish()


def test_failures():
    for failure in ("record", "query", "wait"):
        runtime = EventRuntime()
        queue = CudaOwnerQueue(runtime)
        owner = Owner()
        reference = weakref.ref(owner)
        if failure == "record":
            runtime.fail_record = True
            expect_error(lambda: queue.retain([owner], 10), "record failed")
        else:
            queue.retain([owner], 10)
            setattr(runtime, f"fail_{failure}", True)
            expect_error(queue.reap if failure == "query" else queue.finish, f"{failure} failed")
        del owner
        gc.collect()
        assert reference() is not None, "An event error must not free in-flight buffers"
        setattr(runtime, f"fail_{failure}", False)
        queue.finish()
        gc.collect()
        assert reference() is None
        assert runtime.syncs == int(failure == "record")
    runtime = EventRuntime()
    expect_error(lambda: CudaOwnerQueue(runtime, max_pending_bytes=0), "positive integer")
    expect_error(lambda: CudaOwnerQueue(runtime, max_pending_operations=True), "positive integer")
    expect_error(lambda: CudaOwnerQueue(SimpleNamespace()), "CUDA events")
    queue = CudaOwnerQueue(runtime)
    expect_error(lambda: queue.retain([], -1), "nonnegative integer")


def test_cuda():
    from preview import RuntimeConfig
    from reconstruction_common import initialize_runtime

    with tempfile.TemporaryDirectory(prefix="mitsuba-cuda-owners-") as temporary:
        mi, dr, np, handles = initialize_runtime(RuntimeConfig(), Path(temporary), {})
        library = ctypes.WinDLL("nvcuda.dll") if sys.platform == "win32" else ctypes.CDLL("libcuda.so.1")
        copy = library.cuMemcpyDtoDAsync_v2
        copy.argtypes = [ctypes.c_uint64, ctypes.c_uint64, ctypes.c_size_t, ctypes.c_void_p]
        copy.restype = ctypes.c_int
        context = mi._reconstruction_cuda_context()
        queue = CudaOwnerQueue(dr, max_pending_bytes=64 * 1024 ** 2, max_pending_operations=16)
        count = 131072
        consumers = []
        # Fresh external destinations must not be literal-zero arrays; their
        # values become visible to ordinary JIT kernels through stream order.
        for iteration in range(8):
            queue.reap()
            source = mi.Float(dr.arange(mi.UInt32, count)) + dr.opaque(mi.Float, iteration)
            destination = dr.empty(mi.Float, count)
            dr.eval(source, destination)
            source_info = mi._reconstruction_float_buffer_info(source)
            destination_info = mi._reconstruction_float_buffer_info(destination)
            mi._reconstruction_cuda_push_context(int(context["context"]))
            try:
                code = copy(int(destination_info["pointer"]), int(source_info["pointer"]),
                            count * 4, int(context["stream"]))
                assert code == 0, f"CUDA driver copy failed: {code}"
            finally:
                mi._reconstruction_cuda_pop_context()
            queue.retain((source, destination), count * 8)
            consumer = destination * 2 + 1
            dr.eval(consumer)
            consumers.append(consumer)
            del source, destination
            # Exercise allocator reuse while the queue, rather than local
            # variables, owns pointers captured by an external CUDA operation.
            scratch = dr.full(mi.Float, -987., count)
            dr.eval(scratch)
            del scratch
        before_drain = queue.statistics()
        assert before_drain["event_waits"] == 0 and before_drain["fallback_syncs"] == 0
        queue.finish()
        for iteration, consumer in enumerate(consumers):
            np.testing.assert_array_equal(np.asarray(consumer),
                (np.arange(count, dtype=np.float32) + iteration) * 2 + 1)
        assert queue.statistics()["pending_operations"] == 0
        print("CUDA external writes, consumer stream ordering and allocator churn: passed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args()
    test_lifetime()
    test_limits()
    test_failures()
    print("CUDA owner lifetime, nonblocking reaping, backpressure and error retention: passed")
    if args.cuda:
        test_cuda()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
