"""Launch reconstruction or evaluation workers in one Slurm GPU allocation.

This parent process imports only the Python standard library. Both workers keep
the scheduler's complete CUDA namespace, and the runtime selects logical device
``rank`` after spawning. Run with one Slurm task; workers default to all allocated
GPUs. Training assigns one whole view per worker and updates once per view batch.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import math
import multiprocessing as mp
import os
from pathlib import Path
import queue
import re
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parent))


@dataclass(frozen=True)
class Allocation:
    cuda_visible_devices: str
    device_tokens: tuple[str, ...]
    cpus_per_rank: int
    world_size: int = 2


def validate_allocation(environ=None, *, platform_name=None, workers=None) -> Allocation:
    """Read allocation metadata without modifying the environment or using CUDA."""
    env = os.environ if environ is None else environ
    platform_name = sys.platform if platform_name is None else platform_name
    if not platform_name.startswith("linux"):
        raise RuntimeError("Multi-GPU training currently requires Linux HPC Slurm; "
                           "use train.py for local Windows training.")
    if not (env.get("SLURM_JOB_ID") or env.get("SLURM_JOBID")):
        raise RuntimeError("Start this launcher inside a Slurm GPU job allocation.")
    ntasks = env.get("SLURM_STEP_NUM_TASKS", env.get("SLURM_NTASKS", "1"))
    if re.fullmatch(r"[0-9]+", ntasks) is None or int(ntasks) != 1:
        raise RuntimeError("The worker launcher needs one Slurm task (--ntasks=1), "
                           f"not SLURM task count {ntasks!r}.")
    visible = env.get("CUDA_VISIBLE_DEVICES")
    tokens = tuple(part.strip() for part in visible.split(",")) if visible is not None else ()
    uuid = r"[0-9a-fA-F]+(?:-[0-9a-fA-F]+)*"
    pattern = rf"(?:[0-9]+|GPU-{uuid}|MIG-{uuid}|MIG-GPU-{uuid}/[0-9]+/[0-9]+)"
    canonical = [str(int(token)) if token.isdecimal() else token.lower() for token in tokens]
    if (not tokens or any(re.fullmatch(pattern, token) is None for token in tokens)
            or len(set(canonical)) != len(canonical)):
        raise RuntimeError("Distinct scheduler-assigned GPUs are required in "
                           f"CUDA_VISIBLE_DEVICES; got {visible!r}. The value will not be rewritten.")
    workers = len(tokens) if workers is None else workers
    if type(workers) is not int or not 1 <= workers <= len(tokens):
        raise ValueError(f"--workers must be between 1 and {len(tokens)} allocated GPUs; got {workers!r}")
    cpus = env.get("SLURM_CPUS_PER_TASK")
    if cpus is None or re.fullmatch(r"[0-9]+", cpus) is None or int(cpus) < workers:
        raise RuntimeError(f"SLURM_CPUS_PER_TASK must be an integer of at least {workers} for {workers} workers; "
                           f"got {cpus!r}.")
    return Allocation(visible, tokens, int(cpus) // workers, workers)


@dataclass
class WorkerContext:
    rank: int
    world_size: int
    connection: object
    cpus_per_rank: int
    event_queue: object = None
    _last_heartbeat: float = field(default=0., init=False, repr=False)
    _barrier_index: int = field(default=0, init=False, repr=False)
    _gather_index: int = field(default=0, init=False, repr=False)
    _broadcast_index: int = field(default=0, init=False, repr=False)

    def __post_init__(self):
        if (type(self.world_size) is not int or self.world_size < 1
                or type(self.rank) is not int or not 0 <= self.rank < self.world_size):
            raise ValueError("Worker rank must fit within a positive world size")
        if isinstance(self.cpus_per_rank, bool) or not isinstance(self.cpus_per_rank, int) or self.cpus_per_rank < 1:
            raise ValueError("Each worker requires a positive integer CPU budget")

    def _peers(self):
        if self.rank != 0:
            raise ValueError("Only rank zero owns peer connections")
        if self.world_size == 1:
            return ()
        peers = self.connection if isinstance(self.connection, (list, tuple)) else (self.connection,)
        if len(peers) != self.world_size - 1:
            raise ValueError("Rank zero requires one connection for every other rank")
        return peers

    def broadcast(self, value=None):
        """Broadcast a CPU object from rank 0 before or between GPU collectives."""
        sequence = self._broadcast_index
        self._broadcast_index += 1
        if self.rank == 0:
            for connection in self._peers():
                connection.send(("broadcast", sequence, value))
            return value
        message = self.connection.recv()
        if (not isinstance(message, tuple) or len(message) != 3
                or message[:2] != ("broadcast", sequence)):
            raise RuntimeError(f"Expected rank-zero broadcast, got {message!r}")
        return message[2]

    def barrier(self):
        message = ("barrier", self._barrier_index)
        if self.rank == 0:
            for connection in self._peers():
                received = connection.recv()
                if received != message:
                    raise RuntimeError(f"Worker barrier mismatch: {received!r} != {message!r}")
            for connection in self._peers():
                connection.send(message)
        else:
            self.connection.send(message)
            received = self.connection.recv()
            if received != message:
                raise RuntimeError(f"Worker barrier mismatch: {received!r} != {message!r}")
        self._barrier_index += 1
        self.heartbeat("barrier")

    def gather(self, value):
        """Gather small CPU results in rank order; non-primary ranks return None.

        Evaluation sends metrics and metadata here, never images or gradients.
        The parent supervises every process while a control endpoint waits.
        """
        sequence = self._gather_index
        self._gather_index += 1
        if self.rank == 0:
            result = [value]
            for connection in self._peers():
                message = connection.recv()
                if (not isinstance(message, tuple) or len(message) != 3
                        or message[:2] != ("gather", sequence)):
                    raise RuntimeError("Worker result gather order differs between ranks")
                result.append(message[2])
        else:
            self.connection.send(("gather", sequence, value))
            result = None
        self.heartbeat("gather")
        return result

    def heartbeat(self, stage, *, force=False):
        """Notify the parent of progress, at most once per second by default."""
        now = time.monotonic()
        if self.event_queue is not None and (force or now - self._last_heartbeat >= 1.):
            self.event_queue.put((self.rank, "heartbeat", str(stage)))
            self._last_heartbeat = now


def _run_training(context, arguments):
    # This import is deliberately inside the spawned process. CUDA state must
    # never be inherited from the launcher or selected by rewriting Slurm vars.
    import train
    sys.argv = [str(Path(train.__file__).resolve()), *arguments]
    return train.main(distributed=context)


def _run_evaluation(context, arguments):
    import evaluate
    sys.argv = [str(Path(evaluate.__file__).resolve()), *arguments]
    return evaluate.main(distributed=context)


def _worker_entry(rank, allocation, connection, events, arguments, worker_main):
    context = WorkerContext(rank, allocation.world_size, connection, allocation.cpus_per_rank, events)
    try:
        context.heartbeat("starting", force=True)
        code = worker_main(context, arguments)
        code = 0 if code is None else int(code)
        events.put((rank, "finished", code))
        if code:
            print(f"Reconstruction worker {rank} exited with status {code}.", file=sys.stderr, flush=True)
        raise SystemExit(code)
    except SystemExit:
        raise
    except BaseException:
        detail = traceback.format_exc()
        events.put((rank, "error", detail))
        print(f"Reconstruction worker {rank} failed:\n{detail}", file=sys.stderr, flush=True)
        raise SystemExit(1)
    finally:
        for endpoint in (connection if isinstance(connection, (tuple, list)) else (connection,)):
            if endpoint is not None:
                endpoint.close()


def _stop_workers(processes):
    for process in processes:
        if process.is_alive():
            process.terminate()
    deadline = time.monotonic() + 5.
    for process in processes:
        process.join(max(0., deadline - time.monotonic()))
    for process in processes:
        if process.is_alive():
            process.kill()
            process.join(5.)


def run_workers(arguments, allocation, *, worker_timeout=600., worker_main=_run_training):
    """Supervise all spawned ranks, terminating every worker on any failure."""
    if not math.isfinite(worker_timeout) or worker_timeout <= 0:
        raise ValueError("worker_timeout must be finite and positive")
    if type(allocation.world_size) is not int or allocation.world_size < 1:
        raise ValueError("At least one worker is required")
    ctx = mp.get_context("spawn")
    pipes = [ctx.Pipe(duplex=True) for _ in range(allocation.world_size - 1)]
    connections = [[pair[0] for pair in pipes], *[pair[1] for pair in pipes]]
    if allocation.world_size == 2:
        connections[0] = connections[0][0]  # Preserve the existing two-rank pipe interface.
    endpoints = [connection for pair in pipes for connection in pair]
    events = ctx.Queue()
    processes = []
    last_progress = [time.monotonic()] * allocation.world_size
    last_stage = ["starting"] * allocation.world_size
    try:
        for rank in range(allocation.world_size):
            process = ctx.Process(target=_worker_entry, name=f"reconstruction-rank-{rank}",
                                  args=(rank, allocation, connections[rank], events, list(arguments), worker_main))
            process.start()
            processes.append(process)
        # Only children retain endpoints: peer failures can also surface as EOF.
        for connection in endpoints:
            connection.close()
        while True:
            try:
                rank, kind, detail = events.get(timeout=.1)
            except queue.Empty:
                pass
            else:
                last_progress[rank] = time.monotonic()
                if kind == "heartbeat":
                    last_stage[rank] = detail
                elif kind == "finished":
                    last_stage[rank] = "worker cleanup"
                    if detail != 0:
                        return 1
                elif kind == "error":
                    return 1
            for rank, process in enumerate(processes):
                if process.exitcode is not None and process.exitcode != 0:
                    print(f"Worker {rank} stopped with exit code {process.exitcode}; stopping all workers.",
                          file=sys.stderr, flush=True)
                    return 1
                if (process.is_alive()
                        and time.monotonic() - last_progress[rank] > worker_timeout):
                    print(f"Worker {rank} made no reported progress for {worker_timeout:g} seconds "
                          f"(last stage: {last_stage[rank]}); stopping all workers. "
                          "Increase --worker-timeout for unusually long initialization or evaluation.",
                          file=sys.stderr, flush=True)
                    return 1
            if all(process.exitcode == 0 for process in processes):
                return 0
    except KeyboardInterrupt:
        print("Interrupted; stopping all reconstruction workers.", file=sys.stderr, flush=True)
        return 130
    finally:
        _stop_workers(processes)
        for connection in endpoints:
            connection.close()
        events.close()
        events.join_thread()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False,
        epilog="Other arguments go to train.py, or evaluate.py with --evaluate.")
    parser.add_argument("--evaluate", action="store_true",
                        help="Render and evaluate a saved checkpoint on allocated GPUs without training or NCCL.")
    parser.add_argument("--workers", type=int, default=None,
                        help="Worker count; default: all scheduler-assigned CUDA_VISIBLE_DEVICES.")
    parser.add_argument("--worker-timeout", type=float, default=600.,
                        help="Maximum seconds without a worker heartbeat (default: 600).")
    args, training_arguments = parser.parse_known_args(argv)
    if not math.isfinite(args.worker_timeout) or args.worker_timeout <= 0:
        parser.error("--worker-timeout must be finite and positive")
    try:
        allocation = validate_allocation(workers=args.workers)
        mode = "evaluation" if args.evaluate else "training"
        print(f"{allocation.world_size}-GPU {mode}: logical devices {list(range(allocation.world_size))}, "
              f"CUDA_VISIBLE_DEVICES={allocation.cuda_visible_devices!r}, "
              f"Dr.Jit CPU threads per worker={allocation.cpus_per_rank}", flush=True)
        return run_workers(training_arguments, allocation, worker_timeout=args.worker_timeout,
                           worker_main=_run_evaluation if args.evaluate else _run_training)
    except (RuntimeError, ValueError) as error:
        print(str(error), file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
