"""Launch two reconstruction or evaluation workers in one Slurm GPU allocation.

This parent process imports only the Python standard library. Both workers keep
the scheduler's complete CUDA namespace, and the runtime selects logical device
``rank`` after spawning. Run with one Slurm task and at least two allocated GPUs;
do not start this launcher once per GPU using ``srun --ntasks=2``.
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


def validate_allocation(environ=None, *, platform_name=None) -> Allocation:
    """Read allocation metadata without modifying the environment or using CUDA."""
    env = os.environ if environ is None else environ
    platform_name = sys.platform if platform_name is None else platform_name
    if not platform_name.startswith("linux"):
        raise RuntimeError("Two-GPU training currently requires Linux HPC Slurm; "
                           "use train.py for local Windows training.")
    if not (env.get("SLURM_JOB_ID") or env.get("SLURM_JOBID")):
        raise RuntimeError("Start this launcher inside a Slurm GPU job allocation.")
    ntasks = env.get("SLURM_STEP_NUM_TASKS", env.get("SLURM_NTASKS", "1"))
    if re.fullmatch(r"[0-9]+", ntasks) is None or int(ntasks) != 1:
        raise RuntimeError("The two-worker launcher needs one Slurm task (--ntasks=1), "
                           f"not SLURM task count {ntasks!r}.")
    visible = env.get("CUDA_VISIBLE_DEVICES")
    tokens = tuple(part.strip() for part in visible.split(",")) if visible is not None else ()
    uuid = r"[0-9a-fA-F]+(?:-[0-9a-fA-F]+)*"
    pattern = rf"(?:[0-9]+|GPU-{uuid}|MIG-{uuid}|MIG-GPU-{uuid}/[0-9]+/[0-9]+)"
    canonical = [str(int(token)) if token.isdecimal() else token.lower() for token in tokens]
    if (len(tokens) < 2 or any(re.fullmatch(pattern, token) is None for token in tokens)
            or len(set(canonical)) != len(canonical)):
        raise RuntimeError("Two distinct scheduler-assigned GPUs are required in "
                           f"CUDA_VISIBLE_DEVICES; got {visible!r}. The value will not be rewritten.")
    cpus = env.get("SLURM_CPUS_PER_TASK")
    if cpus is None or re.fullmatch(r"[0-9]+", cpus) is None or int(cpus) < 2:
        raise RuntimeError("SLURM_CPUS_PER_TASK must be an integer of at least 2 for two workers; "
                           f"got {cpus!r}.")
    return Allocation(visible, tokens, int(cpus) // 2)


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

    def __post_init__(self):
        if self.world_size != 2 or self.rank not in (0, 1):
            raise ValueError("WorkerContext requires exactly ranks 0 and 1")
        if isinstance(self.cpus_per_rank, bool) or not isinstance(self.cpus_per_rank, int) or self.cpus_per_rank < 1:
            raise ValueError("Each worker requires a positive integer CPU budget")

    def broadcast(self, value=None):
        """Broadcast a CPU object from rank 0 before or between GPU collectives."""
        if self.rank == 0:
            self.connection.send(("broadcast", value))
            return value
        message = self.connection.recv()
        if not isinstance(message, tuple) or len(message) != 2 or message[0] != "broadcast":
            raise RuntimeError(f"Expected rank-zero broadcast, got {message!r}")
        return message[1]

    def barrier(self):
        message = ("barrier", self._barrier_index)
        if self.rank == 0:
            received = self.connection.recv()
            if received != message:
                raise RuntimeError(f"Worker barrier mismatch: {received!r} != {message!r}")
            self.connection.send(message)
        else:
            self.connection.send(message)
            received = self.connection.recv()
            if received != message:
                raise RuntimeError(f"Worker barrier mismatch: {received!r} != {message!r}")
        self._barrier_index += 1
        self.heartbeat("barrier")

    def gather(self, value):
        """Gather small CPU results in rank order; rank 1 returns None.

        Evaluation sends metrics and metadata here, never images or gradients.
        The parent supervises both processes while either endpoint waits.
        """
        sequence = self._gather_index
        self._gather_index += 1
        if self.rank == 0:
            message = self.connection.recv()
            if (not isinstance(message, tuple) or len(message) != 3
                    or message[:2] != ("gather", sequence)):
                raise RuntimeError("Worker result gather order differs between ranks")
            result = [value, message[2]]
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
        connection.close()


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
    """Supervise both spawned ranks, terminating the peer on any failure."""
    if not math.isfinite(worker_timeout) or worker_timeout <= 0:
        raise ValueError("worker_timeout must be finite and positive")
    if allocation.world_size != 2:
        raise ValueError("Exactly two workers are supported")
    ctx = mp.get_context("spawn")
    connections = ctx.Pipe(duplex=True)
    events = ctx.Queue()
    processes = []
    last_progress = [time.monotonic()] * 2
    last_stage = ["starting", "starting"]
    try:
        for rank in range(2):
            process = ctx.Process(target=_worker_entry, name=f"reconstruction-rank-{rank}",
                                  args=(rank, allocation, connections[rank], events, list(arguments), worker_main))
            process.start()
            processes.append(process)
        # Only children retain endpoints: peer failures can also surface as EOF.
        for connection in connections:
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
                    print(f"Worker {rank} stopped with exit code {process.exitcode}; stopping its peer.",
                          file=sys.stderr, flush=True)
                    return 1
                if (process.is_alive()
                        and time.monotonic() - last_progress[rank] > worker_timeout):
                    print(f"Worker {rank} made no reported progress for {worker_timeout:g} seconds "
                          f"(last stage: {last_stage[rank]}); stopping both workers. "
                          "Increase --worker-timeout for unusually long initialization or evaluation.",
                          file=sys.stderr, flush=True)
                    return 1
            if all(process.exitcode == 0 for process in processes):
                return 0
    except KeyboardInterrupt:
        print("Interrupted; stopping both reconstruction workers.", file=sys.stderr, flush=True)
        return 130
    finally:
        _stop_workers(processes)
        for connection in connections:
            connection.close()
        events.close()
        events.join_thread()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False,
        epilog="Other arguments go to train.py, or evaluate.py with --evaluate.")
    parser.add_argument("--evaluate", action="store_true",
                        help="Render and evaluate a saved checkpoint on both GPUs without training or NCCL.")
    parser.add_argument("--workers", type=int, choices=(2,), default=2)
    parser.add_argument("--worker-timeout", type=float, default=600.,
                        help="Maximum seconds without a worker heartbeat (default: 600).")
    args, training_arguments = parser.parse_known_args(argv)
    if not math.isfinite(args.worker_timeout) or args.worker_timeout <= 0:
        parser.error("--worker-timeout must be finite and positive")
    try:
        allocation = validate_allocation()
        mode = "evaluation" if args.evaluate else "training"
        print(f"Two-GPU {mode}: logical devices 0 and 1, "
              f"CUDA_VISIBLE_DEVICES={allocation.cuda_visible_devices!r}, "
              f"Dr.Jit CPU threads per worker={allocation.cpus_per_rank}", flush=True)
        return run_workers(training_arguments, allocation, worker_timeout=args.worker_timeout,
                           worker_main=_run_evaluation if args.evaluate else _run_training)
    except (RuntimeError, ValueError) as error:
        print(str(error), file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
