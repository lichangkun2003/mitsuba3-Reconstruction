"""CPU-only allocation, communication and process-lifecycle checks for N GPUs.

The workers below never import Mitsuba, Dr.Jit, or NCCL. Real CUDA numerical
equivalence must be checked separately on a node with two allocated devices.
"""

from __future__ import annotations

from contextlib import redirect_stderr
import io
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import tempfile
import time
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_multigpu import Allocation, WorkerContext, main, run_workers, validate_allocation


def _fake_success(context, arguments):
    output = Path(arguments[0])
    inherited = {name: value for name, value in os.environ.items()
                 if name.startswith("SLURM_") or name == "CUDA_VISIBLE_DEVICES"}
    value = context.broadcast({"sent_by": 0, "nested": [2, None, {"x": 4}]} if context.rank == 0 else None)
    assert value == {"sent_by": 0, "nested": [2, None, {"x": 4}]}
    context.barrier()
    context.barrier()
    for index in range(2):
        result = context.gather({"rank": context.rank, "step": index})
        if context.rank == 0:
            assert result == [{"rank": rank, "step": index} for rank in range(context.world_size)]
        else:
            assert result is None
    context.heartbeat("completed fake GPU work", force=True)
    (output / f"rank-{context.rank}.json").write_text(json.dumps({
        "rank": context.rank, "world_size": context.world_size,
        "cpus": context.cpus_per_rank, "environment": inherited,
        "arguments": arguments[1:],
        "cuda_imported": any(name in sys.modules for name in ("drjit", "mitsuba", "cupy")),
    }), encoding="utf-8")
    return 0


def _fake_failure(context, arguments):
    if context.rank == context.world_size - 1:
        return 7
    time.sleep(30)
    return 0


def _fake_timeout(context, arguments):
    context.heartbeat("simulated stall", force=True)
    time.sleep(30)
    return 0


def _fake_heartbeats(context, arguments):
    for _ in range(6):
        context.heartbeat("fake long calculation", force=True)
        time.sleep(.2)
    return 0


def _expect_failure(action):
    try:
        action()
    except (RuntimeError, ValueError):
        return
    raise AssertionError("Expected allocation/context validation to fail")


def test_allocation():
    environment = {"SLURM_JOB_ID": "12345", "SLURM_CPUS_PER_TASK": "9",
                   "SLURM_NTASKS": "1", "CUDA_VISIBLE_DEVICES": " 2, 3 "}
    before = dict(environment)
    allocation = validate_allocation(environment, platform_name="linux")
    assert environment == before
    assert allocation == Allocation(" 2, 3 ", ("2", "3"), 4)
    uuid = dict(environment, CUDA_VISIBLE_DEVICES="GPU-abcd-1234,MIG-GPU-abcd-5678/1/0")
    assert validate_allocation(uuid, platform_name="linux").device_tokens == (
        "GPU-abcd-1234", "MIG-GPU-abcd-5678/1/0")
    assert validate_allocation(dict(environment, CUDA_VISIBLE_DEVICES="3"), platform_name="linux").world_size == 1
    for workers in (1, 2):
        chosen = validate_allocation(environment, platform_name="linux", workers=workers)
        assert chosen.world_size == workers and chosen.cpus_per_rank == 9 // workers
        assert chosen.cuda_visible_devices == environment["CUDA_VISIBLE_DEVICES"]
    for workers in (0, -1, 3, True, 1.5, "2"):
        _expect_failure(lambda: validate_allocation(environment, platform_name="linux", workers=workers))
    for invalid in (None, "", "0,", "-1,0", "NoDevFiles", "0,0", "01,1", "GPU-a,GPU-A", "0,wat"):
        candidate = dict(environment)
        if invalid is None:
            candidate.pop("CUDA_VISIBLE_DEVICES")
        else:
            candidate["CUDA_VISIBLE_DEVICES"] = invalid
        _expect_failure(lambda: validate_allocation(candidate, platform_name="linux"))
    for cpus in (None, "", "0", "1", "2.0", "-2", " 8", "eight"):
        candidate = dict(environment)
        if cpus is None:
            candidate.pop("SLURM_CPUS_PER_TASK")
        else:
            candidate["SLURM_CPUS_PER_TASK"] = cpus
        _expect_failure(lambda: validate_allocation(candidate, platform_name="linux"))
    for key, value in (("SLURM_JOB_ID", ""), ("SLURM_NTASKS", "2"), ("SLURM_STEP_NUM_TASKS", "2")):
        _expect_failure(lambda: validate_allocation(dict(environment, **{key: value}), platform_name="linux"))
    _expect_failure(lambda: validate_allocation(environment, platform_name="win32"))
    _expect_failure(lambda: WorkerContext(2, 2, None, 4))
    _expect_failure(lambda: WorkerContext(0, 0, None, 4))
    _expect_failure(lambda: WorkerContext(True, 2, None, 4))
    _expect_failure(lambda: WorkerContext(0, 2, None, 0))
    _expect_failure(lambda: run_workers([], allocation, worker_timeout=float("inf")))
    return allocation


def test_lifecycle(world_size):
    # The lower-level lifecycle is portable so it can be verified locally;
    # only the actual launcher rejects Windows before spawning CUDA workers.
    visible = ",".join(str(index + 2) for index in range(world_size))
    environment = {"SLURM_JOB_ID": "12345", "SLURM_CPUS_PER_TASK": str(world_size * 4 + 1), "SLURM_NTASKS": "1",
                   "SLURM_STEP_ID": "0", "SLURM_STEP_GPUS": visible, "CUDA_VISIBLE_DEVICES": visible}
    allocation = validate_allocation(environment, platform_name="linux")
    with patch.dict(os.environ, environment), tempfile.TemporaryDirectory(prefix="mitsuba-ngpu-cpu-") as output:
        before = dict(os.environ)
        inherited = {name: value for name, value in before.items()
                     if name.startswith("SLURM_") or name == "CUDA_VISIBLE_DEVICES"}
        assert run_workers([output, "--epochs", "3", "--no-profile-stages"], allocation,
                           worker_timeout=10., worker_main=_fake_success) == 0
        assert dict(os.environ) == before
        for rank in range(world_size):
            record = json.loads((Path(output) / f"rank-{rank}.json").read_text(encoding="utf-8"))
            assert record == {"rank": rank, "world_size": world_size, "cpus": (world_size * 4 + 1) // world_size,
                              "environment": inherited,
                              "arguments": ["--epochs", "3", "--no-profile-stages"], "cuda_imported": False}
        started = time.monotonic()
        with redirect_stderr(io.StringIO()):
            assert run_workers([], allocation, worker_timeout=10., worker_main=_fake_failure) != 0
            assert run_workers([], allocation, worker_timeout=.6, worker_main=_fake_timeout) != 0
        assert time.monotonic() - started < 12., "Failed worker's peers were not terminated promptly"
        assert run_workers([], allocation, worker_timeout=.6, worker_main=_fake_heartbeats) == 0
        assert not mp.active_children(), "Worker processes leaked after completion/failure"
        assert dict(os.environ) == before
        with patch("train_multigpu.sys.platform", "win32"), redirect_stderr(io.StringIO()):
            assert main(["--workers", str(world_size), "--epochs", "3"]) == 1
        with patch("train_multigpu.validate_allocation", return_value=allocation), \
                patch("train_multigpu.run_workers", return_value=0) as launched:
            assert main(["--evaluate", "--checkpoint", "model.npz", "--spp", "64"]) == 0
            assert launched.call_args.args[0] == ["--checkpoint", "model.npz", "--spp", "64"]
            assert launched.call_args.kwargs["worker_main"].__name__ == "_run_evaluation"


def main_test():
    test_allocation()
    for world_size in (1, 2, 3, 4):
        test_lifecycle(world_size)
    assert all(name not in sys.modules for name in ("drjit", "mitsuba", "cupy"))
    print(json.dumps({"status": "passed", "gpu_required": False,
                      "checks": ["allocation_validation", "scheduler_environment_preserved",
                                 "spawn_without_cuda_import", "broadcast_gather_and_barriers",
                                 "standalone_evaluation_dispatch",
                                 "training_arguments_forwarded", "peer_failure_cleanup",
                                 "timeout_cleanup", "heartbeat_progress", "windows_rejected",
                                 "one_two_three_four_workers", "workers_auto_or_explicit"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main_test())
