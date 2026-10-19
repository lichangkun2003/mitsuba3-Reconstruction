"""Full training/control regression with two processes sharing one physical GPU.

THIS IS NOT A MULTI-GPU PERFORMANCE OR NCCL TEST. The test replaces only the
collective with host Pipe reductions and maps both simulated ranks to GPU zero.
Production training, WorkerContext control messages, checkpoint/resume, contour
sampling and the coarse-to-fine transition run unchanged. No production CPU
fallback is installed. All generated data and outputs default to a temp folder.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import time
import zlib

sys.path.insert(0, str(Path(__file__).resolve().parent))

from reconstruction_multigpu import Allocation, run_workers
from reconstruction_preview import RuntimeConfig


class HostPipeCollective:
    """Test-only synchronous reference reduction; never used by production."""

    def __init__(self, rank, world_size, connection, mi, dr, np, library=None):
        self.rank, self.world_size, self.connection = rank, world_size, connection
        self.mi, self.dr, self.np = mi, dr, np
        self.views, self.seconds, self.payload_bytes = 0, 0., 0

    def reduce_view(self, accumulator, loss_sum, finite, raw_alpha=None, alpha=None):
        np, mi = self.np, self.mi
        started = time.perf_counter()
        self.dr.sync_thread()
        local = {"gradients": {key: np.asarray(value).copy()
                              for key, value in accumulator["gradients"].items()},
                 "counters": {key: np.asarray(value).copy()
                              for key, value in accumulator.items() if key != "gradients"},
                 "loss": np.asarray(loss_sum).copy(), "finite": np.asarray(finite).copy(),
                 "raw": None if raw_alpha is None else np.asarray(raw_alpha).copy(),
                 "alpha": None if alpha is None else np.asarray(alpha).copy()}
        if self.rank == 0:
            tag, sequence, peer = self.connection.recv()
            if tag != "test_view_shard" or sequence != self.views:
                raise RuntimeError("Test collective order differs between workers")
            global_state = {"gradients": {key: value + peer["gradients"][key]
                                          for key, value in local["gradients"].items()},
                            "counters": {}}
            for key, value in local["counters"].items():
                global_state["counters"][key] = (np.maximum(value, peer["counters"][key])
                    if key in ("foreground_evidence", "background_evidence")
                    else value + peer["counters"][key])
            global_state["loss"] = local["loss"] + peer["loss"]
            global_state["finite"] = local["finite"] & peer["finite"]
            for key in ("raw", "alpha"):
                if (local[key] is None) != (peer[key] is None):
                    raise RuntimeError("Test workers disagree on required image reductions")
                global_state[key] = None if local[key] is None else local[key] + peer[key]
            self.connection.send(("test_view_global", self.views, global_state))
        else:
            self.connection.send(("test_view_shard", self.views, local))
            tag, sequence, global_state = self.connection.recv()
            if tag != "test_view_global" or sequence != self.views:
                raise RuntimeError("Test collective result order differs between workers")
        self.views += 1
        self.seconds += time.perf_counter() - started
        self.payload_bytes += sum(value.nbytes for value in local["gradients"].values())
        self.payload_bytes += sum(value.nbytes for value in local["counters"].values())
        self.payload_bytes += local["loss"].nbytes + local["finite"].nbytes
        self.payload_bytes += sum(local[key].nbytes for key in ("raw", "alpha") if local[key] is not None)
        result = {"gradients": {key: mi.Float(value) for key, value in global_state["gradients"].items()}}
        result.update({key: mi.UInt32(value) for key, value in global_state["counters"].items()})
        return (result, mi.Float(global_state["loss"]), mi.Bool(global_state["finite"]),
                None if global_state["raw"] is None else mi.Float(global_state["raw"]),
                None if global_state["alpha"] is None else mi.Float(global_state["alpha"]))

    def statistics(self):
        return {"backend": "TEST ONLY: host Pipe, same physical GPU", "rank": self.rank,
                "world_size": self.world_size, "views": self.views, "calls": self.views,
                "seconds": self.seconds, "payload_bytes": self.payload_bytes}

    def close(self, abort=False):
        pass


def simulated_worker(context, arguments):
    """Spawn entry point; monkeypatch only inside this test-owned child process."""
    import reconstruction_collective
    import reconstruction_train

    actual_initialize = reconstruction_train.initialize_runtime

    def initialize_same_gpu(runtime, output, report, *, cuda_device=None, cpu_threads=None):
        # Ignore the requested logical rank here only: this machine has one GPU.
        return actual_initialize(runtime, output, report)

    reconstruction_train.initialize_runtime = initialize_same_gpu
    reconstruction_collective.NcclCollective = HostPipeCollective
    sys.argv = [str(Path(reconstruction_train.__file__)), *arguments]
    return reconstruction_train.main(distributed=context)


def write_test_png(path, width=16, height=16):
    """Write an RGBA PNG using only the standard library; no parent CUDA context."""
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xffffffff)

    rows = bytearray()
    for y in range(height):
        rows.append(0)
        for x in range(width):
            inside = 4 <= x < 12 and 4 <= y < 12
            alpha = 128 if inside and y == 4 else 255 if inside else 0
            rows.extend((153, 51, 26, alpha))
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0))
                     + chunk(b"sRGB", b"\x00") + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b""))


def generate_dataset(root):
    dataset = root / "dataset"
    (dataset / "train").mkdir(parents=True)
    (dataset / "init_points.ply").write_text(
        "ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n0 0 0 100 140 180\n",
        encoding="ascii")
    matrix = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 3], [0, 0, 0, 1]]
    frames = []
    for index in range(2):
        write_test_png(dataset / "train" / f"r_{index}.png")
        frames.append({"file_path": f"./train/r_{index}", "transform_matrix": matrix})
    (dataset / "transforms_train.json").write_text(json.dumps({"camera_angle_x": .7, "frames": frames}), encoding="utf-8")
    return dataset


def compare_checkpoint(expected, actual, np):
    assert expected["model_metadata"] == actual["model_metadata"]
    for key, value in expected["model_state"].items():
        if key == "cell_indices":
            np.testing.assert_array_equal(value, actual["model_state"][key])
        else:
            np.testing.assert_allclose(value, actual["model_state"][key], atol=3e-6, rtol=3e-5, err_msg=key)
    for name, state in expected["optimizer_state"]["groups"].items():
        for key, value in state.items():
            if key == "steps":
                np.testing.assert_array_equal(value, actual["optimizer_state"]["groups"][name][key])
            else:
                np.testing.assert_allclose(value, actual["optimizer_state"]["groups"][name][key],
                                           atol=3e-6, rtol=3e-5, err_msg=f"Adam {name}.{key}")
    for key in ("completed_epochs", "global_step", "rng_state", "pending_epoch_transition", "dataset_signature"):
        assert expected["training_state"][key] == actual["training_state"][key], key
    first = expected["training_state"]["contour_state"]
    second = actual["training_state"]["contour_state"]
    assert first.keys() == second.keys()
    for key, state in first.items():
        np.testing.assert_array_equal(state["pixels"], second[key]["pixels"])
        np.testing.assert_allclose(state["mean_alpha"], second[key]["mean_alpha"], atol=3e-6, rtol=3e-5)
    first_topology = expected["training_state"]["topology_state"]
    second_topology = actual["training_state"]["topology_state"]
    assert first_topology.keys() == second_topology.keys()
    for key, value in first_topology.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(value, second_topology[key], err_msg=f"topology {key}")
        else:
            assert value == second_topology[key], f"topology {key}"


def run_regression(root, build_dir, gpu):
    import numpy as np
    from reconstruction_checkpoint import load_checkpoint

    dataset = generate_dataset(root)
    config = {"data": {"dataset": str(dataset)},
              "model": {"grid_resolution": 4, "bounds_min": [-1., -1., -1.], "bounds_max": [1., 1., 1.],
                        "sh_count": 1, "opacity_sh_count": 1},
              "render": {"width": 16, "height": 16, "spp": 1},
              "runtime": {"build_dir": build_dir, "gpu": gpu},
              "training": {"num_epochs": 3, "chunk_size": 0, "profile_views": 0,
                           "profile_interval_epochs": 0, "save_count": 3},
              "schedule": {"enabled": True, "start_resolution": 4, "target_resolution": 8,
                           "enable_coarse_growth": False},
              "optimizer": {"opacity_warmup_epochs": 0, "opacity_ramp_epochs": 0,
                            "lr_center": 0., "lr_log_scale": 0., "lr_rotation": 0.},
              "topology": {"enable_deletion": True, "enable_growth": False},
              "contour": {"enabled": True}, "evaluation": {"enabled": False}}
    configuration = root / "config.json"
    configuration.write_text(json.dumps(config), encoding="utf-8")

    def run_case(name, mode, resume=None, stop_after=None):
        output = root / name
        arguments = ["--output-dir", str(output), "--build-dir", build_dir, "--gpu", str(gpu), "--skip-evaluation"]
        arguments += ["--resume", str(resume)] if resume else ["--config", str(configuration)]
        if stop_after is not None:
            arguments += ["--stop-after-epochs", str(stop_after)]
        command = [sys.executable, "-I", str(Path(__file__).resolve()), "--case", mode, *arguments]
        result = subprocess.run(command, text=True, capture_output=True, timeout=600)
        (root / f"{name}.log").write_text(result.stdout + result.stderr, encoding="utf-8")
        if result.returncode:
            raise RuntimeError(f"{name} failed (code {result.returncode}):\n{result.stdout}\n{result.stderr}")
        report = json.loads((output / "training-report.json").read_text(encoding="utf-8"))
        assert report["training_status"] == ("stopped" if stop_after else "passed")
        assert report["evaluation"]["status"] == "skipped"
        expected_epochs = stop_after or 3
        assert report["completed_epochs"] == expected_epochs and report["global_step"] == expected_epochs * 2
        if mode == "distributed":
            assert report["distributed"]["world_size"] == 2 and report["distributed"]["rank"] == 0
            assert report["collective"]["backend"].startswith("TEST ONLY")
            assert report["collective"]["views"] == (2 * stop_after if stop_after else 4 if resume else 6)
        if stop_after:
            assert report["status"] == "stopped"
            assert not (output / "model.npz").exists()
            assert not (output / "evaluation-report.json").exists()
            checkpoint_path = output / f"epoch-{stop_after:05d}.npz"
            assert Path(report["checkpoint"]) == checkpoint_path
            stopped = load_checkpoint(checkpoint_path)
            assert stopped["training_state"]["completed_epochs"] == stop_after
            assert stopped["training_state"]["global_step"] == stop_after * 2
            assert not stopped["training_state"]["pending_epoch_transition"]
            assert stopped["config"]["training"]["num_epochs"] == 3
            assert stopped["config"]["schedule"]["total_epochs"] == 3
            return output, stopped
        return output, load_checkpoint(output / "model.npz")

    single_path, single = run_case("single", "single")
    dual_path, dual = run_case("simulated-two", "distributed")
    compare_checkpoint(single, dual, np)
    first = load_checkpoint(dual_path / "epoch-00001.npz")
    assert first["model_metadata"]["resolution"] == 8
    assert first["training_state"]["contour_state"] == {}
    assert single["model_metadata"]["resolution"] == 8
    resumed_path, resumed = run_case("single-to-two", "distributed", single_path / "epoch-00001.npz")
    compare_checkpoint(single, resumed, np)
    _, returned = run_case("single-to-two-to-single", "single", resumed_path / "epoch-00002.npz")
    compare_checkpoint(single, returned, np)
    stopped_path, stopped = run_case("simulated-two-stopped", "distributed", stop_after=1)
    assert stopped["model_metadata"]["resolution"] == 8
    assert {path.name for path in stopped_path.glob("*.npz")} == {"epoch-00001.npz"}
    _, completed = run_case("simulated-two-stopped", "distributed", stopped_path / "epoch-00001.npz")
    compare_checkpoint(single, completed, np)
    result = {"status": "passed", "scope": "test-only host reductions; two processes on one physical GPU",
              "checks": ["same-view SUM/MAX reductions through full training loop",
                         "coarse 4 to fine 8 topology synchronization",
                         "three epochs and six logical Adam updates",
                         "contour, RNG, optimizer and lifecycle state agreement",
                         "single to two to single checkpoint portability",
                         "bounded stop preserves total schedule, skips final export, resumes identically",
                         "primary-only model and report output"],
              "nccl_validated": False, "physical_multi_gpu_validated": False}
    (root / "distributed-training-tests.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--case", choices=("single", "distributed"), help=argparse.SUPPRESS)
    parser.add_argument("--build-dir", default=RuntimeConfig().build_dir)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output-dir")
    args, remaining = parser.parse_known_args()
    if args.case:
        arguments = ["--build-dir", args.build_dir, "--gpu", str(args.gpu)]
        if args.output_dir:
            arguments += ["--output-dir", args.output_dir]
        arguments += remaining
        if args.case == "single":
            import reconstruction_train
            sys.argv = [str(Path(reconstruction_train.__file__)), *arguments]
            return reconstruction_train.main()
        allocation = Allocation("TEST ONLY: same GPU", ("same_gpu_rank_0", "same_gpu_rank_1"), 2)
        return run_workers(arguments, allocation, worker_main=simulated_worker, worker_timeout=180.)
    if remaining:
        parser.error(f"Unknown arguments: {remaining}")
    if args.output_dir:
        root = Path(args.output_dir).resolve()
        root.mkdir(parents=True, exist_ok=False)
        result = run_regression(root, args.build_dir, args.gpu)
    else:
        with tempfile.TemporaryDirectory(prefix="mitsuba-two-process-training-") as directory:
            result = run_regression(Path(directory), args.build_dir, args.gpu)
    print(json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
