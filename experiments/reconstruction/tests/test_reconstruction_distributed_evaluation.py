"""Generated evaluation regression: two processes share one physical CUDA GPU.

This validates view assignment, report merging, failure handling and automatic
post-training evaluation. It does not validate NCCL, physical two-GPU execution
or speed. Only test child processes substitute device selection and, for the
training case, host Pipe gradient reductions. All outputs default to temporary
directories outside the checkout.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_multigpu import Allocation, run_workers
from preview import RuntimeConfig
from test_reconstruction_distributed_training import HostPipeCollective, write_test_png


def audited_run(context, arguments, *, training=False):
    """Audit process ownership and substitute one physical device in this test."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--test-audit-dir", required=True)
    parser.add_argument("--test-fail-rank-one", action="store_true")
    test_args, production_args = parser.parse_known_args(arguments)
    import reconstruction_collective
    import reconstruction_common as common
    import evaluate
    import train

    rank = 0 if context is None else context.rank
    audit = {"rank": rank, "renders": [], "shared_report_writes": [], "status": "running"}
    original_runtime = common.initialize_runtime
    original_render = common.render_view
    original_reference = common.read_reference
    original_write = Path.write_text

    def initialize_same_gpu(runtime, output, report, *, cuda_device=None, cpu_threads=None):
        # A test-only override. Production workers select distinct logical GPUs.
        return original_runtime(runtime, output, report)

    def observed_render(model, render, camera, spp, seed, chunk_size, mi, dr, np):
        item = {"seed": int(seed), "spp": int(spp), "status": "running"}
        audit["renders"].append(item)
        result = original_render(model, render, camera, spp, seed, chunk_size, mi, dr, np)
        item["status"] = "passed"
        return result

    def reference_with_optional_failure(path, *args, **kwargs):
        if rank == 1 and test_args.test_fail_rank_one:
            audit["injected_reference_failure"] = str(path)
            audit["status"] = "failed"
            # Persist evidence before the supervised peer shutdown can stop
            # this process after the root publishes the coordinated failure.
            original_write(Path(test_args.test_audit_dir) / f"rank-{rank}.json",
                           json.dumps(audit, indent=2) + "\n", encoding="utf-8")
            path = Path(path).with_name("missing-rank-one-reference.png")
        return original_reference(path, *args, **kwargs)

    def observed_write(path, data, *args, **kwargs):
        if path.name in ("evaluation-report.json", "evaluation-config.json", "metrics.json"):
            audit["shared_report_writes"].append(str(path))
            if rank != 0:
                raise AssertionError(f"Rank {rank} attempted to write shared evaluation report: {path}")
        return original_write(path, data, *args, **kwargs)

    common.initialize_runtime = initialize_same_gpu
    train.initialize_runtime = initialize_same_gpu
    common.render_view = observed_render
    common.read_reference = reference_with_optional_failure
    Path.write_text = observed_write
    if training:
        reconstruction_collective.NcclCollective = HostPipeCollective
    module = train if training else evaluate
    sys.argv = [str(Path(module.__file__)), *production_args]
    try:
        code = module.main(distributed=context)
        audit["status"] = "passed" if code == 0 else "failed"
        audit["exit_code"] = code
        return code
    except BaseException as error:
        audit.update(status="failed", error=repr(error))
        raise
    finally:
        # Separate audit files cannot race with the application's shared reports.
        original_write(Path(test_args.test_audit_dir) / f"rank-{rank}.json",
                       json.dumps(audit, indent=2) + "\n", encoding="utf-8")


def evaluation_worker(context, arguments):
    return audited_run(context, arguments)


def training_worker(context, arguments):
    return audited_run(context, arguments, training=True)


def make_fixture(root):
    import numpy as np
    from reconstruction_checkpoint import save_checkpoint
    from preview import RenderConfig

    dataset = root / "dataset"
    dataset.mkdir()
    (dataset / "init_points.ply").write_text(
        "ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n0 0 0 100 140 180\n",
        encoding="ascii")
    # Both max_views=3 and max_views=5 produce odd totals across these splits.
    for split, count in (("train", 5), ("test", 2)):
        (dataset / split).mkdir()
        frames = []
        for index in range(count):
            write_test_png(dataset / split / f"r_{index}.png")
            position = (index - (count - 1) * .5) * .08
            matrix = [[1, 0, 0, position], [0, 1, 0, 0], [0, 0, 1, 3], [0, 0, 0, 1]]
            frames.append({"file_path": f"./{split}/r_{index}", "transform_matrix": matrix})
        (dataset / f"transforms_{split}.json").write_text(
            json.dumps({"camera_angle_x": .7, "frames": frames}), encoding="utf-8")
    y00 = 1 / (2 * math.sqrt(math.pi))
    model = {"cell_indices": np.array([0], np.uint32),
             "centers": np.array([.5, .5, .5], np.float32),
             "log_scales": np.log(np.array([.65, .55, .6], np.float32)),
             "rotations": np.array([1, 0, 0, 0], np.float32),
             "radiance_sh": np.array([.2, .3, .4], np.float32) / y00,
             "opacity_sh": np.array([.55], np.float32) / y00}
    metadata = {"resolution": 1, "grid_min": [-1., -1., -1.], "grid_max": [1., 1., 1.],
                "sh_count": 1, "opacity_sh_count": 1}
    checkpoint = root / "generated-model.npz"
    save_checkpoint(checkpoint, model, metadata,
        config={"data": {"dataset": str(dataset), "cameras": "transforms_train.json"},
                "render": asdict(RenderConfig(width=16, height=16, background=(1., 1., 1.)))})
    return dataset, checkpoint


def compare_evaluations(serial, parallel, splits, selected, np, *, save_linear=True):
    first = json.loads((serial / "evaluation-report.json").read_text(encoding="utf-8"))
    second = json.loads((parallel / "evaluation-report.json").read_text(encoding="utf-8"))
    assert first["status"] == second["status"] == "passed"
    assert first["splits"].keys() == second["splits"].keys() == set(splits)
    expected_seeds = []
    for split in splits:
        expected_indices = list(range(selected[split]))
        a, b = first["splits"][split], second["splits"][split]
        assert [row["index"] for row in a["views"]] == expected_indices
        assert [row["index"] for row in b["views"]] == expected_indices
        assert a["aggregate"] == b["aggregate"]
        assert a["evaluated_view_count"] == b["evaluated_view_count"] == selected[split]
        assert a["selected_view_count"] == b["selected_view_count"] == selected[split]
        assert json.loads((parallel / split / "metrics.json").read_text(encoding="utf-8")) == b
        expected_pngs = {f"{index:04d}.png" for index in expected_indices}
        assert {path.name for path in (parallel / split).glob("*.png")} == expected_pngs
        if save_linear:
            assert {path.name for path in (parallel / split).glob("*.npz")} == {
                f"{index:04d}.npz" for index in expected_indices}
        for reference, actual in zip(a["views"], b["views"]):
            index = reference["index"]
            seed = int(np.random.SeedSequence([first["requested_config"]["seed"],
                       0 if split == "train" else 1, index]).generate_state(1)[0])
            assert reference["seed"] == actual["seed"] == seed
            assert reference["metrics"] == actual["metrics"]
            assert reference["reference"] == actual["reference"]
            expected_seeds.append(seed)
            assert (serial / split / f"{index:04d}.png").read_bytes() == (
                parallel / split / f"{index:04d}.png").read_bytes()
            if save_linear:
                with np.load(serial / split / f"{index:04d}.npz") as expected:
                    with np.load(parallel / split / f"{index:04d}.npz") as image:
                        assert expected.files == image.files == ["rgb_linear", "alpha"]
                        for key in expected.files:
                            np.testing.assert_array_equal(expected[key], image[key])
    return expected_seeds


def check_audit(directory, seeds, *, spp, world_size=2):
    audits = [json.loads((directory / f"rank-{rank}.json").read_text(encoding="utf-8"))
              for rank in range(world_size)]
    for rank, audit in enumerate(audits):
        assert audit["status"] == "passed", audit
        assert [item["seed"] for item in audit["renders"]] == seeds[rank::world_size]
        assert all(item["spp"] == spp and item["status"] == "passed" for item in audit["renders"])
        if rank == 0:
            assert audit["shared_report_writes"]
        else:
            assert audit["shared_report_writes"] == []
    assert sum(len(audit["renders"]) for audit in audits) == len(seeds)
    return audits


def run_regression(root, build_dir, gpu):
    import numpy as np
    from train import GradientSettings

    dataset, checkpoint = make_fixture(root)

    def run_case(name, mode, arguments, *, expect_success=True, fail_rank_one=False):
        output = root / name
        if mode == "auto-training":
            output /= "training"
        audit = root / f"{name}-audit"
        audit.mkdir()
        command = [sys.executable, "-I", str(Path(__file__).resolve()), "--case", mode,
                   "--build-dir", build_dir, "--gpu", str(gpu), "--output-dir", str(output),
                   "--audit-dir", str(audit), *arguments]
        if fail_rank_one:
            command.append("--fail-rank-one")
        result = subprocess.run(command, text=True, capture_output=True, timeout=600)
        (root / f"{name}.log").write_text(result.stdout + result.stderr, encoding="utf-8")
        if expect_success != (result.returncode == 0):
            raise RuntimeError(f"{name} returned {result.returncode}:\n{result.stdout}\n{result.stderr}")
        return output, audit

    for limit in (3, 5):
        arguments = ["--checkpoint", str(checkpoint), "--dataset", str(dataset), "--split", "both",
                     "--spp", "2", "--max-views", str(limit), "--save-linear"]
        serial, _ = run_case(f"serial-first-{limit}", "serial", arguments)
        parallel, audit = run_case(f"two-first-{limit}", "distributed", arguments)
        seeds = compare_evaluations(serial, parallel, ("train", "test"),
                                   {"train": limit, "test": 2}, np)
        check_audit(audit, seeds, spp=2)

    arguments = ["--checkpoint", str(checkpoint), "--dataset", str(dataset), "--split", "test",
                 "--spp", "2", "--max-views", "1", "--save-linear"]
    serial, _ = run_case("serial-one-view", "serial", arguments)
    parallel, audit = run_case("two-one-view", "distributed", arguments)
    seeds = compare_evaluations(serial, parallel, ("test",), {"test": 1}, np)
    audits = check_audit(audit, seeds, spp=2)
    assert audits[1]["renders"] == []

    failed, audit = run_case("rank-one-reference-failure", "distributed",
        ["--checkpoint", str(checkpoint), "--dataset", str(dataset), "--split", "train",
         "--spp", "1", "--max-views", "2"], expect_success=False, fail_rank_one=True)
    failure_report = failed / "evaluation-report.json"
    assert failure_report.exists(), "Root must record the coordinated evaluation failure"
    failure = json.loads(failure_report.read_text(encoding="utf-8"))
    assert failure["status"] != "passed"
    assert not any(json.loads(path.read_text(encoding="utf-8")).get("status") == "passed"
                   for path in failed.glob("*/metrics.json"))
    failed_worker = json.loads((audit / "rank-1.json").read_text(encoding="utf-8"))
    assert failed_worker["status"] == "failed" and "injected_reference_failure" in failed_worker

    training_config = root / "auto-training.json"
    training_config.write_text(json.dumps({"data": {"dataset": str(dataset)},
        "model": {"grid_resolution": 4, "bounds_min": [-1., -1., -1.], "bounds_max": [1., 1., 1.],
                  "sh_count": 1, "opacity_sh_count": 1},
        "render": {"width": 16, "height": 16, "spp": 1},
        "training": {"rgb_only": False, "num_epochs": 1, "save_count": 1,
                     "profile_views": 0, "profile_interval_epochs": 0},
        "gradient": asdict(GradientSettings()),
        "schedule": {"enabled": False}, "topology": {"enable_deletion": False, "enable_growth": False},
        "optimizer": {"opacity_warmup_epochs": 0, "opacity_ramp_epochs": 0,
                      "lr_center": 0., "lr_log_scale": 0., "lr_rotation": 0.},
        "contour": {"enabled": True}, "evaluation": {"enabled": True, "spp": 64, "max_views": 2}}), encoding="utf-8")
    training, audit = run_case("automatic-64spp", "auto-training", ["--config", str(training_config)])
    training_report = json.loads((training / "training-report.json").read_text(encoding="utf-8"))
    assert training_report["training_status"] == training_report["evaluation"]["status"] == "passed"
    assert training_report["status"] == "passed" and training_report["global_step"] == 5
    trained_model = training.parent / "model.npz"
    assert trained_model.exists()
    serial, _ = run_case("automatic-model-serial-64spp", "serial",
        ["--checkpoint", str(trained_model), "--dataset", str(dataset), "--split", "both",
         "--spp", "64", "--max-views", "2"])
    seeds = compare_evaluations(serial, training.parent, ("train", "test"), {"train": 2, "test": 2},
                               np, save_linear=False)
    audits = check_audit(audit, seeds, spp=64)
    assert all(len(item["renders"]) == 2 for item in audits)
    result = {"status": "passed", "scope": "two processes on one physical CUDA GPU; test-only host training reductions",
              "checks": ["first 3 and first 5 views per split with odd combined task totals",
                         "per-view seeds, PNGs, linear buffers and merged metrics match serial",
                         "ordered exact coverage and root-only shared report writes",
                         "one total view with an empty second rank",
                         "rank-one reference failure cannot publish successful merged results",
                         "both ranks perform automatic 64-spp evaluation after training"],
              "physical_multi_gpu_validated": False, "nccl_validated": False}
    (root / "distributed-evaluation-tests.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--case", choices=("serial", "distributed", "auto-training"), help=argparse.SUPPRESS)
    parser.add_argument("--build-dir", default=RuntimeConfig().build_dir)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output-dir")
    parser.add_argument("--audit-dir", help=argparse.SUPPRESS)
    parser.add_argument("--fail-rank-one", action="store_true", help=argparse.SUPPRESS)
    args, remaining = parser.parse_known_args()
    if args.case:
        if not args.audit_dir:
            parser.error("Child test cases require --audit-dir")
        arguments = ["--test-audit-dir", args.audit_dir, "--build-dir", args.build_dir,
                     "--gpu", str(args.gpu), "--output-dir", args.output_dir, *remaining]
        if args.fail_rank_one:
            arguments.append("--test-fail-rank-one")
        if args.case == "serial":
            return audited_run(None, arguments)
        allocation = Allocation("TEST ONLY: same GPU", ("same_gpu_rank_0", "same_gpu_rank_1"), 2)
        worker = training_worker if args.case == "auto-training" else evaluation_worker
        return run_workers(arguments, allocation, worker_main=worker, worker_timeout=300.)
    if remaining:
        parser.error(f"Unknown arguments: {remaining}")
    if args.output_dir:
        root = Path(args.output_dir).resolve()
        root.mkdir(parents=True, exist_ok=False)
        result = run_regression(root, args.build_dir, args.gpu)
    else:
        with tempfile.TemporaryDirectory(prefix="mitsuba-two-process-evaluation-") as directory:
            result = run_regression(Path(directory), args.build_dir, args.gpu)
    print(json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
