"""Compare warmed reconstruction steps with native replay and GPU references.

Runs real 800x800 Lego views at 1 spp, including ray generation, reference
uploads, loss/backward, validity/loss readback, and one Adam update per view.
This is a fixed-topology step benchmark, not a full epoch or a training run.
It never opens or overwrites an existing experiment output directory.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import statistics
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))

from reconstruction_common import (
    REPOSITORY, WORKSPACE, ContourSampler, file_sha256, initialize_runtime,
    load_views, model_metadata, read_reference, restore_model, workspace_path,
)
from reconstruction_optimizer import FalcorOptimizer
from reconstruction_reference import GPUReferenceCache
from reconstruction_replay import NativeLossReplay
from reconstruction_train import TrainConfig, train_view


SEED = 194


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="datasets/lego")
    parser.add_argument("--cameras", type=int, nargs="+", default=[0, 21, 56],
                        help="Zero-based training camera indices, in the same order for every mode")
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument("--warmup", type=int, default=1,
                        help="Excluded rounds through all selected cameras (default: 1)")
    parser.add_argument("--repeats", type=int, default=3,
                        help="Timed rounds through all selected cameras (default: 3)")
    parser.add_argument("--chunk-size", type=int, default=0)
    parser.add_argument("--gpu-cache-mib", type=int, default=1024)
    parser.add_argument("--ablations", action="store_true",
                        help="Also measure replay-only and GPU-reference-only modes")
    parser.add_argument("--repeat-baseline", action="store_true",
                        help="Add an independent uncached direct run to measure baseline repeatability")
    parser.add_argument("--gpu", type=int, help="Local physical GPU index; ignored under Slurm")
    parser.add_argument("--build-dir", help="Source build directory; default follows the runtime policy")
    parser.add_argument("--output-dir", help="New or empty directory outside the repository; default: temporary directory")
    args = parser.parse_args(argv)
    if min(args.width, args.height, args.warmup, args.repeats, args.gpu_cache_mib) < 1:
        parser.error("width, height, warmup, repeats and gpu-cache-mib must be positive")
    if args.chunk_size < 0 or (args.gpu is not None and args.gpu < 0):
        parser.error("chunk-size and gpu must be nonnegative")
    if any(index < 0 for index in args.cameras) or len(set(args.cameras)) != len(args.cameras):
        parser.error("camera indices must be distinct and nonnegative")
    if args.output_dir and workspace_path(args.output_dir, WORKSPACE).is_relative_to(REPOSITORY):
        parser.error("output-dir must be outside the source repository")
    return args


def mode_state(name, use_replay, use_gpu_reference, payload, config, args, mi, dr, np):
    model = restore_model(payload, mi)
    optimizer = FalcorOptimizer(model, config.optimizer)
    dr.eval(*model.snapshot().values(),
            *[array for group in optimizer.state.values() for array in group.values()])
    return {"name": name, "model": model, "optimizer": optimizer,
            "rng": np.random.default_rng(SEED), "sampler": ContourSampler(np),
            "replay": NativeLossReplay(dr, enabled=use_replay),
            "reference_cache": GPUReferenceCache(args.width, args.height, mi, dr, np,
                capacity_mib=args.gpu_cache_mib if use_gpu_reference else 0),
            "native_replay": use_replay, "gpu_reference_cache": use_gpu_reference,
            "samples": []}


def execute_step(mode, view, rgba, round_index, config, mi, dr, np):
    # Independent states run serially on the same device. Include all work in
    # this step, while keeping the previous mode's queued work out of the timer.
    dr.sync_thread()
    started = time.perf_counter()
    reference = mode["reference_cache"].get(view["reference_path"], rgba)
    gradients, loss, _ = train_view(mode["model"], view, rgba, config,
        mode["sampler"], mode["rng"], mi, dr, np, collect_deletion_evidence=False,
        gpu_reference=reference, native_replay=mode["replay"])
    mode["optimizer"].step(mode["model"], gradients, sample_count=1,
                           epoch=round_index, coarse_to_fine=config.schedule.enabled)
    dr.sync_thread()
    return {"round": round_index, "camera": view["index"], "loss": loss,
            "wall_seconds": time.perf_counter() - started}


def final_arrays(mode, np):
    # These complete readbacks are only for a final numerical comparison and
    # occur after every timing sample, never inside the measured steps.
    result = {f"model.{name}": np.asarray(array).copy()
              for name, array in mode["model"].snapshot().items()}
    for group, values in mode["optimizer"].state.items():
        result.update({f"optimizer.{group}.{name}": np.asarray(array).copy()
                       for name, array in values.items()})
    result["losses"] = np.asarray([sample["loss"] for sample in mode["samples"]])
    return result


def compare_final(reference, result, np):
    if reference.keys() != result.keys():
        raise AssertionError("Compared modes produced different model/optimizer fields")
    fields = {}
    for name, expected in reference.items():
        actual = result[name]
        if actual.shape != expected.shape:
            raise AssertionError(f"Different shape for {name}: {actual.shape} != {expected.shape}")
        absolute_error = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
        error = float(np.max(absolute_error, initial=0))
        integer = np.issubdtype(expected.dtype, np.integer)
        atol = 1e-5 if name.startswith("optimizer.") else 2e-6
        matches = (actual == expected if integer
                   else np.isclose(actual, expected, atol=atol, rtol=3e-5))
        maximum_index = int(np.argmax(absolute_error)) if expected.size else None
        fields[name] = {"passed": bool(np.all(matches)), "max_absolute_error": error,
                        "mismatched_count": int(np.count_nonzero(~matches)),
                        "total_elements": int(expected.size),
                        "rms_absolute_error": float(np.sqrt(np.mean(absolute_error**2))) if expected.size else 0.,
                        # Flat, zero-based index in the serialized parameter or moment array.
                        "maximum_error_index": maximum_index,
                        "expected_at_max_error": expected.flat[maximum_index].item() if maximum_index is not None else None,
                        "actual_at_max_error": actual.flat[maximum_index].item() if maximum_index is not None else None,
                        "atol": 0 if integer else atol, "rtol": 0 if integer else 3e-5}
    return {"passed": all(value["passed"] for value in fields.values()),
            "max_absolute_error": max(value["max_absolute_error"] for value in fields.values()),
            "fields": fields}


def summarize_mode(mode, cameras, reference, np):
    measured = [sample for sample in mode["samples"] if not sample["warmup"]]
    times = [sample["wall_seconds"] for sample in measured]
    result = {"name": mode["name"], "native_replay": mode["native_replay"],
              "gpu_reference_cache": mode["gpu_reference_cache"],
              "step_wall_median_ms": statistics.median(times) * 1000,
              "step_wall_mean_ms": statistics.mean(times) * 1000,
              "step_wall_min_ms": min(times) * 1000, "step_wall_max_ms": max(times) * 1000,
              "camera_step_wall_median_ms": {
                  str(index): statistics.median(sample["wall_seconds"] for sample in measured
                                                 if sample["camera"] == index) * 1000
                  for index in cameras},
              "reference_cache": mode["reference_cache"].statistics(),
              "replay": mode["replay"].statistics(), "samples": mode["samples"],
              "comparison": compare_final(reference, final_arrays(mode, np), np)}
    print(json.dumps({name: value for name, value in result.items()
                      if name not in ("samples", "comparison")}
                     | {"comparison_passed": result["comparison"]["passed"],
                        "max_absolute_error": result["comparison"]["max_absolute_error"]}), flush=True)
    return result


def main(argv=None):
    args = parse_arguments(argv)
    config = TrainConfig()
    config.data.dataset = args.dataset
    config.model.grid_resolution = config.schedule.start_resolution = 256
    config.render.width, config.render.height = args.width, args.height
    config.render.spp = 1
    config.render.seed = config.training.seed = SEED
    config.render.pixel_sampling = "per_pixel_uniform"
    config.training.chunk_size = args.chunk_size
    config.training.profile_stages = False
    if args.gpu is not None:
        config.runtime.gpu = args.gpu
    if args.build_dir:
        config.runtime.build_dir = args.build_dir
    dataset = workspace_path(args.dataset, WORKSPACE)
    if not dataset.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset}")
    output = (workspace_path(args.output_dir, WORKSPACE) if args.output_dir
              else Path(tempfile.mkdtemp(prefix="mitsuba-execution-benchmark-")))
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output directory must be empty; existing outputs are preserved: {output}")
    output.mkdir(parents=True, exist_ok=True)
    print(f"Benchmark output: {output}", flush=True)
    report_path = output / "benchmark.json"
    report = {
        "status": "running", "dataset": str(dataset), "cameras": args.cameras,
        "scope": "Warmed full train_view + one Adam step on fixed initial 256 topology. "
                 "Includes CPU sampling, ray construction, reference upload/cache lookup, native forward/backward, "
                 "loss/validity readback, optimizer work, and final GPU synchronization. "
                 "Excludes PNG decoding, model initialization, topology operations, checkpointing and evaluation. "
                 "A round traverses only the selected cameras; it is not a full training epoch.",
        "comparison_protocol": "Every mode starts from the same model snapshot and empty Adam state, "
                               "uses the same ordered cameras and independent identically seeded NumPy RNG. "
                               "Mode order rotates for each camera to reduce systematic ordering bias. "
                               "Warmup updates are retained identically in every mode. Coarse-to-fine learning-rate "
                               "semantics are retained, while no topology changes or deletion evidence are requested.",
        "config": asdict(config), "seed": SEED, "warmup_rounds": args.warmup,
        "timed_rounds": args.repeats, "steps_per_mode": (args.warmup + args.repeats) * len(args.cameras),
        "modes": [],
    }
    modes = []
    try:
        mi, dr, np, handles = initialize_runtime(config.runtime, output, report)
        report["provenance"]["source_files_sha256"]["resources/benchmark_reconstruction_execution.py"] = file_sha256(Path(__file__))
        views = load_views(dataset, config.data.cameras, config.render, np)
        if max(args.cameras) >= len(views):
            raise ValueError(f"Camera indices must be below {len(views)}")
        selected = [views[index] for index in args.cameras]
        references = {}
        for view in selected:
            rgba = read_reference(view["reference_path"], args.width, args.height, mi, np)
            rgba.setflags(write=False)
            references[view["index"]] = rgba
        required_bytes = sum(array.nbytes for array in references.values())
        if required_bytes > args.gpu_cache_mib * 1024 * 1024:
            raise ValueError("gpu-cache-mib must fit all selected images for a warmed-cache comparison")
        report["reference_bytes"] = required_bytes
        report["reference_sha256"] = {
            str(view["index"]): file_sha256(Path(view["reference_path"])) for view in selected}
        report["cameras_sha256"] = file_sha256(dataset / config.data.cameras)
        point_cloud = dataset / config.data.point_cloud
        report["point_cloud_sha256"] = file_sha256(point_cloud)
        low, high = np.asarray(config.model.bounds_min), np.asarray(config.model.bounds_max)
        center, half = (low + high) / 2, (high - low) / 2 * config.model.bounds_padding
        initialized = mi.VoxelReconstruction(str(point_cloud), config.model.grid_resolution,
            (center - half).tolist(), (center + half).tolist(), config.model.opacity_threshold,
            config.model.sh_count, config.model.opacity_sh_count)
        payload = {"model_metadata": model_metadata(initialized),
                   "model_state": {name: np.asarray(array).copy()
                                   for name, array in initialized.snapshot().items()}}
        report["active_voxels"] = int(initialized.active_voxel_count)
        del initialized
        definitions = [("baseline", False, False), ("replay_and_gpu_reference", True, True)]
        if args.ablations:
            definitions = [("baseline", False, False), ("replay_only", True, False),
                           ("gpu_reference_only", False, True), ("replay_and_gpu_reference", True, True)]
        if args.repeat_baseline:
            definitions.insert(1, ("baseline_repeat", False, False))
        modes = [mode_state(name, replay, cache, payload, config, args, mi, dr, np)
                 for name, replay, cache in definitions]
        for round_index in range(args.warmup + args.repeats):
            for view_offset, view in enumerate(selected):
                rotation = (round_index * len(selected) + view_offset) % len(modes)
                for mode in modes[rotation:] + modes[:rotation]:
                    sample = execute_step(mode, view, references[view["index"]], round_index,
                                          config, mi, dr, np)
                    sample["warmup"] = round_index < args.warmup
                    mode["samples"].append(sample)
            print(json.dumps({"completed_round": round_index + 1,
                              "warmup": round_index < args.warmup}), flush=True)
        reference = final_arrays(modes[0], np)
        for mode in modes:
            if mode["rng"].bit_generator.state != modes[0]["rng"].bit_generator.state:
                raise AssertionError("Modes consumed different random sequences")
            report["modes"].append(summarize_mode(mode, args.cameras, reference, np))
        baseline_ms = report["modes"][0]["step_wall_median_ms"]
        for result in report["modes"]:
            result["speedup_vs_baseline"] = baseline_ms / result["step_wall_median_ms"]
        report["status"] = ("complete" if all(mode["comparison"]["passed"] for mode in report["modes"])
                            else "numerical_mismatch")
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        for mode in modes:
            mode["replay"].clear()
            mode["reference_cache"].clear()
    print(f"Report: {report_path}", flush=True)
    return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
