"""Benchmark fixed reconstruction inputs without optimizer or topology updates.

Times cover the complete native loss_and_grad call, including its backward work.
Separate diagnostic samples expose forward/hit_backward/candidate_backward.
Relative dataset/build/output paths use the parent of this source checkout.
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
    REPOSITORY, WORKSPACE, file_sha256, generate_ray_batch, initialize_runtime,
    load_views, prepare_camera, read_reference, workspace_path,
)
from reconstruction_train import TrainConfig, kernel_profile, settings


SEED = 194
PROFILE_REPEATS = 3
HISTORY_REPEATS = 3
STAGES = ("forward", "hit_backward", "candidate_backward")


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="datasets/lego")
    parser.add_argument("--cameras", type=int, nargs="+", default=[0, 21, 56],
                        help="Zero-based training camera indices (default: 0 21 56)")
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--gpu", type=int, help="Local physical GPU index; ignored under Slurm")
    parser.add_argument("--build-dir", help="Source build directory; defaults to the existing runtime policy")
    parser.add_argument("--output-dir", help="Directory outside the repository; defaults to a new temporary directory")
    parser.add_argument("--save-arrays", action="store_true",
                        help="Save complete native result arrays and counters for each camera as NPZ")
    args = parser.parse_args(argv)
    if min(args.width, args.height, args.repeats) < 1 or args.warmup < 0:
        parser.error("width, height and repeats must be positive; warmup must be nonnegative")
    if any(index < 0 for index in args.cameras) or len(set(args.cameras)) != len(args.cameras):
        parser.error("camera indices must be distinct and nonnegative")
    if args.gpu is not None and args.gpu < 0:
        parser.error("gpu must be nonnegative")
    if args.output_dir and workspace_path(args.output_dir, WORKSPACE).is_relative_to(REPOSITORY):
        parser.error("output-dir must be outside the source repository")
    return args


def result_arrays(result):
    """Flatten GPU result fields without traversing diagnostic scalar dictionaries."""
    for name, value in result.items():
        if name == "stage_profile":
            continue
        if isinstance(value, dict):
            for field, array in value.items():
                yield f"{name}.{field}", array
        else:
            yield name, value


def native_call(model, inputs, options, dr):
    result = model.loss_and_grad(*inputs, options)
    dr.eval(*(array for _, array in result_arrays(result)))
    dr.sync_thread()
    return result


def ordinary_kernel_history(model, inputs, options, dr):
    """Capture whole-call device work without changing native stage boundaries."""
    ordinary_options = dict(options, profile_stages=False)
    samples = []
    for sample in range(1 + HISTORY_REPEATS):
        with dr.kernel_history() as history:
            started = time.perf_counter()
            result = native_call(model, inputs, ordinary_options, dr)
            wall_seconds = time.perf_counter() - started
        del result
        if sample:  # One independently captured warmup is excluded.
            samples.append(dict(kernel_profile(history), wall_seconds=wall_seconds))
    medians = {key: statistics.median(sample[key] for sample in samples)
               for key, value in samples[0].items() if isinstance(value, (int, float))}
    return samples, medians


def benchmark_view(model, view, config, options, args, output, mi, dr, np):
    count = config.render.width * config.render.height
    reference = read_reference(view["reference_path"], config.render.width,
                               config.render.height, mi, np).reshape(-1, 4)
    seed = SEED + view["index"]
    offsets = np.random.default_rng(seed).random((count, 2)).astype(np.float32)
    np.minimum(offsets, np.nextafter(np.float32(1), np.float32(0)), out=offsets)
    camera = prepare_camera(config.render, view["camera"], mi, dr)
    ray = generate_ray_batch(config.render, view["camera"], np.arange(count, dtype=np.uint32),
                             offsets, mi, dr, prepared_camera=camera)
    rgb = mi.Color3f(*(mi.Float(reference[:, channel]) for channel in range(3)))
    alpha = mi.Float(reference[:, 3])
    background = dr.opaque(mi.Color3f, config.render.background)
    inverse_pdf = dr.opaque(mi.Float, 1.)
    inputs = (ray, rgb, alpha, background, inverse_pdf)
    # Upload and finish all ray construction before either timing mode starts.
    dr.eval(ray.o, ray.d, ray.maxt, rgb, alpha, background, inverse_pdf)
    dr.sync_thread()

    for _ in range(args.warmup):
        result = native_call(model, inputs, options, dr)
        del result
    wall_seconds = []
    for sample in range(args.repeats):
        started = time.perf_counter()
        result = native_call(model, inputs, options, dr)
        wall_seconds.append(time.perf_counter() - started)
        if args.save_arrays and sample == args.repeats - 1:
            arrays_path = output / f"camera-{view['index']:03d}.npz"
            np.savez(arrays_path, **{name: np.asarray(array).copy()
                                    for name, array in result_arrays(result)})
        del result

    history_samples, history_medians = ordinary_kernel_history(model, inputs, options, dr)

    profile_options = dict(options, profile_stages=True)
    # Profiling changes evaluation boundaries; warm its kernels independently.
    result = native_call(model, inputs, profile_options, dr)
    if "stage_profile" not in result:
        raise RuntimeError("Native stage profiling is missing; rebuild this checkout")
    del result
    profile_samples = []
    for _ in range(PROFILE_REPEATS):
        result = native_call(model, inputs, profile_options, dr)
        profile_samples.append({stage: {key: float(value) for key, value in metrics.items()}
                                for stage, metrics in result["stage_profile"].items()})
        del result
    stage_medians = {stage: {key: statistics.median(sample[stage][key] for sample in profile_samples)
                            for key in profile_samples[0][stage]} for stage in STAGES}
    report = {
        "camera_index": view["index"], "seed": seed, "reference": view["reference_path"],
        "reference_sha256": file_sha256(Path(view["reference_path"])),
        "ordinary_native_wall_seconds": wall_seconds,
        "ordinary_native_wall_median_seconds": statistics.median(wall_seconds),
        "ordinary_kernel_history_samples": history_samples,
        "ordinary_kernel_history_medians": history_medians,
        "profile_stage_medians": stage_medians, "profile_samples": profile_samples,
    }
    if args.save_arrays:
        report["arrays"] = str(arrays_path)
    print(json.dumps({"camera": view["index"],
                      "ordinary_native_wall_median_ms": statistics.median(wall_seconds) * 1000,
                      "ordinary_kernel_history_device_ms": history_medians["device_ms"],
                      "profile_forward_device_ms": stage_medians["forward"]["device_ms"]}), flush=True)
    return report


def main(argv=None):
    args = parse_arguments(argv)
    config = TrainConfig()
    config.data.dataset = args.dataset
    config.model.grid_resolution = 256
    config.render.width, config.render.height = args.width, args.height
    config.render.spp = 1
    config.render.seed = SEED
    config.render.pixel_sampling = "per_pixel_uniform"
    if args.gpu is not None:
        config.runtime.gpu = args.gpu
    if args.build_dir:
        config.runtime.build_dir = args.build_dir
    dataset = workspace_path(config.data.dataset, WORKSPACE)
    if not dataset.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset}")
    output = (workspace_path(args.output_dir, WORKSPACE) if args.output_dir
              else Path(tempfile.mkdtemp(prefix="mitsuba-forward-benchmark-")))
    output.mkdir(parents=True, exist_ok=True)
    print(f"Benchmark output: {output}", flush=True)
    report_path = output / "benchmark.json"
    options = settings(config)
    # Match ordinary views without a scheduled topology-evidence collection.
    options.update(collect_deletion_evidence=False, profile_stages=False)
    report = {
        "scope": "Fixed initialized model, 1 spp full-image native loss_and_grad; not an epoch. "
                 "Timed calls exclude input upload, image I/O, ray generation, optimizer, topology "
                 "updates and result readback. All native GPU result fields are evaluated and synchronized.",
        "profiling_note": "Ordinary wall samples disable native profiling. Separate diagnostic samples "
                          "synchronize forward/hit_backward/candidate_backward boundaries and may change execution.",
        "ordinary_kernel_history_note": "Separate whole-call kernel-history samples keep profile_stages=false "
                                        "and evaluate every native GPU output before synchronization. Device time "
                                        "is the sum of captured operation times; these are not stage-profile samples.",
        "dataset": str(dataset), "cameras": args.cameras,
        "render": asdict(config.render), "model": asdict(config.model),
        "native_settings": options, "warmup": args.warmup, "repeats": args.repeats,
        "profile_warmup": 1, "profile_repeats": PROFILE_REPEATS,
        "ordinary_kernel_history_warmup": 1, "ordinary_kernel_history_repeats": HISTORY_REPEATS,
        "seed_rule": f"NumPy PCG64 seed {SEED} + zero-based camera index",
        "views": [],
    }
    try:
        mi, dr, np, handles = initialize_runtime(config.runtime, output, report)
        report["provenance"]["source_files_sha256"]["resources/benchmark_reconstruction_forward.py"] = file_sha256(Path(__file__))
        views = load_views(dataset, config.data.cameras, config.render, np)
        if max(args.cameras) >= len(views):
            raise ValueError(f"Camera indices must be below {len(views)}")
        low, high = np.asarray(config.model.bounds_min), np.asarray(config.model.bounds_max)
        center, half = (low + high) / 2, (high - low) / 2 * config.model.bounds_padding
        point_cloud = dataset / config.data.point_cloud
        report["point_cloud_sha256"] = file_sha256(point_cloud)
        report["cameras_sha256"] = file_sha256(dataset / config.data.cameras)
        model = mi.VoxelReconstruction(str(point_cloud), config.model.grid_resolution,
            (center - half).tolist(), (center + half).tolist(), config.model.opacity_threshold,
            config.model.sh_count, config.model.opacity_sh_count)
        dr.eval(*model.snapshot().values())
        dr.sync_thread()
        report["active_voxels"] = int(model.active_voxel_count)
        for index in args.cameras:
            report["views"].append(benchmark_view(model, views[index], config, options, args,
                                                   output, mi, dr, np))
            report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        report["status"] = "complete"
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Report: {report_path}", flush=True)


if __name__ == "__main__":
    main()
