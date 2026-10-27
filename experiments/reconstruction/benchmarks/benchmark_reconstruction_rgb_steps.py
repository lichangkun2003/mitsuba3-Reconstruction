"""Measure warmed RGB training steps on a real scene and fixed initial topology.

The default is bicycle, full on-disk images, and first/middle/last training
cameras. Every view updates Adam once. A round reads deferred checks and waits
for all GPU work; its wall time includes sampling, rays, loss/backward, Adam,
and that final wait. Initialization, PNG decoding, topology, model saving and
evaluation are excluded. Optional profiling runs AFTER all throughput rounds.
"""

from __future__ import annotations

import argparse
import copy
import cProfile
from dataclasses import asdict
import io
import json
from pathlib import Path
import pstats
import statistics
import sys
import time
import traceback
import uuid

# Python -I deliberately excludes the script directory; import this checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reconstruction_checks import DeferredViewChecks
from reconstruction_common import (
    REPOSITORY, WORKSPACE, file_sha256, initialize_model,
    initialize_runtime, load_views, prepare_dataset, read_reference, workspace_path,
)
from reconstruction_optimizer import FalcorOptimizer
from reconstruction_reference import GPUReferenceCache
from reconstruction_replay import NativeLossReplay
from reconstruction_sampling import PrefetchedUniformSampler
from train import CONFIG, kernel_profile, train_view


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="datasets/bicycle")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--views", type=int, default=3,
                           help="Select N evenly spaced cameras including first/last (default: 3)")
    selection.add_argument("--cameras", type=int, nargs="+",
                           help="Explicit zero-based positions in the training camera list")
    parser.add_argument("--warmup", type=int, default=2,
                        help="Excluded complete rounds, retaining their Adam updates (default: 2)")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--chunk-size", type=int, default=0,
                        help="0: complete image; positive: rays per chunk, still one Adam update per view")
    parser.add_argument("--sampling-workers", type=int, default=0,
                        help="CPU uniform-sampling prefetch workers; 0 keeps the synchronous baseline")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--build-dir", help="Source build directory, resolved relative to the workspace")
    parser.add_argument("--gpu-cache-mib", type=int, default=CONFIG.training.gpu_reference_cache_mib)
    parser.add_argument("--output-dir", help="New or empty directory outside the source repository")
    parser.add_argument("--cprofile", action="store_true",
                        help="Profile a separate round; write cprofile.prof and top 50 entries")
    parser.add_argument("--kernel-history", action="store_true",
                        help="After throughput measurement, capture CUDA times/counts in two diagnostic rounds")
    parser.add_argument("--native-profile", action="store_true",
                        help="After throughput measurement, run synchronized native phase diagnostics")
    args = parser.parse_args(argv)
    if min(args.views, args.warmup, args.repeats, args.gpu_cache_mib) < 1:
        parser.error("views, warmup, repeats and gpu-cache-mib must be positive")
    if args.chunk_size < 0 or args.gpu < 0 or args.sampling_workers < 0:
        parser.error("chunk-size, gpu and sampling-workers must be nonnegative")
    if args.cameras and (min(args.cameras) < 0 or len(set(args.cameras)) != len(args.cameras)):
        parser.error("camera indices must be distinct and nonnegative")
    if args.output_dir and workspace_path(args.output_dir, WORKSPACE).is_relative_to(REPOSITORY):
        parser.error("output-dir must be outside the source repository")
    return args


def selected_indices(count, args):
    if args.cameras is not None:
        if max(args.cameras) >= count:
            raise ValueError(f"Camera indices must be below {count}")
        return args.cameras
    if args.views > count:
        raise ValueError(f"Requested {args.views} views, but only {count} cameras exist")
    return ([0] if args.views == 1 else
            [round(index * (count - 1) / (args.views - 1)) for index in range(args.views)])


def execute_round(state, views, references, config, round_index, mi, dr, np, *, native_profile=False):
    """One checked, fully synchronized sample; never synchronize per view."""
    checks = DeferredViewChecks(mi, dr, np)
    dr.sync_thread()
    started = time.perf_counter()
    submissions = []
    for view in views:
        view_started = time.perf_counter()
        reference = references[view["index"]]
        gpu_reference = state["cache"].get(view["reference_path"], reference)
        stages = {} if native_profile else None
        gradients, _, _ = train_view(
            state["model"], view, reference, config, state["sampler"], state["rng"], mi, dr, np,
            collect_deletion_evidence=False, collect_growth_evidence=False,
            gpu_reference=gpu_reference, native_replay=state["replay"],
            deferred_checks=checks, stage_profile=stages,
        )
        result = state["optimizer"].step(
            state["model"], gradients, sample_count=1, epoch=round_index,
            coarse_to_fine=config.schedule.enabled, update_allowed=gradients.get("_update_allowed"),
        )
        checks.updated(result["accepted"])
        submission = {"camera": int(view["index"]),
                      "submit_wall_seconds": time.perf_counter() - view_started}
        if stages is not None:
            submission["native_stages"] = stages
        submissions.append(submission)
        # Do not keep the previous view's dense gradient table alive while
        # the next native call allocates its output buffers.
        del gradients, result
    validation_started = time.perf_counter()
    measurements = checks.read(rgb_only=True, distortion_weight=config.gradient.distortion_loss_weight)
    validation_seconds = time.perf_counter() - validation_started
    sync_started = time.perf_counter()
    dr.sync_thread()
    finished = time.perf_counter()
    return {
        "round": round_index, "wall_seconds": finished - started,
        "seconds_per_view": (finished - started) / len(views),
        "validation_seconds": validation_seconds, "final_sync_seconds": finished - sync_started,
        "mean_loss": statistics.mean(row["loss"] for row in measurements),
        "views": [dict(submission, **measurement)
                  for submission, measurement in zip(submissions, measurements)],
    }


def write_report(output, report):
    (output / "benchmark.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def main(argv=None):
    args = parse_arguments(argv)
    config = copy.deepcopy(CONFIG)
    config.data.dataset = args.dataset
    config.data.scene_mode = "auto"  # bicycle resolves to the current unbounded initializer.
    config.data.initialization = None
    config.data.background_final_layers = 8
    config.data.background_opacity = .5
    config.model.grid_resolution = config.schedule.start_resolution = 256
    # Match the requested training schedule's initialization/LR conventions;
    # this driver never constructs a topology manager or advances levels.
    config.schedule.enabled = True
    config.schedule.target_resolution = 1024
    config.schedule.level_weights = (4, 3, 3)
    config.render.spp = 1
    config.render.traversal = "octree"
    config.training.rgb_only = True
    config.training.chunk_size = args.chunk_size
    config.training.sampling_workers = args.sampling_workers
    config.training.freeze_native = True
    config.training.sync_view_checks = False
    config.training.profile_views = config.training.profile_interval_epochs = 0
    config.training.profile_stages = False
    config.training.gpu_reference_cache_mib = args.gpu_cache_mib
    config.gradient.distortion_loss_weight = .1
    config.topology.enable_growth = config.topology.enable_deletion = False
    config.evaluation.enabled = False
    config.runtime.gpu = args.gpu
    if args.build_dir:
        config.runtime.build_dir = args.build_dir
    dataset = workspace_path(args.dataset, WORKSPACE)
    if not dataset.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset}")
    # Ordinary mkdir preserves workspace ACL inheritance on Windows. mkdtemp's
    # private permissions can make a directory inaccessible to sandbox workers.
    output = (workspace_path(args.output_dir, WORKSPACE) if args.output_dir else
              WORKSPACE / ("benchmark-rgb-steps-" + uuid.uuid4().hex[:12]))
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError(f"Output directory must be new or empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    print(f"Benchmark output: {output}", flush=True)
    report = {
        "status": "running", "dataset": str(dataset), "command": sys.argv,
        "scope": "Fixed initial topology, full images, RGB-only 1 spp, one Adam update per view. "
                 "Throughput includes CPU sampling, rays, hot GPU reference lookup, loss/backward, "
                 "Adam, deferred checks and GPU synchronization. Excludes initialization, PNG decoding, "
                 "reference upload warmup, topology updates, checkpointing and evaluation.",
        "protocol": "Selected cameras repeat in order. Warmup updates are retained. A round is only "
                    "the selected cameras, not a training epoch. Optional diagnostics run afterwards "
                    "and continue updating this disposable model; they never enter throughput medians.",
        "per_view_timing_note": "submit_wall_seconds measures host submission including any implicit "
                                "waits; only the full round includes all deferred GPU work.",
        "warmup_rounds": args.warmup, "timed_rounds": args.repeats, "rounds": [],
        "diagnostics": {},
    }
    state = None
    try:
        mi, dr, np, handles = initialize_runtime(config.runtime, output, report)
        report["provenance"]["source_files_sha256"][Path(__file__).relative_to(REPOSITORY).as_posix()] = file_sha256(Path(__file__))
        prepared = prepare_dataset(config, np)
        all_views = (prepared["views"] if prepared is not None else
                     load_views(dataset, config.data.cameras, config.render, np, data=config.data))
        indices = selected_indices(len(all_views), args)
        views = [all_views[index] for index in indices]
        references = {}
        required_bytes = len(views) * config.render.width * config.render.height * 3 * 4
        if required_bytes > args.gpu_cache_mib * 1024 * 1024:
            raise ValueError(f"Hot-cache benchmark requires {required_bytes / 1024**2:.1f} MiB; "
                             "increase --gpu-cache-mib or reduce --views")
        for view in views:
            reference = read_reference(view["reference_path"], config.render.width, config.render.height,
                                       mi, np, rgb_only=True, background=config.render.background,
                                       reference_compositing=config.data.reference_compositing)
            reference.setflags(write=False)
            references[view["index"]] = reference
        model = initialize_model(config, dataset, mi, np, prepared=prepared)
        del prepared
        cache = GPUReferenceCache(config.render.width, config.render.height, mi, dr, np,
                                  capacity_mib=args.gpu_cache_mib, rgb_only=True)
        state = dict(model=model, optimizer=FalcorOptimizer(model, config.optimizer), cache=cache,
                     replay=NativeLossReplay(dr, enabled=True),
                     sampler=PrefetchedUniformSampler(np, workers=args.sampling_workers),
                     rng=np.random.default_rng(config.training.seed))
        for view in views:
            if cache.get(view["reference_path"], references[view["index"]]) is None:
                raise RuntimeError("Hot-cache benchmark failed to retain a reference")
        dr.eval(*model.snapshot().values())
        dr.sync_thread()
        report.update(config=asdict(config), camera_indices=indices,
                      cameras=[int(view["index"]) for view in views], total_training_views=len(all_views),
                      active_voxels=int(model.active_voxel_count), storage_resolution=int(model.resolution),
                      reference_bytes=required_bytes,
                      reference_sha256={str(view["index"]): file_sha256(Path(view["reference_path"])) for view in views})
        write_report(output, report)
        print(json.dumps({key: report[key] for key in
                          ("cameras", "total_training_views", "active_voxels", "storage_resolution", "reference_bytes")}), flush=True)
        for round_index in range(args.warmup + args.repeats):
            row = execute_round(state, views, references, config, round_index, mi, dr, np)
            row["warmup"] = round_index < args.warmup
            report["rounds"].append(row)
            print(json.dumps({key: row[key] for key in
                              ("round", "warmup", "wall_seconds", "seconds_per_view", "mean_loss")}), flush=True)
            if round_index + 1 == args.warmup:
                report["after_warmup"] = dict(replay=state["replay"].statistics(),
                                               optimizer=state["optimizer"].statistics(),
                                               sampling_execution=state["sampler"].statistics(),
                                               reference_cache=cache.statistics())
        seconds = [row["wall_seconds"] for row in report["rounds"] if not row["warmup"]]
        report["throughput"] = dict(round_median_seconds=statistics.median(seconds),
            round_mean_seconds=statistics.mean(seconds), round_min_seconds=min(seconds),
            round_max_seconds=max(seconds), step_median_ms=statistics.median(seconds) * 1000 / len(views),
            views_per_second=len(views) / statistics.median(seconds), all_round_seconds=seconds,
            replay=state["replay"].statistics(), optimizer=state["optimizer"].statistics(),
            sampling_execution=state["sampler"].statistics(),
            reference_cache=cache.statistics())
        write_report(output, report)
        print("Throughput: " + json.dumps(report["throughput"]), flush=True)
        next_round = args.warmup + args.repeats
        if args.cprofile:
            profiler = cProfile.Profile()
            profiler.enable()
            try:
                row = execute_round(state, views, references, config, next_round, mi, dr, np)
            finally:
                profiler.disable()
            profiler.dump_stats(str(output / "cprofile.prof"))
            stream = io.StringIO()
            stats = pstats.Stats(profiler, stream=stream).strip_dirs()
            stats.sort_stats("cumulative").print_stats(50)
            stats.sort_stats("tottime").print_stats(50)
            top = stream.getvalue()
            (output / "cprofile-top.txt").write_text(top, encoding="utf-8")
            report["diagnostics"]["cprofile"] = dict(round=row, top=top)
            print(top, flush=True)
            next_round += 1
        if args.kernel_history:
            captured = []
            # Instrumentation changes execution flags; discard its first round
            # to allow Dr.Jit frozen functions to record that variant separately.
            for index in range(3):
                with dr.kernel_history() as history:
                    row = execute_round(state, views, references, config, next_round, mi, dr, np)
                cuda = [entry for entry in history if entry.backend == dr.JitBackend.CUDA]
                row.update(kernel_profile(cuda))
                row["operations_by_type"] = {
                    name: sum(entry.type.name == name for entry in cuda)
                    for name in sorted({entry.type.name for entry in cuda})}
                row["warmup"] = index == 0
                captured.append(row)
                next_round += 1
            report["diagnostics"]["cuda_kernel_history"] = dict(rounds=captured,
                median_device_ms=statistics.median(row["device_ms"] for row in captured[1:]),
                median_jit_kernels=statistics.median(row["jit_kernels"] for row in captured[1:]),
                note="Instrumented CUDA operation times, including copies/reductions; not GPU utilization. "
                     "These rounds are excluded from throughput. JIT kernel counts are separate from all operations.")
        if args.native_profile:
            rows = []
            for index in range(2):
                row = execute_round(state, views, references, config, next_round, mi, dr, np, native_profile=True)
                row["warmup"] = index == 0
                rows.append(row)
                next_round += 1
            report["diagnostics"]["native_profile"] = dict(rounds=rows,
                note="Synchronized native phases bypass replay; excluded from throughput. First round is warmup.")
        report["sampling_execution"] = state["sampler"].statistics()
        report["status"] = "complete"
        write_report(output, report)
        print(f"Report: {output / 'benchmark.json'}", flush=True)
    except BaseException as error:
        report.update(status="failed", error=str(error), traceback=traceback.format_exc())
        write_report(output, report)
        raise
    finally:
        if state is not None:
            state["sampler"].close()
            state["replay"].clear()
            state["optimizer"].clear_replay()
            state["cache"].clear()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
