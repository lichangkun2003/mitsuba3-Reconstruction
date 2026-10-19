"""Render saved voxel reconstruction models on train/test cameras and report PSNR.

All relative paths use the source repository's parent directory. CUDA and source
build validation are shared with training; Slurm GPU allocation is preserved.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
import json
import io
import math
from pathlib import Path
import platform
import sys
import time
import traceback

# Python -I excludes the script directory; only explicitly add our own sources.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import reconstruction_preview as preview
from reconstruction_checkpoint import load_checkpoint
from reconstruction_progress import ProgressLine


@dataclass
class EvaluationConfig:
    checkpoint: str = "outputs/mitsuba/reconstruction-train/model.npz"
    dataset: str | None = None  # None: use the checkpoint's dataset configuration.
    splits: tuple[str, ...] = ("train", "test")
    cameras: dict[str, str | None] = field(default_factory=lambda: {
        "train": None, "test": "transforms_test.json"})
    spp: int = 64  # Independent evaluation sampling; training remains 1 spp.
    max_views: int | None = None  # First N cameras per split; None evaluates all.
    seed: int = 42
    chunk_size: int = 65536
    metric_color_space: str = "linear"
    background: tuple[float, float, float] | None = None  # None: checkpoint setting.
    save_linear: bool = False
    output_directory: str = "outputs/mitsuba/reconstruction-evaluation"
    runtime: preview.RuntimeConfig = field(default_factory=preview.RuntimeConfig)


CONFIG = EvaluationConfig()


def _update_config(config, values):
    allowed = {item.name for item in fields(config)}
    if not isinstance(values, dict) or set(values) - allowed:
        raise ValueError("Unknown or malformed evaluation configuration fields")
    for key, value in values.items():
        if key == "runtime":
            if not isinstance(value, dict) or set(value) - {item.name for item in fields(config.runtime)}:
                raise ValueError("Unknown or malformed runtime configuration fields")
            for name, item in value.items():
                setattr(config.runtime, name, item)
        else:
            setattr(config, key, value)


def parse_config(workspace):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--checkpoint")
    parser.add_argument("--dataset")
    parser.add_argument("--split", choices=("train", "test", "both"))
    parser.add_argument("--train-cameras")
    parser.add_argument("--test-cameras")
    parser.add_argument("--spp", type=int)
    parser.add_argument("--max-views", type=int, help="Evaluate the first N cameras in each selected split")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--chunk-size", type=int)
    parser.add_argument("--metric-color-space", choices=("linear", "srgb"))
    parser.add_argument("--background", nargs=3, type=float, metavar=("R", "G", "B"))
    parser.add_argument("--save-linear", action="store_true", default=None)
    parser.add_argument("--output-dir")
    parser.add_argument("--build-dir")
    parser.add_argument("--gpu", type=int, help="Local nvidia-smi index; ignored in Slurm")
    args = parser.parse_args()
    config = EvaluationConfig()
    _update_config(config, asdict(CONFIG))
    if args.config:
        _update_config(config, json.loads(preview.workspace_path(args.config, workspace).read_text(encoding="utf-8")))
    for name in ("checkpoint", "dataset", "spp", "max_views", "seed", "chunk_size",
                 "metric_color_space", "background", "save_linear"):
        value = getattr(args, name)
        if value is not None:
            setattr(config, name, value)
    if args.split:
        config.splits = ("train", "test") if args.split == "both" else (args.split,)
    for split in ("train", "test"):
        value = getattr(args, split + "_cameras")
        if value is not None:
            config.cameras[split] = value
    if args.output_dir:
        config.output_directory = args.output_dir
    for name in ("gpu", "build_dir"):
        if getattr(args, name) is not None:
            setattr(config.runtime, name, getattr(args, name))
    return config


def validate_config(config):
    for name in ("spp", "chunk_size"):
        value = getattr(config, name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if config.max_views is not None and (
            isinstance(config.max_views, bool) or not isinstance(config.max_views, int)
            or config.max_views < 1):
        raise ValueError("max_views must be None or a positive integer")
    if isinstance(config.seed, bool) or not isinstance(config.seed, int) or config.seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if (not isinstance(config.splits, (tuple, list)) or not config.splits
            or any(value not in ("train", "test") for value in config.splits)
            or len(set(config.splits)) != len(config.splits)):
        raise ValueError("splits must contain train and/or test without duplicates")
    if (not isinstance(config.cameras, dict)
            or any(not isinstance(config.cameras.get(split), str)
                   and not (split == "train" and config.cameras.get(split) is None)
                   for split in config.splits)):
        raise ValueError("Each selected split requires a camera JSON filename")
    if config.metric_color_space not in ("linear", "srgb"):
        raise ValueError("metric_color_space must be linear or srgb")
    if config.background is not None and (
            len(config.background) != 3 or any(not math.isfinite(x) or not 0 <= x <= 1
                                             for x in config.background)):
        raise ValueError("background must contain three finite linear RGB values in [0, 1]")


def linear_to_display_srgb(values, np):
    """The explicit [0,1] display transform used by the sRGB PSNR protocol."""
    linear = np.clip(np.asarray(values, dtype=np.float64), 0.0, 1.0)
    return np.where(linear <= 0.0031308, 12.92 * linear,
                    1.055 * np.power(linear, 1.0 / 2.4) - 0.055)


def image_metrics(prediction, reference, np):
    """Full-image RGB MSE, peak=1; no mask and no PNG quantization."""
    if prediction.shape != reference.shape or prediction.ndim != 3 or prediction.shape[-1] != 3:
        raise ValueError("Prediction and reference must have identical H x W x 3 shapes")
    if not np.isfinite(prediction).all() or not np.isfinite(reference).all():
        raise ValueError("Cannot evaluate images containing nonfinite values")
    result = {}
    for color_space in ("linear", "srgb"):
        if color_space == "linear":
            a, b = prediction.astype(np.float64), reference.astype(np.float64)
        else:
            a, b = linear_to_display_srgb(prediction, np), linear_to_display_srgb(reference, np)
        mse = float(np.mean((a - b) ** 2, dtype=np.float64))
        result[color_space] = {"mse": mse, "psnr_db": None if mse == 0 else -10 * math.log10(mse),
                               "perfect_match": mse == 0}
    return result


def aggregate_metrics(records):
    if not records:
        raise ValueError("Cannot aggregate an empty evaluation split")
    result = {"view_count": len(records)}
    for space in ("linear", "srgb"):
        scores = [record["metrics"][space] for record in records]
        mean_mse = math.fsum(score["mse"] for score in scores) / len(scores)
        has_infinite_psnr = any(score["perfect_match"] for score in scores)
        result[space] = {
            "mean_mse": mean_mse,
            "mean_psnr_db": None if has_infinite_psnr else math.fsum(score["psnr_db"] for score in scores) / len(scores),
            "mean_psnr_is_infinite": has_infinite_psnr,
            "psnr_from_mean_mse_db": None if mean_mse == 0 else -10 * math.log10(mean_mse),
            "all_views_perfect": mean_mse == 0,
        }
    return result


def render_evaluation_record(config, common, model, render, view, split, output,
                             mi, dr, np, view_timing, stage_seconds):
    """Render one whole view with the same seed, metrics and files in either mode."""
    started = time.perf_counter()
    index = int(view["index"])
    seed = int(np.random.SeedSequence([config.seed, 0 if split == "train" else 1, index]).generate_state(1)[0])
    with _wall_stage(view_timing, stage_seconds, "render_seconds"):
        rgb, alpha = common.render_view(model, render, view["camera"], config.spp,
                                       seed, config.chunk_size, mi, dr, np)
    with _wall_stage(view_timing, stage_seconds, "reference_seconds"):
        reference = common.read_reference(view["reference_path"], render.width, render.height, mi, np)
        target = (reference[..., :3] * reference[..., 3:4]
                  + (1 - reference[..., 3:4]) * np.asarray(render.background, dtype=np.float32))
    with _wall_stage(view_timing, stage_seconds, "metrics_seconds"):
        metrics = image_metrics(rgb, target, np)
    prediction_path = output / split / f"{index:04d}.png"
    with _wall_stage(view_timing, stage_seconds, "write_seconds"):
        preview.write_png(prediction_path, rgb, mi, np)
        if config.save_linear:
            np.savez_compressed(output / split / f"{index:04d}.npz", rgb_linear=rgb, alpha=alpha)
    return {"index": index, "reference": str(view["reference_path"]),
            "prediction": str(prediction_path), "seed": seed,
            "seconds": time.perf_counter() - started, "metrics": metrics, **view_timing}


def evaluation_view_owner(task_index, world_size):
    """Assign whole views after applying the global per-split view limit."""
    if (type(task_index) is not int or task_index < 0
            or type(world_size) is not int or world_size < 1):
        raise ValueError("Evaluation task index must be nonnegative and world size positive")
    return task_index % world_size


def merge_evaluation_workers(workers, selected_indices_by_split, metric_color_space="linear"):
    """Validate complete, disjoint camera coverage and aggregate individual views.

    A failed worker may have partial coverage, which stays explicitly failed.
    Stage times are sums of worker work; split seconds are the maximum worker
    duration, not a synchronized measurement of the global split wall time.
    """
    if not workers or sorted(worker["rank"] for worker in workers) != list(range(len(workers))):
        raise ValueError("Evaluation workers must contain each rank exactly once")
    workers = sorted(workers, key=lambda worker: worker["rank"])
    statuses = [worker["status"] for worker in workers]
    if any(status not in ("passed", "failed", "interrupted") for status in statuses):
        raise ValueError("Every evaluation worker must report a terminal status")
    status = ("interrupted" if "interrupted" in statuses else
              "failed" if "failed" in statuses else "passed")
    splits, offset = {}, 0
    for worker in workers:
        if set(worker["splits"]) - set(selected_indices_by_split):
            raise ValueError("Worker reported an unexpected evaluation split")
    for split, indices in selected_indices_by_split.items():
        indices = list(indices)
        if (not indices or any(type(index) is not int or index < 0 for index in indices)
                or len(indices) != len(set(indices))):
            raise ValueError("Selected evaluation indices must be nonempty and unique")
        owners = {index: evaluation_view_owner(offset + position, len(workers))
                  for position, index in enumerate(indices)}
        offset += len(indices)
        records, durations = {}, {}
        stages = dict.fromkeys(("render_seconds", "reference_seconds", "metrics_seconds", "write_seconds"), 0.)
        template = None
        for worker in workers:
            part = worker["splits"].get(split)
            if part is None:
                continue
            if template is None:
                template = part
            durations[str(worker["rank"])] = float(part.get("seconds", 0.))
            for stage in stages:
                stages[stage] += float(part.get("stage_seconds", {}).get(stage, 0.))
            for record in part["views"]:
                index = record["index"]
                if type(index) is not int or index not in owners:
                    raise ValueError(f"Unexpected evaluation view {split}/{index}")
                if owners[index] != worker["rank"]:
                    raise ValueError(f"Evaluation view {split}/{index} was rendered by the wrong rank")
                if index in records:
                    raise ValueError(f"Duplicate evaluation view {split}/{index}")
                records[index] = record
        missing = [index for index in indices if index not in records]
        if status == "passed" and missing:
            raise ValueError(f"Missing evaluation views in {split}: {missing}")
        base = {} if template is None else {
            key: template[key] for key in ("cameras", "max_views", "view_order", "available_view_count")
            if key in template}
        ordered = [records[index] for index in indices if index in records]
        result = dict(base, views=ordered, selected_view_count=len(indices), evaluated_view_count=len(ordered),
                      status="passed" if not missing else status, missing_indices=missing,
                      seconds=max(durations.values(), default=0.), worker_seconds=durations,
                      stage_seconds=stages,
                      timing_scope="seconds=max worker split duration; stage_seconds=sum of worker work, not global wall")
        if ordered:
            result["aggregate"] = aggregate_metrics(ordered)
        splits[split] = result
    if metric_color_space not in ("linear", "srgb"):
        raise ValueError("Unknown evaluation metric color space")
    return {"status": status, "splits": splits}


def _run_distributed_evaluation(config, distributed, progress_callback):
    """Evaluate disjoint whole cameras on two devices without NCCL or image transfer."""
    started = time.perf_counter()
    primary = distributed.rank == 0
    repository = Path(__file__).resolve().parents[1]
    workspace = repository.parent
    output = preview.workspace_path(config.output_directory, workspace)
    if output.is_relative_to(repository):
        raise ValueError("Keep evaluation outputs and caches outside the Git checkout")
    if distributed.world_size != 2 or distributed.rank not in (0, 1):
        raise ValueError("Distributed evaluation requires exactly two ranks")
    report = {
        "status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Source CUDA checkpoint evaluation; no parameter updates",
        "python_executable": sys.executable, "python_version": platform.python_version(),
        "requested_config": asdict(config),
        "resolved": {"workspace": str(workspace), "output": str(output)}, "splits": {},
        "distributed": {"world_size": distributed.world_size,
            "partition": "whole views, global selected train/test task ordinal modulo world_size",
            "sample_scope": "all spp for each image stay on its assigned GPU; no NCCL or image exchange"},
        "timing": {
            "measurement": "wall seconds via perf_counter; not pure GPU execution time",
            "scope": "rank-zero wall through worker merge and metrics writing, excluding final report serialization",
            "render_scope": "render_view including tracing/JIT, transfers and existing device waits/readbacks",
            "worker_stage_scope": "sum of per-worker work; overlaps across GPUs and must not be added to global wall",
            "split_scope": "split_seconds is max worker-local split duration, not synchronized global split wall",
            "total_seconds": 0., "setup_seconds": None, "split_seconds": {},
        },
        "metric_protocol": {
            "primary_color_space": config.metric_color_space,
            "mse": "mean over all image pixels and RGB channels; no mask; peak value 1",
            "linear": "unclamped floating linear RGB after alpha compositing",
            "srgb": "linear RGB clipped to [0,1], IEC sRGB transfer, floating values before PNG quantization",
            "aggregation": "arithmetic mean of all per-view PSNR; PSNR of mean MSE also reported separately; no rank averaging",
            "view_selection": "first max_views cameras per split before sharding; None selects all",
            "perfect_match": "PSNR is +infinity, encoded as null with an explicit boolean flag",
        },
    }
    progress = ProgressLine() if primary else ProgressLine(stream=io.StringIO())
    callback = progress_callback or distributed.heartbeat
    plans, setup_error, setup_status = {}, None, "passed"
    # Setup failures also take part in the same control exchange. A rank must
    # not wait for camera records while its peer is still expecting a setup plan.
    try:
        validate_config(config)
        import reconstruction_common as common
        callback("evaluation setup")
        mi, dr, np, dll_handles = common.initialize_runtime(config.runtime, output, report,
            cuda_device=distributed.rank, cpu_threads=distributed.cpus_per_rank)
        checkpoint_path = preview.workspace_path(config.checkpoint, workspace)
        checkpoint = load_checkpoint(checkpoint_path)
        report["checkpoint"] = {"path": str(checkpoint_path), "sha256": preview.file_sha256(checkpoint_path),
            "model_metadata": checkpoint["model_metadata"],
            "completed_epochs": checkpoint["training_state"].get("completed_epochs"),
            "global_step": checkpoint["training_state"].get("global_step"), "provenance": checkpoint["provenance"]}
        saved = checkpoint["config"]
        render = preview.RenderConfig()
        for key, value in saved.get("render", {}).items():
            if key not in {item.name for item in fields(render)}:
                raise ValueError(f"Unsupported checkpoint render setting: {key}")
            setattr(render, key, value)
        render.spp = config.spp
        if config.background is not None:
            render.background = config.background
        preview.validate_config(preview.PreviewConfig(render=render, runtime=config.runtime))
        dataset_value = config.dataset or saved.get("data", {}).get("dataset")
        if not dataset_value:
            raise ValueError("No dataset in the checkpoint; supply --dataset")
        dataset = preview.workspace_path(dataset_value, workspace)
        report["resolved"]["dataset"] = str(dataset)
        report["render"] = asdict(render)
        report["actual_sampling"] = {
            "layout": "independent uniform subpixel samples per pixel/sample",
            "seeding": "deterministic per split and view", "legacy_preview_pixel_sampling_ignored": render.pixel_sampling}
        report["metric_protocol"]["background_linear_rgb"] = list(render.background)
        model = common.restore_model(checkpoint, mi)
        report["model_statistics"] = model.statistics()
        for split in config.splits:
            cameras_file = config.cameras[split] or saved.get("data", {}).get("cameras", "transforms_train.json")
            views = common.load_views(dataset, cameras_file, render, np)
            plans[split] = {"cameras": cameras_file, "available_view_count": len(views),
                            "views": views[:config.max_views] if config.max_views is not None else views}
            if not plans[split]["views"]:
                raise ValueError(f"Empty camera split: {split}")
    except KeyboardInterrupt:
        setup_status, setup_error = "interrupted", traceback.format_exc()
    except Exception:
        setup_status, setup_error = "failed", traceback.format_exc()
    setup = {"rank": distributed.rank, "status": setup_status, "error": setup_error,
             "config": json.loads(json.dumps(asdict(config))),
             "checkpoint_sha256": report.get("checkpoint", {}).get("sha256"), "plans": plans}
    setups = distributed.gather(setup)
    readiness = None
    if primary:
        try:
            if any(worker["status"] != "passed" for worker in setups):
                readiness = {"status": "interrupted" if any(worker["status"] == "interrupted" for worker in setups) else "failed",
                             "error": "\n".join(f"rank {worker['rank']}: {worker['error']}" for worker in setups if worker["error"])}
            else:
                for worker in setups:
                    for key in ("config", "checkpoint_sha256", "plans"):
                        if worker[key] != setups[0][key]:
                            raise ValueError(f"Evaluation workers disagree on {key}")
                output.mkdir(parents=True, exist_ok=True)
                for split in config.splits:
                    (output / split).mkdir(parents=True, exist_ok=True)
                (output / "evaluation-config.json").write_text(
                    json.dumps(asdict(config), indent=2) + "\n", encoding="utf-8")
                readiness = {"status": "passed", "error": None}
        except Exception:
            readiness = {"status": "failed", "error": traceback.format_exc()}
    readiness = distributed.broadcast(readiness)
    report["timing"]["setup_seconds"] = time.perf_counter() - started
    status, error = readiness["status"], readiness["error"]
    task_offset = 0
    if status == "passed":
        try:
            for split, plan in plans.items():
                split_started = time.perf_counter()
                all_views = plan["views"]
                assigned = [view for ordinal, view in enumerate(all_views)
                            if evaluation_view_owner(task_offset + ordinal, distributed.world_size) == distributed.rank]
                task_offset += len(all_views)
                records = []
                stages = dict.fromkeys(("render_seconds", "reference_seconds", "metrics_seconds", "write_seconds"), 0.)
                split_report = {"cameras": plan["cameras"], "views": records,
                    "max_views": config.max_views, "view_order": "camera file order",
                    "available_view_count": plan["available_view_count"], "selected_view_count": len(all_views),
                    "assigned_view_count": len(assigned), "evaluated_view_count": 0, "status": "running",
                    "seconds": 0., "stage_seconds": stages}
                report["splits"][split] = split_report
                try:
                    for view in assigned:
                        callback(f"evaluation rank {distributed.rank} {split} view {view['index']}")
                        index = int(view["index"])
                        view_timing = dict.fromkeys(stages, 0.)
                        split_report["incomplete_view"] = {"index": index, "stage_seconds": view_timing}
                        record = render_evaluation_record(config, common, model, render, view, split,
                            output, mi, dr, np, view_timing, stages)
                        record["rank"] = distributed.rank
                        records.append(record)
                        split_report.pop("incomplete_view", None)
                        split_report["evaluated_view_count"] = len(records)
                        progress.update(f"Eval rank 0 {split} {len(records)}/{len(assigned)} assigned views | {config.spp} spp")
                    split_report["status"] = "passed"
                except KeyboardInterrupt:
                    split_report["status"] = "interrupted"
                    raise
                except Exception:
                    split_report["status"] = "failed"
                    raise
                finally:
                    split_report["seconds"] = time.perf_counter() - split_started
        except KeyboardInterrupt:
            status, error = "interrupted", traceback.format_exc()
        except Exception:
            status, error = "failed", traceback.format_exc()
    progress.finish()
    callback(f"evaluation rank {distributed.rank} merging {status}")
    local = {"rank": distributed.rank, "status": status, "error": error, "splits": report["splits"],
             "seconds": time.perf_counter() - started, "setup_seconds": report["timing"]["setup_seconds"],
             "runtime": {key: report.get(key) for key in ("gpu", "cpu_threads", "worker_cuda", "python_executable",
                         "mitsuba_version", "drjit_version", "actual_variant", "dlpack_device")}}
    workers = distributed.gather(local)
    outcome = None
    if primary:
        try:
            report["distributed"]["workers"] = [{key: worker[key] for key in
                ("rank", "status", "error", "seconds", "setup_seconds", "runtime")} for worker in workers]
            if readiness["status"] == "passed":
                selected = {split: [int(view["index"]) for view in plan["views"]] for split, plan in plans.items()}
                report.update(merge_evaluation_workers(workers, selected, config.metric_color_space))
                for split, result in report["splits"].items():
                    (output / split / "metrics.json").write_text(
                        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
                    report["timing"]["split_seconds"][split] = result["seconds"]
                    if result.get("aggregate"):
                        score = result["aggregate"][config.metric_color_space]["mean_psnr_db"]
                        text = "+inf" if score is None else f"{score:.4f}"
                        print(f"Eval {split} {result['evaluated_view_count']}/{result['selected_view_count']} "
                              f"| {config.spp} spp | mean PSNR({config.metric_color_space})={text} dB", flush=True)
            else:
                report.update(status=readiness["status"], error=readiness["error"])
            errors = [f"rank {worker['rank']}: {worker['error']}" for worker in workers if worker["error"]]
            if errors:
                report["error"] = "\n".join(errors)
        except Exception:
            report.update(status="failed", error=traceback.format_exc())
        report["timing"]["total_seconds"] = time.perf_counter() - started
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        try:
            output.mkdir(parents=True, exist_ok=True)
            report_path = output / "evaluation-report.json"
            report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
            print("Report:", report_path, flush=True)
        except Exception:
            report.update(status="failed", error=traceback.format_exc())
        if report["status"] != "passed":
            print(report.get("error", "Distributed evaluation failed"), file=sys.stderr, flush=True)
        outcome = report["status"]
    outcome = distributed.broadcast(outcome)
    if outcome == "interrupted":
        raise KeyboardInterrupt
    return 0 if outcome == "passed" else 1


@contextmanager
def _wall_stage(view_timing, split_timing, stage):
    """Measure elapsed host time, including any existing device waits/readbacks."""
    started = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - started
        view_timing[stage] += elapsed
        split_timing[stage] += elapsed


def run_evaluation(config: EvaluationConfig, *, progress_callback=None, distributed=None) -> int:
    """Evaluate a saved model using the same validated runtime as the CLI."""
    if distributed is not None:
        return _run_distributed_evaluation(config, distributed, progress_callback)
    evaluation_started = time.perf_counter()
    repository = Path(__file__).resolve().parents[1]
    workspace = repository.parent
    output = preview.workspace_path(config.output_directory, workspace)
    if output.is_relative_to(repository):
        raise ValueError("Keep evaluation outputs and caches outside the Git checkout")
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Source CUDA checkpoint evaluation; no parameter updates",
        "python_executable": sys.executable, "python_version": platform.python_version(),
        "requested_config": asdict(config),
        "resolved": {"workspace": str(workspace), "output": str(output)},
        "splits": {},
        "timing": {
            "measurement": "wall seconds via perf_counter; not pure GPU execution time; no additional GPU synchronization",
            "scope": "total through finalization, excluding evaluation-report.json serialization; split stages include partial failed/interrupted views",
            "render_scope": "render_view including tracing/JIT, transfers and its existing device waits/readbacks",
            "total_seconds": 0.0, "setup_seconds": None, "split_seconds": {},
        },
        "metric_protocol": {
            "primary_color_space": config.metric_color_space,
            "mse": "mean over all image pixels and RGB channels; no mask; peak value 1",
            "linear": "unclamped floating linear RGB after alpha compositing",
            "srgb": "linear RGB clipped to [0,1], IEC sRGB transfer, floating values before PNG quantization",
            "aggregation": "arithmetic mean of per-view PSNR; PSNR of mean MSE also reported separately",
            "view_selection": "first max_views cameras per split in camera file order; None selects all",
            "perfect_match": "PSNR is +infinity, encoded as null with an explicit boolean flag",
        },
    }
    progress = ProgressLine()
    try:
        validate_config(config)
        import reconstruction_common as common
        mi, dr, np, dll_handles = common.initialize_runtime(config.runtime, output, report)
        checkpoint_path = preview.workspace_path(config.checkpoint, workspace)
        checkpoint = load_checkpoint(checkpoint_path)
        report["checkpoint"] = {"path": str(checkpoint_path), "sha256": preview.file_sha256(checkpoint_path),
                                 "model_metadata": checkpoint["model_metadata"],
                                 "completed_epochs": checkpoint["training_state"].get("completed_epochs"),
                                 "global_step": checkpoint["training_state"].get("global_step"),
                                 "provenance": checkpoint["provenance"]}
        saved = checkpoint["config"]
        render = preview.RenderConfig()
        for key, value in saved.get("render", {}).items():
            if key not in {item.name for item in fields(render)}:
                raise ValueError(f"Unsupported checkpoint render setting: {key}")
            setattr(render, key, value)
        render.spp = config.spp
        if config.background is not None:
            render.background = config.background
        # pixel_sampling is a legacy preview field. Evaluation always uses the
        # independent uniform sampler in render_view, recorded separately below.
        validation_config = preview.PreviewConfig(render=render, runtime=config.runtime)
        preview.validate_config(validation_config)
        dataset_value = config.dataset or saved.get("data", {}).get("dataset")
        if not dataset_value:
            raise ValueError("No dataset in the checkpoint; supply --dataset")
        dataset = preview.workspace_path(dataset_value, workspace)
        report["resolved"]["dataset"] = str(dataset)
        report["render"] = asdict(render)
        report["actual_sampling"] = {
            "layout": "independent uniform subpixel samples per pixel/sample",
            "seeding": "deterministic per split and view",
            "legacy_preview_pixel_sampling_ignored": render.pixel_sampling,
        }
        report["metric_protocol"]["background_linear_rgb"] = list(render.background)
        model = common.restore_model(checkpoint, mi)
        report["model_statistics"] = model.statistics()
        (output / "evaluation-config.json").write_text(json.dumps(asdict(config), indent=2) + "\n", encoding="utf-8")
        report["timing"]["setup_seconds"] = time.perf_counter() - evaluation_started
        for split in config.splits:
            split_started = time.perf_counter()
            cameras_file = config.cameras[split] or saved.get("data", {}).get("cameras", "transforms_train.json")
            split_output = output / split
            records = []
            stage_seconds = dict.fromkeys(
                ("render_seconds", "reference_seconds", "metrics_seconds", "write_seconds"), 0.0)
            split_report = {
                "cameras": cameras_file, "views": records,
                "max_views": config.max_views, "view_order": "camera file order",
                "evaluated_view_count": 0, "status": "running",
                "seconds": 0.0, "stage_seconds": stage_seconds,
            }
            report["splits"][split] = split_report
            try:
                views = common.load_views(dataset, cameras_file, render, np)
                if not views:
                    raise ValueError(f"Empty camera split: {split}")
                split_report["available_view_count"] = len(views)
                if config.max_views is not None:
                    views = views[:config.max_views]
                split_report["selected_view_count"] = len(views)
                split_output.mkdir(parents=True, exist_ok=True)
                progress.update(f"Eval {split} 0/{len(views)} (0.0%) | {config.spp} spp", force=True)
                for view in views:
                    if progress_callback is not None:
                        progress_callback(f"evaluation {split} view {view['index']}")
                    index = int(view["index"])
                    view_timing = dict.fromkeys(stage_seconds, 0.0)
                    split_report["incomplete_view"] = {"index": index, "stage_seconds": view_timing}
                    record = render_evaluation_record(config, common, model, render, view, split,
                        output, mi, dr, np, view_timing, stage_seconds)
                    records.append(record)
                    del split_report["incomplete_view"]
                    split_report["evaluated_view_count"] = len(records)
                    score = record["metrics"][config.metric_color_space]["psnr_db"]
                    text = "+inf" if score is None else f"{score:.4f}"
                    progress.update(
                        f"Eval {split} {len(records)}/{len(views)} "
                        f"({100 * len(records) / len(views):.1f}%) | {config.spp} spp | "
                        f"PSNR({config.metric_color_space})={text} dB")
                split_report["aggregate"] = aggregate_metrics(records)
                split_report["status"] = "passed"
            except KeyboardInterrupt:
                split_report["status"] = "interrupted"
                raise
            except Exception:
                split_report["status"] = "failed"
                raise
            finally:
                split_report["seconds"] = time.perf_counter() - split_started
                report["timing"]["split_seconds"][split] = split_report["seconds"]
            (split_output / "metrics.json").write_text(
                json.dumps(split_report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
            mean_score = split_report["aggregate"][config.metric_color_space]["mean_psnr_db"]
            mean_text = "+inf" if mean_score is None else f"{mean_score:.4f}"
            progress.finish(
                f"Eval {split} {len(records)}/{len(views)} (100%) | {config.spp} spp | "
                f"mean PSNR({config.metric_color_space})={mean_text} dB | "
                f"{time.perf_counter() - split_started:.1f}s")
        report["status"] = "passed"
    except KeyboardInterrupt:
        progress.finish()
        report["status"] = "interrupted"
        raise
    except Exception:
        progress.finish()
        report["status"] = "failed"
        report["error"] = traceback.format_exc()
        print(report["error"], file=sys.stderr, flush=True)
    finally:
        progress.finish()
        report["timing"]["total_seconds"] = time.perf_counter() - evaluation_started
        if report["timing"]["setup_seconds"] is None:
            report["timing"]["setup_seconds"] = report["timing"]["total_seconds"]
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        report_path = output / "evaluation-report.json"
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
        print("Report:", report_path, flush=True)
    return 0 if report["status"] == "passed" else 1


def main(distributed=None):
    workspace = Path(__file__).resolve().parents[2]
    return run_evaluation(parse_config(workspace), distributed=distributed)


if __name__ == "__main__":
    raise SystemExit(main())
