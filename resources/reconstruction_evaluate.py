"""Render saved voxel reconstruction models on train/test cameras and report PSNR.

All relative paths use the source repository's parent directory. CUDA and source
build validation are shared with training; Slurm GPU allocation is preserved.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
import json
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


def run_evaluation(config: EvaluationConfig) -> int:
    """Evaluate a saved model using the same validated runtime as the CLI."""
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
        for split in config.splits:
            cameras_file = config.cameras[split] or saved.get("data", {}).get("cameras", "transforms_train.json")
            views = common.load_views(dataset, cameras_file, render, np)
            if not views:
                raise ValueError(f"Empty camera split: {split}")
            available_view_count = len(views)
            if config.max_views is not None:
                views = views[:config.max_views]
            split_output = output / split
            split_output.mkdir(parents=True, exist_ok=True)
            records = []
            split_report = {
                "cameras": cameras_file, "views": records,
                "max_views": config.max_views, "view_order": "camera file order",
                "available_view_count": available_view_count,
                "selected_view_count": len(views), "evaluated_view_count": 0,
            }
            report["splits"][split] = split_report
            for view in views:
                started = time.perf_counter()
                index = int(view["index"])
                seed = int(np.random.SeedSequence([config.seed, 0 if split == "train" else 1, index])
                           .generate_state(1)[0])
                rgb, alpha = common.render_view(model, render, view["camera"], config.spp,
                                               seed, config.chunk_size, mi, dr, np)
                reference = common.read_reference(view["reference_path"], render.width, render.height, mi, np)
                target = (reference[..., :3] * reference[..., 3:4]
                          + (1 - reference[..., 3:4]) * np.asarray(render.background, dtype=np.float32))
                metrics = image_metrics(rgb, target, np)
                prediction_path = split_output / f"{index:04d}.png"
                preview.write_png(prediction_path, rgb, mi, np)
                if config.save_linear:
                    np.savez_compressed(split_output / f"{index:04d}.npz", rgb_linear=rgb, alpha=alpha)
                record = {"index": index, "reference": str(view["reference_path"]),
                          "prediction": str(prediction_path), "seed": seed,
                          "seconds": time.perf_counter() - started, "metrics": metrics}
                records.append(record)
                split_report["evaluated_view_count"] = len(records)
                score = metrics[config.metric_color_space]["psnr_db"]
                text = "+inf" if score is None else f"{score:.4f}"
                print(f"{split} [{len(records)}/{len(views)}] PSNR ({config.metric_color_space}) = {text} dB", flush=True)
            split_report["aggregate"] = aggregate_metrics(records)
            (split_output / "metrics.json").write_text(
                json.dumps(split_report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
        report["status"] = "passed"
    except Exception:
        report["status"] = "failed"
        report["error"] = traceback.format_exc()
        print(report["error"], file=sys.stderr, flush=True)
    finally:
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        report_path = output / "evaluation-report.json"
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
        print("Report:", report_path, flush=True)
    return 0 if report["status"] == "passed" else 1


def main():
    workspace = Path(__file__).resolve().parents[2]
    return run_evaluation(parse_config(workspace))


if __name__ == "__main__":
    raise SystemExit(main())
