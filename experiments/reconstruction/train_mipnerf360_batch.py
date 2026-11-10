"""Run the remaining Mip-NeRF 360 scenes sequentially using the latest template.

No dataset files or template settings are rewritten. Each scene has its own
configuration snapshot, child process, checkpoint directory and evaluation.
Default queue: room, kitchen, garden, bicycle. Without explicit overrides only
data.dataset changes; train.py resolves images exactly as for an individual run.
Use --dry-run for CPU input checks or --smoke-test for one complete epoch per
scene (same training budget/schedule, no evaluation).
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
import hashlib
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY = SCRIPT_DIR.parents[1]
WORKSPACE = REPOSITORY.parent
TRAIN = SCRIPT_DIR / "train.py"
DEFAULT_CONFIG = SCRIPT_DIR / "configs/bicycle_1024.json"
DEFAULT_SCENES = ("room", "kitchen", "garden", "bicycle")
OUTDOOR = frozenset(("bicycle", "flowers", "garden", "stump", "treehill"))
INDOOR = frozenset(("bonsai", "counter", "kitchen", "room"))
SCENES = OUTDOOR | INDOOR
sys.path.insert(0, str(SCRIPT_DIR))

from reconstruction_experiment import allocate_training_output, EXPERIMENT_TIMEZONE


def discover_scenes(root, requested=None):
    root = Path(root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")
    available = {child.name for child in root.iterdir() if child.is_dir()}
    names = list(requested) if requested is not None else (
        sorted(available & INDOOR) + sorted(available & OUTDOOR))
    if not names:
        raise ValueError(f"No Mip-NeRF 360 scenes found in {root}")
    if len(names) != len(set(names)):
        raise ValueError("Duplicate scene names are not allowed")
    for name in names:
        if name not in SCENES:
            raise ValueError(f"Unknown Mip-NeRF 360 scene: {name}")
        if not (root / name).is_dir():
            raise FileNotFoundError(f"Scene directory does not exist: {root / name}")
    return [root / name for name in names]


def choose_images(dataset, scale="auto"):
    dataset = Path(dataset)
    if scale not in ("auto", "1", "2", "4", "8"):
        raise ValueError("Image scale must be auto, 1, 2, 4 or 8")
    if scale == "auto":
        if dataset.name in OUTDOOR:
            scales = ("4", "8", "2", "1")
        elif dataset.name in INDOOR:
            scales = ("2", "4", "8", "1")
        else:
            raise ValueError(f"No automatic image policy for scene {dataset.name}")
    else:
        scales = (scale,)
    for value in scales:
        name = "images" if value == "1" else "images_" + value
        if (dataset / name).is_dir():
            return name
    raise FileNotFoundError(f"No requested image directory ({scale}) in {dataset}")


def make_scene_config(base, dataset, images=None, gpu=None):
    result = copy.deepcopy(base)
    data = result.setdefault("data", {})
    data["dataset"] = str(Path(dataset).absolute())
    if images is not None:
        data["images"] = images
    # Saved scene normalization/output must not leak into the next scene.
    # The normal template has no initialization and output=null, so its only
    # changed field remains dataset. Do not inject unrelated defaults.
    if "initialization" in data:
        data["initialization"] = None
    if "output" in result.get("training", {}):
        result["training"]["output"] = None
    if gpu is not None:
        result.setdefault("runtime", {})["gpu"] = gpu
    return result


def training_command(config_path, output, gpu, smoke_test=False):
    command = [sys.executable, "-u", "-I", "-X", "utf8", str(TRAIN),
        "--config", str(Path(config_path).absolute()), "--gpu", str(gpu),
        "--output-dir", str(Path(output).absolute())]
    if smoke_test:
        command.extend(("--stop-after-epochs", "1", "--skip-evaluation"))
    return command


def inspect_completion(output, returncode, smoke_test, expected_epochs, evaluation_enabled):
    report_path = Path(output) / "training-report.json"
    result = {"status": "failed", "returncode": returncode, "report": str(report_path)}
    if returncode != 0:
        result["error"] = f"train.py exited with code {returncode}"
        return result
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        completed = report.get("completed_epochs")
        result.update(completed_epochs=completed, training_status=report.get("training_status"),
                      evaluation_status=report.get("evaluation", {}).get("status"))
        if smoke_test:
            expected_status = "stopped" if expected_epochs > 1 else "passed"
            valid = (report.get("status") == expected_status
                and report.get("training_status") == expected_status and completed == 1
                and result["evaluation_status"] == "skipped")
        else:
            valid = (report.get("status") == "passed" and report.get("training_status") == "passed"
                and completed == expected_epochs
                and (not evaluation_enabled or result["evaluation_status"] == "passed"))
        if not valid:
            raise ValueError("Training report does not confirm the requested completed run")
        result["status"] = "smoke_passed" if smoke_test else "passed"
        result["checkpoint"] = report.get("checkpoint")
        result["evaluation_report"] = str(Path(output).parent / "evaluation-report.json") if evaluation_enabled and not smoke_test else None
        if not smoke_test and evaluation_enabled:
            evaluation = json.loads(Path(result["evaluation_report"]).read_text(encoding="utf-8"))
            if evaluation.get("status") != "passed":
                raise ValueError("Evaluation report does not confirm successful evaluation")
            result["mean_psnr_srgb_png_db"] = {split: details.get("aggregate", {}).get("srgb_png", {}).get("mean_psnr_db")
                for split, details in evaluation.get("splits", {}).items()}
    except (OSError, ValueError, TypeError, AttributeError) as error:
        result.update(status="failed", error=str(error))
    return result


def _write_json(path, value):
    path = Path(path)
    pending = path.with_suffix(path.suffix + ".tmp")
    pending.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    pending.replace(path)


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Shared training JSON template")
    parser.add_argument("--datasets-root", type=Path, default=WORKSPACE / "datasets")
    parser.add_argument("--scenes", nargs="+", choices=sorted(SCENES), default=DEFAULT_SCENES,
        help="Default: room kitchen garden bicycle; explicit order is preserved")
    parser.add_argument("--images", choices=("auto", "1", "2", "4", "8"),
        help="Default: preserve template data.images and train.py auto selection; explicit auto prefers outdoor images_4/indoor images_2 with coarser fallback")
    parser.add_argument("--gpu", type=int, help="Override template GPU for every scene")
    parser.add_argument("--epochs", type=int, help="Override template epoch budget for every scene")
    parser.add_argument("--output-dir", type=Path, help="Experiment root; default: workspace outputs, using normal scene-date-Eindex names")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Check cameras, image headers, point paths and configs, without GPU or output files")
    mode.add_argument("--smoke-test", action="store_true", help="One complete epoch per scene, retaining schedule; save checkpoint and skip evaluation")
    args = parser.parse_args(argv)
    if args.gpu is not None and args.gpu < 0:
        parser.error("--gpu must be nonnegative")
    if args.epochs is not None and args.epochs < 1:
        parser.error("--epochs must be positive")
    return args


def prepare_plan(args):
    # Imports here are CPU-only. Never initialize a Mitsuba/Dr.Jit context in
    # the batch parent: process exit must release each scene's CUDA ownership.
    import numpy as np
    import train
    from reconstruction_common import prepare_dataset
    from reconstruction_dataset import detect_dataset

    config_path = args.config.expanduser().resolve()
    template_bytes = config_path.read_bytes()
    base = json.loads(template_bytes.decode("utf-8-sig"))
    if not isinstance(base, dict):
        raise ValueError("The training template must be a JSON object")
    scenes = discover_scenes(args.datasets_root, args.scenes)
    root = (args.output_dir or WORKSPACE / "outputs").expanduser().absolute()
    batch_id = "mipnerf360-batch-" + datetime.now(EXPERIMENT_TIMEZONE).strftime("%Y%m%dT%H%M%S%f")
    resolved = root.resolve()
    protected = (REPOSITORY.resolve(), args.datasets_root.resolve(), *(scene.resolve() for scene in scenes))
    if any(resolved == path or resolved.is_relative_to(path) or path.is_relative_to(resolved) for path in protected):
        raise ValueError("Batch output must be separate from the source repository and datasets")
    if root.exists() and not root.is_dir():
        raise NotADirectoryError(f"Experiment root is not a directory: {root}")
    entries = []
    for dataset in scenes:
        if detect_dataset(dataset) != "colmap":
            raise ValueError(f"Expected a COLMAP Mip-NeRF 360 scene: {dataset}")
        images = choose_images(dataset, args.images) if args.images is not None else None
        values = make_scene_config(base, dataset, images, args.gpu)
        if args.epochs is not None:
            values.setdefault("training", {})["num_epochs"] = args.epochs
        # Save the actual input, not CPU-preflight-derived camera dimensions or
        # scene normalization. The child follows the same path as train.py with
        # this JSON and a manually changed dataset.
        snapshot = copy.deepcopy(values)
        config = copy.deepcopy(train.CONFIG)
        train.apply_arguments(config, train.parse_arguments(["--config", str(config_path)]), values)
        train.validate(config)
        # Only unbounded COLMAP supports point-free preflight. Batch is for
        # the user's world-shell 360 experiments, not synthetic/bounded inputs.
        if config.data.scene_mode not in ("auto", "unbounded"):
            raise ValueError("360 batch template must use data.scene_mode='auto' or 'unbounded'")
        prepared = prepare_dataset(config, np, load_points=False)
        point_source = Path(prepared["point_source"])
        if not point_source.is_file() or point_source.stat().st_size == 0:
            raise FileNotFoundError(f"No usable initial point-cloud file: {point_source}")
        images = prepared["initialization"]["image_directory"]
        entries.append(dict(scene=dataset.name, dataset=str(dataset), images=images,
            width=config.render.width, height=config.render.height, point_source=str(point_source),
            train_views=len(prepared["views"]), test_views=prepared["initialization"]["camera_count"]-len(prepared["views"]),
            epochs=config.training.num_epochs, gpu=config.runtime.gpu, config=snapshot,
            evaluation_enabled=config.evaluation.enabled,
            config_path=None, output=None, log=None, command=None, status="pending"))
    return dict(version=3, status="planned", mode="smoke" if args.smoke_test else "train",
        batch_id=batch_id, manifest=str(root / (batch_id + ".json")),
        template=str(config_path), template_sha256=hashlib.sha256(template_bytes).hexdigest(),
        output=str(root), created_utc=datetime.now(timezone.utc).isoformat(), scenes=entries)


def run_child(command, log_path):
    options = dict(cwd=str(REPOSITORY), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                   text=True, encoding="utf-8", errors="replace", bufsize=1)
    if os.name == "nt":
        options["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    with Path(log_path).open("x", encoding="utf-8") as log:
        process = subprocess.Popen(command, **options)
        try:
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
            return process.wait()
        except BaseException:
            # Do not start another scene while any work of this one survives.
            if process.poll() is None:
                try:
                    process.send_signal(signal.CTRL_BREAK_EVENT if os.name == "nt" else signal.SIGINT)
                    process.wait(timeout=20)
                except (OSError, subprocess.TimeoutExpired):
                    process.kill()
                    process.wait(timeout=20)
            raise
        finally:
            process.stdout.close()


def resolve_scene_output(allocated_output):
    """Locate only this reserved experiment, including train.py's duration suffix."""
    allocated = Path(allocated_output)
    root = allocated.parent.parent.resolve()
    pattern = re.compile(re.escape(allocated.parent.name) + r"(?:-[0-9]+min)?")
    matches = [entry / "training" for entry in root.iterdir()
        if pattern.fullmatch(entry.name) and entry.is_dir() and (entry / "training").is_dir()]
    if not matches:
        raise FileNotFoundError(f"Experiment directory disappeared: {allocated}")
    if len(matches) != 1:
        raise RuntimeError(f"Ambiguous experiment directories for {allocated.parent.name}: {matches}")
    result = matches[0].resolve()
    if result.parent.parent != root or result.name != "training":
        raise ValueError(f"Experiment path escaped its output root: {result}")
    return result


def _allocate_scene(plan, item):
    root = Path(plan["output"])
    output = allocate_training_output(item["dataset"], WORKSPACE, output_root=root)
    snapshot = output / "batch-config.json"
    # Keep an open stdout log outside the experiment until train.py finishes:
    # Windows cannot rename a directory containing this parent's open log.
    log = root / (plan["batch_id"] + "-" + item["scene"] + ".log")
    item.update(allocated_output=str(output), output=str(output), config_path=str(snapshot),
        log=str(log), command=training_command(snapshot, output, item["gpu"], plan["mode"] == "smoke"))
    _write_json(snapshot, item["config"])


def _collect_scene_output(item):
    output = resolve_scene_output(item["allocated_output"])
    item.update(output=str(output), config_path=str(output / "batch-config.json"))
    log = Path(item["log"])
    if log.is_file():
        destination = output.parent / "console.log"
        if destination.exists():
            raise FileExistsError(f"Refusing to replace an existing log: {destination}")
        log.rename(destination)
        item["log"] = str(destination)


def run_batch(plan, runner=run_child):
    root = Path(plan["output"])
    root.mkdir(parents=True, exist_ok=True)
    manifest = Path(plan["manifest"])
    plan["status"] = "running"
    with manifest.open("x", encoding="utf-8") as stream:
        json.dump(plan, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    for index, item in enumerate(plan["scenes"], 1):
        print(f"\n[{index}/{len(plan['scenes'])}] {item['scene']} | {item['images']} | "
              f"{item['width']}x{item['height']} | {item['epochs']} epochs", flush=True)
        item.update(status="running", started_utc=datetime.now(timezone.utc).isoformat())
        _write_json(manifest, plan)
        started = time.perf_counter()
        code = None
        try:
            _allocate_scene(plan, item)
            _write_json(manifest, plan)
            code = runner(item["command"], Path(item["log"]))
        except KeyboardInterrupt:
            item.update(status="interrupted", error="Interrupted by user")
        except Exception as error:
            item.update(status="failed", error=f"{type(error).__name__}: {error}")
        finally:
            if item.get("allocated_output"):
                try:
                    _collect_scene_output(item)
                except Exception as error:
                    item["output_error"] = f"{type(error).__name__}: {error}"
                    if item["status"] == "running":
                        item.update(status="failed", error=item["output_error"])
            item["seconds"] = time.perf_counter() - started
        if item["status"] == "interrupted":
            plan["status"] = "interrupted"
            _write_json(manifest, plan)
            return 130
        if item["status"] == "running":
            item.update(inspect_completion(Path(item["output"]), code, plan["mode"] == "smoke",
                item["epochs"], item.get("evaluation_enabled",
                    item["config"].get("evaluation", {}).get("enabled", True))))
        _write_json(manifest, plan)
        if item["status"] not in ("passed", "smoke_passed"):
            plan["status"] = "failed"
            _write_json(manifest, plan)
            print(f"Batch stopped at {item['scene']}: {item.get('error')}\nLog: {item['log']}", file=sys.stderr)
            return 1
    plan["status"] = "smoke_passed" if plan["mode"] == "smoke" else "passed"
    _write_json(manifest, plan)
    print(f"\nBatch {plan['status']}: {manifest}", flush=True)
    return 0


def main(argv=None):
    try:
        args = parse_arguments(argv)
        plan = prepare_plan(args)
        for item in plan["scenes"]:
            print(f"{item['scene']}: {item['images']} {item['width']}x{item['height']}, "
                  f"train/test={item['train_views']}/{item['test_views']}, "
                  f"epochs={item['epochs']}, points={item['point_source']}", flush=True)
        print(f"Batch output: {plan['output']}", flush=True)
        if args.dry_run:
            print("CPU preflight passed; no GPU work, output directories or training launched.")
            return 0
        return run_batch(plan)
    except KeyboardInterrupt:
        return 130
    except (OSError, ValueError, TypeError, RuntimeError) as error:
        print(f"Batch failed: {error}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
