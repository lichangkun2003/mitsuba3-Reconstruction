"""Train the Falcor NoLightTransport voxel model: every view, 1 spp, one update.

All relative paths use the parent of this checkout. Edit CONFIG, provide a JSON
configuration, or override the main options on the command line.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field, fields, replace
from datetime import datetime, timezone
import gc
import json
import math
from pathlib import Path
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parent))

from reconstruction_preview import DataConfig, ModelConfig, RenderConfig, RuntimeConfig, PreviewConfig, validate_config
from reconstruction_common import (REPOSITORY, WORKSPACE, ContourSampler, ReferenceImageCache,
    generate_ray_batch, prepare_camera, initialize_runtime, load_views, model_metadata,
    restore_model, workspace_path, file_sha256)
from reconstruction_checkpoint import load_checkpoint, save_checkpoint
from reconstruction_experiment import allocate_training_output, resumed_training_output
from reconstruction_optimizer import FalcorOptimizer, OptimizerConfig
from reconstruction_performance import PerformanceRecorder
from reconstruction_progress import ProgressLine
from reconstruction_topology import TopologyManager, TopologySettings, CoarseToFineConfig


@dataclass
class TrainingSettings:
    num_epochs: int = 1000
    chunk_size: int = 0  # 0: submit the whole image; positive: explicit ray batches.
    profile_views: int = 0  # Record JIT/cache/device timings for the first N views.
    profile_interval_epochs: int = 50  # Sample one view periodically and on entering each resolution; 0 disables.
    reference_cache_mib: int = 1024  # Decoded float32 RGBA in host RAM; 0 disables caching.
    seed: int = 42
    shuffle: bool = True
    save_count: int = 3  # Total snapshots, including the final model.npz.
    output: str | None = None  # None: automatically allocate a daily experiment.


@dataclass
class GradientSettings:
    alpha_loss_weight: float = .5
    geometry_tau_voxel_fraction: float = .12
    geometry_grad_clamp: float = 5
    alpha_geometry_weight: float = .1
    background_carve_weight: float = .01
    geometry_two_sided_boundary: bool = False
    geometry_boundary_band_tau: float = 1
    geometry_transition_only: bool = True
    geometry_hit_expansion_band_tau: float = 0
    min_evidence_transmittance: float = .05
    min_removal_loss_delta: float = 1e-4
    max_candidates: int = 16


@dataclass
class ContourSettings:
    enabled: bool = False
    uniform_fraction: float = .5
    history_rate: float = .3


@dataclass
class EvaluationSettings:
    enabled: bool = True
    spp: int = 64
    max_views: int | None = 10  # First N cameras of each split; None: all views.
    chunk_size: int = 65536
    seed: int = 42
    test_cameras: str = "transforms_test.json"


@dataclass
class TrainConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    render: RenderConfig = field(default_factory=lambda: RenderConfig(background=(1., 1., 1.)))
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    training: TrainingSettings = field(default_factory=TrainingSettings)
    gradient: GradientSettings = field(default_factory=GradientSettings)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    topology: TopologySettings = field(default_factory=TopologySettings)
    schedule: CoarseToFineConfig = field(default_factory=CoarseToFineConfig)
    contour: ContourSettings = field(default_factory=ContourSettings)
    evaluation: EvaluationSettings = field(default_factory=EvaluationSettings)


# ==================== 训练实验配置：常用参数在这里修改 ====================
# 路径相对于 mitsuba 仓库的父目录；切换场景只需修改下面的 dataset。
CONFIG = TrainConfig()
CONFIG.data.dataset = "datasets/lego"
CONFIG.data.point_cloud = "init_points.ply"
CONFIG.data.cameras = "transforms_train.json"

CONFIG.render.width = 800
CONFIG.render.height = 800
CONFIG.render.spp = 1  # 训练固定每视角 1 spp；多 spp 出图使用评价脚本。
CONFIG.training.chunk_size = 0  # 0 = 整图提交；显存不足时可显式改为 65536 等。
CONFIG.training.profile_views = 0  # 设为 3 可记录前三个视角的编译、缓存和 GPU 耗时。
CONFIG.training.profile_interval_epochs = 50  # 每 50 轮及首次进入各分辨率时抽样记录内核耗时；0 关闭。
CONFIG.training.reference_cache_mib = 1024  # 内存缓存参考图，避免每轮重新解码；0 关闭。

CONFIG.schedule.enabled = True
CONFIG.schedule.start_resolution = 256
CONFIG.schedule.target_resolution = 512
CONFIG.training.num_epochs = 1000  # 整个实验总轮数，每轮遍历全部训练视角。
CONFIG.training.save_count = 3  # 包含最终 model.npz，共保存三次。
CONFIG.training.output = None  # 自动创建 outputs/场景-MM.DD-E编号/training。
# 需要手动指定目录时，将 None 改为路径字符串，或传 --output-dir。
CONFIG.evaluation.enabled = True  # 最终 model.npz 保存后，自动评价训练集和测试集。
CONFIG.evaluation.spp = 64  # 仅用于评价出图，不改变训练的 1 spp。
CONFIG.evaluation.max_views = 10  # 各取相机文件中的前 10 个视角；None 表示全部。
# 高级参数使用 CONFIG.optimizer / gradient / topology / contour / runtime。
# ============================ 配置区结束 ============================


def update(config, values):
    if not isinstance(values, dict):
        raise ValueError("Config must be a dictionary of parameter groups")
    groups = {f.name for f in fields(config)}
    if set(values) - groups:
        raise ValueError(f"Unknown configuration groups: {set(values) - groups}")
    for name, entries in values.items():
        if name == "training" and isinstance(entries, dict):
            # Old configs/checkpoints remain loadable; interval-based saving
            # no longer controls the number of reconstruction snapshots.
            entries = {key: value for key, value in entries.items() if key != "save_every_epochs"}
        group = getattr(config, name)
        allowed = {f.name for f in fields(group)}
        if not isinstance(entries, dict) or set(entries) - allowed:
            raise ValueError(f"Unknown or malformed configuration fields in {name}")
        for key, value in entries.items():
            setattr(group, key, value)


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--resume", help="Resume a completed-epoch checkpoint")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--dataset")
    parser.add_argument("--output-dir", help="Override automatic outputs/SCENE-MM.DD-Eindex/training; on resume the default reuses the checkpoint's experiment")
    parser.add_argument("--build-dir")
    parser.add_argument("--gpu", type=int, help="Local physical index; ignored under Slurm")
    parser.add_argument("--chunk-size", type=int, help="0 (default): full image; positive: maximum rays per batch")
    parser.add_argument("--profile-views", type=int, help="Record kernel/cache timings for the first N processed views")
    parser.add_argument("--profile-interval-epochs", type=int, help="Sample one view every N epochs and on entering a resolution (default: 50); 0 disables automatic samples")
    parser.add_argument("--reference-cache-mib", type=int, help="RAM budget for decoded reference images in MiB (default: 1024); 0 disables caching")
    parser.add_argument("--skip-evaluation", action="store_true", help="Save the final model without automatic train/test evaluation")
    parser.add_argument("--eval-spp", type=int, help="Samples per pixel for final evaluation (default: 64)")
    parser.add_argument("--eval-max-views", type=int, help="Evaluate the first N cameras of each split (default: 10)")
    return parser.parse_args()


def apply_arguments(config, args, values):
    update(config, values)
    for value, group, key in ((args.epochs, "training", "num_epochs"),
        (args.dataset, "data", "dataset"), (args.output_dir, "training", "output"),
        (args.build_dir, "runtime", "build_dir"), (args.gpu, "runtime", "gpu"),
        (args.chunk_size, "training", "chunk_size"),
        (args.profile_views, "training", "profile_views"),
        (args.profile_interval_epochs, "training", "profile_interval_epochs"),
        (args.reference_cache_mib, "training", "reference_cache_mib"),
        (args.eval_spp, "evaluation", "spp"),
        (args.eval_max_views, "evaluation", "max_views")):
        if value is not None:
            setattr(getattr(config, group), key, value)
    if args.skip_evaluation:
        config.evaluation.enabled = False
    config.schedule.total_epochs = config.training.num_epochs


def validate(config):
    validate_config(PreviewConfig(data=config.data, model=config.model, render=config.render, runtime=config.runtime))
    if config.render.spp != 1:
        raise ValueError("Training requires exactly 1 spp per view; use reconstruction_evaluate.py for multi-spp images")
    if list(config.render.background) != [1., 1., 1.]:
        raise ValueError("Falcor's training reference is composited onto white; training requires background [1,1,1]. Evaluation background is configurable.")
    for name in ("num_epochs", "save_count"):
        value = getattr(config.training, name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"training.{name} must be a positive integer")
    for name in ("chunk_size", "profile_views", "profile_interval_epochs", "reference_cache_mib"):
        value = getattr(config.training, name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"training.{name} must be a nonnegative integer")
    if config.training.seed < 0:
        raise ValueError("training.seed must be nonnegative")
    for name, value in asdict(config.gradient).items():
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"Invalid gradient.{name}: {value}")
    if config.gradient.geometry_tau_voxel_fraction <= 0:
        raise ValueError("geometry_tau_voxel_fraction must be positive")
    if not isinstance(config.evaluation.enabled, bool):
        raise ValueError("evaluation.enabled must be a boolean")
    from reconstruction_evaluate import validate_config as validate_evaluation
    validate_evaluation(final_evaluation_config(config, Path("model.npz")))


def final_evaluation_config(config, model_path):
    """Use the saved model, current runtime and experiment directory for evaluation."""
    from reconstruction_evaluate import EvaluationConfig
    return EvaluationConfig(checkpoint=str(model_path), dataset=config.data.dataset,
        splits=("train", "test"), cameras={"train": config.data.cameras, "test": config.evaluation.test_cameras},
        spp=config.evaluation.spp, max_views=config.evaluation.max_views,
        seed=config.evaluation.seed, chunk_size=config.evaluation.chunk_size,
        output_directory=str(model_path.parent), runtime=config.runtime)


def settings(config):
    result = asdict(config.gradient)
    result.update(foreground_alpha_min=config.topology.foreground_alpha_min,
        background_alpha_max=config.topology.background_alpha_max,
        collect_deletion_evidence=config.topology.enable_deletion,
        transmittance_threshold=config.render.transmittance_threshold,
        max_hits=config.render.max_hits, entry_offset_fraction=config.render.entry_offset_fraction)
    return result


def effective_chunk_size(config):
    count = config.render.width * config.render.height
    return min(config.training.chunk_size or count, count)


def train_view(model, view, rgba, config, sampler, rng, mi, dr, np, *, need_rendered_alpha=False):
    """Compute a full view on the GPU; only transfer data needed by CPU consumers.

    Gradients stay on the GPU. Loss/validity are read once per view. Optional
    contour and growth alpha are assembled on the GPU before their one readback.
    Positive chunk sizes retain bounded intermediate storage and one Adam update.
    """
    count = config.render.width * config.render.height
    chunk_size = effective_chunk_size(config)
    offsets, inverse_pdf, token = sampler.sample(view["index"], rgba[..., 3], rng)
    reference = rgba.reshape(-1, 4)
    raw_alpha = dr.empty(mi.Float, count) if token is not None else None
    estimated_alpha = dr.empty(mi.Float, count) if need_rendered_alpha else None
    accumulator = {"gradients": {}}
    total_loss, finite = mi.Float(0), mi.Bool(True)
    camera = prepare_camera(config.render, view["camera"], mi, dr)
    for begin in range(0, count, chunk_size):
        end = min(begin + chunk_size, count)
        ray = generate_ray_batch(config.render, view["camera"],
            np.arange(begin, end, dtype=np.uint32), offsets[begin:end], mi, dr, prepared_camera=camera)
        ref = reference[begin:end]
        result = model.loss_and_grad(ray, mi.Color3f(mi.Float(ref[:, 0]), mi.Float(ref[:, 1]), mi.Float(ref[:, 2])),
            mi.Float(ref[:, 3]), mi.Color3f(config.render.background), mi.Float(inverse_pdf[begin:end]), settings(config))
        total_loss += dr.sum(result["loss"])
        finite &= dr.all(dr.isfinite(result["raw_alpha"])) & dr.all(dr.isfinite(result["alpha"]))
        for name, target in (("raw_alpha", raw_alpha), ("alpha", estimated_alpha)):
            if target is not None:
                # An opaque offset avoids a different scatter kernel per batch.
                indices = dr.arange(mi.UInt32, end - begin) + dr.opaque(mi.UInt32, begin)
                dr.scatter(target, result[name], indices)
        for key, value in result["gradients"].items():
            accumulator["gradients"][key] = accumulator["gradients"].get(key, 0) + value
        for key in ("appearance_valid", "geometry_valid", "background_carve_valid"):
            accumulator[key] = accumulator.get(key, 0) + result[key]
        for key in ("foreground_evidence", "background_evidence"):
            accumulator[key] = dr.maximum(accumulator.get(key, 0), result[key])
        # Materialize per-chunk sums; never retain all image path records/graphs.
        dr.eval(total_loss, finite, *[x for x in (raw_alpha, estimated_alpha) if x is not None],
                *accumulator["gradients"].values(),
                *[value for key, value in accumulator.items() if key != "gradients"])
    for gradient in accumulator["gradients"].values():
        finite &= dr.all(dr.isfinite(gradient))
    finite &= dr.isfinite(total_loss)
    # Read loss and the combined validity flag together, rather than performing
    # a separate host synchronization for every parameter group and ray batch.
    summary = dr.zeros(mi.Float, 2)
    dr.scatter(summary, total_loss / count, 0)
    dr.scatter(summary, mi.Float(finite), 1)
    dr.eval(summary)
    host_summary = np.asarray(summary)
    if not bool(host_summary[1]) or not math.isfinite(float(host_summary[0])):
        raise RuntimeError(f"Nonfinite loss/alpha/gradient at training view {view['index']}")
    if raw_alpha is not None:
        sampler.observe(token, np.asarray(raw_alpha))
    alpha = (np.asarray(estimated_alpha).reshape(config.render.height, config.render.width)
             if estimated_alpha is not None else None)
    return accumulator, float(host_summary[0]), alpha


def kernel_profile(history):
    """Separate JIT compilation from device work; all Dr.Jit times are ms."""
    entries = list(history)
    jit = [entry for entry in entries if entry.type.name == "JIT"]
    return {"operations": len(entries), "jit_kernels": len(jit),
            "cache_hits": sum(bool(entry.cache_hit) for entry in jit),
            "cache_misses": sum(not entry.cache_hit for entry in jit),
            "codegen_ms": sum(entry.codegen_time for entry in entries),
            "backend_compile_ms": sum(entry.backend_time for entry in entries),
            "device_ms": sum(entry.execution_time for entry in entries),
            "jit_hashes": sorted({str(entry.hash) for entry in jit})}


def dataset_signature(views, dataset, camera_file):
    return {"cameras": file_sha256(dataset / camera_file),
            "references": [file_sha256(Path(view["reference_path"])) for view in views]}


def checkpoint_epochs(num_epochs: int, save_count: int) -> tuple[int, ...]:
    """Evenly space completed-epoch saves; the final model counts as one save."""
    for value in (num_epochs, save_count):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError("Epoch and checkpoint counts must be positive integers")
    count = min(num_epochs, save_count)
    return tuple((index * num_epochs + count - 1) // count for index in range(1, count + 1))


def final_model_path(output: Path) -> Path:
    """Keep the finished reconstruction beside train/test evaluation results."""
    experiment = output.parent if output.name == "training" else output
    return experiment / "model.npz"


def prepare_training_output(config, resume_path=None, checkpoint_config=None):
    """Resolve and record one output directory before initializing the GPU."""
    requested = config.training.output
    if requested is None:
        if resume_path is not None:
            output = resumed_training_output(resume_path, checkpoint_config or {}).resolve()
        else:
            if workspace_path("outputs", WORKSPACE).is_relative_to(REPOSITORY):
                raise ValueError("Keep checkpoints, logs and caches outside the Git checkout")
            output = allocate_training_output(config.data.dataset, WORKSPACE)
    elif not isinstance(requested, str) or not requested.strip():
        raise ValueError("training.output must be null/None for automatic naming or a nonempty path")
    else:
        output = workspace_path(requested, WORKSPACE)
    model_path = final_model_path(output)
    if output.is_relative_to(REPOSITORY) or model_path.is_relative_to(REPOSITORY):
        raise ValueError("Keep checkpoints, logs and caches outside the Git checkout")
    if resume_path is None and (any(path.exists() for path in (
            output / "latest.npz", output / "training.jsonl", model_path))
            or any(output.glob("epoch-*.npz"))):
        raise ValueError("Training output already contains a run; choose --output-dir or resume its checkpoint")
    output.mkdir(parents=True, exist_ok=True)
    # Save a concrete, portable path, never the automatic naming sentinel.
    config.training.output = (output.relative_to(WORKSPACE).as_posix()
                              if output.is_relative_to(WORKSPACE) else str(output))
    return output


def validate_existing_model(path: Path, resumed_checkpoint: dict) -> None:
    """A resume must not replace another experiment's exported reconstruction."""
    if not path.exists():
        return
    existing = load_checkpoint(path)
    previous = resumed_checkpoint["config"]
    for group in ("data", "model", "render", "gradient", "optimizer", "topology", "schedule", "contour"):
        if existing["config"].get(group) != previous.get(group):
            raise ValueError(f"Exported model belongs to a different {group} configuration: {path}; choose a new experiment directory")
    for key in ("num_epochs", "seed", "shuffle"):
        if existing["config"].get("training", {}).get(key) != previous.get("training", {}).get(key):
            raise ValueError(f"Exported model has different training.{key}: {path}; choose a new experiment directory")
    signature = resumed_checkpoint["training_state"].get("dataset_signature")
    if signature is None or existing["training_state"].get("dataset_signature") != signature:
        raise ValueError(f"Exported model belongs to a different training dataset: {path}; choose a new experiment directory")


def run_training():
    """Finish and save training before releasing its GPU state for evaluation."""
    run_started = time.perf_counter()
    args = parse_arguments()
    overrides = json.loads(workspace_path(args.config, WORKSPACE).read_text(encoding="utf-8")) if args.config else {}
    config = TrainConfig()
    update(config, asdict(CONFIG))
    apply_arguments(config, args, overrides)
    resume_path = workspace_path(args.resume, WORKSPACE) if args.resume else None
    checkpoint = load_checkpoint(resume_path) if resume_path is not None else None
    output = prepare_training_output(config, resume_path, checkpoint["config"] if checkpoint else None)
    model_path = final_model_path(output)
    print("Training output:", output, flush=True)
    print("Final reconstructed model:", model_path, flush=True)
    report = {"status": "running", "training_status": "running", "evaluation": {"status": "not_started"},
              "started_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "Falcor NoLightTransport 1-spp full-view training; explicit geometry surrogate gradients",
              "falcor_reference_commit": "398f634ec788bbb2e86a6f4d0ab78d41158d9905",
              "training_protocol": {"spp": 1, "reference_background": [1, 1, 1],
                  "gradient_reduction": "sum over pixels, no hit-count normalization",
                  "reported_loss": "mean over pixels, then mean over training views",
                  "actual_sampling": "per-pixel uniform, optionally contour-guided; frozen for a view"}}
    progress_line = ProgressLine()
    reference_cache = None
    history_name = ("training-resume-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f") + ".jsonl"
                    if checkpoint else "training.jsonl")
    performance_name = ("performance-summary-" + Path(history_name).stem.removeprefix("training-") + ".json"
                        if checkpoint else "performance-summary.json")
    performance = PerformanceRecorder(output / performance_name,
        {"history_file": str(output / history_name), "resumed_from": str(resume_path) if resume_path else None},
        started_at=run_started)
    report["performance_file"] = str(output / performance_name)
    training_loop_started = None
    print("Performance summary:", report["performance_file"], flush=True)
    try:
        validate(config)
        mi, dr, np, handles = initialize_runtime(config.runtime, output, report)
        performance.metadata.update({key: report.get(key) for key in (
            "gpu", "cpu_threads", "python_executable", "mitsuba_version", "drjit_version", "actual_variant",
            "mitsuba_file", "drjit_file", "numpy_version", "provenance")})
        if checkpoint:
            validate_existing_model(model_path, checkpoint)
            # Runtime and output are host-specific; all algorithm settings resume from the checkpoint.
            runtime, training_output = config.runtime, config.training.output
            update(config, checkpoint["config"])
            config.runtime, config.training.output = runtime, training_output
            apply_arguments(config, args, overrides)
            # JSON may request automatic naming with null; keep the already
            # selected location instead of restoring null or a stale host path.
            config.training.output = training_output
            validate(config)
            previous = checkpoint["config"]
            current = asdict(config)
            for group in ("model", "render", "gradient", "optimizer", "topology", "contour"):
                if current[group] != previous[group]:
                    raise ValueError(f"Cannot change {group} while resuming; start a new run for a different algorithm")
            for key in ("seed", "shuffle", "save_count"):
                previous_value = previous["training"].get(key, 3) if key == "save_count" else previous["training"][key]
                if current["training"][key] != previous_value:
                    raise ValueError(f"Cannot change training.{key} while resuming")
            # Changing the total budget changes the coarse-to-fine schedule retroactively.
            if current["schedule"] != previous["schedule"]:
                raise ValueError("Resume must preserve its original resolution schedule and epoch budget")
            existing_config = output / "config.json"
            if existing_config.exists():
                existing = json.loads(existing_config.read_text(encoding="utf-8"))
                for group in ("data", "model", "render", "gradient", "optimizer", "topology", "schedule", "contour"):
                    if existing.get(group) != previous[group]:
                        raise ValueError("Resume output belongs to a different configuration; choose a new --output-dir")
        report["requested_config"] = asdict(config)
        performance.metadata["config"] = asdict(config)
        batch_size = effective_chunk_size(config)
        ray_count = config.render.width * config.render.height
        report["training_protocol"].update(rays_per_view=ray_count, rays_per_batch=batch_size,
            batches_per_view=(ray_count + batch_size - 1) // batch_size,
            alpha_readback="only contour history and scheduled growth evidence; no visualization readback",
            camera_parameters="opaque runtime inputs")
        print(f"Training rays: {ray_count} per view, {batch_size} per batch, "
              f"{report['training_protocol']['batches_per_view']} batch(es) per update", flush=True)
        save_epochs = checkpoint_epochs(config.training.num_epochs, config.training.save_count)
        report["checkpoint_epochs"] = list(save_epochs)
        (output / "config.json").write_text(json.dumps(asdict(config), indent=2) + "\n", encoding="utf-8")
        dataset = workspace_path(config.data.dataset, WORKSPACE)
        print("Loading training views and checking reference files:", dataset, flush=True)
        views = load_views(dataset, config.data.cameras, config.render, np)
        signature = dataset_signature(views, dataset, config.data.cameras)
        reference_cache = ReferenceImageCache(config.render.width, config.render.height, mi, np,
                                             config.training.reference_cache_mib)
        print(f"Reference image cache: {config.training.reference_cache_mib} MiB RAM, "
              "loaded on first use; dataset must remain unchanged during training", flush=True)
        rng = np.random.default_rng(config.training.seed)
        sampler = ContourSampler(np, **asdict(config.contour))
        completed, global_step = 0, 0
        if checkpoint:
            print("Restoring voxel model and training state", flush=True)
            state = checkpoint["training_state"]
            if state["dataset_signature"] != signature:
                raise ValueError("Training cameras/reference images differ from the checkpoint")
            model = restore_model(checkpoint, mi)
            completed, global_step = state["completed_epochs"], state["global_step"]
            if (type(completed) is not int or not 0 <= completed <= config.training.num_epochs
                    or type(global_step) is not int or global_step != completed * len(views)):
                raise ValueError("Checkpoint epoch/update count does not match a complete traversal of the training views")
            rng.bit_generator.state = state["rng_state"]
            sampler.load_state_dict(state["contour_state"])
        else:
            print("Initializing voxel model from point cloud:", dataset / config.data.point_cloud, flush=True)
            bounds_min = np.asarray(config.model.bounds_min)
            bounds_max = np.asarray(config.model.bounds_max)
            center, half = (bounds_min + bounds_max) / 2, (bounds_max - bounds_min) / 2 * config.model.bounds_padding
            resolution = config.schedule.start_resolution if config.schedule.enabled else config.model.grid_resolution
            model = mi.VoxelReconstruction(str(dataset / config.data.point_cloud), resolution,
                (center-half).tolist(), (center+half).tolist(), config.model.opacity_threshold,
                config.model.sh_count, config.model.opacity_sh_count)
        optimizer = FalcorOptimizer(model, config.optimizer)
        topology = TopologyManager(model, len(views), settings=config.topology,
            schedule=replace(config.schedule, save_each_level=False),
            opacity_warmup=config.optimizer.opacity_warmup_epochs, opacity_ramp=config.optimizer.opacity_ramp_epochs)
        if checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            topology.load_state_dict(checkpoint["training_state"]["topology_state"])
            # Loading legacy lifecycle state restores its schedule too. Avoid
            # making additional pre-refinement snapshots under the new policy.
            topology.schedule.save_each_level = False
            if checkpoint["training_state"].get("pending_epoch_transition", False):
                replacement, transition = topology.finish_epoch(model, optimizer, completed)
                if replacement is not None:
                    model = restore_model(replacement, mi)
                if transition.get("refined", False):
                    sampler = ContourSampler(np, **asdict(config.contour))
                topology.before_refine_checkpoint = None

        def save(path, epochs):
            save_started = time.perf_counter()
            dr.eval(*model.snapshot().values())
            dr.sync_thread()
            arrays = {key: np.array(value, copy=True) for key, value in model.snapshot().items()}
            metadata = model_metadata(model)
            optimizer_state, topology_state = optimizer.state_dict(), topology.state_dict()
            save_checkpoint(path, arrays, metadata,
                training_state={"completed_epochs": epochs, "global_step": global_step,
                    "rng_state": rng.bit_generator.state, "contour_state": sampler.state_dict(),
                    "topology_state": topology_state, "dataset_signature": signature,
                    "pending_epoch_transition": False},
                optimizer_state=optimizer_state, config=asdict(config), provenance=report["provenance"])
            report["checkpoint"] = str(path)
            performance.record_checkpoint(path, time.perf_counter()-save_started)

        report.update(view_count=len(views), resumed=checkpoint is not None)
        # A restart writes a new history segment, preserving any later entries
        # from the interrupted run instead of appending ambiguous duplicate epochs.
        report["history_file"] = str(output / history_name)
        profile_path = output / ("profile-" + Path(history_name).stem + ".jsonl")
        if config.training.profile_views or config.training.profile_interval_epochs:
            report["profile_file"] = str(profile_path)
            performance.metadata["profile_file"] = str(profile_path)
            print("Kernel profile log:", profile_path, flush=True)
        processed_views = 0
        last_training_resolution = None
        performance.set_phase_seconds("setup", time.perf_counter()-run_started)
        performance.write("running")
        training_loop_started = time.perf_counter()
        print("Starting training; first kernels may require JIT compilation", flush=True)
        for epoch in range(completed, config.training.num_epochs):
            training_resolution = int(model.resolution)
            active_voxels_before = int(model.active_voxel_count)
            entering_resolution = last_training_resolution != training_resolution
            growth_evidence_requested = topology.needs_rendered_alpha(epoch)
            deletion_evidence_collected = topology.collect_deletion(epoch)
            interval = config.training.profile_interval_epochs
            order = rng.permutation(len(views)) if config.training.shuffle else np.arange(len(views))
            epoch_loss = 0.
            stage_totals = dict(reference=0., gradients=0., topology_observation=0., optimizer=0.)
            started = time.perf_counter()
            progress_line.update(f"Epoch {epoch+1}/{config.training.num_epochs} | view 0/{len(views)} (0%) "
                                 f"| grid={int(model.resolution)}", force=True)
            for progress, index in enumerate(order, 1):
                view = views[int(index)]
                view_started = time.perf_counter()
                rgba = reference_cache.get(view["reference_path"])
                reference_done = time.perf_counter()
                reasons = []
                if processed_views < config.training.profile_views:
                    reasons.append("initial_views")
                if interval and progress == 1:
                    if entering_resolution:
                        reasons.append("enter_resolution")
                    if (epoch + 1) % interval == 0:
                        reasons.append("periodic_epoch")
                profiling = bool(reasons)
                # Sparse kernel-history samples add instrumentation. Wall timings
                # below include tracing, compilation, transfers and waits.
                with dr.kernel_history() if profiling else nullcontext() as history:
                    gradients, loss, rendered_alpha = train_view(model, view, rgba, config, sampler, rng, mi, dr, np,
                        need_rendered_alpha=growth_evidence_requested)
                    gradients_done = time.perf_counter()
                    topology.observe_view(model, gradients, view_id=int(index), epoch=epoch,
                        camera=view["camera"], reference_alpha=rgba[..., 3], rendered_alpha=rendered_alpha)
                    topology_done = time.perf_counter()
                    optimizer.step(model, gradients, sample_count=1, epoch=epoch, coarse_to_fine=config.schedule.enabled)
                    optimizer_done = time.perf_counter()
                stage = dict(reference=reference_done-view_started, gradients=gradients_done-reference_done,
                             topology_observation=topology_done-gradients_done, optimizer=optimizer_done-topology_done)
                for key, duration in stage.items():
                    stage_totals[key] += duration
                processed_views += 1
                global_step += 1
                epoch_loss += loss
                progress_line.update(f"Epoch {epoch+1}/{config.training.num_epochs} "
                    f"| view {progress}/{len(views)} ({100*progress/len(views):.0f}%) "
                    f"| cam {int(index)+1} | loss={loss:.6f} | {optimizer_done-view_started:.2f}s/view")
                if profiling:
                    profile = dict(epoch=epoch+1, progress=progress, camera=int(index)+1,
                                   training_resolution=training_resolution, active_voxels=int(model.active_voxel_count),
                                   sample_reasons=reasons, growth_evidence_requested=growth_evidence_requested,
                                   wall_seconds=optimizer_done-view_started, stage_seconds=stage,
                                   **kernel_profile(history))
                    with profile_path.open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(profile) + "\n")
                    performance.record_profile(profile)
            progress_line.update(f"Epoch {epoch+1}/{config.training.num_epochs} "
                                 f"| view {len(views)}/{len(views)} (100%) | updating topology...", force=True)
            topology_started = time.perf_counter()
            new_state, event = topology.finish_epoch(model, optimizer, epoch+1)
            topology.before_refine_checkpoint = None
            if new_state is not None:
                model = restore_model(new_state, mi)
            if event.get("refined", False):
                # Falcor resets contour proposals whenever grid resolution changes.
                sampler = ContourSampler(np, **asdict(config.contour))
            entry = {"epoch": epoch+1, "global_step": global_step, "mean_loss": epoch_loss / len(views),
                     "seconds": time.perf_counter()-started, "resolution": int(model.resolution),
                     "training_resolution": training_resolution, "active_voxels_before": active_voxels_before,
                     "growth_evidence_requested": growth_evidence_requested,
                     "deletion_evidence_collected": deletion_evidence_collected,
                     "active_voxels": int(model.active_voxel_count), "topology": event,
                     "reference_cache": reference_cache.statistics(),
                     "view_stage_seconds": stage_totals,
                     "epoch_topology_seconds": time.perf_counter()-topology_started}
            with (output / history_name).open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(entry) + "\n")
            report["last_epoch"] = entry
            performance.record_epoch(entry)
            last_training_resolution = training_resolution
            progress_line.finish(f"Epoch {epoch+1}/{config.training.num_epochs} done "
                f"| loss={entry['mean_loss']:.6f} | {entry['seconds']:.2f}s "
                f"| grid={entry['resolution']} | voxels={entry['active_voxels']}")
            if epoch+1 in save_epochs and epoch+1 < config.training.num_epochs:
                save(output / f"epoch-{epoch+1:05d}.npz", epoch+1)
                print("Saved checkpoint:", output / f"epoch-{epoch+1:05d}.npz", flush=True)
            performance.set_phase_seconds("training_loop", time.perf_counter()-training_loop_started)
            performance.write("running", training_status="running")
        performance.set_phase_seconds("training_loop", time.perf_counter()-training_loop_started)
        training_loop_started = None
        # Publish only the completed reconstruction, using the same portable,
        # atomic checkpoint format understood by independent evaluation scripts.
        export_started = time.perf_counter()
        save(model_path, config.training.num_epochs)
        print("Saved reconstructed voxel model:", model_path, flush=True)
        report.update(status="running" if config.evaluation.enabled else "passed", training_status="passed",
                      evaluation={"status": "pending" if config.evaluation.enabled else "skipped"},
                      completed_epochs=config.training.num_epochs, global_step=global_step,
                      checkpoint=str(model_path),
                      reconstructed_model={"path": str(model_path), "sha256": file_sha256(model_path),
                          "completed_epochs": config.training.num_epochs, "global_step": global_step})
        performance.set_phase_seconds("final_export", time.perf_counter()-export_started)
    except KeyboardInterrupt:
        progress_line.finish()
        report.update(status="interrupted", training_status="interrupted")
        raise
    except Exception:
        progress_line.finish()
        report.update(status="failed", training_status="failed", error=traceback.format_exc())
        print(report["error"], file=sys.stderr, flush=True)
    finally:
        progress_line.finish()
        if reference_cache is not None:
            report["reference_cache"] = reference_cache.statistics()
        if training_loop_started is not None:
            performance.set_phase_seconds("training_loop", time.perf_counter()-training_loop_started)
        report["training_finished_utc"] = datetime.now(timezone.utc).isoformat()
        if report["status"] != "running":
            report["finished_utc"] = report["training_finished_utc"]
        (output / "training-report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
        performance.write(report["status"], training_status=report["training_status"],
            reference_cache=report.get("reference_cache"), training_report=str(output / "training-report.json"))
    return config, model_path, output, report, performance


def main():
    config, model_path, output, report, performance = run_training()
    if report["training_status"] != "passed":
        return 1
    if not config.evaluation.enabled:
        return 0
    evaluation = final_evaluation_config(config, model_path)
    evaluation_report = model_path.parent / "evaluation-report.json"
    report["evaluation"] = {"status": "running", "spp": evaluation.spp,
        "max_views_per_split": evaluation.max_views, "splits": list(evaluation.splits),
        "output": evaluation.output_directory, "report": str(evaluation_report)}
    (output / "training-report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    limit = "all views" if evaluation.max_views is None else f"first {evaluation.max_views} views per split"
    print(f"Starting final train/test evaluation: {evaluation.spp} spp, {limit}", flush=True)
    evaluation_started = time.perf_counter()
    try:
        # run_training's model, optimizer and gradients are out of scope. Reclaim
        # their allocations before restoring the published checkpoint for evaluation.
        gc.collect()
        import drjit as dr
        dr.sync_thread()
        dr.flush_malloc_cache()
        from reconstruction_evaluate import run_evaluation
        result = run_evaluation(evaluation)
        report["evaluation"]["status"] = "passed" if result == 0 else "failed"
        report["status"] = "passed" if result == 0 else "failed"
        if result != 0:
            print(f"Training completed and model is saved at {model_path}; "
                  f"evaluation failed. See {evaluation_report}", file=sys.stderr, flush=True)
    except KeyboardInterrupt:
        report["status"] = "interrupted"
        report["evaluation"]["status"] = "interrupted"
        raise
    except Exception:
        report["status"] = "failed"
        report["evaluation"].update(status="failed", error=traceback.format_exc())
        print(report["evaluation"]["error"], file=sys.stderr, flush=True)
        print(f"Training completed; the saved model is preserved at {model_path}", file=sys.stderr, flush=True)
    finally:
        performance.set_phase_seconds("evaluation", time.perf_counter()-evaluation_started)
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        (output / "training-report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
        performance.write(report["status"], training_status=report["training_status"],
            evaluation=report["evaluation"], reference_cache=report.get("reference_cache"),
            training_report=str(output / "training-report.json"), evaluation_report=str(evaluation_report))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
